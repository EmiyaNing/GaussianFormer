import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler
from one_cycle_lr import OneCycleLR

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


def get_occ3d_eval_mask_name(cfg):
    mask_name = cfg.get('occ3d_eval_mask', None)
    if mask_name is not None:
        return mask_name
    return 'camera' if cfg.get('eval_mask_flag', True) else 'none'


def select_occ3d_eval_mask(metas, idx, cfg, expected_numel=None):
    mask_name = get_occ3d_eval_mask_name(cfg)
    if mask_name == 'none':
        return None
    if mask_name == 'camera':
        mask = metas.get('occ_cam_mask', None)
    elif mask_name == 'lidar':
        mask = metas.get('occ_lidar_mask', None)
    elif mask_name == 'nonempty':
        mask = metas.get('occ_nonempty_mask', None)
        if mask is None:
            mask = metas.get('occ_mask', None)
    else:
        raise NotImplementedError(f'Unsupported Occ3D eval mask: {mask_name}')

    if mask is None:
        return None
    # ``metas`` is the local-rank batch, never a globally gathered batch.
    # Select its leading batch dimension for both dense [B,X,Y,Z] and flat
    # [B,V] masks.  A rank can have a smaller tail batch, so validate idx.
    if mask.dim() >= 2:
        if idx >= mask.shape[0]:
            raise IndexError(f'Occ3D mask batch index {idx} is outside local shape {tuple(mask.shape)}')
        mask = mask[idx]
    mask = mask.reshape(-1).bool()
    if expected_numel is not None and mask.numel() != expected_numel:
        raise ValueError(
            'Occ3D evaluation mask/prediction size mismatch on one local rank: '
            f'mask={mask.numel()}, prediction={expected_numel}. '
            'Check that occ masks retain [B, X, Y, Z] or [B, V] layout.')
    return mask


def find_nonfinite_gradients(model):
    return [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]


def loss_value_to_float(value):
    """Convert a loss component/metric to a detached Python float for logging."""
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().cpu().item()
    elif hasattr(value, 'item'):
        value = value.item()
    return float(value)


def main(local_rank, args):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    seed = cfg.get('seed', 42) if args.seed is None else args.seed
    set_random_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', True)

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        from misc.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('selfocc', log_dir=osp.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['selfocc'] = writer
    else:
        writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(f'Runtime cuDNN benchmark: {torch.backends.cudnn.benchmark}')
    logger.info('CUDA allocator configuration: ' + str(
        os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'))))

    # build model
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    logger.info(f'Params require grad: {[n for n, p in my_model.named_parameters() if p.requires_grad]}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=args.iter_resume,
        seed=cfg.get('dataloader_seed', None))

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config
        )
    elif cfg.get('onecyclelr'):
        scheduler = OneCycleLR(
            optimizer.optimizer,
            num_steps=len(train_dataset_loader) * max_num_epochs,
            lr_range=(2e-5, 2e-4)
        )
    else:
        # Most historical configs use the repository's 1e-6 warm-up start.
        # A configuration can opt into a different start without changing the
        # scheduler behavior of configurations that omit this field.
        warmup_lr_init = cfg.get('warmup_lr_init', 1e-6)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=cfg.optimizer["optimizer"]["lr"] * cfg.get("min_lr_ratio", 0.1), #1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=False)
    amp = cfg.get('amp', False)
    if amp:
        # Official OPUS-V1 uses a fixed FP16 loss scale of 512.  A very large
        # growth interval preserves that scale while retaining GradScaler's
        # overflow handling in this runner.
        scaler = torch.cuda.amp.GradScaler(
            init_scale=cfg.get('amp_loss_scale', 65536.0),
            growth_interval=cfg.get('amp_growth_interval', 2000))
        os.environ['amp'] = 'true'
    else:
        os.environ['amp'] = 'false'
    
    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(
            ckpt['state_dict'], strict=cfg.get('strict_load', False)))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.get('train_load_from', cfg.get('load_from', None)):
        initial_checkpoint = cfg.get('train_load_from', cfg.get('load_from'))
        if cfg.get('sparse_gaussian_warmstart', False):
            from misc.sparseworld_compat_checkpoint import \
                load_sparse_gaussian_world_warmstart
            report = load_sparse_gaussian_world_warmstart(
                raw_model, initial_checkpoint)
            print('SparseGaussianWorld warm-start:', report)
        else:
            ckpt = torch.load(initial_checkpoint, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            revise_keys = cfg.get('train_load_revise_keys', [])
            if revise_keys:
                revised = {}
                for key, value in state_dict.items():
                    target_key = key
                    for source_prefix, target_prefix in revise_keys:
                        if target_key.startswith(source_prefix):
                            target_key = target_prefix + target_key[len(source_prefix):]
                            break
                    revised[target_key] = value
                state_dict = revised
            try:
                print(raw_model.load_state_dict(
                    state_dict, strict=cfg.get('train_load_strict',
                                               cfg.get('strict_load', False))))
            except:
                from misc.checkpoint_util import refine_load_from_sd
                print(raw_model.load_state_dict(
                    refine_load_from_sd(state_dict), strict=False))
        
    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_accumulation = args.gradient_accumulation
    grad_norm = 0
    from misc.metric_util import MeanIoU
    if cfg.dataset_name_flag == 'surroundocc':
        miou_metric = MeanIoU(
            list(range(1, 17)),
            17, #17,
            ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'],
            True, 17, filter_minmax=False)
    elif cfg.dataset_name_flag == 'occ3d':
        miou_metric = MeanIoU(
            list(range(17)),
            17, #17,
            ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'],
            True, 17, filter_minmax=False)
    else:
        print("Not emplement this dataset:", cfg.dataset_name_flag)
        exit(0)
    miou_metric.reset()

    iter_checkpoint_interval = int(cfg.get('iter_checkpoint_interval', 50))
    if iter_checkpoint_interval < 1:
        raise ValueError('iter_checkpoint_interval must be positive')

    while epoch < max_num_epochs:
        my_model.train()
        if cfg.get('enable_model_set_epoch', False) and hasattr(raw_model, 'set_epoch'):
            raw_model.set_epoch(epoch)
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        loss_component_history = {}
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            if cfg.get('cuda_memory_log_interval', 0):
                torch.cuda.reset_peak_memory_stats()
            if first_run:
                i_iter = i_iter + last_iter

            if cfg.get('scheduler_step_before_optimizer', False):
                scheduler.step_update(global_iter)

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda(non_blocking=True)
            input_imgs = data.pop('img')            
            data_time_e = time.time()

            with torch.cuda.amp.autocast(amp):
                # forward + backward + optimize
                result_dict = my_model(imgs=input_imgs, metas=data, global_iter=global_iter)

                loss_input = {
                    'metas': data,
                    'global_iter': global_iter
                }
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    loss_input.update({
                        loss_input_key: result_dict[loss_input_val]})
                loss, loss_dict = loss_func(loss_input)
                if (cfg.get('fail_on_nonfinite_grad', False) and
                        not torch.isfinite(loss)):
                    raise FloatingPointError(
                        f'Non-finite loss at global_iter={global_iter}: {loss.item()}')
                loss = loss / grad_accumulation
            if not amp:
                loss.backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    if cfg.get('fail_on_nonfinite_grad', False):
                        bad_gradients = find_nonfinite_gradients(my_model)
                        if bad_gradients:
                            raise FloatingPointError(
                                f'Non-finite gradients at global_iter={global_iter}: '
                                + ', '.join(bad_gradients[:20]))
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        my_model.parameters(), cfg.grad_max_norm,
                        error_if_nonfinite=cfg.get('fail_on_nonfinite_grad', False))
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                scaler.scale(loss).backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    scaler.unscale_(optimizer)
                    if cfg.get('fail_on_nonfinite_grad', False):
                        bad_gradients = find_nonfinite_gradients(my_model)
                        if bad_gradients:
                            raise FloatingPointError(
                                f'Non-finite gradients at global_iter={global_iter}: '
                                + ', '.join(bad_gradients[:20]))
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        my_model.parameters(), cfg.grad_max_norm,
                        error_if_nonfinite=cfg.get('fail_on_nonfinite_grad', False))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            loss_list.append(loss.detach().cpu().item())
            # Keep a per-component history over the same logging window as
            # ``loss_list``.  MultiLoss already returns detached scalar values,
            # but the conversion also handles tensor/NumPy metrics from custom
            # loss implementations.
            for loss_name, loss_value in loss_dict.items():
                try:
                    scalar_value = loss_value_to_float(loss_value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(scalar_value):
                    loss_component_history.setdefault(loss_name, []).append(scalar_value)
            if not cfg.get('scheduler_step_before_optimizer', False):
                scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0:
                #lr = max([p['lr'] for p in optimizer.param_groups])
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    current_value = loss_value_to_float(loss_value)
                    history = loss_component_history.get(loss_name, [])
                    history_mean = np.mean(history) if history else float('nan')
                    detailed_loss.append(
                        f'{loss_name}: current={current_value:.5f}, '
                        f'history_avg={history_mean:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info('[TRAIN][LossComponents] ' + detailed_loss)
                loss_list = []
                loss_component_history = {}
            del result_dict, loss_input, loss, loss_dict, input_imgs, data
            memory_log_interval = cfg.get('cuda_memory_log_interval', 0)
            if memory_log_interval and i_iter % memory_log_interval == 0 and local_rank == 0:
                memory_stats = torch.cuda.memory_stats()
                logger.info(
                    '[CUDA_MEMORY] after_cleanup epoch=%d iter=%d accumulation_pending=%d/%d '
                    'allocated=%.1f MiB reserved=%.1f MiB '
                    'peak_allocated=%.1f MiB peak_reserved=%.1f MiB inactive_split=%.1f MiB '
                    'alloc_retries=%d benchmark=%s',
                    epoch, i_iter, global_iter % grad_accumulation, grad_accumulation,
                    torch.cuda.memory_allocated() / 2**20,
                    torch.cuda.memory_reserved() / 2**20,
                    torch.cuda.max_memory_allocated() / 2**20,
                    torch.cuda.max_memory_reserved() / 2**20,
                    memory_stats.get('inactive_split_bytes.all.current', 0) / 2**20,
                    memory_stats.get('num_alloc_retries', 0), torch.backends.cudnn.benchmark)

            if args.iter_resume:
                if (i_iter + 1) % iter_checkpoint_interval == 0 and local_rank == 0:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                    }
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    symlink(save_file_name, dst_file)
                    logger.info(f'iter ckpt {i_iter + 1} saved!')
            # Checkpoint I/O must not be reported as next-batch data loading.
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if local_rank == 0:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        first_run = False
        
        # eval
        if epoch % cfg.get('eval_every_epochs', 1) != 0:
            continue
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []
        val_detail_sum = {}
        val_detail_count = {}

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):
                for k in list(data.keys()):
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda(non_blocking=True)
                input_imgs = data.pop('img')
                
                with torch.cuda.amp.autocast(amp):
                    result_dict = my_model(imgs=input_imgs, metas=data)

                    loss_input = {
                        'metas': data,
                        'global_iter': global_iter
                    }
                    for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                        loss_input.update({
                            loss_input_key: result_dict[loss_input_val]})
                    loss, loss_dict = loss_func(loss_input)
                    for loss_name, loss_value in loss_dict.items():
                        # Loss details are CUDA scalar tensors during eval.
                        # NumPy attempts an implicit ``Tensor.numpy()`` here,
                        # which is invalid for CUDA storage.  Keep the
                        # finiteness check in PyTorch and accumulate detached
                        # host scalars only; this is logging/aggregation logic
                        # and does not alter any model or loss computation.
                        if isinstance(loss_value, torch.Tensor):
                            if loss_value.numel() != 1:
                                raise ValueError(
                                    f'eval loss detail {loss_name!r} must be a '
                                    f'scalar tensor, got {tuple(loss_value.shape)}')
                            is_finite = torch.isfinite(loss_value).all().item()
                            scalar_value = loss_value.detach().item()
                        else:
                            is_finite = np.isfinite(loss_value)
                            scalar_value = loss_value
                        if is_finite:
                            val_detail_sum[loss_name] = (
                                val_detail_sum.get(loss_name, 0.0) + scalar_value
                            )
                            val_detail_count[loss_name] = (
                                val_detail_count.get(loss_name, 0) + 1
                            )
                
                if 'final_occ' in result_dict:
                    for idx, pred in enumerate(result_dict['final_occ']):
                        pred_occ = pred
                        gt_occ = result_dict['sampled_label'][idx]
                        if cfg.dataset_name_flag == 'surroundocc':
                            occ_mask = result_dict['occ_cam_mask'][idx].flatten()
                        elif cfg.dataset_name_flag == 'occ3d':
                            # Use the local dataloader batch as the source of
                            # truth.  Head outputs may be flattened adapters;
                            # the raw batch preserves its per-rank B dimension.
                            occ_mask = select_occ3d_eval_mask(
                                data, idx, cfg, expected_numel=pred_occ.numel())
                        miou_metric._after_step(pred_occ, gt_occ, occ_mask)
                
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)
                del result_dict, loss_input, loss, loss_dict, input_imgs, data
                        
        miou, iou2 = miou_metric._after_epoch()
        if distributed:
            gathered_details = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered_details,
                (val_detail_sum, val_detail_count),
            )
            val_detail_sum = {}
            val_detail_count = {}
            for rank_sum, rank_count in gathered_details:
                for name, value in rank_sum.items():
                    val_detail_sum[name] = val_detail_sum.get(name, 0.0) + value
                    val_detail_count[name] = (
                        val_detail_count.get(name, 0) + rank_count[name]
                    )
        logger.info(f'mIoU: {miou}, iou2: {iou2}')
        logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))
        if val_detail_sum:
            averaged_details = [
                f'{name}: {val_detail_sum[name] / val_detail_count[name]:.5f}'
                for name in sorted(val_detail_sum)
            ]
            logger.info('[EVAL] Averaged losses/metrics: ' + ', '.join(averaged_details))
        miou_metric.reset()
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
