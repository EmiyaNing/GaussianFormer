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
from opus_v2_lr import OPUSV2WarmupCosineLR

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


def training_checkpoint_meta(cfg):
    """Tag only native OPUSv2 checkpoints; other models remain unchanged."""
    if cfg.get('checkpoint_mapping') == 'official_opusv2':
        return {'checkpoint_format': 'gaussianformer_opusv2_training_v1'}
    return None


def eval_amp_enabled(cfg, train_amp):
    """Keep AMP training while allowing spconv models to evaluate in FP32."""
    return bool(train_amp and not cfg.get('force_fp32_eval', False))


def native_amp_optimizer(optimizer):
    """Return the torch optimizer expected by GradScaler."""
    return getattr(optimizer, 'optimizer', optimizer)


def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

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

    # Official OPUS configs specify a global batch size.  Keep this opt-in so
    # every existing model retains its current per-GPU loader semantics.
    if cfg.get('global_batch_size', None) is not None:
        if cfg.global_batch_size % world_size:
            raise ValueError('global_batch_size must be divisible by world_size')
        cfg.train_loader.batch_size = cfg.global_batch_size // world_size
    
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
        iter_resume=args.iter_resume)

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('lr_scheduler_type', None) == 'official_opusv2':
        scheduler = OPUSV2WarmupCosineLR(
            optimizer.optimizer,
            total_steps=len(train_dataset_loader) * max_num_epochs,
            warmup_iters=cfg.get('warmup_iters', 500),
            warmup_ratio=cfg.get('warmup_ratio', 1. / 3.),
            min_lr_ratio=cfg.get('min_lr_ratio', 1e-3))
    elif cfg.get('multisteplr', False):
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
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=cfg.optimizer["optimizer"]["lr"] * cfg.get("min_lr_ratio", 0.1), #1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
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
            ckpt['state_dict'], strict=cfg.get('strict_train_resume', False)))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if amp and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        load_only_prefixes = cfg.get('load_only_prefixes', None)
        if load_only_prefixes:
            state_dict = {
                key: value for key, value in state_dict.items()
                if any(key.startswith(prefix) for prefix in load_only_prefixes)}
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
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

    while epoch < max_num_epochs:
        my_model.train()
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        loss_component_history = {}
        epoch_start_sleep = cfg.get('epoch_start_sleep', 10)
        if epoch_start_sleep > 0:
            time.sleep(epoch_start_sleep)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            if first_run:
                i_iter = i_iter + last_iter

            non_blocking_transfer = cfg.get('non_blocking_transfer', False)
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda(non_blocking=non_blocking_transfer)
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
                    # GradScaler requires the native torch optimizer.  Passing
                    # MMEngine's OptimWrapper exposes its synthetic
                    # base_param_settings as an extra (duplicated) param group,
                    # so one real gradient is unscaled twice and found-inf
                    # bookkeeping no longer matches the AdamW step.
                    amp_optimizer = native_amp_optimizer(optimizer)
                    scaler.unscale_(amp_optimizer)
                    if cfg.get('fail_on_nonfinite_grad', False):
                        bad_gradients = find_nonfinite_gradients(my_model)
                        if bad_gradients:
                            raise FloatingPointError(
                                f'Non-finite gradients at global_iter={global_iter}: '
                                + ', '.join(bad_gradients[:20]))
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        my_model.parameters(), cfg.grad_max_norm,
                        error_if_nonfinite=cfg.get('fail_on_nonfinite_grad', False))
                    scaler.step(amp_optimizer)
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
            data_time_s = time.time()
            time_s = time.time()

            if args.iter_resume:
                if (i_iter + 1) % 50 == 0 and local_rank == 0:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                    }
                    if amp:
                        dict_to_save['scaler'] = scaler.state_dict()
                    checkpoint_meta = training_checkpoint_meta(cfg)
                    if checkpoint_meta is not None:
                        dict_to_save['meta'] = checkpoint_meta
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    symlink(save_file_name, dst_file)
                    logger.info(f'iter ckpt {i_iter + 1} saved!')
        
        # save checkpoint
        if local_rank == 0:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            if amp:
                dict_to_save['scaler'] = scaler.state_dict()
            checkpoint_meta = training_checkpoint_meta(cfg)
            if checkpoint_meta is not None:
                dict_to_save['meta'] = checkpoint_meta
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
                        data[k] = data[k].cuda(
                            non_blocking=cfg.get('non_blocking_transfer', False))
                input_imgs = data.pop('img')
                
                with torch.cuda.amp.autocast(eval_amp_enabled(cfg, amp)):
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
                        if np.isfinite(loss_value):
                            val_detail_sum[loss_name] = (
                                val_detail_sum.get(loss_name, 0.0) + loss_value
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
    parser.add_argument('--seed', type=int, default=42)
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
