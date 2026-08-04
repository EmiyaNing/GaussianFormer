# try:
#     from vis import save_occ
# except:
#     print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass


def get_occ3d_eval_mask_name(cfg):
    mask_name = cfg.get('occ3d_eval_mask', None)
    if mask_name is not None:
        return mask_name
    return 'camera' if cfg.get('eval_mask_flag', True) else 'none'


def occ3d_mask_to_numpy(result_dict, key, idx, grid_shape):
    mask = result_dict.get(key, None)
    if mask is None:
        return None
    if mask.dim() == 4:
        mask = mask[idx]
    elif mask.dim() == 2:
        mask = mask[idx]
        if mask.numel() != int(np.prod(grid_shape)):
            raise ValueError(
                f'Flattened {key} has {mask.numel()} elements, expected '
                f'{int(np.prod(grid_shape))} for grid_shape={grid_shape}.')
        mask = mask.reshape(*grid_shape)
    elif tuple(mask.shape) != tuple(grid_shape):
        raise ValueError(
            f'Unsupported {key} shape {tuple(mask.shape)}; expected '
            f'[B, *{tuple(grid_shape)}], [B, V], or {tuple(grid_shape)}.')
    return mask.cpu().numpy()


def get_occ_grid_shape(cfg, result_dict):
    if 'final_occ_grid' in result_dict:
        return tuple(result_dict['final_occ_grid'].shape[1:])
    return tuple(cfg.get('grid_shape', (200, 200, 16)))


def build_sparseworld_future_metrics(cfg):
    """Create one independent metric accumulator for each requested horizon.

    This function is called only when a config explicitly declares
    ``sparseworld_future_eval.enabled=True``.  Generic OPUS/Gaussian configs
    therefore retain their historical evaluation behavior unchanged.
    """
    options = cfg.get('sparseworld_future_eval', None)
    if not options or not options.get('enabled', False):
        return None, None
    if cfg.dataset_name_flag != 'occ3d':
        raise ValueError('SparseWorld future occupancy evaluation currently requires Occ3D metrics')
    horizons = tuple(options.get('horizons', (1, 2, 3, 4, 5, 6)))
    if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError('sparseworld_future_eval.horizons must be unique positive one-based indices')
    from misc.sparseworld_eval import SparseWorldHorizonMetric
    mask_name = get_occ3d_eval_mask_name(cfg)
    metrics = {
        horizon: SparseWorldHorizonMetric(
            num_classes=18, use_lidar_mask=mask_name == 'lidar',
            use_image_mask=mask_name == 'camera', free_label=17,
            report_occ_iou=options.get('report_occ_iou', True))
        for horizon in horizons
    }
    return options, metrics


def update_sparseworld_future_metrics(metrics, result_dict, data, raw_model, grid_shape):
    """Rasterize and score each requested future horizon independently.

    Future predictions are expressed in the current ego coordinate system,
    while future Occ3D grids are native to their own future ego frame.  The
    function therefore maps predictions by the inverse of
    ``future_ego_to_current`` before applying the unchanged OPUS rasterizer.
    """
    from misc.sparseworld_eval import points_current_to_future
    required = ('future_predictions', 'future_logits')
    if any(key not in result_dict for key in required):
        raise KeyError(f'SparseWorld future evaluation requires result keys {required}')
    required = ('future_occ_labels', 'future_occ_cam_masks', 'future_occ_lidar_masks',
                'future_ego_to_current')
    if any(key not in data for key in required):
        raise KeyError(f'SparseWorld future evaluation requires data keys {required}')
    if not hasattr(raw_model, 'head') or not hasattr(raw_model, 'num_refines'):
        raise TypeError('SparseWorld future evaluation requires a SparseWorldTrajSegmentor-like model')

    predictions, logits = result_dict['future_predictions'], result_dict['future_logits']
    available_horizon = min(len(predictions), len(logits), data['future_occ_labels'].shape[1])
    rasterizer = raw_model.head.rasterizer
    for horizon, metric in metrics.items():
        step = horizon - 1
        if step >= available_horizon:
            raise ValueError(
                f'configured future horizon t+{horizon} is unavailable; model/data provide '
                f'{available_horizon} future steps')
        if predictions[step].shape[:2] != logits[step].shape[:2]:
            raise ValueError(f'future point/logit shape mismatch at t+{horizon}')
        for batch_index in range(predictions[step].shape[0]):
            # SparseWorldStrict follows the upstream decoder/evaluation
            # contract: its recurrence directly produces native future-ego
            # coordinates. Legacy SparseWorldTrajSegmentor retains the local
            # current-ego convention and is transformed as before.
            future_points = predictions[step][batch_index]
            if result_dict.get('future_prediction_coordinate') != 'native_future':
                future_points = points_current_to_future(
                    future_points, data['future_ego_to_current'][batch_index, step])
            predicted_grid = rasterizer(
                future_points.unsqueeze(0), logits[step][batch_index].unsqueeze(0),
                group_size=raw_model.num_refines)[0]
            metric.add_batch(
                predicted_grid.reshape(*grid_shape).cpu().numpy(),
                data['future_occ_labels'][batch_index, step].reshape(*grid_shape).cpu().numpy(),
                data['future_occ_lidar_masks'][batch_index, step].reshape(*grid_shape).cpu().numpy(),
                data['future_occ_cam_masks'][batch_index, step].reshape(*grid_shape).cpu().numpy())

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
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(args.work_dir, exist_ok=True)
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
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
        val_only=True)
    
    # resume and load
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
        raw_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
        except:
            from misc.checkpoint_util import refine_load_from_sd
            print(raw_model.load_state_dict(
                refine_load_from_sd(state_dict), strict=False))
        
    print_freq = cfg.print_freq
    if cfg.dataset_name_flag == 'surroundocc':
        from misc.metric_util import MeanIoU
        miou_metric = MeanIoU(
            list(range(1, 17)),
            17, #17,
            ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'],
            True, 17, filter_minmax=False)
        miou_metric.reset()
    elif cfg.dataset_name_flag == 'occ3d':
        from misc.occ3d_nus_metrics import Metric_mIoU
        occ3d_eval_mask = get_occ3d_eval_mask_name(cfg)
        miou_metric = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=occ3d_eval_mask == 'lidar',
            use_image_mask=occ3d_eval_mask == 'camera')
    else:
        print("Not emplement this dataset:", cfg.dataset_name_flag)
        exit(0)
    future_eval_options, future_metrics = build_sparseworld_future_metrics(cfg)
    

    my_model.eval()
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)
            if 'final_occ' in result_dict:
                for idx, pred in enumerate(result_dict['final_occ']):
                    pred_occ = pred
                    gt_occ = result_dict['sampled_label'][idx]
                    if cfg.dataset_name_flag == 'surroundocc':
                        occ_mask = result_dict['occ_cam_mask'][idx].flatten()
                        miou_metric._after_step(pred_occ, gt_occ, occ_mask)
                    elif cfg.dataset_name_flag == 'occ3d':
                        grid_shape = get_occ_grid_shape(cfg, result_dict)
                        pred_occ = pred_occ.reshape(*grid_shape).cpu().numpy()
                        gt_occ   = gt_occ.reshape(*grid_shape).cpu().numpy()
                        occ_cam_mask = occ3d_mask_to_numpy(
                            result_dict, 'occ_cam_mask', idx, grid_shape)
                        occ_lidar_mask = occ3d_mask_to_numpy(
                            result_dict, 'occ_lidar_mask', idx, grid_shape)
                        miou_metric.add_batch(
                            pred_occ, gt_occ, occ_lidar_mask, occ_cam_mask)
                    # breakpoint()
            # Future metrics are strictly opt-in and deliberately separate
            # from the current-frame accumulator above.  This prevents any
            # future horizon from changing historical non-SparseWorld scores.
            if future_metrics is not None:
                update_sparseworld_future_metrics(
                    future_metrics, result_dict, data, raw_model,
                    get_occ_grid_shape(cfg, result_dict))
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))

    if cfg.dataset_name_flag == 'surroundocc':       
        miou, iou2 = miou_metric._after_epoch()
        logger.info(f'mIoU: {miou}, iou2: {iou2}')
        miou_metric.reset()
    elif cfg.dataset_name_flag == 'occ3d':
        eval_results = miou_metric.count_miou_metric()
        logger.info(eval_results)

    if future_metrics is not None:
        # Unlike the legacy current-frame evaluator, future metrics are
        # reduced explicitly so each t+k value is valid under multi-GPU eval.
        metric_device = torch.device('cuda', torch.cuda.current_device())
        for metric in future_metrics.values():
            metric.synchronize(metric_device)
        if local_rank == 0:
            for horizon, metric in future_metrics.items():
                logger.info('[SparseWorld Future t+%d] %s', horizon, metric.results())

    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
