try:
    from vis_open3d_voxel import save_occ, save_gaussian, save_gaussian_topdown, save_occ_error
except:
    try:
        from vis import save_occ, save_gaussian, save_gaussian_topdown
        # save_occ_error 可能在 vis.py 中不存在，导入兜底
        try:
            from vis_open3d_voxel import save_occ_error
        except:
            pass
    except:
        print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os, json
import torch, numpy as np
import torch.distributed as dist

from PIL import Image
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

# ──────────────────────────────────────────────
# FIFO 时序融合 import（流式可视化使用）
# ──────────────────────────────────────────────
try:
    from fifo_eval import (
        FIFOQueue,
        fuse_predictions,
        fused_soft_to_hard,
        soft_pred_to_grid,
    )
except ImportError as e:
    _fifo_available = False
    def fuse_predictions(*args, **kwargs):
        raise RuntimeError('fifo_eval 不可用，请确保 fifo_eval.py 存在')
else:
    _fifo_available = True

# ─── 网格参数（与 eval_stream.py 保持一致）───
_H, _W, _D = 200, 200, 16
_PC_MIN = torch.tensor([-50.0, -50.0, -5.0], dtype=torch.float32)
_PC_MAX = torch.tensor([50.0, 50.0, 3.0], dtype=torch.float32)
_GRID_PARAMS = {
    'H': _H, 'W': _W, 'D': _D,
    'pc_min': _PC_MIN,
    'pc_max': _PC_MAX,
}


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

    # 注意: 不添加 num_samples 到 val_dataset_config，
    #       NuScenesDataset.__init__() 不接受该参数（只在模型中用到）
    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "vis_scene_index": args.vis_scene_index})

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
        try:
            # raw_model.load_state_dict(ckpt['state_dict'], strict=True)
            raw_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        except:
            os.system(f"python modify_weight.py --work-dir {args.work_dir} --epoch {args.epoch}")
            cfg.resume_from = os.path.join(args.work_dir, f"epoch_{args.epoch}_mod.pth")
            ckpt = torch.load(cfg.resume_from, map_location=map_location)
            raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq
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

    my_model.eval()
    os.environ['eval'] = 'true'
    if args.vis_occ or args.vis_occ_error or args.vis_gaussian or args.vis_gaussian_point or args.vis_gaussian_topdown:
        save_dir = os.path.join(args.work_dir, f'vis_ep{args.epoch}')
        os.makedirs(save_dir, exist_ok=True)
    if args.model_type == "base":
        draw_gaussian_params = dict(
            scalar = 1.5,
            ignore_opa = False,
            filter_zsize = False
        )
    elif args.model_type == "prob":
        draw_gaussian_params = dict(
            scalar = 2.0,
            ignore_opa = True,
            filter_zsize = True
        )

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            ori_imgs = data.pop('ori_img')
            for i in range(ori_imgs.shape[-1]):
                ori_img = ori_imgs[0, ..., i].cpu().numpy()
                ori_img = ori_img[..., [2, 1, 0]]
                ori_img = Image.fromarray(ori_img.astype(np.uint8))
                ori_img.save(os.path.join(save_dir, f'{i_iter_val}_image_{i}.png'))
            
            # breakpoint()
            result_dict = my_model(imgs=input_imgs, metas=data)
            if args.vis_gaussian_gt:
                gaussian_ctr = result_dict['gaussian'].means[0]
                gt_occ = result_dict['sampled_label'][0]
                from vis_open3d_voxel import get_grid_coords
                grids = get_grid_coords([200, 200, 16], [0.4, 0.4, 0.4])
                grids = torch.tensor(grids, device=gaussian_ctr.device) - torch.tensor([40, 40, 1], device=gaussian_ctr.device)
                occ_mask = gt_occ < 17
                gt_grid_occ = grids[occ_mask]
                cated_points = torch.cat([gaussian_ctr, gt_grid_occ], dim=0)
                color_gauss  = torch.ones_like(gaussian_ctr, dtype=torch.float32) * torch.tensor([1.0, 1.0, 1.0], device=gt_occ.device)
                color_gts    = torch.ones_like(gt_grid_occ, dtype=torch.float32) * torch.tensor([0, 1.0, 1.0], device=gt_occ.device)
                cated_colors = torch.cat([color_gauss, color_gts], dim=0)

                origin_poitns= data['lidar_points'][0][:, :3]
                origin_mask_x= (origin_poitns[:, 0] > -40) & (origin_poitns[:, 0] < 40)
                origin_mask_y= (origin_poitns[:, 1] > -40) & (origin_poitns[:, 1] < 40)
                origin_mask_z= (origin_poitns[:, 2] > -1) & (origin_poitns[:, 2] < 5.4)
                origin_mask = origin_mask_x & origin_mask_y & origin_mask_z
                filter_points= origin_poitns[origin_mask]



                cated_points = torch.cat([cated_points, filter_points], dim=0)
                color_points = torch.ones_like(filter_points, dtype=torch.float32) * torch.tensor([1.0, 1.0, 0], device=gt_occ.device)
                cated_colors = torch.cat([cated_colors, color_points], dim=0)

                from open3d_vis_utils import draw_scenes
                draw_scenes(points=cated_points.detach().cpu().numpy(), point_colors=cated_colors.detach().cpu().numpy())
            
            #import pdb
            #pdb.set_trace()
            for idx, pred in enumerate(result_dict['final_occ']):
                pred_occ = pred
                gt_occ = result_dict['sampled_label'][idx]
                occ_shape = [200, 200, 16]
                if args.vis_gaussian_topdown:
                    save_gaussian_topdown(
                        save_dir,
                        result_dict['anchor_init'],
                        result_dict['gaussians'],
                        f'val_{i_iter_val}_topdown'
                    )
                if args.vis_occ:
                    save_occ(
                        save_dir,
                        pred_occ.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_pred',
                        True, 0, dataset=args.dataset)
                    save_occ(
                        save_dir,
                        gt_occ.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_gt',
                        True, 0, dataset=args.dataset)
                if args.vis_gaussian:
                    save_gaussian(
                        save_dir,
                        result_dict['gaussian'],
                        f'val_{i_iter_val}_gaussian',
                        **draw_gaussian_params)
                if args.vis_gaussian_each_stage:
                    for gaussian in result_dict['gaussians']:
                        save_gaussian(
                            save_dir,
                            gaussian,
                            f'val_{i_iter_val}_gaussian',
                            **draw_gaussian_params
                        )
                # ── 预测错误分类可视化 ──
                if args.vis_occ_error:
                    save_occ_error(
                        save_dir,
                        pred_occ.reshape(*occ_shape),
                        gt_occ.reshape(*occ_shape),
                        f'val_{i_iter_val}',
                        dataset=args.dataset)

                miou_metric._after_step(pred_occ, gt_occ)
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
                    
    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()
    
    if writer is not None:
        writer.close()


# ══════════════════════════════════════════════
# 流式评估可视化 (NuScenesFlowDataset)
# ══════════════════════════════════════════════

def main_stream(local_rank, args):
    """流式评估可视化主函数。

    使用 NuScenesFlowDataset + SceneStream 按场景顺序逐帧读取数据，
    支持 FIFO 时序融合可视化，并按场景组织可视化结果。
    """
    # 检查 FIFO 依赖
    if args.fifo and not _fifo_available:
        raise ImportError(
            '启用 FIFO 时序融合需要 fifo_eval.py，请确保该文件存在。'
        )

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
        hosts = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        gpus = torch.cuda.device_count()
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
    from dataset import get_stream_dataloader

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

    # ── 流式 Config 适配 ──
    cfg.val_dataset_config['type'] = 'NuScenesFlowDataset'
    cfg.val_dataset_config.pop('phase', None)

    # 构建流式 DataLoader
    val_dataset_loader = get_stream_dataloader(
        dataset_config=cfg.val_dataset_config,
        loader_config=cfg.val_loader,
        dist=distributed,
        shuffle_scenes=False,
    )

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
        try:
            raw_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        except:
            os.system(f"python modify_weight.py --work-dir {args.work_dir} --epoch {args.epoch}")
            cfg.resume_from = os.path.join(args.work_dir, f"epoch_{args.epoch}_mod.pth")
            ckpt = torch.load(cfg.resume_from, map_location=map_location)
            raw_model.load_state_dict(ckpt['state_dict'], strict=True)
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

    # ── draw_gaussian_params ──
    if args.model_type == "base":
        draw_gaussian_params = dict(
            scalar=1.5,
            ignore_opa=False,
            filter_zsize=False,
        )
    elif args.model_type == "prob":
        draw_gaussian_params = dict(
            scalar=2.0,
            ignore_opa=True,
            filter_zsize=True,
        )

    # ── FIFO 队列初始化 ──
    if args.fifo:
        fifo_queue = FIFOQueue(maxlen=args.temporal_windows)
        logger.info(
            f'[FIFO] 启用时序融合可视化: '
            f'windows={args.temporal_windows}, '
            f'alpha={args.fusion_alpha}'
        )
    else:
        fifo_queue = None

    # ── 可视化根目录 ──
    stream_vis_root = osp.join(args.work_dir, f'stream_vis_ep{args.epoch}')
    os.makedirs(stream_vis_root, exist_ok=True)
    logger.info(f'Stream visualization root: {stream_vis_root}')

    # ── 指标 ──
    from misc.metric_util import MeanIoU
    CLASS_INDICES = list(range(1, 17))
    NUM_CLASSES = 17
    CLASS_NAMES = [
        'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation',
    ]
    global_miou = MeanIoU(
        CLASS_INDICES, NUM_CLASSES, CLASS_NAMES,
        True, NUM_CLASSES, filter_minmax=False)
    global_miou.reset()
    per_scene_mious = {}
    scene_results = {}
    scene_frame_counts = {}

    # ─── 流式评估 & 可视化主循环 ──────────────────────────────
    my_model.eval()
    os.environ['eval'] = 'true'

    current_scene = None
    total_frames = 0

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):

            # ── 场景元数据 ──
            scene_token = data['scene_token'][0]
            frame_in_scene = data.get('frame_index_in_scene', [0])[0]
            is_first = data.get('is_first_frame', [False])[0]
            is_last = data.get('is_last_frame', [False])[0]

            # ── 场景切换检测 ──
            if scene_token != current_scene:
                if current_scene is not None:
                    # 上一个场景收尾
                    scene_miou, scene_occiou = per_scene_mious[current_scene]._after_epoch()
                    scene_results[current_scene] = {
                        'mIoU': round(float(scene_miou), 2),
                        'occIoU': round(float(scene_occiou), 2),
                        'num_frames': scene_frame_counts[current_scene],
                    }
                    if local_rank == 0:
                        logger.info(
                            f'{"═" * 55}\n'
                            f' Scene {current_scene} completed: '
                            f'{scene_frame_counts[current_scene]} frames\n'
                            f'   mIoU   = {float(scene_miou):.2f}%\n'
                            f'   occIoU = {float(scene_occiou):.2f}%\n'
                            f'{"═" * 55}'
                        )

                current_scene = scene_token
                per_scene_mious[current_scene] = MeanIoU(
                    CLASS_INDICES, NUM_CLASSES, CLASS_NAMES,
                    True, NUM_CLASSES, filter_minmax=False,
                    name=current_scene)
                per_scene_mious[current_scene].reset()
                scene_frame_counts[current_scene] = 0

                # FIFO: 场景切换时清空队列
                if args.fifo and fifo_queue is not None:
                    fifo_queue.clear()
                    logger.info(f'[FIFO] 场景切换, FIFO 队列已清空')

                # 场景级可视化目录
                scene_vis_dir = osp.join(stream_vis_root, current_scene)
                os.makedirs(scene_vis_dir, exist_ok=True)

                if local_rank == 0:
                    logger.info(f'[STREAM VIS] Starting scene: {scene_token}')

            # ── 场景级可视化目录（已在上方创建）──
            scene_vis_dir = osp.join(stream_vis_root, current_scene)

            # ── 数据移至 GPU ──
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
                elif isinstance(data[k], np.ndarray):
                    data[k] = torch.from_numpy(data[k]).cuda()

            input_imgs = data.pop('img')
            ori_imgs = data.pop('ori_img', None)

            # ── 保存原始图片 ──
            if ori_imgs is not None and local_rank == 0:
                for i in range(ori_imgs.shape[-1]):
                    ori_img = ori_imgs[0, ..., i].cpu().numpy()
                    ori_img = ori_img[..., [2, 1, 0]]
                    ori_img = Image.fromarray(ori_img.astype(np.uint8))
                    ori_img.save(osp.join(
                        scene_vis_dir,
                        f'frame_{frame_in_scene:04d}_image_{i}.png'))

            # ── 模型推理 ──
            result_dict = my_model(imgs=input_imgs, metas=data)

            # ── 可视化 Gaussian GT ──
            if args.vis_gaussian_gt and local_rank == 0:
                gaussian_ctr = result_dict['gaussian'].means[0]
                gt_occ = result_dict['sampled_label'][0]
                from vis_open3d_voxel import get_grid_coords
                grids = get_grid_coords([200, 200, 16], [0.4, 0.4, 0.4])
                grids = torch.tensor(grids, device=gaussian_ctr.device) - torch.tensor(
                    [40, 40, 1], device=gaussian_ctr.device)
                occ_mask = gt_occ < 17
                gt_grid_occ = grids[occ_mask]
                cated_points = torch.cat([gaussian_ctr, gt_grid_occ], dim=0)
                color_gauss = torch.ones_like(
                    gaussian_ctr, dtype=torch.float32) * torch.tensor(
                    [1.0, 1.0, 1.0], device=gt_occ.device)
                color_gts = torch.ones_like(
                    gt_grid_occ, dtype=torch.float32) * torch.tensor(
                    [0, 1.0, 1.0], device=gt_occ.device)
                cated_colors = torch.cat([color_gauss, color_gts], dim=0)

                origin_points = data['lidar_points'][0][:, :3]
                origin_mask_x = (origin_points[:, 0] > -40) & (origin_points[:, 0] < 40)
                origin_mask_y = (origin_points[:, 1] > -40) & (origin_points[:, 1] < 40)
                origin_mask_z = (origin_points[:, 2] > -1) & (origin_points[:, 2] < 5.4)
                origin_mask = origin_mask_x & origin_mask_y & origin_mask_z
                filter_points = origin_points[origin_mask]

                cated_points = torch.cat([cated_points, filter_points], dim=0)
                color_points = torch.ones_like(
                    filter_points, dtype=torch.float32) * torch.tensor(
                    [1.0, 1.0, 0], device=gt_occ.device)
                cated_colors = torch.cat([cated_colors, color_points], dim=0)

                from open3d_vis_utils import draw_scenes
                draw_scenes(
                    points=cated_points.detach().cpu().numpy(),
                    point_colors=cated_colors.detach().cpu().numpy())

            # ── FIFO 时序融合 ──
            if args.fifo and fifo_queue is not None:
                soft_pred_batch = result_dict['pred_occ'][-1]  # (B, C, N)
                batch_size = soft_pred_batch.shape[0]

                fused_occ_list = []
                for idx in range(batch_size):
                    soft_pred = soft_pred_batch[idx]           # (C, N)
                    curr_lidar2prev = data['lidar2prev'][idx]  # (4, 4)
                    curr_xyz = result_dict['sampled_xyz'][idx]  # (N, 3)

                    fused_soft_flat = fuse_predictions(
                        curr_soft_flat=soft_pred,
                        fifo_queue=fifo_queue,
                        alpha=args.fusion_alpha,
                        curr_lidar2prev=curr_lidar2prev,
                        sampled_xyz_curr=curr_xyz,
                        grid_params=_GRID_PARAMS,
                    )
                    fused_hard = fused_soft_to_hard(fused_soft_flat)
                    fused_occ_list.append(fused_hard)

                    # 推入当前帧到 FIFO 队列
                    soft_pred_grid = soft_pred_to_grid(
                        soft_pred, _W, _H, _D)
                    fifo_queue.push(
                        soft_pred_grid=soft_pred_grid,
                        lidar2prev=curr_lidar2prev,
                        frame_idx=scene_frame_counts.get(current_scene, 0) + idx,
                        scene_token=scene_token,
                    )

                pred_occ_for_metric = fused_occ_list  # list of (N,)
            else:
                batch_size = result_dict['final_occ'].shape[0]
                pred_occ_for_metric = [
                    result_dict['final_occ'][idx] for idx in range(batch_size)
                ]

            # ── 帧计数 ──
            scene_frame_counts[current_scene] += batch_size
            total_frames += batch_size

            # ── 指标累积与可视化 ──
            for idx in range(batch_size):
                pred_occ = pred_occ_for_metric[idx]
                gt_occ = result_dict['sampled_label'][idx]
                occ_shape = [_H, _W, _D]

                # 指标
                occ_mask = result_dict.get(
                    'occ_cam_mask',
                    torch.ones_like(gt_occ, dtype=torch.bool))[idx].flatten()
                global_miou._after_step(pred_occ, gt_occ, occ_mask)
                per_scene_mious[current_scene]._after_step(
                    pred_occ, gt_occ, occ_mask)

                # ── 可视化（仅 rank 0）──
                if local_rank != 0:
                    continue

                # 命名前缀：包含场景和帧索引
                frame_tag = f'frame_{frame_in_scene:04d}'

                # ── 预测错误分类可视化 ──
                if args.vis_occ_error:
                    save_occ_error(
                        scene_vis_dir,
                        pred_occ.reshape(*occ_shape),
                        gt_occ.reshape(*occ_shape),
                        frame_tag,
                        dataset=args.dataset)

                # 俯视图 Gaussian
                if args.vis_gaussian_topdown:
                    save_gaussian_topdown(
                        scene_vis_dir,
                        result_dict['anchor_init'],
                        result_dict['gaussians'],
                        f'{frame_tag}_topdown',
                    )

                # 占用预测 / GT
                if args.vis_occ:
                    save_occ(
                        scene_vis_dir,
                        pred_occ.reshape(1, *occ_shape),
                        f'{frame_tag}_pred',
                        True, 0, dataset=args.dataset)
                    save_occ(
                        scene_vis_dir,
                        gt_occ.reshape(1, *occ_shape),
                        f'{frame_tag}_gt',
                        True, 0, dataset=args.dataset)

                # Gaussian 分布
                if args.vis_gaussian:
                    save_gaussian(
                        scene_vis_dir,
                        result_dict['gaussian'],
                        f'{frame_tag}_gaussian',
                        **draw_gaussian_params)

                # 每阶段 Gaussian
                if args.vis_gaussian_each_stage:
                    for stage_i, gaussian in enumerate(result_dict['gaussians']):
                        save_gaussian(
                            scene_vis_dir,
                            gaussian,
                            f'{frame_tag}_gaussian_stage{stage_i}',
                            **draw_gaussian_params)

            # ── 日志 ──
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info(
                    f'[STREAM VIS] Iter {i_iter_val:6d} | '
                    f'Scene: {scene_token} | '
                    f'Frame: {frame_in_scene:04d}/{data.get("num_frames_in_scene", ["?"])[0]} | '
                    f'Total in scene: {scene_frame_counts[current_scene]}'
                )

    # ──────────────────────────────────────────────
    # 最后一个场景收尾
    # ──────────────────────────────────────────────
    if current_scene is not None:
        scene_miou, scene_occiou = per_scene_mious[current_scene]._after_epoch()
        scene_results[current_scene] = {
            'mIoU': round(float(scene_miou), 2),
            'occIoU': round(float(scene_occiou), 2),
            'num_frames': scene_frame_counts[current_scene],
        }
        if local_rank == 0:
            logger.info(
                f'{"═" * 55}\n'
                f' Scene {current_scene} completed: '
                f'{scene_frame_counts[current_scene]} frames\n'
                f'   mIoU   = {float(scene_miou):.2f}%\n'
                f'   occIoU = {float(scene_occiou):.2f}%\n'
                f'{"═" * 55}'
            )

    # ── 结果汇总 ──
    if distributed:
        all_scene_results = [None] * world_size
        dist.all_gather_object(all_scene_results, scene_results)
        if local_rank == 0:
            merged_scene_results = {}
            for rank_results in all_scene_results:
                merged_scene_results.update(rank_results)
            scene_results = merged_scene_results

    if local_rank == 0:
        global_miou_result, global_occiou_result = global_miou._after_epoch()
        global_miou_result = float(global_miou_result)
        global_occiou_result = float(global_occiou_result)

        scene_miou_list = [r['mIoU'] for r in scene_results.values()]
        scene_occiou_list = [r['occIoU'] for r in scene_results.values()]

        mean_scene_miou = float(np.mean(scene_miou_list)) if scene_miou_list else 0.0
        std_scene_miou = float(np.std(scene_miou_list)) if scene_miou_list else 0.0
        mean_scene_occiou = float(np.mean(scene_occiou_list)) if scene_occiou_list else 0.0

        logger.info('')
        logger.info(f'{"═" * 55}')
        logger.info(' Streaming Visualization Results')
        logger.info(f'{"═" * 55}')
        logger.info(f' Total scenes     : {len(scene_results)}')
        logger.info(f' Total frames     : {total_frames}')
        logger.info(f'{"─" * 55}')
        logger.info(f' Global mIoU      : {global_miou_result:.2f}%')
        logger.info(f' Global occIoU    : {global_occiou_result:.2f}%')
        logger.info(f'{"─" * 55}')
        logger.info(f' Per-scene mIoU   : {mean_scene_miou:.2f}% ± {std_scene_miou:.2f}%')
        logger.info(f' Per-scene occIoU : {mean_scene_occiou:.2f}%')
        logger.info(f'{"─" * 55}')
        logger.info(f' Per-scene details:')
        for sc_token in sorted(scene_results.keys()):
            r = scene_results[sc_token]
            logger.info(
                f'   {sc_token:20s}: '
                f'mIoU={r["mIoU"]:6.2f}%  '
                f'occIoU={r["occIoU"]:6.2f}%  '
                f'frames={r["num_frames"]:4d}'
            )
        logger.info(f'{"═" * 55}')

        # 保存结果
        eval_results = {
            'config': args.py_config,
            'work_dir': args.work_dir,
            'global_miou': round(global_miou_result, 2),
            'global_occiou': round(global_occiou_result, 2),
            'per_scene': scene_results,
            'per_scene_stats': {
                'mean_miou': round(mean_scene_miou, 2),
                'std_miou': round(std_scene_miou, 2),
                'mean_occiou': round(mean_scene_occiou, 2),
            },
            'num_scenes': len(scene_results),
            'total_frames': total_frames,
            'fifo_enabled': args.fifo,
        }
        save_path = osp.join(args.work_dir, 'stream_vis_results.json')
        with open(save_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f'Stream vis results saved to: {save_path}')

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
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis_gaussian_topdown', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--vis_scene_index', type=int, default=-1)
    parser.add_argument('--vis-scene', action='store_true', default=False)
    parser.add_argument('--vis-gaussian-each-stage', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='nusc')
    parser.add_argument('--model-type', type=str, default="base", choices=["base", "prob"])
    parser.add_argument('--vis-gaussian-gt', action='store_true', default=False)
    parser.add_argument('--vis-occ-error', action='store_true', default=False,
                        help='可视化预测错误分类: 绿色(正确) / 深红(类别错) / 黑色(假阳性)')
    # 流式可视化参数
    parser.add_argument('--stream', action='store_true', default=False,
                        help='启用流式可视化模式 (使用 NuScenesFlowDataset)')
    parser.add_argument('--fifo', action='store_true', default=False,
                        help='启用 FIFO 时序融合可视化')
    parser.add_argument('--temporal-windows', type=int, default=3,
                        help='FIFO 队列长度（时序窗口大小）')
    parser.add_argument('--fusion-alpha', type=float, default=0.7,
                        help='时序融合权重 α: P_fused = P_curr * α + P_history * (1-α)')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    # 自动模式选择：
    #   - 显式指定 --stream → main_stream
    #   - 未指定 --stream 但指定了 --fifo → 自动启用流式模式（因为 FIFO 依赖 NuScenesFlowDataset）
    #   - 其他情况 → 传统 eval 可视化 main
    if args.stream or args.fifo:
        if not args.stream:
            print('[INFO] --fifo 已启用，自动切换到流式可视化模式 (--stream)')
        args.stream = True
        entry_func = main_stream
    else:
        entry_func = main

    if ngpus > 1:
        torch.multiprocessing.spawn(entry_func, args=(args,), nprocs=args.gpus)
    else:
        entry_func(0, args)
