# try:
#     from vis import save_occ
# except:
#     print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os, json
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

from fifo_eval import (
    FIFOQueue,
    fuse_predictions,
    fused_soft_to_hard,
    soft_pred_to_grid,
    GaussianFIFOQueue,
    gaussian_fifo_fuse_and_render,
)
from model.encoder.gaussian_encoder.utils import GaussianPrediction


def pass_print(*args, **kwargs):
    pass


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
    # 确保 work_dir 存在
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

    # ──────────────────────────────────────────────
    # 流式 Config 适配：将 NuScenesDataset 替换为 NuScenesFlowDataset
    # ──────────────────────────────────────────────
    cfg.val_dataset_config['type'] = 'NuScenesFlowDataset'
    cfg.val_dataset_config.pop('phase', None)

    # 构建流式 DataLoader
    val_dataset_loader = get_stream_dataloader(
        dataset_config=cfg.val_dataset_config,
        loader_config=cfg.val_loader,
        dist=distributed,
        shuffle_scenes=args.scene_shuffle,
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

    # ──────────────────────────────────────────────
    # 初始化指标
    # ──────────────────────────────────────────────
    # 只支持 surroundocc（按需求）
    assert cfg.dataset_name_flag == 'surroundocc', \
        f"流式评估目前仅支持 surroundocc，当前数据集: {cfg.dataset_name_flag}"

    from misc.metric_util import MeanIoU

    CLASS_INDICES = list(range(1, 17))
    NUM_CLASSES = 17
    CLASS_NAMES = [
        'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation',
    ]

    # 全局指标（等效于 eval.py 的累积方式）
    global_miou = MeanIoU(
        CLASS_INDICES, NUM_CLASSES, CLASS_NAMES,
        True, NUM_CLASSES, filter_minmax=False)
    global_miou.reset()

    # 场景级指标
    per_scene_mious = {}       # scene_token -> MeanIoU instance
    scene_results = {}         # scene_token -> {mIoU, occIoU, num_frames}
    scene_frame_counts = {}    # scene_token -> int

    # ── FIFO 网格参数 ──
    # 从 config/nuscenes_gs25600_voxel.py 提取
    # pc_range=[-50, -50, -5, 50, 50, 3], grid_size=0.5
    # H=200, W=200, D=16
    H, W, D = 200, 200, 16
    pc_min = torch.tensor([-50.0, -50.0, -5.0], dtype=torch.float32)
    pc_max = torch.tensor([50.0, 50.0, 3.0], dtype=torch.float32)
    grid_params = {
        'H': H, 'W': W, 'D': D,
        'pc_min': pc_min,
        'pc_max': pc_max,
    }

    # ── FIFO 队列初始化 ──
    if args.fifo:
        fifo_queue = FIFOQueue(maxlen=args.temporal_windows)
        logger.info(
            f'[FIFO] 启用时序融合: '
            f'windows={args.temporal_windows}, '
            f'alpha={args.fusion_alpha}'
        )
    else:
        fifo_queue = None

    # ── Gaussian FIFO 队列初始化 ──
    if args.stream_gaussian_fusion:
        gaussian_fifo_queue = GaussianFIFOQueue(maxlen=args.temporal_windows)
        logger.info(
            f'[GaussianFIFO] 启用高斯流式融合: '
            f'windows={args.temporal_windows}'
        )
    else:
        gaussian_fifo_queue = None

    # ──────────────────────────────────────────────
    # 流式评估主循环
    # ──────────────────────────────────────────────
    my_model.eval()
    os.environ['eval'] = 'true'

    current_scene = None
    total_frames = 0

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):

            # ── 提取场景元数据 ──
            # data['scene_token'] 经 collate_fn 处理后为 list[str]
            # batch_size=1 时长度为 1
            scene_token = data['scene_token'][0]

            # ── 场景边界检测 ──
            if scene_token != current_scene:
                if current_scene is not None:
                    # 上一个场景结束：计算场景级指标
                    scene_miou, scene_occiou = per_scene_mious[current_scene]._after_epoch()
                    scene_miou = float(scene_miou)
                    scene_occiou = float(scene_occiou)
                    scene_results[current_scene] = {
                        'mIoU': round(scene_miou, 2),
                        'occIoU': round(scene_occiou, 2),
                        'num_frames': scene_frame_counts[current_scene],
                    }
                    if local_rank == 0:
                        logger.info(
                            f'{"═" * 55}\n'
                            f' Scene {current_scene} completed: '
                            f'{scene_frame_counts[current_scene]} frames\n'
                            f'   mIoU   = {scene_miou:.2f}%\n'
                            f'   occIoU = {scene_occiou:.2f}%\n'
                            f'{"═" * 55}'
                        )

                # 新场景开始
                current_scene = scene_token
                per_scene_mious[current_scene] = MeanIoU(
                    CLASS_INDICES, NUM_CLASSES, CLASS_NAMES,
                    True, NUM_CLASSES, filter_minmax=False,
                    name=current_scene)
                per_scene_mious[current_scene].reset()
                scene_frame_counts[current_scene] = 0

                # FIFO: 场景切换时清空队列
                if args.fifo:
                    fifo_queue.clear()
                    logger.info(f'[FIFO] 场景切换, FIFO 队列已清空')
                if args.stream_gaussian_fusion:
                    gaussian_fifo_queue.clear()
                    logger.info(f'[GaussianFIFO] 场景切换, Gaussian FIFO 队列已清空')

                if local_rank == 0:
                    logger.info(f'[STREAM] Starting scene: {scene_token}')

            # ── 数据移至 GPU ──
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
                elif isinstance(data[k], np.ndarray):
                    data[k] = torch.from_numpy(data[k]).cuda()

            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)

            # ── FIFO 时序融合 ──
            if args.fifo:
                # 提取软预测: pred_occ[-1] shape (B, C, N)
                soft_pred_batch = result_dict['pred_occ'][-1]  # (B, C, N)
                batch_size = soft_pred_batch.shape[0]

                fused_occ_list = []
                for idx in range(batch_size):
                    soft_pred = soft_pred_batch[idx]            # (C, N)
                    curr_lidar2prev = data['lidar2prev'][idx]  # (4, 4)
                    curr_xyz = result_dict['sampled_xyz'][idx] # (N, 3)

                    # 时序融合：当前帧 * α + 历史帧 warped * (1-α)
                    fused_soft_flat = fuse_predictions(
                        curr_soft_flat=soft_pred,
                        fifo_queue=fifo_queue,
                        alpha=args.fusion_alpha,
                        curr_lidar2prev=curr_lidar2prev,
                        sampled_xyz_curr=curr_xyz,
                        grid_params=grid_params,
                        mode=args.fusion_mode,
                    )
                    # fused_soft_flat: (C, N) → 硬标签 (N,)
                    fused_hard = fused_soft_to_hard(fused_soft_flat)
                    fused_occ_list.append(fused_hard)

                    # 推入当前帧到 FIFO 队列（供后续帧融合使用）
                    soft_pred_grid = soft_pred_to_grid(soft_pred, W, H, D)  # (C, D, H, W)
                    fifo_queue.push(
                        soft_pred_grid=soft_pred_grid,
                        lidar2prev=curr_lidar2prev,
                        frame_idx=scene_frame_counts.get(current_scene, 0) + idx,
                        scene_token=scene_token,
                    )

                pred_occ_for_metric = fused_occ_list  # list of (N,)
            elif args.stream_gaussian_fusion:
                # ── Gaussian FIFO 融合 ──
                gaussian_batch = result_dict['gaussian']
                batch_size = gaussian_batch.means.shape[0]
                fused_occ_list = []
                for idx in range(batch_size):
                    curr_g = GaussianPrediction(
                        means=gaussian_batch.means[idx:idx+1],
                        scales=gaussian_batch.scales[idx:idx+1],
                        rotations=gaussian_batch.rotations[idx:idx+1],
                        opacities=gaussian_batch.opacities[idx:idx+1],
                        semantics=gaussian_batch.semantics[idx:idx+1],
                        original_means=(
                            gaussian_batch.original_means[idx:idx+1]
                            if gaussian_batch.original_means is not None else None),
                        delta_means=(
                            gaussian_batch.delta_means[idx:idx+1]
                            if gaussian_batch.delta_means is not None else None),
                    )
                    fused_hard, _ = gaussian_fifo_fuse_and_render(
                        curr_g, gaussian_fifo_queue,
                        data['lidar2prev'][idx],
                        result_dict['sampled_xyz'][idx:idx+1],
                        raw_model.head,
                        grid_params,
                    )
                    fused_occ_list.append(fused_hard)
                    # 推入当前帧到 Gaussian FIFO 队列（CPU 存储，节省 GPU 显存）
                    gaussian_fifo_queue.push(
                        gaussian=GaussianPrediction(
                            means=gaussian_batch.means[idx:idx+1].detach().cpu(),
                            scales=gaussian_batch.scales[idx:idx+1].detach().cpu(),
                            rotations=gaussian_batch.rotations[idx:idx+1].detach().cpu(),
                            opacities=gaussian_batch.opacities[idx:idx+1].detach().cpu(),
                            semantics=gaussian_batch.semantics[idx:idx+1].detach().cpu(),
                            original_means=(
                                gaussian_batch.original_means[idx:idx+1].detach().cpu()
                                if gaussian_batch.original_means is not None else None),
                            delta_means=(
                                gaussian_batch.delta_means[idx:idx+1].detach().cpu()
                                if gaussian_batch.delta_means is not None else None),
                        ),
                        lidar2prev=data['lidar2prev'][idx],
                        frame_idx=scene_frame_counts.get(current_scene, 0) + idx,
                        scene_token=scene_token,
                    )
                pred_occ_for_metric = fused_occ_list
            else:
                batch_size = result_dict['final_occ'].shape[0]
                pred_occ_for_metric = [
                    result_dict['final_occ'][idx] for idx in range(batch_size)
                ]

            # ── 场景帧计数 ──
            scene_frame_counts[current_scene] += batch_size
            total_frames += batch_size

            # ── 指标累积 ──
            for idx in range(batch_size):
                pred_occ = pred_occ_for_metric[idx]
                gt_occ = result_dict['sampled_label'][idx]
                occ_mask = result_dict['occ_cam_mask'][idx].flatten()

                # 全局指标
                global_miou._after_step(pred_occ, gt_occ, occ_mask)
                # 场景级指标
                per_scene_mious[current_scene]._after_step(pred_occ, gt_occ, occ_mask)

            # ── 逐帧可视化（可选） ──
            if args.vis_occ and local_rank == 0:
                try:
                    from vis import save_occ
                    for idx in range(batch_size):
                        pred_occ = result_dict['final_occ'][idx]
                        gt_occ = result_dict['sampled_label'][idx]
                        vis_dir = osp.join(args.work_dir, 'stream_vis', current_scene)
                        os.makedirs(vis_dir, exist_ok=True)
                        save_occ(pred_occ, gt_occ, vis_dir, current_scene,
                                 scene_frame_counts[current_scene] - batch_size + idx)
                except Exception as e:
                    logger.info(f'Visualization failed: {e}')

            # ── 日志 ──
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info(
                    f'[STREAM EVAL] Iter {i_iter_val:6d} | '
                    f'Scene: {scene_token} | '
                    f'Frames in scene: {scene_frame_counts[current_scene]}'
                )

    # ──────────────────────────────────────────────
    # 最后一个场景收尾
    # ──────────────────────────────────────────────
    if current_scene is not None:
        scene_miou, scene_occiou = per_scene_mious[current_scene]._after_epoch()
        scene_miou = float(scene_miou)
        scene_occiou = float(scene_occiou)
        scene_results[current_scene] = {
            'mIoU': round(scene_miou, 2),
            'occIoU': round(scene_occiou, 2),
            'num_frames': scene_frame_counts[current_scene],
        }
        if local_rank == 0:
            logger.info(
                f'{"═" * 55}\n'
                f' Scene {current_scene} completed: '
                f'{scene_frame_counts[current_scene]} frames\n'
                f'   mIoU   = {scene_miou:.2f}%\n'
                f'   occIoU = {scene_occiou:.2f}%\n'
                f'{"═" * 55}'
            )

    # ──────────────────────────────────────────────
    # 指标汇总与输出
    # ──────────────────────────────────────────────
    # 全局指标
    global_miou_result, global_occiou_result = global_miou._after_epoch()
    global_miou_result = float(global_miou_result)
    global_occiou_result = float(global_occiou_result)

    # 分布式场景结果收集
    if distributed:
        all_scene_results = [None] * world_size
        dist.all_gather_object(all_scene_results, scene_results)
        # rank 0 合并所有结果
        if local_rank == 0:
            merged_scene_results = {}
            for rank_results in all_scene_results:
                merged_scene_results.update(rank_results)
            scene_results = merged_scene_results

    if local_rank == 0:
        # 场景级统计
        scene_miou_list = [r['mIoU'] for r in scene_results.values()]
        scene_occiou_list = [r['occIoU'] for r in scene_results.values()]

        mean_scene_miou = float(np.mean(scene_miou_list)) if scene_miou_list else 0.0
        std_scene_miou = float(np.std(scene_miou_list)) if scene_miou_list else 0.0
        mean_scene_occiou = float(np.mean(scene_occiou_list)) if scene_occiou_list else 0.0

        # ── 日志输出 ──
        logger.info('')
        logger.info(f'{"═" * 55}')
        logger.info(' Streaming Evaluation Results')
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

        # ── 结构化结果保存 ──
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
        }
        save_path = osp.join(args.work_dir, 'stream_eval_results.json')
        with open(save_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f'Results saved to: {save_path}')

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Streaming Evaluation')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    # 流式评估特有参数
    parser.add_argument('--scene-shuffle', action='store_true', default=False,
                        help='是否打乱场景评估顺序')
    # FIFO 时序融合参数
    parser.add_argument('--fifo', action='store_true', default=False,
                        help='启用 FIFO 时序融合评估')
    parser.add_argument('--temporal-windows', type=int, default=3,
                        help='FIFO 队列长度（时序窗口大小）')
    parser.add_argument('--fusion-alpha', type=float, default=0.7,
                        help='时序融合权重 α: P_fused = P_curr * α + P_history * (1-α)')
    parser.add_argument('--fusion-mode', type=str, default='conditional',
                        choices=['simple', 'conditional'],
                        help='融合模式: simple=所有体素加权平均, conditional=根据空/非空分情况融合')
    # 基于 Semantic Gaussian 的 FIFO 流式融合
    parser.add_argument('--stream-gaussian-fusion', action='store_true', default=False,
                        help='启用基于 Semantic Gaussian 的 FIFO 流式融合评估')
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
