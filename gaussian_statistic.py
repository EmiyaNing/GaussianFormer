"""Gaussian Statistic —— 主入口脚本。

参照 eval.py 的架构：加载 config、构建模型、加载权重、遍历验证集，
对每帧 Gaussian 预测执行多维度属性统计。

用法:
    python gaussian_statistic.py --py-config config/xxx.py --work-dir out/xxx
    python gaussian_statistic.py --py-config config/xxx.py --work-dir out/xxx \\
        --t-sphere 1.5 --t-scale 0.3 --cov-threshold 3.0
"""

import time, argparse, os, os.path as osp
import torch, numpy as np
import torch.distributed as dist
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
import warnings
warnings.filterwarnings("ignore")

from gaussian_statistic.aggregator import GaussianStatAggregator
from gaussian_statistic.reporter import report_statistics


def pass_print(*args, **kwargs):
    pass


def main(local_rank, args):
    # ---- 全局设置 ----
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # ---- 加载 Config ----
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # ---- DDP 初始化 ----
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        gpus = torch.cuda.device_count()
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

    # ---- Logger ----
    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # ---- 构建模型 ----
    import model  # noqa: F401  注册所有 MODELS
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

    # ---- 数据加载（仅验证集） ----
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)

    # ---- 加载权重 ----
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
        logger.info('successfully resumed.')
    elif cfg.get('load_from', None):
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            raw_model.load_state_dict(state_dict, strict=False)
            logger.info('loaded state_dict (strict=False).')
        except:
            from misc.checkpoint_util import refine_load_from_sd
            raw_model.load_state_dict(
                refine_load_from_sd(state_dict), strict=False)
            logger.info('loaded state_dict via refine_load_from_sd.')

    my_model.eval()
    os.environ['eval'] = 'true'

    # ---- 统计聚合器 ----
    # 从模型 head 或 config 中获取类别数量（优先使用 --num-classes）
    if args.num_classes is not None:
        num_classes = args.num_classes
    elif hasattr(raw_model, 'head') and hasattr(raw_model.head, 'num_classes'):
        num_classes = raw_model.head.num_classes
    elif hasattr(cfg.model, 'head') and 'num_classes' in cfg.model.head:
        num_classes = cfg.model.head.num_classes
    else:
        num_classes = 17
    logger.info(f'num_classes for Gaussian statistics: {num_classes}')
    logger.info(f'empty_label for occupancy statistics: {args.empty_label}')
    logger.info(f'ignore_empty for coverage/purity: {args.ignore_empty}')
    logger.info(f'purity threshold rho: {args.purity_threshold_rho}')
    logger.info('Mixed-Gaussian rule: valid purity < rho')
    aggregator = GaussianStatAggregator(
        t_sphere=args.t_sphere,
        t_scale=args.t_scale,
        distance_bins=args.distance_bins,
        percentiles=args.percentiles,
        num_classes=num_classes,
        cov_threshold=args.cov_threshold,
        chunk_size=args.chunk_size,
        exclude_classes=args.exclude_classes,
        exclude_gaussian_classes=args.exclude_gaussian_classes,
        exclude_voxel_classes=args.exclude_voxel_classes,
        empty_label=args.empty_label,
        ignore_empty=args.ignore_empty,
        scale_range=cfg.get('scale_range', None),
        max_pair_elements=args.max_pair_elements,
        histogram_bins=args.histogram_bins,
        purity_threshold_rho=args.purity_threshold_rho,
    )

    stat_freq = args.stat_freq

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            # 移动到 GPU
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()

            input_imgs = data.pop('img')
            metas = data
            # Statistics only consume the final Gaussian prediction. Skipping the
            # GaussianHead avoids an unnecessary local-aggregation render for
            # every validation frame.
            representation = my_model(
                imgs=input_imgs,
                metas=metas,
                rep_only=True,
            )
            gaussian = (
                representation[-1].get('gaussian')
                if isinstance(representation, (list, tuple)) and representation
                else None
            )
            if gaussian is None:
                logger.warning(f'[SKIP] Iter {i_iter_val}: no gaussian in result_dict')
                continue

            aggregator.add_frame(gaussian, metas)
            #print("Processed iter:", i_iter_val)

            # 中间统计快照（方案 3）
            if i_iter_val % stat_freq == 0 and local_rank == 0:
                snap = aggregator.get_snapshot()
                if snap:
                    logger.info(
                        f'[STAT] Iter {i_iter_val:5d} | Frames: {snap["frames"]} | '
                        f'Gaussians: {snap["gaussians"]} | MeanScale: {snap["mean_scale"]:.4f} | '
                        f'NSR: {snap["nsr"]:.4f} | LIGR: {snap["ligr"]:.4f} | '
                        f'MeanVol: {snap["mean_vol"]:.4f} | MeanAR: {snap["mean_ar"]:.4f} | '
                        f'Cov: {snap["mean_coverage"]:.4f} | '
                        f'Purity(valid): {snap.get("mean_purity_valid", 0):.4f} | '
                        f'Purity(penalized): {snap.get("mean_purity_penalized", 0):.4f} | '
                        f'Purity(old): {snap.get("mean_purity_old", 0):.4f} | '
                        f'Mixed-G: {snap.get("mixed_gaussian_ratio", 0):.4f} | '
                        f'Mixed-G(valid): {snap.get("mixed_gaussian_valid_ratio", 0):.4f} | '
                        f'Sem-Sup: {snap.get("sem_sup", 0):.4f}'
                    )
                else:
                    logger.info(f'[STAT] Iter {i_iter_val:5d} (no frames yet)')

            # The aggregator kebounded streaming summaries. Releasing
            # frame-local references here prevents delayed Python reclamation
            # from extending the peak across validation iterations.
            del gaussian, representation, input_imgs, metas, data

    # ---- 多卡汇总（简化方案：仅 rank 0 输出） ----
    if distributed:
        dist.barrier()

    if local_rank == 0:
        stats = aggregator.finalize()
        report_statistics(stats, logger, args.work_dir)
        logger.info('Gaussian statistic completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Statistic')
    parser.add_argument('--py-config', default='config/nuscenes_gs25600_voxel.py')
    parser.add_argument('--work-dir', type=str, default='./out/gaussian_statistic')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    # 统计阈值参数
    parser.add_argument('--t-sphere', type=float, default=2.0,
                        help='Near-Spherical 阈值（AR < t_sphere 视为近球）')
    parser.add_argument('--t-scale', type=float, default=0.5,
                        help='Large Gaussian 尺寸阈值（s_hat > t_scale 视为大高斯）')
    parser.add_argument('--cov-threshold', type=float, default=1.0,
                        help='Coverage 马氏距离阈值（默认 1.0，对应 1σ 椭球）')
    parser.add_argument('--purity-threshold-rho', type=float, default=0.5,
                        help='Mixed-Gaussian 阈值，严格使用 valid purity < rho')
    parser.add_argument('--distance-bins', type=float, nargs='+',
                        default=[0, 10, 20, 30, 40, 50],
                        help='距离分桶边界，如: 0 10 20 30 40 50 70')
    parser.add_argument('--percentiles', type=float, nargs='+',
                        default=[50, 75, 90, 95],
                        help='分位数列表，如: 50 75 90 95')
    parser.add_argument('--stat-freq', type=int, default=100,
                        help='每隔多少帧打印一次中间统计快照（默认 100）')
    parser.add_argument('--chunk-size', type=int, default=30000,
                        help='Coverage/Purity 遍历时的 chunk 大小（默认 30000）')
    parser.add_argument('--max-pair-elements', type=int, default=64_000_000,
                        help='单个 coverage chunk 最多保留的 voxel-Gaussian 距离元素数')
    parser.add_argument('--histogram-bins', type=int, default=4096,
                        help='流式分位数统计的直方图 bin 数')
    parser.add_argument('--exclude-classes', type=int, nargs='+', default=None,
                        help='[已弃用] 请使用 --exclude-gaussian-classes')
    parser.add_argument('--exclude-gaussian-classes', type=int, nargs='+', default=None,
                        help='Coverage 中排除的 Gaussian 预测类别索引')
    parser.add_argument('--exclude-voxel-classes', type=int, nargs='+', default=None,
                        help='Coverage/Purity 中排除的 GT voxel 类别索引')
    parser.add_argument('--empty-label', type=int, default=17,
                        help='GT occupancy 中 empty/free 类别标签，默认 17')
    parser.add_argument('--ignore-empty', action='store_true', default=True,
                        help='Coverage/Purity 是否排除 empty voxels')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='语义类别数；None 表示从模型自动推断')

    args = parser.parse_args()
    if not 0.0 <= args.purity_threshold_rho <= 1.0:
        parser.error('--purity-threshold-rho must be in [0, 1]')

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
