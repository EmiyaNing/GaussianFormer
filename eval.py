# try:
#     from vis import save_occ
# except:
#     print('Load Occupancy Visualization Tools Failed.')
#
# SparseWorld was released against MMCV 1/MMDet 2, while this repository uses
# MMCV 2/MMDet 3.  Detect the compatibility config before importing either
# stack, then execute the exact released graph in an isolated interpreter.
import argparse
import os
import os.path as osp
from pathlib import Path
import runpy
import re
import subprocess
import sys
import time


def _format_gaussianformer_log(message):
    """Format delegated output like the MMEngine logs used by this project."""
    timestamp = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
    return f'{timestamp} - mmengine - INFO - {message}'


def _append_gaussianformer_log(log_file, message):
    """Append one or more plain-text messages in GaussianFormer log format."""
    if not log_file:
        return
    log_path = Path(log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as stream:
        for line in str(message).splitlines():
            stream.write(_format_gaussianformer_log(line) + '\n')


def make_eval_separator(epoch_label):
    """Build the visible boundary between checkpoint evaluation results."""
    label = str(epoch_label)
    if label.isdigit():
        title = f'Epoch {label}'
    else:
        title = f'Checkpoint {label}'
    return f'-------------------- {title} --------------------'


def _make_unique_log_path(work_dir, suffix):
    """Choose a timestamped log path without overwriting an older run."""
    directory = Path(work_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    candidate = directory / f'{timestamp}_{suffix}.log'
    collision = 1
    while candidate.exists():
        candidate = directory / f'{timestamp}_{suffix}_{collision}.log'
        collision += 1
    return candidate


def _delegate_sparseworld_official_eval() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--py-config')
    parser.add_argument('--work-dir', default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', default='')
    parser.add_argument('--eval-all', '--eval_all', action='store_true')
    parser.add_argument('--sparseworld-python', default=os.environ.get('SPARSEWORLD_OFFICIAL_PYTHON', ''))
    parser.add_argument('--eval-log-file', default='')
    parser.add_argument('--eval-epoch', default='')
    known, _ = parser.parse_known_args()
    # ``eval_all`` is expanded into one ordinary eval subprocess per
    # checkpoint in __main__. Each child then reaches this compatibility
    # delegate with an explicit --resume-from.
    if known.eval_all:
        return
    if not known.py_config:
        return
    config_path = Path(known.py_config).resolve()
    if not config_path.is_file():
        return
    config = runpy.run_path(str(config_path))
    if not config.get('sparseworld_official_compat', False):
        return
    checkpoint = known.resume_from or config.get('strict_checkpoint') or config.get('official_checkpoint')
    if not checkpoint:
        raise RuntimeError('SparseWorld official config requires --resume-from or strict_checkpoint.')
    root = Path(__file__).resolve().parent
    command = [sys.executable, str(root / 'tools' / 'sparseworld_official_compat.py'), 'eval',
               '--official-root', str(config['official_root']), '--config', str(config['official_config']),
               '--checkpoint', str(Path(checkpoint).resolve())]
    if known.sparseworld_python:
        command += ['--python', known.sparseworld_python]
    command += list(config.get('official_eval_args', []))
    print('delegating exact SparseWorld evaluation:', ' '.join(command), flush=True)
    log_file = (Path(known.eval_log_file).expanduser().resolve()
                if known.eval_log_file else
                _make_unique_log_path(known.work_dir, 'eval'))
    if known.eval_epoch:
        _append_gaussianformer_log(log_file, make_eval_separator(known.eval_epoch))
    completed = subprocess.run(
        command, cwd=root, check=False, env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.stdout:
        print(completed.stdout, end='', flush=True)
        _append_gaussianformer_log(log_file, completed.stdout)
    completed.check_returncode()
    print(f'GaussianFormer-format evaluation log saved to {log_file}', flush=True)
    raise SystemExit(0)


_delegate_sparseworld_official_eval()

import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def _checkpoint_sort_key(path):
    """Natural filename order, e.g. epoch_2.pth before epoch_10.pth."""
    return tuple((1, int(part)) if part.isdigit() else (0, part.lower())
                 for part in re.split(r'(\d+)', path.name))


def discover_eval_checkpoints(work_dir):
    """Return every direct ``*.pth`` child of work_dir in stable order."""
    directory = Path(work_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f'--work-dir is not a directory: {directory}')
    checkpoints = sorted(
        (path for path in directory.glob('*.pth') if path.is_file()),
        key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f'No .pth checkpoints found in --work-dir: {directory}')
    return checkpoints


def checkpoint_epoch_label(checkpoint):
    """Extract the epoch number from a conventional checkpoint filename."""
    stem = Path(checkpoint).stem
    match = re.search(r'(?:^|[_-])epoch[_-]?(\d+)(?:[_-]|$)', stem,
                      flags=re.IGNORECASE)
    return match.group(1) if match else stem


def select_eval_checkpoints(checkpoints, start_from=None):
    """Select checkpoints at or after ``start_from`` for eval-all."""
    if start_from is None:
        return list(checkpoints)
    if start_from < 0:
        raise ValueError('--start-from must be zero or greater')
    selected = []
    for checkpoint in checkpoints:
        epoch_label = checkpoint_epoch_label(checkpoint)
        if epoch_label.isdigit() and int(epoch_label) >= start_from:
            selected.append(checkpoint)
    if not selected:
        raise FileNotFoundError(
            f'No epoch checkpoint at or after --start-from {start_from}')
    return selected


def make_eval_all_log_path(work_dir):
    """Choose a unique aggregate log path without overwriting an older run."""
    return _make_unique_log_path(work_dir, 'eval_all')


def run_eval_all(args):
    """Run the existing evaluation entry point once for every checkpoint."""
    if args.resume_from:
        raise ValueError('--eval-all and --resume-from cannot be used together')
    checkpoints = select_eval_checkpoints(
        discover_eval_checkpoints(args.work_dir),
        getattr(args, 'start_from', None))
    root = Path(__file__).resolve().parent
    log_file = make_eval_all_log_path(args.work_dir)
    common = [
        sys.executable, str(Path(__file__).resolve()),
        '--py-config', str(Path(args.py_config).expanduser().resolve()),
        '--work-dir', str(Path(args.work_dir).expanduser().resolve()),
        '--eval-log-file', str(log_file),
    ]
    if args.sparseworld_python:
        common += ['--sparseworld-python', args.sparseworld_python]
    if args.seed is not None:
        common += ['--seed', str(args.seed)]
    if args.vis_occ:
        common.append('--vis-occ')

    print(f'[eval_all] Found {len(checkpoints)} checkpoints in '
          f'{Path(args.work_dir).expanduser().resolve()}', flush=True)
    for index, checkpoint in enumerate(checkpoints, start=1):
        command = common + [
            '--resume-from', str(checkpoint),
            '--eval-tag', checkpoint.stem,
            '--eval-epoch', checkpoint_epoch_label(checkpoint),
        ]
        if index > 1:
            command.append('--skip-config-log')
        print(f'\n[eval_all] [{index}/{len(checkpoints)}] Evaluating '
              f'{checkpoint.name}', flush=True)
        subprocess.run(command, cwd=root, check=True, env=os.environ.copy())
    print(f'\n[eval_all] Completed {len(checkpoints)} checkpoints.', flush=True)
    print(f'[eval_all] Results saved to {log_file}', flush=True)


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


def main(local_rank, args):
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    # global settings (strict SparseWorld records seed 17 in checkpoint meta)
    seed = cfg.get('seed', 42) if args.seed is None else args.seed
    set_random_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

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
    os.makedirs(args.work_dir, exist_ok=True)
    shared_log_file = getattr(args, 'eval_log_file', '')
    if shared_log_file:
        log_file = str(Path(shared_log_file).expanduser().resolve())
        log_mode = 'a'
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        eval_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', args.eval_tag).strip('_.')
        log_suffix = f'_{eval_tag}' if eval_tag else ''
        log_file = osp.join(args.work_dir, f'{timestamp}{log_suffix}.log')
        log_mode = 'w'
    logger = MMLogger('selfocc', log_file=log_file, file_mode=log_mode)
    MMLogger._instance_dict['selfocc'] = logger
    if not getattr(args, 'skip_config_log', False):
        logger.info(f'Config:\n{cfg.pretty_text}')
    eval_epoch = getattr(args, 'eval_epoch', '')
    if eval_epoch:
        logger.info(make_eval_separator(eval_epoch))

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
        val_only=True,
        seed=cfg.get('dataloader_seed', None))
    
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
        if cfg.get('sparseworld_official_compat', False):
            from misc.sparseworld_compat_checkpoint import load_sparseworld_compat_checkpoint
            verified = load_sparseworld_compat_checkpoint(
                raw_model, cfg.load_from, cfg.get('checkpoint_manifest', None))
            logger.info('Strictly loaded %d SparseWorld official-compat tensors.', verified)
        else:
            ckpt = torch.load(cfg.load_from, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            try:
                print(raw_model.load_state_dict(
                    state_dict, strict=cfg.get('strict_load', False)))
            except:
                from misc.checkpoint_util import refine_load_from_sd
                print(raw_model.load_state_dict(
                    refine_load_from_sd(state_dict), strict=False))
        
    print_freq = cfg.print_freq
    strict_metric = None
    if cfg.get('sparseworld_strict_eval', False):
        from misc.sparseworld_strict_eval import SparseWorldStrictMetric
        # Opt-in per-sample dump: without it a report can only quote one
        # aggregate per variant and cannot produce a bootstrap interval.
        strict_metric = SparseWorldStrictMetric(
            per_sample_path=os.environ.get('DSH_EVAL_PER_SAMPLE') or None,
            spatial_samples=int(os.environ.get('DSH_EVAL_SPATIAL_SAMPLES', '0')))
    elif cfg.dataset_name_flag == 'surroundocc':
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

    my_model.eval()
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)
            if strict_metric is not None:
                strict_metric.add_batch(result_dict, data)
            elif 'final_occ' in result_dict:
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
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))

    if strict_metric is not None:
        strict_metric.synchronize(torch.device('cuda', torch.cuda.current_device()))
        eval_results = strict_metric.results()
        logger.info(eval_results)
    elif cfg.dataset_name_flag == 'surroundocc':
        miou, iou2 = miou_metric._after_epoch()
        logger.info(f'mIoU: {miou}, iou2: {iou2}')
        miou_metric.reset()
    elif cfg.dataset_name_flag == 'occ3d':
        eval_results = miou_metric.count_miou_metric()
        logger.info(eval_results)

    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--eval-all', '--eval_all', dest='eval_all',
                        action='store_true',
                        help='sequentially evaluate every direct *.pth file in --work-dir')
    parser.add_argument('--start-from', type=int, default=None,
                        help='in --eval-all mode, evaluate epochs at or after this number')
    parser.add_argument('--sparseworld-python', type=str, default='',
                        help='legacy SparseWorld Python; used only by the official compatibility config')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--eval-tag', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--eval-log-file', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--eval-epoch', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--skip-config-log', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.start_from is not None and not args.eval_all:
        parser.error('--start-from can only be used together with --eval-all')
    if args.start_from is not None and args.start_from < 0:
        parser.error('--start-from must be zero or greater')
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.eval_all:
        run_eval_all(args)
        raise SystemExit(0)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
