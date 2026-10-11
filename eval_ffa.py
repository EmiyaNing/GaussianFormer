#!/usr/bin/env python3
"""Evaluate current occupancy and selected SparseWorld-FFA horizons."""

import argparse
import json
import os
import os.path as osp
import time

import torch
from mmengine import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor


def _json_safe(value):
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _build_metric(cfg, metric_type):
    return metric_type(
        num_classes=18, use_lidar_mask=False,
        use_image_mask=cfg.occ3d_eval_mask == 'camera',
        free_label=17, report_occ_iou=True)


def _seconds_to_horizons(seconds, interval, available):
    horizons = []
    for value in seconds:
        raw = float(value) / float(interval)
        horizon = int(round(raw))
        if horizon < 1 or horizon > available or abs(raw - horizon) > 1e-6:
            raise ValueError(
                f'{value:g}s is not an available keyframe horizon; '
                f'interval={interval:g}s, available=1..{available}')
        if horizon not in horizons:
            horizons.append(horizon)
    return horizons


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('SparseWorld-FFA evaluation requires CUDA')
    if args.deterministic:
        # Required by deterministic CuBLAS matrix multiplication on CUDA
        # 10.2+. This must be set before the first CUDA computation.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    set_random_seed(args.seed, deterministic=args.deterministic)
    cfg = Config.fromfile(args.py_config)
    if args.batch_size is not None:
        cfg.val_loader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.val_loader.num_workers = args.num_workers
        cfg.val_loader.persistent_workers = args.num_workers > 0
        if args.num_workers == 0:
            cfg.val_loader.pop('prefetch_factor', None)

    import model  # noqa: F401
    from dataset import get_dataloader
    from misc.sparseworld_eval import SparseWorldHorizonMetric
    from model.sparse_latent_dynamics import transform_points

    model_instance = build_segmentor(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('ema_state_dict') if args.use_ema else None
    if state is None:
        state = checkpoint.get('state_dict', checkpoint)
    model_instance.load_state_dict(state, strict=True)
    if hasattr(model_instance, 'set_epoch'):
        model_instance.set_epoch(int(checkpoint.get(
            'model_epoch', max(0, int(checkpoint.get('epoch', 1)) - 1))))
    model_instance.to(device).eval()
    _, loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, val_only=True)
    interval = float(cfg.keyframe_interval_seconds)
    available = int(cfg.num_future)
    horizons = _seconds_to_horizons(
        args.forecast_seconds, interval, available)
    current_metric = _build_metric(cfg, SparseWorldHorizonMetric)
    future_metrics = {
        horizon: _build_metric(cfg, SparseWorldHorizonMetric)
        for horizon in horizons
    }
    tokens = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for index, data in enumerate(loader):
            batch_tokens = data.get('sample_idx', ())
            if isinstance(batch_tokens, str):
                batch_tokens = (batch_tokens,)
            tokens.extend(str(token) for token in batch_tokens)
            for key, value in list(data.items()):
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(device, non_blocking=True)
            imgs = data.pop('img')
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
                output = model_instance(
                    imgs=imgs, metas=data, return_all_horizons=True)

            current_prediction = model_instance.head.rasterizer(
                output['current_pred_points'].float(),
                output['current_pred_logits'].float(),
                model_instance.num_refines).view(
                    output['current_pred_points'].shape[0],
                    *model_instance.head.rasterizer.grid_shape).cpu().numpy()
            current_label = data['occ_label'].cpu().numpy()
            current_camera = data['occ_cam_mask'].cpu().numpy()
            current_lidar = data['occ_lidar_mask'].cpu().numpy()
            for item in range(current_prediction.shape[0]):
                current_metric.add_batch(
                    current_prediction[item], current_label[item],
                    current_lidar[item], current_camera[item])

            for horizon, metric in future_metrics.items():
                step = horizon - 1
                points = output['eval_future_pred_points'][step].float()
                logits = output['eval_future_pred_logits'][step].float()
                coordinate = output['eval_future_coordinates'][step]
                if coordinate == 'current_ego':
                    future_to_current = data[
                        'future_ego_to_current'][:, step]
                    points = transform_points(
                        points.view(points.shape[0], -1,
                                    model_instance.num_refines, 3),
                        torch.linalg.inv(future_to_current.float())
                    ).flatten(1, 2)
                elif coordinate != 'native_future':
                    raise ValueError(
                        f'unknown prediction coordinate: {coordinate}')
                auxiliary_points = output[
                    'eval_future_aux_points'][step]
                auxiliary_logits = output[
                    'eval_future_aux_logits'][step]
                if auxiliary_points is not None:
                    if auxiliary_logits is None:
                        raise ValueError(
                            'future auxiliary points have no logits')
                    # Future-born SparseWorld slots are already expressed in
                    # the native coordinate frame of this future horizon.
                    points = torch.cat(
                        [points, auxiliary_points.float()], dim=1)
                    logits = torch.cat(
                        [logits, auxiliary_logits.float()], dim=1)
                prediction = model_instance.head.rasterizer(
                    points, logits, model_instance.num_refines).view(
                        points.shape[0],
                        *model_instance.head.rasterizer.grid_shape
                    ).cpu().numpy()
                label = data['future_occ_labels'][:, step].cpu().numpy()
                camera = data['future_occ_cam_masks'][:, step].cpu().numpy()
                lidar = data['future_occ_lidar_masks'][:, step].cpu().numpy()
                for item in range(prediction.shape[0]):
                    metric.add_batch(
                        prediction[item], label[item], lidar[item],
                        camera[item])
            if index % cfg.print_freq == 0:
                print(f'[EVAL] {index}/{len(loader)}')
            if args.max_batches is not None and index + 1 >= args.max_batches:
                break
    torch.cuda.synchronize()
    future_results = {}
    for horizon, metric in future_metrics.items():
        step = horizon - 1
        seconds = horizon * interval
        future_results[f'{seconds:g}'] = dict(
            horizon_index=horizon,
            seconds=seconds,
            source=('sparseworld_ffa' if
                    horizon == int(cfg.target_horizon_index) else
                    'sparseworld_baseline'),
            metrics=metric.results())
    summary = dict(
        current=dict(seconds=0.0, source='sparseworld_baseline',
                     metrics=current_metric.results()),
        future=future_results,
        ffa_supervised_horizon=dict(
            horizon_index=int(cfg.target_horizon_index),
            seconds=float(cfg.target_horizon_seconds)),
        sample_tokens=tokens,
        elapsed_seconds=time.perf_counter() - started,
        peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)))
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    if args.metrics_out:
        parent = osp.dirname(osp.abspath(args.metrics_out))
        os.makedirs(parent, exist_ok=True)
        with open(args.metrics_out, 'w', encoding='utf-8') as handle:
            json.dump(_json_safe(summary), handle, indent=2,
                      ensure_ascii=False, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', default=(
        'config/opus/sparseworld_ffa_r50_704x256_5f_to_6f.py'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metrics-out', default='')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument(
        '--forecast-seconds', type=float, nargs='+', default=(1., 2., 3.),
        help='Future times to evaluate; defaults to 1 2 3 seconds.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--use-ema', action='store_true')
    main(parser.parse_args())
