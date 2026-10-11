#!/usr/bin/env python3
"""Evaluate current occupancy and 1/2/3-second teacher-aligned forecasts."""

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
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _move_batch(data, device):
    for key, value in list(data.items()):
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device, non_blocking=True)
    return data


def _metric(cfg):
    from misc.sparseworld_eval import SparseWorldHorizonMetric
    return SparseWorldHorizonMetric(
        num_classes=18, use_lidar_mask=False,
        use_image_mask=cfg.occ3d_eval_mask == 'camera',
        free_label=17, report_occ_iou=True)


@torch.no_grad()
def evaluate_model(model, loader, cfg, device, max_batches=None):
    """Return separate 0/1/2/3-second metrics; never average over time."""
    model.eval()
    horizons = (2, 4, 6)
    metrics = {0: _metric(cfg), **{horizon: _metric(cfg)
                                  for horizon in horizons}}
    tokens = []
    for batch_index, data in enumerate(loader):
        batch_tokens = data.get('sample_idx', ())
        if isinstance(batch_tokens, str):
            batch_tokens = (batch_tokens,)
        tokens.extend(str(token) for token in batch_tokens)
        data = _move_batch(data, device)
        imgs = data.pop('img')
        with torch.cuda.amp.autocast(enabled=bool(cfg.amp)):
            output = model(imgs=imgs, metas=data)

        predictions = {0: output['final_occ_grid']}
        for horizon in horizons:
            step = horizon - 1
            points = output['future_pred_points'][step].float()
            logits = output['future_pred_logits'][step].float()
            predictions[horizon] = model.head.rasterizer(
                points, logits, model.num_refines).view(
                    points.shape[0], *model.head.rasterizer.grid_shape)
        for horizon, prediction in predictions.items():
            if horizon == 0:
                labels = data['occ_label']
                camera = data['occ_cam_mask']
                lidar = data['occ_lidar_mask']
            else:
                step = horizon - 1
                labels = data['future_occ_labels'][:, step]
                camera = data['future_occ_cam_masks'][:, step]
                lidar = data['future_occ_lidar_masks'][:, step]
            prediction = prediction.cpu().numpy()
            labels = labels.cpu().numpy()
            camera = camera.cpu().numpy()
            lidar = lidar.cpu().numpy()
            for item in range(prediction.shape[0]):
                metrics[horizon].add_batch(
                    prediction[item], labels[item], lidar[item], camera[item])
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return dict(
        current=dict(seconds=0., metrics=metrics[0].results()),
        future={f'{horizon * .5:g}s': dict(
            horizon_index=horizon, seconds=horizon * .5,
            metrics=metrics[horizon].results()) for horizon in horizons},
        sample_tokens=tokens)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('teacher-aligned evaluation requires CUDA')
    if args.deterministic:
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
    import model.segmentor.sparseworld_teacher_aligned_segmentor  # noqa: F401
    from dataset import get_dataloader

    model = build_segmentor(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f'checkpoint load failed: {incompat}')
    model.to(device).eval()
    _, loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, val_only=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = evaluate_model(
        model, loader, cfg, device, max_batches=args.max_batches)
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)))
    result = _json_safe(result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.metrics_out:
        os.makedirs(osp.dirname(osp.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', default=(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metrics-out', default='')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    main(parser.parse_args())
