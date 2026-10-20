"""Audit Gaussian attributes and empty-class calibration on Occ3D validation.

This reads a trained checkpoint and never changes its weights.  The empty-logit
sweep reuses the same rendered tensor, so it tests decision calibration rather
than claiming a new trained model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine import Config
from mmseg.models import build_segmentor
from torch.utils.data import DataLoader, Subset

import model  # noqa: F401 - register modules
from dataset import get_dataloader
from dataset.utils import custom_collate_fn_temporal
from tools.center_support_common import write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/mini_occ3d_4096_1024/opusv1_gaussian_sparse_render.py'
CHECKPOINT = ROOT / 'work_dirs/mini_occ3d_4096_1024/opusv1_gaussian_sparse_render/epoch_5.pth'
CLASSES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation')


def file_sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def calibrated_predictions(fg_max, fg_class, empty_logit, shift, empty_label=17,
                           foreground_wins_ties=True):
    threshold = empty_logit - shift
    occupied = fg_max >= threshold if foreground_wins_ties else fg_max > threshold
    return torch.where(occupied, fg_class, empty_label)


def training_selection(length, limit, seed, imageset, manifest=None):
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed))[:limit].tolist()
    record = dict(imageset=str(imageset), imageset_sha256=file_sha256(imageset),
                  population=length, selection_seed=seed, indices=indices)
    if manifest:
        path = Path(manifest)
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError('Training calibration manifest differs from the fixed selection')
        write_json(path, record)
    return indices


def choose_empty_shift(result):
    if result.get('split') != 'train':
        raise ValueError('Empty bias must be selected exclusively from training data')
    options = result['empty_logit_shift_metrics']
    # Prefer the smallest correction when measured mIoU ties.
    return float(min(options, key=lambda s: (-options[s]['mIoU'], abs(float(s)), float(s))))


def metrics(confusion):
    correct = confusion.diag()
    seen = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    intersection = correct[:17]
    union = seen[:17] + predicted[:17] - intersection
    per_class = torch.where(union > 0, intersection / union.clamp_min(1), 0)
    tp = confusion[:17, :17].sum()
    occupied_seen = seen[:17].sum()
    occupied_pred = predicted[:17].sum()
    return {
        'mIoU': float(per_class.mean() * 100),
        'IoU': float(tp / (occupied_seen + occupied_pred - tp).clamp_min(1) * 100),
        'precision': float(tp / occupied_pred.clamp_min(1) * 100),
        'recall': float(tp / occupied_seen.clamp_min(1) * 100),
        'predicted_occupied_fraction': float(occupied_pred / confusion.sum() * 100),
        'gt_occupied_fraction': float(occupied_seen / confusion.sum() * 100),
        'per_class_IoU': {
            name: float(value * 100)
            for name, value in zip(CLASSES, per_class)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--skip-first', type=int, default=0)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--selection-seed', type=int, default=928)
    parser.add_argument('--selection-manifest', type=Path,
                        help='Record/verify identical training subset indices and annotation SHA256')
    parser.add_argument('--shifts', type=float, nargs='+', default=[-2., 0., 2., 4., 6., 8.])
    parser.add_argument('--py-config', type=Path, default=CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--out', type=Path, default=ROOT /
                        'work_dirs/mini_occ3d_4096_1024/opusv1_gaussian_diagnostic.json')
    args = parser.parse_args()
    if args.limit < 1 or args.skip_first < 0:
        parser.error('--limit must be positive and --skip-first nonnegative')
    if len(set(args.shifts)) != len(args.shifts) or 0. not in args.shifts or not all(map(math.isfinite, args.shifts)):
        parser.error('--shifts must be unique finite values including zero')
    if args.selection_manifest and args.split != 'train':
        parser.error('--selection-manifest is only for training calibration')
    cfg = Config.fromfile(str(args.py_config))
    cfg.val_loader.batch_size = 1
    cfg.val_loader.num_workers = 4
    if args.split == 'train':
        # Use the validation transforms for reproducible calibration, but
        # sample exclusively from the training split.
        cfg.val_dataset_config.imageset = cfg.train_dataset_config.imageset
    _, loader = get_dataloader(cfg.train_dataset_config, cfg.val_dataset_config,
                               cfg.train_loader, cfg.val_loader, val_only=True)
    if args.split == 'train':
        subset_indices = training_selection(len(loader.dataset), args.limit, args.selection_seed,
                                            cfg.val_dataset_config.imageset, args.selection_manifest)
        loader = DataLoader(
            Subset(loader.dataset, subset_indices), batch_size=1, shuffle=False,
            num_workers=cfg.val_loader.num_workers, pin_memory=True,
            collate_fn=custom_collate_fn_temporal)
    network = build_segmentor(cfg.model).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    shifts = args.shifts
    confusion = {shift: torch.zeros((18, 18), dtype=torch.float64) for shift in shifts}
    sums = {key: 0.0 for key in (
        'opacity', 'scale_x', 'scale_y', 'scale_z', 'opacity_under_005',
        'opacity_over_09', 'scale_over_039', 'semantic_max_logit',
        'foreground_max_gt_occupied', 'empty_logit_gt_occupied',
        'foreground_max_gt_empty', 'empty_logit_gt_empty',
        'foreground_correct_gt_occupied')}
    denominators = {'gaussian': 0, 'voxel_occupied': 0, 'voxel_empty': 0}
    processed = 0
    legacy_ties = bool(network.head.empty_logit_offset)
    with torch.inference_mode():
        for index, data in enumerate(loader):
            if index < args.skip_first:
                continue
            if index >= args.skip_first + args.limit:
                break
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.cuda(non_blocking=True)
            imgs = data.pop('img')
            output = network(imgs=imgs, metas=data)
            gaussian = output['gaussian']
            opacity = gaussian.opacities.reshape(-1)
            scales = gaussian.scales.reshape(-1, 3)
            semantic = gaussian.semantics.reshape(-1, 17)
            count = opacity.numel()
            denominators['gaussian'] += count
            sums['opacity'] += opacity.sum().item()
            for axis, key in enumerate(('scale_x', 'scale_y', 'scale_z')):
                sums[key] += scales[:, axis].sum().item()
            sums['opacity_under_005'] += (opacity < 0.05).sum().item()
            sums['opacity_over_09'] += (opacity > 0.9).sum().item()
            sums['scale_over_039'] += (scales.max(dim=1).values > 0.39).sum().item()
            sums['semantic_max_logit'] += semantic.max(dim=1).values.sum().item()

            logits = output['pred_occ'][-1][0]
            fg_max, fg_class = logits[:17].max(dim=0)
            empty_logit = logits[17]
            gt = output['sampled_label'][0].reshape(-1).long()
            mask = data['occ_cam_mask'][0].reshape(-1).bool() & (gt >= 0) & (gt < 18)
            occupied = mask & (gt < 17)
            empty = mask & (gt == 17)
            sums['foreground_correct_gt_occupied'] += ((fg_class == gt) & occupied).sum().item()
            for name, values, which in (
                    ('foreground_max_gt_occupied', fg_max, occupied),
                    ('empty_logit_gt_occupied', empty_logit, occupied),
                    ('foreground_max_gt_empty', fg_max, empty),
                    ('empty_logit_gt_empty', empty_logit, empty)):
                sums[name] += values[which].sum().item()
            denominators['voxel_occupied'] += occupied.sum().item()
            denominators['voxel_empty'] += empty.sum().item()
            for shift in shifts:
                pred = calibrated_predictions(fg_max, fg_class, empty_logit, shift,
                                              foreground_wins_ties=not legacy_ties)
                if shift == 0 and not legacy_ties:
                    assert torch.equal(pred, output['final_occ'][0].reshape(-1))
                if network.head.empty_logit_offset and shift == network.head.empty_logit_offset:
                    assert torch.equal(pred, output['final_occ'][0].reshape(-1)), (
                        'head decision differs from direct logit calibration')
                flat = gt[mask] * 18 + pred[mask]
                confusion[shift] += torch.bincount(flat, minlength=18 * 18).reshape(
                    18, 18).cpu().double()
            processed += 1
            if processed % 100 == 0:
                print(f'processed {processed} samples', flush=True)
    result = {
        'status': 'complete',
        'samples': processed,
        'skip_first': args.skip_first,
        'split': args.split,
        'selection_seed': args.selection_seed if args.split == 'train' else None,
        'checkpoint': str(args.checkpoint),
        'py_config': str(args.py_config),
        'protocol': dict(py_config=str(args.py_config), checkpoint=str(args.checkpoint),
                         split=args.split, limit=args.limit, skip_first=args.skip_first,
                         selection_seed=args.selection_seed, shifts=shifts,
                         selection_manifest=str(args.selection_manifest) if args.selection_manifest else None),
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'imageset': str(cfg.val_dataset_config.imageset),
        'imageset_sha256': file_sha256(cfg.val_dataset_config.imageset),
        'decision_ties': 'empty' if legacy_ties else 'foreground (matches argmax class order)',
        'head_empty_logit_offset': network.head.empty_logit_offset,
        'learned_empty_bias': (
            float(network.head.empty_bias.detach())
            if hasattr(network.head, 'empty_bias') else None),
        'empty_scalar': float(network.head.empty_scalar.detach()),
        'foreground_class_accuracy_on_gt_occupied': (
            100 * sums['foreground_correct_gt_occupied'] / denominators['voxel_occupied']),
        'gaussian_statistics': {
            key: sums[key] / denominators['gaussian']
            for key in ('opacity', 'scale_x', 'scale_y', 'scale_z',
                        'opacity_under_005', 'opacity_over_09',
                        'scale_over_039', 'semantic_max_logit')},
        'render_logit_means': {
            key: sums[key] / denominators['voxel_occupied' if 'gt_occupied' in key
                                           else 'voxel_empty']
            for key in ('foreground_max_gt_occupied', 'empty_logit_gt_occupied',
                        'foreground_max_gt_empty', 'empty_logit_gt_empty')},
        'empty_logit_shift_metrics': {str(shift): metrics(value)
                                      for shift, value in confusion.items()},
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
