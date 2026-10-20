"""Frozen Gaussian logit head: train-selected empty calibration across masks."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: F401
from dataset import get_dataloader
from dataset.utils import custom_collate_fn_temporal
from model.head.gaussian_head import GaussianHead
from tools.center_support_common import write_json
from tools.diagnose_opus_v1_gaussian import (
    calibrated_predictions, file_sha256, metrics, training_selection)


def extended_metrics(confusion):
    result = metrics(confusion)
    occupied = confusion[:17].sum()
    empty = confusion[17].sum()
    result['occupied_missed_as_empty_percent'] = float(100 * confusion[:17, 17].sum() / occupied.clamp_min(1))
    result['empty_falsely_occupied_percent'] = float(100 * confusion[17, :17].sum() / empty.clamp_min(1))
    result['confusion'] = confusion.long().tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--selection-manifest', type=Path)
    parser.add_argument('--include-camera-invisible', action='store_true',
                        help='Also report the complement of the camera evaluation mask')
    parser.add_argument('--shifts', type=float, nargs='+', default=list(map(float, range(-8, 9))))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or len(set(args.shifts)) != len(args.shifts) or 0. not in args.shifts or not np.isfinite(args.shifts).all():
        parser.error('Positive sample count and unique finite shifts including zero are required')
    if args.selection_manifest and args.split != 'train':
        parser.error('Selection manifest is exclusively for training calibration')
    torch.manual_seed(928)
    random.seed(928)
    np.random.seed(928)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(args.py_config)
    if cfg.model.head.get('use_localaggprob', False):
        raise ValueError('This diagnostic requires a Gaussian logit/argmax head, not probability aggregation')
    if cfg.model.head.get('empty_logit_offset', 0):
        raise ValueError('Checkpoint must be evaluated without an existing extra empty offset')
    dataset = copy.deepcopy(cfg.val_dataset_config)
    if args.split == 'train':
        dataset.imageset = cfg.train_dataset_config.imageset
    loader_cfg = dict(cfg.val_loader, batch_size=1, num_workers=4)
    _, loader = get_dataloader(cfg.train_dataset_config, dataset, cfg.train_loader,
                              loader_cfg, val_only=True, seed=928)
    if args.split == 'train':
        indices = training_selection(len(loader.dataset), args.limit, 928,
                                     dataset.imageset, args.selection_manifest)
        loader = DataLoader(Subset(loader.dataset, indices), batch_size=1, shuffle=False,
                            num_workers=4, collate_fn=custom_collate_fn_temporal, pin_memory=True)
    network = build_segmentor(cfg.model).cuda().eval()
    if not isinstance(network.head, GaussianHead):
        raise ValueError('Expected GaussianHead or a compatible OPUS Gaussian subclass')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    checkpoint_epoch = checkpoint.get('epoch')
    del checkpoint
    mask_names = ['camera', 'all_valid']
    if args.include_camera_invisible:
        mask_names.append('camera_invisible')
    confusion = {mask: {b: torch.zeros(18, 18, dtype=torch.float64) for b in args.shifts}
                 for mask in mask_names}
    statistics = {mask: dict(occupied=0, empty=0, foreground_correct=0,
                            margin_occupied_sum=0., margin_empty_sum=0.)
                  for mask in confusion}
    class_counts = {mask: torch.zeros(17, 2, dtype=torch.float64) for mask in confusion}
    start = time.monotonic()
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            if processed >= args.limit:
                break
            batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            images = batch.pop('img')
            output = network(imgs=images, metas=batch)
            logits = output['pred_occ'][-1][0]
            if logits.shape != (18, 200 * 200 * 16):
                raise ValueError(f'Unexpected dense Occ3D output: {tuple(logits.shape)}')
            fg_max, fg_class = logits[:17].max(dim=0)
            empty_score = logits[17]
            gt = output['sampled_label'][0].reshape(-1).long()
            valid = (gt >= 0) & (gt < 18)
            masks = dict(camera=valid & batch['occ_cam_mask'][0].reshape(-1).bool(), all_valid=valid)
            if args.include_camera_invisible:
                masks['camera_invisible'] = valid & ~batch['occ_cam_mask'][0].reshape(-1).bool()
            predictions = {b: calibrated_predictions(fg_max, fg_class, empty_score, b) for b in args.shifts}
            if not torch.equal(predictions[0.], output['final_occ'][0].reshape(-1)):
                raise ValueError('Zero-offset diagnostic does not reproduce native model prediction')
            margin = fg_max - empty_score
            for mask_name, mask in masks.items():
                occupied, empty = mask & (gt < 17), mask & (gt == 17)
                stats = statistics[mask_name]
                stats['occupied'] += int(occupied.sum())
                stats['empty'] += int(empty.sum())
                stats['foreground_correct'] += int(((fg_class == gt) & occupied).sum())
                stats['margin_occupied_sum'] += float(margin[occupied].sum())
                stats['margin_empty_sum'] += float(margin[empty].sum())
                class_counts[mask_name][:, 0] += torch.bincount(gt[occupied], minlength=17).cpu().double()
                missed = occupied & (predictions[0.] == 17)
                class_counts[mask_name][:, 1] += torch.bincount(gt[missed], minlength=17).cpu().double()
                for b, pred in predictions.items():
                    confusion[mask_name][b] += torch.bincount(gt[mask] * 18 + pred[mask], minlength=324).reshape(18, 18).cpu().double()
            processed += 1
            if processed % 100 == 0:
                print(f'processed {processed}/{args.limit}', flush=True)
    if processed != args.limit:
        raise ValueError(f'Only processed {processed}, expected {args.limit}')
    reports = {}
    for mask_name, stats in statistics.items():
        reports[mask_name] = dict(
            empty_logit_shift_metrics={str(b): extended_metrics(c) for b, c in confusion[mask_name].items()},
            foreground_accuracy_on_gt_occupied=100 * stats['foreground_correct'] / max(stats['occupied'], 1),
            mean_fg_minus_empty_on_gt_occupied=stats['margin_occupied_sum'] / max(stats['occupied'], 1),
            mean_fg_minus_empty_on_gt_empty=stats['margin_empty_sum'] / max(stats['empty'], 1),
            default_per_class_gt_and_empty_misses=class_counts[mask_name].long().tolist())
    result = dict(status='complete', finished_at=datetime.now().astimezone().isoformat(),
        samples=processed, split=args.split, elapsed_seconds=time.monotonic()-start,
        protocol=dict(py_config=args.py_config, checkpoint=args.checkpoint, split=args.split,
            limit=args.limit, shifts=args.shifts, selection_seed=928,
            selection_manifest=str(args.selection_manifest) if args.selection_manifest else None,
            include_camera_invisible=args.include_camera_invisible,
            calibration='Train camera-mask mIoU only; the selected shift also used unmodified without camera mask',
            offset='Subtract b from the rendered empty logit; positive b favors occupancy',
            ties='Foreground wins ties, identical to native class-index argmax'),
        checkpoint_sha256=file_sha256(args.checkpoint), checkpoint_epoch=checkpoint_epoch,
        config_sha256=file_sha256(args.py_config), resolved_config_sha256=__import__('hashlib').sha256(cfg.pretty_text.encode()).hexdigest(),
        imageset=str(dataset.imageset), imageset_sha256=file_sha256(dataset.imageset),
        diagnostic_source_sha256=file_sha256(__file__),
        empty_scalar=float(network.head.empty_scalar.detach()),
        learned_empty_bias=(float(network.head.empty_bias.detach())
                            if hasattr(network.head, 'empty_bias') else None),
        native_zero_offset_parity_passed=True, masks=reports)
    write_json(args.out, result)
    print(f'Completed {processed} {args.split} frames: {args.out}', flush=True)


if __name__ == '__main__':
    main()
