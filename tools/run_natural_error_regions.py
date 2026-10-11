#!/usr/bin/env python3
"""E06: natural local errors and privileged local geometry correction."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.utils import custom_collate_fn_temporal
from misc.error_propagation import paired_damage, region_masks, voxel_centers
from misc.sparseworld_diagnostics import (
    clone_snapshot, occupied_voxel_centers, privileged_query_translation)
from tools.run_error_propagation import (
    RegionalAccumulator, eligible_blocks, get_clean_sequence,
    select_distinct_scene_indices)
from tools.run_sparseworld_diagnostics import build, json_safe, rasterize, sha256, to_cuda


def block_error(grid, label, mask):
    occupied = (label != 17) & mask
    valid = mask
    return dict(
        occupied_voxels=int(occupied.sum().item()),
        occupied_errors=int(((grid != label) & occupied).sum().item()),
        valid_voxels=int(valid.sum().item()),
        all_errors=int(((grid != label) & valid).sum().item()),
        occupied_error_rate=float(((grid != label) & occupied).sum().item() /
                                  max(occupied.sum().item(), 1)))


def group_name(value, thresholds):
    if value <= thresholds[0]:
        return 'low'
    if value >= thresholds[1]:
        return 'high'
    return 'middle'


@torch.no_grad()
def run(args):
    protocol = json.load(open(args.protocol, encoding='utf-8'))
    cfg, network, full_loader = build(args)
    indices, selection = select_distinct_scene_indices(
        full_loader.dataset, set(protocol['development_scenes']),
        args.max_samples, pose_extremes=False)
    split = args.calibration_scenes
    calibration_indices, evaluation_indices = indices[:split], indices[split:]
    calibration_scenes = [selection[i]['scene_token'] for i in range(split)]
    evaluation_scenes = [selection[i]['scene_token'] for i in range(split, len(indices))]
    grid_points = voxel_centers(cfg.grid_shape, cfg.pc_range, cfg.grid_size, 'cuda')
    calibration = []
    started = time.perf_counter()

    calibration_loader = DataLoader(
        Subset(full_loader.dataset, calibration_indices), batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn_temporal)
    for cpu_data in calibration_loader:
        data = to_cuda(cpu_data); images = data.pop('img')
        token = str(data['sample_idx'][0]); scene = str(data['scene_token'][0])
        result = network(imgs=images, metas=data,
                         forecast_snapshot_step=args.injection_steps)
        transforms = data['future_ego_to_current'][0]
        clean_grids = [rasterize(network, p, l)[0].reshape(*cfg.grid_shape)
                       for p, l in zip(result['future_predictions'], result['future_logits'])]
        for injection in args.injection_steps:
            snapshot = result['forecast_snapshots'][injection]
            label = data['future_occ_labels'][0, injection - 1]
            blocks = eligible_blocks(protocol, grid_points, snapshot, label,
                                     transforms, injection)
            for block, query_mask, _ in blocks:
                masks = region_masks(grid_points, block['bounds_m'], transforms,
                                    injection, injection)[0]
                error = block_error(clean_grids[injection - 1], label, masks['R0'])
                classes = torch.bincount(label[((label != 17) & masks['R0'])], minlength=18)
                calibration.append(dict(
                    sample_token=token, scene_token=scene, injection_step=injection,
                    block_id=block['block_id'], bounds_m=block['bounds_m'],
                    selected_queries=int(query_mask.sum().item()),
                    class_counts=classes[:17].cpu().tolist(), **error))
    values = np.asarray([x['occupied_error_rate'] for x in calibration])
    if values.size < 30:
        raise RuntimeError('too few calibration blocks for natural-error tertiles')
    thresholds = np.quantile(values, (1 / 3, 2 / 3)).tolist()

    natural_records, correction_cases, correction_runs = [], [], []
    accumulator = RegionalAccumulator()
    evaluation_loader = DataLoader(
        Subset(full_loader.dataset, evaluation_indices), batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn_temporal)
    for cpu_data in evaluation_loader:
        data = to_cuda(cpu_data); images = data.pop('img')
        token = str(data['sample_idx'][0]); scene = str(data['scene_token'][0])
        transforms = data['future_ego_to_current'][0]
        result = network(imgs=images, metas=data,
                         forecast_snapshot_step=args.injection_steps)
        all_clean_grids = [rasterize(network, p, l)[0].reshape(*cfg.grid_shape)
                           for p, l in zip(result['future_predictions'], result['future_logits'])]
        for injection in args.injection_steps:
            snapshot = result['forecast_snapshots'][injection]
            clean_sequence, _ = get_clean_sequence(network, result, snapshot, injection)
            label_at_injection = data['future_occ_labels'][0, injection - 1]
            blocks = eligible_blocks(protocol, grid_points, snapshot, label_at_injection,
                                     transforms, injection)
            high_candidates = []
            for block, query_mask, occupied in blocks:
                injection_masks = region_masks(
                    grid_points, block['bounds_m'], transforms, injection, injection)[0]
                early = block_error(all_clean_grids[injection - 1],
                                    label_at_injection, injection_masks['R0'])
                group = group_name(early['occupied_error_rate'], thresholds)
                classes = torch.bincount(
                    label_at_injection[((label_at_injection != 17) &
                                        injection_masks['R0'])], minlength=18)[:17]
                center_x = .5 * (block['bounds_m'][0] + block['bounds_m'][1])
                center_y = .5 * (block['bounds_m'][2] + block['bounds_m'][3])
                for evaluation in range(injection, network.future_steps + 1):
                    masks = region_masks(
                        grid_points, block['bounds_m'], transforms,
                        injection, evaluation)[0]
                    label = data['future_occ_labels'][0, evaluation - 1]
                    ring_errors = {ring: block_error(
                        all_clean_grids[evaluation - 1], label, mask)
                        for ring, mask in masks.items()}
                    natural_records.append(dict(
                        sample_token=token, scene_token=scene,
                        injection_step=injection, evaluation_step=evaluation,
                        block_id=block['block_id'], bounds_m=block['bounds_m'],
                        group=group, early_occupied_error_rate=early['occupied_error_rate'],
                        selected_queries=int(query_mask.sum().item()),
                        source_occupied_voxels=occupied,
                        block_center_distance_m=float(np.hypot(center_x, center_y)),
                        source_class_counts=classes.cpu().tolist(), rings=ring_errors))
                if group == 'high':
                    high_candidates.append((block, query_mask, injection_masks, early))
            high_candidates.sort(key=lambda item: hashlib.sha256(
                f'{token}:{injection}:{item[0]["block_id"]}:natural'.encode()).hexdigest())
            for block, query_mask, injection_masks, early in high_candidates[:args.corrections_per_injection]:
                targets = occupied_voxel_centers(
                    label_at_injection, network.pc_range, cfg.grid_size,
                    mask=injection_masks['R0'], max_points=args.max_match_points).to(images.device)
                source = snapshot['positions_m'][0, query_mask].flatten(0, 1)
                if source.numel() == 0 or targets.numel() == 0:
                    continue
                with torch.enable_grad():
                    delta, losses = privileged_query_translation(
                        source, targets, network.num_refines, args.correction_steps,
                        args.correction_lr, args.correction_cap_m)
                corrected = clone_snapshot(snapshot)
                full_delta = snapshot['positions'].new_zeros(
                    snapshot['positions'].shape[0], snapshot['positions'].shape[1], 1, 3)
                full_delta[0, query_mask, 0] = delta
                extent = full_delta.new_tensor(network.pc_range[3:]) - full_delta.new_tensor(
                    network.pc_range[:3])
                corrected['positions'] = snapshot['positions'] + full_delta / extent
                corrected['positions_m'] = snapshot['positions_m'] + full_delta
                replay = network._replay_forecast(corrected, network.future_steps)
                correction_runs.append(dict(
                    sample_token=token, scene_token=scene, injection_step=injection,
                    block_id=block['block_id'], selected_queries=int(query_mask.sum()),
                    early_occupied_error_rate=early['occupied_error_rate'],
                    mean_translation_m=float(delta.norm(dim=-1).mean()),
                    max_translation_m=float(delta.norm(dim=-1).max()),
                    fit_loss_start=losses[0], fit_loss_end=losses[-1]))
                for offset, (points, logits) in enumerate(zip(replay['points'], replay['logits'])):
                    evaluation = injection + 1 + offset
                    clean_grid = all_clean_grids[evaluation - 1]
                    corrected_grid = rasterize(network, points, logits)[0].reshape(*cfg.grid_shape)
                    masks = region_masks(grid_points, block['bounds_m'], transforms,
                                         injection, evaluation)[0]
                    label = data['future_occ_labels'][0, evaluation - 1]
                    regional = paired_damage(clean_grid, corrected_grid, label, masks)
                    rings = {}
                    for ring, value in regional.items():
                        accumulator.add(('privileged_local_correction', injection,
                                         evaluation, ring), value)
                        rings[ring] = {name: value[name] for name in (
                            'valid_voxels', 'occupied_voxels', 'new_errors',
                            'recovered_errors', 'net_extra_errors', 'changed_voxels')}
                    correction_cases.append(dict(
                        sample_token=token, scene_token=scene,
                        injection_step=injection, evaluation_step=evaluation,
                        block_id=block['block_id'], rings=rings))
    return dict(
        task='E06', status='completed', protocol=os.path.abspath(args.protocol),
        protocol_sha256=sha256(args.protocol),
        split=dict(calibration_scenes=calibration_scenes,
                   evaluation_scenes=evaluation_scenes,
                   overlap=sorted(set(calibration_scenes) & set(evaluation_scenes))),
        thresholds=dict(metric='occupied semantic error rate in source block',
                        low_middle=thresholds[0], middle_high=thresholds[1],
                        source='calibration scenes only'),
        calibration_records=calibration, natural_records=natural_records,
        correction_runs=correction_runs, correction_cases=correction_cases,
        correction_aggregate=accumulator.results(),
        experiment=dict(injection_steps=list(args.injection_steps),
                        region_groups=['R0', 'R1', 'R2', 'R3', 'R4'],
                        correction_cap_m=args.correction_cap_m,
                        correction_is_label_assisted=True),
        elapsed_seconds=time.perf_counter() - started,
        cuda_device=torch.cuda.get_device_name(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='out/error_propagation/E00_protocol/protocol.json')
    parser.add_argument('--py-config', default='config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    parser.add_argument('--checkpoint', default='out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth')
    parser.add_argument('--metrics-out', required=True)
    parser.add_argument('--max-samples', type=int, default=40)
    parser.add_argument('--calibration-scenes', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--injection-steps', type=lambda x: tuple(int(v) for v in x.split(',')),
                        default=(1, 2, 3))
    parser.add_argument('--corrections-per-injection', type=int, default=1)
    parser.add_argument('--correction-steps', type=int, default=20)
    parser.add_argument('--correction-lr', type=float, default=.05)
    parser.add_argument('--correction-cap-m', type=float, default=.4)
    parser.add_argument('--max-match-points', type=int, default=4096)
    args = parser.parse_args()
    if not 0 < args.calibration_scenes < args.max_samples:
        raise ValueError('calibration-scenes must split max-samples into two nonempty groups')
    result = run(args)
    output = os.path.abspath(args.metrics_out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(json_safe(result), handle, indent=2, allow_nan=False)
    print(output)


if __name__ == '__main__':
    main()
