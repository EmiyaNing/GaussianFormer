#!/usr/bin/env python3
"""Execute E01--E03 local error-propagation diagnostics."""
import argparse
import hashlib
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.utils import custom_collate_fn_temporal
from misc.error_propagation import (
    evaluation_to_injection_transform, paired_damage, region_masks,
    sum_confusions, transform_points, validate_partition, voxel_centers)
from misc.sparseworld_diagnostics import confusion_matrix
from misc.sparseworld_diagnostics import (
    metrics_from_confusion, position_intervention_mask)
from tools.run_sparseworld_diagnostics import (
    build, json_safe, rasterize, sha256, to_cuda)


def select_distinct_scene_indices(dataset, allowed_scenes, count, pose_extremes=False):
    first = {}
    for index, (scene, _) in enumerate(dataset.keyframes):
        if scene in allowed_scenes and scene not in first:
            first[scene] = index
    records = []
    for scene in sorted(first):
        index = first[scene]
        frame = dataset.scene_infos[scene][dataset.keyframes[index][1]]
        token = frame['token']
        transforms = dataset._future_infos(token)
        # Pose ranking uses metadata only, never a prediction or error result.
        from dataset.sparseworld_trajectory import _pose
        relative = np.linalg.inv(_pose(frame)) @ _pose(transforms[-1])
        yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
        records.append((abs(yaw), scene, index))
    if pose_extremes and count >= 2:
        records.sort()
        selected = [records[0], records[-1]]
        remaining = [item for item in records[1:-1]
                     if item[1] not in {selected[0][1], selected[1][1]}]
        selected.extend(remaining[:count - 2])
    else:
        records.sort(key=lambda item: hashlib.sha256(item[1].encode()).hexdigest())
        selected = records[:count]
    return [item[2] for item in selected], [dict(scene_token=item[1],
                                                 absolute_six_step_yaw_rad=item[0])
                                            for item in selected]


def get_clean_sequence(network, result, snapshot, injection):
    replay = network._replay_forecast(snapshot, network.future_steps,
                                      collect_trace=True)
    expected_points = result['future_predictions'][injection:]
    expected_logits = result['future_logits'][injection:]
    if not all(torch.equal(a, b) for a, b in zip(expected_points, replay['points'])):
        raise AssertionError('D0 replay points are not bit-exact')
    if not all(torch.equal(a, b) for a, b in zip(expected_logits, replay['logits'])):
        raise AssertionError('D0 replay logits are not bit-exact')
    immediate_points = network._world(snapshot['positions']).flatten(1, 2)
    immediate_logits = snapshot['semantic'].flatten(1, 2)
    if not torch.equal(immediate_points, result['future_predictions'][injection - 1]):
        raise AssertionError('snapshot immediate points differ from original output')
    if not torch.equal(immediate_logits, result['future_logits'][injection - 1]):
        raise AssertionError('snapshot immediate logits differ from original output')
    return [(immediate_points, immediate_logits)] + list(zip(replay['points'], replay['logits'])), replay


def eligible_blocks(protocol, grid_points, snapshot, label, transforms, injection):
    centers = snapshot['positions_m'][0].mean(1)
    selected = []
    for block in protocol['region_protocol']['blocks']:
        bounds = block['bounds_m']
        xmin, xmax, ymin, ymax = bounds
        query_mask = ((centers[:, 0] >= xmin) & (centers[:, 0] < xmax) &
                      (centers[:, 1] >= ymin) & (centers[:, 1] < ymax))
        masks, _, _ = region_masks(grid_points, bounds, transforms,
                                   injection, injection)
        occupied = ((label != 17) & masks['R0']).sum()
        if (query_mask.sum() >= protocol['region_protocol']['minimum_active_queries'] and
                occupied >= protocol['region_protocol']['minimum_occupied_voxels']):
            selected.append((block, query_mask, int(occupied.item())))
    return selected


def plot_masks(path, masks_by_step, grid_z):
    steps = sorted(masks_by_step)
    figure, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
    axes = np.atleast_1d(axes)
    colors = np.zeros((*masks_by_step[steps[0]]['R0'].shape[:2], 3), dtype=np.float32)
    palette = dict(R0=(1., 0., 0.), R1=(1., .5, 0.), R2=(1., 1., 0.),
                   R3=(0., .7, 1.), R4=(.15, .15, .15))
    for axis, step in zip(axes, steps):
        image = colors.copy()
        for name, mask in masks_by_step[step].items():
            image[mask[:, :, grid_z].cpu().numpy()] = palette[name]
        axis.imshow(image.transpose(1, 0, 2), origin='lower')
        axis.set_title(f'future step {step}')
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


class RegionalAccumulator:
    def __init__(self):
        self.values = {}

    def add(self, key, value):
        if key not in self.values:
            self.values[key] = dict(
                clean_confusion=np.zeros((18, 18), dtype=np.int64),
                branch_confusion=np.zeros((18, 18), dtype=np.int64),
                valid_voxels=0, occupied_voxels=0, new_errors=0,
                recovered_errors=0, net_extra_errors=0,
                changed_voxels=0, observations=0)
        target = self.values[key]
        for name in ('clean_confusion', 'branch_confusion'):
            target[name] += np.asarray(value[name], dtype=np.int64)
        for name in ('valid_voxels', 'occupied_voxels', 'new_errors',
                     'recovered_errors', 'net_extra_errors', 'changed_voxels'):
            target[name] += int(value[name])
        target['observations'] += 1

    def results(self):
        output = {}
        for key, value in sorted(self.values.items(), key=lambda item: str(item[0])):
            clean = metrics_from_confusion(value['clean_confusion'])
            branch = metrics_from_confusion(value['branch_confusion'])
            name = '|'.join(map(str, key))
            output[name] = dict(
                **{field: value[field] for field in (
                    'valid_voxels', 'occupied_voxels', 'new_errors',
                    'recovered_errors', 'net_extra_errors', 'changed_voxels',
                    'observations')},
                clean_mIoU=clean['mIoU'], branch_mIoU=branch['mIoU'],
                delta_mIoU=branch['mIoU'] - clean['mIoU'],
                clean_OccIoU=clean['OccIoU'], branch_OccIoU=branch['OccIoU'],
                delta_OccIoU=branch['OccIoU'] - clean['OccIoU'],
                clean_confusion=value['clean_confusion'].tolist(),
                branch_confusion=value['branch_confusion'].tolist())
        return output


@torch.no_grad()
def run_e01(args):
    protocol = json.load(open(args.protocol, encoding='utf-8'))
    cfg, network, full_loader = build(args)
    indices, selection = select_distinct_scene_indices(
        full_loader.dataset, set(protocol['development_scenes']),
        args.max_samples, pose_extremes=True)
    loader = DataLoader(Subset(full_loader.dataset, indices), batch_size=1,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=custom_collate_fn_temporal)
    grid = voxel_centers(cfg.grid_shape, cfg.pc_range, cfg.grid_size, 'cuda')
    records, selected_samples = [], []
    figure_written = False
    started = time.perf_counter()
    for sample_number, cpu_data in enumerate(loader):
        data = to_cuda(cpu_data); images = data.pop('img')
        token = str(data['sample_idx'][0]); scene = str(data['scene_token'][0])
        transforms = data['future_ego_to_current'][0]
        result = network(imgs=images, metas=data,
                         forecast_snapshot_step=args.injection_steps)
        sample_record = dict(sample_token=token, scene_token=scene,
                             eligible_blocks={})
        for injection in args.injection_steps:
            snapshot = result['forecast_snapshots'][injection]
            sequence, _ = get_clean_sequence(network, result, snapshot, injection)
            label_at_injection = data['future_occ_labels'][0, injection - 1]
            blocks = eligible_blocks(protocol, grid, snapshot,
                                     label_at_injection, transforms, injection)
            sample_record['eligible_blocks'][str(injection)] = [
                item[0]['block_id'] for item in blocks]
            for block, query_mask, occupied in blocks[:args.max_blocks_per_sample]:
                masks_for_figure = {}
                for offset, (points, logits) in enumerate(sequence):
                    evaluation = injection + offset
                    label = data['future_occ_labels'][0, evaluation - 1]
                    clean_grid = rasterize(network, points, logits)[0].reshape(*cfg.grid_shape)
                    masks, _, injection_points = region_masks(
                        grid, block['bounds_m'], transforms, injection, evaluation)
                    if not validate_partition(masks):
                        raise AssertionError('R0--R4 masks do not partition the grid')
                    zero = paired_damage(clean_grid, clean_grid.clone(), label, masks)
                    global_hist = confusion_matrix(
                        clean_grid.cpu().numpy(), label.cpu().numpy())
                    if not np.array_equal(sum_confusions(zero, 'clean_confusion'), global_hist):
                        raise AssertionError('regional confusion does not sum to global confusion')
                    if any(value['changed_voxels'] or value['new_errors'] or
                           value['recovered_errors'] for value in zero.values()):
                        raise AssertionError('zero branch produced regional damage')
                    transform = evaluation_to_injection_transform(
                        transforms, injection, evaluation).to(grid)
                    sampled = grid[::40, ::40, ::8]
                    roundtrip = transform_points(
                        transform_points(sampled, transform), torch.linalg.inv(transform))
                    roundtrip_error = float((roundtrip - sampled).abs().max().item())
                    records.append(dict(
                        sample_token=token, scene_token=scene,
                        injection_step=injection, evaluation_step=evaluation,
                        block_id=block['block_id'], bounds_m=block['bounds_m'],
                        selected_queries=int(query_mask.sum().item()),
                        source_occupied_voxels=occupied,
                        partition_valid=True, zero_damage=True,
                        regional_confusion_equals_global=True,
                        transform_roundtrip_max_abs_m=roundtrip_error,
                        ring_voxels={name: int(mask.sum().item())
                                     for name, mask in masks.items()}))
                    if not figure_written and evaluation in (injection, 6):
                        masks_for_figure[evaluation] = masks
                if not figure_written and len(masks_for_figure) == 2:
                    plot_masks(args.figure_out, masks_for_figure, cfg.grid_shape[2] // 2)
                    figure_written = True
        selected_samples.append(sample_record)
    if not records:
        raise RuntimeError('no eligible local block was found in E01 samples')
    return dict(
        task='E01', status='passed', protocol=os.path.abspath(args.protocol),
        protocol_sha256=sha256(args.protocol), selected_by_pose_metadata=selection,
        samples=selected_samples, records=records,
        checks=dict(
            snapshot_and_d0_bit_exact=True, masks_partition_grid=True,
            regional_confusions_sum_to_global=True, zero_damage_is_zero=True,
            maximum_transform_roundtrip_abs_m=max(
                item['transform_roundtrip_max_abs_m'] for item in records)),
        evaluated_samples=len(selected_samples), evaluated_records=len(records),
        elapsed_seconds=time.perf_counter() - started,
        figure=os.path.abspath(args.figure_out), cuda_device=torch.cuda.get_device_name(0))


def deterministic_random_mask(query_mask, seed):
    count = int(query_mask.sum().item())
    candidates = torch.nonzero(~query_mask, as_tuple=False).flatten().cpu()
    if candidates.numel() < count:
        candidates = torch.arange(query_mask.numel())
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
    result = torch.zeros_like(query_mask)
    result[chosen.to(result.device)] = True
    return result


def branch_sequence(network, snapshot, intervention, clean_sequence):
    immediate_points = network._world(intervention['positions']).flatten(1, 2)
    immediate_logits = snapshot['semantic'].flatten(1, 2)
    replay = network._replay_forecast(
        intervention, network.future_steps, collect_trace=True)
    return [(immediate_points, immediate_logits)] + list(
        zip(replay['points'], replay['logits'])), replay


@torch.no_grad()
def run_e02(args):
    protocol = json.load(open(args.protocol, encoding='utf-8'))
    cfg, network, full_loader = build(args)
    indices, selection = select_distinct_scene_indices(
        full_loader.dataset, set(protocol['development_scenes']),
        args.max_samples, pose_extremes=False)
    loader = DataLoader(Subset(full_loader.dataset, indices), batch_size=1,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=custom_collate_fn_temporal)
    grid = voxel_centers(cfg.grid_shape, cfg.pc_range, cfg.grid_size, 'cuda')
    accumulator = RegionalAccumulator()
    cases, perturbations, sample_records = [], [], []
    started = time.perf_counter()
    for sample_number, cpu_data in enumerate(loader):
        data = to_cuda(cpu_data); images = data.pop('img')
        token = str(data['sample_idx'][0]); scene = str(data['scene_token'][0])
        transforms = data['future_ego_to_current'][0]
        result = network(imgs=images, metas=data,
                         forecast_snapshot_step=args.injection_steps)
        sample_record = dict(sample_token=token, scene_token=scene,
                             injections={})
        for injection in args.injection_steps:
            snapshot = result['forecast_snapshots'][injection]
            clean_sequence, clean_replay = get_clean_sequence(
                network, result, snapshot, injection)
            clean_grids = [rasterize(network, points, logits)[0].reshape(*cfg.grid_shape)
                           for points, logits in clean_sequence]
            label_at_injection = data['future_occ_labels'][0, injection - 1]
            candidates = eligible_blocks(protocol, grid, snapshot,
                                         label_at_injection, transforms, injection)
            candidates.sort(key=lambda item: hashlib.sha256(
                f'{token}:{injection}:{item[0]["block_id"]}'.encode()).hexdigest())
            blocks = candidates[:args.max_blocks_per_sample]
            sample_record['injections'][str(injection)] = [
                item[0]['block_id'] for item in blocks]
            for block, source_query_mask, occupied in blocks:
                masks_by_evaluation = {}
                labels = {}
                for offset in range(len(clean_sequence)):
                    evaluation = injection + offset
                    masks_by_evaluation[evaluation] = region_masks(
                        grid, block['bounds_m'], transforms,
                        injection, evaluation)[0]
                    labels[evaluation] = data['future_occ_labels'][0, evaluation - 1]
                for sigma in args.sigmas:
                    for mode in args.position_modes:
                        shared = mode == 'shared_translation'
                        for seed in args.seeds:
                            local_state, local_meta, _ = position_intervention_mask(
                                snapshot, sigma, source_query_mask[None], seed,
                                shared=shared, pc_range=network.pc_range,
                                metadata=dict(block_id=block['block_id'],
                                              bounds_m=block['bounds_m'], mode=mode,
                                              control='local'))
                            perturbations.append(dict(
                                sample_token=token, scene_token=scene,
                                injection_step=injection, **local_meta))
                            branches = {}
                            if 'C1' in args.branches:
                                immediate_points = network._world(
                                    local_state['positions']).flatten(1, 2)
                                immediate_logits = snapshot['semantic'].flatten(1, 2)
                                branches['C1'] = [(immediate_points, immediate_logits)] + clean_sequence[1:]
                            if 'C2' in args.branches:
                                local_sequence, _ = branch_sequence(
                                    network, snapshot, local_state, clean_sequence)
                                branches['C2'] = local_sequence
                            if 'C5' in args.branches:
                                random_mask = deterministic_random_mask(
                                    source_query_mask,
                                    seed + 100000 * injection + block['block_id'])
                                random_state, random_meta, _ = position_intervention_mask(
                                    snapshot, sigma, random_mask[None], seed,
                                    shared=shared, pc_range=network.pc_range,
                                    metadata=dict(block_id=block['block_id'],
                                                  bounds_m=block['bounds_m'], mode=mode,
                                                  control='random_dispersed'))
                                perturbations.append(dict(
                                    sample_token=token, scene_token=scene,
                                    injection_step=injection, **random_meta))
                                random_sequence, _ = branch_sequence(
                                    network, snapshot, random_state, clean_sequence)
                                branches['C5'] = random_sequence
                            for branch_name, sequence in branches.items():
                                for offset, (points, logits) in enumerate(sequence):
                                    evaluation = injection + offset
                                    if branch_name == 'C1' and offset > 0:
                                        branch_grid = clean_grids[offset]
                                    else:
                                        branch_grid = rasterize(
                                            network, points, logits)[0].reshape(*cfg.grid_shape)
                                    regional = paired_damage(
                                        clean_grids[offset], branch_grid,
                                        labels[evaluation], masks_by_evaluation[evaluation])
                                    ring_counts = {}
                                    for ring, value in regional.items():
                                        key = (branch_name, mode, sigma, seed,
                                               injection, evaluation, ring)
                                        accumulator.add(key, value)
                                        ring_counts[ring] = {
                                            name: value[name] for name in (
                                                'valid_voxels', 'occupied_voxels',
                                                'new_errors', 'recovered_errors',
                                                'net_extra_errors', 'changed_voxels')}
                                    cases.append(dict(
                                        sample_token=token, scene_token=scene,
                                        block_id=block['block_id'], bounds_m=block['bounds_m'],
                                        source_occupied_voxels=occupied,
                                        selected_queries=int(source_query_mask.sum().item()),
                                        branch=branch_name, position_mode=mode,
                                        sigma_m=sigma, seed=seed,
                                        injection_step=injection,
                                        evaluation_step=evaluation,
                                        rings=ring_counts))
        sample_records.append(sample_record)
    if not cases:
        raise RuntimeError('E02 found no eligible block/case')
    return dict(
        task=args.stage, status='completed', protocol=os.path.abspath(args.protocol),
        protocol_sha256=sha256(args.protocol), selected_samples=selection,
        sample_records=sample_records,
        experiment=dict(
            branches=dict(C0='clean replay',
                          C1='local position changed only for injection-step output',
                          C2='local position written into recurrent state',
                          C5='same number of spatially dispersed queries perturbed'),
            injection_steps=list(args.injection_steps), sigmas_m=list(args.sigmas),
            position_modes=list(args.position_modes), seeds=list(args.seeds),
            enabled_branches=list(args.branches),
            max_blocks_per_sample=args.max_blocks_per_sample),
        perturbations=perturbations, cases=cases,
        aggregate=accumulator.results(),
        evaluated_samples=len(sample_records), elapsed_seconds=time.perf_counter() - started,
        cuda_device=torch.cuda.get_device_name(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('E01', 'E02', 'E03'), required=True)
    parser.add_argument('--protocol', default='out/error_propagation/E00_protocol/protocol.json')
    parser.add_argument('--py-config', default='config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    parser.add_argument('--checkpoint', default='out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth')
    parser.add_argument('--metrics-out', required=True)
    parser.add_argument('--figure-out', default='')
    parser.add_argument('--max-samples', type=int, default=2)
    parser.add_argument('--max-blocks-per-sample', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--injection-steps', type=lambda x: tuple(int(v) for v in x.split(',')), default=(1, 2, 3))
    parser.add_argument('--sigmas', type=lambda x: tuple(float(v) for v in x.split(',')), default=(0., .1, .2, .4, .8))
    parser.add_argument('--position-modes', type=lambda x: tuple(v for v in x.split(',')), default=('independent', 'shared_translation'))
    parser.add_argument('--seeds', type=lambda x: tuple(int(v) for v in x.split(',')), default=(2701, 2702, 2703))
    parser.add_argument('--branches', type=lambda x: tuple(v for v in x.split(',')),
                        default=('C1', 'C2', 'C5'))
    args = parser.parse_args()
    if args.figure_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.figure_out)), exist_ok=True)
    result = run_e01(args) if args.stage == 'E01' else run_e02(args)
    output = os.path.abspath(args.metrics_out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(json_safe(result), handle, indent=2, allow_nan=False)
    print(output)


if __name__ == '__main__':
    main()
