#!/usr/bin/env python3
"""E04: attribute local error propagation to recurrent forecast paths."""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.utils import custom_collate_fn_temporal
from misc.error_propagation import paired_damage, region_masks, voxel_centers
from misc.sparseworld_diagnostics import position_intervention_mask
from tools.run_error_propagation import (
    RegionalAccumulator, eligible_blocks, get_clean_sequence,
    select_distinct_scene_indices)
from tools.run_sparseworld_diagnostics import build, json_safe, rasterize, sha256, to_cuda


def expanded_mask(source_mask, count):
    result = torch.zeros(count, dtype=torch.bool, device=source_mask.device)
    result[:source_mask.numel()] = source_mask
    return result


def masked_mean(values, mask):
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else 0.


def trace_difference(network, clean, branch, source_mask):
    input_mask = expanded_mask(source_mask, branch['input_features'].shape[1])
    output_mask = expanded_mask(source_mask, branch['output_features'].shape[1])
    input_feature = (branch['input_features'] - clean['input_features']).norm(dim=-1)[0]
    output_feature = (branch['output_features'] - clean['output_features']).norm(dim=-1)[0]
    clean_input_center = network._world(clean['input_positions']).mean(2)[0]
    branch_input_center = network._world(branch['input_positions']).mean(2)[0]
    clean_output_center = network._world(clean['output_positions']).mean(2)[0]
    branch_output_center = network._world(branch['output_positions']).mean(2)[0]
    input_position = (branch_input_center - clean_input_center).norm(dim=-1)
    output_position = (branch_output_center - clean_output_center).norm(dim=-1)
    return dict(
        selected_input_feature_l2=masked_mean(input_feature, input_mask),
        unselected_input_feature_l2=masked_mean(input_feature, ~input_mask),
        selected_output_feature_l2=masked_mean(output_feature, output_mask),
        unselected_output_feature_l2=masked_mean(output_feature, ~output_mask),
        selected_input_position_error_m=masked_mean(input_position, input_mask),
        unselected_input_position_error_m=masked_mean(input_position, ~input_mask),
        selected_output_position_error_m=masked_mean(output_position, output_mask),
        unselected_output_position_error_m=masked_mean(output_position, ~output_mask),
        raw_fused_ego_l2=float((branch['raw_fused_ego'] - clean['raw_fused_ego']).norm().item()),
        used_fused_ego_l2=float((branch['used_fused_ego'] - clean['used_fused_ego']).norm().item()),
        distance_prior_mean_abs=float(
            (branch['distance_prior'] - clean['distance_prior']).abs().mean().item()))


@torch.no_grad()
def run(args):
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
    cases, sample_records = [], []
    started = time.perf_counter()
    for cpu_data in loader:
        data = to_cuda(cpu_data); images = data.pop('img')
        token = str(data['sample_idx'][0]); scene = str(data['scene_token'][0])
        transforms = data['future_ego_to_current'][0]
        result = network(imgs=images, metas=data,
                         forecast_snapshot_step=args.injection_steps)
        sample_record = dict(sample_token=token, scene_token=scene, injections={})
        for injection in args.injection_steps:
            snapshot = result['forecast_snapshots'][injection]
            clean_sequence, clean_replay = get_clean_sequence(
                network, result, snapshot, injection)
            clean_grids = [rasterize(network, points, logits)[0].reshape(*cfg.grid_shape)
                           for points, logits in clean_sequence[1:]]
            candidates = eligible_blocks(
                protocol, grid, snapshot,
                data['future_occ_labels'][0, injection - 1], transforms, injection)
            candidates.sort(key=lambda item: hashlib.sha256(
                f'{token}:{injection}:{item[0]["block_id"]}'.encode()).hexdigest())
            blocks = candidates[:args.max_blocks_per_sample]
            sample_record['injections'][str(injection)] = [x[0]['block_id'] for x in blocks]
            clean_distance = [x['input_positions'] for x in clean_replay['trace']]
            clean_encoding = [x['interaction_positions'] for x in clean_replay['trace']]
            clean_fused = [x['used_fused_ego'] for x in clean_replay['trace']]
            for block, source_mask, occupied in blocks:
                masks = {}
                labels = {}
                for offset in range(len(clean_replay['points'])):
                    evaluation = injection + 1 + offset
                    masks[evaluation] = region_masks(
                        grid, block['bounds_m'], transforms, injection, evaluation)[0]
                    labels[evaluation] = data['future_occ_labels'][0, evaluation - 1]
                for sigma in args.sigmas:
                    for mode in args.position_modes:
                        for seed in args.seeds:
                            state, intervention, _ = position_intervention_mask(
                                snapshot, sigma, source_mask[None], seed,
                                shared=mode == 'shared_translation',
                                pc_range=network.pc_range,
                                metadata=dict(block_id=block['block_id'],
                                              bounds_m=block['bounds_m'], mode=mode))
                            branches = {}
                            if any(name in args.path_branches for name in (
                                    'P0_full', 'P4_clean_output_positions',
                                    'P5_clean_output_logits')):
                                full_replay = network._replay_forecast(
                                    state, network.future_steps, collect_trace=True)
                                for name in ('P0_full', 'P4_clean_output_positions',
                                             'P5_clean_output_logits'):
                                    if name in args.path_branches:
                                        branches[name] = full_replay
                            if 'P1_clean_distance_prior' in args.path_branches:
                                branches['P1_clean_distance_prior'] = network._replay_forecast(
                                    state, network.future_steps, collect_trace=True,
                                    distance_positions=clean_distance)
                            if 'P2_clean_position_encoding' in args.path_branches:
                                branches['P2_clean_position_encoding'] = network._replay_forecast(
                                    state, network.future_steps, collect_trace=True,
                                    encoding_positions=clean_encoding)
                            if 'P3_clean_fused_ego' in args.path_branches:
                                branches['P3_clean_fused_ego'] = network._replay_forecast(
                                    state, network.future_steps, collect_trace=True,
                                    fused_ego_features=clean_fused)
                            for branch_name, replay in branches.items():
                                for offset, (points, logits) in enumerate(
                                        zip(replay['points'], replay['logits'])):
                                    evaluation = injection + 1 + offset
                                    raster_points = (clean_replay['points'][offset]
                                                     if branch_name == 'P4_clean_output_positions'
                                                     else points)
                                    raster_logits = (clean_replay['logits'][offset]
                                                     if branch_name == 'P5_clean_output_logits'
                                                     else logits)
                                    branch_grid = rasterize(
                                        network, raster_points, raster_logits)[0].reshape(
                                        *cfg.grid_shape)
                                    regional = paired_damage(
                                        clean_grids[offset], branch_grid,
                                        labels[evaluation], masks[evaluation])
                                    ring_counts = {}
                                    for ring, value in regional.items():
                                        accumulator.add((branch_name, mode, sigma, seed,
                                                         injection, evaluation, ring), value)
                                        ring_counts[ring] = {name: value[name] for name in (
                                            'valid_voxels', 'occupied_voxels', 'new_errors',
                                            'recovered_errors', 'net_extra_errors',
                                            'changed_voxels')}
                                    cases.append(dict(
                                        sample_token=token, scene_token=scene,
                                        block_id=block['block_id'], bounds_m=block['bounds_m'],
                                        source_occupied_voxels=occupied,
                                        selected_queries=int(source_mask.sum().item()),
                                        branch=branch_name, position_mode=mode,
                                        sigma_m=sigma, seed=seed,
                                        injection_step=injection, evaluation_step=evaluation,
                                        rings=ring_counts,
                                        trace=trace_difference(
                                            network, clean_replay['trace'][offset],
                                            replay['trace'][offset], source_mask)))
        sample_records.append(sample_record)
    if not cases:
        raise RuntimeError('E04 found no eligible block/case')
    return dict(
        task='E04', status='completed', protocol=os.path.abspath(args.protocol),
        protocol_sha256=sha256(args.protocol), selected_samples=selection,
        sample_records=sample_records,
        experiment=dict(
            branches={
                'P0_full': 'ordinary local position error written into state',
                'P1_clean_distance_prior': 'distance bias reads clean positions',
                'P2_clean_position_encoding': 'forecast position encoder reads clean positions',
                'P3_clean_fused_ego': 'shared ego-to-query aggregation is replaced by clean value',
                'P4_clean_output_positions': 'rasterize with clean points and perturbed logits',
                'P5_clean_output_logits': 'rasterize with perturbed points and clean logits'},
            injection_steps=list(args.injection_steps), sigmas_m=list(args.sigmas),
            position_modes=list(args.position_modes), seeds=list(args.seeds),
            enabled_branches=list(args.path_branches),
            max_blocks_per_sample=args.max_blocks_per_sample),
        cases=cases, aggregate=accumulator.results(),
        evaluated_samples=len(sample_records), elapsed_seconds=time.perf_counter() - started,
        cuda_device=torch.cuda.get_device_name(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='out/error_propagation/E00_protocol/protocol.json')
    parser.add_argument('--py-config', default='config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    parser.add_argument('--checkpoint', default='out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth')
    parser.add_argument('--metrics-out', required=True)
    parser.add_argument('--max-samples', type=int, default=40)
    parser.add_argument('--max-blocks-per-sample', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--injection-steps', type=lambda x: tuple(int(v) for v in x.split(',')),
                        default=(1, 2, 3))
    parser.add_argument('--sigmas', type=lambda x: tuple(float(v) for v in x.split(',')),
                        default=(.4,))
    parser.add_argument('--position-modes', type=lambda x: tuple(v for v in x.split(',')),
                        default=('independent', 'shared_translation'))
    parser.add_argument('--seeds', type=lambda x: tuple(int(v) for v in x.split(',')),
                        default=(2701, 2702, 2703))
    parser.add_argument('--path-branches', type=lambda x: tuple(v for v in x.split(',')),
                        default=('P0_full', 'P1_clean_distance_prior',
                                 'P2_clean_position_encoding', 'P3_clean_fused_ego',
                                 'P4_clean_output_positions', 'P5_clean_output_logits'))
    args = parser.parse_args()
    result = run(args)
    output = os.path.abspath(args.metrics_out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(json_safe(result), handle, indent=2, allow_nan=False)
    print(output)


if __name__ == '__main__':
    main()
