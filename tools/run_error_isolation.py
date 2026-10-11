#!/usr/bin/env python3
"""E05: privileged isolation upper bounds for known perturbed queries."""
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
from misc.sparseworld_diagnostics import clone_snapshot, position_intervention_mask
from tools.run_error_propagation import (
    RegionalAccumulator, eligible_blocks, get_clean_sequence,
    select_distinct_scene_indices)
from tools.run_sparseworld_diagnostics import build, json_safe, rasterize, sha256, to_cuda


def state_from_delta(snapshot, delta_m, pc_range):
    result = clone_snapshot(snapshot)
    extent = delta_m.new_tensor(pc_range[3:]) - delta_m.new_tensor(pc_range[:3])
    result['positions'] = snapshot['positions'] + delta_m / extent
    result['positions_m'] = snapshot['positions_m'] + delta_m
    return result


def cap_delta(delta_m, maximum_m):
    xy = delta_m[..., 0, :2]
    scale = torch.clamp(float(maximum_m) / xy.norm(dim=-1, keepdim=True).clamp_min(1e-12),
                        max=1.)
    result = delta_m.clone()
    result[..., 0, :2] = xy * scale
    return result


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
            clean_fused = [x['used_fused_ego'] for x in clean_replay['trace']]
            for block, source_mask, occupied in blocks:
                masks, labels = {}, {}
                for offset in range(len(clean_replay['points'])):
                    evaluation = injection + 1 + offset
                    masks[evaluation] = region_masks(
                        grid, block['bounds_m'], transforms, injection, evaluation)[0]
                    labels[evaluation] = data['future_occ_labels'][0, evaluation - 1]
                for mode in args.position_modes:
                    for seed in args.seeds:
                        dirty, intervention, delta = position_intervention_mask(
                            snapshot, args.sigma, source_mask[None], seed,
                            shared=mode == 'shared_translation', pc_range=network.pc_range,
                            metadata=dict(block_id=block['block_id'],
                                          bounds_m=block['bounds_m'], mode=mode))
                        bounded_delta = cap_delta(delta, args.position_cap_m)
                        gated_delta = delta * args.writeback_weight
                        bounded = state_from_delta(snapshot, bounded_delta, network.pc_range)
                        gated = state_from_delta(snapshot, gated_delta, network.pc_range)
                        branches = {
                            'I0_no_isolation': network._replay_forecast(
                                dirty, network.future_steps),
                            'I1_restore_known_query': clean_replay,
                            'I2_clean_fused_ego': network._replay_forecast(
                                dirty, network.future_steps,
                                fused_ego_features=clean_fused),
                            'I3_cap_position_update': network._replay_forecast(
                                bounded, network.future_steps),
                            'I4_half_writeback': network._replay_forecast(
                                gated, network.future_steps),
                        }
                        displacement = {
                            'I0_no_isolation': float(delta[..., 0, :2].norm(dim=-1)[source_mask[None]].mean()),
                            'I1_restore_known_query': 0.,
                            'I2_clean_fused_ego': float(delta[..., 0, :2].norm(dim=-1)[source_mask[None]].mean()),
                            'I3_cap_position_update': float(bounded_delta[..., 0, :2].norm(dim=-1)[source_mask[None]].mean()),
                            'I4_half_writeback': float(gated_delta[..., 0, :2].norm(dim=-1)[source_mask[None]].mean()),
                        }
                        for branch_name, replay in branches.items():
                            for offset, (points, logits) in enumerate(
                                    zip(replay['points'], replay['logits'])):
                                evaluation = injection + 1 + offset
                                branch_grid = rasterize(network, points, logits)[0].reshape(
                                    *cfg.grid_shape)
                                regional = paired_damage(
                                    clean_grids[offset], branch_grid,
                                    labels[evaluation], masks[evaluation])
                                ring_counts = {}
                                for ring, value in regional.items():
                                    accumulator.add((branch_name, mode, args.sigma, seed,
                                                     injection, evaluation, ring), value)
                                    clean_hist = np.asarray(value['clean_confusion'])
                                    branch_hist = np.asarray(value['branch_confusion'])
                                    ring_counts[ring] = dict(
                                        **{name: value[name] for name in (
                                            'valid_voxels', 'occupied_voxels', 'new_errors',
                                            'recovered_errors', 'net_extra_errors',
                                            'changed_voxels')},
                                        clean_occupied_hits=int(clean_hist[:17, :17].sum()),
                                        branch_occupied_hits=int(branch_hist[:17, :17].sum()))
                                cases.append(dict(
                                    sample_token=token, scene_token=scene,
                                    block_id=block['block_id'], bounds_m=block['bounds_m'],
                                    source_occupied_voxels=occupied,
                                    selected_queries=int(source_mask.sum().item()),
                                    branch=branch_name, position_mode=mode,
                                    sigma_m=args.sigma, seed=seed,
                                    injection_step=injection, evaluation_step=evaluation,
                                    mean_initial_displacement_m=displacement[branch_name],
                                    query_count=int(snapshot['positions'].shape[1]),
                                    points_per_query=int(network.num_refines), rings=ring_counts))
        sample_records.append(sample_record)
    return dict(
        task='E05', status='completed', protocol=os.path.abspath(args.protocol),
        protocol_sha256=sha256(args.protocol), selected_samples=selection,
        sample_records=sample_records,
        experiment=dict(
            branches={
                'I0_no_isolation': '0.4 m error normally written into state',
                'I1_restore_known_query': 'known perturbed query restored to clean state',
                'I2_clean_fused_ego': 'shared fused ego restored at every future step',
                'I3_cap_position_update': f'known displacement norm capped at {args.position_cap_m} m',
                'I4_half_writeback': f'known displacement written with weight {args.writeback_weight}'},
            sigma_m=args.sigma, position_cap_m=args.position_cap_m,
            writeback_weight=args.writeback_weight,
            position_modes=list(args.position_modes), seeds=list(args.seeds),
            injection_steps=list(args.injection_steps), query_and_point_budget_fixed=True),
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
    parser.add_argument('--sigma', type=float, default=.4)
    parser.add_argument('--position-cap-m', type=float, default=.2)
    parser.add_argument('--writeback-weight', type=float, default=.5)
    parser.add_argument('--position-modes', type=lambda x: tuple(v for v in x.split(',')),
                        default=('independent', 'shared_translation'))
    parser.add_argument('--seeds', type=lambda x: tuple(int(v) for v in x.split(',')),
                        default=(2701, 2702, 2703))
    args = parser.parse_args()
    result = run(args)
    output = os.path.abspath(args.metrics_out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(json_safe(result), handle, indent=2, allow_nan=False)
    print(output)


if __name__ == '__main__':
    main()
