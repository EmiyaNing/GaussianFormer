#!/usr/bin/env python3
"""Run the paired SparseWorld T02--T04 diagnostics from one encoded state."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from mmengine import Config
from mmseg.models import build_segmentor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import model  # noqa: F401 - registry side effects
from dataset import get_dataloader
from misc.sparseworld_diagnostics import (
    InterventionAccumulator, bidirectional_set_distance, confusion_matrix,
    feature_intervention, metrics_from_confusion, occupied_voxel_centers,
    position_intervention, privileged_query_translation)


def csv_values(text, cast):
    return tuple(cast(item) for item in text.split(',') if item)


def json_safe(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', 'diff', '--binary'])
    return commit, hashlib.sha256(diff).hexdigest()


def to_cuda(data):
    return {key: (value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in data.items()}


def build(args, use_train=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; the current process cannot see a CUDA device')
    cfg = Config.fromfile(args.py_config)
    cfg.val_loader.batch_size = args.batch_size
    cfg.val_loader.num_workers = args.num_workers
    cfg.train_loader.batch_size = args.batch_size
    cfg.train_loader.num_workers = args.num_workers
    network = build_segmentor(cfg.model)
    network.init_weights()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    network.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    network.cuda().eval()
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, val_only=not use_train)
    return cfg, network, (train_loader if use_train else val_loader)


def rasterize(network, points, logits):
    return network.head.rasterizer(points, logits, group_size=network.num_refines)


def label_and_mask(cfg, data, step, batch_index):
    label = data['future_occ_labels'][batch_index, step]
    mask_name = cfg.get('occ3d_eval_mask', 'camera' if cfg.get('eval_mask_flag', True) else 'none')
    if mask_name == 'camera':
        mask = data['future_occ_cam_masks'][batch_index, step]
    elif mask_name == 'lidar':
        mask = data['future_occ_lidar_masks'][batch_index, step]
    else:
        mask = None
    return label, mask


def score(accumulator, key, cfg, data, network, step, points, logits,
          clean_points, tokens):
    grids = rasterize(network, points, logits)
    for batch_index, grid in enumerate(grids):
        label, mask = label_and_mask(cfg, data, step, batch_index)
        accumulator.add(
            key, grid.detach().cpu().numpy(), label.detach().cpu().numpy(),
            None if mask is None else mask.detach().cpu().numpy(),
            points[batch_index], clean_points[batch_index], tokens[batch_index])


def replay_exact_check(result, injection, clean):
    expected_points = result['future_predictions'][injection:]
    expected_logits = result['future_logits'][injection:]
    if len(expected_points) != len(clean['points']):
        raise AssertionError('clean replay horizon mismatch')
    for expected, actual in zip(expected_points, clean['points']):
        if not torch.equal(expected, actual):
            raise AssertionError('D0 point replay is not bit-exact')
    for expected, actual in zip(expected_logits, clean['logits']):
        if not torch.equal(expected, actual):
            raise AssertionError('D0 logit replay is not bit-exact')


def intervention_metadata(args, cfg, stage, processed, elapsed):
    commit, diff_hash = git_state()
    return dict(
        task=stage, experiment_id=args.experiment_id, exit_status=0,
        git_commit=commit, workspace_diff_sha256=diff_hash,
        config=os.path.abspath(args.py_config), config_sha256=sha256(args.py_config),
        checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        command=' '.join(subprocess.list2cmdline([item]) for item in os.sys.argv),
        perturbation_seeds=list(args.seeds), injection_steps=list(args.injection_steps),
        query_subsets=list(args.subsets), sigmas_m=list(args.sigmas),
        ego_condition='model-predicted ego feature; no ground-truth future ego is injected',
        query_point_budget='fixed baseline slots, 48 output points per active query',
        evaluated_batches=processed, elapsed_seconds=elapsed,
        grid_size_m=float(cfg.grid_size), cuda_device=torch.cuda.get_device_name(0))


@torch.no_grad()
def run_t02_t03(args, stage):
    cfg, network, loader = build(args)
    accumulator = InterventionAccumulator()
    perturbation_records = []
    path_records = []
    feature_std = None
    if stage == 'T03':
        if not args.feature_std:
            raise ValueError('T03 requires --feature-std generated from training snapshots')
        feature_std = np.load(args.feature_std)
    started = time.perf_counter()
    processed = 0
    for batch_index, cpu_data in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        data = to_cuda(cpu_data)
        images = data.pop('img')
        result = network(
            imgs=images, metas=data,
            forecast_snapshot_step=args.injection_steps)
        tokens = data.get('sample_idx', [f'batch{batch_index}:{i}' for i in range(images.shape[0])])
        if isinstance(tokens, str):
            tokens = [tokens]
        for injection in args.injection_steps:
            snapshot = result['forecast_snapshots'][injection]
            clean = network._replay_forecast(snapshot, network.future_steps, collect_trace=True)
            replay_exact_check(result, injection, clean)
            for local_step, (clean_points, clean_logits) in enumerate(zip(clean['points'], clean['logits'])):
                evaluation = injection + local_step
                score(accumulator, ('D0', injection, evaluation, 0., 'none', 0),
                      cfg, data, network, evaluation, clean_points, clean_logits,
                      clean_points, tokens)
            for subset in args.subsets:
                for sigma in args.sigmas:
                    for seed in args.seeds:
                        perturbed, record, _ = position_intervention(
                            snapshot, sigma, subset, seed, network.pc_range, args.block)
                        record.update(injection_step=injection, batch=batch_index)
                        perturbation_records.append(record)
                        # D1 changes only the output coordinates at the completed
                        # injection step and never enters a recurrent state.
                        d1_points = network._world(perturbed['positions']).flatten(1, 2)
                        d1_logits = snapshot['semantic'].flatten(1, 2)
                        score(accumulator, ('D1', injection, injection - 1, sigma, subset, seed),
                              cfg, data, network, injection - 1, d1_points, d1_logits,
                              result['future_predictions'][injection - 1], tokens)
                        d2 = network._replay_forecast(
                            perturbed, network.future_steps, collect_trace=True)
                        branches = {'D2': d2}
                        if stage == 'T03':
                            branches['D3'] = network._replay_forecast(
                                perturbed, network.future_steps,
                                distance_positions=[item['input_positions'] for item in clean['trace']],
                                collect_trace=True)
                            branches['D4'] = network._replay_forecast(
                                perturbed, network.future_steps,
                                encoding_positions=[item['interaction_positions'] for item in clean['trace']],
                                collect_trace=True)
                            # No image/feature sampling consumer exists after the
                            # forecast snapshot. D5 is therefore an explicit
                            # structural negative control and must equal D2.
                            branches['D5'] = d2
                            feature_state, feature_record = feature_intervention(
                                snapshot, sigma / max(float(cfg.grid_size), 1e-8),
                                feature_std, subset, seed, args.block)
                            feature_record.update(injection_step=injection, batch=batch_index)
                            perturbation_records.append(feature_record)
                            branches['D6'] = network._replay_forecast(
                                feature_state, network.future_steps, collect_trace=True)
                        for branch_name, branch in branches.items():
                            for local_step, (points, logits) in enumerate(zip(branch['points'], branch['logits'])):
                                evaluation = injection + local_step
                                score(accumulator,
                                      (branch_name, injection, evaluation, sigma, subset, seed),
                                      cfg, data, network, evaluation, points, logits,
                                      clean['points'][local_step], tokens)
                                clean_trace = clean['trace'][local_step]
                                branch_trace = branch['trace'][local_step]
                                path_records.append(dict(
                                    branch=branch_name, injection_step=injection,
                                    evaluation_step=evaluation, sigma_m=sigma,
                                    subset=subset, seed=seed, batch=batch_index,
                                    input_position_l2_m=float((
                                        network._world(branch_trace['input_positions']) -
                                        network._world(clean_trace['input_positions'])).norm(dim=-1).mean().item()),
                                    output_position_l2_m=float((
                                        network._world(branch_trace['output_positions']) -
                                        network._world(clean_trace['output_positions'])).norm(dim=-1).mean().item()),
                                    feature_l2=float((branch_trace['output_features'] -
                                                      clean_trace['output_features']).norm(dim=-1).mean().item()),
                                    distance_prior_l1=float((branch_trace['distance_prior'] -
                                                             clean_trace['distance_prior']).abs().mean().item())))
        processed += 1
    elapsed = time.perf_counter() - started
    return dict(
        manifest=intervention_metadata(args, cfg, stage, processed, elapsed),
        protocol=dict(
            D0='unchanged snapshot replay',
            D1='decode-coordinate-only; state remains immutable',
            D2='position changed before transition; all forecast consumers updated',
            D3='D2 noise with clean positions only for ego distance bias',
            D4='D2 noise with clean positions only for forecast position encoding',
            D5='no post-snapshot sampling consumer; exact D2 negative control',
            D6='feature-only noise normalized by external training channel std'),
        perturbations=perturbation_records, path_statistics=path_records,
        matrix=accumulator.results())


@torch.no_grad()
def feature_stats(args):
    cfg, network, loader = build(args, use_train=True)
    total = torch.zeros(network.position[0].out_features, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = 0
    tokens = []
    for batch_index, cpu_data in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        data = to_cuda(cpu_data)
        images = data.pop('img')
        result = network(imgs=images, metas=data,
                         forecast_snapshot_step=args.injection_steps)
        for snapshot in result['forecast_snapshots'].values():
            values = snapshot['features'].detach().double().flatten(0, 1).cpu()
            total += values.sum(0); total_sq += values.square().sum(0); count += values.shape[0]
        sample_tokens = data.get('sample_idx', ())
        tokens.extend([str(item) for item in sample_tokens])
    mean = total / max(count, 1)
    variance = (total_sq / max(count, 1) - mean.square()).clamp_min(0.)
    path = os.path.abspath(args.metrics_out).replace('.json', '.npy')
    np.save(path, variance.sqrt().numpy())
    return dict(task='T03-feature-stats', source_split='train', count=count,
                sample_tokens=tokens, feature_std_path=path,
                feature_std_sha256=sha256(path))


def sample_occ_error(grid, label, mask):
    hist = confusion_matrix(grid.detach().cpu().numpy(), label.detach().cpu().numpy(),
                            mask=None if mask is None else mask.detach().cpu().numpy())
    metrics = metrics_from_confusion(hist)
    return 1. - metrics['OccIoU'] / 100., hist


def controlled_correlation(rows):
    """Residual correlation after fixed, predeclared geometry controls."""
    if len(rows) < 3:
        return dict(n=len(rows), pearson=None, controlled_pearson=None)
    x = np.asarray([row['natural_oneway_point_error_m'] for row in rows])
    y = np.asarray([row['future_query_geometry_error_m'] for row in rows])
    raw = (float(np.corrcoef(x, y)[0, 1])
           if x.std() > 0 and y.std() > 0 else None)
    classes = np.asarray([row['predicted_class'] for row in rows])
    distance = np.asarray([row['center_distance_m'] for row in rows])
    span = np.asarray([row['local_geometry_span_m'] for row in rows])
    # Controls are fixed before looking at results: semantic class, 10 m range
    # bin, one-metre local span bin, and point count (constant in this model).
    categories = list(zip(classes, np.floor(distance / 10.).astype(int),
                          np.floor(span).astype(int)))
    unique = {value: index for index, value in enumerate(sorted(set(categories)))}
    design = np.zeros((len(rows), len(unique) + 1), dtype=np.float64)
    design[:, 0] = 1.
    for row_index, value in enumerate(categories):
        design[row_index, unique[value] + 1] = 1.
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    controlled = (float(np.corrcoef(residual_x, residual_y)[0, 1])
                  if residual_x.std() > 0 and residual_y.std() > 0 else None)
    return dict(n=len(rows), pearson=raw, controlled_pearson=controlled,
                controls='class + floor(distance/10m) + floor(local_span/1m); point_count fixed at 48')


def run_t04(args):
    cfg, network, loader = build(args)
    rows, correction_records = [], []
    baseline_hist = defaultdict(lambda: np.zeros((18, 18), dtype=np.int64))
    corrected_hist = defaultdict(lambda: np.zeros((18, 18), dtype=np.int64))
    started = time.perf_counter(); processed = 0
    for batch_index, cpu_data in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        data = to_cuda(cpu_data); images = data.pop('img')
        result = network(imgs=images, metas=data,
                         forecast_snapshot_step=args.injection_steps)
        tokens = data.get('sample_idx', [f'batch{batch_index}:{i}' for i in range(images.shape[0])])
        if isinstance(tokens, str): tokens = [tokens]
        if images.shape[0] != 1:
            raise ValueError('T04 privileged correction currently requires --batch-size 1')
        for injection in args.injection_steps:
            snapshot = result['forecast_snapshots'][injection]
            clean = network._replay_forecast(snapshot, network.future_steps)
            label, mask = label_and_mask(cfg, data, injection - 1, 0)
            targets = occupied_voxel_centers(label, network.pc_range, cfg.grid_size,
                                             mask=mask, max_points=args.max_match_points).to(images.device)
            source = snapshot['positions_m'][0].flatten(0, 1)
            natural_distance = bidirectional_set_distance(source, targets).item()
            semantic = snapshot['semantic'][0].argmax(-1)
            centers = snapshot['positions_m'][0].mean(1)
            local_span = (snapshot['positions_m'][0] - centers[:, None]).norm(dim=-1).mean(1)
            nearest = []
            for start in range(0, source.shape[0], 2048):
                nearest.append(torch.cdist(source[start:start + 2048], targets).amin(1))
            query_error = torch.cat(nearest).view(-1, network.num_refines).mean(1)
            for local_step, (points, logits) in enumerate(zip(clean['points'], clean['logits'])):
                evaluation = injection + local_step
                clean_grid = rasterize(network, points, logits)[0]
                future_label, future_mask = label_and_mask(cfg, data, evaluation, 0)
                future_error, hist = sample_occ_error(clean_grid, future_label, future_mask)
                baseline_hist[(injection, evaluation)] += hist
                future_targets = occupied_voxel_centers(
                    future_label, network.pc_range, cfg.grid_size,
                    mask=future_mask, max_points=args.max_match_points).to(images.device)
                persistent_points = points[0].view(-1, network.num_refines, 3)[:query_error.numel()]
                future_nearest = []
                flattened = persistent_points.flatten(0, 1)
                for start in range(0, flattened.shape[0], 2048):
                    future_nearest.append(torch.cdist(
                        flattened[start:start + 2048], future_targets).amin(1))
                future_query_error = torch.cat(future_nearest).view(
                    -1, network.num_refines).mean(1)
                for query_index in range(query_error.numel()):
                    rows.append(dict(
                        sample_token=str(tokens[0]), injection_step=injection,
                        evaluation_step=evaluation, query_index=query_index,
                        slot_id=int(snapshot['slot_ids'][query_index].item()),
                        predicted_class=int(torch.mode(semantic[query_index]).values.item()),
                        center_distance_m=float(centers[query_index, :2].norm().item()),
                        point_count=network.num_refines,
                        local_geometry_span_m=float(local_span[query_index].item()),
                        natural_oneway_point_error_m=float(query_error[query_index].item()),
                        future_query_geometry_error_m=float(future_query_error[query_index].item()),
                        sample_future_occupancy_error=float(future_error)))
            with torch.enable_grad():
                delta, losses = privileged_query_translation(
                    source, targets, network.num_refines, args.correction_steps,
                    args.correction_lr, args.correction_cap_m)
            corrected = {key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
                         for key, value in snapshot.items()}
            delta_group = delta[None, :, None]
            extent = snapshot['positions'].new_tensor(network.pc_range[3:]) - snapshot['positions'].new_tensor(network.pc_range[:3])
            corrected['positions'] = snapshot['positions'] + delta_group / extent
            corrected_rollout = network._replay_forecast(corrected, network.future_steps)
            moved = source.view(-1, network.num_refines, 3) + delta[:, None]
            after_distance = bidirectional_set_distance(moved.flatten(0, 1), targets).item()
            correction_records.append(dict(
                sample_token=str(tokens[0]), injection_step=injection,
                fit_steps=args.correction_steps, learning_rate=args.correction_lr,
                translation_cap_m=args.correction_cap_m,
                natural_symmetric_distance_m=float(natural_distance),
                corrected_symmetric_distance_m=float(after_distance), fit_losses=losses,
                mean_translation_m=float(delta.norm(dim=-1).mean().item()),
                max_translation_m=float(delta.norm(dim=-1).max().item())))
            for local_step, (points, logits) in enumerate(zip(
                    corrected_rollout['points'], corrected_rollout['logits'])):
                evaluation = injection + local_step
                grid = rasterize(network, points, logits)[0]
                future_label, future_mask = label_and_mask(cfg, data, evaluation, 0)
                _, hist = sample_occ_error(grid, future_label, future_mask)
                corrected_hist[(injection, evaluation)] += hist
        processed += 1
    correlations = {}
    for injection in args.injection_steps:
        for evaluation in range(injection, network.future_steps):
            selected = [row for row in rows if row['injection_step'] == injection and
                        row['evaluation_step'] == evaluation]
            correlations[f'{injection}->{evaluation}'] = controlled_correlation(selected)
    matrices = {}
    for key in baseline_hist:
        base = metrics_from_confusion(baseline_hist[key]); fixed = metrics_from_confusion(corrected_hist[key])
        matrices[f'{key[0]}->{key[1]}'] = dict(
            baseline=base, privileged_correction=fixed,
            delta_mIoU=fixed['mIoU'] - base['mIoU'],
            delta_OccIoU=fixed['OccIoU'] - base['OccIoU'],
            baseline_confusion=baseline_hist[key].tolist(),
            corrected_confusion=corrected_hist[key].tolist())
    return dict(
        manifest=intervention_metadata(args, cfg, 'T04', processed,
                                       time.perf_counter() - started),
        interpretation='label-assisted privileged diagnosis; never used in normal inference',
        natural_query_rows=rows, correlations=correlations,
        correction_runs=correction_records, matrix=matrices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('T02', 'T03', 'T04', 'feature-stats'))
    parser.add_argument('--py-config', default='config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    parser.add_argument('--checkpoint', default='out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth')
    parser.add_argument('--metrics-out', required=True)
    parser.add_argument('--experiment-id', default='')
    parser.add_argument('--max-batches', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--injection-steps', type=lambda x: csv_values(x, int), default=(1, 2, 3, 4, 5))
    parser.add_argument('--sigmas', type=lambda x: csv_values(x, float), default=(0., .1, .2, .4, .8))
    parser.add_argument('--subsets', type=lambda x: csv_values(x, str), default=('all', 'random10', 'block', 'block_shared'))
    parser.add_argument('--seeds', type=lambda x: csv_values(x, int), default=(1701, 1702, 1703))
    parser.add_argument('--block', type=lambda x: csv_values(x, float), default=(-20., 0., -10., 10.))
    parser.add_argument('--feature-std', default='')
    parser.add_argument('--correction-steps', type=int, default=20)
    parser.add_argument('--correction-lr', type=float, default=.05)
    parser.add_argument('--correction-cap-m', type=float, default=.8)
    parser.add_argument('--max-match-points', type=int, default=8192)
    args = parser.parse_args()
    if not args.experiment_id:
        args.experiment_id = f'{args.stage}_{time.strftime("%Y%m%d_%H%M%S")}'
    torch.manual_seed(0); np.random.seed(0)
    if args.stage == 'feature-stats': result = feature_stats(args)
    elif args.stage in ('T02', 'T03'): result = run_t02_t03(args, args.stage)
    else: result = run_t04(args)
    output = os.path.abspath(args.metrics_out); os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(json_safe(result), handle, indent=2, allow_nan=False)
    print(output)


if __name__ == '__main__':
    main()
