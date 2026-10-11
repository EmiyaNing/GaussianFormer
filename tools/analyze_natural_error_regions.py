#!/usr/bin/env python3
"""Analyze E06 natural-region association and privileged correction."""
import argparse
import hashlib
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def interval(values, rng, repeats):
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, (repeats, values.size), replace=True).mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(draws, (.025, .975)).tolist(),
                n_scenes=int(values.size))


def control_vector(record):
    classes = np.asarray(record['source_class_counts'], dtype=np.float64)
    classes /= max(classes.sum(), 1.)
    bounds = record['bounds_m']
    return np.r_[record['selected_queries'], np.log1p(record['source_occupied_voxels']),
                 record['block_center_distance_m'],
                 .5 * (bounds[0] + bounds[1]), .5 * (bounds[2] + bounds[3]),
                 classes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--figure', required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2701)
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as handle:
        raw = json.load(handle)
    records = raw['natural_records']
    scenes = raw['split']['evaluation_scenes']
    rng = np.random.default_rng(args.bootstrap_seed)

    # Aggregate all blocks within a scene before bootstrap.
    natural_pool = defaultdict(lambda: [0, 0])
    for record in records:
        lag = record['evaluation_step'] - record['injection_step']
        for ring, values in record['rings'].items():
            key = (record['scene_token'], record['group'], lag, ring)
            natural_pool[key][0] += values['occupied_voxels']
            natural_pool[key][1] += values['occupied_errors']
    group_contrasts = {}
    for ring in ('R0', 'R1', 'R2', 'R3', 'R4'):
        for lag in range(6):
            differences = []
            for scene in scenes:
                high = natural_pool.get((scene, 'high', lag, ring))
                low = natural_pool.get((scene, 'low', lag, ring))
                if high and low and high[0] and low[0]:
                    differences.append((high[1] / high[0] - low[1] / low[0]) * 100)
            if differences:
                group_contrasts[f'{ring}|{lag}'] = interval(
                    differences, rng, args.bootstrap_repeats)

    # Nearest low-error block within the same scene/injection/evaluation.
    unique = {}
    for record in records:
        key = (record['sample_token'], record['injection_step'], record['block_id'])
        unique.setdefault(key, record)
    control_matrix = np.stack([control_vector(record) for record in unique.values()])
    control_mean, control_std = control_matrix.mean(0), control_matrix.std(0)
    control_std[control_std < 1e-6] = 1.
    matched_contrasts, matching = {}, {}
    for ring in ('R0', 'R1', 'R2', 'R3', 'R4'):
        for lag in range(6):
            scene_differences, distances = defaultdict(list), []
            keys = sorted({(r['scene_token'], r['injection_step'], r['evaluation_step'])
                           for r in records
                           if r['evaluation_step'] - r['injection_step'] == lag})
            pair_count = 0
            for key in keys:
                high = [r for r in records if
                        (r['scene_token'], r['injection_step'], r['evaluation_step']) == key
                        and r['group'] == 'high']
                low = [r for r in records if
                       (r['scene_token'], r['injection_step'], r['evaluation_step']) == key
                       and r['group'] == 'low']
                if not high or not low:
                    continue
                low_vectors = np.stack([(control_vector(r) - control_mean) / control_std
                                        for r in low])
                for high_record in high:
                    high_vector = (control_vector(high_record) - control_mean) / control_std
                    distances_to_low = np.linalg.norm(low_vectors - high_vector, axis=1)
                    index = int(distances_to_low.argmin())
                    low_record = low[index]
                    high_ring, low_ring = high_record['rings'][ring], low_record['rings'][ring]
                    if high_ring['occupied_voxels'] and low_ring['occupied_voxels']:
                        difference = (high_ring['occupied_errors'] /
                                      high_ring['occupied_voxels'] -
                                      low_ring['occupied_errors'] /
                                      low_ring['occupied_voxels']) * 100
                        scene_differences[high_record['scene_token']].append(difference)
                        distances.append(float(distances_to_low[index])); pair_count += 1
            if scene_differences:
                values = [float(np.mean(value)) for value in scene_differences.values()]
                name = f'{ring}|{lag}'
                matched_contrasts[name] = interval(values, rng, args.bootstrap_repeats)
                matching[name] = dict(pairs=pair_count,
                                      mean_standardized_distance=float(np.mean(distances)),
                                      controls='class composition + query count + occupied count + distance + block XY; exact scene/injection/evaluation')

    correction_pool = defaultdict(lambda: [0, 0, 0])
    for case in raw['correction_cases']:
        for ring, values in case['rings'].items():
            target = correction_pool[(case['scene_token'], ring)]
            target[0] += values['valid_voxels']
            target[1] += values['new_errors']
            target[2] += values['recovered_errors']
    correction_recovery = {}
    for name, rings in {'source': ('R0',), 'inner_adjacent': ('R1',),
                        'outer_adjacent': ('R2',), 'adjacent': ('R1', 'R2'),
                        'far': ('R3', 'R4')}.items():
        values = []
        for scene in scenes:
            total = [0, 0, 0]
            for ring in rings:
                part = correction_pool.get((scene, ring), (0, 0, 0))
                total = [a + b for a, b in zip(total, part)]
            if total[0]:
                values.append((total[2] - total[1]) / total[0] * 1e6)
        summary = interval(values, rng, args.bootstrap_repeats)
        summary.update(positive_scenes=sum(value > 0 for value in values),
                       negative_scenes=sum(value < 0 for value in values))
        correction_recovery[name] = summary

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for axis, ring in zip(axes, ('R0', 'R1', 'R2')):
        for group, color in (('low', '#2ca02c'), ('middle', '#ff7f0e'), ('high', '#d62728')):
            means = []
            for lag in range(6):
                values = []
                for scene in scenes:
                    counts = natural_pool.get((scene, group, lag, ring))
                    if counts and counts[0]:
                        values.append(counts[1] / counts[0] * 100)
                means.append(np.mean(values) if values else np.nan)
            axis.plot(range(6), means, marker='o', label=group, color=color)
        axis.set_title(ring); axis.set_xlabel('steps after early region')
        axis.set_ylabel('occupied semantic error (%)')
    axes[-1].legend()
    figure.tight_layout(); figure.savefig(args.figure, dpi=160); plt.close(figure)

    fit = raw['correction_runs']
    result = dict(
        task='E06', source=os.path.abspath(args.input),
        source_sha256=file_sha256(args.input), scene_count=len(scenes),
        bootstrap_unit='scene', bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed, group_contrasts=group_contrasts,
        matched_contrasts=matched_contrasts, matching=matching,
        correction_recovery=correction_recovery,
        correction_fit=dict(
            runs=len(fit), mean_translation_m=float(np.mean(
                [x['mean_translation_m'] for x in fit])),
            maximum_translation_m=float(max(x['max_translation_m'] for x in fit)),
            mean_fit_loss_start=float(np.mean([x['fit_loss_start'] for x in fit])),
            mean_fit_loss_end=float(np.mean([x['fit_loss_end'] for x in fit]))))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(os.path.abspath(args.output))


if __name__ == '__main__':
    main()
