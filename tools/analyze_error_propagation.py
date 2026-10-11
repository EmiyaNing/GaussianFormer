#!/usr/bin/env python3
"""Create scene-level summaries and bootstrap intervals for E02/E03."""
import argparse
import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np


REGION_GROUPS = {
    'source': ('R0',),
    'inner_adjacent': ('R1',),
    'outer_adjacent': ('R2',),
    'adjacent': ('R1', 'R2'),
    'far': ('R3', 'R4'),
    'outside': ('R1', 'R2', 'R3', 'R4'),
}
COUNT_FIELDS = (
    'valid_voxels', 'occupied_voxels', 'changed_voxels', 'new_errors',
    'recovered_errors', 'net_extra_errors')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def add_counts(target, case, rings):
    for ring in rings:
        values = case['rings'][ring]
        for field in COUNT_FIELDS:
            target[field] += int(values[field])


def scene_rows(cases, branch='C2'):
    grouped = {}
    for case in cases:
        if case['branch'] != branch:
            continue
        lag = case['evaluation_step'] - case['injection_step']
        for region, rings in REGION_GROUPS.items():
            key = (case['scene_token'], case['position_mode'], float(case['sigma_m']),
                   int(case['seed']), int(lag), region)
            target = grouped.setdefault(key, {field: 0 for field in COUNT_FIELDS})
            add_counts(target, case, rings)
    rows = []
    for key, counts in sorted(grouped.items()):
        scene, mode, sigma, seed, lag, region = key
        valid = max(counts['valid_voxels'], 1)
        changed = counts['changed_voxels']
        rows.append(dict(
            scene_token=scene, position_mode=mode, sigma_m=sigma, seed=seed,
            lag_steps=lag, lag_seconds=lag * .5, region=region, **counts,
            changed_per_million=changed / valid * 1e6,
            new_errors_per_million=counts['new_errors'] / valid * 1e6,
            recovered_per_million=counts['recovered_errors'] / valid * 1e6,
            net_extra_per_million=counts['net_extra_errors'] / valid * 1e6))
    return rows


def bootstrap_mean(values, rng, repeats):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return dict(mean=None, ci95=[None, None], n_scenes=0)
    draws = rng.choice(values, size=(repeats, values.size), replace=True).mean(1)
    return dict(mean=float(values.mean()),
                ci95=[float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                n_scenes=int(values.size))


def summaries(rows, seed, repeats):
    rng = np.random.default_rng(seed)
    keys = sorted({(r['position_mode'], r['sigma_m'], r['seed'],
                    r['lag_steps'], r['region']) for r in rows})
    output = {}
    fields = ('changed_per_million', 'new_errors_per_million',
              'recovered_per_million', 'net_extra_per_million')
    for key in keys:
        selected = [r for r in rows if
                    (r['position_mode'], r['sigma_m'], r['seed'],
                     r['lag_steps'], r['region']) == key]
        output['|'.join(map(str, key))] = {
            field: bootstrap_mean([r[field] for r in selected], rng, repeats)
            for field in fields}
    return output


def paired_temporal(rows, seed, repeats):
    """Compare future lags with lag zero within each scene, pooling seeds."""
    rng = np.random.default_rng(seed + 1)
    grouped = {}
    for row in rows:
        if row['sigma_m'] not in (.2, .4):
            continue
        key = (row['scene_token'], row['position_mode'], row['sigma_m'], row['region'])
        target = grouped.setdefault(key, {})
        for field in ('changed_per_million', 'net_extra_per_million'):
            target.setdefault((row['lag_steps'], field), []).append(row[field])
    results = {}
    for mode, sigma, region in sorted({key[1:] for key in grouped}):
        selected = [(key, value) for key, value in grouped.items()
                    if key[1:] == (mode, sigma, region)]
        for window, lags in [('early_future', (1, 2)), ('late_future', (3, 4, 5))]:
            for field in ('changed_per_million', 'net_extra_per_million'):
                differences = []
                for _, values in selected:
                    baseline = np.mean(values.get((0, field), [0.]))
                    future = [v for lag in lags for v in values.get((lag, field), [])]
                    if future:
                        differences.append(float(np.mean(future) - baseline))
                name = '|'.join(map(str, (mode, sigma, region, window, field)))
                results[name] = bootstrap_mean(differences, rng, repeats)
    return results


def seed_direction(rows):
    output = {}
    for mode in sorted({r['position_mode'] for r in rows}):
        for sigma in (.2, .4):
            for region in ('adjacent', 'far', 'outside'):
                values = {}
                for perturbation_seed in sorted({r['seed'] for r in rows}):
                    selected = [r for r in rows if r['position_mode'] == mode and
                                r['sigma_m'] == sigma and r['region'] == region and
                                r['seed'] == perturbation_seed and r['lag_steps'] > 0]
                    values[str(perturbation_seed)] = float(np.mean(
                        [r['net_extra_per_million'] for r in selected]))
                output['|'.join(map(str, (mode, sigma, region)))] = dict(
                    per_seed_net_extra_per_million=values,
                    all_positive=all(value > 0 for value in values.values()))
    return output


def future_bootstrap(rows, seed, repeats):
    """Pool all post-injection steps, preserving scenes as independent units."""
    rng = np.random.default_rng(seed + 2)
    pooled = {}
    for row in rows:
        if row['lag_steps'] == 0:
            continue
        key = (row['scene_token'], row['position_mode'], row['sigma_m'],
               row['seed'], row['region'])
        target = pooled.setdefault(key, {field: 0 for field in COUNT_FIELDS})
        for field in COUNT_FIELDS:
            target[field] += row[field]
    output = {}
    settings = sorted({key[1:] for key in pooled})
    for mode, sigma, perturbation_seed, region in settings:
        selected = [(key[0], value) for key, value in pooled.items()
                    if key[1:] == (mode, sigma, perturbation_seed, region)]
        rates = {scene: value['net_extra_errors'] / max(value['valid_voxels'], 1) * 1e6
                 for scene, value in selected}
        name = '|'.join(map(str, (mode, sigma, perturbation_seed, region)))
        output[name] = dict(
            net_extra_per_million=bootstrap_mean(list(rates.values()), rng, repeats),
            positive_scenes=sum(value > 0 for value in rates.values()),
            negative_scenes=sum(value < 0 for value in rates.values()),
            zero_scenes=sum(value == 0 for value in rates.values()))
    # The core CI pools the three fixed perturbation seeds within each scene first.
    for mode, sigma, region in sorted({(key[1], key[2], key[4]) for key in pooled}):
        scene_values = {}
        for (scene, row_mode, row_sigma, _, row_region), value in pooled.items():
            if (row_mode, row_sigma, row_region) == (mode, sigma, region):
                rate = value['net_extra_errors'] / max(value['valid_voxels'], 1) * 1e6
                scene_values.setdefault(scene, []).append(rate)
        averaged = [float(np.mean(values)) for values in scene_values.values()]
        name = '|'.join(map(str, (mode, sigma, 'all_seeds', region)))
        output[name] = dict(
            net_extra_per_million=bootstrap_mean(averaged, rng, repeats),
            positive_scenes=sum(value > 0 for value in averaged),
            negative_scenes=sum(value < 0 for value in averaged),
            zero_scenes=sum(value == 0 for value in averaged))
    return output


def plot_lag(rows, output):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for column, mode in enumerate(('independent', 'shared_translation')):
        for sigma, style in ((.2, '--'), (.4, '-')):
            for region, color in (('source', '#d62728'), ('adjacent', '#ff7f0e'),
                                  ('far', '#1f77b4')):
                means = []
                for lag in range(6):
                    selected = [r['changed_per_million'] for r in rows
                                if r['position_mode'] == mode and r['sigma_m'] == sigma and
                                r['lag_steps'] == lag and r['region'] == region]
                    means.append(np.mean(selected) if selected else np.nan)
                axes[0, column].plot(range(6), means, style, color=color,
                                     label=f'{region}, {sigma} m')
                net = []
                for lag in range(6):
                    selected = [r['net_extra_per_million'] for r in rows
                                if r['position_mode'] == mode and r['sigma_m'] == sigma and
                                r['lag_steps'] == lag and r['region'] == region]
                    net.append(np.mean(selected) if selected else np.nan)
                axes[1, column].plot(range(6), net, style, color=color,
                                     label=f'{region}, {sigma} m')
        axes[0, column].set_title(mode.replace('_', ' '))
        axes[0, column].set_ylabel('changed voxels / million')
        axes[1, column].set_ylabel('net extra errors / million')
        axes[1, column].set_xlabel('steps after injection (0.5 s each)')
        axes[1, column].axhline(0, color='black', linewidth=.7)
    axes[0, 1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


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
    rows = scene_rows(raw['cases'])
    result = dict(
        task=raw['task'], source=os.path.abspath(args.input), source_sha256=sha256(args.input),
        scene_count=len({r['scene_token'] for r in rows}),
        bootstrap_unit='scene', bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed, scene_rows=rows,
        bootstrap=summaries(rows, args.bootstrap_seed, args.bootstrap_repeats),
        paired_temporal=paired_temporal(rows, args.bootstrap_seed, args.bootstrap_repeats),
        seed_direction=seed_direction(rows),
        future_bootstrap=future_bootstrap(
            rows, args.bootstrap_seed, args.bootstrap_repeats))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    plot_lag(rows, args.figure)
    print(os.path.abspath(args.output))


if __name__ == '__main__':
    main()
