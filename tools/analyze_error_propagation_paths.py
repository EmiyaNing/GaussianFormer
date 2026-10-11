#!/usr/bin/env python3
"""Scene-bootstrap analysis for E04 path interventions."""
import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np


REGIONS = {
    'source': ('R0',), 'inner_adjacent': ('R1',),
    'outer_adjacent': ('R2',), 'far': ('R3', 'R4'),
    'outside': ('R1', 'R2', 'R3', 'R4')}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(values, rng, repeats):
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, (repeats, values.size), replace=True).mean(1)
    return dict(mean=float(values.mean()),
                ci95=np.quantile(draws, (.025, .975)).tolist(),
                n_scenes=int(values.size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2701)
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as handle:
        raw = json.load(handle)
    counts = defaultdict(lambda: [0, 0, 0])
    traces = defaultdict(list)
    for case in raw['cases']:
        base = (case['scene_token'], case['position_mode'], case['seed'], case['branch'])
        for region, rings in REGIONS.items():
            target = counts[base + (region,)]
            for ring in rings:
                values = case['rings'][ring]
                target[0] += values['valid_voxels']
                target[1] += values['new_errors']
                target[2] += values['recovered_errors']
        for field, value in case['trace'].items():
            traces[(case['position_mode'], case['branch'], field)].append(value)
    rates = {key: (value[1] - value[2]) / max(value[0], 1) * 1e6
             for key, value in counts.items()}
    scenes = sorted({key[0] for key in rates})
    modes = sorted({key[1] for key in rates})
    branches = sorted({key[3] for key in rates})
    rng = np.random.default_rng(args.bootstrap_seed)
    branch_effects, paired_recovery = {}, {}
    for mode in modes:
        for region in REGIONS:
            for branch in branches:
                scene_values = []
                for scene in scenes:
                    fixed_seeds = [rates[(scene, mode, seed, branch, region)]
                                   for seed in raw['experiment']['seeds']]
                    scene_values.append(float(np.mean(fixed_seeds)))
                key = '|'.join((mode, branch, region))
                branch_effects[key] = bootstrap(
                    scene_values, rng, args.bootstrap_repeats)
            for branch in branches:
                if branch == 'P0_full':
                    continue
                recovery = []
                for scene in scenes:
                    clean = np.mean([rates[(scene, mode, seed, 'P0_full', region)]
                                     for seed in raw['experiment']['seeds']])
                    path = np.mean([rates[(scene, mode, seed, branch, region)]
                                    for seed in raw['experiment']['seeds']])
                    recovery.append(float(clean - path))
                key = '|'.join((mode, branch, region))
                paired_recovery[key] = bootstrap(
                    recovery, rng, args.bootstrap_repeats)
    trace_means = {'|'.join(key): float(np.mean(values))
                   for key, values in sorted(traces.items())}
    result = dict(
        task='E04', source=os.path.abspath(args.input),
        source_sha256=file_sha256(args.input), scene_count=len(scenes),
        bootstrap_unit='scene', bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed, branch_effects=branch_effects,
        paired_recovery=paired_recovery, trace_means=trace_means)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(os.path.abspath(args.output))


if __name__ == '__main__':
    main()
