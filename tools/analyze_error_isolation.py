#!/usr/bin/env python3
"""Scene-bootstrap summaries for E05 privileged isolation."""
import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np


REGIONS = {'source': ('R0',), 'outer_adjacent': ('R2',),
           'outside': ('R1', 'R2', 'R3', 'R4')}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2701)
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as handle:
        raw = json.load(handle)
    pool = defaultdict(lambda: dict(valid=0, new=0, recovered=0, occupied=0, hits=0))
    for case in raw['cases']:
        for region, rings in REGIONS.items():
            key = (case['scene_token'], case['position_mode'], case['seed'],
                   case['branch'], region)
            target = pool[key]
            for ring in rings:
                values = case['rings'][ring]
                target['valid'] += values['valid_voxels']
                target['new'] += values['new_errors']
                target['recovered'] += values['recovered_errors']
                target['occupied'] += values['occupied_voxels']
                target['hits'] += values['branch_occupied_hits']
    error_rate = {key: (value['new'] - value['recovered']) /
                  max(value['valid'], 1) * 1e6 for key, value in pool.items()}
    recall = {key: value['hits'] / max(value['occupied'], 1) * 100
              for key, value in pool.items()}
    scenes = sorted({key[0] for key in pool})
    modes = sorted({key[1] for key in pool})
    seeds = raw['experiment']['seeds']
    branches = list(raw['experiment']['branches'])
    rng = np.random.default_rng(args.bootstrap_seed)
    effects, recovery, per_seed_recovery, recall_delta = {}, {}, {}, {}
    for mode in modes:
        for region in REGIONS:
            for branch in branches:
                values = [np.mean([error_rate[(scene, mode, seed, branch, region)]
                                   for seed in seeds]) for scene in scenes]
                key = '|'.join((mode, branch, region))
                effects[key] = interval(values, rng, args.bootstrap_repeats)
                if branch == 'I0_no_isolation':
                    continue
                paired = []
                for scene in scenes:
                    baseline = np.mean([error_rate[(scene, mode, seed,
                                                   'I0_no_isolation', region)]
                                        for seed in seeds])
                    isolated = np.mean([error_rate[(scene, mode, seed, branch, region)]
                                        for seed in seeds])
                    paired.append(baseline - isolated)
                summary = interval(paired, rng, args.bootstrap_repeats)
                baseline_mean = effects['|'.join((mode, 'I0_no_isolation', region))]['mean']
                summary['recovery_percent_of_mean_baseline'] = (
                    summary['mean'] / baseline_mean * 100 if baseline_mean else None)
                recovery[key] = summary
                for perturbation_seed in seeds:
                    paired_seed = [
                        error_rate[(scene, mode, perturbation_seed,
                                    'I0_no_isolation', region)] -
                        error_rate[(scene, mode, perturbation_seed, branch, region)]
                        for scene in scenes]
                    seed_key = '|'.join(map(str, (mode, branch, region,
                                                  perturbation_seed)))
                    per_seed_recovery[seed_key] = interval(
                        paired_seed, rng, args.bootstrap_repeats)
        for branch in branches:
            paired_recall = []
            for scene in scenes:
                baseline = np.mean([recall[(scene, mode, seed,
                                            'I0_no_isolation', 'source')]
                                    for seed in seeds])
                isolated = np.mean([recall[(scene, mode, seed, branch, 'source')]
                                    for seed in seeds])
                paired_recall.append(isolated - baseline)
            recall_delta['|'.join((mode, branch))] = interval(
                paired_recall, rng, args.bootstrap_repeats)
    result = dict(
        task='E05', source=os.path.abspath(args.input),
        source_sha256=file_sha256(args.input), scene_count=len(scenes),
        bootstrap_unit='scene', bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed, effects=effects, recovery=recovery,
        per_seed_recovery=per_seed_recovery,
        source_occupied_recall_delta_percentage_points=recall_delta)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(os.path.abspath(args.output))


if __name__ == '__main__':
    main()
