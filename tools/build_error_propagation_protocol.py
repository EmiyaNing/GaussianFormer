#!/usr/bin/env python3
"""Freeze the scene split and region protocol for error-propagation studies."""
import argparse
import hashlib
import json
import os
import subprocess
import sys

from mmengine import Config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import dataset  # noqa: F401
from dataset import OPENOCC_DATASET


def digest_rank(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', default='config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--development-scenes', type=int, default=40)
    parser.add_argument('--split-seed', type=int, default=2701)
    parser.add_argument('--previous-token-file', default='out/sparseworld_uncertainty/T02/d0_d2_scan_s10.json')
    args = parser.parse_args()

    cfg = Config.fromfile(args.py_config)
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    scene_samples = {}
    token_to_scene = {}
    for scene, index in validation.keyframes:
        token = validation.scene_infos[scene][index]['token']
        scene_samples.setdefault(scene, []).append(token)
        token_to_scene[token] = scene

    previously_used = []
    if os.path.isfile(args.previous_token_file):
        prior = json.load(open(args.previous_token_file, encoding='utf-8'))
        for value in prior.get('matrix', {}).values():
            previously_used.extend(value.get('sample_tokens', ()))
    previously_used = sorted(set(previously_used))
    forced_development = sorted({token_to_scene[token] for token in previously_used
                                 if token in token_to_scene})
    candidates = sorted((scene for scene in scene_samples
                         if scene not in forced_development),
                        key=lambda scene: digest_rank(scene, args.split_seed))
    development = forced_development + candidates[
        :max(args.development_scenes - len(forced_development), 0)]
    development = sorted(development)
    holdout = sorted(set(scene_samples) - set(development))
    if set(development) & set(holdout):
        raise AssertionError('development and holdout scenes overlap')
    if len(development) != args.development_scenes:
        raise ValueError('unable to construct requested development scene count')

    lower, upper = cfg.pc_range[:2], cfg.pc_range[3:5]
    block_size = 10.
    blocks = []
    block_id = 0
    x = float(lower[0])
    while x < float(upper[0]):
        y = float(lower[1])
        while y < float(upper[1]):
            blocks.append(dict(block_id=block_id, bounds_m=[x, x + block_size,
                                                            y, y + block_size]))
            block_id += 1
            y += block_size
        x += block_size

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    protocol = dict(
        task='E00', status='frozen', git_commit=commit,
        config=os.path.abspath(args.py_config), config_sha256=sha256(args.py_config),
        split_seed=args.split_seed,
        validation_samples=len(validation), validation_scenes=len(scene_samples),
        development_scenes=development, holdout_scenes=holdout,
        development_sample_count=sum(len(scene_samples[x]) for x in development),
        holdout_sample_count=sum(len(scene_samples[x]) for x in holdout),
        previously_used_sample_tokens=previously_used,
        forced_development_scenes=forced_development,
        samples_by_scene=scene_samples,
        region_protocol=dict(
            main_block_size_m=10., supplementary_block_sizes_m=[5., 20.],
            blocks=blocks, minimum_active_queries=5,
            minimum_occupied_voxels=50,
            rings_m=[dict(name='R0', inside_source=True),
                     dict(name='R1', distance=[0., 2.]),
                     dict(name='R2', distance=[2., 4.]),
                     dict(name='R3', distance=[4., 8.]),
                     dict(name='R4', distance=[8., None])],
            coordinate_tracking='world-fixed source block transformed between native future ego frames'),
        intervention_protocol=dict(
            injection_steps=[1, 2, 3], step_seconds=.5,
            position_sigmas_m=[0., .1, .2, .4, .8],
            primary_sigma_m=.4, stress_sigma_m=.8,
            position_modes=['independent', 'shared_translation'],
            feature_scales=[.25, .5, 1.], seeds=[2701, 2702, 2703]),
        gate=dict(
            require_non_source_damage=True,
            require_near_to_far_order_for_diffusion=True,
            require_signal_at_or_below_m=.4,
            require_all_three_seeds_same_direction=True,
            require_scene_bootstrap_interval_excludes_zero=True,
            require_effect_above_d0_repeat_noise_factor=3.))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
    print(output)


if __name__ == '__main__':
    main()
