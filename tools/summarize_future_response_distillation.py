#!/usr/bin/env python3
"""Scene-cluster bootstrap for the 2048/512 response-distillation experiment."""

import argparse
import json
from pathlib import Path

import numpy as np


def miou(hist):
    diag = np.diagonal(hist, axis1=-2, axis2=-1)
    union = hist.sum(axis=-1) + hist.sum(axis=-2) - diag
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = 100. * diag[..., :17] / union[..., :17]
    return np.nanmean(iou, axis=-1)


def occ_iou(intersection, union):
    return 100. * intersection / np.maximum(union, 1)


def main(args):
    root = Path(args.output_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = json.loads((root / 'report.json').read_text())
    scenes = np.asarray(manifest['validation']['scenes'])
    distinct = np.unique(scenes)
    scene_indices = [np.nonzero(scenes == scene)[0] for scene in distinct]
    arms = list(report['runs'][str(args.seed)])
    names = ['pretrained_student', 'teacher_native'] + arms
    filename = {'pretrained_student': 'pretrained_student.npz',
                'teacher_native': 'teacher_native.npz',
                **{arm: f'{args.seed}_{arm}.npz' for arm in arms}}
    bank = {}
    for name in names:
        with np.load(root / filename[name]) as values:
            hist = values['hist'].astype(np.float64)
            intersection = values['intersection'].astype(np.float64)
            union = values['union'].astype(np.float64)
        if hist.shape[:2] != (len(scenes), 3):
            raise RuntimeError(f'Confusion shape mismatch: {name}')
        bank[name] = dict(hist=np.stack([hist[indices].sum(axis=0)
                                         for indices in scene_indices]),
                          intersection=np.stack([intersection[indices].sum(axis=0)
                                                 for indices in scene_indices]),
                          union=np.stack([union[indices].sum(axis=0)
                                          for indices in scene_indices]))
    rng = np.random.default_rng(args.bootstrap_seed)
    counts = np.zeros((args.bootstrap_reps, len(distinct)), dtype=np.float64)
    for row in counts:
        row[:] = np.bincount(rng.integers(len(distinct), size=len(distinct)),
                             minlength=len(distinct))
    boot = {}
    for name, data in bank.items():
        hist = np.einsum('bs,shij->bhij', counts, data['hist'], optimize=True)
        intersection = counts @ data['intersection']
        union = counts @ data['union']
        boot[name] = dict(miou=miou(hist), occ_iou=occ_iou(intersection, union))
        official = (report['pretrained_student'] if name == 'pretrained_student'
                    else report['teacher_native'] if name == 'teacher_native'
                    else report['runs'][str(args.seed)][name]['metrics'])
        for index, horizon in enumerate(('1s', '2s', '3s')):
            expected = float(official[horizon]['mIoU'])
            actual = float(miou(data['hist'].sum(axis=0))[index])
            if abs(expected - actual) > 1e-6:
                raise RuntimeError(f'Metric reconstruction mismatch: {name}/{horizon}')
    comparisons = {}
    for name in names:
        if name == 'task_only':
            continue
        baseline = 'pretrained_student' if name == 'teacher_native' else 'task_only'
        comparisons[name] = {}
        for index, horizon in enumerate(('1s', '2s', '3s')):
            for key in ('miou', 'occ_iou'):
                delta = boot[name][key][:, index] - boot[baseline][key][:, index]
                data = bank[name]
                base = bank[baseline]
                if key == 'miou':
                    observed = float(miou(data['hist'].sum(0))[index] -
                                     miou(base['hist'].sum(0))[index])
                else:
                    observed = float(occ_iou(data['intersection'].sum(0), data['union'].sum(0))[index] -
                                     occ_iou(base['intersection'].sum(0), base['union'].sum(0))[index])
                comparisons[name].setdefault(horizon, {})[key] = dict(
                    baseline=baseline, observed_delta_pp=observed,
                    scene_bootstrap_95ci_pp=[float(x) for x in np.quantile(delta, [.025, .975])],
                    bootstrap_positive_fraction=float(np.mean(delta > 0)))
    summary = dict(seed=args.seed, sample_count=len(scenes), scene_count=len(distinct),
                   bootstrap_reps=args.bootstrap_reps,
                   bootstrap_seed=args.bootstrap_seed,
                   comparisons=comparisons)
    destination = (root / 'summary.json' if args.seed == 20260926 else
                   root / f'summary_{args.seed}.json')
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='out/future_distillation_response_based/2048x512')
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--bootstrap-reps', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260929)
    main(parser.parse_args())
