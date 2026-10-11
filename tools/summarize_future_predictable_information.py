#!/usr/bin/env python3
"""Paired scene-level summary of the isolated H3 predictable-information run."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.experiment_future2current_identity import save_json


HORIZONS = ('1s', '2s', '3s')
ARMS = ('task_only', 'teacher_all', 'teacher_shuffled')


def miou(hist):
    hist = np.asarray(hist, dtype=np.float64)
    intersection = np.diag(hist)[:17]
    union = hist.sum(0)[:17] + hist.sum(1)[:17] - intersection
    iou = np.divide(intersection, union, out=np.full(17, np.nan), where=union > 0)
    return float(np.nanmean(iou) * 100)


def by_scene(hist, scenes):
    names = sorted(set(scenes))
    output = np.zeros((len(names), 3, 18, 18), dtype=np.float64)
    locations = {name: index for index, name in enumerate(names)}
    for row, scene in zip(hist, scenes):
        output[locations[scene]] += row
    return output


def run(root, repeats, seed):
    report = json.loads((root / 'results.json').read_text())
    settings = json.loads((root / 'settings.json').read_text())
    if report['status'] != 'complete' or set(report['arms']) != set(ARMS):
        raise RuntimeError('All three arms must finish before final summarization')
    scenes = settings['val_indices']
    manifest = json.loads(Path(settings['source_manifest']).read_text())
    index_to_scene = dict(zip(manifest['val']['indices'], manifest['val']['scenes']))
    scenes = [index_to_scene[index] for index in scenes]
    seeds = settings['seeds']
    grouped = {}
    for arm in ARMS:
        if set(report['arms'][arm]) != {str(seed) for seed in seeds}:
            raise RuntimeError(f'Missing seed for {arm}')
        grouped[arm] = {}
        for train_seed in seeds:
            record = report['arms'][arm][str(train_seed)]
            with np.load(record['rows']) as data:
                hist = data['student_hist']
            if hist.shape != (len(scenes), 3, 18, 18):
                raise RuntimeError(f'Unexpected confusion-matrix shape: {hist.shape}')
            grouped[arm][train_seed] = by_scene(hist, scenes)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(set(scenes)), size=(repeats, len(set(scenes))))
    comparisons = {}
    for left, right in [('teacher_all', 'task_only'),
                        ('teacher_all', 'teacher_shuffled')]:
        pair = f'{left}_minus_{right}'
        comparisons[pair] = {}
        for h_index, horizon in enumerate(HORIZONS):
            observed = np.mean([
                miou(grouped[left][s][:, h_index].sum(0)) -
                miou(grouped[right][s][:, h_index].sum(0)) for s in seeds])
            samples = []
            for draw in draws:
                samples.append(np.mean([
                    miou(grouped[left][s][draw, h_index].sum(0)) -
                    miou(grouped[right][s][draw, h_index].sum(0)) for s in seeds]))
            comparisons[pair][horizon] = dict(delta_miou_pp=float(observed),
                                               ci95_pp=[float(np.quantile(samples, .025)),
                                                        float(np.quantile(samples, .975))])
    output = dict(settings=str(root / 'settings.json'),
                  validation_samples=len(scenes), validation_scenes=len(set(scenes)),
                  bootstrap_repeats=repeats, comparisons=comparisons,
                  frozen={h: {side: report['frozen']['metrics'][side][h]
                              for side in ('baseline', 'teacher')} for h in HORIZONS},
                  arms={})
    for arm in ARMS:
        output['arms'][arm] = {}
        for horizon in HORIZONS:
            rows = [report['arms'][arm][str(s)] for s in seeds]
            output['arms'][arm][horizon] = dict(
                miou_mean=float(np.mean([r['metrics']['student'][horizon]['mIoU'] for r in rows])),
                miou_seeds=[r['metrics']['student'][horizon]['mIoU'] for r in rows],
                occ_iou_mean=float(np.mean([r['metrics']['student'][horizon]['OccIoU'] for r in rows])),
                strata={name: {
                    key: (None if rows[0]['strata'][horizon][name][key] is None else
                          float(np.mean([r['strata'][horizon][name][key] for r in rows])))
                    for key in ('feature_error', 'point_ce', 'point_accuracy',
                                'teacher_ce', 'baseline_ce')}
                    | {'queries': rows[0]['strata'][horizon][name]['queries']}
                    for name in rows[0]['strata'][horizon]})
    save_json(root / 'summary.json', output)
    print(root / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('out/future2current/predictable_information_1024x256'))
    parser.add_argument('--bootstrap-repeats', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260930)
    arguments = parser.parse_args()
    run(arguments.root, arguments.bootstrap_repeats, arguments.seed)
