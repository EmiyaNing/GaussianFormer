"""Additional-label diagnostic: are evidence signals useful outside camera mask?"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import joblib
from tools.analyze_empty_decision_mini import (RUN, REGIONS, GROUPS, MODELS, load_data,
    scores, bootstrap_delta, sha)
from tools.center_support_common import write_json


def main():
    import json
    train, _ = load_data('train', 1)
    val, _ = load_data('val', 1)
    manifest = json.loads((RUN/'probe_split_manifest.json').read_text())
    fit_scenes = [i for i, s in enumerate(train['scenes']) if s in set(manifest['fit_scenes'])]
    tune_scenes = [i for i, s in enumerate(train['scenes']) if s in set(manifest['tune_scenes'])]
    rng = np.random.default_rng(928)
    fit = np.where(np.isin(train['scene'], fit_scenes))[0]
    tune = np.where(np.isin(train['scene'], tune_scenes))[0]
    if len(fit) > 200000:
        fit = rng.choice(fit, 200000, replace=False)
    if len(tune) > 100000:
        tune = rng.choice(tune, 100000, replace=False)
    result = dict(status='running', source_sha256=sha(__file__),
        training_regions=['camera', 'camera_complement'], fit_rows=len(fit), tune_rows=len(tune),
        interpretation='Additional region labels: diagnostic only, not a fair camera-only generalization result',
        scene_manifest_sha256=sha(RUN/'probe_split_manifest.json'), models={})
    for task in ['correct_rescue', 'occupied']:
        yt = (train['gt'] < 17) & (train['gt'] == train['fg']) if task == 'correct_rescue' else train['gt'] < 17
        yv = (val['gt'] < 17) & (val['gt'] == val['fg']) if task == 'correct_rescue' else val['gt'] < 17
        result['models'][task] = {}
        predictions = {}
        for name in ['margin_position_class', 'all_class']:
            cols = sum([GROUPS[g] for g in MODELS[name]], [])
            choices = []
            w = train['weight'][fit]
            for it in [40, 100]:
                est = HistGradientBoostingClassifier(max_iter=it, learning_rate=.07,
                    max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=1.,
                    early_stopping=False, random_state=928)
                est.fit(train['X'][fit][:, cols], yt[fit], sample_weight=w/w.mean())
                p = est.predict_proba(train['X'][tune][:, cols])[:, 1]
                ap = average_precision_score(yt[tune], p, sample_weight=train['weight'][tune])
                choices.append((ap, -it, est))
            ap, negit, est = max(choices, key=lambda c: (c[0], c[1]))
            p = est.predict_proba(val['X'][:, cols])[:, 1]
            predictions[name] = p
            joblib.dump(est, RUN/f'all_region_probe_{task}_{name}.joblib')
            metrics = {}
            for ri, region in enumerate(REGIONS):
                keep = val['region'] == ri if ri < 2 else np.ones(len(yv), bool)
                metrics[region] = scores(yv[keep], p[keep], val['weight'][keep])
            result['models'][task][name] = dict(validation=metrics, selected_iterations=-negit,
                tuning_weighted_ap=ap)
            print('all-region', task, name, {r: m['auprc'] for r, m in metrics.items()}, flush=True)
        comparisons = {}
        for ri, region in enumerate(REGIONS):
            keep = val['region'] == ri if ri < 2 else np.ones(len(yv), bool)
            comparisons[region] = bootstrap_delta(yv[keep].astype(float),
                predictions['margin_position_class'][keep], predictions['all_class'][keep],
                val['weight'][keep], val['scene'][keep])
        result['models'][task]['paired_comparison'] = comparisons
        write_json(RUN/'all_region_probes_partial.json', result)
    result['status'] = 'complete'
    write_json(RUN/'all_region_probes.json', result)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
