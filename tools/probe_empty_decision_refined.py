"""Separate rendered-score confidence from contributor semantic consistency."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import joblib
from tools.analyze_empty_decision_mini import RUN, REGIONS, load_data, scores, bootstrap_delta, sha
from tools.center_support_common import write_json


def main():
    train, _ = load_data('train', 1)
    val, _ = load_data('val', 1)
    rng = np.random.default_rng(928)
    unique = np.unique(train['scene'])
    rng.shuffle(unique)
    cut = int(len(unique)*.75)
    fit = np.where((train['region']==0) & np.isin(train['scene'], unique[:cut]))[0]
    tune = np.where((train['region']==0) & np.isin(train['scene'], unique[cut:]))[0]
    if len(fit)>200000:
        fit = rng.choice(fit, 200000, replace=False)
    if len(tune)>100000:
        tune = rng.choice(tune, 100000, replace=False)
    manifest = json.loads((RUN/'probe_split_manifest.json').read_text())
    assert set(train['scenes'][i] for i in unique[:cut]) == set(manifest['fit_scenes'])
    base = list(range(5))+list(range(20,37))+[9,10,11]
    variants = dict(rendered_control=base, plus_support=base+[5,6,7,8],
        plus_contributor_semantics=base+[12,13,14], plus_image=base+[15,16,17,18,19])
    yt = (train['gt']<17) & (train['gt']==train['fg'])
    yv = (val['gt']<17) & (val['gt']==val['fg'])
    result = dict(status='running', source_sha256=sha(__file__),
        scene_manifest_sha256=sha(RUN/'probe_split_manifest.json'),
        target='correct_rescue', training_regions=['camera'],
        motivation='Exploratory refinement: control rendered-score confidence before testing contributor consistency',
        models={})
    predictions = {}
    for name, columns in variants.items():
        choices = []
        w = train['weight'][fit]
        for it in [40,100]:
            est = HistGradientBoostingClassifier(max_iter=it, learning_rate=.07,
                max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=1.,
                early_stopping=False, random_state=928)
            est.fit(train['X'][fit][:,columns],yt[fit],sample_weight=w/w.mean())
            p = est.predict_proba(train['X'][tune][:,columns])[:,1]
            ap = average_precision_score(yt[tune],p,sample_weight=train['weight'][tune])
            choices.append((ap,-it,est))
        ap, negit, est = max(choices,key=lambda a:(a[0],a[1]))
        p = est.predict_proba(val['X'][:,columns])[:,1]
        predictions[name] = p
        joblib.dump(est,RUN/f'refined_probe_{name}.joblib')
        metrics = {}
        for ri,r in enumerate(REGIONS):
            keep = val['region']==ri if ri<2 else np.ones(len(yv),bool)
            metrics[r] = scores(yv[keep],p[keep],val['weight'][keep])
        result['models'][name] = dict(columns=columns,selected_iterations=-negit,
            tuning_weighted_ap=ap,validation=metrics)
        print('refined',name,{r:m['auprc'] for r,m in metrics.items()},flush=True)
    result['comparisons'] = {}
    for name in list(variants)[1:]:
        result['comparisons'][name] = {}
        for ri,r in enumerate(REGIONS):
            keep = val['region']==ri if ri<2 else np.ones(len(yv),bool)
            result['comparisons'][name][r] = bootstrap_delta(yv[keep].astype(float),
                predictions['rendered_control'][keep],predictions[name][keep],
                val['weight'][keep],val['scene'][keep])
    result['status']='complete'
    write_json(RUN/'refined_probes.json',result)


if __name__=='__main__':
    with threadpool_limits(limits=8):
        main()
