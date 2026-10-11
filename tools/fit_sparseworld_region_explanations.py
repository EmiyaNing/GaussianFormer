#!/usr/bin/env python3
"""Scene-separated multinomial explanations fitted to exact categorical cell counts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression

from tools.sparseworld_region_utils import ROOT, DYNAMIC, STATE_NAMES, write_json


def features(indices, reduced=False):
    # Input: horizon, GT class, distance, boundary, camera, speed, frustum.
    cardinality=[3,17,4,2,2,4,5]
    labels=['horizon','class','distance','boundary','camera','speed','frustum']
    cols=[]
    names=[]
    for axis,(size,label) in enumerate(zip(cardinality,labels)):
        if reduced and axis not in [0,1]:
            continue
        cols.append(np.eye(size,dtype=np.float64)[indices[:,axis]])
        names.extend([f'{label}/{level}' for level in range(size)])
    if not reduced:
        t=(indices[:,0]-1).astype(float)[:,None]
        for axis,label in [(5,'speed'),(6,'frustum')]:
            cols.append(np.eye(cardinality[axis])[indices[:,axis]]*t)
            names.extend([f'{label}/{level} x centered_time' for level in range(cardinality[axis])])
    return np.concatenate(cols,axis=1),names


def train(cov,C=1,reduced=False):
    table=cov[:,:,:17].sum(0)
    idx=np.argwhere(table.sum(-1)>0)
    counts=table[tuple(idx.T)]
    x,names=features(idx,reduced)
    row,state=np.nonzero(counts)
    weight=counts[row,state].astype(float)
    weight*=100000/weight.sum()
    model=LogisticRegression(C=C,max_iter=5000,tol=1e-5,solver='lbfgs',multi_class='multinomial')
    model.fit(x[row],state,sample_weight=weight)
    if model.n_iter_.max() >= model.max_iter:
        raise RuntimeError('Multinomial optimization did not converge')
    return model,names


def evaluate(model,cov,reduced=False):
    table=cov[:,:,:17].sum(0)
    idx=np.argwhere(table.sum(-1)>0)
    counts=table[tuple(idx.T)]
    x,_=features(idx,reduced)
    probability=model.predict_proba(x)
    loss=-(counts*np.log(np.clip(probability,1e-12,1))).sum()/counts.sum()
    freq=counts/counts.sum(1,keepdims=True)
    # Exact individual-observation multiclass Brier score, not cell-mean Brier.
    brier=((probability**2).sum(1)-2*(freq*probability).sum(1)+1)
    score=np.average(brier,weights=counts.sum(1))
    predicted=np.average(probability,axis=0,weights=counts.sum(1))
    observed=counts.sum(0)/counts.sum()
    return dict(log_loss=loss,brier=score,predicted_state_percent=predicted*100,
                observed_state_percent=observed*100,cells=len(idx),voxels=int(counts.sum()))


def conditional_effects(model,cov):
    table=cov[:,:,:17].sum(0)
    idx=np.argwhere(table.sum(-1)>0)
    counts=table[tuple(idx.T)].sum(1)
    result={}
    for label,axis,low,high in [('camera1_vs0',4,0,1),('boundary_vs_interior',3,0,1),
                              ('fast_vs_stopped',5,0,2),('future_only_vs_both',6,3,1)]:
        allowed=np.ones(len(idx),bool)
        if axis==5:
            allowed&=np.isin(idx[:,1],DYNAMIC)&(idx[:,5]<3)
        if axis==6:
            allowed&=idx[:,6]<4
        # Restrict to GT x distance strata with both factor levels present.
        support=set(map(tuple,idx[idx[:,axis]==low][:,[0,1,2]])) & set(map(tuple,idx[idx[:,axis]==high][:,[0,1,2]]))
        allowed&=np.asarray([tuple(row) in support for row in idx[:,[0,1,2]]])
        lowidx,highidx=idx.copy(),idx.copy()
        lowidx[:,axis]=low
        highidx[:,axis]=high
        pl=model.predict_proba(features(lowidx)[0])
        ph=model.predict_proba(features(highidx)[0])
        diff=((ph[:,1]-ph[:,2])-(pl[:,1]-pl[:,2]))*100
        out=[]
        for h in range(3):
            keep=allowed&(idx[:,0]==h)
            out.append(float(np.average(diff[keep],weights=counts[keep])) if keep.any() else None)
        result[label]=dict(advantage_difference_pp=out,
                          caveat='Point diagnostic after multivariate adjustment; no refit bootstrap interval')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    data=dict(np.load(args.root/'region_counts.npz'))
    cov=data['covariates']
    scenes=data['scene_names'].tolist()
    split=json.loads((args.root/'scene_split.json').read_text())
    discovery=np.asarray([i for i,s in enumerate(scenes) if s in split['discovery']])
    holdout=np.asarray([i for i,s in enumerate(scenes) if s in split['holdout']])
    folds=np.array_split(discovery,3)
    cv=[]
    for C in [.1,1,10]:
        losses=[]
        for val in folds:
            training=np.setdiff1d(discovery,val)
            model,_=train(cov[training],C)
            losses.append(evaluate(model,cov[val])['log_loss'])
        cv.append(dict(C=C,scene_cv_log_loss=losses,mean=float(np.mean(losses))))
        print('[multinomial CV]',C,losses,flush=True)
    best=min(cv,key=lambda r:r['mean'])['C']
    model,names=train(cov[discovery],best)
    # Freeze coefficients and tuning decision before calculating holdout scores.
    write_json(args.root/'explanation_model.json',dict(status='frozen',C=best,
        state_order=STATE_NAMES,feature_names=names,coef=model.coef_,intercept=model.intercept_,
        classes=model.classes_,cross_validation=cv,training_scenes=split['discovery'],
        sampling='Exact aggregated categorical cells with original voxel-frequency weights; no independent-voxel inferential claim',
        interactions=['speed x time','frustum x time'],n_iter=model.n_iter_))
    base,_=train(cov[discovery],1,reduced=True)
    output=dict(status='complete',chosen_C=best,cross_validation=cv,
          discovery=evaluate(model,cov[discovery]),holdout=evaluate(model,cov[holdout]),
          category_time_baseline_holdout=evaluate(base,cov[holdout],reduced=True),
          adjusted_effects_discovery=conditional_effects(model,cov[discovery]),
          adjusted_effects_holdout=conditional_effects(model,cov[holdout]),
          uncertainty='No coefficient or refit bootstrap intervals; scene intervals are supplied for fixed observed rules and standardized contrasts',
          meaning='Conditional association and predictive explanation only; input interventions not included')
    write_json(args.root/'explanation_summary.json',output)
    print('[multinomial] holdout',output['holdout'],flush=True)


if __name__=='__main__':
    main()
