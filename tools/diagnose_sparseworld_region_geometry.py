#!/usr/bin/env python3
"""Symmetric same-class tolerance diagnosis on 20 fixed holdout scenes."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import distance_transform_edt

from tools.sparseworld_region_utils import ROOT,write_json
from tools.analyze_sparseworld_region_pairs import bootstrap_weights


def main():
    root=ROOT
    records=json.loads((root/'manifest.json').read_text())['samples']
    scenes=json.loads((root/'scene_split.json').read_text())['holdout'][:20]
    selected=[]
    for scene in scenes:
        rows=[r for r in records if r['scene']==scene]
        selected.append(min(rows,key=lambda r:hashlib.sha256(('geometry:'+r['token']).encode()).hexdigest()))
    counts=np.zeros((20,3,2,3,4),np.int64)
    for i,rec in enumerate(selected):
        with np.load(root/'pairs'/(rec['token']+'.npz')) as d:
            for h in range(3):
                gt=d['GT'][h]
                valid=gt<18
                for c in range(17):
                    truth=valid&(gt==c)
                    ntruth=truth.sum()
                    dist_truth=distance_transform_edt(~truth,sampling=.4) if ntruth else None
                    for m,prefix in enumerate(['F','P']):
                        pred=valid&(d[prefix][h]==c)
                        npred=pred.sum()
                        inter=np.count_nonzero(pred&truth)
                        counts[i,h,m,0]+=[inter,npred,inter,ntruth]
                        if npred and ntruth:
                            dist_pred=distance_transform_edt(~pred,sampling=.4)
                            for t,radius in [(1,.4),(2,.8)]:
                                counts[i,h,m,t]+=[np.count_nonzero(dist_truth[pred]<=radius+1e-6),npred,
                                                  np.count_nonzero(dist_pred[truth]<=radius+1e-6),ntruth]
                        else:
                            counts[i,h,m,1:]+=[0,npred,0,ntruth]
        print('[geometry]',i+1,'/20',flush=True)
    total=counts.sum(0)
    weights=bootstrap_weights(20,np.random.default_rng(20261008))
    boot=(weights@counts.reshape(20,-1)).reshape(2000,3,2,3,4)
    precision=total[...,0]/total[...,1]*100
    recall=total[...,2]/total[...,3]*100
    bp=boot[...,0]/boot[...,1]*100
    br=boot[...,2]/boot[...,3]*100
    np.savez_compressed(root/'geometry_counts.npz',counts=counts)
    write_json(root/'geometry_diagnosis.json',dict(status='complete',scenes=20,samples=20,
        selected=[dict(scene=r['scene'],token=r['token'],position=r['position']) for r in selected],
        tolerances_m=[0,.4,.8],models=['F','P'],precision=precision,recall=recall,
        precision_ci95=np.percentile(bp,[2.5,97.5],axis=0).transpose(1,2,3,0),
        recall_ci95=np.percentile(br,[2.5,97.5],axis=0).transpose(1,2,3,0),
        meaning='Pooled occupied same-class precision/recall; both directions measured; strict labels retained',
        caveat='Many-to-one tolerance matching is a diagnosis, not IoU or evidence that geometric shifts are the sole cause'))


if __name__=='__main__':
    main()
