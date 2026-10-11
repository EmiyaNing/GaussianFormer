#!/usr/bin/env python3
"""Oracle complementarity, instance-weighted motion, and block-size sensitivity."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from tools.analyze_sparseworld_region_pairs import scene_sum, bootstrap_weights
from tools.sparseworld_region_utils import ROOT, metric_from_hist, sha256, write_json


def main():
    root=ROOT
    records=json.loads((root/'manifest.json').read_text())['samples']
    data=dict(np.load(root/'region_counts.npz'))
    scene_names=data['scene_names'].tolist()
    sample_scene=data['sample_scene']
    joint=np.load(root/'joint_counts.npz')['joint']
    g,f,p=np.indices((18,18,18))
    selected=np.where((f==g)|(p==g),g,f)
    code=(g*18+selected).reshape(-1)
    oracle=np.stack([np.bincount(code,weights=row.reshape(-1),minlength=324).reshape(18,18)
                     for row in joint.reshape(-1,18,18,18)]).reshape(512,3,3,18,18)
    hist=scene_sum(oracle,sample_scene)
    point=metric_from_hist(hist.sum(0))
    weights=bootstrap_weights(150,np.random.default_rng(20261008))
    boot=(weights@hist.reshape(150,-1)).reshape(2000,3,3,18,18)
    bmetric=metric_from_hist(boot)
    oracle_result=dict(metrics=point,
                mIoU_ci95=np.percentile(bmetric['mIoU'],[2.5,97.5],axis=0).transpose(1,2,0),
                OccIoU_ci95=np.percentile(bmetric['OccIoU'],[2.5,97.5],axis=0).transpose(1,2,0),
                rule='Select GT if F or P is exactly correct; otherwise keep F label',
                interpretation='Accuracy complementarity reference, not a deployable system or an mIoU optimum')
    frame=pd.read_parquet(root/'instance_stats.parquet')
    frame['speed_bin']=np.where(frame.speed.isna(),3,np.digitize(frame.speed,[.5,2]))
    frame['scene_id']=sample_scene[frame.sample_position.to_numpy()]
    counts=np.zeros((150,3,4,5),float)
    for row in frame.itertuples():
        counts[row.scene_id,row.horizon-1,row.speed_bin,:4]+=np.asarray(
             [row.both_wrong,row.only_perception,row.only_forecast,row.both_correct])/row.voxels
        counts[row.scene_id,row.horizon-1,row.speed_bin,4]+=1
    total=counts.sum(0)
    percent=np.divide(total[...,:4],total[...,4,None],out=np.full_like(total[...,:4],np.nan),where=total[...,4,None]>0)*100
    boot_counts=(weights@counts.reshape(150,-1)).reshape(2000,3,4,5)
    proportions=np.divide(boot_counts[...,:4],boot_counts[...,4,None],
                          out=np.full_like(boot_counts[...,:4],np.nan),where=boot_counts[...,4,None]>0)*100
    instance_result=dict(instance_frames=len(frame),distinct_instances=frame.instance_token.nunique(),
              counts=total[...,4],state_percent=percent,
              state_ci95=np.nanpercentile(proportions,[2.5,97.5],axis=0).transpose(1,2,3,0),
              advantage=percent[...,1]-percent[...,2],
              advantage_ci95=np.nanpercentile(proportions[...,1]-proportions[...,2],[2.5,97.5],axis=0).transpose(1,2,0),
              weighting='Each class-compatible matched instance-frame equally weighted; scene bootstrap',
              limitation='Box-supported occupied voxels only; repeated instance tasks retained within scene clusters')
    sensitivity=data['sensitivity'].sum(0)
    size_result=dict(sizes_m=[1,2,4],count_order=['supported','F_advantage','P_advantage','tie','both_wrong_majority'],
                     counts=sensitivity,
                     percent=sensitivity[...,1:]/sensitivity[...,:1]*100,
                     threshold='At least 20 GT-occupied voxels in the block; results are descriptive')
    write_json(root/'extras.json',dict(status='complete',oracle=oracle_result,instance_motion=instance_result,
               block_sensitivity=size_result,
               replay_audit=json.loads((root/'replay_audit.json').read_text()),
               source_hashes={str(root/p):sha256(root/p) for p in ['protocol.json','manifest.json','scene_split.json']}))
    print('[extras] complete',flush=True)


if __name__=='__main__':
    main()
