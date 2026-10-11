#!/usr/bin/env python3
"""Summarize the completed validation census without declaring train complete."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_full_regions_common import ROOT
from tools.sparseworld_region_utils import state_metrics,percentile_interval,write_json
from tools.analyze_sparseworld_region_pairs import attribute_counts


def main():
    records=[r for r in json.loads((ROOT/'manifest.json').read_text())['samples'] if r['split']=='val']
    scenes=sorted({r['scene'] for r in records});lookup={s:i for i,s in enumerate(scenes)}
    counts=np.zeros((len(scenes),3,4,4),np.int64)
    assert all((ROOT/'sample_stats'/(r['token']+'.npz')).exists() for r in records),'Validation collection incomplete'
    for r in records:
        with np.load(ROOT/'sample_stats'/(r['token']+'.npz')) as d:
            counts[lookup[r['scene']]]+=attribute_counts(d['covariates'][None])['speed'][0]
    point=state_metrics(counts.sum(0));w=np.random.default_rng(20261010).multinomial(len(scenes),np.ones(len(scenes))/len(scenes),size=2000)
    boot=state_metrics((w.astype(float)@counts.reshape(len(scenes),-1)).reshape(2000,3,4,4))
    result=dict(status='validation_complete_train_in_progress',sources=len(records),scenes=len(scenes),
                fast={k:dict(value=np.asarray(point[k])[:,2],ci95=percentile_interval(boot[k])[:,2]) for k in ('forecast_accuracy','perception_accuracy','advantage')},
                fast_voxels=point['total'][:,2],scope='Full validation only; does not satisfy the full train/val gate')
    write_json(ROOT/'validation_census_progress.json',result)
    print(json.dumps(result,default=lambda a:a.tolist(),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
