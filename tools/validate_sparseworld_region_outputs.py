#!/usr/bin/env python3
"""Independent cross-artifact validation of the completed paired experiment."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from tools.sparseworld_region_utils import ROOT,write_json
from tools.analyze_sparseworld_region_pairs import scene_sum


def main():
    root=ROOT
    data=dict(np.load(root/'region_counts.npz'))
    joint=np.load(root/'joint_counts.npz')['joint']
    basic=data['basic']
    assert basic.shape==(512,3,3,3,2,4)
    assert np.array_equal(basic[:,:,:,0],basic[:,:,:,1]+basic[:,:,:,2])
    assert np.array_equal(basic.sum(-1)[...,0],basic.sum(-1)[...,1])
    assert np.array_equal(joint.sum(axis=(-3,-2,-1)),basic[:,:,:,0,0].sum(-1))
    old_root=Path('out/future2current/native_2048x512')
    for model,path,axis in [('F','baseline_replay.npz',-1),('P','teacher.npz',-2)]:
        hist=joint[:,:,0].sum(axis=axis)
        old=dict(np.load(old_root/path))
        assert np.array_equal(hist,old['hist'])
        assert np.array_equal(hist[:,:,:17,:17].sum(axis=(-2,-1)),old['intersection'])
        assert np.array_equal(hist.sum(axis=(-2,-1))-hist[:,:,17,17],old['union'])
    cov=data['covariates']
    state_scene=scene_sum(basic[:,:,0,0,0],data['sample_scene'])
    assert np.array_equal(cov.sum(axis=(2,3,4,5,6,7)),state_scene)
    gt_scene=scene_sum(joint[:,:,0].sum(axis=(-2,-1)),data['sample_scene'])
    assert np.array_equal(cov.sum(axis=(3,4,5,6,7,8)),gt_scene)
    errors=data['errors']
    assert np.array_equal(errors[:,:,:5].sum(-1),basic[:,:,0,1,0,0])
    assert np.array_equal(errors[:,:,5:].sum(-1),basic[:,:,0,2,0,0])
    summary=json.loads((root/'summary.json').read_text())
    proportions=np.asarray(summary['splits']['all']['states']['proportions']['value'])
    assert np.allclose(proportions.sum(-1),100)
    advantage=np.asarray(summary['splits']['all']['states']['advantage']['value'])
    assert np.allclose(advantage,proportions[...,1]-proportions[...,2])
    for entry in summary['splits'].values():
        for model_name in ['F','P']:
            for measure in ['mIoU','OccIoU']:
                point=np.asarray(entry['metrics'][model_name][measure])
                interval=np.asarray(entry['metrics'][model_name][measure+'_ci95'])
                assert interval.shape==(*point.shape,2)
                assert np.all((point>=interval[...,0])&(point<=interval[...,1])), 'Horizon/mask interval axes inconsistent'
    model=json.loads((root/'explanation_model.json').read_text())
    assert max(model['n_iter'])<5000
    records=json.loads((root/'manifest.json').read_text())['samples']
    split=json.loads((root/'scene_split.json').read_text())
    assert len(split['discovery'])==100 and len(split['holdout'])==50
    assert not set(split['discovery'])&set(split['holdout'])
    assert len({r['token'] for r in records})==512
    cases=json.loads((root/'selected_cases.json').read_text())['cases']
    assert len(cases)==12 and len({c['scene'] for c in cases})==12
    for state in ['both_wrong','only_perception','only_forecast','both_correct']:
        assert sum(c['state']==state for c in cases)==3
    write_json(root/'final_validation.json',dict(status='passed',samples=512,scenes=150,
       checks=['domain conservation','semantic/binary validity count conservation','joint total conservation',
               '512-row F and P exact legacy histogram replay','exact occupied intersection/union replay',
               'covariate state margins','covariate GT margins','error subtype conservation',
               'four-state percentages','advantage identity','horizon/mask metric interval axes','regression convergence',
               'disjoint analytical scenes','unique sample identities','12 cases in distinct scenes'],
       all_valid_voxel_tasks=int(basic[:,:,0,0,0].sum()),
       gt_occupied_voxel_tasks=int(basic[:,:,0,1,0].sum()),
       limitations=['No new scene evaluation','No new training seeds','No input/renderer causal interventions']))
    print('[validation] all cross-artifact checks passed',flush=True)


if __name__=='__main__':
    main()
