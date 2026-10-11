#!/usr/bin/env python3
"""Cross-artifact conservation, replay, training-budget and link checks."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_region_utils import ROOT,sha256,write_json
from tools.sparseworld_region_ablation_common import OUT,REPORT,records,masks,evaluate

def main():
    checks=[];recs=records();s=json.loads((OUT/'summary.json').read_text())
    assert len(recs)==150==len({r['scene'] for r in recs});checks.append('150 deterministic distinct-scene samples')
    total=0;occupied=0
    for r in recs:
        pair=dict(np.load(ROOT/'pairs'/(r['token']+'.npz')));attrs=dict(np.load(ROOT/'attributes'/(r['token']+'.npz')))
        with np.load(OUT/'val'/(r['token']+'.npz')) as d:
            names=d['names'].tolist()
            for a in ('F','P'):
                for j in range(3):
                    h,c=evaluate(pair['GT'][j],pair[a][j],masks(r,j,pair,attrs))
                    assert np.array_equal(h,d['hist'][names.index(a),j]);assert np.array_equal(c,d['counts'][names.index(a),j])
            for k,name in enumerate(names):
                if '_static_' in name:continue
                assert np.array_equal(d['counts'][k,:,:,0],d['counts'][names.index('F'),:,:,0])
                assert np.all(d['hist'][k].sum((1,2))==640000)
            total+=int(d['hist'][names.index('F')].sum());occupied+=int(d['counts'][names.index('F'),:,0,0].sum())
        with np.load(OUT/'budget_followup'/(r['token']+'.npz')) as b:
            assert len(b['names'])==42
            assert np.all(b['hist'].sum((1,2))==640000)
            assert np.all(b['counts'][:,:,1]<=b['counts'][:,:,0])
    checks+=['native F/P histograms and regional counts exactly replay paired caches','all common intervention domains conserved','150 x 42 budget interventions complete']
    original=json.loads((ROOT/'manifest.json').read_text())['samples'];distance_total=0
    for r in original:
        with np.load(OUT/'observation'/(r['token']+'.npz')) as d:
            assert int(d['version'])==2
            assert np.array_equal(d['distance_class'].sum(1),d['distance'][:,1])
            assert np.array_equal(d['frustum'].sum((1,2,3,4)),d['distance'][:,1].sum((1,2,3)))
            if d['quality'].sum():
                expected=d['distance'][:,1].sum((1,2,3))
                assert np.all(d['quality'].sum((3,4,5,6))==expected[:,None,None])
            distance_total+=int(d['distance'][:,0].sum())
    assert distance_total==50954208
    checks+=['512 distance partitions conserve all 50,954,208 occupied tasks','class/frustum/quality partitions conserve static targets']
    train=json.loads((OUT/'train_selection.json').read_text())
    assert len(train)==256==len({r['scene'] for r in train})
    assert not {r['scene'] for r in train}&{r['scene'] for r in recs}
    assert len(s['adaptation'])==9
    for arm in s['adaptation']:
        meta=json.loads((OUT/'adaptation'/(arm+'.json')).read_text());assert meta['updates']==768 and meta['log'][-1]['step']==768
        with np.load(OUT/'adaptation'/(arm+'.npz')) as d:
            assert d['tokens'].tolist()==[r['token'] for r in recs]
            assert d['hist'].shape==(150,3,18,18) and np.all(d['hist'].sum((2,3))==640000)
        assert (OUT/'adaptation'/(arm+'.pth')).exists()
    checks+=['256 training scenes disjoint from validation','9 matched head runs x 768 updates = 6912; checkpoints and complete evaluation retained']
    for p in REPORT.glob('*.md'):
        assert '{{' not in p.read_text(),p
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http:','https:','#')):
                resolved=p.parent/target.split('#')[0]
                if resolved.resolve()!=(OUT/'final_validation.json').resolve():
                    assert resolved.exists(),(p,target)
    checks.append('all local report links resolve')
    checkpoint='out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth'
    assert sha256(checkpoint)=='fb9f955f15362cb23244fd551145e499b6a28b1a46e45849b129d9b60a879828'
    checks.append('original checkpoint hash unchanged')
    files=list(REPORT.glob('*.md'))+list(REPORT.glob('*.csv'))+list(REPORT.glob('figures/*.png'))+list(REPORT.glob('figures/*.svg'))
    code=['tools/sparseworld_region_ablation_common.py','tools/run_sparseworld_region_ablations.py','tools/analyze_sparseworld_observation_migration.py','tools/train_sparseworld_window_heads.py','tools/check_sparseworld_point_budget.py','tools/summarize_sparseworld_region_ablations.py','tools/write_sparseworld_ablation_report.py','tools/visualize_sparseworld_ablation_mask.py',__file__,'tests/test_sparseworld_region_ablations.py']
    code+=['tools/illustrate_sparseworld_ablations.py','tools/polish_sparseworld_ablation_reports.py']
    code += [str(p) for p in Path('tools/report_templates/sparseworld_ablations').glob('*.md')]
    write_json(OUT/'final_validation.json',dict(status='passed',checks=checks,valid_gpu_voxel_tasks=total,occupied_gpu_voxel_tasks=occupied,
            occupied_distance_tasks=distance_total,trained_updates=6912,code_sha256={p:sha256(p) for p in code},report_sha256={str(p):sha256(p) for p in files}))
    print(json.dumps(dict(status='passed',checks=len(checks),valid_voxel_tasks=total,occupied_voxel_tasks=occupied,training_updates=6912),indent=2))
if __name__=='__main__':main()
