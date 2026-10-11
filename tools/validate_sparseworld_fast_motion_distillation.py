#!/usr/bin/env python3
"""Validate complete 2048/512 paired occupancy distillation artifacts."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from tools.sparseworld_fast_motion_common import OUT,REPORT,FORECAST,build
from tools.sparseworld_region_utils import sha256,write_json


def main():
    p=json.loads((OUT/'protocol.json').read_text());checks=[]
    assert len(p['train']['indices'])==2048 and len(p['val']['indices'])==512
    assert not set(p['train']['scenes'])&set(p['val']['scenes'])
    checks.append('2048/512 identities, 700/150 scene separation')
    assert sha256(p['checkpoint'])==p['checkpoint_sha256']
    checks.append('original checkpoint unchanged')
    for split in ('train','val'):
        assert len(list((OUT/'targets'/split).glob('*.npz')))==len(p[split]['indices'])
        for ix,scene in zip(p[split]['indices'],p[split]['scenes']):
            with np.load(OUT/'targets'/split/f'{ix:06d}.npz') as d:
                assert int(d['index'])==ix and str(d['scene'])==scene
                assert d['teacher'].shape==(3,200,200,16) and d['teacher'].max()<=17
                bins=d['speed']
                for j in range(3):assert np.array_equal(d[f'fast_{j}'],np.flatnonzero(bins[j].reshape(-1)==2))
    checks.append('2560 teacher/motion caches aligned; KD mask exactly speed bin 2')
    reference_rows=np.load(OUT/'native_baseline.npz');counts=reference_rows['counts'][:,:,0,0]
    assert counts.sum()==50954208
    cfg=Config.fromfile('config/opus/sparseworld_fast_motion_occupancy_kd.py');strict=Config.fromfile(cfg.student_model_config)
    reference=build(cfg,strict).eval().requires_grad_(False)
    initial={n:v.cpu().clone() for n,v in reference.state_dict().items()}
    del reference;torch.cuda.empty_cache()
    for arm in p['arms']:
        root=OUT/arm;complete=json.loads((root/'completion.json').read_text())
        expected=2048*p['epochs'];assert complete['updates']==expected and complete['status']=='complete'
        rows={r['step']:r for r in [json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()]}
        assert sorted(rows)==list(range(1,expected+1))
        for ep in range(1,p['epochs']+1):
            indices=[rows[k]['index'] for k in range((ep-1)*2048+1,ep*2048+1)]
            assert sorted(indices)==sorted(p['train']['indices'])
        with np.load(root/f'epoch_{p["epochs"]}.npz') as d:
            assert d['indices'].tolist()==p['val']['indices'] and d['scenes'].tolist()==p['val']['scenes']
            assert d['hist'].shape==(512,3,18,18) and np.all(d['hist'].sum((2,3))==640000)
            assert np.array_equal(d['counts'][:,:,0,0],counts)
        audit=json.loads((root/'gradient_audit.json').read_text());assert not audit['no_gradient_names']
        saved=torch.load(root/'latest.pth',map_location='cpu');assert saved['step']==expected and saved['protocol']==p
        if arm.startswith('frozen'):
            assert all(torch.equal(v,saved['state_dict'][n]) for n,v in initial.items() if n.split('.')[0] not in FORECAST)
            assert all(n.split('.')[0] in FORECAST for n in audit['trainable_names'])
        else:
            assert json.loads((root/'initial_parity.json').read_text())['status']=='passed'
            for n,v in initial.items():
                if n.endswith(('running_mean','running_var','num_batches_tracked')):assert torch.equal(v,saved['state_dict'][n])
        del saved
    checks += ['four arms complete full ordered epochs with all 2048 identities','four 512-row evaluations conserve domains',
               'frozen perception tensors identical to initialization','full Student initial parity and fixed BN statistics',
               'all intended trainable parameters have gradients']
    summary=json.loads((OUT/'summary.json').read_text());assert summary['status']=='complete'
    for prefix in ('frozen','full'):
        for key in ('mIoU','OccIoU','accuracy'):
            a=np.asarray(summary['models'][prefix+'_fast_kd'][key]['value']);b=np.asarray(summary['models'][prefix+'_task'][key]['value'])
            np.testing.assert_allclose(a-b,summary['kd_minus_task'][prefix][key]['value'])
    checks.append('paired deltas consistent with absolute scores')
    for doc in REPORT.glob('*.md'):
        text=doc.read_text();assert text.count('```')%2==0 and '{{' not in text
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if not target.startswith(('http:','https:','#')):assert (doc.parent/target.split('#')[0]).exists(),(doc,target)
    checks.append('all local report links and fenced blocks valid')
    code=['tools/sparseworld_fast_motion_common.py','tools/run_sparseworld_fast_motion_distillation.py','tools/summarize_sparseworld_fast_motion_distillation.py',__file__,'config/opus/sparseworld_fast_motion_occupancy_kd.py','tests/test_sparseworld_fast_motion_distillation.py','tools/illustrate_sparseworld_fast_motion_distillation.py']
    docs=list(REPORT.glob('*.md'))+list((REPORT/'figures').glob('*'))
    result=dict(status='passed',checks=checks,trained_updates=8192*p['epochs'],validation_voxel_tasks_per_model=512*3*640000,occupied_tasks_per_model=50954208,
                code_sha256={str(x):sha256(x) for x in code},report_sha256={str(x):sha256(x) for x in docs})
    write_json(OUT/'final_validation.json',result);print(json.dumps(dict(status='passed',checks=len(checks),updates=result['trained_updates']),indent=2))
if __name__=='__main__':main()
