#!/usr/bin/env python3
"""Verify repaired renderer/rollout parity and completed-run invariants."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from tools.diagnose_teacher_alignment import tree_to
from tools.experiment_future_response_distillation import paths_for
from tools.sparseworld_region_utils import write_json,sha256
from tools.sparseworld_fast_motion_common import build,native_voxel_logits,FORECAST,HORIZONS,occupancy_kd
from model.response_distillation_forecast import forecast_with_position_gradient

DEFAULT=Path('out/sparseworld_fast_motion_distillation/20261009_repaired_2048x512')

def preflight(cfg,root):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads(Path('out/future2current/feature_identity_2048x512/manifest.json').read_text())
    instance=build(cfg,Config.fromfile(cfg.student_model_config)).eval()
    sources=paths_for(Path(manifest['source_cache']),'train',manifest['train'])
    # Fixed identity-based probes, independent of model scores.
    candidates=[]
    for i,ix in enumerate(manifest['train']['indices']):
        with np.load(root/'targets'/'train'/f'{ix:06d}.npz') as data:
            if sum(len(data[f'fast_{j}']) for j in range(3)):
                import hashlib
                candidates.append((hashlib.sha256(str(data['token']).encode()).hexdigest(),i))
    selected=[i for _,i in sorted(candidates)[:12]];records=[]
    for i in selected:
        source=torch.load(sources[i],map_location='cpu');encoded=tree_to(source['encoded'],'cuda')
        ego=source['ego'].cuda();stamps=source['stamps'].cuda();current=stamps==0
        with torch.no_grad():
            out=forecast_with_position_gradient(instance,encoded,stamps,ego,position_gradient=False)
            positions=encoded['query_points'][:,current].detach()
            native=instance._forecast_rollout(dict(features=encoded['query_features'][:,current],positions=positions,
                timestamp=positions.new_zeros(*positions.shape[:-1],1),ego=ego,all_features=encoded['query_features'],
                all_points=encoded['query_points'],stamps=stamps,slot_ids=torch.nonzero(current).flatten(),completed_steps=0),6)
            for p,a,l,b in zip(out['points'],native['points'],out['logits'],native['logits']):
                torch.testing.assert_close(p,a,atol=0,rtol=0);torch.testing.assert_close(l,b,atol=0,rtol=0)
            disagreements=[]
            for h in HORIZONS:
                z=native_voxel_logits(instance,out['points'][h-1],out['logits'][h-1]);prob=z.sigmoid()
                occupied=(prob>instance.head.rasterizer.threshold.to(prob)).any(-1)
                labels=torch.full((len(prob),),17,device='cuda',dtype=torch.long);labels[occupied]=prob[occupied].argmax(-1)
                truth=instance.head.rasterizer(out['points'][h-1],out['logits'][h-1])[0]
                mismatch=int((labels!=truth).sum());assert mismatch==0
                disagreements.append(mismatch)
        # Validate actual KD backward, not only label agreement.
        with np.load(root/'targets'/'train'/f'{source["index"]:06d}.npz') as d:targets={k:d[k] for k in ('teacher','fast_0','fast_1','fast_2')}
        instance.zero_grad(set_to_none=True)
        out=forecast_with_position_gradient(instance,encoded,stamps,ego,position_gradient=False)
        kd,detail=occupancy_kd(instance,out,targets,policy='positive_supported_native');assert torch.isfinite(kd)
        kd.backward()
        assert all(p.grad is None for name in ('reg','velocity') for p in getattr(instance,name).parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in instance.parameters())
        records.append(dict(index=source['index'],renderer_disagreements=disagreements,kd=float(kd.detach()),fast=detail))
        print(f'[repair preflight] {len(records)}/12',flush=True)
    write_json(root/'preflight_validation.json',dict(status='passed',checks=['12 native rollout exact-forward comparisons','36 native raster occupancy comparisons','12 finite real-data KD backwards; no reg/velocity KD gradients'],records=records,checkpoint_sha256=sha256(cfg.student_checkpoint)))

def final(cfg,root):
    protocol=json.loads((root/'protocol.json').read_text());assert protocol['position_gradient'] is False
    original=torch.load(cfg.student_checkpoint,map_location='cpu')['state_dict'];checks=[]
    assert not set(protocol['train']['scenes'])&set(protocol['val']['scenes']);checks.append('train/validation scenes disjoint')
    assert sha256(cfg.student_checkpoint)==protocol['checkpoint_sha256'];checks.append('initial checkpoint identity')
    for arm in protocol['arms']:
        directory=root/arm;completion=json.loads((directory/'completion.json').read_text())
        assert completion['status']=='complete' and completion['updates']==2048 and completion['val_samples']==512
        rows={r['step']:r for r in map(json.loads,(directory/'train.jsonl').read_text().splitlines())}
        assert sorted(rows)==list(range(1,2049))
        assert sorted(r['index'] for r in rows.values())==sorted(protocol['train']['indices'])
        assert all(np.isfinite([r['task'],r['kd'],r['total'],r['grad_norm']]).all() for r in rows.values())
        saved=torch.load(directory/'latest.pth',map_location='cpu');assert saved['protocol']==protocol
        assert sha256(directory/'latest.pth')==completion['checkpoint_sha256']
        weights=saved['state_dict']
        assert all(torch.equal(v,weights[n]) for n,v in original.items() if n.split('.')[0] in ('reg','velocity'))
        assert all(torch.equal(v,weights[n]) for n,v in original.items() if 'running_mean' in n or 'running_var' in n or n.endswith('num_batches_tracked'))
        if arm.startswith('frozen'):
            assert all(torch.equal(v,weights[n]) for n,v in original.items() if n.split('.')[0] not in FORECAST)
        else:
            parity=json.loads((directory/'initial_parity.json').read_text())
            assert parity['status']=='passed' and parity['history_images']==30
        audit=json.loads((directory/'gradient_audit.json').read_text())
        expected=sorted(n for n in weights if n.split('.')[0] in ('reg','velocity'))
        assert sorted(audit['no_gradient_names'])==expected
        with np.load(directory/'epoch_1.npz') as data:
            assert data['indices'].tolist()==protocol['val']['indices'] and data['scenes'].tolist()==protocol['val']['scenes']
            assert data['hist'].shape==(512,3,18,18) and np.all(data['hist'].sum((2,3))==640000)
        if arm.endswith('kd'):
            assert all(d['retained']+d['positive_unsupported']==d['voxels']-d['teacher_free'] for r in rows.values() for d in r['fast'])
            assert all(0<=d['retained']<=d['score_supported']<=d['voxels'] for r in rows.values() for d in r['fast'])
        legacy=Path(cfg.fast_target_cache)/arm
        first_old=json.loads((legacy/'train.jsonl').read_text().splitlines()[0])
        assert abs(first_old['task']-rows[1]['task'])<1e-5,'Initial task values changed'
        assert sha256(legacy/'latest.pth')==json.loads((legacy/'completion.json').read_text())['checkpoint_sha256']
        checks.append(arm+': updates/identities/finite losses/weights/BN/gradients/validation/legacy checkpoint passed')
        del saved,weights
    pre=json.loads((root/'preflight_validation.json').read_text());assert pre['status']=='passed';checks.extend(pre['checks'])
    audit=json.loads((root/'repair_gradient_diagnosis.json').read_text());assert audit['status']=='complete'
    assert all(sha256(root/name/'latest.pth')==value for name,value in audit['checkpoints_unchanged'].items())
    checks.append('36 read-only repaired gradient snapshots; audited checkpoint hashes unchanged')
    s=json.loads((root/'summary.json').read_text());assert s['status']=='complete'
    legacy_summary=json.loads((Path(cfg.fast_target_cache)/'summary.json').read_text())
    for prefix in ('frozen','full'):
        for key in ('mIoU','OccIoU','accuracy','occupancy_recall'):
            a=np.asarray(s['models'][prefix+'_fast_kd'][key]['value'])
            for pair,baseline in [('kd_minus_task',prefix+'_task'),('kd_minus_native','native')]:
                np.testing.assert_allclose(a-np.asarray(s['models'][baseline][key]['value']),s[pair][prefix][key]['value'])
        arm=prefix+'_fast_kd'
        assert s['repair_signal'][arm]['voxels']==legacy_summary['training'][arm]['fast_voxel_tasks']
    checks.append('paired summary deltas and original fast-mask denominators consistent')
    report=Path('analysis/hypotheses/prediction_perception_regions/fast_motion_distillation/repair')
    documents=list(report.glob('*.md'));assert len(documents)==5
    for doc in documents:
        contents=doc.read_text();assert contents.count('```')%2==0
        for target in re.findall(r'\]\(([^)]+)\)',contents):
            if not target.startswith(('http:','https:','#')):
                path=(doc.parent/target.split('#')[0]).resolve()
                assert path.exists() or path==(root/'final_validation.json').resolve(),(doc,target)
    checks.append('five repair documents: valid local links and fenced blocks')
    code=['config/opus/sparseworld_fast_motion_occupancy_kd_repaired.py','model/response_distillation_forecast.py','tools/sparseworld_fast_motion_common.py','tools/run_sparseworld_fast_motion_distillation.py','tools/summarize_sparseworld_fast_motion_distillation.py','tools/summarize_sparseworld_fast_motion_repair.py','tools/check_sparseworld_fast_motion_repair.py','tools/audit_sparseworld_fast_motion_repair_gradients.py','tests/test_sparseworld_fast_motion_repair.py']
    artifacts=documents+list((report/'figures').glob('*'))+[report/'metrics.csv']
    write_json(root/'final_validation.json',dict(status='passed',checks=checks,check_count=len(checks),regression_tests=8,
                code_sha256={str(p):sha256(p) for p in code},report_sha256={str(p):sha256(p) for p in artifacts}))
    print(json.dumps(dict(status='passed',check_count=len(checks),regression_tests=8)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--config',default='config/opus/sparseworld_fast_motion_occupancy_kd_repaired.py');p.add_argument('--preflight',action='store_true');args=p.parse_args()
    cfg=Config.fromfile(args.config)
    if args.preflight:preflight(cfg,args.root)
    else:final(cfg,args.root)
