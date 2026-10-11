#!/usr/bin/env python3
"""Read-only fixed-student gradient audit after the repair; no updates."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from loss import OPENOCC_LOSS
from model.response_distillation_forecast import forecast_with_position_gradient
from tools.check_sparseworld_fast_motion_repair import DEFAULT
from tools.sparseworld_fast_motion_common import build,cached_task,occupancy_kd
from tools.sparseworld_region_utils import write_json,sha256
from tools.experiment_future_response_distillation import paths_for
from tools.diagnose_teacher_alignment import tree_to

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile('config/opus/sparseworld_fast_motion_occupancy_kd_repaired.py')
    strict=Config.fromfile(cfg.student_model_config);root=DEFAULT
    manifest=json.loads(Path('out/future2current/feature_identity_2048x512/manifest.json').read_text())
    sources=paths_for(Path(manifest['source_cache']),'train',manifest['train'])
    indices=[r['index'] for r in json.loads((root/'preflight_validation.json').read_text())['records']]
    positions={ix:i for i,ix in enumerate(manifest['train']['indices'])}
    objective=OPENOCC_LOSS.build(strict.loss).cuda();result={};hashes={}
    for name in ('initial','frozen_task','frozen_fast_kd'):
        instance=build(cfg,strict).eval()
        if name!='initial':
            path=root/name/'latest.pth';hashes[name]=sha256(path)
            instance.load_state_dict(torch.load(path,map_location='cpu')['state_dict'],strict=True)
        parameters=[(n,p) for n,p in instance.named_parameters() if p.requires_grad]
        rows=[]
        for ix in indices:
            source=torch.load(sources[positions[ix]],map_location='cpu')
            with np.load(root/'targets'/'train'/f'{ix:06d}.npz') as d:target={k:d[k] for k in ('teacher','fast_0','fast_1','fast_2','constant_task')}
            out=forecast_with_position_gradient(instance,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda(),position_gradient=False)
            task,_=cached_task(objective,out,tree_to(source['metas'],'cuda'),target['constant_task'])
            kd,detail=occupancy_kd(instance,out,target,policy='positive_supported_native')
            gt=torch.autograd.grad(task,[p for _,p in parameters],retain_graph=True,allow_unused=True)
            gk=torch.autograd.grad(kd,[p for _,p in parameters],allow_unused=True)
            row=dict(index=ix,task=float(task.detach()),kd=float(kd.detach()),fast=detail,groups={})
            for group in ('semantic','position','ego_attention','reg','velocity'):
                selected=[i for i,(n,_) in enumerate(parameters) if n.split('.')[0]==group]
                nt=sum(float(gt[i].square().sum()) for i in selected if gt[i] is not None)**.5
                nk=sum(float(gk[i].square().sum()) for i in selected if gk[i] is not None)**.5
                dot=sum(float((gt[i]*gk[i]).sum()) for i in selected if gt[i] is not None and gk[i] is not None)
                row['groups'][group]=dict(task_norm=nt,kd_norm=nk,weighted_ratio=.1*nk/nt if nt else None,cosine=dot/(nt*nk) if nt and nk else None)
            assert row['groups']['reg']['task_norm']==row['groups']['reg']['kd_norm']==0
            assert row['groups']['velocity']['task_norm']==row['groups']['velocity']['kd_norm']==0
            rows.append(row)
        stats={}
        for group in ('semantic','position','ego_attention'):
            gs=[r['groups'][group] for r in rows];cos=[g['cosine'] for g in gs if g['cosine'] is not None];ratios=[g['weighted_ratio'] for g in gs if g['weighted_ratio'] is not None]
            stats[group]=dict(mean_cosine=np.mean(cos),negative_cosines=sum(v<0 for v in cos),probes_with_cosine=len(cos),median_weighted_ratio=np.median(ratios))
        result[name]=dict(rows=rows,statistics=stats);del instance;torch.cuda.empty_cache()
        print(name,stats,flush=True)
    assert all(sha256(root/name/'latest.pth')==value for name,value in hashes.items())
    write_json(root/'repair_gradient_diagnosis.json',dict(status='complete',probe_count=len(indices),mode='FP32 eval; no optimizer updates; fixed-perception snapshots only',snapshots=result,checkpoints_unchanged=hashes))

if __name__=='__main__':main()
