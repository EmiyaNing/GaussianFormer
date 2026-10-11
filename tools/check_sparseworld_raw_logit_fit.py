#!/usr/bin/env python3
"""Read-only matched-voxel fitting check: initial versus final raw logits."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from tools.sparseworld_fast_motion_common import build,pack_output,encode
from tools.sparseworld_raw_logit_kd import paired_raw_logits
from tools.run_sparseworld_fast_motion_distillation import attach_raw_target,cache_path,image_loader,loaders
from tools.train_future_predictable_full import historical_config
from tools.diagnose_teacher_alignment import tree_to
from tools.experiment_future_response_distillation import paths_for
from model.response_distillation_forecast import forecast_with_position_gradient
from tools.sparseworld_region_utils import write_json,sha256
from tools.audit_sparseworld_raw_logit_kl import ROOT

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile('config/opus/sparseworld_fast_motion_raw_logit_kl.py');strict=Config.fromfile(cfg.student_model_config)
    protocol=json.loads((ROOT/'protocol.json').read_text());manifest=json.loads((ROOT/'raw_cache_protocol.json').read_text())
    probes=json.loads(Path('out/sparseworld_fast_motion_distillation/20261009_repaired_2048x512/preflight_validation.json').read_text())['records']
    positions={ix:i for i,ix in enumerate(manifest['train']['indices'])};selected=[positions[p['index']] for p in probes]
    sources=paths_for(Path(manifest['source_cache']),'train',manifest['train']);original=build(cfg,strict).eval();reference={};targets={}
    with torch.no_grad():
        for i in selected:
            source=torch.load(sources[i],map_location='cpu');ix=source['index']
            with np.load(cache_path(ROOT,'train',ix)) as d:target={k:d[k] for k in ('fast_0','fast_1','fast_2')}
            attach_raw_target(ROOT,'train',ix,source['scene'],target);targets[ix]=target
            out=forecast_with_position_gradient(original,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda(),position_gradient=False)
            reference[ix]=[(s.cpu(),t.cpu(),valid.cpu()) for j in range(3) for s,t,valid,_ in [paired_raw_logits(original,out,target,j)]]
    del original;torch.cuda.empty_cache();results={};hashes={}
    for arm in protocol['arms']:
        directory=ROOT/arm;assert json.loads((directory/'completion.json').read_text())['status']=='complete'
        hashes[arm]=sha256(directory/'latest.pth');full=arm.startswith('full')
        instance=build(cfg,strict,full=full).eval().requires_grad_(False)
        instance.load_state_dict(torch.load(directory/'latest.pth',map_location='cpu')['state_dict'],strict=True)
        if full:
            images,selection=loaders(historical_config(cfg),'train');assert selection==manifest['train']
            stream=image_loader(images.dataset,cfg.fast_workers,selected)
        else:stream=range(len(selected))
        totals=dict(voxels=0,fast_voxels=0,initial_pair_voxels=0,final_pair_voxels=0,initial_squared_error=0.,final_squared_error=0.,initial_sum=0.,final_sum=0.,teacher_sum=0.);rows=[]
        with torch.no_grad():
            for i,item in zip(selected,stream):
                source=torch.load(sources[i],map_location='cpu');ix=source['index']
                if full:
                    data=tree_to(item,'cuda');state,ego=encode(instance,data.pop('img'),data)
                    packed=pack_output(instance,state,ego,data,position_gradient=False);out=dict(points=packed['future_pred_points'],logits=packed['future_pred_logits'])
                else:out=forecast_with_position_gradient(instance,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda(),position_gradient=False)
                for j in range(3):
                    current,teacher,valid,_=paired_raw_logits(instance,out,targets[ix],j)
                    initial,initial_teacher,initial_valid=reference[ix][j];valid=valid.cpu();common=valid&initial_valid
                    totals['fast_voxels']+=len(valid);totals['initial_pair_voxels']+=int(initial_valid.sum());totals['final_pair_voxels']+=int(valid.sum())
                    a=initial[common[initial_valid]];b=current.cpu()[common[valid]];t=teacher.cpu()[common[valid]]
                    torch.testing.assert_close(t,initial_teacher[common[initial_valid]],rtol=0,atol=0)
                    if not len(a):continue
                    totals['voxels']+=len(a);totals['initial_squared_error']+=float((a-t).double().square().sum());totals['final_squared_error']+=float((b-t).double().square().sum())
                    for key,x in [('initial_sum',a),('final_sum',b),('teacher_sum',t)]:totals[key]+=float(x.double().sum())
                    rows.append(dict(index=ix,horizon_seconds=j+1,common_voxels=len(a),initial_rmse=float((a-t).square().mean().sqrt()),final_rmse=float((b-t).square().mean().sqrt())))
        count=totals['voxels']*17
        results[arm]=dict(common_voxels=totals['voxels'],fast_voxels=totals['fast_voxels'],initial_pair_voxels=totals['initial_pair_voxels'],final_pair_voxels=totals['final_pair_voxels'],initial_rmse=(totals['initial_squared_error']/count)**.5,final_rmse=(totals['final_squared_error']/count)**.5,
                         initial_mean=totals['initial_sum']/count,final_mean=totals['final_sum']/count,teacher_mean=totals['teacher_sum']/count,rows=rows)
        del instance;torch.cuda.empty_cache()
        print(arm,{k:v for k,v in results[arm].items() if k!='rows'},flush=True)
    assert all(sha256(ROOT/a/'latest.pth')==v for a,v in hashes.items())
    write_json(ROOT/'raw_logit_fit_audit.json',dict(status='complete',probes=12,scope='FP32 eval; same physical voxels supported initially and finally, fixed teacher; no optimizer update',results=results,checkpoint_hashes=hashes))

if __name__=='__main__':main()
