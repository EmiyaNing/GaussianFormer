#!/usr/bin/env python3
"""Real-data validity audit for literal raw KL calls; no optimizer updates."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from loss import OPENOCC_LOSS
from tools.sparseworld_region_utils import write_json
from tools.sparseworld_fast_motion_common import build,cached_task
from tools.sparseworld_raw_logit_kd import paired_raw_logits,elementwise_loss,raw_occupancy_kd
from tools.run_sparseworld_fast_motion_distillation import attach_raw_target,cache_path
from tools.experiment_future_response_distillation import paths_for
from tools.diagnose_teacher_alignment import tree_to
from model.response_distillation_forecast import forecast_with_position_gradient

ROOT=Path('out/sparseworld_fast_motion_distillation/20261009_raw_logits_kl_2048x512')

def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile('config/opus/sparseworld_fast_motion_raw_logit_kl.py');root=args.root
    manifest=json.loads((root/'raw_cache_protocol.json').read_text())
    source_root=Path(manifest['source_cache']);instance=build(cfg,Config.fromfile(cfg.student_model_config)).eval()
    result={};example=None
    for split in ('train','val'):
        sources=paths_for(source_root,split,manifest[split]);records=[];teacher_mass=[];student_mass=[]
        totals=dict(fast_voxels=0,teacher_supported=0,student_supported=0,retained=0,components=0,default_nonfinite_components=0,default_nonfinite_voxels=0,teacher_negative_components=0)
        for i,path in enumerate(sources):
            source=torch.load(path,map_location='cpu');ix=source['index']
            with np.load(cache_path(root,split,ix)) as d:target={k:d[k] for k in ('fast_0','fast_1','fast_2')}
            attach_raw_target(root,split,ix,source['scene'],target);record=dict(index=ix,scene=source['scene'],horizons=[])
            with torch.no_grad():
                out=forecast_with_position_gradient(instance,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda(),position_gradient=False)
                for j in range(3):
                    s,t,_,detail=paired_raw_logits(instance,out,target,j)
                    for key in ('voxels','teacher_supported','student_supported','retained'):totals['fast_voxels' if key=='voxels' else key]+=detail[key]
                    if len(s):
                        d0=elementwise_loss(s,t,'raw_api_default');d1=elementwise_loss(s,t,'raw_api_log_target');g=elementwise_loss(s,t,'gaussian_raw')
                        assert torch.isfinite(d1).all() and torch.isfinite(g).all() and (g>=0).all()
                        totals['components']+=d0.numel();totals['default_nonfinite_components']+=int((~torch.isfinite(d0)).sum())
                        totals['default_nonfinite_voxels']+=int((~torch.isfinite(d0)).any(-1).sum());totals['teacher_negative_components']+=int((t<0).sum())
                        detail.update(default_finite=bool(torch.isfinite(d0).all()),raw_api_loss=float(d1.sum()/(detail['voxels']*17)),gaussian_loss=float(g.sum()/(detail['voxels']*17)),rmse=float((s-t).square().mean().sqrt()))
                        teacher_mass.append(torch.logsumexp(t,dim=-1).cpu().numpy());student_mass.append(torch.logsumexp(s,dim=-1).cpu().numpy())
                        if example is None:
                            probe=t[:1].detach()
                            # The surrounding no_grad context is intentional for
                            # the full dataset; use a small enabled-grad probe.
                            with torch.enable_grad():
                                matched=probe.clone().requires_grad_();equal=elementwise_loss(matched,probe,'raw_api_log_target').mean()
                                gradient=torch.autograd.grad(equal,matched)[0]
                                same=probe.clone().requires_grad_();zero=elementwise_loss(same,probe,'gaussian_raw').mean();gg=torch.autograd.grad(zero,same)[0]
                            shifts=np.linspace(-5,5,41);example=dict(index=ix,horizon=j+1,teacher_logits=probe.cpu().numpy(),shifts=shifts,
                                raw_api_curve=[float(elementwise_loss(probe+float(c),probe,'raw_api_log_target').mean()) for c in shifts],
                                gaussian_curve=[float(elementwise_loss(probe+float(c),probe,'gaussian_raw').mean()) for c in shifts],
                                raw_equal_loss=float(equal),raw_equal_gradient=gradient.cpu().numpy(),gaussian_equal_loss=float(zero),gaussian_equal_gradient=gg.cpu().numpy())
                    record['horizons'].append(detail)
            records.append(record)
            if (i+1)%128==0:print(f'[raw KL audit {split}] {i+1}/{len(sources)}',flush=True)
        for name,chunks in [('teacher',teacher_mass),('student',student_mass)]:
            flat=np.concatenate(chunks);totals[name+'_log_mass_quantiles']=np.quantile(flat,[0,.01,.5,.99,1])
            totals[name+'_normalized_fraction']=float((np.abs(flat)<1e-5).mean())
        result[split]=dict(totals=totals,records=records)
    # Compare KD/task gradient norms on the fixed identity-based 12 probes used
    # in the preceding repair. These are FP32 eval snapshots, not train averages.
    objective=OPENOCC_LOSS.build(Config.fromfile(cfg.student_model_config).loss).cuda()
    probes=json.loads(Path('out/sparseworld_fast_motion_distillation/20261009_repaired_2048x512/preflight_validation.json').read_text())['records']
    positions={ix:i for i,ix in enumerate(manifest['train']['indices'])};sources=paths_for(source_root,'train',manifest['train']);gradients=[]
    for p in probes:
        if p['index'] not in positions:continue
        source=torch.load(sources[positions[p['index']]],map_location='cpu');ix=source['index']
        with np.load(cache_path(root,'train',ix)) as d:target={k:d[k] for k in ('fast_0','fast_1','fast_2','constant_task')}
        attach_raw_target(root,'train',ix,source['scene'],target)
        out=forecast_with_position_gradient(instance,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda(),position_gradient=False)
        task,_=cached_task(objective,out,tree_to(source['metas'],'cuda'),target['constant_task'])
        params=[(n,p) for n,p in instance.named_parameters() if p.requires_grad]
        gt=torch.autograd.grad(task,[p for _,p in params],retain_graph=True,allow_unused=True)
        row=dict(index=ix,policies={})
        for policy in ('raw_api_log_target','gaussian_raw'):
            kd,detail=raw_occupancy_kd(instance,out,target,policy)
            gk=torch.autograd.grad(kd,[p for _,p in params],retain_graph=True,allow_unused=True);groups={}
            for group in ('semantic','position','ego_attention','reg','velocity'):
                sel=[i for i,(n,_) in enumerate(params) if n.split('.')[0]==group]
                nt=sum(float(gt[i].square().sum()) for i in sel if gt[i] is not None)**.5
                nk=sum(float(gk[i].square().sum()) for i in sel if gk[i] is not None)**.5
                dot=sum(float((gt[i]*gk[i]).sum()) for i in sel if gt[i] is not None and gk[i] is not None)
                groups[group]=dict(task_norm=nt,kd_norm=nk,weighted_ratio=.1*nk/nt if nt else None,cosine=dot/(nt*nk) if nt and nk else None)
            assert groups['reg']['kd_norm']==groups['velocity']['kd_norm']==0
            row['policies'][policy]=dict(loss=float(kd.detach()),retained=sum(d['retained'] for d in detail),groups=groups)
        gradients.append(row)
    write_json(root/'raw_kl_input_audit.json',dict(status='complete',torch_version=torch.__version__,updates=0,results=result,translation_probe=example,gradient_probes=gradients,
        default_training_status='invalid_inputs_no_optimizer_update',literal_log_target_status='finite_but_unnormalized_and_unbounded'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);main(p.parse_args())
