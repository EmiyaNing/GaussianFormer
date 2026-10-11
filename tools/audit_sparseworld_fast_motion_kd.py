#!/usr/bin/env python3
"""Read-only diagnosis of existing fast-motion KD artifacts; no optimizer updates."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from loss import OPENOCC_LOSS
from tools.sparseworld_fast_motion_common import OUT,HORIZONS,FORECAST,build,cached_task,occupancy_kd,voxel_scores
from tools.sparseworld_region_utils import write_json,sha256,percentile_interval
from tools.diagnose_teacher_alignment import tree_to
from model.response_distillation_forecast import forecast_with_position_gradient


def training_labels(root,p):
    counts=np.zeros((3,4),np.int64);fast_total=np.zeros(3,np.int64);gt_total=np.zeros(3,np.int64);ratios=[]
    for i,ix in enumerate(p['train']['indices']):
        with np.load(root/'targets/train'/f'{ix:06d}.npz') as d:
            teacher=d['teacher'];ids=[d[f'fast_{j}'] for j in range(3)]
        source=torch.load(Path('out/future_distillation_response_based/2048x512/source_cache/train')/f'{ix:06d}.pth',map_location='cpu')
        dense=source['metas']['future_occ_labels'][0].numpy()
        for j,h in enumerate(HORIZONS):
            gt=dense[h-1].reshape(-1);t=teacher[j].reshape(-1)[ids[j]];g=gt[ids[j]]
            n=len(t);right=int((t==g).sum());free=int((t==17).sum());wrong=n-right-free
            counts[j]+=[n,right,free,wrong];fast_total[j]+=n;gt_total[j]+=(gt<17).sum()
            if n:ratios.append([right/n,free/n,wrong/n])
        if (i+1)%512==0:print(f'[teacher labels] {i+1}/2048',flush=True)
    return dict(horizon_counts=counts,columns=['fast_gt','teacher_correct','teacher_free','teacher_wrong_nonempty'],fast_fraction_of_occupied=fast_total/gt_total,
                voxel_weighted_fractions=counts[:,1:].sum(0)/counts[:,0].sum(),nonempty_horizon_equal_weighted_fractions=np.asarray(ratios).mean(0))


def gradients(root,p,count):
    cfg=Config.fromfile('config/opus/sparseworld_fast_motion_occupancy_kd.py');strict=Config.fromfile(cfg.student_model_config)
    import hashlib
    chosen=sorted(p['train']['indices'],key=lambda ix:hashlib.sha256(f'fast_kd_gradient_audit:{ix}'.encode()).hexdigest())
    eligible=[]
    for ix in chosen:
        with np.load(root/'targets/train'/f'{ix:06d}.npz') as d:
            if sum(len(d[f'fast_{j}']) for j in range(3)):eligible.append(ix)
        if len(eligible)==count:break
    model=build(cfg,strict).eval();objective=OPENOCC_LOSS.build(strict.loss).cuda()
    params=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    rows=[]
    for phase in ['initial','frozen_task_final','frozen_fast_kd_final']:
        if phase!='initial':
            arm='frozen_task' if phase=='frozen_task_final' else 'frozen_fast_kd'
            model.load_state_dict(torch.load(root/arm/'latest.pth',map_location='cpu')['state_dict'],strict=True)
        model.eval()
        for ix in eligible:
            source=torch.load(Path('out/future_distillation_response_based/2048x512/source_cache/train')/f'{ix:06d}.pth',map_location='cpu')
            with np.load(root/'targets/train'/f'{ix:06d}.npz') as d:target=dict(d)
            out=forecast_with_position_gradient(model,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda())
            task,_=cached_task(objective,out,tree_to(source['metas'],'cuda'),target['constant_task'])
            kd,detail=occupancy_kd(model,out,target)
            gs=torch.autograd.grad(task,[v for _,v in params],retain_graph=True,allow_unused=True)
            gk=torch.autograd.grad(kd,[v for _,v in params],retain_graph=True,allow_unused=True)
            module={}
            for prefix in FORECAST:
                pairs=[(a,b,v) for (n,v),a,b in zip(params,gs,gk) if n.split('.')[0]==prefix]
                tn=sum(float(a.detach().square().sum()) for a,_,_ in pairs if a is not None)**.5
                kn=sum(float(b.detach().square().sum()) for _,b,_ in pairs if b is not None)**.5
                dot=sum(float((a.detach()*b.detach()).sum()) for a,b,_ in pairs if a is not None and b is not None)
                module[prefix]=dict(task_norm=tn,kd_norm=kn,weighted_ratio=.1*kn/tn if tn else None,cosine=dot/(tn*kn) if tn*kn else None)
            geom=torch.autograd.grad(kd,[out['points'][h-1] for h in HORIZONS],allow_unused=True)
            support_rows=[]
            with torch.no_grad():
                for j,h in enumerate(HORIZONS):
                    ids=torch.as_tensor(target[f'fast_{j}'],device='cuda',dtype=torch.long)
                    if not len(ids):continue
                    scores=voxel_scores(model,out['points'][h-1],out['logits'][h-1])[ids]
                    labels=torch.as_tensor(target['teacher'][j].reshape(-1),device='cuda',dtype=torch.long)[ids]
                    nonempty=labels<17
                    target_score=scores[torch.arange(len(ids),device='cuda'),labels.clamp(max=16)]
                    losses=torch.nn.functional.binary_cross_entropy(scores.clamp(1e-6,1-1e-6),torch.nn.functional.one_hot(labels,18)[:,:17].float(),reduction='none').sum(-1)
                    unsupported=scores.amax(-1)<=1e-6
                    support_rows.append(dict(horizon=h,voxels=len(ids),teacher_nonempty=int(nonempty.sum()),target_class_supported=int((nonempty&(target_score>1e-6)).sum()),
                                             all_class_unsupported=int(unsupported.sum()),loss_sum=float(losses.sum()),unsupported_loss_sum=float(losses[unsupported].sum())))
            rows.append(dict(phase=phase,index=ix,task=float(task.detach()),kd=float(kd.detach()),modules=module,kd_point_gradients_all_none=all(x is None for x in geom),support=support_rows))
            print(f'[gradient {phase}] sample={ix} geometry_none={rows[-1]["kd_point_gradients_all_none"]}',flush=True)
            del out,task,kd,gs,gk,geom,source
    return dict(samples=eligible,selection='SHA256(index), no prediction-error selection, fast nonempty eligibility',mode='eval, no dropout; snapshot diagnostics, not training-average gradients',rows=rows)


def main():
    args=argparse.ArgumentParser();args.add_argument('--probes',type=int,default=12);a=args.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    decomposition={}
    for name,m in summary['models'].items():
        acc=np.asarray(m['accuracy']['value'])[:,1];rec=np.asarray(m['occupancy_recall']['value'])[:,1]
        decomposition[name]=dict(fast_accuracy=acc,fast_occupancy_recall=rec,semantic_accuracy_given_predicted_occupied=100*acc/rec)
    result=dict(status='complete',decomposition=decomposition,teacher_labels=training_labels(OUT,p),gradients=gradients(OUT,p,a.probes),restoration=restore_diagnosis(OUT))
    write_json(OUT/'code_diagnosis.json',result)
    print('Saved code_diagnosis.json',flush=True)


def restore_diagnosis(root):
    from tools.run_sparseworld_region_ablations import render
    from tools.sparseworld_fast_motion_common import region_counts
    from tools.sparseworld_region_utils import metric_from_hist
    cfg=Config.fromfile('config/opus/sparseworld_fast_motion_occupancy_kd.py');strict=Config.fromfile(cfg.student_model_config)
    p=json.loads((root/'protocol.json').read_text())
    specs=[('frozen_task_restore_geometry','frozen_task',True),('frozen_kd_restore_geometry','frozen_fast_kd',True),
           ('full_task_restore_perception','full_task',False),('full_kd_restore_perception','full_fast_kd',False),
           ('full_task_restore_perception_geometry','full_task',True),('full_kd_restore_perception_geometry','full_fast_kd',True)]
    models={};original=torch.load(cfg.student_checkpoint,map_location='cpu')['state_dict']
    for label,arm,geo in specs:
        m=build(cfg,strict).eval().requires_grad_(False)
        trained=torch.load(root/arm/'latest.pth',map_location='cpu')['state_dict']
        for name in FORECAST:
            state=original if geo and name in ('reg','velocity') else trained
            getattr(m,name).load_state_dict({key[len(name)+1:]:v for key,v in state.items() if key.startswith(name+'.')},strict=True)
        models[label]=m
        del trained
    hist={key:[] for key in models};counts={key:[] for key in models}
    with torch.no_grad():
        for i,ix in enumerate(p['val']['indices']):
            source=torch.load(Path('out/future_distillation_response_based/2048x512/source_cache/val')/f'{ix:06d}.pth',map_location='cpu')
            with np.load(root/'targets/val'/f'{ix:06d}.npz') as d:gt=d['GT'];bins=d['speed']
            encoded=tree_to(source['encoded'],'cuda');stamps=source['stamps'].cuda();ego=source['ego'].cuda()
            for label,m in models.items():
                out=forecast_with_position_gradient(m,encoded,stamps,ego);hs=[];cs=[]
                for j,h in enumerate(HORIZONS):
                    pred=render(m,out['points'][h-1],out['logits'][h-1]);a,b=region_counts(gt[j],pred,bins[j]);hs.append(a);cs.append(b)
                hist[label].append(hs);counts[label].append(cs)
            if (i+1)%128==0:print(f'[restore] {i+1}/512',flush=True)
    result={}
    for label in models:
        hs=np.asarray(hist[label]);cs=np.asarray(counts[label]);c=cs.sum(0);met=metric_from_hist(hs.sum(0))
        result[label]=dict(mIoU=met['mIoU'],fast_accuracy=100*c[:,1,1]/c[:,1,0],fast_occupancy_recall=100*c[:,1,2]/c[:,1,0],semantic_accuracy_given_predicted_occupied=100*c[:,1,1]/c[:,1,2])
        np.savez_compressed(root/(label+'.npz'),hist=hs,counts=cs,indices=p['val']['indices'],scenes=p['val']['scenes'])
    return dict(status='complete',samples=512,operation='Snapshot inference interventions, not retraining; geometry restores reg+velocity; perception restore uses original encoded cache, stamps and ego',variants=result)

if __name__=='__main__':main()
