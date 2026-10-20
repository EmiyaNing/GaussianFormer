"""Paired final-epoch interventions on suppressive semantic-amplitude paths."""
from __future__ import annotations
import argparse
from datetime import datetime
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage2 import read,write_json,file_sha256,ce_parts
from tools.empty_mechanism_stage1 import network_output
from tools.train_empty_mechanism_stage2 import build,training_dataset,load_item
from tools.empty_decision_optimization import seed_all,build_loader,to_cuda

RUN=ROOT/'work_dirs/empty_mechanism_stage3'
CONDITIONS=['B0','S','E','SE']
ORDER_SEED=31928


def initialize():
    RUN.mkdir(exist_ok=True)
    path=RUN/'protocol.json'
    if path.exists():return read(path)
    from mmengine import Config
    from tools.empty_mechanism_stage2 import CONFIG
    cfg=Config.fromfile(CONFIG)
    prior=read(ROOT/'work_dirs/empty_mechanism_stage2/protocol.json')
    p=dict(created_at=datetime.now().astimezone().isoformat(),
           purpose='Contribution to endpoint recoverable empty misses; test conditional dominance in the complete final epoch',
           config=CONFIG,origin_epoch=4,final_epoch=5,original_training_seed=928,
           original_checkpoint=prior['checkpoint']['5'],
           checkpoint=dict(path='work_dirs/center_support_contribution1/train/a0_original_seed928/epoch_4.pth'),
           train_frames=4096,validation_frames=1024,optimizer_steps=1024,accumulation=4,
           order_seed=ORDER_SEED,conditions=CONDITIONS,endpoints=[64,256,1024],
           stage_scope='All six Set classification outputs; all occupancy-supervised stages for CE empty',
           intervention='Remove only positive amplitude derivative of Set-negative and/or CE-empty output gradients. Preserve forward loss values, all geometry/empty gradients, positive-class and other-loss gradients.',
           matched_fpr='Descriptive validation ROC operating-point comparison at B0 native empty FPR; no deployed threshold claim',
           calibration='One deterministic frame per training scene, previous fixed fit/tune scene split; per-class IoU thresholds and common tuning shift; all validation frames',
           primary='Correct semantic rescue fraction of fixed B0-endpoint recoverable-miss cohort, native and matched FPR; rank losses and new harms reported separately',
           dominance_rule='Within this final-epoch intervention: paired-scene lower95 rescue fraction >0.5, rank accuracy lower95 delta >= -0.5pp, net final foreground semantic accuracy delta >=0; matched-FPR criterion required for evidence beyond an operating-point shift',
           result_policy='Failure rejects dominance of the tested direct path within this final epoch only; does not settle whole-history or cross-method causes',
           bootstrap=1000,scene_samples=prior['scene_samples']['val'],
           original_rng_unavailable=True,optimizer_moments_restored=True,
           annotation_sha256={s:file_sha256(cfg[s+'_dataset_config'].imageset) for s in ['train','val']},
           source_sha256=file_sha256(__file__))
    p['checkpoint']['sha256']=file_sha256(ROOT/p['checkpoint']['path'])
    p['preexisting_results_sha256']={str(f.relative_to(ROOT)):file_sha256(f) for f in [
        ROOT/'work_dirs/empty_mechanism_stage2/summary.json',
        ROOT/'work_dirs/empty_mechanism_stage2/completion_audit.json',
        ROOT/'work_dirs/empty_mechanism_metric_effects/summary.json']}
    write_json(path,p)
    return p


def suppressive_part(raw,g):
    """Positive d dot g gives a negative amplitude derivative for gradient descent."""
    d=raw.detach().double().sigmoid();g=g.detach().double()
    dot=(d*g).sum(-1,keepdim=True);den=d.square().sum(-1,keepdim=True)
    if torch.any((den<=1e-30)&(dot>0)):
        raise FloatingPointError('Vanishing amplitude derivative with positive source gradient')
    removed=d*dot.clamp_min(0)/den.clamp_min(1e-30)
    remainder=g-removed
    bound=2e-12*(d*g).abs().sum(-1,keepdim=True).clamp_min(1e-30)
    assert torch.all((d*remainder).sum(-1,keepdim=True)<=bound)
    return removed.to(raw),dict(
        source_norm=float(g.norm()),removed_norm=float(removed.norm()),
        suppressive_derivative=float(dot.clamp_min(0).sum()),
        removed_derivative=float((d*removed).sum()),
        removed_fraction=float(removed.square().sum()/g.square().sum().clamp_min(1e-30)))


@torch.no_grad()
def negative_focal_gradient(raw,labels,weights,module,multiplier):
    q=raw.sigmoid();gamma=module.focal_gamma
    g=(1-module.focal_alpha)*(gamma*q.pow(gamma)*(1-q)*F.softplus(raw)+q.pow(gamma+1))
    target=F.one_hot(labels,raw.shape[-1]).bool()
    return g.masked_fill(target,0)*weights[:,None]*multiplier/max(len(raw),1)


def set_negative_gradients(output,batch,module):
    from loss.opus_set_loss import _nearest_indices
    yy=batch['occ_label'][0].long().flatten();xyz=batch['occ_xyz'][0].reshape(-1,3)
    camera=batch['occ_cam_mask'][0].bool().flatten()
    result=[]
    for stage,(raw,points,valid,stage_weight) in enumerate(zip(output['opus_pred_logits'],
          output['opus_pred_points'],output['opus_point_valid_masks'],module.stage_weights)):
        with torch.no_grad():
            domain=module.final_gt_domain if stage==len(module.stage_weights)-1 else module.gt_domain
            mask=(yy>=0)&(yy<17)
            if domain=='camera':mask&=camera
            pm=valid[0].bool();g=torch.zeros_like(raw)
            if pm.any() and mask.any():
                nearest=_nearest_indices(points[0,pm],xyz[mask],module.chunk_size)
                labels=yy[mask][nearest]
                weights=module.class_weights.to(raw)[labels]*module._distance_weight(xyz[mask][nearest])
                multiplier=stage_weight*module.lambda_cls
                if stage==len(module.stage_weights)-1:multiplier*=module.final_classification_weight
                g[0,pm]=negative_focal_gradient(raw[0,pm],labels,weights,module,multiplier)
        result.append(g)
    return result


def loss_with_intervention(output,batch,set_loss,mods,condition):
    inp=dict(metas=batch)
    for key in ['opus_pred_points','opus_pred_logits','opus_point_valid_masks']:inp[key]=output[key]
    components,_=set_loss.components(inp)
    voxel=dict(pred_occ=output['pred_occ'],sampled_xyz=output['sampled_xyz'],
               sampled_label=output['sampled_label'],occ_mask=output['occ_cam_mask'])
    ce=mods[1](voxel);lov=mods[2](voxel)
    base=sum(components.values())+ce+lov
    loss=base;audits=[]
    if condition in ['S','SE']:
        for stage,g in enumerate(set_negative_gradients(output,batch,set_loss)):
            raw=output['opus_pred_logits'][stage]
            removed,audit=suppressive_part(raw,g)
            correction=-(raw*removed).sum()
            loss=loss+(correction-correction.detach())
            audits.append(dict(source='S',stage=stage,**audit))
    if condition in ['E','SE']:
        y=output['sampled_label'][0].long()
        mask=output['occ_cam_mask'][0].flatten().bool()&(y>=0)&(y<18)
        empty=sum(ce_parts(scores[:,:,mask],y[mask],mods[1].class_weights,
                           mods[1].loss_voxel_ce_weight)[1] for scores in output['pred_occ'])/len(output['pred_occ'])
        semantics=[output['gaussians'][stage].semantics for stage in output['supervised_stage_indices']]
        gs=torch.autograd.grad(empty,semantics,retain_graph=True)
        for stage,gradient in zip(output['supervised_stage_indices'],gs):
            raw=output['opus_pred_logits'][stage]
            gu=gradient.detach()*raw.detach().sigmoid()
            removed,audit=suppressive_part(raw,gu)
            correction=-(raw*removed).sum()
            loss=loss+(correction-correction.detach())
            audits.append(dict(source='E',stage=stage,**audit))
    torch.testing.assert_close(loss,base,atol=0,rtol=0)
    return loss,dict(base=float(base.detach()),audits=audits)


def brief_evaluate(net,selected):
    old=net.training;net.eval();rows=[]
    with torch.no_grad():
        _,loader,_=build_loader('val',[m['index'] for m in selected])
        for meta,batch in zip(selected,loader):
            batch=to_cuda(batch);out=network_output(net,batch)
            scores=out['pred_occ'][-1][0]
            y=out['sampled_label'][0].long();mask=batch['occ_cam_mask'][0].bool().flatten()&(y>=0)&(y<18)
            y=y[mask];z=scores[:,mask];top,fg=z[:17].max(0);pred=z.argmax(0)
            counts=torch.bincount(y*18+pred,minlength=324).reshape(18,18)
            rows.append(dict(frame=meta,counts=counts.cpu().tolist(),
                rank_correct=int(((y<17)&(fg==y)).sum()),
                recoverable=int(((y<17)&(fg==y)&(pred==17)).sum()),
                margin_sum=float((top-z[17])[y<17].double().sum()),
                amplitude=float(F.softplus(out['opus_pred_logits'][-1]).sum(-1).mean()),
                bias=float(net.head.empty_bias),empty_scalar=float(net.head.empty_scalar)))
            del batch,out,scores,z
    net.train(old);return rows


def build_origin():
    # Reuse the validated constructor, then restore the actual epoch-4 origin.
    net,opt,sched,cfg,old,set_loss,mods=build(3,'full')
    del old
    p=initialize();checkpoint=torch.load(ROOT/p['checkpoint']['path'],map_location='cpu',weights_only=False)
    net.load_state_dict(checkpoint['state_dict'],strict=True)
    opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    sched.load_state_dict(copy.deepcopy(checkpoint['scheduler']))
    return net,opt,sched,cfg,checkpoint,set_loss,mods


def train(condition,pilot=False):
    p=initialize();dest=RUN/('pilot' if pilot else 'train')/condition;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    seed_all(ORDER_SEED)
    net,opt,sched,cfg,checkpoint,set_loss,mods=build_origin()
    dataset=training_dataset();assert len(dataset)==4096
    sequence=np.random.default_rng(ORDER_SEED).permutation(len(dataset)).tolist()
    steps=1 if pilot else p['optimizer_steps']
    endpoints=[1] if pilot else p['endpoints']
    selected=p['scene_samples'][:2] if pilot else p['scene_samples']
    trace=[];global_iter=checkpoint['global_iter'];start=time.monotonic();begin_step=1
    resume=dest/'resume.pth'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        net.load_state_dict(state['state_dict']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        global_iter=state['global_iter'];begin_step=state['step']+1;trace=state['trace']
        torch.cuda.set_rng_state_all(state['cuda_rng']);torch.set_rng_state(state['torch_rng'])
    elif not (dest/'baseline.json').exists():
        write_json(dest/'baseline.json',dict(records=brief_evaluate(net,selected)))
    os.environ['amp']='false';net.train()
    for step in range(begin_step,steps+1):
        opt.zero_grad();audits=[]
        for micro in range(4):
            index=(step-1)*4+micro
            batch=load_item(dataset,sequence[index],ORDER_SEED*100000+index)
            out=network_output(net,batch)
            loss,audit=loss_with_intervention(out,batch,set_loss,mods,condition)
            assert torch.isfinite(loss)
            (loss/4).backward()
            if micro<3:sched.step_update(global_iter)
            global_iter+=1;audits.append(audit)
            del out,batch,loss
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),cfg.grad_max_norm,error_if_nonfinite=True)
        opt.step();opt.zero_grad();sched.step_update(global_iter-1)
        sources={}
        for source in ['S','E']:
            a=[v for micro in audits for v in micro['audits'] if v['source']==source]
            if a:sources[source]={k:float(np.mean([v[k] for v in a])) for k in
                       ['source_norm','removed_norm','suppressive_derivative','removed_derivative','removed_fraction']}
        trace.append(dict(step=step,loss=float(np.mean([a['base'] for a in audits])),grad_norm=float(norm),sources=sources))
        if step in endpoints:
            write_json(dest/f'step_{step:04d}.json',dict(records=brief_evaluate(net,selected),step=step,condition=condition))
        if step%64==0 or step==steps:
            state=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                       step=step,global_iter=global_iter,trace=trace,torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all(),epoch_origin=4,condition=condition)
            temp=dest/'resume.tmp.pth';torch.save(state,temp);temp.replace(resume)
            write_json(dest/'progress.json',dict(step=step,total=steps,elapsed_seconds=time.monotonic()-start))
        if step==1 or step%16==0:
            print(f'{condition} step={step}/{steps} elapsed={time.monotonic()-start:.1f}s',flush=True)
    write_json(dest/'complete.json',dict(status='complete',condition=condition,steps=steps,frames=steps*4,
        sequence=sequence[:steps*4],trace=trace,elapsed_seconds=time.monotonic()-start,
        protocol_sha256=file_sha256(RUN/'protocol.json'),source_sha256=file_sha256(__file__)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['initialize','train'])
    parser.add_argument('--condition',choices=CONDITIONS,default='B0')
    parser.add_argument('--pilot',action='store_true')
    a=parser.parse_args()
    if a.stage=='initialize':print(initialize())
    else:train(a.condition,a.pilot)


if __name__=='__main__':main()
