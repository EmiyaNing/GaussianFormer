"""Fit 17 biases with exact CE gradients and controlled metric objectives."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import math
import numpy as np
import torch
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.learnable_class_bias import (RUN,ORIGINAL,CACHE,SHRINK,OUT_COST,ALL_COST,
    initialize,read,write_json,file_sha256,load_samples)
from tools.eval_empty_decision_optimization import metrics,region_metrics


def weighted_confusion(data,a):
    pred=np.where(data['X'][:,0]+np.asarray(a)[data['fg']]>=0,data['fg'],17)
    code=data['region'].astype(np.int64)*324+data['gt'].astype(np.int64)*18+pred
    return data['fixed_confusion']+np.bincount(code,weights=data['weight'],minlength=648).reshape(2,18,18)


def grid_candidates(data):
    cuts=np.arange(-64,65)/32;bins=np.searchsorted(cuts,data['X'][:,0],side='right')
    code=(data['region'].astype(np.int64)*306+data['gt'].astype(np.int64)*17+data['fg'])*130+bins
    h=np.bincount(code,weights=data['weight'],minlength=612*130).reshape(2,18,17,130)
    above=np.flip(np.cumsum(np.flip(h,-1),axis=-1),axis=-1)[...,1:]
    fixed=data['fixed_confusion'];gt=fixed[0].sum(-1)+h[0].sum((-1,-2))
    tp=fixed[0].diagonal()[:17,None]+above[0,np.arange(17),np.arange(17)]
    fp=(fixed[0].sum(0)[:17]-fixed[0].diagonal()[:17])[:,None]+above[0].sum(0)-above[0,np.arange(17),np.arange(17)]
    iou=tp/(gt[:17,None]+fp).clip(1)
    out=fixed[1,17,:17,None]+above[1,17];n_empty=fixed[1,17].sum()+h[1,17].sum()
    proposals=[np.zeros(17)]
    for lam in np.r_[0.,np.logspace(-5,4,181)]:
        index=(iou/17-lam*out/max(n_empty,1)).argmax(-1);proposals.append(cuts[index])
    thresholds=np.unique(np.stack(proposals),axis=0)
    cc=np.stack([weighted_confusion(data,-t) for t in thresholds])
    return thresholds,cc


def fit_reference(data,budget=.005):
    thresholds,cc=grid_candidates(data)
    ref=metrics(weighted_confusion(data,np.zeros(17))[1])['empty_FPR']/100
    risk=metrics(cc[:,1])['empty_FPR']/100;cam=metrics(cc[:,0])['mIoU']
    feasible=np.where(risk<=ref+budget+1e-12)[0]
    i=min(feasible,key=lambda k:(-cam[k],risk[k],k))
    return dict(a=-thresholds[i],counts=cc[i],reference_risk=ref,
                reference_all=metrics(cc[i].sum(0))['mIoU']/100),thresholds


def ce_objective(a,lse,target,empty,weight,shrink=0.):
    per=torch.logaddexp(lse,-a)-target+empty*a
    return (weight*per).sum()+shrink*(a-a.mean()).square().mean()


def ce_fit(data,ce,shrink,initial=None):
    fg=torch.as_tensor(data['fg'].astype(np.int64),device='cuda')
    lse=torch.as_tensor(ce['lse'],device='cuda');target=torch.as_tensor(ce['target'],device='cuda')
    empty=torch.as_tensor(data['gt']==17,device='cuda',dtype=torch.float64)
    weight=torch.as_tensor(data['weight']/data['weight'].sum(),device='cuda')
    a=torch.nn.Parameter(torch.zeros(17,device='cuda',dtype=torch.float64))
    def objective(value):
        with torch.no_grad():a.copy_(torch.as_tensor(value,device='cuda'))
        a.grad=None
        loss=(weight*(torch.logaddexp(lse,-a[fg])-target+empty*a[fg])).sum()
        loss=loss+shrink*(a-a.mean()).square().mean()
        loss.backward();return float(loss.detach()),a.grad.detach().cpu().numpy().copy()
    result=minimize(objective,np.zeros(17) if initial is None else initial,method='L-BFGS-B',jac=True,
                    bounds=[(-2.,2.)]*17,options=dict(maxiter=300,ftol=1e-12,gtol=1e-8,maxls=40))
    loss,grad=objective(result.x)
    projected=grad.copy();projected[(result.x<=-2+1e-8)&(grad>0)]=0;projected[(result.x>=2-1e-8)&(grad<0)]=0
    return dict(a=result.x.tolist(),shrink=shrink,loss=loss,success=bool(result.success),
                message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),
                projected_gradient_max=float(np.max(np.abs(projected))))


def ce_value(data,ce,a):
    v=np.asarray(a)[data['fg']]
    return float(np.average(np.logaddexp(ce['lse'],-v)-ce['target']+(data['gt']==17)*v,weights=data['weight']))


def smooth_context(data):
    bins=1024;index=np.minimum(((data['X'][:,0]+2)*bins/4).astype(np.int64),bins-1).clip(0)
    outcome=np.where(data['gt']==data['fg'],0,np.where(data['gt']==17,1,2))
    code=((data['region'].astype(np.int64)*3+outcome)*17+data['fg'])*bins+index
    h=np.bincount(code,weights=data['weight'],minlength=2*3*17*bins).reshape(2,3,17,bins)
    fixed=data['fixed_confusion'].astype(np.float64)
    gt=fixed.sum(-1)+np.bincount(data['region'].astype(np.int64)*18+data['gt'],weights=data['weight'],minlength=36).reshape(2,18)
    return dict(h=torch.as_tensor(h,device='cuda'),
        centers=torch.as_tensor(-2+(np.arange(bins)+.5)*4/bins,device='cuda'),
        tp=torch.as_tensor(np.diagonal(fixed,axis1=1,axis2=2)[:,:17].copy(),device='cuda'),
        fp=torch.as_tensor(fixed[:,:,:17].sum(1)-np.diagonal(fixed,axis1=1,axis2=2)[:,:17],device='cuda'),
        gt=torch.as_tensor(gt[:,:17],device='cuda'),
        out_fixed=float(fixed[1,17,:17].sum()),out_count=float(gt[1,17]))


def smooth_metrics(a,context,temperature):
    a=a.reshape(-1,17);gate=torch.sigmoid((context['centers'][None,None,:]+a[:,:,None])/temperature)
    selected=(context['h'][None]*gate[:,None,None,:,:]).sum(-1)
    tp=context['tp'][None]+selected[:,:,0];fp=context['fp'][None]+selected[:,:,1:].sum(2)
    camera=(tp[:,0]/(context['gt'][0]+fp[:,0]).clamp_min(1)).mean(-1)
    overall=(tp.sum(1)/(context['gt'].sum(0)+fp.sum(1)).clamp_min(1)).mean(-1)
    risk=(context['out_fixed']+selected[:,1,1].sum(-1))/max(context['out_count'],1)
    return camera,overall,risk


def metric_fit(data,warm,shrink,configs,save_prefix=''):
    protocol=initialize();context=smooth_context(data);reference,_=fit_reference(data)
    ref_out=reference['reference_risk']+.005;ref_all=reference['reference_all']-.0005
    a0=np.stack([warm if c['start']=='CE_shrunk' else np.zeros(17) for c in configs])
    a=torch.nn.Parameter(torch.as_tensor(a0,device='cuda',dtype=torch.float64))
    opt=torch.optim.Adam([a],lr=.03)
    rout=torch.as_tensor([c['rho_out'] for c in configs],device='cuda')
    rall=torch.as_tensor([c['rho_all'] for c in configs],device='cuda')
    def objective(temp):
        cam,overall,risk=smooth_metrics(a,context,temp)
        reg=(a-a.mean(-1,keepdim=True)).square().mean(-1)
        loss=1-cam+shrink*reg+rout*(risk-ref_out).clamp_min(0).square()+rall*(ref_all-overall).clamp_min(0).square()
        return loss,cam,overall,risk
    best=np.full(len(configs),np.inf);best_a=a0.copy();best_step=np.zeros(len(configs),int);history=[];start=time.monotonic()
    for step in range(1601):
        if step:
            frac=(step-1)/1599;lr=.003+.027*.5*(1+math.cos(math.pi*frac));temp=.4*(.05/.4)**frac
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True);loss,*_=objective(temp);loss.sum().backward();opt.step()
            with torch.no_grad():a.clamp_(-2.,2.)
        if step%40==0:
            with torch.no_grad():loss,cam,overall,risk=objective(.05)
            values=loss.cpu().numpy();better=values<best-1e-12
            best[better]=values[better];best_a[better]=a.detach().cpu().numpy()[better];best_step[better]=step
            history.append(dict(step=step,loss=values.tolist(),camera=cam.cpu().tolist(),all_miou=overall.cpu().tolist(),risk=risk.cpu().tolist()))
        if step and step%400==0:print('metric fit',save_prefix,step,'/1600','models',len(configs),'seconds',round(time.monotonic()-start,1),flush=True)
    records=[]
    for i,c in enumerate(configs):
        record=dict(**c,a=best_a[i].tolist(),shrink=shrink,selected_step=int(best_step[i]),
            fitting_loss=float(best[i]),fitting_reference_all=reference['reference_all'],
            fitting_reference_risk=reference['reference_risk'],
            history=[{k:(v[i] if isinstance(v,list) else v) for k,v in h.items()} for h in history],
            elapsed_seconds=time.monotonic()-start,source_sha256=file_sha256(__file__))
        records.append(record)
    return records


def fit_all():
    p=initialize();dest=RUN/'models';dest.mkdir(exist_ok=True)
    if (RUN/'fit_summary.json').exists():return read(RUN/'fit_summary.json')
    fit=load_samples('fit');tune=load_samples('tune')
    with np.load(RUN/'ce_fit.npz') as z:cefit={k:z[k] for k in z.files}
    with np.load(RUN/'ce_tune.npz') as z:cetune={k:z[k] for k in z.files}
    reference,proposals=fit_reference(fit)
    np.savez_compressed(RUN/'fitting_B2_proposals.npz',thresholds=proposals)
    ce_records=[]
    for lam in [0.]+SHRINK:
        name='C2' if lam==0 else f'C3_s{lam:g}'
        path=dest/(name+'.json')
        if path.exists():record=read(path)
        else:
            record=ce_fit(fit,cefit,lam);record.update(id=name,family='C2' if lam==0 else 'C3',
                tune_candidate_ce=ce_value(tune,cetune,record['a']),source_sha256=file_sha256(__file__))
            write_json(path,record)
        ce_records.append(record);print('CE',name,record['tune_candidate_ce'],'KKT',record['projected_gradient_max'],flush=True)
    chosen=min(ce_records[1:],key=lambda c:(c['tune_candidate_ce'],c['shrink']))
    write_json(RUN/'ce_shrink_selection.json',dict(selected=chosen['id'],shrink=chosen['shrink'],
        criterion='minimum weighted 200k tuning candidate full18 CE',candidates=ce_records))
    configs=[]
    for family,alls in [('C4',[0.]),('C5',ALL_COST)]:
        for rout in OUT_COST:
            for rall in alls:
                for start in ['CE_shrunk','zero']:
                    configs.append(dict(id=f'{family}_out{rout:g}_all{rall:g}_{start}',family=family,rho_out=rout,rho_all=rall,start=start))
    records=metric_fit(fit,np.asarray(chosen['a']),chosen['shrink'],configs)
    for r in records:write_json(dest/(r['id']+'.json'),r)
    result=dict(status='complete',models={r['id']:dict(path=str((dest/(r['id']+'.json')).relative_to(ROOT)),sha256=file_sha256(dest/(r['id']+'.json'))) for r in ce_records+records},
        selected_C3=chosen['id'],shrink=chosen['shrink'],same_data_B2_proposals=len(proposals),
        fitting_reference=dict(all_miou=reference['reference_all'],risk=reference['reference_risk']),
        source_sha256=file_sha256(__file__))
    write_json(RUN/'fit_summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit']);args=parser.parse_args();fit_all()
