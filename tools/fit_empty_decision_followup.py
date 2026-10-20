"""Train controlled probability/decision heads; selection uses training scenes only."""
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from tools.empty_decision_followup import RUN,OLD,LAMBDAS,SEEDS,initialize,samples,read,write_json,file_sha256,seed_all
from tools.fit_empty_decision_optimization import Dense,loss_context,decision_statistics,load_artifact


def scale_fit(data,cols):
    x=data['X'][:,cols].astype(np.float64);w=data['weight']
    mean=np.average(x,axis=0,weights=w)
    std=np.sqrt(np.average((x-mean)**2,axis=0,weights=w)).clip(.05)
    return dict(cols=list(cols),mean=mean.astype(np.float32),std=std.astype(np.float32))


def encode(x,fg,scale,region=None):
    if not torch.is_tensor(x):x=torch.as_tensor(x,device='cuda')
    if not torch.is_tensor(fg):fg=torch.as_tensor(fg.astype(np.int64),device=x.device)
    values=(x[:,scale['cols']]-torch.as_tensor(scale['mean'],device=x.device))/torch.as_tensor(scale['std'],device=x.device)
    result=torch.cat([values,F.one_hot(fg.long(),17).to(values)],-1)
    if region is not None:
        result=torch.cat([result,torch.as_tensor(region,device=x.device,dtype=x.dtype).reshape(-1,1)],-1)
    return result


def targets(data):return np.where(data['gt']==17,1,np.where(data['gt']==data['fg'],0,2)).astype(np.int64)


class Probability:
    def __init__(self,path,device='cuda'):
        self.path=Path(path);self.record=torch.load(path,map_location='cpu',weights_only=False)
        r=self.record
        self.mode=r.get('flavour','mix_800');self.scale=r['scale']
        self.net=Dense(len(self.scale['cols'])+17+(self.mode=='mask_oracle'),r['hidden'],r['outputs']).to(device).eval()
        self.net.load_state_dict(r['state_dict']);self.device=device
    def __call__(self,x,fg,region=None):
        if self.mode=='mask_oracle':assert region is not None,'GT region is oracle-only'
        a=encode(x,fg,self.scale,region if self.mode=='mask_oracle' else None)
        return (self.net(a)/self.record.get('temperature',1.)).softmax(-1)


def probability_fit(mode,seed=928):
    initialize();dest=RUN/'models';dest.mkdir(exist_ok=True)
    path=dest/f'Q_{mode}_seed{seed}.pt'
    if path.exists():return path
    fit=samples('fit',mode=='mix_expanded');tune=samples('tune')
    seed_all(seed);scale=scale_fit(fit,list(range(8)))
    a=encode(fit['X'],fit['fg'],scale,fit['region'] if mode=='mask_oracle' else None)
    b=encode(tune['X'],tune['fg'],scale,tune['region'] if mode=='mask_oracle' else None)
    y=torch.as_tensor(targets(fit)+(fit['region'].astype(np.int64)*3 if mode=='joint6' else 0),device='cuda')
    yt=torch.as_tensor(targets(tune)+(tune['region'].astype(np.int64)*3 if mode=='joint6' else 0),device='cuda')
    w=fit['weight'].copy();wt=tune['weight'].copy()
    if mode=='balanced':
        for region in [0,1]:
            w[fit['region']==region]/=w[fit['region']==region].sum()*2
            wt[tune['region']==region]/=wt[tune['region']==region].sum()*2
    w=torch.as_tensor(w/w.sum(),device='cuda',dtype=torch.float32)
    wt=torch.as_tensor(wt/wt.sum(),device='cuda',dtype=torch.float32)
    hidden=96 if mode=='mix_wide' else 32;outputs=6 if mode=='joint6' else 3
    net=Dense(a.shape[1],hidden,outputs).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
    best=None;history=[];torch.cuda.synchronize();start=time.monotonic()
    for step in range(1,2401):
        lr=.003 if step<=800 else .0003+.0027*.5*(1+math.cos(math.pi*(step-800)/1600))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);loss=(F.cross_entropy(net(a),y,reduction='none')*w).sum()
        loss.backward();opt.step()
        if step%100==0:
            with torch.no_grad():value=(F.cross_entropy(net(b),yt,reduction='none')*wt).sum().item()
            history.append(dict(step=step,fit_nll=loss.item(),tune_nll=value,lr=lr))
            if best is None or value<best[0]:best=(value,step,copy.deepcopy(net.state_dict()))
    torch.cuda.synchronize()
    record=dict(kind='probability',flavour=mode,scale=scale,hidden=hidden,outputs=outputs,temperature=1.,seed=seed,
                state_dict={k:v.cpu() for k,v in best[2].items()})
    torch.save(record,path)
    write_json(path.with_suffix('.json'),dict(mode=mode,seed=seed,selected_step=best[1],tune_nll=best[0],
        parameters=sum(p.numel() for p in net.parameters()),history=history,fit_rows=len(fit['gt']),
        elapsed_seconds=time.monotonic()-start,artifact_sha256=file_sha256(path),
        oracle_only=mode=='mask_oracle',source_sha256=file_sha256(__file__)))
    print('probability',mode,seed,'NLL',round(best[0],6),'step',best[1],'seconds',round(time.monotonic()-start,2),flush=True)
    return path


def probability_metrics(prob,data):
    with torch.no_grad():
        p=prob(torch.as_tensor(data['X'],device='cuda'),torch.as_tensor(data['fg'],device='cuda'),data['region']).cpu().numpy()
    outcomes=targets(data);w=data['weight'];result={}
    variants={'marginal':p.reshape(-1,2,3).sum(1)} if p.shape[1]==6 else {'direct':p}
    if p.shape[1]==6:
        pp=p.reshape(-1,2,3);conditional=pp[np.arange(len(pp)),data['region']]
        variants['conditional_diagnostic']=conditional/conditional.sum(-1,keepdims=True).clip(1e-20)
        mass=pp[:,0].sum(-1);truth=data['region']==0
        result['region_prediction']=dict(AP=float(average_precision_score(truth,mass,sample_weight=w)),
            AUC=float(roc_auc_score(truth,mass,sample_weight=w)),prevalence=float(np.average(truth,weights=w)))
    for label,values in variants.items():
        rows={}
        for r,name in [(0,'camera'),(1,'complement'),(None,'all')]:
            mask=np.ones(len(w),bool) if r is None else data['region']==r
            v,ww,y=values[mask],w[mask],outcomes[mask]
            row=dict(NLL=float(np.average(-np.log(v[np.arange(len(y)),y].clip(1e-20)),weights=ww)))
            for event,key in [(0,'q'),(1,'e')]:
                score=v[:,event];truth=y==event;bins=np.minimum((score*15).astype(int),14)
                row[key]=dict(mean=float(np.average(score,weights=ww)),prevalence=float(np.average(truth,weights=ww)),
                    ECE=float(sum(abs((ww[bins==i]*(score[bins==i]-truth[bins==i])).sum()) for i in range(15))/ww.sum()),
                    AP=float(average_precision_score(truth,score,sample_weight=ww)))
            rows[name]=row
        result[label]=rows
    return result


def probability_path(family,seed):
    if family=='Mix':return OLD/'models'/f'prob_no_support_seed{seed}.pt'
    mode={'Long':'mix_long','Balanced':'balanced','Joint':'joint6','Conditional':'joint6'}.get(family)
    return probability_fit(mode,seed) if mode else None


def probability_features(values,family):
    if family=='Conditional':
        joint=values.reshape(-1,2,3)
        return (joint/joint.sum(-1,keepdim=True).clamp_min(1e-20)).flatten(1)
    return values


class Policy:
    def __init__(self,path,device='cuda'):
        self.path=Path(path);self.record=torch.load(path,map_location='cpu',weights_only=False);r=self.record
        self.prob=Probability(ROOT/r['probability_path'],device=device) if r['probability_path'] else None
        self.net=Dense(len(r['scale']['cols'])+17+(self.prob.record['outputs'] if self.prob else 0),r['hidden'],1).to(device).eval()
        self.net.load_state_dict(r['state_dict'])
        self.anchor=torch.as_tensor(r['anchor_thresholds'],device=device,dtype=torch.float32)
    def __call__(self,x,fg,cache=None):
        cache={} if cache is None else cache
        a=encode(x,fg,self.record['scale'])
        if self.prob:
            key=str(self.prob.path)
            if key not in cache:cache[key]=self.prob(x,fg)
            a=torch.cat([a,probability_features(cache[key],self.record['family'])],-1)
        return x[:,0]-self.anchor[fg.long()]+self.net(a)[:,0].tanh()


def fit_policy(family,seed,lam):
    protocol=initialize();tag=str(lam).replace('.','p')
    dest=RUN/('width_control/models' if family=='H16' else 'models')
    path=dest/f'D_{family}_seed{seed}_lam{tag}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return path
    fit=samples('fit',family=='Expanded');tune=samples('tune')
    cols=list(range(12 if family in ['Support','Safe'] else 9 if family=='S' else 8))
    assert fit['X'].shape[1]>=len(cols)
    seed_all(seed);scale=scale_fit(fit,cols);a=encode(fit['X'],fit['fg'],scale);b=encode(tune['X'],tune['fg'],scale)
    probability=probability_path(family,seed)
    prob_parameters=0
    if probability:
        q=Probability(probability)
        with torch.no_grad():
            qa=q(torch.as_tensor(fit['X'],device='cuda'),torch.as_tensor(fit['fg'],device='cuda'))
            qb=q(torch.as_tensor(tune['X'],device='cuda'),torch.as_tensor(tune['fg'],device='cuda'))
            a=torch.cat([a,probability_features(qa,family)],-1)
            b=torch.cat([b,probability_features(qb,family)],-1)
        prob_parameters=sum(p.numel() for p in q.net.parameters())
    ca,cb=loss_context(fit),loss_context(tune)
    anchor=torch.as_tensor(protocol['anchor_thresholds'],device='cuda')
    ma=torch.as_tensor(fit['X'][:,0],device='cuda')-anchor[ca['fg']]
    mb=torch.as_tensor(tune['X'][:,0],device='cuda')-anchor[cb['fg']]
    hidden=16 if probability or family=='H16' else 40
    net=Dense(a.shape[1],hidden,1,zero_output=True).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
    wa=ca['w']/ca['w'].sum();wb=cb['w']/cb['w'].sum()
    safe=family=='Safe';safe_a=(~ca['camera'])&ca['correct'];safe_b=(~cb['camera'])&cb['correct']
    initial_a=(ma>=0).float();initial_b=(mb>=0).float()
    def objective(d,context,reg,w,initial,correct_out):
        iou,risk=decision_statistics(d,context)
        removal=(context['w']*correct_out*(initial-d).clamp_min(0)).sum()/(context['w']*correct_out).sum().clamp_min(1)
        return iou-lam*risk-.001*(w*reg.square()).sum()-(.05*removal if safe else 0),iou,risk,removal
    best=None;history=[];torch.cuda.synchronize();start=time.monotonic()
    for step in range(1201):
        if step:
            progress=step/1200;temp=.15+.25*(1-progress)
            for group in optimizer.param_groups:group['lr']=.0003+.0027*.5*(1+math.cos(math.pi*progress))
            optimizer.zero_grad(set_to_none=True);r=net(a)[:,0].tanh();d=torch.sigmoid((ma+r)/temp)
            utility,*_=objective(d,ca,r,wa,initial_a,safe_a)
            (-utility).backward();optimizer.step()
        if step%40==0:
            with torch.no_grad():
                r=net(b)[:,0].tanh();value,iou,risk,removal=objective((mb+r>=0).float(),cb,r,wb,initial_b,safe_b)
                value=value.item()
                history.append(dict(step=step,utility=value,camera_mIoU=iou.item()*100,out_FPR=risk.item()*100,
                    residual_rms=float((wb*r.square()).sum().sqrt()),correct_out_removal=removal.item()))
                if best is None or value>best[0]:best=(value,step,copy.deepcopy(net.state_dict()))
    torch.cuda.synchronize()
    record=dict(kind='policy',family=family,seed=seed,lam=lam,scale=scale,hidden=hidden,
        probability_path=str(probability.relative_to(ROOT)) if probability else None,
        anchor_thresholds=protocol['anchor_thresholds'],state_dict={k:v.cpu() for k,v in best[2].items()})
    torch.save(record,path)
    write_json(path.with_suffix('.json'),dict(family=family,seed=seed,lam=lam,selected_step=best[1],history=history,
        fit_rows=len(fit['gt']),parameters=sum(p.numel() for p in net.parameters())+prob_parameters,
        probability_parameters=prob_parameters,elapsed_seconds=time.monotonic()-start,
        artifact_sha256=file_sha256(path),source_sha256=file_sha256(__file__)))
    print('policy',family,seed,lam,'step',best[1],'seconds',round(time.monotonic()-start,2),flush=True)
    return path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['base','probability','support','all','metrics'])
    args=parser.parse_args();protocol=initialize()
    if args.stage in ['probability','all']:
        for mode in ['mix_long','balanced','joint6','mask_oracle','mix_wide','mix_expanded']:
            for seed in (SEEDS if mode=='joint6' else [928]):probability_fit(mode,seed)
    families={'base':['H','Mix'],'probability':['Long','Balanced','Joint','Conditional'],
              'support':['S','Support','Expanded','Safe'],'all':protocol['policy_families']}.get(args.stage,[])
    for family in families:
        for seed in protocol['seeds'][family]:
            for lam in LAMBDAS:fit_policy(family,seed,lam)
    if args.stage in ['metrics','all']:
        result={}
        paths=[OLD/'models'/'prob_no_support_seed928.pt']+list((RUN/'models').glob('Q_*seed928.pt'))
        for path in paths:
            p=Probability(path);result[path.stem]={}
            for cohort in ['fit','tune']:result[path.stem][cohort]=probability_metrics(p,samples(cohort))
            print('metrics',path.stem,flush=True)
        write_json(RUN/'probability_training_metrics.json',result)
    write_json(RUN/'fit_manifest.json',dict(models={p.stem:dict(path=str(p.relative_to(ROOT)),**read(p.with_suffix('.json')))
        for p in (RUN/'models').glob('*.pt')},source_sha256=file_sha256(__file__)))


if __name__=='__main__':main()
