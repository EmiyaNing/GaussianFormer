#!/usr/bin/env python3
"""U1 auxiliary feature predictability, not an occupancy readout or U2 proof."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmengine import Config
from tools.future_distillation_causal_common import *


@torch.no_grad()
def feature_dataset(cfg,root,manifest,split,queries=64):
    instance,_=model_for(cfg); instance.requires_grad_(False)
    inputs,targets,metadata=[],[],[]
    for source in source_records(root,manifest,split):
        output=cached_rollout(instance,source)
        teacher=teacher_record(cfg,split,source['index'],source['scene'])
        matched=matched_targets(instance,output,teacher)
        for h,(target,valid) in matched.items():
            indices=valid.nonzero().flatten()
            generator=torch.Generator(device='cuda').manual_seed(source['index']*31+h)
            indices=indices[torch.randperm(len(indices),device='cuda',generator=generator)[:queries]]
            if not len(indices): continue
            state=output['future_query_states'][h-1]
            features=F.layer_norm(state['features'][0,indices].float(),(256,))
            center=instance._world(state['positions']).mean(2)[0,indices]
            phase=features.new_full((len(indices),1),h/6)
            future_ego=output['metas']['temporal_trajs'][0].flatten().float()/10.
            x=torch.cat([features,center/40.,phase,future_ego[None].expand(len(indices),-1)],-1)
            y=F.layer_norm(target[0,indices].float(),(256,))
            grid=output['metas']['future_occ_labels'][0,h-1]
            index=torch.floor((center-center.new_tensor(instance.pc_range[:3]))/.4).long()
            inside=((index>=0)&(index<index.new_tensor(grid.shape))).all(-1)
            labels=torch.full((len(index),),17,device='cuda',dtype=torch.long)
            p=index[inside];labels[inside]=grid[p[:,0],p[:,1],p[:,2]].long()
            inputs.append(x.cpu());targets.append(y.cpu())
            metadata.extend([dict(index=source['index'],scene=source['scene'],horizon=h,label=label) for label in labels.cpu().tolist()])
    del instance;torch.cuda.empty_cache()
    return torch.cat(inputs),torch.cat(targets),metadata


class ConditionalFeatureModel(nn.Module):
    def __init__(self,inputs):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inputs,256),nn.GELU(),nn.Linear(256,512))

    def forward(self,x):
        mean,logvar=self.net(x).chunk(2,-1)
        return mean,logvar.clamp(-6,3)


class AuxiliaryBaseline(nn.Module):
    def __init__(self,mean,logvar,identity=False):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('logvar',logvar);self.identity=identity

    def forward(self,x):
        return (x[:,:256] if self.identity else self.mean.expand(len(x),-1)),self.logvar.expand(len(x),-1)


def gaussian_nll(mean,logvar,target):
    return .5*((target-mean).square()*(-logvar).exp()+logvar+np.log(2*np.pi)).mean()


@torch.no_grad()
def evaluate(model,x,y,metadata,fixed_logvar=None):
    sums={}
    dynamic={2,3,4,5,6,7,9,10};static={1,8,11,12,13,14,15,16}
    for start in range(0,len(x),1024):
        a,b=x[start:start+1024].cuda(),y[start:start+1024].cuda()
        mean,logvar=model(a)
        if fixed_logvar is not None: logvar=fixed_logvar.expand_as(mean)
        mse=(mean-b).square().mean(-1)
        nll=.5*((mean-b).square()*(-logvar).exp()+logvar+np.log(2*np.pi)).mean(-1)
        cosine=1-F.cosine_similarity(mean,b,dim=-1)
        observed=torch.stack([mse,nll,cosine,logvar.exp().mean(-1)],-1).cpu().tolist()
        for record,values in zip(metadata[start:start+1024],observed):
            group='dynamic' if record['label'] in dynamic else 'static' if record['label'] in static else 'other_or_empty'
            for name in ('all',group):
                key=f'{record["horizon"]/2:g}s/{name}'
                row=sums.setdefault(key,dict(count=0,mse=0.,nll=0.,cosine_error=0.,variance=0.))
                row['count']+=1;row['mse']+=values[0];row['nll']+=values[1]
                row['cosine_error']+=values[2];row['variance']+=values[3]
    for row in sums.values():
        for key in ('mse','nll','cosine_error','variance'):row[key]/=row['count']
    return sums


def main(args):
    cfg=Config.fromfile(args.config);deterministic();torch.cuda.set_per_process_memory_fraction(.15)
    root=Path(args.output_dir or (cfg.causal_root+'_smoke2' if args.smoke else cfg.causal_root))
    manifest=json.loads((root/'manifest.json').read_text())
    if (root/'s2_screens.json').exists():
        from tools.audit_future_distillation_screen_alignment import audit
        audit(cfg,root)
    rows={}
    for split in ('train','val'):
        cache=root/f'u1_vectors_{split}.pth'
        if cache.exists():rows[split]=torch.load(cache,map_location='cpu',weights_only=False)
        else:
            x,y,meta=feature_dataset(cfg,root,manifest[split],split)
            rows[split]=dict(x=x,y=y,metadata=meta);torch.save(rows[split],cache)
    report=dict(status='running',arms={},protocol='auxiliary conditional teacher-feature prediction; no occupancy head changed',
                u2=dict(status='not_run',carla_python=importlib.util.find_spec('carla') is not None,
                        server_on_path=shutil.which('CarlaUE4.sh'),
                        reason='No verified identical-history/multiple-future branch dataset or executable environment has been supplied; do not substitute synthetic features for real scene evidence'))
    report['auxiliary_baselines']={}
    y=rows['train']['y'];mean=y.mean(0,keepdim=True)
    for label,identity in [('train_mean',False),('uncalibrated_student',True)]:
        prediction=rows['train']['x'][:,:256] if identity else mean.expand_as(y)
        logvar=(prediction-y).square().mean(0,keepdim=True).clamp_min(1e-3).log()
        predictor=AuxiliaryBaseline(mean,logvar,identity).cuda()
        report['auxiliary_baselines'][label]=evaluate(predictor,rows['val']['x'],rows['val']['y'],rows['val']['metadata'])
        del predictor
    updates=2 if args.smoke else 1024
    seeds=[20260930] if args.smoke else cfg.causal_seeds
    for condition in ('position_time','history','privileged_ego'):
        privileged=condition=='privileged_ego'
        for probabilistic in (False,True):
            for seed in seeds:
                seed_all(seed)
                x=rows['train']['x'].clone();vx=rows['val']['x'].clone()
                if not privileged:x[:,260:]=0.;vx[:,260:]=0.
                if condition=='position_time':x[:,:256]=0.;vx[:,:256]=0.
                y=rows['train']['y']
                model=ConditionalFeatureModel(x.shape[1]).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                generator=torch.Generator().manual_seed(seed)
                for step in range(updates):
                    index=torch.randint(len(x),(min(512,len(x)),),generator=generator)
                    mean,logvar=model(x[index].cuda());target=y[index].cuda()
                    objective=gaussian_nll(mean,logvar,target) if probabilistic else F.mse_loss(mean,target)
                    optimizer.zero_grad(set_to_none=True);objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                model.eval()
                fixed=None
                if not probabilistic:
                    residual=[]
                    with torch.no_grad():
                        for start in range(0,len(x),1024):
                            mean,_=model(x[start:start+1024].cuda());residual.append((mean-y[start:start+1024].cuda()).square().cpu())
                    fixed=torch.cat(residual).mean(0).clamp_min(1e-3).log().cuda()[None]
                label=f'{condition}_{"gaussian" if probabilistic else "deterministic"}_seed{seed}'
                result=evaluate(model,vx,rows['val']['y'],rows['val']['metadata'],fixed)
                report['arms'][label]=dict(metrics=result,updates=updates,seed=seed,privileged=privileged,probabilistic=probabilistic)
                save_json(root/'s4_conditions.json',report)
                print('[U1]',label,result.get('3s/all'),flush=True)
                del model,optimizer
    report['status']='complete';report['train_queries']=len(rows['train']['x']);report['val_queries']=len(rows['val']['x'])
    report['limitations']=['Features are conditioned on the frozen historical forecast representation, not an unlimited raw-history predictor.',
                          'Native predicted query-center voxel classes define strata, not majority labels or measured motion.',
                          'Future ego trajectory is actual GT and privileged, not an available planned action.',
                          'Positive conditional variance or improved Gaussian NLL does not prove an irreducible uncertainty bound.',
                          'U2 and history-visible/newly-visible strata are not evaluated.']
    save_json(root/'s4_conditions.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_causal.py')
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args())
