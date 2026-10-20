"""Data/protocol for controlled follow-up of frozen empty-decision policies."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.empty_decision_optimization import (
    RUN as OLD, BAND, CONFIG, CHECKPOINT, read, write_json, file_sha256,
    build_loader, build_network, to_cuda, candidate_features, confusion, seed_all)

RUN = ROOT/'work_dirs/empty_decision_followup'
FEATURES = ['margin','x','y','z','range_xy','fg_top_gap','fg_entropy','log_fg_total',
            'log_support','log_neff','weighted_opacity','weighted_scale']
LAMBDAS = [0., .1, .3, 1., 3.]
BUDGETS = [0., .005, .02, .05, None]
SEEDS = [928,929,930]


def initialize():
    RUN.mkdir(exist_ok=True)
    path = RUN/'protocol.json'
    if path.exists():
        return read(path)
    original = read(OLD/'protocol.json')
    point = next(p for p in read(OLD/'operating_points.json')['operating_points']
                 if p['family']=='B2' and p['budget']==.005)
    record = dict(created_at=datetime.now().astimezone().isoformat(), config=CONFIG,checkpoint=CHECKPOINT,
        checkpoint_sha256=original['checkpoint_sha256'], original_protocol_sha256=file_sha256(OLD/'protocol.json'),
        original_selection_sha256=file_sha256(OLD/'operating_points.json'),
        train_frames=4096,validation_frames=1024, fit_scenes=original['fit_scenes'],
        tune_scenes=original['tune_scenes'],validation_scenes=original['validation_scenes'],
        features=FEATURES, band=BAND, anchor_thresholds=point['threshold'],
        fit_rows=400000,expanded_fit_rows=1200000,tune_rows=200000,
        primary_budget=.005,budgets=BUDGETS,validation_risk_matching_tolerance_pp=.25,
        decision=dict(steps=1200,checkpoint_interval=40,lambda_grid=LAMBDAS,
            optimizer='AdamW',lr=.003,lr_final=.0003,residual_amplitude=1.,
            regularization=.001,temperature_start=.4,temperature_end=.15,
            checkpoint_selection='maximum weighted tune IoU-lambda*FPR-regularization; includes zero residual',
            final_thresholds='-1 to +1, step 1/32; choose on full 994 tune frames',
            anchor='B2 primary point fixed before follow-up; B2_shift gets identical global threshold sweep'),
        probability=dict(max_steps=2400,checkpoint_interval=100,hidden=32,lr=.003,
            lr_final=.0003,weights='inverse inclusion; balanced variant normalizes each region to 1/2',
            variants=['mix_800_reuse','mix_long','balanced','joint6','mask_oracle','mix_wide','mix_expanded'],
            mask_oracle='diagnostic only, never a deployable policy'),
        stages=['matched ordinary/probability residuals','region and convergence probability diagnostics',
                'single vs full Gaussian support','expanded sampling and correct-foreground preservation'],
        policy_families=['H','Mix','Long','Balanced','Joint','Conditional','S','Support','Expanded','Safe'],
        seeds=dict(H=SEEDS,Mix=SEEDS,Joint=SEEDS,Conditional=SEEDS,S=SEEDS,Support=SEEDS,
                   Long=[928],Balanced=[928],Expanded=[928],Safe=[928]),
        safe_correct_removal_weight=.05,
        support_information_probes=dict(scopes=['camera','all'],groups=['base','S','support'],
            classifier='HistGradientBoosting',max_iter=200,max_leaf_nodes=15,min_samples_leaf=200,
            learning_rate=.06,l2_regularization=1.,early_stopping=False,seed=928,role='diagnostic only'),
        evaluation='all 1024 frames; freeze all models and threshold choices before evaluation',
        confirmation='existing validation has informed research choices; results are exploratory follow-up',
        no_joint_backbone_training=True,source_sha256=file_sha256(__file__))
    write_json(path,record)
    return record


def support_group(head,out):
    """Two production aggregations, same bins/truncation, foreground Gaussians only."""
    args = tuple(x[:,:-1] for x in head.prepare_gaussian_args(out['gaussian']))
    means,alpha,sem,scales,inverse = args
    payload = torch.zeros_like(sem)
    payload[:,:,0]=1.
    payload[:,:,1]=alpha.reshape(1,-1)
    payload[:,:,2]=scales.mean(-1)
    def aggregate(opacity,values,inv):
        return head.aggregator(out['sampled_xyz'].float().contiguous(),means.contiguous(),
            opacity.reshape(1,-1).contiguous(),values.contiguous(),scales.contiguous(),inv.contiguous())
    moments=aggregate(alpha,payload,inverse)[:,:3].clone()
    payload.zero_();payload[:,:,0]=1.
    square=aggregate(alpha.square(),payload,inverse*2)[:,0].clone()
    s=moments[:,0]
    group=torch.stack([torch.log1p(s),torch.log1p(s.square()/square.clamp_min(1e-30)),
                       moments[:,1]/s.clamp_min(1e-20),moments[:,2]/s.clamp_min(1e-20)],-1)
    assert torch.isfinite(group).all()
    return s,group


def extract(split,limit=0):
    protocol=initialize();seed_all(928)
    if split=='train':
        indices=list(range(4096))
    else:
        indices=list(range(1024))
    if limit:indices=indices[:limit]
    dataset,loader,_=build_loader(split,indices)
    net=build_network();dest=RUN/(split+'_cache');dest.mkdir(exist_ok=True)
    frames=[];start=time.monotonic()
    tune=set(protocol['tune_scenes'])
    with torch.inference_mode():
        for index,batch in zip(indices,loader):
            path=dest/f'frame_{index:04d}.npz';meta_path=path.with_suffix('.json')
            scene,sf=dataset.keyframes[index]
            if path.exists() and meta_path.exists():
                frames.append(read(meta_path));continue
            batch=to_cuda(batch)
            out=net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k!='img'})
            logits=out['pred_occ'][-1][0];top,fg=logits[:17].max(0)
            margin=top-logits[17];gt=out['sampled_label'][0].long()
            region=(~batch['occ_cam_mask'][0].reshape(-1).bool()).long()
            native=logits.argmax(0);valid=(gt>=0)&(gt<18)
            candidate=valid&(margin.abs()<=BAND)
            is_full=split=='val' or scene in tune
            if split=='train':
                with np.load(OLD/'train'/f'frame_{index:04d}.npz') as old:
                    sampled={k:old[k] for k in old.files}
                voxel=torch.as_tensor(sampled['voxel_id'],device='cuda',dtype=torch.long)
            if is_full:ids=torch.where(candidate)[0]
            else:ids=voxel
            support,group=support_group(net.head,out)
            audit={}
            if index<2:
                from tools.diagnose_empty_decision_mini import evidence
                reference,_=evidence(net.head,out,check=True)
                expected=torch.stack([torch.log1p(reference['support']),torch.log1p(reference['neff']),
                    reference['averages'][:,0],reference['averages'][:,1]],-1)
                audit['max_support_group_error']=float((group[ids]-expected[ids]).abs().max())
                assert torch.allclose(group[ids],expected[ids],atol=2e-5,rtol=2e-5),audit
            x=candidate_features(out,support,ids)
            x=torch.cat([x[:,:8],group[ids]],-1)
            outside=valid&(~candidate)
            values=dict(X=x.cpu().numpy(),gt=gt[ids].cpu().numpy().astype(np.int8),
                fg=fg[ids].cpu().numpy().astype(np.int8),region=region[ids].cpu().numpy().astype(np.int8),
                voxel_id=ids.cpu().numpy().astype(np.int32),
                fixed_confusion=confusion(gt[outside],native[outside],region[outside]),
                native_confusion=confusion(gt,native,region))
            if split=='train':
                sampled_x=x[torch.searchsorted(ids,voxel)] if is_full else x
                assert np.allclose(sampled_x[:,:9].cpu().numpy(),sampled['X'],atol=2e-5,rtol=2e-5)
                values.update(sampled_X=sampled_x.cpu().numpy(),sampled_gt=sampled['gt'],
                    sampled_fg=sampled['fg'],sampled_region=sampled['region'],sampled_weight=sampled['weight'])
            else:
                values.update(grid_gt=gt.cpu().numpy().astype(np.uint8),
                    grid_native=native.cpu().numpy().astype(np.uint8),grid_region=region.cpu().numpy().astype(np.uint8))
            np.savez_compressed(path,**values)
            meta=dict(index=index,scene=scene,scene_frame=sf,full_candidates=is_full,
                path=str(path.relative_to(ROOT)),rows=len(ids),checkpoint_sha256=protocol['checkpoint_sha256'],audit=audit)
            write_json(meta_path,meta);frames.append(meta)
            if index==0 or (index+1)%128==0:
                print('extract',split,index+1,'/',len(indices),'elapsed',round(time.monotonic()-start,1),flush=True)
    result=dict(status='complete' if not limit else 'smoke',frames=frames,elapsed_seconds=time.monotonic()-start,
                source_sha256=file_sha256(__file__))
    write_json(RUN/(split+('_smoke' if limit else '')+'_cache.json'),result)
    return result


def samples(cohort,expanded=False):
    if not (RUN/'train_cache.json').exists():
        assert not expanded
        with np.load(OLD/f'samples_{cohort}.npz') as z:return {k:z[k] for k in z.files}
    path=RUN/f'samples_{cohort}{"_expanded" if expanded else ""}.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    protocol=initialize();fit_scenes=set(protocol['fit_scenes'])
    arrays={k:[] for k in ['X','gt','fg','region','weight','frame']}
    fixed=np.zeros((2,18,18),np.int64);native=fixed.copy()
    for meta in read(RUN/'train_cache.json')['frames']:
        if (meta['scene'] in fit_scenes)!=(cohort=='fit'):continue
        with np.load(ROOT/meta['path']) as z:
            for key in ['X','gt','fg','region','weight']:arrays[key].append(z['sampled_'+key])
            arrays['frame'].append(np.full(len(z['sampled_gt']),meta['index'],np.int16))
            fixed+=z['fixed_confusion'];native+=z['native_confusion']
    data={k:np.concatenate(v) for k,v in arrays.items()};n=len(data['gt'])
    cap=1200000 if expanded else (400000 if cohort=='fit' else 200000)
    if n>cap:
        ids=np.random.default_rng(928 if cohort=='fit' else 929).choice(n,cap,replace=False)
        data={k:v[ids] for k,v in data.items()};data['weight']*=n/cap
    data.update(fixed_confusion=fixed,native_confusion=native)
    if not expanded:
        with np.load(OLD/f'samples_{cohort}.npz') as z:
            for key in ['gt','fg','region','weight','frame']:assert np.array_equal(data[key],z[key]),key
            assert np.allclose(data['X'][:,:9],z['X'],atol=2e-5,rtol=2e-5)
    np.savez_compressed(path,**data)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','train','val','prepare'])
    parser.add_argument('--limit',type=int,default=0);a=parser.parse_args()
    if a.stage=='initialize':print(initialize())
    elif a.stage=='prepare':
        for cohort,expanded in [('fit',False),('tune',False),('fit',True)]:
            d=samples(cohort,expanded);print(cohort,expanded,d['X'].shape,flush=True)
    else:extract(a.stage,a.limit)
