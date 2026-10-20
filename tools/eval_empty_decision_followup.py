"""Full-tuning selection and immutable full-validation evaluation of follow-up heads."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.empty_decision_followup import RUN,OLD,BUDGETS,SEEDS,initialize,read,write_json,file_sha256
from tools.fit_empty_decision_followup import Policy,Probability,probability_metrics
from tools.eval_empty_decision_optimization import (
    EDGES,hist_tensor,histogram_curve_counts,histogram_counts,histogram_counts_torch,
    metrics,region_metrics,chosen_index,calibration_statistics,calibration_metrics)


def source_paths():return {p.stem:p for p in sorted((RUN/'models').glob('D_*.pt'))}


def tuning():
    protocol=initialize();dest=RUN/'tuning';dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():return read(dest/'summary.json')
    paths=source_paths()
    expected=sum(len(x) for x in protocol['seeds'].values())*5
    assert len(paths)==expected,(len(paths),expected)
    models={k:Policy(p) for k,p in paths.items()}
    anchor=torch.as_tensor(protocol['anchor_thresholds'],device='cuda')
    hists={k:torch.zeros((2,18,17,2,len(EDGES)+1),dtype=torch.int64,device='cuda') for k in ['anchor']+list(models)}
    fixed=np.zeros((2,18,18),np.int64);native=fixed.copy();frames=[];start=time.monotonic()
    frame_list=[f for f in read(RUN/'train_cache.json')['frames'] if f['full_candidates']]
    assert len(frame_list)==994
    with torch.inference_mode():
        for n,meta in enumerate(frame_list):
            with np.load(ROOT/meta['path']) as z:
                x=torch.as_tensor(z['X'],device='cuda');fg=torch.as_tensor(z['fg'].astype(np.int64),device='cuda')
                gt=torch.as_tensor(z['gt'].astype(np.int64),device='cuda');region=torch.as_tensor(z['region'],device='cuda')
                fixed+=z['fixed_confusion'];native+=z['native_confusion']
            cache={};native_fg=x[:,0]>=0
            hists['anchor']+=hist_tensor(x[:,0]-anchor[fg],gt,fg,region,native_fg)
            for name,model in models.items():hists[name]+=hist_tensor(model(x,fg,cache),gt,fg,region,native_fg)
            frames.append(meta['index'])
            if n==0 or (n+1)%64==0:
                print('tuning',n+1,'/994','seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(dest/'histograms.npz',**{k:v.cpu().numpy() for k,v in hists.items()},fixed=fixed,native=native)
    result=dict(status='complete',samples=994,frames=frames,models={k:dict(path=str(p.relative_to(ROOT)),sha256=file_sha256(p)) for k,p in paths.items()},
                elapsed_seconds=time.monotonic()-start,source_sha256=file_sha256(__file__))
    write_json(dest/'summary.json',result);return result


def select():
    if (RUN/'operating_points.json').exists():return read(RUN/'operating_points.json')
    protocol=initialize();summary=read(RUN/'tuning/summary.json')
    with np.load(RUN/'tuning/histograms.npz') as z:
        fixed,native=z['fixed'],z['native'];reference=region_metrics(native)['camera_complement']['empty_FPR']/100
        allowed=np.where((EDGES>=-1)&(EDGES<=1))[0]+1;zero=int(np.searchsorted(EDGES,0))+1
        b2,_=histogram_counts(z['anchor'],fixed,zero)
        old_point=next(p for p in read(OLD/'operating_points.json')['operating_points'] if p['family']=='B2' and p['budget']==.005)
        assert abs(region_metrics(b2)['camera']['mIoU']-old_point['tuning']['camera']['mIoU'])<1e-9
        records=[dict(id='B0',family='B0',seed=None,budget=None,source='native',threshold=0.,cuts=zero,tuning=region_metrics(native)),
                 dict(id='B2',family='B2',seed=None,budget=.005,source='anchor',threshold=0.,cuts=zero,tuning=region_metrics(b2))]
        groups=defaultdict(list)
        anchor_counts=histogram_curve_counts(z['anchor'],fixed,allowed)
        groups[('B2_shift',None)]=[dict(source='anchor',cuts=int(c),threshold=float(EDGES[c-1]),counts=cc) for c,cc in zip(allowed,anchor_counts)]
        for family,seeds in protocol['seeds'].items():
            for seed in seeds:
                groups[(family,seed)].append(dict(source='anchor',cuts=zero,threshold=0.,counts=b2,zero_residual=True))
        for name in summary['models']:
            meta=read((ROOT/summary['models'][name]['path']).with_suffix('.json'))
            curve=histogram_curve_counts(z[name],fixed,allowed)
            groups[(meta['family'],meta['seed'])]+=[dict(source=name,cuts=int(c),threshold=float(EDGES[c-1]),counts=cc,lam=meta['lam']) for c,cc in zip(allowed,curve)]
        for (family,seed),options in groups.items():
            cc=np.stack([o['counts'] for o in options]);camera=metrics(cc[:,0])['mIoU'];risk=metrics(cc[:,1])['empty_FPR']/100
            for budget in BUDGETS:
                i,feasible=chosen_index(camera,risk,budget,reference);choice=options[i]
                tag='unconstrained' if budget is None else f'eps{budget:g}'
                records.append(dict(id=f'{family}_{tag}'+('' if seed is None else f'_seed{seed}'),family=family,seed=seed,budget=budget,
                    **{k:v for k,v in choice.items() if k!='counts'},feasible=feasible,tuning=region_metrics(cc[i])))
    result=dict(status='frozen_before_validation',created_at=datetime.now().astimezone().isoformat(),reference_out_fpr=reference,
        operating_points=records,model_sha256={k:v['sha256'] for k,v in summary['models'].items()},
        probability_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in
            list((RUN/'models').glob('Q_*.pt'))+[OLD/'models'/f'prob_no_support_seed{seed}.pt' for seed in SEEDS]},
        probe_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in (RUN/'probes').glob('*.joblib')},
        histogram_sha256=file_sha256(RUN/'tuning/histograms.npz'),protocol_sha256=file_sha256(RUN/'protocol.json'),
        source_sha256=file_sha256(__file__))
    candidates=[p for p in records if p['budget']==.005 and p['seed'] in [None,928]]
    feasible=[p for p in candidates if p['tuning']['camera_complement']['empty_FPR']/100<=reference+.005+1e-12]
    result['training_selected_primary_id']=max(feasible,key=lambda p:p['tuning']['camera']['mIoU'])['id']
    unconstrained=[p for p in records if p['budget'] is None and p['seed'] in [None,928] and p['family']!='B0']
    result['oracle_region_reference']=dict(camera_policy_id=max(unconstrained,key=lambda p:p['tuning']['camera']['mIoU'])['id'],
        complement_policy_id='B2',role='GT-region routing diagnostic only; excluded from deployable ranking')
    write_json(RUN/'operating_points.json',result)
    for p in records:
        if p['budget']==.005 and p['seed'] in [None,928]:
            print(p['id'],'tune',round(p['tuning']['camera']['mIoU'],4),round(p['tuning']['camera_complement']['empty_FPR'],4),p['source'],p['threshold'],flush=True)
    return result


def validate():
    selection=read(RUN/'operating_points.json');sha=file_sha256(RUN/'operating_points.json')
    for name,digest in selection['model_sha256'].items():assert file_sha256(RUN/'models'/f'{name}.pt')==digest
    for name,digest in selection['probability_sha256'].items():assert file_sha256(ROOT/name)==digest
    for name,digest in selection['probe_sha256'].items():assert file_sha256(ROOT/name)==digest
    assert file_sha256(RUN/'protocol.json')==selection['protocol_sha256']
    dest=RUN/'validation';dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():
        cached=read(dest/'summary.json');assert cached['selection_sha256']==sha;return cached
    points=selection['operating_points'];grouped=defaultdict(list)
    for i,p in enumerate(points):
        if p['source']!='native':grouped[p['source']].append((i,p))
    models={k:Policy(RUN/'models'/f'{k}.pt') for k in grouped if k!='anchor'}
    anchor=torch.as_tensor(initialize()['anchor_thresholds'],device='cuda')
    probability_paths=[OLD/'models'/'prob_no_support_seed928.pt']+sorted((RUN/'models').glob('Q_*seed928.pt'))
    probability_models={p.stem:Probability(p) for p in probability_paths}
    primary=[i for i,p in enumerate(points) if p['family'] in ['B0','B2'] or (p['budget']==.005 and p['seed'] in [None,928])]
    frames=[];all_counts=[];all_events=[];calibration={};sample_arrays={k:[] for k in ['X','gt','fg','region','weight','frame']}
    start=time.monotonic()
    with torch.inference_mode():
        for n,meta in enumerate(read(RUN/'val_cache.json')['frames']):
            stem=dest/f'frame_{meta["index"]:04d}';meta_file=stem.with_suffix('.json')
            if meta_file.exists():
                saved=read(meta_file);assert saved['selection_sha256']==sha
                with np.load(stem.with_suffix('.npz')) as z:cc,ev=z['confusion'],z['events']
                for key,value in saved['calibration'].items():calibration[key]=calibration.get(key,0)+np.asarray(value)
            else:
                with np.load(ROOT/meta['path']) as z:data={k:z[k] for k in z.files}
                x=torch.as_tensor(data['X'],device='cuda');fg=torch.as_tensor(data['fg'].astype(np.int64),device='cuda')
                gt=torch.as_tensor(data['gt'].astype(np.int64),device='cuda');region=torch.as_tensor(data['region'],device='cuda')
                fixed=torch.as_tensor(data['fixed_confusion'],device='cuda');native=x[:,0]>=0
                cc=np.empty((len(points),2,18,18),np.int64);ev=np.zeros((len(points),2,6),np.int64)
                cc[0]=data['native_confusion'];scores={};cache={}
                for source,items in grouped.items():
                    score=x[:,0]-anchor[fg] if source=='anchor' else models[source](x,fg,cache)
                    scores[source]=score
                    h=hist_tensor(score,gt,fg,region,native)
                    counts,events=histogram_counts_torch(h,fixed,[p['cuts'] for _,p in items])
                    rows=[i for i,_ in items];cc[rows]=counts.cpu().numpy();ev[rows]=events.cpu().numpy()
                target=torch.where(gt==17,1,torch.where(gt==fg,0,2));local={}
                for name,model in probability_models.items():
                    value=model(x,fg,region);variants={'direct':value}
                    if value.shape[1]==6:
                        joint=value.reshape(-1,2,3)
                        conditional=joint[torch.arange(len(gt),device='cuda'),region.long()]
                        variants={'marginal':joint.sum(1),'conditional_diagnostic':conditional/conditional.sum(-1,keepdim=True).clamp_min(1e-20)}
                    for label,pred in variants.items():
                        key=name+'/'+label;stat=calibration_statistics(pred,target,region)
                        local[key]=stat.tolist();calibration[key]=calibration.get(key,0)+stat
                arrays={'gt':data['grid_gt'],'confusion':cc,'events':ev}
                for i in primary:
                    point=points[i];pred=data['grid_native'].copy()
                    if point['source']!='native':
                        occupied=(scores[point['source']]>=point['threshold']).cpu().numpy()
                        pred[data['voxel_id']]=np.where(occupied,data['fg'],17)
                    arrays[point['id']]=pred
                    if n<2:
                        from tools.empty_decision_optimization import confusion
                        direct=confusion(torch.as_tensor(data['grid_gt'].astype(np.int64)),torch.as_tensor(pred.astype(np.int64)),torch.as_tensor(data['grid_region']))
                        assert np.array_equal(direct,cc[i]),point['id']
                np.savez_compressed(stem.with_suffix('.npz'),**arrays)
                saved=dict(index=meta['index'],scene=meta['scene'],scene_frame=meta['scene_frame'],path=str(stem.with_suffix('.npz').relative_to(ROOT)),
                    selection_sha256=sha,calibration=local)
                write_json(meta_file,saved)
            all_counts.append(cc);all_events.append(ev);frames.append(saved)
            # Identical sampling locations and weights to the previous experiment.
            with np.load(OLD/'validation'/f'frame_{meta["index"]:04d}.npz') as old:
                voxel=old['probability_sample_voxel'];weight=old['probability_sample_weight']
                truth=old['probability_sample_outcome'];sample_region=old['probability_sample_region']
            with np.load(ROOT/meta['path']) as z:
                ids=np.searchsorted(z['voxel_id'],voxel)
                assert np.array_equal(z['voxel_id'][ids],voxel)
                for k in ['X','gt','fg','region']:sample_arrays[k].append(z[k][ids])
                assert np.array_equal(z['region'][ids],sample_region)
                outcome=np.where(z['gt'][ids]==17,1,np.where(z['gt'][ids]==z['fg'][ids],0,2))
                assert np.array_equal(outcome,truth)
            sample_arrays['weight'].append(weight);sample_arrays['frame'].append(np.full(len(ids),meta['index'],np.int16))
            if n==0 or (n+1)%64==0:print('validation',n+1,'/1024','seconds',round(time.monotonic()-start,1),flush=True)
    counts=np.stack(all_counts);events=np.stack(all_events)
    np.savez_compressed(dest/'per_frame_counts.npz',confusion=counts,events=events)
    samples={k:np.concatenate(v) for k,v in sample_arrays.items()};np.savez_compressed(RUN/'validation_samples.npz',**samples)
    outcomes={}
    for i,p in enumerate(points):
        outcomes[p['id']]={**p,'validation':region_metrics(counts[:,i].sum(0)),'transition_counts':events[:,i].sum(0).tolist()}
    with np.load(OLD/'validation/per_frame_counts.npz') as old:
        assert np.array_equal(counts[:,0],old['confusion'][:,0]),'Native predictions must reproduce exactly'
    result=dict(status='complete',samples=1024,frames=frames,operating_points=outcomes,selection_sha256=sha,
        primary_predictions=[points[i]['id'] for i in primary],calibration={k:calibration_metrics(v) for k,v in calibration.items()},
        raw_calibration={k:v.tolist() for k,v in calibration.items()},native_previous_exact_parity=True,
        elapsed_seconds=time.monotonic()-start,source_sha256=file_sha256(__file__))
    write_json(dest/'summary.json',result)
    ap={name:probability_metrics(model,samples) for name,model in probability_models.items()}
    write_json(RUN/'probability_validation_metrics.json',ap)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tune','select','validate']);a=p.parse_args()
    if a.stage=='tune':tuning()
    elif a.stage=='select':select()
    else:validate()
