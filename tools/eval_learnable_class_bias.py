"""Training-only calibration, fitting-scene resampling and exact validation."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.learnable_class_bias import (RUN,ROOT,CACHE,ORIGINAL,BUDGETS,SHIFTS,initialize,
    read,write_json,file_sha256,load_samples,tuning_statistics,exact_counts,sorted_statistics)
from tools.fit_learnable_class_bias import (ce_fit,metric_fit,weighted_confusion)
from tools.eval_empty_decision_optimization import metrics,region_metrics


def options_for(models,stat):
    records=[];a=[]
    for model in models:
        for shift in SHIFTS:
            records.append(dict(source=model['id'],shift=float(shift),a_fit=model['a'],
                                family=model['family']))
            a.append(np.asarray(model['a'])+shift)
    a=np.stack(a);cc=exact_counts(stat,-a)
    return records,a,cc


def choose(records,a,cc,budget,reference,all_min=None,fallback=None):
    cam=metrics(cc[:,0])['mIoU'];risk=metrics(cc[:,1])['empty_FPR']/100
    overall=metrics(cc.sum(1))['mIoU']/100
    feasible=np.ones(len(cc),bool) if budget is None else risk<=reference+budget+1e-12
    if all_min is not None:feasible &= overall>=all_min-1e-12
    choices=np.where(feasible)[0]
    if not len(choices):
        assert fallback is not None,'No feasible point and no declared fallback'
        return {**fallback,'fallback':True,'fallback_reason':'no trained candidate meets tuning hard constraints'}
    i=min(choices,key=lambda k:(-cam[k],risk[k],k));record=records[i]
    return dict(**record,a=a[i].tolist(),tuning=region_metrics(cc[i]),fallback=False,
                budget=budget,counts=cc[i].tolist())


def calibrate():
    p=initialize();stat=tuning_statistics();fit=read(RUN/'fit_summary.json')
    if (RUN/'selection_training.json').exists():return read(RUN/'selection_training.json')
    baseline=exact_counts(stat,np.zeros(17))[0];assert np.array_equal(baseline,stat['native'])
    ref=region_metrics(baseline)['camera_complement']['empty_FPR']/100
    with np.load(RUN/'fitting_B2_proposals.npz') as z:thresholds=z['thresholds']
    b2models=[dict(id=f'B2_fit_{i:03d}',family='C1',a=(-t).tolist()) for i,t in enumerate(thresholds)]
    records,a,cc=options_for(b2models,stat)
    points=[];b2={}
    native=dict(id='C0',family='C0',a=np.zeros(17).tolist(),source='native',shift=0.,fallback=False,
                budget=None,tuning=region_metrics(baseline),counts=baseline.tolist())
    points.append(native)
    for budget in BUDGETS:
        tag='unconstrained' if budget is None else f'eps{budget:g}'
        point=choose(records,a,cc,budget,ref,fallback=native)
        point.update(id='C1_'+tag,family='C1');points.append(point);b2[str(budget)]=point
    learned={name:read(ROOT/meta['path']) for name,meta in fit['models'].items()}
    families={'C2':[learned['C2']],'C3':[learned[fit['selected_C3']]],
              'C4':[v for v in learned.values() if v['family']=='C4'],
              'C5':[v for v in learned.values() if v['family']=='C5']}
    candidate_best={}
    for family,models in families.items():
        records,a,cc=options_for(models,stat)
        np.savez_compressed(RUN/f'tuning/{family}_candidate_metrics.npz',a=a,
            camera=metrics(cc[:,0])['mIoU'],all_miou=metrics(cc.sum(1))['mIoU'],risk=metrics(cc[:,1])['empty_FPR'],
            source=np.array([r['source'] for r in records]),shift=np.array([r['shift'] for r in records]))
        for budget in BUDGETS:
            tag='unconstrained' if budget is None else f'eps{budget:g}'
            all_min=b2[str(budget)]['tuning']['all_valid']['mIoU']/100-.0005 if family=='C5' else None
            fallback=b2[str(budget)] if family=='C5' else native
            point=choose(records,a,cc,budget,ref,all_min,fallback)
            point.update(id=family+'_'+tag,family=family,budget=budget);points.append(point)
        # Fixed configuration for fitting-scene stability; no validation selection.
        primary=next(x for x in points if x['family']==family and x['budget']==.005)
        if family=='C5' and primary['fallback']:
            cam=metrics(cc[:,0])['mIoU']/100;overall=metrics(cc.sum(1))['mIoU']/100;risk=metrics(cc[:,1])['empty_FPR']/100
            violation=(risk-ref-.005).clip(0)**2+(b2['0.005']['tuning']['all_valid']['mIoU']/100-.0005-overall).clip(0)**2
            i=int(np.argmax(cam-1000*violation));candidate_best[family]=records[i]['source']
        else:candidate_best[family]=primary['source']
    for family in ['C2','C3']:
        model=families[family][0];counts=exact_counts(stat,-np.asarray(model['a']))[0]
        points.append(dict(id=family+'_raw',family=family+'_raw',a=model['a'],source=model['id'],shift=0.,
                           fallback=False,budget=None,tuning=region_metrics(counts),counts=counts.tolist()))
    old=read(CACHE/'operating_points.json')
    anchors=np.asarray(read(CACHE/'protocol.json')['anchor_thresholds'])
    for name,family in [('B2_historical','B2'),('B2_shift_historical','B2_shift')]:
        point=next(x for x in old['operating_points'] if x['family']==family and x['budget']==.005)
        value=-anchors-point['threshold'];counts=exact_counts(stat,-value)[0]
        actual=region_metrics(counts)
        assert abs(actual['camera']['mIoU']-point['tuning']['camera']['mIoU'])<1e-9
        points.append(dict(id=name,family=name,a=value.tolist(),source='historical_frozen',shift=0.,
                           fallback=False,budget=.005,tuning=actual,counts=counts.tolist()))
    result=dict(status='training_only_selection',at=datetime.now().astimezone().isoformat(),
        reference_risk=ref,points=points,primary_id='C5_eps0.005',
        bootstrap_config_sources=candidate_best,protocol_sha256=file_sha256(RUN/'protocol.json'),
        source_sha256=file_sha256(__file__))
    write_json(RUN/'selection_training.json',result)
    for q in points:
        if q['budget']==.005:print(q['id'],q['tuning']['camera']['mIoU'],q['tuning']['all_valid']['mIoU'],q['tuning']['camera_complement']['empty_FPR'],q['source'],'fallback',q['fallback'],flush=True)
    return result


def stability():
    protocol=initialize();selection=read(RUN/'selection_training.json');summary=read(RUN/'fit_summary.json')
    dest=RUN/'stability';dest.mkdir(exist_ok=True)
    if (dest/'selection.json').exists():return read(dest/'selection.json')
    fit=load_samples('fit');stat=tuning_statistics();scenes=protocol['fit_scenes'];scene_index={s:i for i,s in enumerate(scenes)}
    frame_info=read(ORIGINAL/'train/summary.json')['frames'];mapping={f['index']:scene_index[f['scene']] for f in frame_info if f['scene'] in scene_index}
    row_scene=np.array([mapping[int(f)] for f in fit['frame']])
    fixed_scene=np.zeros((len(scenes),2,18,18),np.float64)
    for frame in frame_info:
        if frame['index'] not in mapping:continue
        fixed_scene[mapping[frame['index']]]+=np.asarray(read((ROOT/frame['path']).with_suffix('.json'))['fixed_confusion'])
    with np.load(RUN/'ce_fit.npz') as z:ce={k:z[k] for k in z.files}
    c5source=selection['bootstrap_config_sources']['C5'];config=read(RUN/'models'/f'{c5source}.json')
    rng=np.random.default_rng(protocol['bootstrap']['seed']);points=[];models=[]
    for replicate in range(10):
        mult=rng.multinomial(len(scenes),np.full(len(scenes),1/len(scenes)))
        resampled={**fit,'weight':fit['weight']*mult[row_scene],
                   'fixed_confusion':np.tensordot(mult,fixed_scene,axes=(0,0))}
        file=dest/f'fit_{replicate:02d}.json'
        if file.exists():saved=read(file)
        else:
            c2=ce_fit(resampled,ce,0.);c3=ce_fit(resampled,ce,summary['shrink'])
            setup={k:config[k] for k in ['family','rho_out','rho_all','start']};setup['id']=f'boot{replicate:02d}_C5'
            c5=metric_fit(resampled,np.asarray(c3['a']),summary['shrink'],[setup],save_prefix=f'bootstrap{replicate}')[0]
            saved=dict(replicate=replicate,scene_multiplicity=mult.tolist(),models={'C2':c2,'C3':c3,'C5':c5},
                       main_C5_fallback=next(q for q in selection['points'] if q['id']==selection['primary_id'])['fallback'],
                       C5_config_source=c5source,source_sha256=file_sha256(__file__))
            write_json(file,saved)
        for family,record in saved['models'].items():
            model=dict(id=f'boot{replicate:02d}_{family}',family=family,a=record['a'])
            options,a,cc=options_for([model],stat)
            all_min=None
            if family=='C5':all_min=next(x for x in selection['points'] if x['id']=='C1_eps0.005')['tuning']['all_valid']['mIoU']/100-.0005
            fallback=next(x for x in selection['points'] if x['id']==('C1_eps0.005' if family=='C5' else 'C0'))
            point=choose(options,a,cc,.005,selection['reference_risk'],all_min,fallback)
            point.update(id=model['id'],family='bootstrap_'+family,replicate=replicate,budget=.005)
            points.append(point);models.append(model)
        print('fitting-scene resampling',replicate+1,'/10',flush=True)
    result=dict(status='training_selected_before_validation',points=points,models=models,C5_config_source=c5source,
                main_C5_fallback=next(q for q in selection['points'] if q['id']==selection['primary_id'])['fallback'])
    write_json(dest/'selection.json',result);return result


def freeze():
    path=RUN/'operating_points.json'
    if path.exists():return read(path)
    selection=read(RUN/'selection_training.json');extra=read(RUN/'stability/selection.json')
    files=list((RUN/'models').glob('*.json'))+list((RUN/'stability').glob('fit_*.json'))
    result=dict(status='frozen_before_validation',at=datetime.now().astimezone().isoformat(),
                points=selection['points']+extra['points'],primary_id=selection['primary_id'],
                model_sha256={str(f.relative_to(ROOT)):file_sha256(f) for f in files},
                protocol_sha256=file_sha256(RUN/'protocol.json'),selection_training_sha256=file_sha256(RUN/'selection_training.json'),
                stability_selection_sha256=file_sha256(RUN/'stability/selection.json'),source_sha256=file_sha256(__file__))
    snapshots=RUN/'source_snapshots';snapshots.mkdir(exist_ok=True)
    for file in list((ROOT/'tools').glob('*learnable_class_bias.py'))+[ROOT/'model/head/learnable_class_empty_bias.py']:
        shutil.copy2(file,snapshots/file.name)
    write_json(path,result);return result


def validate():
    selection=read(RUN/'operating_points.json');sha=file_sha256(RUN/'operating_points.json')
    assert selection['protocol_sha256']==file_sha256(RUN/'protocol.json')
    for path,digest in selection['model_sha256'].items():assert file_sha256(ROOT/path)==digest
    dest=RUN/'validation';dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():return read(dest/'summary.json')
    points=selection['points'];thresholds=-np.asarray([p['a'] for p in points]);frames=[];all_cc=[];all_ev=[];start=time.monotonic()
    primary=['C0','B2_shift_historical','C1_eps0.005','C2_eps0.005','C3_eps0.005','C4_eps0.005','C5_eps0.005']
    for n,meta in enumerate(read(CACHE/'val_cache.json')['frames']):
        stem=dest/f'frame_{meta["index"]:04d}';path=stem.with_suffix('.npz');jpath=stem.with_suffix('.json')
        if jpath.exists():
            saved=read(jpath);assert saved['selection_sha256']==sha
            with np.load(path) as z:cc,ev=z['confusion'],z['events']
        else:
            with np.load(ROOT/meta['path']) as z:
                margin=z['X'][:,0];fg=z['fg'];gt=z['gt'];region=z['region'];voxel=z['voxel_id']
                fixed=z['fixed_confusion'];native=z['native_confusion'];grid_gt=z['grid_gt'];grid_native=z['grid_native'];grid_region=z['grid_region']
            code=region.astype(np.int16)*306+gt.astype(np.int16)*17+fg
            stat=sorted_statistics(margin,code,fixed,native);cc,ev=exact_counts(stat,thresholds,events=True)
            assert np.array_equal(cc[0],native)
            arrays=dict(gt=grid_gt,confusion=cc,events=ev)
            for name in primary:
                i=next(i for i,p in enumerate(points) if p['id']==name)
                prediction=grid_native.copy();prediction[voxel]=np.where(margin>=thresholds[i,fg],fg,17)
                arrays[name]=prediction
                if n<2:
                    codes=grid_region.astype(np.int64)*324+grid_gt.astype(np.int64)*18+prediction
                    direct=np.bincount(codes,minlength=648).reshape(2,18,18)
                    assert np.array_equal(direct,cc[i]),name
            np.savez_compressed(path,**arrays)
            saved=dict(index=meta['index'],scene=meta['scene'],scene_frame=meta['scene_frame'],
                path=str(path.relative_to(ROOT)),selection_sha256=sha)
            write_json(jpath,saved)
        all_cc.append(cc);all_ev.append(ev);frames.append(saved)
        if n==0 or (n+1)%128==0:
            print('validation',n+1,'/1024','points',len(points),'seconds',round(time.monotonic()-start,1),flush=True)
    counts=np.stack(all_cc);events=np.stack(all_ev)
    np.savez_compressed(dest/'per_frame_counts.npz',confusion=counts,events=events)
    outcomes={p['id']:{**p,'validation':region_metrics(counts[:,i].sum(0)),
                      'transition_counts':events[:,i].sum(0).tolist()} for i,p in enumerate(points)}
    with np.load(CACHE/'validation/per_frame_counts.npz') as old:assert np.array_equal(counts[:,0],old['confusion'][:,0])
    result=dict(status='complete',samples=1024,frames=frames,operating_points=outcomes,primary_predictions=primary,
                selection_sha256=sha,native_previous_exact_parity=True,source_sha256=file_sha256(__file__),
                elapsed_seconds=time.monotonic()-start)
    write_json(dest/'summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['calibrate','stability','freeze','validate']);a=parser.parse_args()
    {'calibrate':calibrate,'stability':stability,'freeze':freeze,'validate':validate}[a.stage]()
