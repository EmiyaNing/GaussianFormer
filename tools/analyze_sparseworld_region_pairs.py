#!/usr/bin/env python3
"""Collect paired regional counts and perform scene-level discovery/holdout statistics."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from tools.sparseworld_region_utils import (ROOT, CLASS_NAMES, DYNAMIC, STATIC,
    STATE_NAMES, boundary_band, joint_counts, metric_from_hist, sha256, states,
    state_metrics, voxel_centers, write_json, percentile_interval)


def collect(root):
    records = json.loads((root/'manifest.json').read_text())['samples']
    scene_names = sorted({r['scene'] for r in records})
    scene_ids = {s: i for i, s in enumerate(scene_names)}
    n = len(records)
    basic = np.zeros((n, 3, 3, 3, 2, 4), np.int64)
    joint = np.zeros((n, 3, 3, 18, 18, 18), np.int64)
    # Scene,horizon,GT,distance,boundary,camera,speed,frustum,state.
    cov = np.zeros((len(scene_names), 3, 18, 4, 2, 2, 4, 5, 4), np.int64)
    lidar_count = np.zeros((len(scene_names), 3, 2, 4), np.int64)
    broad_boundary = np.zeros((len(scene_names), 3, 2, 4), np.int64)
    errors = np.zeros((n, 3, 7), np.int64)
    xy = voxel_centers()
    distance = np.digitize(np.linalg.norm(xy[..., :2], axis=-1), [10, 20, 30]).astype(np.int64)
    block = np.broadcast_to((np.arange(200)//5)[:, None, None]*40 +
                            (np.arange(200)//5)[None, :, None], (200, 200, 16)).reshape(-1)
    region_frames, instance_rows = [], []
    sensitivity = np.zeros((n, 3, 3, 5), np.int64)
    for i, rec in enumerate(records):
        pair_path = root/'pairs'/(rec['token']+'.npz')
        attr_path = root/'attributes'/(rec['token']+'.npz')
        while not pair_path.exists() or not attr_path.exists():
            time.sleep(1)
        sid = scene_ids[rec['scene']]
        with np.load(pair_path) as pair, np.load(attr_path) as attrs:
            assert str(attrs['pair_sha256']) == sha256(pair_path), 'Attribute cache belongs to different predictions'
            inst_meta = json.loads(attr_path.with_suffix('.json').read_text())['instances']
            for j in range(3):
                g, f, p = pair['GT'][j], pair['F'][j], pair['P'][j]
                valid, occupied = g < 18, g < 17
                sem, occ = states(g, f, p), states(g, f, p, binary=True)
                band, band08 = boundary_band(g, .4), boundary_band(g, .8)
                masks = [valid, valid & pair['camera'][j], valid & pair['lidar'][j]]
                for m, mask in enumerate(masks):
                    joint[i, j, m] = joint_counts(g, f, p, mask)
                    for d, domain in enumerate([mask, mask & occupied, mask & (g == 17)]):
                        for k, state in enumerate([sem, occ]):
                            basic[i, j, m, d, k] = np.bincount(state[domain], minlength=4)
                code = g.astype(np.int64)
                for value, radix in [(distance, 4), (band, 2), (pair['camera'][j], 2),
                                      (attrs['speed'][j], 4), (attrs['frustum'][j], 5), (sem, 4)]:
                    code = code * radix + value
                cov[sid, j] += np.bincount(code[valid], minlength=18*4*2*2*4*5*4).reshape(18,4,2,2,4,5,4)
                for attr, dest in [(pair['lidar'][j], lidar_count), (band08, broad_boundary)]:
                    dest[sid, j] += np.bincount((attr[occupied].astype(np.int64)*4 + sem[occupied]),
                                               minlength=8).reshape(2,4)
                wrong = valid & (sem == 0)
                focc, pocc = f != 17, p != 17
                conditions = [occupied & ~focc & ~pocc, occupied & ~focc & pocc,
                              occupied & focc & ~pocc, occupied & focc & pocc & (f == p),
                              occupied & focc & pocc & (f != p), (g == 17) & (f == p),
                              (g == 17) & (f != p)]
                errors[i,j] = [np.count_nonzero(wrong & c) for c in conditions]
                assert errors[i,j].sum() == basic[i,j,0,0,0,0]
                sem_flat, occupied_flat = sem.reshape(-1), occupied.reshape(-1)
                block_counts = np.bincount(block[occupied_flat]*4 + sem_flat[occupied_flat],
                                            minlength=1600*4).reshape(1600,4)
                free = (g == 17).reshape(-1)
                free_counts = np.bincount(block[free]*4 + sem_flat[free], minlength=1600*4).reshape(1600,4)
                region_frames.append(pd.DataFrame(dict(sample_position=i, horizon=j+1,
                    block_id=np.arange(1600), x=np.repeat(np.arange(40)*2-39,40),
                    y=np.tile(np.arange(40)*2-39,40), occupied=block_counts.sum(1),
                    both_wrong=block_counts[:,0], only_perception=block_counts[:,1],
                    only_forecast=block_counts[:,2], both_correct=block_counts[:,3],
                    free=free_counts.sum(1), free_F_false=free_counts[:,0]+free_counts[:,1],
                    free_P_false=free_counts[:,0]+free_counts[:,2])))
                for bidx, size in enumerate([1,2,4]):
                    bins = int(80/size)
                    grid = xy[..., :2]
                    bx = np.floor((grid[...,0]+40)/size).astype(np.int64)
                    by = np.floor((grid[...,1]+40)/size).astype(np.int64)
                    codes = (bx*bins+by)*4+sem
                    counts = np.bincount(codes[occupied],minlength=bins*bins*4).reshape(-1,4)
                    supported = counts.sum(1)>=20
                    advantage = counts[:,1]-counts[:,2]
                    sensitivity[i,j,bidx] = [supported.sum(), np.count_nonzero(supported & (advantage<0)),
                                            np.count_nonzero(supported & (advantage>0)),
                                            np.count_nonzero(supported & (advantage==0)),
                                            np.count_nonzero(supported & (counts[:,0]>.5*counts.sum(1)))]
                for inst in inst_meta[j]:
                    mask = attrs['instance'][j] == inst['id']
                    if not mask.any():
                        continue
                    counts = np.bincount(sem[mask],minlength=4)
                    instance_rows.append(dict(sample_position=i, horizon=j+1, instance_token=inst['token'],
                          category=CLASS_NAMES[inst['label']], speed=inst['speed'],
                          voxels=int(mask.sum()), **dict(zip(STATE_NAMES,counts.tolist()))))
        if i%32==0 or i+1==n:
            print(f'[collect] {i+1}/{n}',flush=True)
    np.savez_compressed(root/'joint_counts.npz',joint=joint)
    np.savez_compressed(root/'region_counts.npz',basic=basic,covariates=cov,scene_names=scene_names,
               sample_scene=np.asarray([scene_ids[r['scene']] for r in records]),errors=errors,
               lidar=lidar_count,boundary08=broad_boundary,sensitivity=sensitivity)
    pd.concat(region_frames,ignore_index=True).to_parquet(root/'region_stats.parquet',index=False)
    pd.DataFrame(instance_rows).to_parquet(root/'instance_stats.parquet',index=False)
    write_json(root/'collection_audit.json',dict(status='complete',samples=n,
        four_state_conservation=True,joint_margins_replay='enforced at export',
        gt_void_voxels=int(512*3*640000-basic[:,:,0,0,0].sum()),
        regions=n*3*1600,instances=len(instance_rows),
        boundary_definition='6-neighbor valid semantic interface, nearest adjacent center distance + 0.2m',
        covariate_dimensions=['scene','horizon','gt','distance','boundary04','camera','speed','frustum','state']))


def scene_sum(values, ids, nscene=150):
    result=np.zeros((nscene,*values.shape[1:]),np.int64)
    np.add.at(result,ids,values)
    return result


def bootstrap_weights(n, rng, repeats=2000):
    return rng.multinomial(n,np.full(n,1/n),size=repeats)


def summarize(counts, weights):
    point=state_metrics(counts.sum(0))
    boot=(weights @ counts.reshape(len(counts),-1)).reshape(len(weights),*counts.shape[1:])
    bm=state_metrics(boot)
    result={k: {'value':v,'ci95':percentile_interval(bm[k])} for k,v in point.items() if k!='total'}
    result['total']=point['total']
    result['scenes']=(counts.sum(-1)>0).sum(0)
    return result


def attribute_counts(cov):
    occupied=cov[:,:,:17]
    out={}
    out['class']=cov.sum(axis=(3,4,5,6,7))
    out['distance']=occupied.sum(axis=(2,4,5,6,7))
    out['boundary']=occupied.sum(axis=(2,3,5,6,7))
    out['camera']=occupied.sum(axis=(2,3,4,6,7))
    out['speed']=cov[:,:,DYNAMIC].sum(axis=(2,3,4,5,7))
    out['frustum']=occupied.sum(axis=(2,3,4,5,6))
    out['static_frustum']=cov[:,:,STATIC].sum(axis=(2,3,4,5,6))
    out['dynamic_frustum']=cov[:,:,DYNAMIC].sum(axis=(2,3,4,5,6))
    out['semantic_group']=np.stack([cov[:,:,DYNAMIC].sum(axis=(2,3,4,5,6,7)),
                          cov[:,:,STATIC].sum(axis=(2,3,4,5,6,7)),cov[:,:,0].sum(axis=(2,3,4,5,6))],axis=2)
    return out


def standardized_contrast(cov, factor, levels, weights, dynamic=False):
    # Retain GT class and distance as strata. Compare A(high)-A(low) only
    # on strata where both groups have support, using pooled stratum weights.
    c=cov[:,:,DYNAMIC] if dynamic else cov[:,:,:17]
    if factor=='boundary':
        c=c.sum(axis=(5,6,7))
    elif factor=='camera':
        c=c.sum(axis=(4,6,7))
    elif factor=='speed':
        c=c.sum(axis=(4,5,7))
    elif factor=='frustum':
        c=c.sum(axis=(4,5,6))
    def calculate(arr):
        low,high=arr[...,levels[0],:],arr[...,levels[1],:]
        nl,nh=low.sum(-1),high.sum(-1)
        common=(nl>0)&(nh>0)
        al=np.divide(low[...,1]-low[...,2],nl,out=np.zeros_like(nl,dtype=float),where=nl>0)
        ah=np.divide(high[...,1]-high[...,2],nh,out=np.zeros_like(nh,dtype=float),where=nh>0)
        w=(nl+nh)*common
        return np.divide(((ah-al)*w).sum(axis=(-2,-1)),w.sum(axis=(-2,-1)),
                         out=np.full(w.shape[:-2],np.nan),where=w.sum(axis=(-2,-1))>0)*100
    point=calculate(c.sum(0))
    boot=(weights@c.reshape(len(c),-1)).reshape(len(weights),*c.shape[1:])
    return dict(value=point,ci95=np.nanpercentile(calculate(boot),[2.5,97.5],axis=0).T,
                method='Class x distance common-support standardization; A(high)-A(low)',levels=levels)


def select_rules(root, cov, ids):
    # Candidate list depends only on preregistered attributes, never holdout data.
    candidates=[]
    marginal=attribute_counts(cov)
    names={'distance':['0-10m','10-20m','20-30m','>=30m'],
           'boundary':['interior','boundary04'], 'camera':['camera0','camera1'],
           'speed':['speed<0.5','speed0.5-2','speed>=2','speed_unknown'],
           'frustum':['neither','future_only','history_only','both','unknown'],
           'class':CLASS_NAMES,'semantic_group':['dynamic_classes','static_classes','others']}
    rng=np.random.default_rng(20261008)
    weights=bootstrap_weights(len(ids),rng)
    for attr in names:
        for group,label in enumerate(names[attr]):
            if (attr=='class' and group==17) or ('unknown' in label):
                continue
            counts=marginal[attr][ids,:,group]
            result=summarize(counts,weights)
            for h in range(3):
                a=result['advantage']['value'][h]
                lo,hi=result['advantage']['ci95'][h]
                support=result['scenes'][h]
                candidates.append(dict(attribute=attr,group=group,label=label,horizon=h+1,
                    discovery_advantage=a,discovery_ci95=[lo,hi],scenes=support,
                    qualified=bool(support>=20 and abs(a)>=1 and (lo>0 or hi<0))))
    picked=[]
    for side in ['F','P']:
        eligible=[x for x in candidates if x['qualified'] and
                  (x['discovery_advantage']<0 if side=='F' else x['discovery_advantage']>0)]
        eligible.sort(key=lambda x:abs(x['discovery_advantage']),reverse=True)
        for candidate in eligible:
            if not any((p['attribute'],p['group'])==(candidate['attribute'],candidate['group']) for p in picked):
                picked.append(dict(candidate,side=side))
            if sum(p['side']==side for p in picked)>=2:
                break
    write_json(root/'rules.json',dict(status='frozen_before_holdout',rules=picked,
       candidates=candidates,selection='Largest absolute discovery advantage; <=2 rules per side; no duplicate attribute/group',
       exploratory=True,comparisons=len(candidates),
       uncertainty='Scene bootstrap for fixed rules; no multiplicity-adjusted significance claim'))
    return picked


def summarize_all(root):
    data=dict(np.load(root/'region_counts.npz'))
    cov=data['covariates']
    scene_names=data['scene_names'].tolist()
    split=json.loads((root/'scene_split.json').read_text())
    split_ids={k:np.asarray([i for i,s in enumerate(scene_names) if s in split[k]])
               for k in ['discovery','holdout']}
    split_ids['all']=np.arange(len(scene_names))
    basic=scene_sum(data['basic'],data['sample_scene'])
    errors=scene_sum(data['errors'],data['sample_scene'])
    rules=select_rules(root,cov,split_ids['discovery'])
    attrs=attribute_counts(cov)
    attrs['lidar']=data['lidar']
    attrs['boundary08']=data['boundary08']
    summary=dict(status='complete',state_order=STATE_NAMES,samples=512,scenes=150,
                 bootstrap_repeats=2000,seed=20261008,splits={})
    boot_json={}
    joint=dict(np.load(root/'joint_counts.npz'))['joint']
    hist_f=joint.sum(-1)
    hist_p=joint.sum(-2)
    hist_f_scene=scene_sum(hist_f,data['sample_scene'])
    hist_p_scene=scene_sum(hist_p,data['sample_scene'])
    for name,ids in split_ids.items():
        weights=bootstrap_weights(len(ids),np.random.default_rng(20261008))
        item=dict(scenes=len(ids),samples=int(np.isin(data['sample_scene'],ids).sum()),
                  states=summarize(basic[ids],weights),
                  attributes={attr:summarize(value[ids],weights) for attr,value in attrs.items()})
        item['errors']=dict(counts=errors[ids].sum(0),names=['common_miss','F_miss_P_wrong_semantic',
               'P_miss_F_wrong_semantic','both_wrong_same_occupied','both_wrong_different_occupied',
               'both_false_same','both_false_different'])
        metric={}
        for model,hist in [('F',hist_f_scene),('P',hist_p_scene)]:
            metric[model]=metric_from_hist(hist[ids].sum(0))
            boot_hist=(weights@hist[ids].reshape(len(ids),-1)).reshape(len(weights),*hist.shape[1:])
            bm=metric_from_hist(boot_hist)
            for measure in ['mIoU','OccIoU']:
                metric[model][measure+'_ci95']=percentile_interval(bm[measure])
            metric[model]['scene_equal_mIoU']=np.nanmean(metric_from_hist(hist[ids])['mIoU'],axis=0)
        item['metrics']=metric
        classes=attrs['class'][ids,:,:17]
        point=np.nanmean(state_metrics(classes.sum(0))['proportions'],axis=1)
        boot=(weights@classes.reshape(len(ids),-1)).reshape(len(weights),*classes.shape[1:])
        item['class_equal_states']=dict(value=point,
              ci95=np.nanpercentile(np.nanmean(state_metrics(boot)['proportions'],axis=2),[2.5,97.5],axis=0).transpose(1,2,0),
              classes_present=(classes.sum(axis=(0,3))>0).sum(1))
        item['standardized_contrasts']={
            'camera1_vs0':standardized_contrast(cov[ids],'camera',(0,1),weights),
            'boundary_vs_interior':standardized_contrast(cov[ids],'boundary',(0,1),weights),
            'fast_vs_stopped':standardized_contrast(cov[ids],'speed',(0,2),weights,dynamic=True),
            'future_only_vs_both':standardized_contrast(cov[ids],'frustum',(3,1),weights)}
        summary['splits'][name]=item
        boot_json[name]=dict(scenes=len(ids),repeats=len(weights),seed=20261008)
        print('[summarize]',name,flush=True)
    holdout=[]
    for rule in rules:
        h=rule['horizon']-1
        counts=attrs[rule['attribute']][:,:,rule['group']]
        result=summary['splits']['holdout']['attributes'][rule['attribute']]
        total=basic[split_ids['holdout'],h,0,1,0].sum(0)
        rule_counts=counts[split_ids['holdout'],h].sum(0)
        state=2 if rule['side']=='F' else 1
        rate=rule_counts[state]/rule_counts.sum()
        baseline=total[state]/total.sum()
        a=result['advantage']['value'][h,rule['group']]
        lo,hi=result['advantage']['ci95'][h,rule['group']]
        holdout.append(dict(rule,holdout_advantage=a,holdout_ci95=[lo,hi],
                holdout_scenes=result['scenes'][h,rule['group']],
                enrichment=rate/baseline,coverage_percent=rule_counts[state]/total[state]*100,
                direction_replicated=bool(np.sign(a)==np.sign(rule['discovery_advantage'])),
                interval_excludes_zero=bool(lo>0 or hi<0)))
    write_json(root/'summary.json',summary)
    write_json(root/'bootstrap.json',boot_json)
    write_json(root/'holdout_summary.json',dict(rules=holdout,status='complete',
               caveat='Analytical scene holdout; not untouched model evaluation data'))
    # Recompute strict labels and check optional camera/lidar protocols too.
    write_json(root/'mask_order.json',dict(masks=['none','camera','lidar'],
               domains=['all_valid','gt_occupied','gt_free'],modes=['semantic','binary'],state_order=STATE_NAMES))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--mode',choices=['collect','summarize','all'],default='all')
    args=parser.parse_args()
    if args.mode in ['collect','all']:
        collect(args.root)
    if args.mode in ['summarize','all']:
        summarize_all(args.root)


if __name__=='__main__':
    main()
