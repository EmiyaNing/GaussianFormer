"""Official rays, real inference timing and independent completion checks."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import hashlib
import re
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.learnable_class_bias import (RUN,CACHE,ORIGINAL,initialize,read,write_json,file_sha256,
    build_loader,build_network,to_cuda,seed_all)
from tools.audit_empty_decision_optimization import official_setup
from tools.eval_center_support import RayCounts


def primary_points():
    validation=read(RUN/'validation/summary.json')
    return {name:validation['operating_points'][name] for name in validation['primary_predictions']}


class Forward:
    def __init__(self,points):
        from model.head.learnable_class_empty_bias import LearnableClassEmptyBias
        self.net=build_network();self.heads={}
        for name,p in points.items():
            head=LearnableClassEmptyBias(initialize()['original_bias']).cuda().eval()
            with torch.no_grad():head.class_bias.copy_(head.original_bias+torch.as_tensor(p['a'],device='cuda',dtype=torch.float64))
            self.heads[name]=head
    def __call__(self,batch,name):
        output=self.net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k!='img'})
        if name=='C0':return output['final_occ'][0].reshape(-1)
        return self.heads[name].predict(output['pred_occ'][-1])[0].reshape(-1)


def ray_evaluate():
    dest=RUN/'ray';dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists():return read(dest/'summary.json')
    seed_all(928);validation=read(RUN/'validation/summary.json');ids=validation['primary_predictions']
    ray,poses,pose_index,cache,mini,provenance=official_setup()
    directions=torch.from_numpy(ray.generate_lidar_rays());counts={k:RayCounts() for k in ids}
    first_pred={k:[] for k in ids};first_gt=[];parity={k:False for k in ids};started=time.monotonic()
    with torch.no_grad():
        for n,frame in enumerate(validation['frames']):
            stem=ROOT/frame['path'];meta=read(stem.with_suffix('.json'));result_path=dest/f'frame_{frame["index"]:04d}.json'
            if result_path.exists() and n>=8:
                saved=read(result_path);assert saved['selection_sha256']==validation['selection_sha256']
                assert saved['official_source_sha256']==provenance['sha256']
                for k,v in saved['counts'].items():counts[k].gt+=v['gt'];counts[k].pred+=v['pred'];counts[k].tp+=v['tp']
                continue
            token=mini['infos'][meta['scene']][meta['scene_frame']]['token'];_,origins=poses[pose_index[token]];origins=origins[None]
            with np.load(stem.with_suffix('.npz')) as z:
                gt=z['gt'].reshape(200,200,16);cached=cache/(token+'.npz')
                assert (gt<18).all();gt_sha=hashlib.sha256(gt.astype(np.int64).tobytes()).hexdigest()
                if cached.exists():
                    with np.load(cached) as c:
                        gt_ray=c['rays'];assert np.array_equal(c['origins'],origins.numpy())
                        if 'gt_sha256' in c:assert str(c['gt_sha256'])==gt_sha
                else:
                    gt_ray=ray.process_one_sample(gt.astype(np.int64),directions,origins)
                    np.savez_compressed(cached,rays=gt_ray,origins=origins.numpy(),gt_sha256=gt_sha)
                valid=gt_ray[:,0].astype(np.int32)!=17;trimmed_gt=gt_ray[valid]
                if n<8:first_gt.append(trimmed_gt)
                saved=dict(selection_sha256=validation['selection_sha256'],official_source_sha256=provenance['sha256'],counts={})
                for name in ids:
                    pred_ray=ray.process_one_sample(z[name].reshape(200,200,16).astype(np.int64),directions,origins)[valid]
                    current=RayCounts();current.add(pred_ray,trimmed_gt);counts[name].add(pred_ray,trimmed_gt)
                    saved['counts'][name]=dict(gt=current.gt.tolist(),pred=current.pred.tolist(),tp=current.tp.tolist())
                    if n<8:
                        first_pred[name].append(pred_ray)
                        assert np.allclose(np.asarray(ray.calc_metrics(first_pred[name],first_gt)),counts[name].iou(),rtol=0,atol=0,equal_nan=True)
                        parity[name]=True
            write_json(result_path,saved)
            if n==0 or (n+1)%128==0:print('official ray',n+1,'/1024','models',len(ids),'seconds',round(time.monotonic()-started,1),flush=True)
    results={}
    for name,c in counts.items():
        iou=c.iou();results[name]=dict(RayIoU=float(np.nanmean(iou)*100),
            **{f'RayIoU@{distance}':float(np.nanmean(iou[i])*100) for i,distance in enumerate([1,2,4])},
            counts=dict(gt=c.gt.tolist(),predicted=c.pred.tolist(),true_positive=c.tp.tolist()))
    old=read(CACHE/'ray/summary.json');assert results['C0']['counts']==old['models']['B0']['counts']
    assert results['B2_shift_historical']['counts']==old['models']['B2_shift_eps0.005']['counts']
    result=dict(status='complete',samples=1024,models=results,official_source=provenance,official_streaming_parity=parity,
        baseline_and_historical_exact_parity=True,elapsed_seconds=time.monotonic()-started,source_sha256=file_sha256(__file__))
    write_json(dest/'summary.json',result);return result


def benchmark():
    if (RUN/'benchmark.json').exists():return read(RUN/'benchmark.json')
    seed_all(928);points=primary_points();forward=Forward(points);_,loader,_=build_loader('val',range(8))
    rng=np.random.default_rng(928);times={k:[] for k in points};memory={k:[] for k in points}
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            batch=to_cuda(batch);keys=list(points);rng.shuffle(keys)
            for name in keys:
                for _ in range(2):forward(batch,name)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();elapsed=[]
                for _ in range(5):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();forward(batch,name);end.record();end.synchronize();elapsed.append(start.elapsed_time(end))
                times[name].append(elapsed);memory[name].append(torch.cuda.max_memory_allocated()/2**20)
            print('full pipeline timing',index+1,'/8',flush=True)
    result=dict(status='complete',frames=8,warmups=2,repeats=5,scope='whole frozen network + bias; excludes loading and H2D',
        gpu=torch.cuda.get_device_name(),batch_size=1,
        models={k:dict(FPS=float(1000/np.mean(v)),latency_ms=float(np.mean(v)),peak_MiB=float(max(memory[k])),repeat_ms=v) for k,v in times.items()},
        source_sha256=file_sha256(__file__))
    write_json(RUN/'benchmark.json',result);return result


def invariance():
    if (RUN/'invariance.json').exists():return read(RUN/'invariance.json')
    seed_all(928);points=primary_points();forward=Forward(points);_,loader,_=build_loader('val',range(2));rows=[]
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            batch=to_cuda(batch);changed=dict(batch)
            for key in ['occ_label','voxel_semantics']:
                if key in changed and torch.is_tensor(changed[key]):changed[key]=torch.full_like(changed[key],17)
            for key in ['occ_cam_mask','occ_mask','occ_loss_mask','occ_lidar_mask','occ_nonempty_mask']:
                if key in changed and torch.is_tensor(changed[key]):changed[key]=~changed[key].bool()
            for name in points:
                a,b=forward(batch,name),forward(changed,name);assert torch.equal(a,b),name
                with np.load(RUN/'validation'/f'frame_{index:04d}.npz') as z:assert np.array_equal(a.cpu().numpy(),z[name]),name
            rows.append(dict(frame=index,methods=len(points),exact_mask_gt_invariance=True,cache_prediction_parity=True))
    result=dict(status='passed',frames=rows,source_sha256=file_sha256(__file__));write_json(RUN/'invariance.json',result);return result


def completion():
    protocol=initialize();selection=read(RUN/'operating_points.json');validation=read(RUN/'validation/summary.json')
    assert file_sha256(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    assert file_sha256(ORIGINAL/'operating_points.json')==protocol['original_selection_sha256']
    assert selection['protocol_sha256']==file_sha256(RUN/'protocol.json')
    assert validation['selection_sha256']==file_sha256(RUN/'operating_points.json')
    assert selection['selection_training_sha256']==file_sha256(RUN/'selection_training.json')
    assert selection['stability_selection_sha256']==file_sha256(RUN/'stability/selection.json')
    assert datetime.fromisoformat(selection['at']).timestamp()<=(RUN/'validation/frame_0000.json').stat().st_mtime
    for p,h in selection['model_sha256'].items():assert file_sha256(ROOT/p)==h
    assert read(RUN/'ce_manifest.json')['frames']==list(range(4096))
    assert len(read(RUN/'tuning/manifest.json')['frames'])==994
    assert len(validation['frames'])==1024
    with np.load(RUN/'validation/per_frame_counts.npz') as z:c,e=z['confusion'],z['events']
    population=c.sum(-1);assert np.array_equal(population,np.broadcast_to(population[:,:1],population.shape))
    tp=np.diagonal(c,axis1=-2,axis2=-1)[...,:17].sum(-1)
    assert np.array_equal(tp-tp[:,:1],e[...,0]-e[...,3])
    fp=c[...,17,:17].sum(-1);assert np.array_equal(fp-fp[:,:1],e[...,2]-e[...,5])
    semantic_fp=c[...,:,:17].sum((-1,-2))-tp
    assert np.array_equal(semantic_fp-semantic_fp[:,:1],e[...,1]+e[...,2]-e[...,4]-e[...,5])
    assert read(RUN/'invariance.json')['status']=='passed'
    ray=read(RUN/'ray/summary.json');assert ray['samples']==1024 and ray['baseline_and_historical_exact_parity']
    matches=re.findall(r'(\d+) passed',(RUN/'tests.log').read_text());assert matches
    assert set(protocol['fit_scenes']).isdisjoint(protocol['tune_scenes'])
    assert (set(protocol['fit_scenes'])|set(protocol['tune_scenes'])).isdisjoint(protocol['validation_scenes'])
    ce=read(RUN/'ce_shrink_selection.json');assert max(x['projected_gradient_max'] for x in ce['candidates'])<1e-5
    archive={file_sha256(p) for p in (RUN/'source_snapshots').glob('*.py')}
    for name in ['ce_manifest.json','fit_summary.json','validation/summary.json']:
        assert read(RUN/name)['source_sha256'] in archive
    bootstrap_models=sum(len(read(path)['models']) for path in (RUN/'stability').glob('fit_*.json'))
    result=dict(status='complete',train_frames=4096,tuning_frames=994,validation_frames=1024,
        fitted_main_models=len(read(RUN/'fit_summary.json')['models']),bootstrap_models=bootstrap_models,points=len(selection['points']),
        official_ray_methods=len(ray['models']),tests_passed=int(matches[-1]),
        selection_and_models_unchanged=True,exact_population_and_event_identities=True,
        full18_ce_verified=True,convex_CE_KKT_checked=True,mask_and_gt_invariance=True,
        baseline_and_historical_parity=True,source_sha256=file_sha256(__file__))
    write_json(RUN/'completion_audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['ray','benchmark','invariance','completion']);a=parser.parse_args()
    result={'ray':ray_evaluate,'benchmark':benchmark,'invariance':invariance,'completion':completion}[a.stage]()
    print(result['status'])
