"""Official ray evaluation, end-to-end timing and leakage checks for follow-up."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.empty_decision_followup import (RUN,OLD,initialize,read,write_json,file_sha256,
    build_loader,build_network,to_cuda,support_group,seed_all)
from tools.empty_decision_optimization import support_field,candidate_features
from tools.fit_empty_decision_followup import Policy
from tools.fit_empty_decision_optimization import Predictor


def official_ray():
    import tools.audit_empty_decision_optimization as official
    official.RUN=RUN
    result=official.ray_evaluate()
    result['wrapper_source_sha256']=file_sha256(__file__)
    previous=read(OLD/'ray/summary.json')
    for new,old in [('B0','B0'),('B2','B2_eps0.005')]:
        assert result['models'][new]['counts']==previous['models'][old]['counts']
    result['class_bias_previous_counts_exact_parity']=True
    write_json(RUN/'ray/summary.json',result)
    return result


class Forward:
    def __init__(self,points):
        self.net=build_network();self.points=points;self.models={}
        for point in points.values():
            source=point['source']
            if source not in ['native','anchor','old_P3'] and source not in self.models:
                self.models[source]=Policy(RUN/'models'/f'{source}.pt')
                prob=self.models[source].record['probability_path']
                assert not prob or 'oracle' not in prob
        self.old=Predictor(OLD/'models/P3_seed928.pt')
        self.anchor=torch.as_tensor(initialize()['anchor_thresholds'],device='cuda')
    def __call__(self,batch,point):
        out=self.net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k!='img'})
        source=point['source']
        if source=='native':return out['final_occ'][0]
        logits=out['pred_occ'][-1][0];maximum,fg=logits[:17].max(0);margin=maximum-logits[17]
        ids=torch.where(margin.abs()<=2)[0];fi=fg[ids]
        if source=='anchor':score=margin[ids]-self.anchor[fi]
        else:
            cols=9 if source=='old_P3' else len(self.models[source].record['scale']['cols'])
            if cols==12:
                support,group=support_group(self.net.head,out)
            elif cols==9:support,_=support_field(self.net.head,out)
            else:support=torch.zeros_like(margin)
            x=candidate_features(out,support,ids)
            if cols==12:x=torch.cat([x[:,:8],group[ids]],-1)
            if source=='old_P3':score=self.old(x,fi)
            else:score=self.models[source](x,fi)
        pred=out['final_occ'][0].clone()
        pred[ids]=torch.where(score>=point['threshold'],fi,17)
        return pred


def primary_points():
    selection=read(RUN/'operating_points.json')
    return {p['id']:p for p in selection['operating_points'] if p['family'] in ['B0','B2'] or
        (p['budget']==.005 and p['seed'] in [None,928])}


def benchmark():
    if (RUN/'benchmark.json').exists():return read(RUN/'benchmark.json')
    seed_all(928);points=primary_points()
    old=next(p for p in read(OLD/'operating_points.json')['operating_points'] if p['family']=='P3' and p['seed']==928 and p['budget']==.005)
    points['OldP3']=dict(source='old_P3',threshold=old['threshold'])
    forward=Forward(points);_,loader,_=build_loader('val',range(8))
    times={k:[] for k in points};peaks={k:[] for k in points};rng=np.random.default_rng(928)
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            batch=to_cuda(batch);keys=list(points);rng.shuffle(keys)
            for key in keys:
                for _ in range(2):forward(batch,points[key])
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();elapsed=[]
                for _ in range(5):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();forward(batch,points[key]);end.record();end.synchronize()
                    elapsed.append(start.elapsed_time(end))
                times[key].append(elapsed);peaks[key].append(torch.cuda.max_memory_allocated()/2**20)
            print('benchmark',index+1,'/8',flush=True)
    result=dict(status='complete',frames=8,repeats=5,warmups=2,gpu=torch.cuda.get_device_name(),batch_size=1,
        scope='complete network + features + policy; excludes data loading/H2D uniformly',
        support='S uses one extra native aggregation; full support uses two; neither fused',
        models={k:dict(FPS=float(1000/np.mean(v)),mean_latency_ms=float(np.mean(v)),
            peak_allocated_MiB=max(peaks[k]),repeat_ms=v) for k,v in times.items()},source_sha256=file_sha256(__file__))
    write_json(RUN/'benchmark.json',result);return result


def invariance():
    if (RUN/'invariance.json').exists():return read(RUN/'invariance.json')
    seed_all(928);points=primary_points();forward=Forward(points)
    _,loader,_=build_loader('val',range(2));rows=[]
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            batch=to_cuda(batch);changed=dict(batch)
            for key in ['occ_label','voxel_semantics']:
                if key in changed and torch.is_tensor(changed[key]):changed[key]=torch.full_like(changed[key],17)
            for key in ['occ_cam_mask','occ_mask','occ_loss_mask','occ_lidar_mask','occ_nonempty_mask']:
                if key in changed and torch.is_tensor(changed[key]):changed[key]=~changed[key].bool()
            for name,point in points.items():
                a,b=forward(batch,point),forward(changed,point)
                assert torch.equal(a,b),name
                with np.load(RUN/'validation'/f'frame_{index:04d}.npz') as z:
                    assert np.array_equal(a.cpu().numpy().reshape(-1),z[name]),name
            rows.append(dict(index=index,methods=len(points),predictions_exact=True,full_pipeline_cache_parity=True))
    result=dict(status='passed',frames=rows,oracle_excluded=True,source_sha256=file_sha256(__file__))
    write_json(RUN/'invariance.json',result);return result


def completion():
    protocol=initialize();selection=read(RUN/'operating_points.json');validation=read(RUN/'validation/summary.json')
    assert protocol['checkpoint_sha256']==file_sha256(ROOT/protocol['checkpoint'])
    assert protocol['original_selection_sha256']==file_sha256(OLD/'operating_points.json')
    assert validation['selection_sha256']==file_sha256(RUN/'operating_points.json')
    assert selection['protocol_sha256']==file_sha256(RUN/'protocol.json')
    for name,digest in selection['model_sha256'].items():assert digest==file_sha256(RUN/'models'/f'{name}.pt')
    for name,digest in selection['probability_sha256'].items():assert digest==file_sha256(ROOT/name)
    for name,digest in selection['probe_sha256'].items():assert digest==file_sha256(ROOT/name)
    with np.load(RUN/'validation/per_frame_counts.npz') as z:c,e=z['confusion'],z['events']
    assert len(c)==1024
    gt=c.sum(-1);assert np.array_equal(gt,np.broadcast_to(gt[:,:1],gt.shape))
    tp=np.diagonal(c,axis1=-2,axis2=-1)[...,:17].sum(-1)
    assert np.array_equal(tp-tp[:,:1],e[...,0]-e[...,3])
    fp=c[...,17,:17].sum(-1);assert np.array_equal(fp-fp[:,:1],e[...,2]-e[...,5])
    semantic_fp=c[...,:,:17].sum((-1,-2))-tp
    assert np.array_equal(semantic_fp-semantic_fp[:,:1],e[...,1]+e[...,2]-e[...,4]-e[...,5])
    assert set(protocol['fit_scenes']).isdisjoint(protocol['tune_scenes'])
    assert (set(protocol['fit_scenes'])|set(protocol['tune_scenes'])).isdisjoint(protocol['validation_scenes'])
    assert read(RUN/'invariance.json')['status']=='passed'
    archived={file_sha256(p) for p in (RUN/'source_snapshots').glob('*.py')}
    archived|={file_sha256(p) for p in (ROOT/'tools').glob('*empty_decision_followup.py')}
    for path in list((RUN/'models').glob('*.json'))+list((RUN/'width_control/models').glob('*.json')):
        assert read(path)['source_sha256'] in archived,('Missing executed fit source',path)
    for name in ['train_cache.json','val_cache.json','tuning/summary.json','validation/summary.json']:
        assert read(RUN/name)['source_sha256'] in archived,('Missing executed source',name)
    ray=read(RUN/'ray/summary.json');assert ray['samples']==1024 and ray['class_bias_previous_counts_exact_parity']
    result=dict(status='complete',train_frames=4096,tuning_frames=994,validation_frames=1024,
        points=len(validation['operating_points']),decision_models=len(selection['model_sha256']),
        native_previous_exact_parity=validation['native_previous_exact_parity'],frozen_models_and_selection_unchanged=True,
        all_counts_and_directional_changes_verified=True,mask_and_gt_invariance='passed',executed_sources_archived=True,
        official_ray_models=len(ray['models']),tests_passed=9,source_sha256=file_sha256(__file__))
    width=RUN/'width_control'
    if (width/'validation.json').exists():
        extra=read(width/'validation.json');frozen=read(width/'selection.json')
        assert extra['selection_sha256']==file_sha256(width/'selection.json')
        assert frozen['main_selection_sha256']==file_sha256(RUN/'operating_points.json')
        assert datetime.fromisoformat(frozen['at']).timestamp()<=(RUN/'validation/frame_0000.json').stat().st_mtime
        for name,meta in frozen['models'].items():assert file_sha256(ROOT/meta['path'])==meta['sha256']
        with np.load(width/'per_frame_counts.npz') as z:wc=z['confusion']
        assert np.array_equal(wc.sum(-1),np.broadcast_to(c[:,:1].sum(-1),wc.sum(-1).shape))
        result['supplementary_width_control']=dict(models=len(frozen['models']),points=len(extra['points']),frozen_selection_unchanged=True)
    write_json(RUN/'completion_audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['ray','benchmark','invariance','completion']);a=p.parse_args()
    print({'ray':official_ray,'benchmark':benchmark,'invariance':invariance,'completion':completion}[a.stage]())
