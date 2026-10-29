"""All-scene evaluation for paired Gaussian geometry/evidence training trials."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tools.gaussian_geometry_causal import RUN, cfg, build, seed_all, write, sha, to_cuda, forward
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.stat_gaussian_voxel_coverage import measure, collect, accumulator, FIELDS, FLAGS
from tools.probe_gaussian_scale_supervision import capture_attributes

# A threshold is selected on the full training split only. Native metrics use
# the exact 18-class argmax, independently of this approximate PR histogram.
EDGES=np.arange(-16.,16.0001,.02)
G_NAMES=('count','sum_A','sum_alpha','sum_b','sum_sigma','maxscale_ge_03999',
         'allaxes_ge_03999','center_camera','center_fg','center_camera_empty',
         'camera_empty_and_high_evidence','sum_max_p')
RAW_NAMES=('gaussians','zero_scale_axes','all_scale_axes_zero','largest_scale_axis_zero',
           'sum_largest_axis_relative_jacobian','opacity_jacobian_zero',
           'sum_opacity_relative_jacobian')


def extra_accumulator():
    a=accumulator()
    a.update(confusion=torch.zeros((18,18),dtype=torch.int64,device='cuda'),
        margin_hist=torch.zeros((18,17,len(EDGES)+1),dtype=torch.int64,device='cuda'),
        gaussian_total=torch.zeros(len(G_NAMES),dtype=torch.float64,device='cuda'),
        raw_attribute=torch.zeros(len(RAW_NAMES),dtype=torch.float64,device='cuda'),
        loss=torch.zeros(3,dtype=torch.float64,device='cuda'))
    return a


def add_extra(a,net,out,batch,values):
    g=out['gaussian']
    means,scales,inverse,alpha,sem,xyz,labels,camera,winner,pred=values
    valid=camera&(labels>=0)&(labels<18)
    y=labels[valid].long();p=pred[valid].long();w=winner[valid].long()
    scores=out['pred_occ'][-1][0,:,valid]
    a['confusion'].view(-1).add_(torch.bincount(y*18+p,minlength=324))
    margin=scores[:17].max(0).values-scores[17]
    bins=torch.bucketize(margin.contiguous(),torch.tensor(EDGES,device=margin.device),right=True)
    key=(y*17+w)*(len(EDGES)+1)+bins
    a['margin_hist'].view(-1).add_(torch.bincount(key,minlength=a['margin_hist'].numel()))
    shape=(net.head.aggregator.H,net.head.aggregator.W,net.head.aggregator.D)
    cells=((means-net.head.aggregator.pc_min)/net.head.aggregator.grid_size).to(torch.int32).long()
    flat=(cells[:,0]*shape[1]+cells[:,1])*shape[2]+cells[:,2]
    center_camera=camera[flat]
    center_fg=(labels[flat]>=0)&(labels[flat]<17)
    center_empty=center_camera&(labels[flat]==17)
    A=sem.sum(1).double();al=alpha.double();b=A*al
    maximum=scales.max(1).values
    # Read the actual pre-sigmoid output. Rounded physical scales cannot tell
    # whether sigmoid's derivative is exactly zero.
    raw=net._scale_probe_raw[0]
    assert len(raw)==len(scales)
    sig=raw[:,:3].sigmoid();jac=.32*sig*(1-sig)
    axis=scales.argmax(1)
    selected=jac.gather(1,axis[:,None]).flatten()
    opa=raw[:,7].sigmoid();ojac=opa*(1-opa)
    a['raw_attribute']+=torch.stack((torch.tensor(float(len(raw)),device=raw.device),
        (jac==0).sum(),(jac==0).all(1).sum(),(selected==0).sum(),
        (selected/.08).double().sum(),(ojac==0).sum(),(ojac/.25).double().sum())).double()
    rows=torch.stack((torch.ones_like(A),A,al,b,scales.mean(1).double(),
        (maximum>=.3999).double(),(scales>=.3999).all(1).double(),center_camera.double(),
        center_fg.double(),center_empty.double(),(center_empty&(b>=4)).double(),
        sem.max(1).values.double()/A.clamp_min(1e-30)),1)
    a['gaussian_total']+=rows.sum(0)
    # Exact per-voxel NLL and class accuracy on all camera-valid classes.
    from torch.nn import functional as F
    weight=torch.tensor([1.01552756,1.06897009,1.30013094,1.07253735,.94637502,
        1.10087012,1.26960524,1.06258364,1.189019,1.06217292,1.00595144,
        .85706115,1.03923299,.90867526,.8936431,.85486129,.8527829,.5],device='cuda')
    nll=F.cross_entropy(scores.T,y,weight=weight,reduction='none')
    a['loss']+=torch.stack((nll.double().sum(),weight[y].double().sum(),torch.ones((),device='cuda')))


def evaluate(name,checkpoint,workers=4,splits=('val','train')):
    from dataset.utils import custom_collate_fn_temporal
    outdir=RUN/'eval'/name
    digest=sha(checkpoint)
    complete=outdir/'complete.json'
    if complete.exists():
        prior=json.loads(complete.read_text())
        assert prior['checkpoint_sha256']==digest and set(prior['splits'])==set(splits)
        expected={'train':(700,28130),'val':(150,6019)}
        for split in splits:
            scenes=sorted((outdir/split).glob('*.json'))
            assert len(scenes)==expected[split][0]
            assert sum(json.loads(p.read_text())['frames'] for p in scenes)==expected[split][1]
            assert all(p.with_suffix('.npz').exists() for p in scenes)
        return prior
    seed_all(928);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    c,net,opt,loss,state=build(checkpoint,freeze_empty=True)
    del opt,loss,state
    net.eval()
    attribute_hook=capture_attributes(net)
    start=time.perf_counter();processed=completed=0
    for split in splits:
        dataset=EvaluationFrames(c,split)
        byscene={}
        for i,(scene,frame) in enumerate(dataset.keyframes):byscene.setdefault(scene,[]).append(i)
        pending=[]
        for scene,ids in byscene.items():
            path=outdir/split/(scene+'.json')
            if path.exists():
                old=json.loads(path.read_text())
                assert old['checkpoint_sha256']==digest and old['indices']==ids
                assert path.with_suffix('.npz').exists()
                completed+=1
            else:pending.extend(ids)
        data=DataLoader(Subset(dataset,pending),batch_size=1,num_workers=workers,
            collate_fn=custom_collate_fn_temporal,pin_memory=True,persistent_workers=workers>0,
            **(dict(prefetch_factor=2) if workers else {}))
        cache=FeatureCache(net,dataset)
        scene_active=None
        for index,batch in zip(pending,data):
            scene,frame=dataset.keyframes[index]
            if scene_active is None:
                scene_active=scene;a=extra_accumulator();records=[]
            assert scene==scene_active
            batch=to_cuda(batch)
            with torch.inference_mode():
                out=forward(net,batch)
                result,info=measure(net,out,batch)
                shape=(net.head.aggregator.H,net.head.aggregator.W,net.head.aggregator.D)
                audit=collect(a,result,info,shape)
                assert audit['maximum_winner_mass_relative_error']<5e-4
                full_scores=out['pred_occ'][-1][0]
                values=(info['means'],out['gaussian'].scales[0],info['inverse'],
                    info['alpha'],info['semantic'],info['xyz'],info['labels'].int(),info['camera'],
                    full_scores[:17].argmax(0).int(),out['final_occ'].flatten().int())
                add_extra(a,net,out,batch,values)
            records.append(dict(index=index,scene_frame=int(frame),**audit));processed+=1
            if len(records)==len(byscene[scene]):
                path=outdir/split/(scene+'.json');path.parent.mkdir(parents=True,exist_ok=True)
                with path.with_suffix('.npz.tmp').open('wb') as f:
                    np.savez_compressed(f,**{k:v.cpu().numpy() for k,v in a.items()})
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                write(path,dict(status='complete',split=split,scene=scene,indices=byscene[scene],
                    frames=len(records),records=records,checkpoint_sha256=digest))
                completed+=1;elapsed=time.perf_counter()-start
                status=dict(status='running',name=name,split=split,completed_scenes=completed,
                    frames=processed,seconds=elapsed,fps=processed/elapsed)
                write(outdir/'progress.json',status)
                if completed%10==0:print(json.dumps(status),flush=True)
                scene_active=None;del a
            del out,result,info,values,batch
        assert scene_active is None
        cache.close();del cache,data,dataset
    result=dict(status='complete',name=name,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        splits=list(splits),frames_this_process=processed,seconds=time.perf_counter()-start,
        source_sha256=sha(__file__))
    write(outdir/'complete.json',result)
    attribute_hook.remove()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--splits',nargs='+',default=['val','train'])
    args=p.parse_args()
    folder=RUN/'eval'/args.name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(evaluate(args.name,args.checkpoint,args.workers,args.splits))
