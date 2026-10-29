"""Resume small census and add all-frame strict 1/2/3-sigma boundaries.

Already committed primary scenes are preserved and get a sigma-only backfill.
Uncommitted primary scenes collect both diagnostics in one inference pass.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnose_small_gaussians as base
from tools.stat_boundary_gaussians import read,write,sha,forward,to_cuda
from tools.probe_gaussian_scale_supervision import layout
from tools.boundary_gaussian_stats.data import FeatureCache
from tools.gaussian_sigma_boundaries import extension

RUN=base.RUN/'sigma_boundaries'
COUNT_NAMES=('predicted','supported','boundary_any','boundary_foreground','boundary_fg_empty')


def initialize():
    cfg,primary=base.initialize()
    files=[Path(__file__),*sorted((ROOT/'tools/gaussian_sigma_boundaries').glob('*.*'))]
    files=[p for p in files if p.is_file()]
    protocol=dict(version='small_sigma_boundaries_v1',primary_protocol_sha256=sha(base.RUN/'protocol.json'),
        checkpoint_sha256=primary['checkpoint_sha256'],expected=primary['expected'],
        sigmas=[1,2,3],scopes=['all_valid_gt','camera_mask'],count_names=COUNT_NAMES,
        queries=base.Q,children=base.CHILDREN,
        support='Production candidate cube, positive FP32 opacity*kernel; power>=-r*r/2 for r=1,2,3',
        denominator='All 76800 final predicted Gaussians, including no-support; exclude global empty Gaussian',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    protocol=json.loads(json.dumps(protocol))
    if (RUN/'protocol.json').exists():
        assert read(RUN/'protocol.json')==protocol
    else:
        write(RUN/'protocol.json',protocol)
        for p in files:
            dst=RUN/'sources'/p.relative_to(ROOT)
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        write(base.RUN/'measurement_resume_extension.json',dict(reason='User requests strict 1/2/3-sigma cross-class GS',
            extension_protocol=str((RUN/'protocol.json').relative_to(ROOT)),
            preserved_primary_progress=read(base.RUN/'progress.json')))
    return cfg,protocol


def measure(network,out,batch):
    _,v=layout(network.head,out['gaussian'])
    means,scales,inverse,alpha,sem,centers,radii=v
    xyz=out['sampled_xyz'][0].contiguous()
    labels=batch['occ_label'].flatten().int().contiguous()
    camera=batch['occ_cam_mask'].flatten().bool().contiguous()
    h=network.head.aggregator
    bits=extension().membership(means,inverse,alpha,centers,radii,xyz,labels,camera,h.H,h.W,h.D)
    assert ((bits[0]&~bits[1])==0).all() and ((bits[1]&~bits[2])==0).all()
    assert ((bits[:,1]&~bits[:,0])==0).all()
    distinct=sum((bits>>c)&1 for c in range(18))
    empty=(bits>>17)&1
    foreground=distinct-empty
    flags=torch.stack((distinct>=2,foreground>=2,(foreground>0)&(empty>0)),dim=-1)
    counts=torch.cat((torch.full((3,2,1),76800,device='cuda',dtype=torch.long),
        (distinct>0).sum(-1,keepdim=True),flags.sum(-2)),dim=-1)
    qb=flags.permute(2,0,1,3).reshape(base.Q,base.CHILDREN,3,2,3).sum(1)
    key=torch.arange(6,device='cuda')[:,None]*19+distinct.reshape(6,-1)
    hist=torch.bincount(key.flatten(),minlength=6*19).reshape(3,2,19)
    transitions=(flags.long()*torch.tensor([1,2,4],device='cuda')[:,None,None,None]).sum(0)
    off=torch.arange(6,device='cuda').reshape(2,1,3)*8
    joint=torch.bincount((transitions+off).permute(0,2,1).flatten(),minlength=48).reshape(2,3,8)
    return dict(counts=counts.cpu().numpy(),hist=hist.cpu().numpy(),
        query_counts=qb.short().cpu().numpy(),transitions=joint.cpu().numpy()),bits


class Recorder:
    def __init__(self,protocol):
        self.protocol=protocol
        availability=read(ROOT/'work_dirs/boundary_gaussian_full/availability.json')
        self.expected={s:availability[s]['scene_frame_counts'] for s in ('train','val')}
        self.active=None
        self.start=time.perf_counter()
        self.initial_frames=sum(read(p)['frames'] for p in (RUN/'scenes').glob('*/*.json'))
        self.initial_scenes=len(list((RUN/'scenes').glob('*/*.json')))
        self.done_frames,self.done_scenes=self.initial_frames,self.initial_scenes
        self.split=None

    def push(self,network,out,batch,comparison=None):
        row,bits=measure(network,out,batch)
        if comparison is not None:
            torch.testing.assert_close(bits[2],comparison[0][2:4],rtol=0,atol=0)
        scene,frame=batch['feature_frame_keys'][0][0]
        source=batch['source_index']
        index=int(source.flatten()[0]) if torch.is_tensor(source) else int(np.asarray(source).reshape(-1)[0])
        if self.active is None:
            self.active=scene
            self.records=[];self.qb=[]
            self.counts=np.zeros((3,2,5),np.int64)
            self.hist=np.zeros((3,2,19),np.int64)
            self.transitions=np.zeros((2,3,8),np.int64)
        assert self.active==scene
        self.records.append(dict(index=index,scene_frame=int(frame)))
        self.qb.append(row['query_counts'])
        self.counts+=row['counts'];self.hist+=row['hist'];self.transitions+=row['transitions']
        if len(self.records)==self.expected[self.split][scene]:
            path=RUN/'scenes'/self.split/(scene+'.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream,counts=self.counts,histogram=self.hist,
                    transitions=self.transitions,query_counts=np.stack(self.qb))
            path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
            write(path,dict(status='complete',split=self.split,scene_token=scene,frames=len(self.records),
                records=self.records,indices=[r['index'] for r in self.records],
                checkpoint_sha256=self.protocol['checkpoint_sha256']))
            self.done_frames+=len(self.records);self.done_scenes+=1
            elapsed=time.perf_counter()-self.start
            fps=(self.done_frames-self.initial_frames)/max(elapsed,1e-6)
            write(RUN/'progress.json',dict(status='running',completed_scenes=self.done_scenes,total_scenes=850,
                completed_frames=self.done_frames,total_frames=34149,split=self.split,
                frames_per_second=fps,remaining_minutes=(34149-self.done_frames)/max(fps,1e-6)/60,
                updated_at=datetime.now().astimezone().isoformat()))
            if self.done_scenes%10==0:
                print(f'Sigma: {self.done_scenes}/850 scenes; {self.done_frames}/34149 frames; {fps:.2f} frames/s',flush=True)
            self.active=None


def pilot():
    from dataset.utils import custom_collate_fn_temporal
    cfg,protocol=initialize()
    network=base.build(cfg)
    checks=[]
    for split in ('val','train'):
        dataset=base.query.QueryFrames(cfg,split)
        cache=FeatureCache(network,dataset)
        batch=to_cuda(custom_collate_fn_temporal([dataset[0]]))
        with torch.inference_mode():
            out=forward(network,batch)
            row,bits=measure(network,out,batch)
            result,values,params=base.diagnostic(network,out,batch)
            torch.testing.assert_close(bits[2],result[0][2:4],rtol=0,atol=0)
            means,scales,inv,alpha,semantic,xyz,labels,camera,winner,pred=values
            subset=np.array([0,63,64,38399,38400,76799])
            m=means[subset].cpu().numpy();iv=inv[subset].cpu().numpy();a=alpha[subset].cpu().numpy()
            x=xyz.cpu().numpy();y=labels.cpu().numpy();cam=camera.cpu().numpy()
            _,layout_values=layout(network.head,out['gaussian'])
            centers=layout_values[5][subset].cpu().numpy();radii=layout_values[6][subset].cpu().numpy()
            cells=((x-params[5:])/params[3]).astype(np.int32)
            expected=np.zeros((3,2,len(subset)),np.int32)
            for g in range(len(subset)):
                delta=m[g]-x
                power=-.5*(iv[g,:3]*delta**2).sum(1)-(iv[g,3:]*delta[:,[0,1,0]]*delta[:,[1,2,2]]).sum(1)
                support=(np.abs(cells-centers[g])<=radii[g]).all(1)&(y>=0)&(y<18)&(a[g]*np.exp(power)>0)
                for s,r in enumerate((1,2,3)):
                    good=support&(power>=-.5*r*r)
                    for scope,valid in enumerate((good,good&cam)):
                        for label in np.unique(y[valid]):expected[s,scope,g]|=1<<int(label)
            np.testing.assert_array_equal(bits[:,:,subset].cpu().numpy(),expected)
            checks.append(dict(split=split,index=0,numpy_all_sigma_membership_exact=True,primary_3sigma_exact=True,
                counts=row['counts'].tolist()))
        cache.close()
        del batch,out,result,values,dataset,cache
    write(RUN/'pilot.json',dict(status='passed',checks=checks,checkpoint_sha256=protocol['checkpoint_sha256']))
    print(json.dumps(dict(status='passed',checks=checks)),flush=True)


def run(workers):
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader,Subset
    cfg,protocol=initialize()
    assert read(RUN/'pilot.json')['status']=='passed'
    recorder=Recorder(protocol)
    original=base.diagnostic
    # The original driver visits only primary scenes not yet committed.
    def extended(network,out,batch,subset=None):
        result,values,params=original(network,out,batch,subset)
        scene=batch['feature_frame_keys'][0][0][0]
        recorder.split='val' if scene in recorder.expected['val'] else 'train'
        recorder.push(network,out,batch,result)
        return result,values,params
    base.diagnostic=extended
    try:
        base.run(workers)
    finally:
        base.diagnostic=original
    network=base.build(cfg)
    for split in ('val','train'):
        recorder.split=split
        dataset=base.query.QueryFrames(cfg,split)
        missing=[i for i,(scene,_) in enumerate(dataset.keyframes) if not (RUN/'scenes'/split/(scene+'.json')).exists()]
        loader=DataLoader(Subset(dataset,missing),batch_size=1,shuffle=False,num_workers=workers,
            collate_fn=custom_collate_fn_temporal,pin_memory=True,
            **(dict(prefetch_factor=2,persistent_workers=True) if workers else {}))
        cache=FeatureCache(network,dataset)
        for batch in loader:
            batch=to_cuda(batch)
            with torch.inference_mode():
                out=forward(network,batch)
                recorder.push(network,out,batch)
            del batch,out
        assert recorder.active is None
        cache.close();del dataset,loader,cache
    assert recorder.done_scenes==850 and recorder.done_frames==34149
    write(RUN/'execution_complete.json',dict(status='complete',frames=34149,scenes=850,
        checkpoint_sha256=protocol['checkpoint_sha256'],seconds=time.perf_counter()-recorder.start))
    write(RUN/'progress.json',dict(status='complete',completed_scenes=850,completed_frames=34149))
    print('All 1/2/3-sigma scenes measured.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['pilot','run'],required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    pilot() if args.action=='pilot' else run(args.workers)
