"""Full-scene frozen inference: disjoint GT support types and evidence factors.

python3.10 tools/stat_gaussian_support_regions.py --action pilot
python3.10 tools/stat_gaussian_support_regions.py --action run
The original full-census protocol and its primary source files are not changed.
"""
import argparse
from datetime import datetime
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
from mmengine import Config
from tools.stat_boundary_gaussians import CONFIG,CHECKPOINT,build,diagnostic,forward,to_cuda,read,write,sha
from tools.boundary_gaussian_stats.data import EvaluationFrames,FeatureCache

OLD=ROOT/'work_dirs/boundary_gaussian_full'
RUN=ROOT/'work_dirs/gaussian_support_regions_full'
SCOPES=('native_camera','core3sigma_camera')
GROUPS=('foreground_mixed','one_foreground_plus_empty','pure_foreground','pure_empty','unsupported')
FIELDS=('log_geometry','log_effective_opacity','log_amplitude','log_concentration','log_empty',
        'log_margin','log_opacity_mass','log_max_foreground','log_opacity_mass_squared',
        'log_amplitude_squared','log_opacity_mass_times_log_amplitude',
        'geometry_mass','opacity_mass','amplitude','concentration','max_foreground','empty')
SHAPE=(2,5,5,18,4)
JOINT_EDGES=np.linspace(-16,16,33)


def gaussian_types(bits):
    """Disjoint: >=2 FG; 1 FG+empty; 1 FG only; empty only; no GT support."""
    fg=bits & ((1<<17)-1)
    many=(fg>0)&((fg&(fg-1))!=0)
    result=torch.full_like(bits,4,dtype=torch.long)
    result[(fg>0)&~many]=2
    result[(fg>0)&~many&((bits&(1<<17))!=0)]=1
    result[bits==(1<<17)]=3
    result[many]=0
    return result


def new_accumulator(device='cuda'):
    return dict(gaussian_counts=torch.zeros((2,5),device=device,dtype=torch.int64),
        gaussian_sums=torch.zeros((2,5,5),device=device,dtype=torch.float64),
        strata_counts=torch.zeros(SHAPE,device=device,dtype=torch.int64),
        field_counts=torch.zeros(SHAPE,device=device,dtype=torch.int64),
        field_sums=torch.zeros((*SHAPE,len(FIELDS)),device=device,dtype=torch.float64),
        joint_hist=torch.zeros((2,5,5,32,32),device=device,dtype=torch.int64))


def native_mass(head,out):
    """Measure C and O with the production renderer, using auxiliary channels."""
    full=head.prepare_gaussian_args(out['gaussian'])
    means,alpha,semantic,scales,inverse=[x[:,:-1].contiguous() for x in full]
    auxiliary=torch.zeros_like(semantic)
    auxiliary[...,0]=(alpha[...,0]>0).to(semantic.dtype)
    auxiliary[...,1]=alpha[...,0]
    rendered=head.aggregator(out['sampled_xyz'],means,torch.ones_like(alpha).reshape(1,-1),
                             auxiliary,scales,inverse)
    return rendered[:,:2].T.contiguous()


def collect(acc,result,values,params,out,head):
    means,scales,inverse,alpha,semantics,xyz,labels,camera,winner,pred=values
    mass=native_mass(head,out)
    mask=camera&(labels>=0)&(labels<18)
    y=labels[mask].long();p=pred[mask].long()
    outcome=torch.where(y==17,torch.where(p==17,4,3),torch.where(p==17,2,torch.where(p==y,0,1)))
    distance=xyz[mask,:2].square().sum(1).sqrt()
    ranges=torch.bucketize(distance,torch.tensor([10.,20.,30.],device=y.device),right=True)
    logits=out['pred_occ'][-1][0,:,mask]
    maximum=logits[:17].max(0).values.double()
    total=logits[:17].sum(0).double();empty=logits[17].double()
    geom=mass[0,mask].double();opacity=mass[1,mask].double()
    # Keep zero/underflow cases in the outcome census; omit them only from logs.
    valid=(geom>1e-30)&(opacity>1e-30)&(total>1e-30)&(maximum>1e-30)&(empty>0)
    C=geom[valid];O=opacity[valid];D=total[valid];M=maximum[valid];E=empty[valid]
    lc,lo,ld,lm,le=[t.log() for t in (C,O,D,M,E)]
    la=ld-lo;lt=lm-ld
    fields=torch.stack((lc,lo-lc,la,lt,le,lm-le,lo,lm,lo.square(),la.square(),lo*la,
                        C,O,D/O,M/D,M,E),1)
    obin=((lo/np.log(2)+16).floor()).long().clamp(0,31)
    abin=((la/np.log(2)+16).floor()).long().clamp(0,31)
    semsum=semantics.sum(1).double()
    gfields=torch.stack((semsum,semantics.max(1).values.double(),alpha.double(),
                         scales.mean(1).double(),scales.max(1).values.double()),1)
    frame_gc=[];frame_ec=[]
    for s,old_scope in enumerate((1,3)):
        gtype=gaussian_types(result[0][old_scope])
        gc=torch.bincount(gtype,minlength=5)
        acc['gaussian_counts'][s]+=gc
        acc['gaussian_sums'][s].index_add_(0,gtype,gfields)
        packed=result[3][s,mask]
        owner=(0xffffffff-(packed&0xffffffff)).long()
        group=torch.full_like(owner,4)
        selected=packed>0
        group[selected]=gtype[owner[selected]]
        assert not ((group==4)&selected).any()
        flat=(((group*5+outcome)*18+y)*4+ranges)
        acc['strata_counts'][s].view(-1).add_(torch.bincount(flat,minlength=5*5*18*4))
        flat_valid=flat[valid]
        acc['field_counts'][s].view(-1).add_(torch.bincount(flat_valid,minlength=5*5*18*4))
        acc['field_sums'][s].view(-1,len(FIELDS)).index_add_(0,flat_valid,fields)
        joint=((group[valid]*5+outcome[valid])*32+obin)*32+abin
        acc['joint_hist'][s].view(-1).add_(torch.bincount(joint,minlength=5*5*32*32))
        ec=torch.bincount(group*5+outcome,minlength=25).reshape(5,5).cpu().numpy()
        previous=result[4][old_scope].cpu().numpy()
        expected=np.stack((previous[5],previous[4]-previous[5],
            previous[7]*[1,1,1,0,0],previous[7]*[0,0,0,1,1],previous[8]))
        np.testing.assert_array_equal(ec,expected)
        frame_gc.append(gc.cpu().tolist());frame_ec.append(ec.tolist())
    return frame_gc,frame_ec,dict(valid_log_voxels=int(valid.sum()),camera_voxels=int(mask.sum()),
        factor_identity_max_error=float((lc+(lo-lc)+la+lt-le-(lm-le)).abs().max()) if len(lc) else 0.)


def initialize():
    previous=read(OLD/'protocol.json')
    # All original inference/diagnostic source hashes must still match.
    for file,digest in previous['source_sha256'].items():
        assert sha(ROOT/file)==digest,file
    files=['tools/stat_gaussian_support_regions.py']
    protocol=dict(config=str(CONFIG.relative_to(ROOT)),checkpoint=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=previous['checkpoint_sha256'],base_protocol_sha256=sha(OLD/'protocol.json'),
        scopes=SCOPES,groups=GROUPS,fields=FIELDS,range_edges_m=[0,10,20,30,'infinity'],
        joint_log2_edges=JOINT_EDGES.tolist(),expected=previous['expected'],
        mode='Frozen model, unchanged full native predictions; no fitting, augmentation or interventions',
        grouping='GT support in selected camera scope; voxel dominant contributor to highest foreground class; no GT used to choose contributor',
        geometry='Production native renderer auxiliary channels. C=sum kernel over positive-opacity Gaussians; O=sum opacity*kernel; D=sum 17 foreground logits; M=max foreground logit; S=D/O; T=M/D; M=C*(O/C)*S*T',
        log_valid='All outcomes counted; factor moments require C,O,D,M>1e-30 and E>0',
        source_sha256={f:sha(ROOT/f) for f in files})
    if (RUN/'protocol.json').exists():assert read(RUN/'protocol.json')==json.loads(json.dumps(protocol))
    else:
        write(RUN/'protocol.json',protocol)
        import shutil
        for f in files:
            target=RUN/'sources'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,target)
    return Config.fromfile(CONFIG),protocol


def pilot():
    cfg,protocol=initialize();network=build(cfg);audits=[]
    from dataset.utils import custom_collate_fn_temporal
    for split in ('val','train'):
        dataset=EvaluationFrames(cfg,split);cache=FeatureCache(network,dataset)
        for index in (0,7):
            batch=to_cuda(custom_collate_fn_temporal([dataset[index]]))
            with torch.inference_mode():
                out=forward(network,batch);result,values,params=diagnostic(network,out,batch)
                acc=new_accumulator();gc,ec,audit=collect(acc,result,values,params,out,network.head)
                # Independent native renderer with semantic channels 1 and alpha=1:
                # C and O must equal its corresponding native accumulated logits.
                means,scales,inverse,alpha,sem,xyz,labels,camera,_,_=values
                full=network.head.prepare_gaussian_args(out['gaussian'])
                args=[x[:,:-1].contiguous() for x in full]
                ones=torch.ones_like(args[2])*(args[1]>0).to(args[2].dtype)
                ones[...,1]=args[1].flatten()
                reference=network.head.aggregator(xyz.unsqueeze(0).contiguous(),args[0],
                    torch.ones_like(args[1]).reshape(1,-1),ones,args[3],args[4])
                torch.testing.assert_close(native_mass(network.head,out).T,reference[:,:2],atol=0,rtol=0)
                old_scene=read(OLD/'scenes'/split/(dataset.keyframes[index][0]+'.json'))
                old=next(r for r in old_scene['records'] if r['index']==index)
                np.testing.assert_array_equal(result[4].cpu().numpy(),old['error_counts'])
                audits.append(dict(split=split,index=index,geometry_native_renderer_parity=True,
                    previous_census_error_counts_exact=True,
                    gaussian_counts=gc,error_counts=ec,**audit))
            del batch,out,result,values,acc,reference
        cache.close()
    write(RUN/'pilot.json',dict(status='passed',audits=audits,checkpoint_sha256=protocol['checkpoint_sha256']))
    print(json.dumps(dict(status='passed',frames=len(audits))),flush=True)


def run(workers):
    cfg,protocol=initialize()
    assert read(RUN/'pilot.json')['status']=='passed'
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader,Subset
    network=build(cfg)
    start=time.perf_counter();processed=0;completed=0
    for split in ('val','train'):
        dataset=EvaluationFrames(cfg,split);by_scene={}
        for index,(scene,frame) in enumerate(dataset.keyframes):by_scene.setdefault(scene,[]).append(index)
        pending=[]
        for scene,ids in by_scene.items():
            path=RUN/'scenes'/split/(scene+'.json')
            if path.exists():
                prior=read(path)
                assert prior['indices']==ids and prior['checkpoint_sha256']==protocol['checkpoint_sha256']
                assert path.with_suffix('.npz').exists()
                completed+=1
            else:pending.extend(ids)
        loader=DataLoader(Subset(dataset,pending),batch_size=1,shuffle=False,num_workers=workers,
            collate_fn=custom_collate_fn_temporal,pin_memory=True,
            **(dict(prefetch_factor=2,persistent_workers=True) if workers else {}))
        cache=FeatureCache(network,dataset);active=None
        for index,batch in zip(pending,loader):
            scene,frame=dataset.keyframes[index]
            if active is None:
                active=scene;acc=new_accumulator();records=[]
                old=read(OLD/'scenes'/split/(scene+'.json'))
                old_records={r['index']:r for r in old['records']}
            assert active==scene
            batch=to_cuda(batch)
            with torch.inference_mode():
                out=forward(network,batch);result,values,params=diagnostic(network,out,batch)
                np.testing.assert_array_equal(result[4].cpu().numpy(),old_records[index]['error_counts'])
                gc,ec,audit=collect(acc,result,values,params,out,network.head)
            records.append(dict(index=index,scene_frame=int(frame),sample_token=old_records[index]['sample_token'],
                                gaussian_counts=gc,error_counts=ec,**audit))
            processed+=1
            if len(records)==len(by_scene[scene]):
                path=RUN/'scenes'/split/(scene+'.json');path.parent.mkdir(parents=True,exist_ok=True)
                arrays={k:v.cpu().numpy() for k,v in acc.items()}
                with open(path.with_suffix('.npz.tmp'),'wb') as f:np.savez_compressed(f,**arrays)
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                write(path,dict(status='complete',split=split,scene_token=scene,scene_name=old['scene_name'],
                    indices=by_scene[scene],frames=len(records),records=records,
                    checkpoint_sha256=protocol['checkpoint_sha256']))
                completed+=1;elapsed=time.perf_counter()-start
                write(RUN/'progress.json',dict(status='running',completed_scenes=completed,total_scenes=850,
                    split=split,last_scene=old['scene_name'],frames_this_process=processed,
                    seconds=elapsed,frames_per_second=processed/elapsed,updated_at=datetime.now().astimezone().isoformat()))
                if completed%20==0:print(f'{completed}/850 scenes; {processed} frames; {processed/elapsed:.2f} frames/s',flush=True)
                active=None;del acc,arrays
            del batch,out,result,values
        assert active is None
        cache.close();del loader,dataset,cache
    write(RUN/'execution_complete.json',dict(status='complete',frames_this_process=processed,
        scenes=completed,seconds=time.perf_counter()-start,checkpoint_sha256=protocol['checkpoint_sha256']))
    print('Full support-region census complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['pilot','run'],required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    pilot() if args.action=='pilot' else run(args.workers)
