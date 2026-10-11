#!/usr/bin/env python3
"""Streaming CPU attributes/statistics for the all-keyframe paired census."""
import argparse,copy,json,multiprocessing,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pyquaternion import Quaternion
from tools.sparseworld_full_regions_common import ROOT
from tools.sparseworld_fast_motion_common import MotionIndex
from tools.build_sparseworld_region_attributes import frustum_seen,speed
from tools.sparseworld_region_utils import DYNAMIC,STATIC,boundary_band,joint_counts,states,voxel_centers,sha256,write_json

MOTION=None
RUN_ROOT=None
PROTOCOL=None

def attributes(pair,rec):
    centers=MOTION.centers;all_speed=[];all_frustum=[];all_instance=[];all_meta=[]
    for j,target in enumerate(rec['targets']):
        labels=pair['GT'][j].reshape(-1)
        motion=np.full(len(centers),3,np.uint8);frustum=np.full(len(centers),4,np.uint8);ident=np.zeros(len(centers),np.uint16)
        instances=[]
        if target['available']:
            static=np.flatnonzero(np.isin(labels,STATIC))
            fs=frustum_seen(centers[static],pair['F_projection'][j],pair['F_wh'][j]);ps=frustum_seen(centers[static],pair['P_projection'][j],pair['P_wh'][j])
            frustum[static]=2*fs.astype(np.uint8)+ps.astype(np.uint8)
            dyn=np.flatnonzero(np.isin(labels,DYNAMIC));pose=np.asarray(target['pose'],np.float32)
            world=centers[dyn]@pose[:3,:3].T+pose[:3,3];matched=np.zeros(len(dyn),np.uint16)
            for a in MOTION.by_sample.get(target['token'],[]):
                rot=Quaternion(a['rotation']).rotation_matrix.astype(np.float32)
                local=(world-np.asarray(a['translation'],np.float32))@rot
                size=np.asarray(a['size'],np.float32)[[1,0,2]]
                inside=(np.abs(local)<=size/2+1e-5).all(-1)&(labels[dyn]==a['label'])
                if not inside.any():continue
                identifier=len(instances)+1
                matched[inside&(matched!=0)]=65535;matched[inside&(matched==0)]=identifier
                instances.append(dict(id=identifier,token=a['instance_token'],annotation_token=a['token'],label=a['label'],
                    speed=speed(a,MOTION.annotations,MOTION.times),translation=a['translation'],rotation=a['rotation'],size=a['size']))
            for inst in instances:
                chosen=np.flatnonzero(matched==inst['id']);ids=dyn[chosen]
                if not len(ids):continue
                v=inst['speed'];ident[ids]=inst['id'];inst['voxels']=len(ids)
                if np.isfinite(v):motion[ids]=0 if v<.5 else (1 if v<2 else 2)
                rot=Quaternion(inst['rotation']).rotation_matrix.astype(np.float32)
                local=(world[chosen]-np.asarray(inst['translation'],np.float32))@rot
                observed=[];supported=[]
                for prefix,key in [('F','forecast_frames'),('P','perception_frames')]:
                    seen=np.zeros(len(ids),bool);complete=True
                    for slot,frame in enumerate(target[key]):
                        prior=FRAME_INSTANCE.get((frame['token'],inst['token']))
                        if prior is None:complete=False;continue
                        r=Quaternion(prior['rotation']).rotation_matrix.astype(np.float32)
                        w=local@r.T+np.asarray(prior['translation'],np.float32)
                        ego=(w-pose[:3,3])@pose[:3,:3]
                        seen|=frustum_seen(ego,pair[prefix+'_projection'][j,6*slot:6*slot+6],pair[prefix+'_wh'][j,6*slot:6*slot+6])
                    observed.append(seen);supported.append(complete)
                inst['frustum_full_track']=all(supported)
                if all(supported):frustum[ids]=2*observed[0].astype(np.uint8)+observed[1].astype(np.uint8)
        all_speed.append(motion.reshape(200,200,16));all_frustum.append(frustum.reshape(200,200,16));all_instance.append(ident.reshape(200,200,16));all_meta.append(instances)
    return dict(speed=np.stack(all_speed),frustum=np.stack(all_frustum),instance=np.stack(all_instance)),all_meta


def process_one(rec):
    token=rec['token'];pair_path=RUN_ROOT/'pairs'/(token+'.npz')
    attr_path=RUN_ROOT/'attributes'/(token+'.npz');dest=RUN_ROOT/'sample_stats'/(token+'.npz')
    with np.load(pair_path) as pair:
        assert str(pair['protocol_sha256'])==PROTOCOL and str(pair['token'])==token
        attrs,meta=attributes(pair,rec)
        basic=np.zeros((3,3,3,2,4),np.int64);joint=np.zeros((3,3,18,18,18),np.int64)
        cov=np.zeros((3,18,4,2,2,4,5,4),np.int64);errors=np.zeros((3,7),np.int64)
        lidar=np.zeros((3,2,4),np.int64);broad=np.zeros((3,2,4),np.int64);xy=np.zeros((3,1600,4),np.int64)
        free_xy=np.zeros((3,1600,4),np.int64);sensitivity=np.zeros((3,3,5),np.int64)
        for j,t in enumerate(rec['targets']):
            if not t['available']:continue
            g,f,p=pair['GT'][j],pair['F'][j],pair['P'][j]
            valid=g<18;occupied=g<17;sem=states(g,f,p);occ=states(g,f,p,binary=True)
            band=boundary_band(g,.4);band08=boundary_band(g,.8)
            for m,mask in enumerate([valid,valid&pair['camera'][j],valid&pair['lidar'][j]]):
                joint[j,m]=joint_counts(g,f,p,mask)
                for d,domain in enumerate([mask,mask&occupied,mask&(g==17)]):
                    for k,s in enumerate([sem,occ]):basic[j,m,d,k]=np.bincount(s[domain],minlength=4)
            code=g.astype(np.int64)
            for val,radix in [(DISTANCE,4),(band,2),(pair['camera'][j],2),(attrs['speed'][j],4),(attrs['frustum'][j],5),(sem,4)]:code=code*radix+val
            cov[j]=np.bincount(code[valid],minlength=18*4*2*2*4*5*4).reshape(18,4,2,2,4,5,4)
            for mask,array in [(pair['lidar'][j],lidar),(band08,broad)]:array[j]=np.bincount(mask[occupied].astype(np.int64)*4+sem[occupied],minlength=8).reshape(2,4)
            bothwrong=valid&(sem==0);fo=f!=17;po=p!=17
            conditions=[occupied&~fo&~po,occupied&~fo&po,occupied&fo&~po,occupied&fo&po&(f==p),occupied&fo&po&(f!=p),(g==17)&(f==p),(g==17)&(f!=p)]
            errors[j]=[np.count_nonzero(bothwrong&c) for c in conditions];assert errors[j].sum()==basic[j,0,0,0,0]
            xy[j]=np.bincount(BLOCK[occupied.reshape(-1)]*4+sem[occupied],minlength=1600*4).reshape(1600,4)
            free=g==17;free_xy[j]=np.bincount(BLOCK[free.reshape(-1)]*4+sem[free],minlength=1600*4).reshape(1600,4)
            for k,size in enumerate((1,2,4)):
                block=CELL[size];n=int(80/size)**2
                c=np.bincount(block[occupied.reshape(-1)]*4+sem[occupied],minlength=n*4).reshape(n,4)
                supported=c.sum(1)>=20;a=c[:,1]-c[:,2]
                sensitivity[j,k]=[supported.sum(),np.count_nonzero(supported&(a<0)),np.count_nonzero(supported&(a>0)),np.count_nonzero(supported&(a==0)),np.count_nonzero(supported&(c[:,0]>.5*c.sum(1)))]
        old=Path('out/sparseworld_region_comparison/20261008_native512/attributes')/(token+'.npz')
        replay=False
        if old.exists():
            with np.load(old) as d:
                for key in attrs:assert np.array_equal(attrs[key],d[key]),f'Old attribute replay failed {token} {key}'
            replay=True
        tmp=attr_path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**attrs,pair_sha256=np.asarray(sha256(pair_path)));tmp.replace(attr_path)
        write_json(attr_path.with_suffix('.json'),dict(instances=meta,old_exact_replay=replay))
        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,basic=basic,joint=joint,covariates=cov,errors=errors,lidar=lidar,boundary08=broad,xy=xy,free_xy=free_xy,sensitivity=sensitivity,
            token=np.asarray(token),available=pair['available'],pair_sha256=np.asarray(sha256(pair_path)));tmp.replace(dest)
    return token


def aggregate(root,records):
    scenes=sorted({r['scene'] for r in records});sid={s:i for i,s in enumerate(scenes)};n=len(scenes)
    arrays={};samples=np.zeros((n,3),np.int64);counts=np.zeros(n,np.int64)
    for i,r in enumerate(records):
        s=sid[r['scene']]
        with np.load(root/'sample_stats'/(r['token']+'.npz')) as d:
            assert str(d['token'])==r['token']
            if not arrays:
                arrays={k:np.zeros((n,*d[k].shape),np.int64) for k in ('basic','joint','covariates','errors','lidar','boundary08','xy','free_xy','sensitivity')}
            for k in arrays:arrays[k][s]+=d[k]
            samples[s]+=d['available'];counts[s]+=1
        if i%1024==0:print(f'[aggregate] {i+1}/{len(records)}',flush=True)
    np.savez_compressed(root/'scene_counts.npz',**arrays,scene_names=scenes,
        split=np.asarray(['train' if s in TRAIN_SCENES else 'val' for s in scenes]),samples=samples,sources=counts)
    # Explanation code consumes scene covariates directly; no fake per-frame arrays.
    np.savez_compressed(root/'region_counts.npz',covariates=arrays['covariates'],scene_names=scenes)
    assert np.array_equal(arrays['covariates'].sum((2,3,4,5,6,7)),arrays['basic'][:,:,0,0,0])
    assert np.array_equal(arrays['joint'].sum((3,4,5)),arrays['basic'].sum(-1)[:,:, :,0,0])
    write_json(root/'collection_audit.json',dict(status='complete',sources=len(records),scenes=n,
        paired_targets=int(samples.sum()),scene_splits=['train','val'],attribute_old_exact_replays=sum((root/'attributes'/(t+'.npz')).exists() for t in OLD_TOKENS),
        four_state_conservation=True,unit='scene; all eligible source frames, no regional frame subsampling'))


def main():
    global MOTION,RUN_ROOT,PROTOCOL,FRAME_INSTANCE,DISTANCE,BLOCK,CELL,TRAIN_SCENES,OLD_TOKENS
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=6);p.add_argument('--pilot',action='store_true')
    args=p.parse_args();RUN_ROOT=args.root/('smoke' if args.pilot else '')
    while not (RUN_ROOT/'manifest.json').exists():time.sleep(1)
    manifest=json.loads((RUN_ROOT/'manifest.json').read_text());records=manifest['samples'];OLD_TOKENS=manifest['old_validation_tokens']
    PROTOCOL=sha256(RUN_ROOT/'protocol.json');TRAIN_SCENES={r['scene'] for r in records if r['split']=='train'}
    for name in ('attributes','sample_stats'):(RUN_ROOT/name).mkdir(exist_ok=True)
    MOTION=MotionIndex();FRAME_INSTANCE={(a['sample_token'],a['instance_token']):a for a in MOTION.annotations.values()}
    xyz=voxel_centers();DISTANCE=np.digitize(np.linalg.norm(xyz[...,:2],axis=-1),[10,20,30])
    BLOCK=np.broadcast_to((np.arange(200)//5)[:,None,None]*40+(np.arange(200)//5)[None,:,None],(200,200,16)).reshape(-1)
    CELL={size:(np.floor((xyz[...,0]+40)/size).astype(np.int64)*int(80/size)+np.floor((xyz[...,1]+40)/size).astype(np.int64)).reshape(-1) for size in (1,2,4)}
    pending={r['token']:r for r in records if not (RUN_ROOT/'sample_stats'/(r['token']+'.npz')).exists()};started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as executor:
        while pending:
            ready=[r for r in pending.values() if (RUN_ROOT/'pairs'/(r['token']+'.npz')).exists()][:256]
            if not ready:
                status=json.loads((RUN_ROOT/'status.json').read_text()) if (RUN_ROOT/'status.json').exists() else {}
                if status.get('status')=='complete':raise RuntimeError('Producer complete but some pairs are missing')
                time.sleep(2);continue
            for token in executor.map(process_one,ready,chunksize=4):pending.pop(token)
            done=len(records)-len(pending)
            write_json(RUN_ROOT/'statistics_status.json',dict(status='running',completed=done,total=len(records),elapsed_seconds=time.monotonic()-started))
            print(f'[full statistics] {done}/{len(records)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    aggregate(RUN_ROOT,records)
    write_json(RUN_ROOT/'statistics_status.json',dict(status='complete',sources=len(records),elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
