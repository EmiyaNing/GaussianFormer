#!/usr/bin/env python3
"""Historical-query identity control: raw pose drift, nonrigidity, native erasure."""
import argparse,json,multiprocessing,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pyquaternion import Quaternion
from tools.sparseworld_full_regions_common import ROOT
from tools.sparseworld_fast_motion_common import MotionIndex
from tools.sparseworld_query_geometry import fit_rigid,native_point_validity
from tools.sparseworld_motion_geometry import overlap
from tools.sparseworld_region_utils import sha256,write_json

MOTION=None
RUN_ROOT=None


def bindings(rec,pair,raw):
    p=raw['source_points'];v=native_point_validity(p,raw['source_labels'],raw['source_max_logits'])
    indices=np.floor((p-np.asarray([-40,-40,-1],np.float32))/np.float32(.4)).astype(np.int64)
    inside=v['inside'];flat=np.zeros(p.shape[:2],np.int64)
    flat[inside]=indices[inside,0]*3200+indices[inside,1]*16+indices[inside,2]
    gtlabels=np.full(p.shape[:2],255,np.uint8);gtlabels[inside]=pair['current_GT'].reshape(-1)[flat[inside]]
    pose=np.asarray(rec['current_pose'],np.float64);world=p.reshape(-1,3).astype(np.float64)@pose[:3,:3].T+pose[:3,3]
    assignment=np.zeros(len(world),np.int32);anns=MOTION.by_sample.get(rec['token'],[])
    for i,a in enumerate(anns,1):
        local=(world-np.asarray(a['translation']))@Quaternion(a['rotation']).rotation_matrix
        mask=(np.abs(local)<=np.asarray(a['size'])[[1,0,2]]/2+1e-5).all(-1)
        mask&=(gtlabels.reshape(-1)==a['label'])&v['kept'].reshape(-1)
        assignment[mask&(assignment!=0)]=-1;assignment[mask&(assignment==0)]=i
    assignment=assignment.reshape(720,48);counts=np.asarray([(assignment==i).sum(1) for i in range(1,len(anns)+1)])
    result={}
    if not len(anns):return result
    best=counts.argmax(0);total=counts.sum(0);accepted=v['kept'].sum(1)
    for i,a in enumerate(anns):
        candidate=(best==i)&(counts[i]>=3)&(counts[i]>=.6*total)&(counts[i]>=.3*np.maximum(accepted,1))
        query=[]
        for q in np.flatnonzero(candidate):
            # Several channels at one voxel do not establish a reliable object.
            if len(np.unique(flat[q,assignment[q]==i+1]))>=3:query.append(int(q))
        if query:result[a['instance_token']]=dict(queries=query,annotation=a)
    return result


def process_one(rec):
    pairpath=RUN_ROOT/'pairs'/(rec['token']+'.npz');rawpath=RUN_ROOT/'query_lineage'/(rec['token']+'.npz')
    ap=RUN_ROOT/'attributes'/(rec['token']+'.npz');meta=json.loads(ap.with_suffix('.json').read_text())['instances'];rows=[]
    with np.load(pairpath) as pair,np.load(rawpath) as raw,np.load(ap) as attrs:
        assert str(raw['token'])==rec['token'] and np.array_equal(raw['available'],pair['available'])
        bound=bindings(rec,pair,raw);cp=np.asarray(rec['current_pose'],np.float64)
        for j,target in enumerate(rec['targets']):
            if not target['available']:continue
            pose=np.asarray(target['pose'],np.float64)
            for inst in meta[j]:
                if inst['speed'] is None or inst['speed']<2:continue
                ids=np.flatnonzero(attrs['instance'][j].reshape(-1)==inst['id'])
                if not len(ids):continue
                row=dict(scene=rec['scene'],split=rec['split'],source_token=rec['token'],target_token=target['token'],instance_token=inst['token'],
                    horizon_seconds=j+1,label=inst['label'],speed=inst['speed'],gt_voxels=len(ids),eligible=False,parent_queries=0)
                seed=bound.get(inst['token'])
                if seed is None:row['exclusion']='no_reliable_historical_query';rows.append(row);continue
                q=seed['queries'];a=seed['annotation'];row['parent_queries']=len(q)
                current=raw['source_points'][q].reshape(-1,3).astype(np.float64)@cp[:3,:3].T+cp[:3,3]
                local=(current-np.asarray(a['translation']))@Quaternion(a['rotation']).rotation_matrix
                expected=local@Quaternion(inst['rotation']).rotation_matrix.T+np.asarray(inst['translation'])
                future=raw['future_points_'+str(j)][q];labels=raw['future_labels_'+str(j)][q]
                logits=raw['future_max_logits_'+str(j)][q]
                actual=future.reshape(-1,3).astype(np.float64)@pose[:3,:3].T+pose[:3,3]
                fit=fit_rigid(expected,actual);row['eligible']=True;row['exclusion']=''
                for k in ('raw_squared','pose_removable_squared','nonrigid_residual_squared','centroid_error_xy','centroid_error_z','yaw_correction_degrees','pair_distance_change_m'):row[k]=float(fit[k])
                truth=MOTION.centers[ids].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
                for name,points in [('template',expected),('actual',actual),('aligned_actual',fit['aligned'])]:
                    # Raw unthresholded geometry, separate from native occupancy.
                    if np.max(np.abs(points@np.linalg.inv(pose)[:3,:3].T+np.linalg.inv(pose)[:3,3]))<700:
                        row[name+'_gt_iou']=overlap(points,truth,pose)['iou']
                    else:row[name+'_gt_iou']=None
                v=native_point_validity(future,labels,logits);n=labels.size
                categories={'drop_geometry':~v['geometry'],'drop_range':v['geometry']&~v['inside'],
                    'drop_confidence':v['geometry']&v['inside']&~v['confident'],
                    'kept_wrong_class':v['kept']&(labels!=inst['label']),'kept_object_class':v['kept']&(labels==inst['label'])}
                for name,mask in categories.items():row[name+'_fraction']=float(mask.sum()/n)
                assert abs(sum(row[k+'_fraction'] for k in categories)-1)<1e-9
                row['source_point_count']=n;row['box_dimension_change']=float(np.max(np.abs(np.asarray(inst['size'])/np.asarray(a['size'])-1)))
                rows.append(row)
    write_json(RUN_ROOT/'query_diagnostics'/(rec['token']+'.json'),dict(token=rec['token'],raw_sha256=sha256(rawpath),rows=rows))
    return rec['token'],len(rows)


def main():
    global MOTION,RUN_ROOT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();RUN_ROOT=args.root/('smoke' if args.pilot else '')
    records=json.loads((RUN_ROOT/'manifest.json').read_text())['samples'];(RUN_ROOT/'query_diagnostics').mkdir(exist_ok=True)
    policy=dict(version=1,parent_queries=720,source_binding='>=3 distinct GT actor voxels; >=.6 of matched actor support; >=.3 of native-kept query points',
        comparison='Same query identity; point sets, not channel correspondence; true GT rigid warp of historical decoded geometry',
        pose='full SE3 multistart symmetric ICP, identity included',shape='squared symmetric Chamfer after achieved rigid correction; full point-set pair-distance distribution invariant',
        limitation='Historical observed-query subset; initial shape error and future-born slots separate; changed raw geometry need not be harmful unless future GT alignment worsens',
        code_sha256={p:sha256(p) for p in ['tools/diagnose_sparseworld_query_lineage.py','tools/sparseworld_query_geometry.py']})
    pp=RUN_ROOT/'query_diagnostic_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==policy
    else:write_json(pp,policy)
    MOTION=MotionIndex();pending={r['token']:r for r in records if not (RUN_ROOT/'query_diagnostics'/(r['token']+'.json')).exists()};started=time.monotonic();nrows=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        while pending:
            ready=[r for r in pending.values() if (RUN_ROOT/'query_lineage'/(r['token']+'.npz')).exists() and (RUN_ROOT/'sample_stats'/(r['token']+'.npz')).exists()][:128]
            if not ready:time.sleep(2);continue
            for token,n in pool.map(process_one,ready,chunksize=2):pending.pop(token);nrows+=n
            done=len(records)-len(pending);write_json(RUN_ROOT/'query_status.json',dict(status='precomputing',sources=done,total=len(records),new_rows=nrows,elapsed_seconds=time.monotonic()-started))
            print(f'[query diagnostics] {done}/{len(records)} rows={nrows} elapsed={time.monotonic()-started:.1f}s',flush=True)
    import pandas as pd
    rows=[]
    for rec in records:rows.extend(json.loads((RUN_ROOT/'query_diagnostics'/(rec['token']+'.json')).read_text())['rows'])
    pd.DataFrame(rows).to_parquet(RUN_ROOT/'query_instance_diagnostics.parquet',index=False)
    write_json(RUN_ROOT/'query_status.json',dict(status='complete',sources=len(records),rows=len(rows),elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
