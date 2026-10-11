#!/usr/bin/env python3
"""GT-assisted fast-instance geometry oracles, with explicit missing/ambiguous cases."""
import argparse,json,multiprocessing,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pyquaternion import Quaternion
from tools.sparseworld_full_regions_common import ROOT
from tools.sparseworld_fast_motion_common import MotionIndex
from tools.sparseworld_motion_geometry import rigid_probe,extent_probe,decomposition,overlap,principal_yaw,corrected_labels_at_truth
from tools.sparseworld_region_utils import DYNAMIC,sha256,write_json

MOTION=None
FRAMES=None
RUN_ROOT=None


def box_distance(points,a):
    local=(points-np.asarray(a['translation']))@Quaternion(a['rotation']).rotation_matrix
    excess=np.maximum(np.abs(local)-np.asarray(a['size'])[[1,0,2]]/2,0.)
    return np.linalg.norm(excess,axis=1)


def border(a,pose):
    half=np.asarray(a['size'])[[1,0,2]]/2
    corners=np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
    world=corners@Quaternion(a['rotation']).rotation_matrix.T+np.asarray(a['translation'])
    inv=np.linalg.inv(np.asarray(pose));ego=world@inv[:3,:3].T+inv[:3,3]
    return bool(((ego<[-40,-40,-1])|(ego>[40,40,5.4])).any())


def assigned_clouds(pred,rec,target):
    ids=np.flatnonzero(np.isin(pred.reshape(-1),DYNAMIC));pose=np.asarray(target['pose'],np.float64)
    world=MOTION.centers[ids].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
    anns=MOTION.by_sample.get(target['token'],[]);h=target['horizon']
    frame_tokens=[rec['token']]+rec['frame_tokens'][5:5+h]
    distances=[]
    for a in anns:
        track=[FRAMES[(t,a['instance_token'])] for t in frame_tokens if (t,a['instance_token']) in FRAMES]
        distances.append(np.minimum.reduce([box_distance(world,x) for x in track]))
    matrix=np.asarray(distances)
    if not len(anns):return {},world,pred.reshape(-1)[ids]
    best=matrix.argmin(0);minimum=matrix.min(0)
    second=np.partition(matrix,1,axis=0)[1] if len(anns)>1 else np.full(len(ids),np.inf)
    result={}
    for i,a in enumerate(anns):
        result[a['instance_token']]={}
        for radius in (2.,4.):
            selected=(best==i)&(minimum<=radius)&(second-minimum>.4)
            ambiguous=(matrix[i]<=radius)&(matrix[i]<=minimum+.4)&(second-minimum<=.4)
            result[a['instance_token']][radius]=dict(indices=np.flatnonzero(selected),ambiguous=int(ambiguous.sum()))
    return result,world,pred.reshape(-1)[ids]


def current_body(rec,inst,pair,truth,target):
    a=FRAMES.get((rec['token'],inst['token']))
    if a is None:return dict(current_instance=False,current_gt_voxels=0,gt_body_iou=None,box_dimension_change=None,current_border=None)
    ids=np.flatnonzero(pair['current_GT'].reshape(-1)==inst['label']);cp=np.asarray(rec['current_pose'],np.float64)
    world=MOTION.centers[ids].astype(np.float64)@cp[:3,:3].T+cp[:3,3]
    local=(world-np.asarray(a['translation']))@Quaternion(a['rotation']).rotation_matrix
    inside=(np.abs(local)<=np.asarray(a['size'])[[1,0,2]]/2+1e-5).all(-1)
    # Exclude other same-class GT boxes; do not mix touching actors.
    for other in MOTION.by_sample.get(rec['token'],[]):
        if other['instance_token']!=inst['token'] and other['label']==inst['label']:
            inside&=box_distance(world,other)>1e-5
    local=local[inside]
    warped=local@Quaternion(inst['rotation']).rotation_matrix.T+np.asarray(inst['translation'])
    score=overlap(warped,truth,np.asarray(target['pose'])) if len(local) else None
    dimension=np.asarray(inst['size'])/np.asarray(a['size'])-1
    return dict(current_instance=True,current_gt_voxels=len(local),gt_body_iou=score['iou'] if score else None,
        box_dimension_change=float(np.max(np.abs(dimension))),current_border=border(a,cp))


def analyze_one(rec):
    path=RUN_ROOT/'pairs'/(rec['token']+'.npz');ap=RUN_ROOT/'attributes'/(rec['token']+'.npz')
    meta=json.loads(ap.with_suffix('.json').read_text())['instances'];rows=[]
    with np.load(path) as pair,np.load(ap) as attrs:
        assert str(attrs['pair_sha256'])==sha256(path)
        for j,target in enumerate(rec['targets']):
            if not target['available']:continue
            pose=np.asarray(target['pose'],np.float64)
            fast=[i for i in meta[j] if i['speed'] is not None and i['speed']>=2 and np.count_nonzero(attrs['instance'][j]==i['id'])]
            if not fast:continue
            assigned={m:assigned_clouds(pair[m][j],rec,target) for m in ('F','P')}
            for inst in fast:
                mask=attrs['instance'][j].reshape(-1)==inst['id'];ids=np.flatnonzero(mask)
                truth=MOTION.centers[ids].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
                native={m:pair[m][j].reshape(-1)[ids] for m in ('F','P')}
                a=MOTION.annotations[inst['annotation_token']]
                base=dict(split=rec['split'],scene=rec['scene'],source_token=rec['token'],target_token=target['token'],instance_token=inst['token'],
                    horizon_seconds=j+1,label=inst['label'],speed=inst['speed'],gt_voxels=len(ids),future_border=border(a,pose),
                    gt_orientation_anisotropy=principal_yaw(truth)[1] if len(ids)>2 else 0.)
                base.update(current_body(rec,inst,pair,truth,target))
                for m in ('F','P'):
                    base.update({m+'_native_correct':int((native[m]==inst['label']).sum()),m+'_native_free':int((native[m]==17).sum()),
                        m+'_native_wrong_nonfree':int(((native[m]!=17)&(native[m]!=inst['label'])).sum()),
                        m+'_native_wrong_static':int(((native[m]!=17)&~np.isin(native[m],DYNAMIC)).sum())})
                for radius in (2.,4.):
                    row=dict(base,radius_m=radius);clouds={};classes={};reasons=[]
                    for m in ('F','P'):
                        mapping,world,labels=assigned[m];selection=mapping[inst['token']][radius]
                        cloud=world[selection['indices']];cls=labels[selection['indices']];clouds[m]=cloud;classes[m]=cls
                        row[m+'_points']=len(cloud);row[m+'_ambiguous']=selection['ambiguous']
                        ambiguity=selection['ambiguous']/max(1,len(cloud)+selection['ambiguous']);row[m+'_ambiguity_fraction']=ambiguity
                        if len(cloud)<3:reasons.append(m+'_missing_or_sparse')
                        if ambiguity>.2:reasons.append(m+'_association_ambiguous')
                    if len(ids)<20:reasons.append('small_GT')
                    if base['future_border']:reasons.append('future_grid_truncated')
                    row['eligible']=not reasons;row['exclusion']=';'.join(reasons)
                    # Unmatched/small cases remain in denominator, unchanged by
                    # the counterfactual, rather than acquiring a "shape" label.
                    for m in ('F','P'):
                        for mode in ('translation','yaw','rigid','size','rigid_size'):
                            row[m+'_'+mode+'_correct']=base[m+'_native_correct']
                    if row['eligible']:
                        for m in ('F','P'):
                            r=rigid_probe(clouds[m],truth,pose)
                            s=extent_probe(clouds[m],truth,pose,Quaternion(inst['rotation']).rotation_matrix,r);d=decomposition(r,s)
                            for mode in ('base','translation','yaw','rigid'):
                                for metric in ('iou','recall','precision'):row[m+'_'+mode+'_'+metric]=r[mode][metric]
                            for mode in ('size','rigid_size'):
                                for metric in ('iou','recall','precision'):row[m+'_'+mode+'_'+metric]=s[mode][metric]
                            for key,value in d.items():row[m+'_'+key]=value
                            row[m+'_centroid_displacement']=r['centroid_displacement'];row[m+'_yaw_abs_degrees']=abs(r['yaw_radians'])*180/np.pi
                            row[m+'_orientation_anisotropy']=r['orientation_anisotropy']
                            for k,v in enumerate(r['transform']['R'].reshape(-1)):row[m+'_rotation_'+str(k)]=float(v)
                            for k,v in enumerate(r['transform']['t']):row[m+'_translation_'+str(k)]=float(v)
                            for mode,changed in {**r['_points'],**s['_points']}.items():
                                moved,conflicts=corrected_labels_at_truth(clouds[m],changed,classes[m],truth,pose,native[m])
                                row[m+'_'+mode+'_correct']=int((moved==inst['label']).sum());row[m+'_'+mode+'_class_collisions']=conflicts
                    rows.append(row)
    dest=RUN_ROOT/'motion_oracles'/(rec['token']+'.json')
    write_json(dest,dict(token=rec['token'],pair_sha256=sha256(path),rows=rows))
    return rec['token'],len(rows)


def main():
    global MOTION,FRAMES,RUN_ROOT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--pilot',action='store_true');p.add_argument('--precompute',action='store_true')
    args=p.parse_args();RUN_ROOT=args.root/('smoke' if args.pilot else '')
    if not args.pilot and not args.precompute:
        gate=json.loads((RUN_ROOT/'summary.json').read_text())['fast_gate'];assert gate['passed'],'Full-data fast advantage not confirmed'
    manifest=json.loads((RUN_ROOT/'manifest.json').read_text());records=manifest['samples']
    (RUN_ROOT/'motion_oracles').mkdir(exist_ok=True)
    policy=dict(version=1,radii_m=[2.,4.],gt_minimum=20,pred_minimum=3,ambiguity_distance_m=.4,ambiguity_fraction_maximum=.2,
        maximum_horizontal_translation_m=30,maximum_vertical_translation_m=2,orientation='global yaw; undirected modulo pi',
        candidate='dynamic occupied native voxels, nearest real GT box trajectory, all dynamic target actors included',
        scale='1--99% body-axis extents; clipped [.5,2]; identity retained',
        counterfactual='GT-mask voxel label relocation; original classes retained; class collisions become free',
        code_sha256={p:sha256(p) for p in ['tools/diagnose_sparseworld_full_motion.py','tools/sparseworld_motion_geometry.py']})
    pp=RUN_ROOT/'motion_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==policy
    else:write_json(pp,policy)
    MOTION=MotionIndex();FRAMES={(a['sample_token'],a['instance_token']):a for a in MOTION.annotations.values()}
    pending={r['token']:r for r in records if not (RUN_ROOT/'motion_oracles'/(r['token']+'.json')).exists()};started=time.monotonic();rows=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        while pending:
            ready=[r for r in pending.values() if (RUN_ROOT/'sample_stats'/(r['token']+'.npz')).exists()][:128]
            if not ready:
                status=json.loads((RUN_ROOT/'statistics_status.json').read_text()) if (RUN_ROOT/'statistics_status.json').exists() else {}
                if status.get('status')=='complete':raise RuntimeError('Statistics complete but expected source stats missing')
                time.sleep(2);continue
            for token,n in pool.map(analyze_one,ready,chunksize=2):pending.pop(token);rows+=n
            done=len(records)-len(pending)
            write_json(RUN_ROOT/'motion_status.json',dict(status='precomputing' if args.precompute else 'running',sources=done,total=len(records),new_rows=rows,elapsed_seconds=time.monotonic()-started))
            print(f'[motion oracles] {done}/{len(records)} rows={rows} elapsed={time.monotonic()-started:.1f}s',flush=True)
    import pandas as pd
    all_rows=[]
    for rec in records:
        d=json.loads((RUN_ROOT/'motion_oracles'/(rec['token']+'.json')).read_text());assert d['pair_sha256']==sha256(RUN_ROOT/'pairs'/(rec['token']+'.npz'))
        all_rows.extend(d['rows'])
    frame=pd.DataFrame(all_rows);frame.to_parquet(RUN_ROOT/'motion_instance_oracles.parquet',index=False)
    write_json(RUN_ROOT/'motion_status.json',dict(status='complete',sources=len(records),rows=len(frame),
        interpretation='Interpret only after full train/val fast_gate passes',elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
