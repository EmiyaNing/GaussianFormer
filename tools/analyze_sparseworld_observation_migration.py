#!/usr/bin/env python3
"""A2: both ego distances and input-time annotation-depth observation proxies."""
import json,pickle,sys,time
from pathlib import Path
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import minimum_filter
from dataset.sparseworld_trajectory import _pose
from tools.sparseworld_region_utils import ROOT,STATIC,states,write_json
from tools.sparseworld_region_ablation_common import OUT,CENTERS,records

@lru_cache(maxsize=32)
def occupied_points(path):
    options=[Path(path),Path('data/occ3d')/path,Path('data/occ3d/gts')/path]
    path=next(p for p in options if p.exists())
    if path.is_dir():path=path/'labels.npz'
    with np.load(path) as f:g=f['semantics']
    return CENTERS[g.reshape(-1)<17]

def observations(points,projection,wh,frames,target_pose,frame_infos):
    count=np.zeros(len(points),np.uint8);support=count.copy();blocked=count.copy();unknown=count.copy();scale=np.zeros(len(points),np.float32)
    for slot,frame in enumerate(frames):
        info=frame_infos[frame['token']]
        pts=occupied_points(info['path'])
        transform=np.linalg.inv(target_pose)@info['pose']
        targetpoints=pts@transform[:3,:3].T+transform[:3,3]
        for cam in range(6):
            k=slot*6+cam;mat=projection[k];w,h=wh[k].astype(int);hh=(h+3)//4;ww=(w+3)//4
            q=points@mat[:3,:3].T+mat[:3,3]
            uv=q[:,:2]/np.maximum(q[:,2:3],1e-6)
            seen=(q[:,2]>.01)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            ids=np.flatnonzero(seen);count[ids]+=1
            # Norm removes camera extrinsic rotation; geometric voxel image size proxy.
            axis=mat[2,:3]
            fx=np.linalg.norm(mat[0,:3]-np.dot(mat[0,:3],axis)/np.dot(axis,axis)*axis)
            fy=np.linalg.norm(mat[1,:3]-np.dot(mat[1,:3],axis)/np.dot(axis,axis)*axis)
            scale[ids]=np.maximum(scale[ids],.4*np.sqrt(fx*fy)/q[ids,2])
            z=targetpoints@mat[:3,:3].T+mat[:3,3]
            pix=z[:,:2]/np.maximum(z[:,2:3],1e-6)
            ok=(z[:,2]>.01)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
            bins=np.floor(pix[ok]/4).astype(int);depth=np.full((hh,ww),np.inf,np.float32)
            np.minimum.at(depth,(bins[:,1],bins[:,0]),z[ok,2])
            depth=minimum_filter(depth,size=3,mode='constant',cval=np.inf)
            loc=np.floor(uv[ids]/4).astype(int);surface=depth[loc[:,1],loc[:,0]];d=q[ids,2]
            consistent=np.isfinite(surface)&(np.abs(d-surface)<=.8)
            behind=np.isfinite(surface)&(d>surface+.8)
            support[ids]+=consistent;blocked[ids]+=behind;unknown[ids]+=~(consistent|behind)
    return count,support,blocked,unknown,scale

def main():
    recs=json.loads((ROOT/'manifest.json').read_text())['samples'];chosen={r['token'] for r in records()}
    raw=pickle.load(open('data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl','rb'))['infos']
    needed={f['token'] for r in recs if r['token'] in chosen for t in r['targets'] for f in t['forecast_frames']+t['perception_frames']}
    frame_infos={f['token']:dict(path=f['occ_path'],pose=_pose(f)) for frames in raw.values() for f in frames if f.get('token') in needed}
    assert needed <= frame_infos.keys()
    del raw
    dest=OUT/'observation';dest.mkdir(exist_ok=True)
    for i,r in enumerate(recs):
        path=dest/(r['token']+'.npz')
        if path.exists():
            with np.load(path) as old:
                if 'version' in old and int(old['version'])==2:continue
        start=time.monotonic();pair=dict(np.load(ROOT/'pairs'/(r['token']+'.npz')));attrs=dict(np.load(ROOT/'attributes'/(r['token']+'.npz')))
        distance=np.zeros((3,2,4,4,4),np.int64);frustum=np.zeros((3,5,4,4,4),np.int64)
        distance_class=np.zeros((3,17,4,4,4),np.int64)
        quality=np.zeros((3,2,5,4,17,4,4),np.int64)
        # quality axes: horizon,F/P,attribute,count-bin,class,target-distance,state.
        cross=np.zeros((3,4,4,4),np.int64)
        for j,t in enumerate(r['targets']):
            gt=pair['GT'][j].reshape(-1);s=states(gt,pair['F'][j].reshape(-1),pair['P'][j].reshape(-1)).astype(int)
            target=np.asarray(t['pose']);tr=np.linalg.inv(np.asarray(r['current_pose']))@target
            current=CENTERS@tr[:3,:3].T+tr[:3,3]
            d0=np.digitize(np.linalg.norm(current[:,:2],axis=1),[10,20,30]);dh=np.digitize(np.linalg.norm(CENTERS[:,:2],axis=1),[10,20,30])
            static=np.isin(gt,STATIC)
            for k,sel in enumerate([gt<17,static]):
                distance[j,k]=np.bincount((d0[sel]*4+dh[sel])*4+s[sel],minlength=64).reshape(4,4,4)
            distance_class[j]=np.bincount(((gt[static].astype(int)*4+d0[static])*4+dh[static])*4+s[static],minlength=17*64).reshape(17,4,4,4)
            fr=attrs['frustum'][j].reshape(-1).astype(int)
            frustum[j]=np.bincount(((fr[static]*4+d0[static])*4+dh[static])*4+s[static],minlength=320).reshape(5,4,4,4)
            if r['token'] not in chosen:continue
            ix=np.flatnonzero(static);seen_counts=[]
            for a,prefix in enumerate(['F','P']):
                obs=observations(CENTERS[ix],pair[prefix+'_projection'][j],pair[prefix+'_wh'][j],t['forecast_frames' if a==0 else 'perception_frames'],target,frame_infos)
                seen_counts.append(obs[0]);bins=[np.digitize(v,[1,2,6]) for v in obs[:4]]+[np.digitize(obs[4],[1,2,4])]
                for k,b in enumerate(bins):
                    code=((b*17+gt[ix].astype(int))*4+dh[ix])*4+s[ix]
                    quality[j,a,k]=np.bincount(code,minlength=4*17*4*4).reshape(4,17,4,4)
            f,p=seen_counts
            cross[j]=np.bincount((np.digitize(f,[1,2,6])*4+np.digitize(p,[1,2,6]))*4+s[ix],minlength=64).reshape(4,4,4)
        np.savez_compressed(path,version=np.array(2),distance=distance,distance_class=distance_class,frustum=frustum,quality=quality,cross=cross)
        if i%8==0:print(f'[A2] {i+1}/{len(recs)} quality={r["token"] in chosen} {time.monotonic()-start:.1f}s',flush=True)
    write_json(OUT/'observation_protocol.json',dict(samples=512,quality_samples=len(chosen),distance_bins=[0,10,20,30],
        attributes=['frustum_view_count','depth_consistent_view_count','depth_occluded_view_count','depth_unknown_view_count','max_voxel_pixel_scale'],
        view_bins=['0','1','2-5','>=6'],scale_bins=['<1','1-2','2-4','>=4'],depth_pixel_stride=4,depth_min_filter=3,depth_tolerance_m=.8,
        geometry='Input-frame GT occupancy depth buffer; static target correspondences only; annotation proxy, not measured camera visibility'))
if __name__=='__main__':main()
