"""Permutation-invariant raw-query rigidity diagnostics in full SE(3)."""
import numpy as np
from scipy.spatial import cKDTree,distance
from tools.sparseworld_motion_geometry import rotation,sample


def chamfer_squared(a,b):
    if not len(a) or not len(b):return np.nan
    return .5*(np.mean(cKDTree(b).query(a)[0]**2)+np.mean(cKDTree(a).query(b)[0]**2))


def fit_rigid(a,b,limit=256,iterations=8):
    """Fit actual b to GT-rigid-warped historical template a, no point ID claim.

    Compare point SETS, not the 48 channel indices. Identity is retained and the
    achieved decrease in symmetric squared Chamfer partitions raw discrepancy.
    Full 3D rotation prevents pitch/roll error being labeled nonrigid deformation.
    """
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);x=sample(b,limit);y=sample(a,limit)
    tree=cKDTree(y);xc=x.mean(0);yc=y.mean(0)
    raw=chamfer_squared(x,y);best=raw;bestR=np.eye(3);bestt=np.zeros(3)
    xa=b-b.mean(0);ya=a-a.mean(0)
    _,vx=np.linalg.eigh(xa.T@xa);_,vy=np.linalg.eigh(ya.T@ya)
    initial=[rotation(t) for t in (0.,np.pi/2,-np.pi/2,np.pi)]
    for signs in ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)):
        R=vy@np.diag(signs)@vx.T
        if np.linalg.det(R)<0:R[:,-1]*=-1
        initial.append(R)
    fullraw=chamfer_squared(b,a);fullbest=fullraw;fullR=np.eye(3);fullt=np.zeros(3)
    def consider_full(R,t):
        nonlocal fullbest,fullR,fullt
        value=chamfer_squared(b@R.T+t,a)
        if value<fullbest:fullbest=value;fullR=R.copy();fullt=t.copy()
    for R in initial:
        consider_full(R,a.mean(0)-b.mean(0)@R.T)
        t=yc-xc@R.T
        for _ in range(iterations+1):
            moved=x@R.T+t;value=chamfer_squared(moved,y)
            if value<best:best=value;bestR=R.copy();bestt=t.copy()
            d,ix=tree.query(moved);reverse=cKDTree(moved).query(y)[1]
            keep=d<=np.quantile(d,.8)+1e-9
            p=np.concatenate([x[keep],x[reverse]]);q=np.concatenate([y[ix[keep]],y])
            pc=p.mean(0);qc=q.mean(0);u,_,vt=np.linalg.svd((p-pc).T@(q-qc))
            R=vt.T@u.T
            if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
            t=qc-pc@R.T
        consider_full(R,t)
    # Evaluate the chosen transform and identity on the same full point sets.
    consider_full(bestR,bestt);bestR=fullR;bestt=fullt;aligned=b@bestR.T+bestt
    da=np.sort(distance.pdist(a));db=np.sort(distance.pdist(b))
    deformation=float(np.mean(np.abs(da-db))) if len(da)==len(db) else np.nan
    angle=np.arctan2(bestR[1,0],bestR[0,0]);angle=(angle+np.pi/2)%np.pi-np.pi/2
    return dict(raw_squared=fullraw,pose_removable_squared=fullraw-fullbest,nonrigid_residual_squared=fullbest,
        centroid_error_xy=float(np.linalg.norm((b.mean(0)-a.mean(0))[:2])),centroid_error_z=float(abs(b.mean(0)[2]-a.mean(0)[2])),
        yaw_correction_degrees=float(abs(angle)*180/np.pi),pair_distance_change_m=deformation,
        R=bestR,t=bestt,aligned=aligned)


def native_point_validity(points,labels,max_logits):
    p=np.asarray(points,np.float32);labels=np.asarray(labels)
    center=p.mean(1,keepdims=True);geom=np.linalg.norm(p-center,axis=-1)<3
    index=np.floor((p-np.asarray([-40,-40,-1],np.float32))/np.float32(.4)).astype(np.int64)
    inside=((index>=0)&(index<[200,200,16])).all(-1)
    score=1/(1+np.exp(-np.clip(max_logits,-80,80)))
    normalized=(p-center)/np.asarray([80,80,6.4],np.float32);dist=np.linalg.norm(normalized,axis=-1)
    compact=(labels==15)|(labels==16);factor=np.ones_like(score)
    factor[compact]=np.minimum(1.,.1/np.maximum(dist[compact],1e-12));score=score*factor
    threshold=np.asarray([.35]*15+[.25,.3],np.float32)[labels]
    confident=score>threshold
    return dict(geometry=geom,inside=inside,confident=confident,kept=geom&inside&confident)
