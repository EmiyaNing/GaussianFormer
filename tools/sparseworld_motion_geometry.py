"""Oracle point-set diagnostics; no deployable model intervention or box head.

Rigid recovery is an achieved lower bound, never a claim of global registration
optimality. Unrecovered error includes missing surfaces and label/GT noise.
"""
import numpy as np
from scipy.spatial import cKDTree


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.asarray([[c,-s,0],[s,c,0],[0,0,1]],np.float64)


def voxel_keys(world,pose):
    # The ego pose is a float32 dataset matrix; use its actual inverse.
    inv=np.linalg.inv(np.asarray(pose,np.float64))
    ego=np.asarray(world,np.float64)@inv[:3,:3].T+inv[:3,3]
    ix=np.floor((ego-np.asarray([-40.,-40.,-1.]))/.4).astype(np.int64)
    # Keep out-of-grid false positives; injective keys over the diagnostic ROI.
    assert (np.abs(ix)<2048).all()
    return np.unique((ix[:,0]+2048)*8192**2+(ix[:,1]+2048)*8192+(ix[:,2]+2048))


def overlap(pred,truth,pose):
    a=voxel_keys(pred,pose);b=voxel_keys(truth,pose)
    inter=np.intersect1d(a,b,assume_unique=True).size
    union=len(a)+len(b)-inter
    return dict(iou=inter/union if union else 0.,recall=inter/len(b) if len(b) else 0.,
                precision=inter/len(a) if len(a) else 0.,predicted=len(a),truth=len(b),intersection=inter)


def sample(points,limit=256):
    return points[np.linspace(0,len(points)-1,min(len(points),limit),dtype=int)]


def principal_yaw(points):
    xy=points[:,:2]-points[:,:2].mean(0)
    values,vectors=np.linalg.eigh(xy.T@xy/max(1,len(points)))
    v=vectors[:,-1]
    return np.arctan2(v[1],v[0]),float(values[-1]/max(values[0],1e-8))


def rigid_probe(pred,truth,pose,max_shift=30.,iterations=8):
    """Translation, centroid-yaw, and multi-start symmetric ICP in global XYZ.

    Yaw acts about the prediction centroid. Horizontal rigid transforms plus
    vertical translation preserve object shape. Identity is always retained.
    Every retained candidate is selected by native-grid IoU, not Chamfer alone.
    """
    pred=np.asarray(pred,np.float64);truth=np.asarray(truth,np.float64)
    identity=dict(R=np.eye(3),t=np.zeros(3));base=overlap(pred,truth,pose)
    if len(pred)<3 or len(truth)<3:
        return dict(base=base,translation=base,yaw=base,rigid=base,transform=identity,
                    centroid_displacement=0.,yaw_radians=0.,orientation_anisotropy=0.,
                    _points={k:pred for k in ('translation','yaw','rigid')})
    source=sample(pred);target=sample(truth);tree=cKDTree(target)
    pc=pred.mean(0);tc=truth.mean(0);pa,anisotropy=principal_yaw(source);ta,_=principal_yaw(target)
    angles=[0.,-np.pi/2,-np.pi/4,np.pi/4,np.pi/2,np.pi,ta-pa,ta-pa+np.pi]
    scores=dict(translation=base,yaw=base,rigid=base);best=identity
    transforms={k:dict(R=np.eye(3),t=np.zeros(3)) for k in scores}
    def consider(R,t,kind):
        nonlocal best
        moved=pred@R.T+t
        if np.linalg.norm((moved.mean(0)-pc)[:2])>max_shift+1e-6:return
        if abs(moved.mean(0)[2]-pc[2])>2.:return
        score=overlap(moved,truth,pose)
        if score['iou']>scores[kind]['iou']+1e-12:
            scores[kind]=score;transforms[kind]=dict(R=R.copy(),t=t.copy())
        if score['iou']>scores['rigid']['iou']+1e-12:
            scores['rigid']=score;best=dict(R=R.copy(),t=t.copy())
        # A rigid candidate can coincide with the kind just updated.
        if kind=='rigid' and score['iou']>=scores['rigid']['iou'] and score['iou']>base['iou']:
            best=dict(R=R.copy(),t=t.copy())
    # Translation-only ICP, including centroid initialization.
    t=tc-pc;R=np.eye(3)
    for _ in range(iterations+1):
        consider(R,t,'translation')
        moved=source+t;distance,indices=tree.query(moved)
        keep=distance<=np.quantile(distance,.8)+1e-9
        t=(target[indices[keep]]-source[keep]).mean(0)
    for angle in angles:
        R=rotation(angle);t=pc-pc@R.T
        consider(R,t,'yaw')
        t=tc-pc@R.T
        for _ in range(iterations+1):
            consider(R,t,'rigid')
            moved=source@R.T+t
            d,ids=tree.query(moved);reverse=cKDTree(moved).query(target)[1]
            keep=d<=np.quantile(d,.8)+1e-9
            x=np.concatenate([source[keep],source[reverse]])
            y=np.concatenate([target[ids[keep]],target])
            xc=x.mean(0);yc=y.mean(0)
            u,_,vt=np.linalg.svd((x[:,:2]-xc[:2]).T@(y[:,:2]-yc[:2]))
            q=vt.T@u.T
            if np.linalg.det(q)<0:vt[-1]*=-1;q=vt.T@u.T
            R=np.eye(3);R[:2,:2]=q;t=yc-xc@R.T
    moved=pred@best['R'].T+best['t']
    transforms['rigid']=best
    # Report undirected orientation; point-set occupancy cannot identify front/back.
    angle=np.arctan2(best['R'][1,0],best['R'][0,0]);angle=(angle+np.pi/2)%np.pi-np.pi/2
    return dict(base=base,**scores,transform=best,centroid_displacement=float(np.linalg.norm((moved.mean(0)-pc)[:2])),
                vertical_displacement=float(moved.mean(0)[2]-pc[2]),yaw_radians=float(angle),orientation_anisotropy=anisotropy,
                _points={k:pred@v['R'].T+v['t'] for k,v in transforms.items()})


def extent_probe(pred,truth,pose,body_rotation,rigid):
    """Size-only oracle at the original pose; compare with rigid+size.

    The scale is fixed from GT 1--99% body-axis extents, clipped to [.5,2].
    This diagnoses only scale/extent mismatch, not arbitrary surface deformation.
    """
    if len(pred)<3:return dict(size=rigid['base'],rigid_size=rigid['rigid'],scale=[1.,1.,1.],_points=dict(size=pred,rigid_size=pred))
    R,t=rigid['transform']['R'],rigid['transform']['t'];aligned=pred@R.T+t
    center=aligned.mean(0);body=np.asarray(body_rotation,np.float64)
    a=(aligned-center)@body;b=(truth-truth.mean(0))@body
    aw=np.maximum(np.diff(np.quantile(a,[.01,.99],axis=0),axis=0)[0],.4)
    bw=np.maximum(np.diff(np.quantile(b,[.01,.99],axis=0),axis=0)[0],.4)
    scale=np.clip(bw/aw,.5,2.)
    size=rigid['base'];both=rigid['rigid'];best_size=pred;best_both=aligned
    for axes in ([0],[1],[2],[0,1],[0,1,2]):
        s=np.ones(3);s[axes]=scale[axes]
        changed=(a*s)@body.T+center
        original_pose=(changed-t)@R
        score=overlap(original_pose,truth,pose)
        if score['iou']>size['iou']:size=score;best_size=original_pose
        score=overlap(changed,truth,pose)
        if score['iou']>both['iou']:both=score;best_both=changed
    # Joint oracle includes each individual intervention, preventing negative
    # contributions introduced only by a worse combination implementation.
    if size['iou']>both['iou']:both=size;best_both=best_size
    return dict(size=size,rigid_size=both,scale=scale.tolist(),_points=dict(size=best_size,rigid_size=best_both))


def decomposition(rigid,extent):
    b=rigid['base']['iou'];t=rigid['translation']['iou'];r=rigid['yaw']['iou'];p=rigid['rigid']['iou']
    s=extent['size']['iou'];ps=extent['rigid_size']['iou']
    return dict(position=.5*((t-b)+(p-r)),orientation=.5*((r-b)+(p-t)),
        pose=.5*((p-b)+(ps-s)),extent=.5*((s-b)+(ps-p)),
        residual=1-ps,raw_error=1-b,pose_only_gain=p-b)


def corrected_labels_at_truth(original_pred,changed_pred,labels,truth,pose,original_truth_labels):
    """Conservative voxel-label relocation on a fixed GT instance mask.

    Keep unassigned original labels; remove the moved cloud; place its unchanged
    semantic labels at transformed voxels. Conflicting classes become free.
    This is an offline voxel oracle, not re-running the network rasterizer.
    """
    def keys_with_inverse(points):
        inv=np.linalg.inv(np.asarray(pose,np.float64));ego=points@inv[:3,:3].T+inv[:3,3]
        ix=np.floor((ego-np.asarray([-40.,-40.,-1.]))/.4).astype(np.int64)
        return (ix[:,0]+2048)*8192**2+(ix[:,1]+2048)*8192+(ix[:,2]+2048)
    truth_keys=keys_with_inverse(truth)
    old=keys_with_inverse(original_pred);new=keys_with_inverse(changed_pred)
    output=np.asarray(original_truth_labels).copy()
    output[np.isin(truth_keys,old)]=17
    unique,inverse=np.unique(new,return_inverse=True)
    bits=np.zeros(len(unique),np.uint32);np.bitwise_or.at(bits,inverse,np.left_shift(np.uint32(1),np.asarray(labels,np.uint32)))
    single=(bits!=0)&((bits&(bits-1))==0)
    rendered=np.full(len(unique),17,np.uint8);rendered[single]=np.log2(bits[single]).astype(np.uint8)
    positions=np.searchsorted(unique,truth_keys);valid=positions<len(unique)
    valid[valid]&=unique[positions[valid]]==truth_keys[valid]
    output[valid]=rendered[positions[valid]]
    return output,int((~single).sum())
