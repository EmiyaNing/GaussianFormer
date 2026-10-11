"""Shared contracts for the four regional mechanism ablations (no model edits)."""
import json
from pathlib import Path
import numpy as np
import torch
from tools.sparseworld_region_utils import ROOT, STATIC, voxel_centers
from tools.experiment_future2current import _frame

OUT = Path('out/sparseworld_region_ablations/20261008')
REPORT = Path('analysis/hypotheses/prediction_perception_regions/ablations')
CENTERS = voxel_centers().reshape(-1, 3)
GROUPS = ['occupied', 'static', 'static_future_only', 'current_near10',
          'target_near10', 'current_near_target_far20', 'rear_edge', 'fast', 'both_wrong']

def windows(h):
    history = list(range(0, -5, -1))
    future = list(range(h, h-5, -1))
    union = sorted(set(history + future), reverse=True)
    mixed = [union[i] for i in np.rint(np.linspace(0, len(union)-1, 5)).astype(int)]
    return dict(H=history, P=future, M=mixed)

def window(data, imgs, relative, h):
    indices = [6*_frame(r)+c for r in relative for c in range(6)]
    metas = dict(data)
    projection = data['projection_mat'][:, indices].clone().float()
    transforms = data['future_ego_to_current'].float()
    dest = transforms[:, h-1] if h else torch.eye(4, device=imgs.device)[None]
    for slot, r in enumerate(relative):
        source = torch.linalg.inv(transforms[:, r-1]) @ dest if r > 0 else dest
        projection[:, slot*6:(slot+1)*6] = projection[:, slot*6:(slot+1)*6] @ source[:, None]
    metas.update(projection_mat=projection, image_wh=data['image_wh'][:, indices])
    return imgs[:, indices].contiguous(), metas

def masks(rec, j, pair, attrs):
    gt = pair['GT'][j].reshape(-1)
    target = np.asarray(rec['targets'][j]['pose'])
    current = np.asarray(rec['current_pose'])
    transform = np.linalg.inv(current) @ target
    coords = CENTERS @ transform[:3,:3].T + transform[:3,3]
    d0 = np.linalg.norm(coords[:,:2], axis=1)
    dh = np.linalg.norm(CENTERS[:,:2], axis=1)
    occupied = gt < 17
    static = np.isin(gt, STATIC)
    return [occupied, static, static & (attrs['frustum'][j].reshape(-1)==1),
            occupied & (d0<10), occupied & (dh<10), occupied & (d0<10) & (dh>=20),
            occupied & (CENTERS[:,0]<-30), occupied & (attrs['speed'][j].reshape(-1)==2),
            occupied & (pair['F'][j].reshape(-1)!=gt) & (pair['P'][j].reshape(-1)!=gt)]

def evaluate(gt, pred, groups):
    gt, pred = gt.reshape(-1).astype(np.int64), np.asarray(pred).reshape(-1).astype(np.int64)
    valid = gt < 18
    hist = np.bincount(gt[valid]*18+pred[valid], minlength=324).reshape(18,18)
    count = np.array([[g.sum(), ((pred==gt)&g).sum(), ((pred<17)&g).sum()] for g in groups], dtype=np.int64)
    return hist, count

def records():
    all_records = json.loads((ROOT/'manifest.json').read_text())['samples']
    # Exactly one sample per scene, deterministic identity-only selection, no score selection.
    chosen={}
    import hashlib
    for r in all_records:
        score=hashlib.sha256(('ablation20261008:'+r['token']).encode()).hexdigest()
        if r['scene'] not in chosen or score<chosen[r['scene']][0]: chosen[r['scene']]=(score,r)
    return sorted([v[1] for v in chosen.values()],key=lambda r:r['position'])

def patch_mask(points, projection, wh, height, width, radius=12):
    """Union of fixed pixel squares; geometric ROI mask, not a visibility oracle."""
    from scipy.ndimage import maximum_filter
    result=np.zeros((len(projection),height,width),bool)
    for k,(mat,size) in enumerate(zip(projection,wh)):
        q=points@mat[:3,:3].T+mat[:3,3]
        good=q[:,2]>.01
        uv=q[good,:2]/q[good,2:3]
        uv=np.floor(uv).astype(int)
        good=(uv[:,0]>=0)&(uv[:,0]<min(width,size[0]))&(uv[:,1]>=0)&(uv[:,1]<min(height,size[1]))
        uv=uv[good]
        result[k,uv[:,1],uv[:,0]]=True
        result[k]=maximum_filter(result[k], size=2*radius+1, mode='constant')
    return result

def shifted_mask(mask, seed):
    rng=np.random.default_rng(seed)
    out=np.stack([np.roll(m,(int(rng.integers(m.shape[0])),int(rng.integers(m.shape[1]))),(0,1)) for m in mask])
    assert np.array_equal(out.sum((1,2)),mask.sum((1,2)))
    return out
