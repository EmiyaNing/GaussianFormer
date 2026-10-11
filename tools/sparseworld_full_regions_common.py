"""Full-census evaluation helpers, isolated from production datasets/models."""
import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from dataset import OPENOCC_DATASET
from dataset.sparseworld_trajectory import NuScenesSparseWorldTrajectoryDataset, _pose
from tools.sparseworld_region_utils import HORIZONS

ROOT = Path('out/sparseworld_region_comparison/20261010_full_nuscenes')
REPORT = Path('analysis/hypotheses/prediction_perception_regions/full_dataset')


@OPENOCC_DATASET.register_module()
class NuScenesFullRegionDataset(NuScenesSparseWorldTrajectoryDataset):
    """All four-history/two-future eligible sources; mark later missing targets.

    Tail frames are repeated ONLY to keep the image/GT loader shape fixed.
    Neither metric nor teacher window ever consumes a repeated target.
    The forecaster consumes only the five real historical frames.
    """
    def actual_future(self, token):
        scene, index, _ = self.trajectory_by_token[token]
        result = []
        for r in self.trajectory_info[scene][index+1:index+7]:
            frame = self.annotation_by_token.get(r['token'])
            if frame is None or not frame.get('occ_path'):
                break
            if not all(c in frame.get('data', {}) for c in self.sensor_types):
                break
            result.append(frame)
        return result

    def _future_infos(self, token):
        result = self.actual_future(token)
        if len(result) < 2:
            raise ValueError('Full-region source needs two real future keyframes')
        return result + [result[-1]] * (6-len(result))


def make_dataset(cfg, split):
    spec = copy.deepcopy(cfg.train_dataset_config if split == 'train' else cfg.val_dataset_config)
    spec.type = 'NuScenesFullRegionDataset'
    spec.phase = 'val'
    spec.history_frames = 4
    spec.future_steps = 2
    spec.official_boundary_filter = False
    spec.pipeline = [copy.deepcopy(x) for x in cfg.train_pipeline
                     if x['type'] != 'PhotoMetricDistortionMultiViewImage']
    for transform in spec.pipeline:
        if transform['type'] == 'LoadMultiViewImageHistory':
            transform['num_history'] = 4
        if transform['type'] == 'NuScenesAdaptor':
            transform['num_cams'] = 66
    ds = OPENOCC_DATASET.build(spec)
    # Constructor also checks state/trajectory/history/GT eligibility.
    ds.keyframes = [(s, i) for s, i in ds.keyframes
                    if len(ds.actual_future(ds.scene_infos[s][i]['token'])) >= 2]
    return ds, spec


def record_for(ds, index, split):
    scene, ix = ds.keyframes[index]
    frames = ds.scene_infos[scene]
    current = frames[ix]
    history = [frames[i] for i in ds._history_indices(frames, ix, 4, True)]
    future = ds.actual_future(current['token'])
    relative = {0: current, **{-i-1:f for i,f in enumerate(history)},
                **{i+1:f for i,f in enumerate(future)}}
    def frame(r):
        f = relative[r]
        return dict(relative=r, token=f['token'], timestamp=f['timestamp'])
    targets=[]
    for h in HORIZONS:
        if h > len(future):
            targets.append(dict(horizon=h, available=False, token=None,
                                forecast_frames=[], perception_frames=[]))
            continue
        f=future[h-1]
        targets.append(dict(horizon=h, available=True, token=f['token'],
            timestamp=f['timestamp'], actual_seconds=(f['timestamp']-current['timestamp'])/1e6,
            pose=_pose(f).tolist(), annotations=f.get('anns', []),
            forecast_frames=[frame(r) for r in range(0,-5,-1)],
            perception_frames=[frame(r) for r in range(h,h-5,-1)]))
    return dict(index=index, split=split, scene=scene, token=current['token'],
        timestamp=current['timestamp'], current_pose=_pose(current).tolist(),
        current_occ_path=current['occ_path'], targets=targets,
        frame_tokens=[relative[r]['token'] for r in range(0,-5,-1)]+[f['token'] for f in future]+[future[-1]['token']]*(6-len(future)))


def future_window(data, imgs, h):
    frames = [-r if r<0 else (0 if r==0 else 4+r) for r in range(h,h-5,-1)]
    ids = [6*f+c for f in frames for c in range(6)]
    metas=dict(data, projection_mat=data['projection_mat'][:,ids].clone().float(),
               image_wh=data['image_wh'][:,ids])
    transforms=data['future_ego_to_current'].float()
    current_from_target=transforms[:,h-1]
    for slot,r in enumerate(range(h,h-5,-1)):
        frame_from_target=current_from_target if r<=0 else torch.linalg.inv(transforms[:,r-1])@current_from_target
        metas['projection_mat'][:,6*slot:6*slot+6] = metas['projection_mat'][:,6*slot:6*slot+6]@frame_from_target[:,None]
    return imgs[:,ids].contiguous(),metas,frames


class WindowFeatureCache:
    """Reuse independent frozen CNN features; original batch-of-30 kernels.

    Every cache fill runs one REAL five-frame window, exactly as the original
    implementation. Historical encoders receive historical feature slices only.
    No teacher query state/feature is supplied to the forecast encoder.
    """
    def __init__(self, model, enabled=True, capacity=16):
        self.model=model;self.enabled=enabled;self.capacity=capacity
        self.cache=OrderedDict();self.misses=0;self.hits=0

    def get(self, imgs, tokens):
        assert imgs.shape[1]==30 and len(tokens)==5
        if not self.enabled:
            self.misses+=1
            return self.model.extract_img_feat(imgs=imgs)['ms_img_feats']
        missing=any(t not in self.cache for t in tokens)
        if missing:
            features=self.model.extract_img_feat(imgs=imgs)['ms_img_feats'];self.misses+=1
            for i,t in enumerate(tokens):
                if t not in self.cache:self.cache[t]=[f[:,6*i:6*i+6].clone() for f in features]
        else:self.hits+=1
        for t in tokens:self.cache.move_to_end(t)
        out=[torch.cat([self.cache[t][i] for t in tokens],1) for i in range(len(self.cache[tokens[0]]))]
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return out


def encode_window(model, imgs, metas, ego, features):
    state=dict(imgs=imgs,metas=metas,points=None,ms_img_feats=features)
    state.update(model.lifter(**state))
    state['query_points']=state['query_points']*((torch.tanh(model.points_scale(ego))+1)*.35+.8)
    state.update(model.encoder(**state))
    return state
