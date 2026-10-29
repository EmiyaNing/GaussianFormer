"""Deterministic evaluation caching, checked against the original 48-image pipeline."""
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch


class EvaluationFrames:
    """Native current-frame transforms, native temporal selection/projection math.

    Read/transform six current images once; history images have exactly the same
    deterministic preprocessing, so their backbone/FPN features can be reused.
    Training records are evaluated with the test pipeline, never augmentation.
    """
    def __init__(self, cfg, split):
        from dataset import OPENOCC_DATASET
        from dataset.opus_temporal_history import LoadOPUSV1ImageHistory
        options=deepcopy(cfg.val_dataset_config)
        options.imageset=cfg[split+'_dataset_config'].imageset
        options.phase='val'
        for transform in options.pipeline:
            if transform['type']=='LoadOPUSV1ImageHistory':
                transform['num_history']=0
            if transform['type']=='NuScenesAdaptor':
                transform['num_cams']=6
        options.return_keys=list(options.return_keys)+['ego_pose','history_frame_indices']
        self.base=OPENOCC_DATASET.build(options)
        self.keyframes=self.base.keyframes
        self.lookup={tuple(key):i for i,key in enumerate(self.keyframes)}
        self.selector=LoadOPUSV1ImageHistory(num_history=7,num_cams=6,test_mode=True,interval=(4,8))
        # Reuse the original image transform to obtain the exact camera homography.
        from PIL import Image
        from dataset.transform_3d import ResizeCropFlipImage
        resize,dims,crop,flip,rotate=self.base._sample_augmentation()
        _,ida=ResizeCropFlipImage()._img_transform(Image.new('RGB',(2,2)),resize,dims,crop,flip,rotate)
        self.homography=np.eye(4)
        self.homography[:3,:3]=ida

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        from dataset.utils import get_img2global
        row=self.base[index]
        scene,frame=self.keyframes[index]
        history,_=self.selector.select_indices(dict(scene_infos=self.base.scene_infos,
            scene_token=scene,frame_index=frame,sensor_types=self.base.sensor_types))
        keys=[(scene,frame)]+[(scene,i) for i in history]
        projections=[row['projection_mat']]
        for _,past in keys[1:]:
            record=self.base.scene_infos[scene][past]
            matrices=[]
            for cam in self.base.sensor_types:
                sensor=record['data'][cam]
                img2global=get_img2global(sensor['calib'],sensor['pose'])
                matrices.append(self.homography @ (np.linalg.inv(img2global) @ row['ego_pose']))
            projections.append(np.asarray(matrices,dtype=np.float32))
        row['projection_mat']=np.concatenate(projections,axis=0)
        row['image_wh']=np.tile(row['image_wh'],(8,1))
        row['history_frame_indices']=np.array([i for _,i in keys],dtype=np.int64)
        row['feature_frame_keys']=keys
        row['source_index']=index
        row.pop('ego_pose')
        return row


class FeatureCache:
    def __init__(self,network,dataset):
        self.network=network
        self.dataset=dataset
        self.cache=OrderedDict()
        self.original=network.extract_img_feat
        network.extract_img_feat=self.extract

    def close(self):
        self.network.extract_img_feat=self.original
        self.cache.clear()

    def extract(self,imgs,metas,**kwargs):
        keys=[tuple(k) for k in metas['feature_frame_keys'][0]]
        if self.cache and next(iter(self.cache))[0]!=keys[0][0]:
            self.cache.clear()
        if keys[0] not in self.cache:
            self.cache[keys[0]]=self.network._extract_chunk(imgs[0])
        for key in keys:
            if key not in self.cache:
                # Needed only for isolated pilot frames or a nonconsecutive resume.
                row=self.dataset.base[self.dataset.lookup[key]]
                self.cache[key]=self.network._extract_chunk(row['img'].to(imgs.device))
        levels=[torch.cat([self.cache[k][j] for k in keys],dim=0).unsqueeze(0)
                for j in range(len(self.cache[keys[0]]))]
        for key in list(self.cache):
            if key not in keys:
                del self.cache[key]
        return dict(ms_img_feats=levels)
