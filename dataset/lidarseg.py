"""Current-scan nuScenes lidarseg targets alongside the unchanged Occ3D input."""

from functools import lru_cache
from pathlib import Path
import random

import numpy as np
from pyquaternion import Quaternion

from . import OPENOCC_DATASET
from .dataset import NuScenesDataset


@lru_cache(None)
def lidarseg_label_mapping():
    from nuscenes.eval.lidarseg.utils import LidarsegClassMapper
    from nuscenes.utils.color_map import get_colormap

    mapper = LidarsegClassMapper.__new__(LidarsegClassMapper)
    mapper.ignore_class = mapper.get_ignore_class()
    names = list(get_colormap())
    coarse = mapper.get_coarse2idx()
    mapping = mapper.get_fine2coarse()
    assert len(names) == 32 and names[17] == 'vehicle.car'
    lut = np.array([coarse[mapping[name]] for name in names], dtype=np.int64)
    classes = [name for name, index in sorted(coarse.items(), key=lambda item: item[1])][1:]
    return lut, classes


def load_current_lidarseg(point_file, label_file, calibration, pc_range):
    """Keep point/label order; transform only this scan, never fused history."""
    raw = np.fromfile(point_file, dtype=np.float32).reshape(-1, 5)
    fine = np.fromfile(label_file, dtype=np.uint8)
    lut, _ = lidarseg_label_mapping()
    if len(raw) != len(fine):
        raise ValueError(f'Point/label length mismatch: {point_file}, {label_file}')
    if fine.size and fine.max() >= len(lut):
        raise ValueError(f'Invalid fine lidarseg class in {label_file}')
    rotation = Quaternion(calibration['rotation']).rotation_matrix
    xyz = (raw[:, :3].astype(np.float64) @ rotation.T +
           np.asarray(calibration['translation'])).astype(np.float32)
    limits = np.asarray(pc_range, dtype=np.float32)
    target = lut[fine]
    keep = ((xyz >= limits[:3]) & (xyz < limits[3:])).all(-1) & (target > 0)
    return (np.ascontiguousarray(xyz[keep]), np.ascontiguousarray(target[keep] - 1),
            np.flatnonzero(keep), dict(raw_points=len(raw), labelled_points=int((target > 0).sum())))


@OPENOCC_DATASET.register_module()
class NuScenesLidarSegDataset(NuScenesDataset):
    def __getitem__(self, index):
        if not hasattr(self, 'augmentation_seed'):
            return super().__getitem__(index)
        # The experiment can seed augmentation per sample and epoch so arms
        # and mid-epoch resumes receive identical images/history selections.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        item_seed = (self.augmentation_seed + int(index)) % (2**32)
        np.random.seed(item_seed)
        random.seed(item_seed)
        try:
            return super().__getitem__(index)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def get_data_info(self, info, scene_token=None, frame_index=None):
        if not self.occ3d or self.occ3d_coord != 'ego':
            raise ValueError('Joint Occ3D/lidarseg experiment requires ego coordinates')
        result = super().get_data_info(info, scene_token, frame_index)
        lidar = info['data']['LIDAR_TOP']
        label = Path(self.data_path) / 'lidarseg/v1.0-trainval' / (lidar['token'] + '_lidarseg.bin')
        xyz, target, indices, counts = load_current_lidarseg(
            Path(self.data_path) / lidar['filename'], label, lidar['calib'], self.pc_range)
        result.update(lidarseg_xyz=xyz, lidarseg_label=target,
                      lidarseg_original_index=indices, lidarseg_counts=counts,
                      lidarseg_token=lidar['token'])
        return result
