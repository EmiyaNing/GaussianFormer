"""Explicit Occ3D-derived point targets, preserving the original input cloud.

This is a derived annotation protocol, distinct from official nuScenes lidarseg.
Free points carry an outlier flag and never enter the 16-class point objective.
"""

import json
from pathlib import Path

import numpy as np

from . import OPENOCC_DATASET
from .lidarseg import NuScenesLidarSegDataset

PROTOCOL = 'occ3d_aligned_lidarseg_v1'
IGNORE_LABEL = -1
OUTLIER_LABEL = -2
STATUS_NAMES = ('unchanged_semantic', 'relabeled_semantic', 'free_outlier',
                'others_ignore', 'original_ignore', 'outside_roi')


def align_point_labels(original_coarse, original_indices, occ_labels):
    """Return full-scan arrays; process only the original valid in-ROI cohort.

    original_coarse: original nuScenes mapped IDs 0(ignore), 1..16.
    labels: training IDs 0..15, -1(ignore), -2(free/outlier).
    occ_label: original Occ3D IDs 0..17, 255(not queried).
    """
    coarse = np.asarray(original_coarse)
    indices = np.asarray(original_indices, dtype=np.int64)
    occ = np.asarray(occ_labels)
    if coarse.ndim != 1 or (coarse.size and (coarse.min() < 0 or coarse.max() > 16)):
        raise ValueError('Invalid original coarse labels')
    if indices.ndim != 1 or occ.shape != indices.shape:
        raise ValueError('Point index and occupancy label lengths differ')
    if indices.size and (indices.min() < 0 or indices.max() >= len(coarse) or
                         not (np.diff(indices) > 0).all() or (coarse[indices] == 0).any()):
        raise ValueError('Indices must be sorted original valid point indices')
    if occ.size and (occ.min() < 0 or occ.max() > 17):
        raise ValueError('Invalid Occ3D labels')
    labels = np.full(len(coarse), IGNORE_LABEL, dtype=np.int8)
    status = np.where(coarse > 0, 5, 4).astype(np.uint8)
    occ_full = np.full(len(coarse), 255, dtype=np.uint8)
    occ_full[indices] = occ
    semantic = (occ >= 1) & (occ <= 16)
    labels[indices[semantic]] = occ[semantic].astype(np.int8)-1
    status[indices[semantic]] = (coarse[indices[semantic]] != occ[semantic]).astype(np.uint8)
    labels[indices[occ == 17]] = OUTLIER_LABEL
    status[indices[occ == 17]] = 2
    status[indices[occ == 0]] = 3
    return dict(labels=labels, status=status, occ_label=occ_full,
                original_coarse=coarse.astype(np.uint8), outlier_mask=status == 2)


def load_aligned_point_targets(path, xyz, target, indices, counts, *,
                               sample_token, lidar_token, pc_range):
    """Validate a derived file and filter supervision, preserving raw indices."""
    with np.load(path, allow_pickle=False) as z:
        if (str(z['protocol']) != PROTOCOL or str(z['sample_token']) != sample_token or
                str(z['lidar_token']) != lidar_token):
            raise ValueError(f'Derived label protocol/token mismatch: {path}')
        if not np.array_equal(z['pc_range'], np.asarray(pc_range, dtype=np.float64)):
            raise ValueError(f'Derived label ROI mismatch: {path}')
        labels, status, outlier = z['labels'], z['status'], z['outlier_mask']
        original, occ = z['original_coarse'], z['occ_label']
    if any(v.shape != (counts['raw_points'],) for v in (labels, status, outlier, original, occ)):
        raise ValueError(f'Derived label/raw scan length mismatch: {path}')
    if labels.dtype != np.int8 or not np.isin(labels, [-2, -1, *range(16)]).all():
        raise ValueError(f'Invalid derived training labels: {path}')
    if not np.array_equal(outlier, status == 2) or not np.array_equal(outlier, labels == -2):
        raise ValueError(f'Outlier flag and label disagree: {path}')
    if not np.array_equal(np.flatnonzero(status <= 3), indices):
        raise ValueError(f'Original valid point cohort changed: {path}')
    if not np.array_equal(original[indices], target+1):
        raise ValueError(f'Original labels changed since derivation: {path}')
    keep = labels[indices] >= 0
    valid = indices[keep]
    if not np.array_equal(labels[valid]+1, occ[valid]):
        raise ValueError(f'Derived semantic label differs from Occ3D: {path}')
    if not (occ[outlier] == 17).all() or not (occ[status == 3] == 0).all():
        raise ValueError(f'Invalid free/others assignment: {path}')
    n = np.bincount(status, minlength=6)
    details = dict(counts, original_valid_points=len(indices), retained_points=int(keep.sum()),
                   outlier_points=int(n[2]), relabeled_points=int(n[1]), others_ignored_points=int(n[3]))
    return (np.ascontiguousarray(xyz[keep]), labels[valid].astype(np.int64),
            valid, details, np.flatnonzero(outlier))


@OPENOCC_DATASET.register_module()
class NuScenesOccAlignedLidarSegDataset(NuScenesLidarSegDataset):
    def __init__(self, *args, lidarseg_label_root, **kwargs):
        super().__init__(*args, **kwargs)
        self.lidarseg_label_root = Path(lidarseg_label_root)
        manifest = json.loads((self.lidarseg_label_root/'manifest.json').read_text())
        state = json.loads((self.lidarseg_label_root/'state.json').read_text())
        if manifest['protocol'] != PROTOCOL or state['status'] != 'complete':
            raise ValueError('Occ3D-derived point labels are incomplete or have another protocol')
        if list(self.pc_range) != manifest['settings']['pc_range']:
            raise ValueError('Dataset ROI differs from the derived annotation ROI')

    def get_data_info(self, info, scene_token=None, frame_index=None):
        result = super().get_data_info(info, scene_token, frame_index)
        token = result['lidarseg_token']
        # Each LiDAR sample token is globally unique; the manifest identifies split.
        split = 'val' if self.test_mode else 'train'
        path = self.lidarseg_label_root/'labels'/split/(token+'.npz')
        if not path.exists():
            # Fixed-preprocessing train readers can have phase='val'. Match token,
            # never silently fall back to original point labels.
            alternate = self.lidarseg_label_root/'labels'/('train' if split == 'val' else 'val')/(token+'.npz')
            if not alternate.exists():
                raise FileNotFoundError(path)
            path = alternate
        xyz, target, indices, counts, outliers = load_aligned_point_targets(
            path, result['lidarseg_xyz'], result['lidarseg_label'], result['lidarseg_original_index'],
            result['lidarseg_counts'], sample_token=info['token'], lidar_token=token, pc_range=self.pc_range)
        result.update(lidarseg_xyz=xyz, lidarseg_label=target, lidarseg_original_index=indices,
                      lidarseg_counts=counts, lidarseg_outlier_original_index=outliers)
        # Do not remove encoder input points based on GT. The mask affects targets only.
        return result
