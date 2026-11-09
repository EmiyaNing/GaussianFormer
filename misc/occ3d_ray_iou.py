"""Occ3D official RayIoU for the nuScenes Occ3D occupancy grid.

Ported from the reference implementation used by OPUS / 4D-Occ-forecasting
(``loaders/ray_metrics.py`` plus the ``lib/dvr`` CUDA renderer).  The metric
casts the nuScenes LiDAR ray pattern from the LiDAR origins of the scene's
keyframes, takes the first occupied voxel along every ray for prediction and
ground truth, and counts a class hit only when the two hit distances agree
within 1 m, 2 m or 4 m.

RayIoU is a *ray-level* metric: it is not the voxel mIoU and it is not the
binary ``iou2`` printed by the training loop.  ``RayIoU`` is the mean over the
three thresholds; ``RayIoU@1/@2/@4`` are the per-threshold means.  The free
class (17) is excluded, and rays whose ground-truth hit is free are dropped,
exactly as in the reference.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import torch

PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
VOXEL_SIZE = 0.4
OCC_SHAPE = (200, 200, 16)
GRID = [1, 16, 200, 200]
FREE_ID = 17
THRESHOLDS = (1, 2, 4)

OCC_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]

_DVR = None
_LIDAR_RAYS = None


def _nanmean(values):
    """Mean over the entries that are not NaN, without numpy's empty-slice warning."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = values[~np.isnan(values)]
    return float(valid.mean()) if valid.size else float('nan')


def dvr_extension():
    """Build/load the dvr renderer once; PyTorch caches the compiled binary."""
    global _DVR
    if _DVR is None:
        from torch.utils.cpp_extension import load

        source = Path(__file__).resolve().with_name('dvr')
        _DVR = load(
            'occ3d_dvr',
            sources=[str(source / 'dvr.cpp'), str(source / 'dvr.cu')],
            verbose=False,
            extra_cuda_cflags=['-allow-unsupported-compiler'])
    return _DVR


def generate_lidar_rays():
    """The nuScenes LiDAR ray pattern used by the official RayIoU."""
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuScenes lidar vertical FOV: [0.2107773983152201, -0.5439104895672159]
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)
            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)
            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


def lidar_rays():
    global _LIDAR_RAYS
    if _LIDAR_RAYS is None:
        _LIDAR_RAYS = torch.from_numpy(generate_lidar_rays())
    return _LIDAR_RAYS


def _occupancy(semantics, device):
    """Binary occupancy in dvr's [B, T, Z, Y, X] layout (free class is empty)."""
    semantics = np.asarray(semantics)
    if semantics.shape != OCC_SHAPE:
        raise ValueError(f'Expected semantics {OCC_SHAPE}, got {semantics.shape}')
    occupancy = torch.from_numpy((semantics != FREE_ID).astype(np.float32))
    return occupancy.permute(2, 1, 0)[None, None].contiguous().to(device)


def hit_distances(semantics, origins, device='cuda'):
    """First-hit (label, distance in metres) for every ray and origin.

    ``semantics`` is the dense ``[200, 200, 16]`` label grid in the current ego
    frame, ``origins`` the ``[T, 3]`` LiDAR origins in the same frame.  Returns
    ``[T * N_rays, 2]`` float64 with the hit class in column 0 and the ray
    parameter in metres in column 1, mirroring ``process_one_sample``.
    """
    semantics = np.asarray(semantics)
    occupancy = _occupancy(semantics, device)
    renderer = dvr_extension()
    rays = lidar_rays().to(device)

    offset = torch.tensor(PC_RANGE[:3], device=device)[None, None]
    scaler = torch.tensor([VOXEL_SIZE] * 3, device=device)[None, None]
    tindex = torch.zeros([1, rays.shape[0]], device=device)

    outputs = []
    for origin in np.asarray(origins, dtype=np.float32):
        origin = torch.from_numpy(origin)[None, None].to(device)
        endpoints = rays[None] + origin
        with torch.no_grad():
            distance, _, coord_index = renderer.render_forward(
                occupancy,
                ((origin - offset) / scaler).float(),
                ((endpoints - offset) / scaler).float(),
                tindex,
                GRID,
                'test')
        distance = (distance * VOXEL_SIZE)[0].cpu().numpy()
        coords = coord_index[0].int().cpu()
        labels = semantics[coords[:, 0], coords[:, 1], coords[:, 2]]
        outputs.append(np.stack([labels.astype(np.float64), distance.astype(np.float64)], axis=-1))
    return np.concatenate(outputs, axis=0)


class Occ3DRayIoU:
    """Streaming Accumulator for the official RayIoU (no per-sample storage)."""

    def __init__(self, thresholds=THRESHOLDS, device='cuda'):
        self.thresholds = tuple(thresholds)
        self.device = device
        self.num_samples = 0
        self.gt_cnt = np.zeros(len(OCC_CLASS_NAMES), dtype=np.float64)
        self.pred_cnt = np.zeros(len(OCC_CLASS_NAMES), dtype=np.float64)
        self.tp_cnt = np.zeros((len(self.thresholds), len(OCC_CLASS_NAMES)), dtype=np.float64)

    def add(self, semantics_pred, semantics_gt, origins):
        """Add one sample; both grids are dense ``[200, 200, 16]`` label arrays."""
        origins = np.asarray(origins, dtype=np.float32)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError(f'ray_origins must be [T, 3], got {origins.shape}')
        predicted = hit_distances(semantics_pred, origins, self.device)
        target = hit_distances(semantics_gt, origins, self.device)

        # The official metric scores only rays whose GT hit is occupied.
        valid = target[:, 0].astype(np.int64) != FREE_ID
        predicted, target = predicted[valid], target[valid]

        for level, threshold in enumerate(self.thresholds):
            close = np.abs(predicted[:, 1] - target[:, 1]) < threshold
            for cls in range(len(OCC_CLASS_NAMES)):
                pred_cls = predicted[:, 0] == cls
                gt_cls = target[:, 0] == cls
                if level == 0:
                    self.pred_cnt[cls] += pred_cls.sum()
                    self.gt_cnt[cls] += gt_cls.sum()
                self.tp_cnt[level, cls] += (pred_cls & gt_cls & close).sum()
        self.num_samples += 1

    def synchronize(self, device=None):
        """Sum the counters across ranks (RayIoU has no per-rank collectives)."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return self
        device = device or self.device
        counters = torch.from_numpy(np.concatenate([
            self.gt_cnt, self.pred_cnt, self.tp_cnt.reshape(-1)])).to(device)
        torch.distributed.all_reduce(counters)
        counters = counters.cpu().numpy()
        classes = len(OCC_CLASS_NAMES)
        self.gt_cnt = counters[:classes].copy()
        self.pred_cnt = counters[classes:2 * classes].copy()
        self.tp_cnt = counters[2 * classes:].reshape(self.tp_cnt.shape).copy()
        return self

    def per_class_iou(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            union = self.gt_cnt + self.pred_cnt
            return [
                (self.tp_cnt[level] / (union - self.tp_cnt[level]))
                for level in range(len(self.thresholds))
            ]

    def results(self):
        per_level = self.per_class_iou()
        occupied = [ious[:-1] for ious in per_level]
        result = {
            # RayIoU averages the 17 occupied classes over the three thresholds;
            # the free class is excluded, exactly as in the reference.
            'RayIoU': _nanmean(np.concatenate(occupied)),
            'RayIoU@1': _nanmean(occupied[0]),
            'RayIoU@2': _nanmean(occupied[1]),
            'RayIoU@4': _nanmean(occupied[2]),
            'RayIoU/samples': int(self.num_samples),
        }
        with warnings.catch_warnings():
            # A class absent from every sample leaves an all-NaN column.
            warnings.simplefilter('ignore', RuntimeWarning)
            per_class = np.nanmean(np.stack(per_level)[:, :-1], axis=0)
        for name, value in zip(OCC_CLASS_NAMES[:-1], per_class):
            result[f'RayIoU/{name}'] = float(value)
        return result
