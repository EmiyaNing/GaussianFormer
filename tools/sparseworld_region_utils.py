"""CPU utilities for paired native SparseWorld regional evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np

HORIZONS = (2, 4, 6)
CLASS_NAMES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free']
DYNAMIC = (2, 3, 4, 5, 6, 7, 9, 10)
STATIC = (1, 8, 11, 12, 13, 14, 15, 16)
STATE_NAMES = ('both_wrong', 'only_perception', 'only_forecast', 'both_correct')
ROOT = Path('out/sparseworld_region_comparison/20261008_native512')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def sanitize(x):
        if isinstance(x, dict):
            return {str(k): sanitize(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [sanitize(v) for v in x]
        if isinstance(x, np.ndarray):
            return sanitize(x.tolist())
        if isinstance(x, np.generic):
            return sanitize(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf8')
    tmp.replace(path)


def joint_counts(gt, forecast, perception, mask=None):
    if gt.shape != forecast.shape or gt.shape != perception.shape:
        raise ValueError('Paired grid shapes differ')
    if np.any(forecast > 17) or np.any(perception > 17):
        raise ValueError('Invalid predicted label')
    valid = (gt >= 0) & (gt < 18)
    if mask is not None:
        valid &= mask
    g, f, p = [a[valid].astype(np.int64) for a in (gt, forecast, perception)]
    return np.bincount(g * 324 + f * 18 + p, minlength=5832).reshape(18, 18, 18)


def states(gt, forecast, perception, binary=False):
    if binary:
        gt, forecast, perception = [a != 17 for a in (gt, forecast, perception)]
    return 2 * (forecast == gt).astype(np.uint8) + (perception == gt).astype(np.uint8)


def metric_from_hist(hist):
    hist = np.asarray(hist, dtype=np.float64)
    diagonal = np.diagonal(hist, axis1=-2, axis2=-1)
    union = hist.sum(-1) + hist.sum(-2) - diagonal
    iou = np.divide(diagonal, union, out=np.full_like(union, np.nan), where=union > 0)
    inter = hist[..., :17, :17].sum(axis=(-2, -1))
    occ_union = hist.sum(axis=(-2, -1)) - hist[..., 17, 17]
    occ = np.divide(inter, occ_union, out=np.full_like(inter, np.nan), where=occ_union > 0)
    return dict(mIoU=np.nanmean(iou[..., :17], axis=-1) * 100,
                OccIoU=occ * 100, per_class_iou=iou * 100)


def state_metrics(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(-1)
    proportions = np.divide(counts, total[..., None],
                            out=np.full_like(counts, np.nan), where=total[..., None] > 0)
    p00, p01, p10, p11 = np.moveaxis(proportions, -1, 0)
    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    return dict(proportions=proportions * 100, advantage=(p01 - p10) * 100,
                forecast_accuracy=(p10 + p11) * 100,
                perception_accuracy=(p01 + p11) * 100,
                perception_corrects_forecast=ratio(p01, p01 + p00) * 100,
                forecast_corrects_perception=ratio(p10, p10 + p00) * 100,
                oracle_accuracy=(1 - p00) * 100, total=total)


def percentile_interval(samples):
    """Preserve all metric axes, append lower/upper CI axis last."""
    return np.moveaxis(np.nanpercentile(samples, [2.5, 97.5], axis=0), 0, -1)


def voxel_centers():
    x = np.arange(200, dtype=np.float32) * .4 - 39.8
    y = np.arange(200, dtype=np.float32) * .4 - 39.8
    z = np.arange(16, dtype=np.float32) * .4 - .8
    return np.stack(np.meshgrid(x, y, z, indexing='ij'), -1)


def boundary_band(gt, radius=.4):
    """Valid GT cells within Euclidean radius of a 6-neighbor interface midpoint."""
    from scipy.ndimage import distance_transform_edt
    valid = gt < 18
    boundary = np.zeros(gt.shape, bool)
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        changed = valid[left] & valid[right] & (gt[left] != gt[right])
        boundary[left] |= changed
        boundary[right] |= changed
    if not boundary.any():
        return np.zeros_like(boundary)
    # Interface-adjacent centers are 0.2 m from the interface. Additional
    # layers are included by the center-to-boundary-center distance + 0.2 m.
    return valid & (distance_transform_edt(~boundary, sampling=.4) + .2 <= radius + 1e-6)
