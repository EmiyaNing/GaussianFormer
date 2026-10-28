"""Evaluation-only region construction from frozen, existing GT metadata.

The legacy JSON attributes retain real instance velocities and global boxes at
1/2/3 seconds. Only the diagnostic runner reads these files. They are never
model inputs or training-loss masks. Regions exclude ambiguous same-class box
overlaps and expose their coverage explicitly.
"""

import json
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from dataset.sparseworld_strict import _ego_pose


HORIZONS = (2, 4, 6)
REGIONS = ("fast", "slow", "medium", "unknown_movable", "static_classes",
           "fast_neighborhood", "F_only", "P_only", "both_correct", "both_wrong")


def token_at(dataset, index, horizon):
    scene, frame = dataset.keyframes[index]
    current = dataset.scene_infos[scene][frame]
    if horizon == 0:
        return current["token"]
    _, future = dataset._future_records(current["token"])
    return future[horizon - 1]["token"]


def record_at(dataset, index, horizon):
    scene, frame = dataset.keyframes[index]
    current = dataset.scene_infos[scene][frame]
    return current if horizon == 0 else dataset._future_records(current["token"])[1][horizon - 1]


def box_regions(target, xyz, instances, ego_pose):
    target, xyz = target.reshape(-1), xyz.reshape(-1, 3)
    owner = np.full(target.size, -1, dtype=np.int32)
    speeds = np.full(target.size, np.nan, dtype=np.float32)
    ambiguous = np.zeros(target.size, dtype=bool)
    near = np.zeros((200, 200, 16), dtype=bool)
    for number, instance in enumerate(instances):
        selected = np.flatnonzero(target == instance["label"])
        if not selected.size:
            continue
        center = np.asarray(instance["translation"])
        rotation = Quaternion(instance["rotation"]).rotation_matrix
        width, length, height = instance["size"]
        half_size = np.array([length, width, height]) * .5
        world = xyz[selected] @ ego_pose[:3, :3].T + ego_pose[:3, 3]
        local = (world - center) @ rotation
        selected = selected[(np.abs(local) <= half_size + 1e-5).all(-1)]
        ambiguous[selected] |= owner[selected] >= 0
        owner[selected] = number
        speed = instance.get("speed")
        if speed is not None and np.isfinite(speed):
            speeds[selected] = speed
        if speed is not None and np.isfinite(speed) and speed >= 2:
            signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
            corners = (signs * half_size) @ rotation.T + center
            corners = (corners - ego_pose[:3, 3]) @ ego_pose[:3, :3]
            lo = np.floor((corners.min(0) - np.array([2., 2., 1.]) - [-40., -40., -1.]) / .4).astype(int)
            hi = np.ceil((corners.max(0) + np.array([2., 2., 1.]) - [-40., -40., -1.]) / .4).astype(int)
            lo, hi = np.maximum(lo, 0), np.minimum(hi, [200, 200, 16])
            if (hi > lo).all():
                near[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    speeds[ambiguous] = np.nan
    occupied = (target >= 0) & (target < 17)
    movable = (target >= 2) & (target <= 10)
    known = np.isfinite(speeds) & occupied
    masks = dict(fast=known & (speeds >= 2),
                 slow=known & (speeds < .5),
                 medium=known & (speeds >= .5) & (speeds < 2),
                 unknown_movable=movable & ~known,
                 static_classes=occupied & ~movable,
                 fast_neighborhood=near.reshape(-1) & (target < 18))
    return masks, dict(known_speed_voxels=int(known.sum()),
                       ambiguous_voxels=int(ambiguous.sum()),
                       occupied_voxels=int(occupied.sum()),
                       cached_instance_voxels=int(sum(x.get("voxels", 0) for x in instances)))


def pack_masks(masks):
    return {name: np.packbits(mask.reshape(-1)) for name, mask in masks.items()}


def load_masks(path, horizon):
    with np.load(path) as arrays:
        return {name: np.unpackbits(arrays[f"h{horizon}_{name}"], count=640000).astype(bool)
                for name in REGIONS}


def histogram(prediction, target, mask=None):
    prediction, target = prediction.reshape(-1), target.reshape(-1)
    valid = (target >= 0) & (target < 18)
    if mask is not None:
        valid &= mask
    return np.bincount(target[valid].astype(np.int64) * 18 + prediction[valid],
                       minlength=324).reshape(18, 18)


def region_result(hist):
    total = int(hist.sum())
    occupied = int(hist[:17].sum())
    recalled = int(hist[:17, :17].sum())
    predicted = int(hist[:, :17].sum())
    correct = int(np.trace(hist[:17, :17]))
    return dict(voxels=total, occupied_gt=occupied,
                semantic_accuracy=100 * int(np.trace(hist)) / total if total else None,
                occupied_semantic_accuracy=100 * correct / occupied if occupied else None,
                occupied_recall=100 * recalled / occupied if occupied else None,
                conditional_semantic_accuracy=100 * correct / recalled if recalled else None,
                occupied_precision=100 * recalled / predicted if predicted else None,
                occupied_iou=100 * recalled / (occupied + predicted - recalled)
                    if occupied + predicted - recalled else None,
                false_positive=int(hist[17, :17].sum()),
                false_negative=int(hist[:17, 17].sum()))


def attributes_for(dataset, index, attribute_root):
    path = Path(attribute_root) / (token_at(dataset, index, 0) + ".json")
    return json.loads(path.read_text())["instances"]


def speed_masks(dataset, index, horizon, target, xyz, attributes):
    return box_regions(target, xyz, attributes[HORIZONS.index(horizon)],
                       _ego_pose(record_at(dataset, index, horizon)))
