"""Count current labels from the immutable training manifest, never validation."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from ..metrics import CLASS_NAMES
from ..train import atomic_json, digest
from .protocol import ANALYSIS


def balance_multipliers(counts):
    counts = np.asarray(counts[:17], dtype=np.float64)
    if counts.shape != (17,) or np.any(counts <= 0):
        raise ValueError('all foreground classes must occur in the training subset')
    reference = float(np.mean(counts))
    return np.clip(np.sqrt(reference / counts), .5, 3.), reference


def count_one(sample):
    path = Path(sample['occupancy_files'][0])
    if path.parent.name != sample['token']:
        raise ValueError('manifest occupancy path differs from the selected keyframe')
    with np.load(path, allow_pickle=False) as data:
        labels = data['semantics']
    if labels.shape != (200, 200, 16):
        raise ValueError(f'unexpected occupancy grid: {path}')
    valid = labels != 255
    if np.any((labels[valid] < 0) | (labels[valid] >= 18)):
        raise ValueError(f'invalid label: {path}')
    return (np.bincount(labels[valid].astype(np.int64), minlength=18), int((~valid).sum()),
            dict(token=sample['token'], scene=sample['scene'], path=str(path), sha256=digest(path)))


def main():
    path = Path('data/sparseworld_opus_gaussian_bias_4096_1024/train.json')
    manifest = json.loads(path.read_text())
    if manifest['split'] != 'train' or manifest['num_samples'] != 4096:
        raise ValueError('class weights must come only from the fixed 4096 training keyframes')
    target = ANALYSIS / 'training_class_frequencies_mean_balance.json'
    if target.exists():
        raise ValueError('refusing to replace frozen training frequency evidence')
    counts = np.zeros(18, dtype=np.int64)
    frame_counts = np.zeros(18, dtype=np.int64)
    scenes = [set() for _ in range(18)]
    inputs, ignored = [], 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (hist, ignore, source) in enumerate(pool.map(count_one, manifest['samples']), 1):
            counts += hist
            frame_counts += hist > 0
            ignored += ignore
            for c in np.flatnonzero(hist):
                scenes[c].add(source['scene'])
            inputs.append(source)
            if index % 512 == 0:
                print(f'TRAIN_FREQUENCIES {index}/4096', flush=True)
    factors, reference = balance_multipliers(counts)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    atomic_json(target, dict(split='train', horizons=[0], num_samples=4096,
        train_manifest=str(path), train_manifest_sha256=digest(path),
        total_voxels=4096 * 200 * 200 * 16, ignored_voxels=ignored,
        class_names=CLASS_NAMES, voxel_counts=counts.tolist(), frame_counts=frame_counts.tolist(),
        scene_counts=[len(v) for v in scenes], occupied_reference_mean_count=reference,
        auxiliary_multipliers=factors.tolist(), multiplier_rule='clip(sqrt(mean(foreground_counts)/class_count),0.5,3.0)',
        scope='Only current training labels; no masks, validation labels, future labels or scene reweighting.',
        occupancy_inputs=inputs))
    print(json.dumps(dict(path=str(target), voxel_counts=counts.tolist(), auxiliary_multipliers=factors.tolist())), flush=True)


if __name__ == '__main__':
    main()
