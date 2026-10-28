"""Build immutable, context-preserving 4096/1024 token manifests."""
import argparse
import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from mmengine.config import Config

from dataset.sparseworld_strict import NuScenesSparseWorldStrictDataset


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def resolve_occ(path, root):
    for candidate in (Path(path), Path(root) / path, Path(root).parent / path):
        if candidate.is_dir():
            candidate = candidate / 'labels.npz'
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f'missing occupancy: {path}')


def sample_context(dataset, scene, index):
    record = dataset.scene_infos[scene][index]
    _, future = dataset._future_records(record['token'])
    positions = dataset.keyframe_indices_by_scene[scene]
    current = positions.index(index)
    if current < dataset.history_frames:
        raise ValueError('insufficient actual keyframe history')
    history_indices = [positions[current - i] for i in range(1, dataset.history_frames + 1)]
    images = []
    for idx in [index] + history_indices:
        for camera in dataset.sensor_types:
            path = Path(dataset.data_path) / dataset.scene_infos[scene][idx]['data'][camera]['filename']
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append(str(path))
    lidar = Path(dataset.data_path) / record['data']['LIDAR_TOP']['filename']
    if not lidar.is_file():
        raise FileNotFoundError(lidar)
    occupancies = [resolve_occ(r['occ_path'], 'data/occ3d') for r in [record] + future]
    # Check labels and masks rather than accepting merely existing files.
    for path in occupancies:
        with np.load(path) as data:
            for field in ('semantics', 'mask_camera', 'mask_lidar'):
                if data[field].shape != (200, 200, 16):
                    raise ValueError(f'invalid {field} shape at {path}')
            if not np.all((data['semantics'] >= 0) & (data['semantics'] <= 17)):
                raise ValueError(f'invalid semantic labels at {path}')
    return dict(token=record['token'], scene=scene, frame_index=index,
                timestamp=int(record['timestamp']), history_indices=history_indices,
                history_tokens=[dataset.scene_infos[scene][i]['token'] for i in history_indices],
                future_tokens=[r['token'] for r in future], images=images,
                current_lidar=str(lidar), occupancy_files=occupancies)


def build_split(config, split, count, seed, directory):
    cfg = dict(config[f'{split}_dataset_config'])
    cfg.pop('type')
    cfg['pipeline'] = []
    print(f'Loading complete {split} metadata...', flush=True)
    dataset = NuScenesSparseWorldStrictDataset(**cfg)
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(len(dataset.keyframes))
    selected, rejected = [], []
    for candidate in candidates:
        scene, index = dataset.keyframes[int(candidate)]
        try:
            selected.append(sample_context(dataset, scene, index))
        except (OSError, ValueError, KeyError) as error:
            rejected.append(dict(scene=scene, index=index, reason=str(error)))
        if len(selected) % 256 == 0 and selected:
            print(f'{split}: checked {len(selected)}/{count}', flush=True)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f'{split}: only {len(selected)} valid samples, need {count}')
    selected.sort(key=lambda sample: (sample['scene'], sample['frame_index']))
    source_paths = (cfg['imageset'], cfg['trajectory_path'], cfg['admlp_state_path'])
    manifest = dict(version=1, split=split, num_samples=count, seed=seed,
                    num_eligible=len(dataset.keyframes),
                    keyframe_interval_seconds=.5, history_frames=4, future_steps=6,
                    source_files={path: sha256(path) for path in source_paths},
                    scene_counts=dict(Counter(s['scene'] for s in selected)),
                    rejected=rejected, samples=selected)
    path = directory / f'{split}.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise RuntimeError(f'refusing to replace a different immutable subset: {path}')
    else:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    del dataset
    gc.collect()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/sparseworld_opus_gaussian_bias_4096_1024')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    config = Config.fromfile('config/opus/sparseworld_strict.py')
    train = build_split(config, 'train', 4096, args.seed, directory)
    val = build_split(config, 'val', 1024, args.seed + 1, directory)
    if set(train['scene_counts']) & set(val['scene_counts']):
        raise RuntimeError('train / held-out scenes overlap')
    if {s['token'] for s in train['samples']} & {s['token'] for s in val['samples']}:
        raise RuntimeError('train / held-out tokens overlap')
    summary = dict(train_samples=4096, test_samples=1024,
                   train_scenes=len(train['scene_counts']), test_scenes=len(val['scene_counts']),
                   train_manifest_sha256=sha256(directory / 'train.json'),
                   test_manifest_sha256=sha256(directory / 'val.json'),
                   scene_overlap=0, token_overlap=0, context_integrity='verified',
                   test_role='held-out validation for epoch selection')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
