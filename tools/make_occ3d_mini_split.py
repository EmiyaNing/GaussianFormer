"""Create a reproducible 4096/1024 keyframe Occ3D mini benchmark.

The source train/val sets are disjoint.  ``infos`` retains complete selected
scenes because dataset indexing and optional history refer to original frame
indices; ``metadata`` alone determines the samples actually loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/nuscenes_cam'
OUT = ROOT / 'data/mini_occ3d_4096_1024'


def make_split(source: Path, output: Path, count: int, seed: int):
    with source.open('rb') as handle:
        data = pickle.load(handle)
    keyframes = [tuple(item) for item in data['metadata']]
    if count > len(keyframes) or len(set(keyframes)) != len(keyframes):
        raise ValueError(f'Invalid or insufficient keyframes in {source}')
    selected = random.Random(seed).sample(sorted(keyframes), count)
    selected.sort(key=lambda item: (item[0], item[1]))
    scenes = {scene for scene, _ in selected}
    reduced = {
        'infos': {scene: data['infos'][scene] for scene in sorted(scenes)},
        'metadata': selected,
    }
    for scene, index in selected:
        if not 0 <= index < len(reduced['infos'][scene]):
            raise ValueError(f'Invalid frame index: {scene}/{index}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        pickle.dump(reduced, handle, protocol=pickle.HIGHEST_PROTOCOL)
    selected_json = [[scene, index] for scene, index in selected]
    digest = hashlib.sha256(json.dumps(selected_json, separators=(',', ':')).encode()).hexdigest()
    return {
        'source': str(source.relative_to(ROOT)),
        'imageset': str(output.relative_to(ROOT)),
        'source_samples': len(keyframes),
        'samples': len(selected),
        'selected_scenes': len(scenes),
        'seed': seed,
        'sample_pairs_sha256': digest,
        'sample_pairs': selected_json,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-count', type=int, default=4096)
    parser.add_argument('--val-count', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=928)
    args = parser.parse_args()
    manifest = {
        'selection': 'uniform keyframe sample without replacement, independently within official train/val',
        'train': make_split(
            SOURCE / 'nuscenes_infos_train_sweeps_occ.pkl', OUT / 'train.pkl',
            args.train_count, args.seed),
        'val': make_split(
            SOURCE / 'nuscenes_infos_val_sweeps_occ.pkl', OUT / 'val.pkl',
            args.val_count, args.seed + 1),
    }
    path = OUT / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split in ('train', 'val'):
        info = manifest[split]
        print(f'{split}: {info["samples"]} samples, {info["selected_scenes"]} scenes, '
              f'sha256={info["sample_pairs_sha256"]}')
    print(path)


if __name__ == '__main__':
    main()
