"""Build reduced train/val splits so a bounded P1 ablation is affordable.

A full SparseWorld epoch is 6,577 iterations at ~0.85 s/iteration (~1.6 h) on a
single RTX 4090, and ``train.py`` only writes a checkpoint at an epoch
boundary.  A plan-compliant P1 (A0-A7 x 3 seeds) would therefore need roughly
190 GPU-hours.  This tool produces smaller ``imageset`` pickles with the same
schema (``infos`` + ``metadata``) so an epoch is minutes instead of hours.

The reduced split is a *hardware-budget compromise*, not a scientific choice:
it is recorded here and in the P1 report so no result is presented as if it
used the full protocol.

    python3.10 tools/make_reduced_split.py --train-scenes 12 --val-scenes 4
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANNO = ROOT / 'data' / 'nuscenes_cam'
TRAIN_PKL = ANNO / 'nuscenes_infos_train_sweeps_occ.pkl'
VAL_PKL = ANNO / 'nuscenes_infos_val_sweeps_occ.pkl'
OUT_DIR = ROOT / 'data' / 'reduced_splits'


def reduce_split(source, scenes_wanted, out_path):
    """Keep the first ``scenes_wanted`` scenes and their keyframes."""
    with source.open('rb') as handle:
        data = pickle.load(handle)
    # Sort for determinism; the loader re-sorts keyframes anyway.
    scene_order = sorted(data['infos'].keys())
    kept = scene_order[:scenes_wanted]
    kept_set = set(kept)
    reduced = {
        'infos': {scene: data['infos'][scene] for scene in kept},
        'metadata': [entry for entry in data['metadata']
                     if entry[0] in kept_set],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as handle:
        pickle.dump(reduced, handle)
    return {
        'source': str(source.relative_to(ROOT)),
        'output': str(out_path.relative_to(ROOT)),
        'scenes': len(reduced['infos']),
        'keyframes': len(reduced['metadata']),
        'original_scenes': len(data['infos']),
        'original_keyframes': len(data['metadata']),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-scenes', type=int, default=12)
    parser.add_argument('--val-scenes', type=int, default=4)
    args = parser.parse_args(argv)

    report = {
        'train': reduce_split(TRAIN_PKL, args.train_scenes,
                              OUT_DIR / 'train_reduced.pkl'),
        'val': reduce_split(VAL_PKL, args.val_scenes,
                            OUT_DIR / 'val_reduced.pkl'),
    }
    print(json.dumps(report, indent=2))
    manifest = OUT_DIR / 'manifest.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {manifest.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
