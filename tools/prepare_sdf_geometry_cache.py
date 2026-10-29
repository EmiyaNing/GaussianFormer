"""Prepare exact voxel-face SDFs for every official labeled train/val frame."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.sdf_geometry.field import VERSION, construct, save

RUN = ROOT/'work_dirs/sdf_geometry_replacement_full'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def manifest():
    scene_names = {x['token']: x['name'] for x in json.loads((ROOT/'data/nuscenes/v1.0-trainval/scene.json').read_text())}
    result = dict(version=VERSION, splits={})
    for split, count, scenes in [('train', 28130, 700), ('val', 6019, 150)]:
        annotation = ROOT/f'data/nuscenes_cam/nuscenes_infos_{split}_sweeps_occ.pkl'
        with annotation.open('rb') as stream: data = pickle.load(stream)
        keys = sorted(data['metadata'], key=lambda x: x[0]+f'{x[1]:03d}')
        assert len(keys) == count and len({s for s, _ in keys}) == scenes
        rows = []
        for index, (scene, frame) in enumerate(keys):
            record = data['infos'][scene][frame]; token = record['token']
            gt = ROOT/'data/occ3d/gts'/scene_names[scene]/token/'labels.npz'
            if not gt.exists(): raise FileNotFoundError(gt)
            rows.append(dict(index=index, scene=scene, frame=int(frame), token=token,
                gt=str(gt.relative_to(ROOT)), cache=str((RUN/'cache'/split/scene/(token+'.npz')).relative_to(ROOT))))
        result['splits'][split] = dict(frames=count, scenes=scenes, annotation_sha256=sha(annotation), rows=rows)
    if (RUN/'manifest.json').exists():
        assert json.loads((RUN/'manifest.json').read_text()) == result
    else: write(RUN/'manifest.json', result)
    return result


def process(row):
    target = ROOT/row['cache']
    if target.exists():
        with np.load(target) as d:
            current = str(d['version']) == VERSION
        if current: return dict(index=row['index'], reused=True)
    gt = ROOT/row['gt']
    with np.load(gt) as d: labels, camera = d['semantics'], d['mask_camera']
    if labels.shape != (200, 200, 16): raise ValueError(labels.shape)
    sdf, valid, normals = construct(labels, camera, return_normals=True)
    save(target, sdf, valid, sha(gt), normals=normals)
    return dict(index=row['index'], reused=False, valid_voxels=int(valid.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--split', choices=['train', 'val', 'all'], default='all')
    args = p.parse_args(); start = time.monotonic(); m = manifest()
    # Validation first enables baseline evaluation while train caches build.
    # Train cache order matches the fixed one-epoch training permutation.
    splits = ['val', 'train'] if args.split == 'all' else [args.split]
    rows = []
    for split in splits:
        items = m['splits'][split]['rows']
        order = np.random.default_rng(928).permutation(len(items)) if split == 'train' else range(len(items))
        rows.extend(items[i] for i in order)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, row in enumerate(pool.map(process, rows, chunksize=1), 1):
            if done == 1 or done % 50 == 0 or done == len(rows):
                status = dict(status='running', completed=done, total=len(rows), seconds=time.monotonic()-start,
                              workers=args.workers, splits=splits)
                write(RUN/('cache_progress_'+args.split+'.json'), status)
                print(json.dumps(status), flush=True)
            if args.split == 'all' and done == m['splits']['val']['frames']:
                write(RUN/'cache_complete_val.json', dict(status='complete', frames=done, splits=['val']))
    write(RUN/('cache_complete_'+args.split+'.json'), dict(status='complete', frames=len(rows), seconds=time.monotonic()-start, splits=splits))


if __name__ == '__main__': main()
