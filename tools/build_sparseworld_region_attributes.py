#!/usr/bin/env python3
"""Offline motion and actual-window frustum attributes for paired occupancy."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pyquaternion import Quaternion

from tools.sparseworld_region_utils import (ROOT, DYNAMIC, STATIC, sha256,
                                           voxel_centers, write_json)


def category_label(name):
    mapping = {'vehicle.bicycle': 2, 'vehicle.bus.bendy': 3, 'vehicle.bus.rigid': 3,
               'vehicle.car': 4, 'vehicle.construction': 5, 'vehicle.motorcycle': 6,
               'vehicle.trailer': 9, 'vehicle.truck': 10}
    if (name.startswith('human.pedestrian.') and name not in
            ('human.pedestrian.stroller', 'human.pedestrian.wheelchair')):
        return 7
    return mapping.get(name)


def annotation_index(root, records):
    base = Path('data/nuscenes/v1.0-trainval')
    cache = root / 'annotation_index.json'
    if cache.exists():
        d = json.loads(cache.read_text())
        return d['annotations'], d['samples']
    wanted = set()
    for rec in records:
        for target in rec['targets']:
            wanted.add(target['token'])
            for frame in target['forecast_frames'] + target['perception_frames']:
                wanted.add(frame['token'])
    categories = {r['token']: r['name'] for r in json.loads((base/'category.json').read_text())}
    instances = {r['token']: category_label(categories[r['category_token']])
                 for r in json.loads((base/'instance.json').read_text())}
    samples = {r['token']: r['timestamp'] for r in json.loads((base/'sample.json').read_text())}
    raw = json.loads((base/'sample_annotation.json').read_text())
    selected = {a['token'] for a in raw if a['sample_token'] in wanted and
                instances.get(a['instance_token']) is not None}
    neighbors = {a[k] for a in raw if a['token'] in selected for k in ('prev', 'next') if a[k]}
    annotations = {}
    for a in raw:
        if a['token'] in selected or a['token'] in neighbors:
            a['label'] = instances.get(a['instance_token'])
            annotations[a['token']] = a
    del raw
    write_json(cache, dict(annotations=annotations, samples=samples))
    write_json(root/'annotation_sources.json', {str(base/(n+'.json')): sha256(base/(n+'.json'))
               for n in ['sample_annotation', 'instance', 'category', 'sample']})
    return annotations, samples


def speed(annotation, annotations, timestamps):
    neighbors = [annotations.get(annotation[k]) for k in ('prev', 'next')]
    prev, nxt = neighbors
    left, right = (prev or annotation), (nxt or annotation)
    dt = (timestamps[right['sample_token']] - timestamps[left['sample_token']])/1e6
    if dt <= 0 or dt > (3.0 if prev and nxt else 1.5):
        return np.nan
    return float(np.linalg.norm((np.asarray(right['translation']) - left['translation'])[:2]) / dt)


def frustum_seen(points, projections, wh):
    seen = np.zeros(len(points), dtype=bool)
    if not len(points):
        return seen
    for mat, size in zip(projections, wh):
        q = points @ mat[:3, :3].T + mat[:3, 3]
        depth = q[:, 2]
        seen |= ((depth > .01) & (q[:, 0] >= 0) & (q[:, 1] >= 0) &
                 (q[:, 0] < size[0] * depth) & (q[:, 1] < size[1] * depth))
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--limit', type=int, default=512)
    args = parser.parse_args()
    records = json.loads((args.root/'manifest.json').read_text())['samples']
    annotations, timestamps = annotation_index(args.root, records)
    by_sample = {}
    for a in annotations.values():
        if a['label'] is not None:
            by_sample.setdefault(a['sample_token'], []).append(a)
    frame_instance = {(a['sample_token'], a['instance_token']): a for a in annotations.values()}
    centers = voxel_centers().reshape(-1, 3)
    dest_root = args.root/'attributes'
    dest_root.mkdir(exist_ok=True)
    for position, rec in enumerate(records[:args.limit]):
        dest = dest_root/(rec['token']+'.npz')
        if dest.exists():
            continue
        pair_path = args.root/'pairs'/(rec['token']+'.npz')
        while not pair_path.exists():
            time.sleep(1)
        started = time.monotonic()
        with np.load(pair_path) as pair:
            gt = pair['GT']
            speed_maps, frustum_maps, instance_maps = [], [], []
            frame_coverage, instance_meta = [], []
            for j, target in enumerate(rec['targets']):
                labels = gt[j].reshape(-1)
                speed_map = np.full(len(centers), 3, np.uint8)  # unknown/not applicable
                frustum_map = np.full(len(centers), 4, np.uint8)
                instance_map = np.zeros(len(centers), np.uint16)
                static = np.flatnonzero(np.isin(labels, STATIC))
                fseen = frustum_seen(centers[static], pair['F_projection'][j], pair['F_wh'][j])
                pseen = frustum_seen(centers[static], pair['P_projection'][j], pair['P_wh'][j])
                frustum_map[static] = 2 * fseen.astype(np.uint8) + pseen.astype(np.uint8)
                pose = np.asarray(target['pose'], dtype=np.float32)
                dyn = np.flatnonzero(np.isin(labels, DYNAMIC))
                global_points = centers[dyn] @ pose[:3, :3].T + pose[:3, 3]
                matched = np.zeros(len(dyn), np.uint16)
                instances = []
                for a in by_sample.get(target['token'], []):
                    rotation = Quaternion(a['rotation']).rotation_matrix.astype(np.float32)
                    local = (global_points - np.asarray(a['translation'], np.float32)) @ rotation
                    size = np.asarray(a['size'], np.float32)[[1, 0, 2]]
                    inside = (np.abs(local) <= size/2 + 1e-5).all(-1) & (labels[dyn] == a['label'])
                    if not inside.any():
                        continue
                    identifier = len(instances)+1
                    matched[inside & (matched != 0)] = 65535  # ambiguous overlap
                    matched[inside & (matched == 0)] = identifier
                    v = speed(a, annotations, timestamps)
                    instances.append(dict(id=identifier, token=a['instance_token'],
                                          annotation_token=a['token'], label=a['label'], speed=v,
                                          translation=a['translation'], rotation=a['rotation'], size=a['size']))
                for inst in instances:
                    selected = np.flatnonzero(matched == inst['id'])
                    if not len(selected):
                        continue
                    voxel_ids = dyn[selected]
                    v = inst['speed']
                    if np.isfinite(v):
                        speed_map[voxel_ids] = 0 if v < .5 else (1 if v < 2 else 2)
                    instance_map[voxel_ids] = inst['id']
                    a = annotations[inst['annotation_token']]
                    rotation = Quaternion(a['rotation']).rotation_matrix.astype(np.float32)
                    local = (global_points[selected]-np.asarray(a['translation'], np.float32)) @ rotation
                    observed = []
                    supported = []
                    for prefix, key in [('F', 'forecast_frames'), ('P', 'perception_frames')]:
                        seen = np.zeros(len(selected), bool)
                        complete = True
                        for slot, frame in enumerate(target[key]):
                            prior = frame_instance.get((frame['token'], inst['token']))
                            if prior is None:
                                complete = False
                                continue
                            r = Quaternion(prior['rotation']).rotation_matrix.astype(np.float32)
                            world = local @ r.T + np.asarray(prior['translation'], np.float32)
                            target_ego = (world-pose[:3, 3]) @ pose[:3, :3]
                            seen |= frustum_seen(target_ego,
                                      pair[prefix+'_projection'][j, slot*6:(slot+1)*6],
                                      pair[prefix+'_wh'][j, slot*6:(slot+1)*6])
                        observed.append(seen)
                        supported.append(complete)
                    if all(supported):
                        frustum_map[voxel_ids] = 2 * observed[0].astype(np.uint8) + observed[1].astype(np.uint8)
                    inst['voxels'] = len(selected)
                    inst['frustum_full_track'] = all(supported)
                speed_maps.append(speed_map.reshape(200, 200, 16))
                frustum_maps.append(frustum_map.reshape(200, 200, 16))
                instance_maps.append(instance_map.reshape(200, 200, 16))
                instance_meta.append(instances)
                frame_coverage.append(dict(dynamic_voxels=len(dyn), matched=int((instance_map > 0).sum()),
                                           speed_known=int((speed_map < 3).sum()),
                                           frustum_known=int((frustum_map < 4).sum())))
            tmp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, speed=np.stack(speed_maps), frustum=np.stack(frustum_maps),
                                instance=np.stack(instance_maps), pair_sha256=np.asarray(sha256(pair_path)))
            tmp.replace(dest)
            write_json(dest.with_suffix('.json'), dict(instances=instance_meta, coverage=frame_coverage))
        if position % 16 == 0 or position+1 == args.limit:
            print(f'[attributes] {position+1}/{args.limit}, {time.monotonic()-started:.2f}s', flush=True)
    write_json(args.root/'attributes_manifest.json', dict(status='complete', samples=args.limit,
               speed_bins=['<0.5 m/s', '0.5-2 m/s', '>=2 m/s', 'unknown/not applicable'],
               frustum_bins=['neither', 'only_future_window', 'only_history_window', 'both', 'unknown'],
               motion='nuScenes global annotation centers with true time differences; class-compatible oriented-box assignment',
               frustum='Actual augmented five-frame six-camera windows; motion-compensated for full-track matched instances',
               occlusion='Not measured; frustum coverage does not imply unobstructed visibility',
               instance_match='Ambiguous overlapping boxes excluded; no outside-box voxel matching'))


if __name__ == '__main__':
    main()
