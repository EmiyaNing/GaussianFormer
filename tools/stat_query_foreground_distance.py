"""Full validation: exact final GS distances for the fixed coverage bins.

Distances use visible GT foreground only and are expressed in 0.4 m grid units.
Every scene stores FP32 per-GS distances, so further summaries need no inference.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools import diagnose_small_gaussians as small
from tools.stat_boundary_gaussians import read, write, sha, forward, to_cuda
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.probe_gaussian_scale_supervision import layout
from tools.query_foreground_distance import extension
from tools.query_final_support import extension as support_extension
from tools.stat_query_budget import FIELDS

RUN = ROOT / 'work_dirs/small_query_foreground_distance_val'
COVERAGE = ROOT / 'work_dirs/small_final_query_coverage_full'
OLD = ROOT / 'work_dirs/small_gaussian_diagnosis_full'
Q, K = 1200, 64


def measure(network, out, batch, expected_core):
    _, values = layout(network.head, out['gaussian'])
    means, scales, inv, alpha, sem, centers, radii = values
    assert means.shape == (Q * K, 3)
    xyz = out['sampled_xyz'][0].contiguous()
    labels = batch['occ_label'].flatten().long()
    camera = batch['occ_cam_mask'].flatten().bool()
    targets = torch.where(camera & (labels >= 0) & (labels < 17))[0]
    gt = xyz[targets].contiguous()
    agg = network.head.aggregator
    h = float(agg.grid_size)
    assert abs(h - .4) < 1e-7 and len(gt) > 0
    distances = extension().distances(means, gt, h)
    assert bool(torch.isfinite(distances).all())
    assert bool((distances[:, 1] <= distances[:, 0] + 1e-5).all())
    lookup = torch.full((len(labels),), -1, dtype=torch.int32, device=means.device)
    lookup[targets] = torch.arange(len(gt), device=means.device, dtype=torch.int32)
    bits, coverage, distinct = support_extension().support(means, inv, alpha, centers, radii, xyz,
        lookup, len(gt), Q, K, agg.H, agg.W, agg.D)
    np.testing.assert_array_equal(coverage[0].cpu().numpy(), expected_core)
    d = distances[:, 1]
    # Bounds for distance from union of untruncated 1sigma ellipsoids to union
    # of foreground voxel solids. No claim of exact ellipsoid/voxel projection.
    lower = (d - scales.amax(1) / h).clamp_min(0).reshape(Q, K).amin(1)
    upper = (d - scales.amin(1) / h).clamp_min(0).reshape(Q, K).amin(1)
    assert bool((lower <= upper + 1e-5).all())
    assert bool((lower[coverage[0] > 0] < 1e-4).all())
    result = dict(distances=distances.reshape(Q, K, 2).cpu().numpy(),
        core_support_gap_bounds=torch.stack((lower, upper), -1).cpu().numpy())
    return result, (means, gt, h)


def initialize(cfg):
    assert read(COVERAGE / 'val_complete.json') == dict(status='complete', frames=6019, scenes=150)
    files = [Path(__file__), ROOT / 'tools/query_foreground_distance/__init__.py',
        ROOT / 'tools/query_foreground_distance/kernels.cu',
        ROOT / 'tools/query_final_support/kernels.cu', small.CONFIG,
        ROOT / 'tools/boundary_gaussian_stats/data.py', ROOT / 'tools/probe_gaussian_scale_supervision.py']
    protocol = dict(version=1, checkpoint=str(small.CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=sha(small.CHECKPOINT), config=str(small.CONFIG.relative_to(ROOT)),
        expected=dict(frames=6019, scenes=150), queries=Q, children=K, grid_meters=.4,
        columns=['nearest_foreground_voxel_center_grid', 'nearest_foreground_voxel_solid_grid'],
        source_sha256={str(f.relative_to(ROOT)): sha(f) for f in files},
        annotation_sha256=sha(cfg.val_dataset_config.imageset),
        previous_coverage_protocol_sha256=sha(COVERAGE / 'protocol.json'),
        distance='Exact exhaustive Euclidean minimum to all camera-valid foreground voxel centers and to union of their closed 0.4m cubes; FP32',
        query='Summarize distances of all 64 final GS centers; no invented query centroid',
        support_gap='Lower/upper bounds from enclosing max-scale sphere and inscribed min-scale sphere; not an exact field distance',
        no_subsampling=True, eval_mode=True, seed=928)
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol
    else:
        write(RUN / 'protocol.json', protocol)
        cfg.dump(RUN / 'resolved_config.py')
        for f in files:
            dst = RUN / 'sources' / f.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
    return protocol


def reference_check(values, distances):
    means, gt, h = values
    mu, centers = means.cpu().numpy(), gt.cpu().numpy()
    rng = np.random.default_rng(928)
    ids = rng.choice(len(mu), 128, replace=False)
    truth = []
    for i in ids:
        delta = np.abs(mu[i].astype(np.float64) - centers.astype(np.float64))
        d_center = np.sqrt(np.square(delta).sum(1).min()) / h
        d_volume = np.sqrt(np.square(np.maximum(delta - h / 2, 0)).sum(1).min()) / h
        truth.append((d_center, d_volume))
    actual = distances.reshape(-1, 2)[ids]
    np.testing.assert_allclose(actual, truth, atol=2e-5, rtol=2e-6)
    # Nearest voxel center need not identify the nearest cube: use exhaustive
    # center and volume minima independently, never a one-neighbor shortcut.
    return float(np.max(np.abs(actual - truth)))


def run(cfg, network, protocol, workers):
    from torch.utils.data import DataLoader, Subset
    from dataset.utils import custom_collate_fn_temporal
    dataset = EvaluationFrames(cfg, 'val')
    by_scene = {}
    for i, (scene, _) in enumerate(dataset.keyframes):
        by_scene.setdefault(scene, []).append(i)
    assert len(dataset) == 6019 and len(by_scene) == 150
    pending = []
    for scene, ids in by_scene.items():
        path = RUN / 'scenes' / (scene + '.json')
        if path.exists():
            meta = read(path)
            assert meta['indices'] == ids and meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
            assert path.with_suffix('.npz').exists()
        else:
            pending.extend(ids)
    loader = DataLoader(Subset(dataset, pending), batch_size=1, shuffle=False, num_workers=workers,
        collate_fn=custom_collate_fn_temporal, pin_memory=True,
        **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))
    cache = FeatureCache(network, dataset)
    active = None
    processed, start = 0, time.monotonic()
    for index, batch in zip(pending, loader):
        scene, scene_frame = dataset.keyframes[index]
        if active is None:
            active, records, distances, bounds = scene, [], [], []
            previous = read(COVERAGE / 'scenes/val' / (scene + '.json'))
            with np.load(COVERAGE / 'scenes/val' / (scene + '.npz')) as z:
                core = z['core_1sigma_query_coverage']
            with np.load(OLD / 'scenes/val' / (scene + '.npz')) as z:
                metrics = z['metrics']
            camera_core = metrics[..., FIELDS.index('gs_camera_core')] > 0
            active_gs = metrics[..., FIELDS.index('core_gs_touches_fg')].astype(np.int32)
            assert previous['indices'] == by_scene[scene]
        assert active == scene
        batch = to_cuda(batch)
        with torch.inference_mode():
            out = forward(network, batch)
            row, values = measure(network, out, batch, core[len(records)])
        if not (RUN / 'pilot.json').exists():
            error = reference_check(values, row['distances'])
            write(RUN / 'pilot.json', dict(status='passed', reference='FP64 exhaustive voxel-center and voxel-cube CPU minima',
                checked_gaussians=128, max_absolute_error_grid=error, historical_per_query_coverage_exact=True))
        token = dataset.base.scene_infos[scene][scene_frame]['token']
        assert token == previous['records'][len(records)]['token']
        records.append(dict(index=index, token=token, foreground_voxels=len(values[1])))
        distances.append(row['distances'])
        bounds.append(row['core_support_gap_bounds'])
        processed += 1
        if len(records) == len(by_scene[scene]):
            path = RUN / 'scenes' / (scene + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez(stream, distances=np.stack(distances), core_support_gap_bounds=np.stack(bounds),
                    core_foreground_voxels=core, camera_core_contact=camera_core, core_contact_gs=active_gs)
            path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
            write(path, dict(status='complete', scene=scene, indices=by_scene[scene], records=records,
                checkpoint_sha256=protocol['checkpoint_sha256']))
            done = len(list((RUN / 'scenes').glob('*.json')))
            elapsed = time.monotonic() - start
            write(RUN / 'progress.json', dict(status='running', pid=os.getpid(), scenes=done, total_scenes=150,
                processed_this_run=processed, fps=processed / elapsed, seconds=elapsed,
                updated_at=datetime.now().astimezone().isoformat()))
            print(f'{done}/150 scenes, {processed} frames, {processed/elapsed:.2f} frames/s', flush=True)
            active = None
        del out, batch, row, values
    assert active is None
    cache.close()
    write(RUN / 'complete.json', dict(status='complete', frames=6019, scenes=150,
        checkpoint_sha256=protocol['checkpoint_sha256']))
    write(RUN / 'progress.json', dict(status='complete', frames=6019, scenes=150, seconds=time.monotonic() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = Config.fromfile(small.CONFIG)
        protocol = initialize(cfg)
        try:
            extension()
            network = small.build(cfg)
            run(cfg, network, protocol, args.workers)
        except BaseException as e:
            write(RUN / 'failure.json', dict(error=repr(e), time=datetime.now().astimezone().isoformat()))
            raise


if __name__ == '__main__':
    main()
