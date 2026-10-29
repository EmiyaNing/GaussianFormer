"""Tiny/small, full train/val query radii around actual decoder sampling centers."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import gc
import hashlib
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
from tools.stat_boundary_gaussians import read, write, sha, forward, to_cuda
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.stat_query_budget import FIELDS as OLD_FIELDS
from tools.query_radius_stats import FIELDS, capture_final_sampler, measure, ellipsoid_radius_bounds

RUN = ROOT / 'work_dirs/query_influence_radius_full'
MODELS = {
    'tiny': dict(config='config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py',
        checkpoint='ckpts/opus_gaussian_8f_tiny.pth', old='work_dirs/query_budget_full', q=600, children=128),
    'small': dict(config='config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_small_8f.py',
        checkpoint='ckpts/opus_gaussian_8f_small.pth', old='work_dirs/small_gaussian_diagnosis_full', q=1200, children=64),
}
EXPECTED = dict(val=dict(frames=6019, scenes=150), train=dict(frames=28130, scenes=700))


def build(name):
    from visualize import load_model
    from mmengine.runner import set_random_seed
    cfg = Config.fromfile(ROOT / MODELS[name]['config'])
    set_random_seed(928)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    return cfg, load_model(cfg, ROOT / MODELS[name]['checkpoint'], 'cuda:0')


def initialize():
    files = [Path(__file__), ROOT / 'tools/query_radius_stats.py',
        ROOT / 'tools/query_final_support/kernels.cu', ROOT / 'tools/query_foreground_distance/kernels.cu',
        ROOT / 'model/encoder/opus_encoder.py', ROOT / 'model/encoder/opus_v1_gaussian_temporal_encoder.py',
        ROOT / 'model/encoder/opus_v1_gaussian_encoder.py', ROOT / 'model/lifter/opus_query_lifter.py',
        ROOT / 'model/head/gaussian_opus_head.py', ROOT / 'tools/boundary_gaussian_stats/data.py']
    models = {}
    for name, options in MODELS.items():
        cfg = Config.fromfile(ROOT / options['config'])
        old = read(ROOT / options['old'] / 'protocol.json')
        digest = sha(ROOT / options['checkpoint'])
        assert digest == old['checkpoint_sha256']
        assert old['expected'] == EXPECTED
        models[name] = dict(**options, checkpoint_sha256=digest,
            resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
            annotation_sha256={s: sha(cfg[s + '_dataset_config'].imageset) for s in EXPECTED},
            old_protocol_sha256=sha(ROOT / options['old'] / 'protocol.json'))
        files.append(ROOT / options['config'])
    assert models['tiny']['annotation_sha256'] == models['small']['annotation_sha256']
    protocol = dict(version=1, expected=EXPECTED, models=models, fields=FIELDS, grid_meters=.4,
        source_sha256={str(f.relative_to(ROOT)): sha(f) for f in files},
        phases=['val/tiny', 'val/small', 'train/tiny', 'train/small'],
        reference='Actual final sampler center observed after positional encoding; decoded stage-5 reference-point mean, before final GS generation',
        all_geometry='All final child GS, equal weight; no opacity/semantic weighting, no clipping of ideal ellipsoid envelopes',
        core_group='Distinct camera-valid foreground voxels in union of final child 1sigma supports, exact historical census',
        field_radius='Mean/max child center radius; 1sigma and 3sigma union enclosing radius bounds; exact axis-aligned ideal support spans',
        candidate_radius='Maximum distance from reference to native candidate voxel centers, including clipped integer-box corners, not a minimum foreground evidence threshold',
        data='Full 700 train + 150 val scenes per model; frozen eval mode and deterministic test transforms', seed=928)
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol, 'Protocol changed; use a new run directory'
    else:
        write(RUN / 'protocol.json', protocol)
        for f in files:
            dst = RUN / 'sources' / f.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
        for name, options in MODELS.items():
            Config.fromfile(ROOT / options['config']).dump(RUN / (name + '_resolved_config.py'))
    return protocol


def analytic_checks():
    # Isotropic envelopes have an exact sphere radius; check both bounds.
    generator = torch.Generator().manual_seed(928)
    d = torch.randn(16, 3, generator=generator)
    d[0] = 0
    sigma = torch.linspace(.08, .4, 16)
    scales = sigma[:, None].expand(-1, 3)
    rotation = torch.eye(3).expand(16, -1, -1)
    for k in (1., 3.):
        lo, hi, _ = ellipsoid_radius_bounds(d, scales, rotation, k)
        truth = d.norm(dim=-1) + k * sigma
        torch.testing.assert_close(lo, truth, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(hi, truth, rtol=1e-5, atol=1e-6)
    # Dense surface samples of anisotropic ellipsoids must fit the upper bound.
    scales = torch.rand(16, 3, generator=generator) * .32 + .08
    unit = torch.nn.functional.normalize(torch.randn(16, 4096, 3, generator=generator), dim=-1)
    for k in (1., 3.):
        lo, hi, _ = ellipsoid_radius_bounds(d, scales, rotation, k)
        sampled = (d[:, None] + k * scales[:, None] * unit).norm(dim=-1).amax(1)
        assert bool((sampled <= hi + 1e-5).all())
        assert bool((lo <= hi + 1e-5).all())
    return dict(isotropic_bounds_exact=True, anisotropic_sampled_surfaces_within_bounds=True)


def pilot(name, cfg, network):
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    options = MODELS[name]
    native = OPENOCC_DATASET.build(cfg.val_dataset_config)
    batch = to_cuda(custom_collate_fn_temporal([native[0]]))
    with torch.inference_mode():
        original = forward(network, batch)['pred_occ'][-1].clone()
        handle = capture_final_sampler(network)
        output = forward(network, batch)
        torch.testing.assert_close(original, output['pred_occ'][-1], atol=0, rtol=0)
        observed = network._query_radius_capture['center'].clone()
    del output, batch, native
    dataset = EvaluationFrames(cfg, 'val')
    scene, _ = dataset.keyframes[0]
    with np.load(ROOT / options['old'] / 'scenes/val' / (scene + '.npz')) as z:
        previous = z['metrics'][0]
    cache = FeatureCache(network, dataset)
    batch = to_cuda(custom_collate_fn_temporal([dataset[0]]))
    with torch.inference_mode():
        output = forward(network, batch)
        torch.testing.assert_close(original, output['pred_occ'][-1], atol=0, rtol=0)
        torch.testing.assert_close(observed, network._query_radius_capture['center'], atol=0, rtol=0)
        row, audit = measure(network, output, batch, options['q'], options['children'],
                             previous[:, OLD_FIELDS.index('core_coverage_all_fg')])
    means = output['gaussian'].means[0].cpu().numpy().reshape(options['q'], options['children'], 3)
    ref = row['reference_center']
    distance = np.linalg.norm(means.astype(np.float64) - ref[:, None].astype(np.float64), axis=-1) / .4
    np.testing.assert_allclose(row['metrics'][:, FIELDS.index('center_radius_mean')], distance.mean(1), atol=2e-5, rtol=2e-6)
    np.testing.assert_allclose(row['metrics'][:, FIELDS.index('center_radius_max')], distance.max(1), atol=2e-5, rtol=2e-6)
    cache.close()
    handle.remove()
    del network._query_radius_capture
    del output, batch, cache, dataset, original, observed
    result = dict(status='passed', model=name, original_hook_logits_exact=True,
        original_cached_logits_exact=True, sampler_center_exact=True,
        fp64_center_radius_reference=True, historical_core_query_counts_exact=True,
        analytic=analytic_checks(), **audit)
    write(RUN / name / 'pilot.json', result)


def run_phase(name, split, cfg, network, protocol, workers):
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Subset
    options = MODELS[name]
    dataset = EvaluationFrames(cfg, split)
    by_scene = {}
    for i, (scene, _) in enumerate(dataset.keyframes):
        by_scene.setdefault(scene, []).append(i)
    assert len(dataset) == EXPECTED[split]['frames'] and len(by_scene) == EXPECTED[split]['scenes']
    folder = RUN / name / 'scenes' / split
    pending = []
    for scene, ids in by_scene.items():
        path = folder / (scene + '.json')
        if path.exists():
            saved = read(path)
            assert saved['indices'] == ids and saved['checkpoint_sha256'] == protocol['models'][name]['checkpoint_sha256']
            assert path.with_suffix('.npz').exists()
        else:
            pending.extend(ids)
    loader = DataLoader(Subset(dataset, pending), batch_size=1, shuffle=False, num_workers=workers,
        collate_fn=custom_collate_fn_temporal, pin_memory=True,
        **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))
    cache, handle = FeatureCache(network, dataset), capture_final_sampler(network)
    active = None
    start, processed = time.monotonic(), 0
    for index, batch in zip(pending, loader):
        scene, scene_frame = dataset.keyframes[index]
        if active is None:
            active, records, arrays = scene, [], {}
            meta = read(ROOT / options['old'] / 'scenes' / split / (scene + '.json'))
            with np.load(ROOT / options['old'] / 'scenes' / split / (scene + '.npz')) as z:
                previous = z['metrics']
            assert meta['indices'] == by_scene[scene]
        assert scene == active
        i = len(records)
        batch = to_cuda(batch)
        with torch.inference_mode():
            output = forward(network, batch)
            row, audit = measure(network, output, batch, options['q'], options['children'],
                previous[i, :, OLD_FIELDS.index('core_coverage_all_fg')])
        token = dataset.base.scene_infos[scene][scene_frame]['token']
        assert token == meta['records'][i]['sample_token']
        assert audit['foreground'] == meta['records'][i]['foreground']
        assert audit['foreground'] - audit['covered'] == meta['records'][i]['no_core']
        row.update(core_foreground=previous[i, :, OLD_FIELDS.index('core_coverage_all_fg')].astype(np.int32),
            camera_core_contact=previous[i, :, OLD_FIELDS.index('gs_camera_core')] > 0,
            core_contact_gs=previous[i, :, OLD_FIELDS.index('core_gs_touches_fg')].astype(np.int16))
        records.append(dict(index=index, token=token, **audit))
        for k, v in row.items():
            arrays.setdefault(k, []).append(v)
        processed += 1
        if len(records) == len(by_scene[scene]):
            path = folder / (scene + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, **{k: np.stack(v) for k, v in arrays.items()})
            path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
            write(path, dict(status='complete', model=name, split=split, scene=scene,
                scene_name=meta['scene_name'], indices=by_scene[scene], records=records,
                checkpoint_sha256=protocol['models'][name]['checkpoint_sha256']))
            done = len(list(folder.glob('*.json')))
            elapsed = time.monotonic() - start
            total_scenes = sum(1 for _ in RUN.glob('*/scenes/*/*.json'))
            write(RUN / 'progress.json', dict(status='running', pid=os.getpid(), model=name, split=split,
                phase_scenes=done, expected_phase=EXPECTED[split], total_scenes=total_scenes,
                expected_total_scenes=1700, processed_this_phase=processed, fps=processed / elapsed,
                seconds_this_phase=elapsed, updated_at=datetime.now().astimezone().isoformat()))
            print(f'{name}/{split}: {done}/{len(by_scene)} scenes; {processed} frames this phase; {processed/elapsed:.2f} fps', flush=True)
            active = None
        del output, batch, row, audit
    assert active is None
    cache.close()
    handle.remove()
    if hasattr(network, '_query_radius_capture'):
        del network._query_radius_capture
    del loader, cache, dataset
    write(RUN / name / (split + '_complete.json'), dict(status='complete', **EXPECTED[split]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = initialize()
        start = time.monotonic()
        try:
            for split in ('val', 'train'):
                for name in ('tiny', 'small'):
                    marker = RUN / name / (split + '_complete.json')
                    if marker.exists():
                        assert read(marker) == dict(status='complete', **EXPECTED[split])
                        continue
                    cfg, network = build(name)
                    if not (RUN / name / 'pilot.json').exists():
                        pilot(name, cfg, network)
                    run_phase(name, split, cfg, network, protocol, args.workers)
                    del network
                    gc.collect()
                    torch.cuda.empty_cache()
            write(RUN / 'complete.json', dict(status='complete', model_frames=68298, model_scenes=1700,
                seconds=time.monotonic() - start))
            write(RUN / 'progress.json', dict(status='complete', model_frames=68298, model_scenes=1700,
                seconds=time.monotonic() - start))
        except BaseException as error:
            write(RUN / 'failure.json', dict(error=repr(error), time=datetime.now().astimezone().isoformat()))
            raise


if __name__ == '__main__':
    main()
