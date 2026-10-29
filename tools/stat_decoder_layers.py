"""Frozen tiny/small/large decoder diagnostics on all 6,019 validation frames.

python3.10 tools/stat_decoder_layers.py --action pilot
python3.10 tools/stat_decoder_layers.py --action run --workers 3
python3.10 tools/report_decoder_layers.py
"""
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
from tools.probe_gaussian_scale_supervision import layout
from tools.eval_gaussian_sizes import paths, build
from tools.decoder_layer_stats import extension
from tools.decoder_layer_stats.common import quintiles, SUPPORTS, GROUPS, DISTANCES, COUNTS
from tools.query_foreground_distance import extension as distance_extension

RUN = ROOT / 'work_dirs/decoder_layers_full_val'
HISTORY = ROOT / 'work_dirs/gaussian_sizes_full_val'
MODELS = ('tiny', 'small', 'large')


def initialize(size):
    config, checkpoint = paths(size)
    cfg = Config.fromfile(config)
    old = read(HISTORY / size / 'protocol.json')
    assert read(HISTORY / size / 'complete.json')['status'] == 'complete'
    assert sha(checkpoint) == old['checkpoint_sha256']
    assert sha(cfg.val_dataset_config.imageset) == old['annotation_sha256']
    assert hashlib.sha256(cfg.pretty_text.encode()).hexdigest() == old['resolved_config_sha256']
    sources = [Path(__file__), ROOT / 'tools/decoder_layer_stats/__init__.py',
               ROOT / 'tools/decoder_layer_stats/common.py', ROOT / 'tools/decoder_layer_stats/kernels.cu',
               ROOT / 'tools/query_foreground_distance/__init__.py',
               ROOT / 'tools/query_foreground_distance/kernels.cu',
               *[ROOT / p for p in old['source_sha256']],
               *sorted((ROOT / 'model/encoder').glob('opus*encoder.py'))]
    sources = list(dict.fromkeys(p for p in sources if p.is_file()))
    protocol = dict(version=1, model=size, checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=old['checkpoint_sha256'], config=str(config.relative_to(ROOT)),
        resolved_config_sha256=old['resolved_config_sha256'], annotation_sha256=old['annotation_sha256'],
        history_protocol_sha256=sha(HISTORY / size / 'protocol.json'),
        queries=cfg.model.lifter.num_queries, children=list(cfg.model.encoder.num_refines),
        expected_frames=6019, expected_scenes=150, seed=928, eval_mode=True, training=False,
        domain='Camera-valid GT foreground labels 0..16; voxel size 0.4m',
        grouping='Per-frame final (sixth) GS nearest-foreground load; stable ascending argsort reversed; fixed query IDs across layers; 0-20% is highest load; zero-load queries included',
        reference_center='Decode raw, unclamped output query_points to meters then average children; this is next-layer sampling reference center for layers 1..5; output reference centroid only for layer 6',
        distance='Exact independent exhaustive Euclidean minima to foreground voxel centers and closed voxel cubes, divided by 0.4m; additionally mean of child GS distances; unweighted',
        support='Production integer candidate cube and positive FP32 opacity*exp(power); geometric 1sigma / strict 3sigma / native / same-class 1sigma',
        correct_GS='GS with at least one camera-valid foreground voxel center in its 1sigma core whose GT equals argmax of that GS 17 semantic scores; denominator ALL predicted GS in layer/group',
        attribute_note='Only final layer has learned scale, rotation, opacity; layers 1..5 use fixed initial attributes. No calibration or extra training.',
        aggregation='Sum unique group foreground unions and foreground denominators over frames; sum GS hit counts and ALL GS denominators; distances use FP64 query sums',
        columns=dict(supports=list(SUPPORTS), groups=list(GROUPS), distances=list(DISTANCES), counts=list(COUNTS)),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    folder = RUN / size
    if (folder / 'protocol.json').exists():
        assert read(folder / 'protocol.json') == protocol, 'Inputs changed; use a new run directory'
    else:
        write(folder / 'protocol.json', protocol)
        cfg.dump(folder / 'resolved_config.py')
        for p in sources:
            dest = folder / 'sources' / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    return cfg, protocol


def capture(net):
    def hook(module, inputs, output):
        net._decoder_raw_points = [s['query_points'].detach() for s in output['representation']]
    return net.encoder.register_forward_hook(hook)


def reference_distances(means, gt, h, actual, seed=928):
    mu, xyz = means.double().cpu().numpy(), gt.double().cpu().numpy()
    ids = np.random.default_rng(seed).choice(len(mu), min(16, len(mu)), replace=False)
    expected = []
    for i in ids:
        delta = np.abs(xyz - mu[i])
        expected.append([np.sqrt((delta ** 2).sum(1).min()) / h,
                         np.sqrt((np.maximum(delta - h / 2, 0) ** 2).sum(1).min()) / h])
    np.testing.assert_allclose(actual[ids], expected, atol=3e-5, rtol=3e-6)
    return float(np.max(np.abs(actual[ids] - expected)))


def validate_support(values, xyz, targets, labels, classes, groups, bits, coverage, unions, gs, agg, q, k):
    """Independent established bitset implementation plus direct FP64 GS checks."""
    from tools.query_final_support import extension as independent
    means, scales, inverse, alpha, sem, centers, radii = values
    lookup = torch.full_like(labels, -1)
    lookup[targets] = torch.arange(len(targets), device=xyz.device, dtype=torch.int32)
    old_bits, old_coverage, _ = independent().support(means, inverse, alpha, centers, radii, xyz,
        lookup, len(targets), q, k, agg.H, agg.W, agg.D)
    assert torch.equal(bits[:3], old_bits) and torch.equal(coverage[:3], old_coverage)
    packed = bits.cpu().numpy().view(np.uint64)
    for scope in range(4):
        assert int(unions[0, scope]) == int((bits[scope] != 0).any(1).sum())
        for group in range(5):
            mask = np.zeros((q + 63) // 64, np.uint64)
            for j in np.flatnonzero(groups == group):
                mask[j // 64] |= np.uint64(1) << np.uint64(j % 64)
            assert int(unions[group + 1, scope]) == int(np.any(packed[scope] & mask, axis=1).sum())
    # Exhaustively enumerate all visible GT for sampled individual GS. Include
    # rotations, native box clipping, no-contact GS and class mismatches.
    x = xyz[targets].double().cpu().numpy()
    y = labels[targets].cpu().numpy()
    cells = ((xyz[targets] - agg.pc_min) / agg.grid_size).long().cpu().numpy()
    mu, inv = means.double().cpu().numpy(), inverse.double().cpu().numpy()
    cc, rr, aa, pred = [t.cpu().numpy() for t in (centers, radii, alpha, classes)]
    ids = np.random.default_rng(928).choice(len(mu), min(32, len(mu)), replace=False)
    for g in ids:
        d = mu[g] - x
        power = -.5 * (inv[g, :3] * d ** 2).sum(1) - (inv[g, 3:] * d[:, [0, 1, 0]] * d[:, [1, 2, 2]]).sum(1)
        core = ((np.abs(cells - cc[g]) <= rr[g]).all(1) & (aa[g] > 0) & (power >= -.5))
        np.testing.assert_array_equal(gs[g].cpu().numpy(), [core.sum(), (core & (y == pred[g])).sum()])


@torch.inference_mode()
def measure(net, out, batch, protocol, previous, frame, validate=False):
    from loss.opus_set_loss import _nearest_indices
    q = protocol['queries']
    children = protocol['children']
    load = previous['load_sixth'][frame].astype(np.int32)
    groups = quintiles(load)
    group_cuda = torch.tensor(groups, device='cuda', dtype=torch.int32)
    xyz = out['sampled_xyz'][0].contiguous()
    labels = batch['occ_label'].flatten().int().contiguous()
    camera = batch['occ_cam_mask'].flatten().bool()
    targets = torch.where(camera & (labels >= 0) & (labels < 17))[0]
    assert len(targets) > 0 and int(load.sum()) == len(targets)
    gt = xyz[targets].contiguous()
    lookup = torch.full_like(labels, -1)
    lookup[targets] = torch.arange(len(gt), device=xyz.device, dtype=torch.int32)
    agg = net.head.aggregator
    assert abs(float(agg.grid_size) - .4) < 1e-7
    ds, cs, us, qs = [], [], [], []
    refs = []
    checks = []
    assert len(out['gaussians']) == len(net._decoder_raw_points) == len(children) == 6
    for stage, (gaussian, raw, k) in enumerate(zip(out['gaussians'], net._decoder_raw_points, children)):
        _, values = layout(net.head, gaussian)
        means, scales, inverse, alpha, sem, centers, radii = values
        assert len(means) == q * k and raw.shape == (1, q, k, 3)
        pc = raw.new_tensor(net.encoder.pc_range)
        ref = (pc[:3] + raw[0] * (pc[3:] - pc[:3])).mean(1).contiguous()
        distances = distance_extension().distances(torch.cat((ref, means)).contiguous(), gt, agg.grid_size)
        assert bool(torch.isfinite(distances).all())
        assert bool((distances[:, 1] <= distances[:, 0] + 1e-5).all())
        query_d = torch.cat((distances[:q].double(), distances[q:].reshape(q, k, 2).double().mean(1)), 1)
        classes = sem.argmax(-1).int().contiguous()
        bits, coverage, unions, gs = extension().support(means, inverse, alpha, centers, radii,
            xyz, lookup, labels, classes, group_cuda, len(gt), q, k, agg.H, agg.W, agg.D)
        assert bool((coverage[3] <= coverage[0]).all())
        assert bool((coverage[0] <= coverage[1]).all()) and bool((coverage[1] <= coverage[2]).all())
        assert bool((gs[:, 1] <= gs[:, 0]).all())
        assert int(gs[:, 0].sum()) >= int(coverage[0].sum()) >= int(unions[0, 0])
        assert int(gs[:, 1].sum()) >= int(coverage[3].sum()) >= int(unions[0, 3])
        if stage == 5:
            np.testing.assert_array_equal(coverage[0].cpu().numpy(), previous['query_core'][frame])
            old_cc = previous['class_coverage'][frame].sum(0)
            np.testing.assert_array_equal(unions[0, [0, 1, 2, 3]].cpu().numpy(), old_cc[[1, 3, 7, 4]])
            valid = camera & (labels >= 0) & (labels < 18)
            pred = out['final_occ'].flatten()
            cm = torch.bincount((labels[valid] * 18 + pred[valid]).long(), minlength=324).reshape(18, 18)
            np.testing.assert_array_equal(cm.cpu().numpy(), previous['confusion'][frame])
            if validate:
                actual = torch.bincount(_nearest_indices(gt, means, 1024) // k, minlength=q).cpu().numpy()
                np.testing.assert_array_equal(actual, load)
        if validate:
            error = reference_distances(torch.cat((ref, means)), gt, agg.grid_size, distances.cpu().numpy(), stage + 928)
            # Separately include references so their different meaning is tested.
            error = max(error, reference_distances(ref, gt, agg.grid_size, distances[:q].cpu().numpy()))
            validate_support(values, xyz, targets, labels, classes, groups, bits, coverage, unions, gs, agg, q, k)
            checks.append(dict(layer=stage + 1, GS=q*k, distance_max_abs_error_grid=error,
                               independent_bitsets_exact=True, FP64_GS_checks=32))
        hit = (gs > 0).reshape(q, k, 2).sum(1)
        pairs = gs.reshape(q, k, 2).long().sum(1)
        ds.append(query_d.cpu().numpy())
        cs.append(torch.cat((hit, pairs), 1).cpu().numpy())
        us.append(unions.cpu().numpy())
        qs.append(coverage.T.cpu().numpy())
        refs.append(ref.cpu().numpy())
        del bits, coverage, unions, gs, distances
    return dict(query_distances=np.stack(ds), query_GS_counts=np.stack(cs),
        group_unions=np.stack(us), query_coverage=np.stack(qs), reference_centers=np.stack(refs),
        load_sixth=load, load_groups=groups, foreground=np.array(len(gt), np.int64)), checks


def history(size, scene):
    path = HISTORY / size / 'scenes' / scene
    meta = read(path.with_suffix('.json'))
    with np.load(path.with_suffix('.npz')) as z:
        previous = {key: z[key].copy() for key in ('load_sixth', 'query_core', 'class_coverage', 'confusion')}
    return meta, previous


def pilot(size, cfg, protocol, net):
    from copy import deepcopy
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    ds = EvaluationFrames(cfg, 'val')
    native_cfg = deepcopy(cfg.val_dataset_config)
    native_cfg.return_keys = list(native_cfg.return_keys) + ['history_frame_indices']
    native = OPENOCC_DATASET.build(native_cfg)
    a = to_cuda(custom_collate_fn_temporal([native[0]]))
    b = to_cuda(custom_collate_fn_temporal([ds[0]]))
    start = time.monotonic()
    with torch.inference_mode():
        original = forward(net, a)
        raw_original = [v.clone() for v in net._decoder_raw_points]
        cache = FeatureCache(net, ds)
        try:
            out = forward(net, b)
            torch.testing.assert_close(original['pred_occ'][-1], out['pred_occ'][-1], rtol=0, atol=0)
            for ga, gb in zip(original['gaussians'], out['gaussians']):
                for key in ga._fields:
                    torch.testing.assert_close(getattr(ga, key), getattr(gb, key), rtol=0, atol=0)
            for va, vb in zip(raw_original, net._decoder_raw_points):
                torch.testing.assert_close(va, vb, rtol=0, atol=0)
            scene = ds.keyframes[0][0]
            _, old = history(size, scene)
            row, checks = measure(net, out, b, protocol, old, 0, validate=True)
        finally:
            cache.close()
    write(RUN / size / 'pilot.json', dict(status='passed', model=size,
        native_and_cached_all_six_layers_exact=True, final_load_core_semantic_confusion_history_exact=True,
        checks=checks, seconds=time.monotonic()-start))
    print(f'PILOT {size}: all six layers passed', flush=True)
    net.flush_memory_phase()
    return row


def evaluate(size, workers, action):
    from torch.utils.data import DataLoader, Subset
    from dataset.utils import custom_collate_fn_temporal
    cfg, protocol = initialize(size)
    folder = RUN / size
    if (folder / 'complete.json').exists():
        assert read(folder / 'complete.json')['checkpoint_sha256'] == protocol['checkpoint_sha256']
        return
    net = build(cfg, size)
    hook = capture(net)
    try:
        if not (folder / 'pilot.json').exists() or action == 'pilot':
            pilot(size, cfg, protocol, net)
        if action == 'pilot':
            return
        ds = EvaluationFrames(cfg, 'val')
        scenes = {}
        for i, (scene, _) in enumerate(ds.keyframes):
            scenes.setdefault(scene, []).append(i)
        assert len(ds) == 6019 and len(scenes) == 150
        pending = []
        for scene, ids in scenes.items():
            path = folder / 'scenes' / f'{scene}.json'
            if path.exists():
                saved = read(path)
                assert saved['indices'] == ids and saved['checkpoint_sha256'] == protocol['checkpoint_sha256']
                assert sha(path.with_suffix('.npz')) == saved['npz_sha256']
            else:
                pending.extend(ids)
        loader = DataLoader(Subset(ds, pending), batch_size=1, shuffle=False, num_workers=workers,
            collate_fn=custom_collate_fn_temporal, pin_memory=True,
            **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))
        cache = FeatureCache(net, ds)
        started, done, active = time.monotonic(), 0, None
        try:
            for index, batch in zip(pending, loader):
                scene, f = ds.keyframes[index]
                if active is None:
                    active, arrays, tokens, ids = scene, {}, [], []
                    old_meta, previous = history(size, scene)
                    assert old_meta['indices'] == scenes[scene]
                assert active == scene
                batch = to_cuda(batch)
                with torch.inference_mode():
                    out = forward(net, batch)
                    row, _ = measure(net, out, batch, protocol, previous, len(ids))
                token = ds.base.scene_infos[scene][f]['token']
                assert token == old_meta['tokens'][len(ids)]
                for key, value in row.items():
                    arrays.setdefault(key, []).append(value)
                ids.append(index)
                tokens.append(token)
                done += 1
                if index == scenes[scene][-1]:
                    path = folder / 'scenes' / f'{scene}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.with_suffix('.npz.tmp').open('wb') as stream:
                        np.savez_compressed(stream, **{key: np.stack(v) for key, v in arrays.items()})
                    path.with_suffix('.npz.tmp').replace(path)
                    write(path.with_suffix('.json'), dict(status='complete', scene=scene, indices=ids,
                        tokens=tokens, frames=len(ids), checkpoint_sha256=protocol['checkpoint_sha256'], npz_sha256=sha(path)))
                    status = dict(status='running', model=size, frames=6019-len(pending)+done,
                        scenes=len(list((folder / 'scenes').glob('*.json'))), expected_frames=6019,
                        expected_scenes=150, frames_per_second=done/(time.monotonic()-started),
                        updated_at=datetime.now().astimezone().isoformat())
                    write(folder / 'progress.json', status)
                    write(RUN / 'status.json', status)
                    print(status, flush=True)
                    active = None
                del row, out, batch
            files = sorted((folder / 'scenes').glob('*.json'))
            indices = [i for p in files for i in read(p)['indices']]
            tokens = [t for p in files for t in read(p)['tokens']]
            assert len(files) == 150 and sorted(indices) == list(range(6019)) and len(set(tokens)) == 6019
            for rel, digest in protocol['source_sha256'].items():
                assert sha(ROOT / rel) == digest, rel
            assert sha(paths(size)[1]) == protocol['checkpoint_sha256']
            status = dict(status='complete', model=size, frames=6019, scenes=150,
                checkpoint_sha256=protocol['checkpoint_sha256'], seconds_this_run=time.monotonic()-started)
            write(folder / 'complete.json', status)
            write(folder / 'progress.json', status)
        finally:
            cache.close()
    finally:
        hook.remove()
        net.flush_memory_phase()
        del net
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('pilot', 'run'), default='run')
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'evaluation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for size in args.models:
            evaluate(size, args.workers, args.action)
    if args.action == 'run' and all((RUN / size / 'complete.json').exists() for size in MODELS):
        from tools.report_decoder_layers import main as report
        report()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(RUN / 'error.json', dict(status='failed', type=type(error).__name__, message=str(error),
                                     updated_at=datetime.now().astimezone().isoformat()))
        raise
