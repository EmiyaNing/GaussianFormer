"""Matched full-validation census of the original tiny/small/large checkpoints.

No training or calibration. Scene-resumable; historical results are read-only.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import fcntl
import gc
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader, Subset
from tools.stat_boundary_gaussians import read, write, sha, forward, to_cuda
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.probe_gaussian_scale_supervision import layout
from tools.semantic_field_support import extension

RUN = ROOT / 'work_dirs/gaussian_sizes_full_val'
MODELS = ('large', 'tiny', 'small')
CLASS_NAMES = ['others', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
               'terrain', 'manmade', 'vegetation']
CLASS_COLUMNS = ['GT', 'geo1', 'geo2', 'geo3', 'same1', 'same2', 'same3', 'native']


def paths(size):
    return (ROOT / f'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_{size}_8f.py',
            ROOT / f'ckpts/opus_gaussian_8f_{size}.pth')


def initialize(size):
    config, checkpoint = paths(size)
    cfg = Config.fromfile(config)
    q, k = cfg.model.lifter.num_queries, cfg.model.encoder.num_refines[-1]
    assert q * k == 76800 and q % 5 == 0
    files = [Path(__file__), ROOT / 'visualize.py',
             ROOT / 'tools/stat_boundary_gaussians.py',
             ROOT / 'tools/boundary_gaussian_stats/data.py',
             ROOT / 'tools/probe_gaussian_scale_supervision.py',
             ROOT / 'tools/semantic_field_support/__init__.py',
             ROOT / 'tools/semantic_field_support/kernels.cu',
             ROOT / 'loss/opus_set_loss.py',
             *sorted((ROOT / 'config/full_occ3d_oups_guassian').glob('*.py')),
             *sorted((ROOT / 'model/head/localagg/src').glob('*'))]
    files += [ROOT / p for p in read(ROOT / 'work_dirs/boundary_gaussian_full/protocol.json')['source_sha256']]
    files = list(dict.fromkeys(p for p in files if p.is_file()))
    protocol = dict(version=1, model=size, checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=sha(checkpoint), config=str(config.relative_to(ROOT)),
        resolved_config_sha256=__import__('hashlib').sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256=sha(cfg.val_dataset_config.imageset),
        queries=q, children=k, previous_children=cfg.model.encoder.num_refines[-2],
        frames=6019, scenes=150, seed=928, eval_mode=True, training=False,
        domain='camera-valid GT labels 0..17; foreground 0..16',
        support='native integer candidate cube, positive FP32 opacity*exp(power), Mahalanobis <=1/2/3',
        load='Each valid GT foreground voxel chooses nearest final (sixth) GS center, then its parent query; fifth-stage counts also saved',
        core='Distinct GT foreground centers in the union of each query final GS 1sigma cores; overlaps across queries allowed',
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files})
    folder = RUN / size
    if (folder / 'protocol.json').exists():
        assert read(folder / 'protocol.json') == protocol, 'Inputs changed; use a new run directory'
    else:
        write(folder / 'protocol.json', protocol)
        cfg.dump(folder / 'resolved_config.py')
    return cfg, protocol


def build(cfg, size):
    from visualize import load_model
    from mmengine.runner import set_random_seed
    set_random_seed(928)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    return load_model(cfg, paths(size)[1], 'cuda:0')


def packed(ids, q, device):
    words = np.zeros((q + 63) // 64, np.uint64)
    for i in ids:
        words[int(i) // 64] |= np.uint64(1) << np.uint64(int(i) % 64)
    return torch.from_numpy(words.view(np.int64)).to(device)


@torch.inference_mode()
def measure(net, out, batch, q, k, previous_k, validate=False):
    from loss.opus_set_loss import _nearest_indices
    _, args = layout(net.head, out['gaussian'])
    means, scales, inverse, alpha, semantic, centers, radii = args
    assert len(means) == q * k
    xyz = out['sampled_xyz'][0].contiguous()
    labels = batch['occ_label'].flatten().int().contiguous()
    camera = batch['occ_cam_mask'].flatten().bool()
    valid = camera & (labels >= 0) & (labels < 18)
    ids = torch.where(valid & (labels < 17))[0]
    y = labels[ids]
    pred = out['final_occ'].flatten().int()
    assert torch.equal(pred, out['pred_occ'][-1][0].argmax(0).flatten())
    cm = torch.bincount((labels[valid] * 18 + pred[valid]).long(), minlength=324).reshape(18, 18)
    classes = semantic.argmax(-1).int().contiguous()
    lookup = torch.full_like(labels, -1)
    lookup[ids] = torch.arange(len(ids), device=xyz.device, dtype=torch.int32)
    agg = net.head.aggregator
    bits, counts, core = extension().support(means, inverse, alpha, centers, radii, xyz,
        lookup, labels, classes, classes, len(ids), q, k, agg.H, agg.W, agg.D)
    assert torch.equal(counts[:, 3], counts[:, 6])
    hit = counts[:, [0, 1, 2, 3, 4, 5, 7]] > 0
    cc = torch.zeros((17, 8), device=xyz.device, dtype=torch.int64)
    cc[:, 0] = torch.bincount(y.long(), minlength=17)
    for j in range(7):
        cc[:, j + 1] = torch.bincount(y[hit[:, j]].long(), minlength=17)
    assert bool((hit[:, 0] <= hit[:, 1]).all()) and bool((hit[:, 1] <= hit[:, 2]).all())
    final_assignment = _nearest_indices(xyz[ids], means, 1024) // k
    coarse = out['gaussians'][-2].means[0].contiguous()
    assert len(coarse) == q * previous_k
    fifth_assignment = _nearest_indices(xyz[ids], coarse, 1024) // previous_k
    load = torch.bincount(final_assignment, minlength=q)
    load5 = torch.bincount(fifth_assignment, minlength=q)
    core_np = core.cpu().numpy()
    order = np.argsort(core_np, kind='stable')[-q // 5:]
    top_core_union = (bits & packed(order, q, bits.device)).ne(0).any(1)
    assert int(load.sum()) == int(load5.sum()) == len(ids) == int(cc[:, 0].sum())
    if validate:
        from tools.query_final_support import extension as independent
        old_bits, old_core, old_distinct = independent().support(means, inverse, alpha, centers,
            radii, xyz, lookup, len(ids), q, k, agg.H, agg.W, agg.D)
        assert torch.equal(old_bits[0], bits) and torch.equal(old_core[0], core)
        assert torch.equal(old_distinct[0] > 0, hit[:, 0])
        for t in np.random.default_rng(928).choice(len(ids), min(16, len(ids)), replace=False):
            x = xyz[ids[t]]
            grid = ((x - agg.pc_min[0]) / agg.grid_size).long()
            box = ((grid >= centers - radii[:, None]) & (grid <= centers + radii[:, None])).all(1)
            delta = means - x
            power = -.5 * (inverse[:, :3] * delta.square()).sum(1) - (
                inverse[:, 3:] * delta[:, [0, 1, 0]] * delta[:, [1, 2, 2]]).sum(1)
            distance = (-2 * power).clamp_min(0).sqrt()
            positive = box & (alpha * power.exp() > 0)
            same = classes == y[t]
            expected = [int((positive & (distance <= r)).sum()) for r in (1, 2, 3)]
            expected += [int((positive & (distance <= r) & same).sum()) for r in (1, 2, 3)]
            expected += [expected[3], int(positive.sum())]
            np.testing.assert_array_equal(counts[t].cpu().numpy(), expected)
    return dict(confusion=cm.cpu().numpy(), class_coverage=cc.cpu().numpy(),
        load_sixth=load.cpu().numpy().astype(np.int32),
        load_fifth=load5.cpu().numpy().astype(np.int32), query_core=core_np,
        core_top20_union=np.array(int(top_core_union.sum()), np.int64))


def history(size, scene):
    if size == 'large':
        return None
    folder = ROOT / 'work_dirs' / ('query_budget_full' if size == 'tiny' else 'small_gaussian_diagnosis_full')
    meta = read(folder / 'scenes/val' / f'{scene}.json')
    with np.load(folder / 'scenes/val' / f'{scene}.npz') as z:
        fields = read(folder / 'protocol.json')['fields']
        load = z['metrics'][..., fields.index('gt_foreground')]
        core = z['metrics'][..., fields.index('core_coverage_all_fg')]
        strata = z['strata'].sum((0, 2, 3))
        cm = z['confusion'] if size == 'small' else None
    final = None
    if size == 'small':
        p = ROOT / 'work_dirs/small_final_query_coverage_full/scenes/val' / f'{scene}.npz'
        with np.load(p) as z:
            final = z['load_sixth'].copy()
    return dict(meta=meta, load=load, core=core, class_gt=strata[:, 0],
                class_covered=strata[:, 0] - strata[:, 1], confusion=cm, final=final)


def pilot(size, cfg, net, protocol):
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    options = deepcopy(cfg.val_dataset_config)
    options.return_keys = list(options.return_keys) + ['history_frame_indices']
    native = OPENOCC_DATASET.build(options)
    dataset = EvaluationFrames(cfg, 'val')
    a = to_cuda(custom_collate_fn_temporal([native[0]]))
    b = to_cuda(custom_collate_fn_temporal([dataset[0]]))
    for key in a:
        torch.testing.assert_close(a[key][:, :6] if key == 'img' else a[key], b[key], rtol=0, atol=0)
    with torch.inference_mode():
        original = forward(net, a)
        original_logits = original['pred_occ'][-1].clone()
        original_gaussian = original['gaussian']
        del original, a
        cache = FeatureCache(net, dataset)
        try:
            out = forward(net, b)
            torch.testing.assert_close(original_logits, out['pred_occ'][-1], rtol=0, atol=0)
            for key in out['gaussian']._fields:
                torch.testing.assert_close(getattr(out['gaussian'], key), getattr(original_gaussian, key), rtol=0, atol=0)
            row = measure(net, out, b, protocol['queries'], protocol['children'], protocol['previous_children'], True)
            previous = history(size, dataset.keyframes[0][0])
            if previous is not None:
                np.testing.assert_array_equal(row['load_fifth'], previous['load'][0])
                np.testing.assert_array_equal(row['query_core'], previous['core'][0])
                if previous['final'] is not None:
                    np.testing.assert_array_equal(row['load_sixth'], previous['final'][0])
        finally:
            cache.close()
    write(RUN / size / 'pilot.json', dict(status='passed', checkpoint_sha256=protocol['checkpoint_sha256'],
        native_images_projections_logits_gaussians_exact=True,
        independent_query_bitsets_exact=True, exhaustive_GT_positions=16,
        historical_query_arrays_exact=previous is not None))
    net.flush_memory_phase()


def evaluate(size, workers):
    from dataset.utils import custom_collate_fn_temporal
    cfg, protocol = initialize(size)
    folder = RUN / size
    if (folder / 'complete.json').exists():
        assert read(folder / 'complete.json')['checkpoint_sha256'] == protocol['checkpoint_sha256']
        return
    net = build(cfg, size)
    pilot(size, cfg, net, protocol)
    dataset = EvaluationFrames(cfg, 'val')
    assert len(dataset) == 6019
    scenes = {}
    for i, (scene, _) in enumerate(dataset.keyframes):
        scenes.setdefault(scene, []).append(i)
    assert len(scenes) == 150
    pending = []
    for scene, ids in scenes.items():
        path = folder / 'scenes' / f'{scene}.json'
        if path.exists():
            meta = read(path)
            assert meta['indices'] == ids and meta['checkpoint_sha256'] == protocol['checkpoint_sha256']
            assert path.with_suffix('.npz').exists()
        else:
            pending += ids
    loader = DataLoader(Subset(dataset, pending), batch_size=1, num_workers=workers,
        shuffle=False, collate_fn=custom_collate_fn_temporal, pin_memory=True,
        persistent_workers=workers > 0, **(dict(prefetch_factor=2) if workers else {}))
    cache = FeatureCache(net, dataset)
    started = time.monotonic()
    done = 0
    active = None
    try:
        for index, batch in zip(pending, loader):
            scene, f = dataset.keyframes[index]
            if active is None:
                active = scene
                arrays, indices, tokens = {}, [], []
                previous = history(size, scene)
            assert active == scene
            batch = to_cuda(batch)
            with torch.inference_mode():
                out = forward(net, batch)
                row = measure(net, out, batch, protocol['queries'], protocol['children'], protocol['previous_children'])
            if previous is not None:
                j = len(indices)
                np.testing.assert_array_equal(row['load_fifth'], previous['load'][j])
                np.testing.assert_array_equal(row['query_core'], previous['core'][j])
                if previous['final'] is not None:
                    np.testing.assert_array_equal(row['load_sixth'], previous['final'][j])
            for key, value in row.items():
                arrays.setdefault(key, []).append(value)
            indices.append(index)
            tokens.append(dataset.base.scene_infos[scene][f]['token'])
            done += 1
            if index == scenes[scene][-1]:
                result = {key: np.stack(value) for key, value in arrays.items()}
                if previous is not None:
                    cc = result['class_coverage'].sum(0)
                    np.testing.assert_array_equal(cc[:, 0], previous['class_gt'])
                    np.testing.assert_array_equal(cc[:, 1], previous['class_covered'])
                    if previous['confusion'] is not None:
                        np.testing.assert_array_equal(result['confusion'].sum(0), previous['confusion'])
                path = folder / 'scenes' / f'{scene}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, **result)
                path.with_suffix('.tmp').replace(path)
                write(path.with_suffix('.json'), dict(status='complete', indices=indices, tokens=tokens,
                    scene=scene, frames=len(indices), checkpoint_sha256=protocol['checkpoint_sha256']))
                nscene = len(list((folder / 'scenes').glob('*.json')))
                status = dict(status='running', model=size, frames=6019-len(pending)+done,
                    scenes=nscene, expected_frames=6019, expected_scenes=150,
                    diagnostic_frames_per_second=done/(time.monotonic()-started))
                write(folder / 'progress.json', status)
                print(json.dumps(status), flush=True)
                active = None
            del row, out, batch
        paths_all = list((folder / 'scenes').glob('*.json'))
        ids = [i for path in paths_all for i in read(path)['indices']]
        assert len(paths_all) == 150 and sorted(ids) == list(range(6019))
        for rel, digest in protocol['source_sha256'].items():
            assert sha(ROOT / rel) == digest, rel
        status = dict(status='complete', model=size, frames=6019, scenes=150,
            checkpoint_sha256=protocol['checkpoint_sha256'], queries=protocol['queries'],
            children=protocol['children'], seconds=time.monotonic()-started,
            class_columns=CLASS_COLUMNS, historical_checks='Every frame and scene checked where available')
        write(folder / 'complete.json', status)
        write(folder / 'progress.json', status)
    finally:
        cache.close()
        net.flush_memory_phase()
        del net, cache, loader, dataset
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    with (RUN / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for size in args.models:
                write(RUN / 'status.json', dict(status='running', phase=size, pid=os.getpid()))
                evaluate(size, args.workers)
            write(RUN / 'status.json', dict(status='evaluation_complete', models=args.models))
        except BaseException as exc:
            write(RUN / 'status.json', dict(status='failed', error=repr(exc), pid=os.getpid()))
            raise


if __name__ == '__main__':
    main()
