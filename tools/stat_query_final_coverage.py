"""Full small-model census: fifth/sixth nearest ownership vs final GS support.

Stores exact GT unions of query groups using three query bitsets per voxel.
Every old fifth-level load and final 1sigma per-query coverage must reproduce
the historical complete census before a frame can enter the new statistics.
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
from tools.stat_boundary_gaussians import forward, to_cuda, read, write, sha
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.probe_gaussian_scale_supervision import layout
from tools.stat_query_budget import FIELDS
from tools.query_final_support import extension

RUN = ROOT / 'work_dirs/small_final_query_coverage_full'
OLD = ROOT / 'work_dirs/small_gaussian_diagnosis_full'
Q, CHILDREN, COARSE = 1200, 64, 32
RANGES = ('core_1sigma', 'strict_3sigma', 'native_support')


def top20(values):
    # Match the original all-query top20 convention, including tied indices.
    return np.argsort(values, kind='stable')[-Q // 5:]


def packed_mask(ids, device):
    words = np.zeros((Q + 63) // 64, np.uint64)
    for q in ids:
        words[int(q) // 64] |= np.uint64(1) << np.uint64(int(q) % 64)
    return torch.from_numpy(words.view(np.int64)).to(device)


def frame(network, out, batch, previous):
    from loss.opus_set_loss import _nearest_indices
    _, values = layout(network.head, out['gaussian'])
    means, scales, inv, alpha, semantic, centers, radii = values
    assert len(means) == Q * CHILDREN
    xyz = out['sampled_xyz'][0].contiguous()
    labels = batch['occ_label'].flatten().long()
    valid = batch['occ_cam_mask'].flatten().bool() & (labels >= 0) & (labels < 17)
    targets = torch.where(valid)[0]
    gt = xyz[targets]
    coarse = out['gaussians'][-2].means[0].contiguous()
    assert len(coarse) == Q * COARSE
    q5 = _nearest_indices(gt, coarse, 1024) // COARSE
    q6 = _nearest_indices(gt, means, 1024) // CHILDREN
    loads = [torch.bincount(q, minlength=Q).cpu().numpy() for q in (q5, q6)]
    np.testing.assert_array_equal(loads[0], previous[:, FIELDS.index('gt_foreground')])
    lookup = torch.full((len(labels),), -1, device=xyz.device, dtype=torch.int32)
    lookup[targets] = torch.arange(len(gt), device=xyz.device, dtype=torch.int32)
    agg = network.head.aggregator
    bits, coverage, distinct = extension().support(means, inv, alpha, centers, radii, xyz,
        lookup, len(gt), Q, CHILDREN, agg.H, agg.W, agg.D)
    coverage_np = coverage.cpu().numpy()
    np.testing.assert_array_equal(coverage_np[0], previous[:, FIELDS.index('core_coverage_all_fg')])
    core_holes = int((distinct[0] == 0).sum())
    assert core_holes == int(previous[:, FIELDS.index('gt_no_core')].sum())
    assert bool((distinct[0] <= distinct[1]).all()) and bool((distinct[1] <= distinct[2]).all())
    arrays = dict(load_fifth=loads[0], load_sixth=loads[1],
        dominant_foreground=previous[:, FIELDS.index('dominant_fg_voxels')].astype(np.int64))
    outrow = dict(foreground=len(gt), nearest={}, supports={})
    for i, name in enumerate(('fifth', 'sixth')):
        assigned = (q5, q6)[i]
        own = ((bits[0, torch.arange(len(gt), device=xyz.device), assigned // 64]
                >> (assigned % 64)) & 1).bool()
        ids = top20(loads[i])
        outrow['nearest'][name] = dict(top20_foreground=int(loads[i][ids].sum()),
            foreground=int(loads[i].sum()), zero_assigned_queries=int((loads[i] == 0).sum()),
            own_core_foreground=int(own.sum()))
    dom = arrays['dominant_foreground']
    outrow['dominant_foreground'] = dict(attributed=int(dom.sum()),
        top20_attributed=int(dom[top20(dom)].sum()))
    for i, name in enumerate(RANGES):
        covered = distinct[i] > 0
        arrays[name + '_query_coverage'] = coverage_np[i]
        assert int(distinct[i].sum()) == int(coverage_np[i].sum())
        groups = dict(fifth_top20=top20(loads[0]), sixth_top20=top20(loads[1]),
            coverage_top20=top20(coverage_np[i]))
        data = dict(all_covered=int(covered.sum()),
            query_voxel_pairs=int(coverage_np[i].sum()), zero_contact_queries=int((coverage_np[i] == 0).sum()),
            groups={})
        for group, ids in groups.items():
            mask = packed_mask(ids, xyz.device)
            hit_top = (bits[i] & mask).ne(0).any(1)
            hit_rest = (bits[i] & ~mask).ne(0).any(1)
            assert torch.equal(hit_top | hit_rest, covered)
            both = int((hit_top & hit_rest).sum())
            union_top, union_rest = int(hit_top.sum()), int(hit_rest.sum())
            assert union_top + union_rest - both == data['all_covered']
            data['groups'][group] = dict(top_union=union_top, remaining_union=union_rest,
                both_groups=both, top_only=union_top - both, remaining_only=union_rest - both,
                top_query_voxel_pairs=int(coverage_np[i, ids].sum()),
                assigned_fifth_foreground=int(loads[0][ids].sum()),
                assigned_sixth_foreground=int(loads[1][ids].sum()))
        outrow['supports'][name] = data
    return outrow, arrays


def initialize(cfg):
    expected = read(OLD / 'protocol.json')['expected']
    files = [Path(__file__), ROOT / 'tools/query_final_support/__init__.py',
        ROOT / 'tools/query_final_support/kernels.cu', small.CONFIG,
        ROOT / 'tools/boundary_gaussian_stats/data.py', ROOT / 'tools/probe_gaussian_scale_supervision.py']
    protocol = dict(version=1, checkpoint=str(small.CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=sha(small.CHECKPOINT), config=str(small.CONFIG.relative_to(ROOT)),
        expected=expected, queries=Q, children=CHILDREN, coarse_children=COARSE,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        historical_protocol_sha256=sha(OLD / 'protocol.json'),
        annotation_sha256={s: sha(cfg[s + '_dataset_config'].imageset) for s in ('val', 'train')},
        top20='Exactly 240 per frame; stable ascending sort, last 240, same historical tie convention',
        primary='Full visible GT foreground; final-level native support and Mahalanobis 1/3sigma; class agnostic',
        nearest='Each GT chooses nearest fifth or sixth center, then parent query; does not require actual support',
        coverage_ranking='Rank each query by unique GT voxels covered by its final children; no semantic condition',
        union='Deduplicate within each query and across selected queries; preserve overlap with remaining queries',
        dominant='Historical highest individual contribution to the winning foreground-class score, even if final is empty',
        no_subsampling=True, seed=928, eval_mode=True)
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol
    else:
        write(RUN / 'protocol.json', protocol)
        cfg.dump(RUN / 'resolved_config.py')
        for p in files:
            dest = RUN / 'sources' / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    return protocol


def pilot(cfg, network):
    from dataset.utils import custom_collate_fn_temporal
    from tools.query_budget_stats import extension as old_extension
    rows = []
    for split in ('val', 'train'):
        dataset = EvaluationFrames(cfg, split)
        scene, f = dataset.keyframes[0]
        meta = read(OLD / 'scenes' / split / (scene + '.json'))
        with np.load(OLD / 'scenes' / split / (scene + '.npz')) as z:
            previous = z['metrics'][meta['indices'].index(0)]
        cache = FeatureCache(network, dataset)
        batch = to_cuda(custom_collate_fn_temporal([dataset[0]]))
        with torch.inference_mode():
            out = forward(network, batch)
            row, arrays = frame(network, out, batch, previous)
            # Independent historical bitset equality and direct 1sigma checks.
            _, val = layout(network.head, out['gaussian'])
            means, scales, inv, alpha, sem, centers, radii = val
            xyz = out['sampled_xyz'][0].contiguous()
            labels = batch['occ_label'].flatten().int().contiguous()
            camera = batch['occ_cam_mask'].flatten().bool().contiguous()
            targets = torch.where(camera & (labels >= 0) & (labels < 17))[0]
            lookup = torch.full_like(labels, -1)
            lookup[targets] = torch.arange(len(targets), device=xyz.device, dtype=torch.int32)
            winner = out['pred_occ'][-1][0, :17].argmax(0).int().contiguous()
            dummy = torch.zeros(len(targets), device=xyz.device, dtype=torch.int32)
            agg = network.head.aggregator
            old_bits = old_extension().census(means, inv, alpha, sem, centers, radii, xyz, labels,
                camera, lookup, winner, dummy, Q, CHILDREN, agg.H, agg.W, agg.D)[1]
            new_bits, _, _ = extension().support(means, inv, alpha, centers, radii, xyz, lookup,
                len(targets), Q, CHILDREN, agg.H, agg.W, agg.D)
            assert torch.equal(old_bits, new_bits[0]), '1sigma bitsets differ from audited original'
            # CPU direct check for all query membership at 16 GT positions.
            from model.utils.utils import get_rotation_matrix
            rng = np.random.default_rng(928)
            for t in rng.choice(len(targets), min(16, len(targets)), replace=False):
                x = xyz[targets[t]]
                center = ((x - agg.pc_min[0]) / agg.grid_size).long()
                in_cube = ((center[None] >= centers.long() - radii[:, None]) &
                           (center[None] <= centers.long() + radii[:, None])).all(1)
                d = means - x
                power = -.5 * (inv[:, :3] * d.square()).sum(1) - (inv[:, 3:] * d[:, [0, 1, 0]] * d[:, [1, 2, 2]]).sum(1)
                positive = in_cube & (alpha * power.exp() > 0)
                for si, keep in enumerate((positive & ((-2 * power).clamp_min(0).sqrt() <= 1),
                                          positive & ((-2 * power).clamp_min(0).sqrt() <= 3), positive)):
                    expected = keep.reshape(Q, CHILDREN).any(1).cpu().numpy()
                    word = new_bits[si, t].cpu().numpy().view(np.uint64)
                    actual = np.array([bool((word[q // 64] >> np.uint64(q % 64)) & np.uint64(1)) for q in range(Q)])
                    np.testing.assert_array_equal(actual, expected)
        rows.append(dict(split=split, foreground=row['foreground'], old_query_counts_exact=True,
            original_1sigma_bitsets_exact=True, direct_membership_checks=16 * 3))
        cache.close()
        del out, batch, cache, dataset
    write(RUN / 'pilot.json', dict(status='passed', records=rows))


def run(cfg, network, protocol, workers):
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Subset
    start, processed = time.monotonic(), 0
    for split in ('val', 'train'):
        dataset = EvaluationFrames(cfg, split)
        by_scene = {}
        for i, (scene, _) in enumerate(dataset.keyframes):
            by_scene.setdefault(scene, []).append(i)
        assert len(dataset) == protocol['expected'][split]['frames']
        pending = []
        for scene, ids in by_scene.items():
            path = RUN / 'scenes' / split / (scene + '.json')
            if path.exists():
                saved = read(path)
                assert saved['indices'] == ids and saved['checkpoint_sha256'] == protocol['checkpoint_sha256']
                assert path.with_suffix('.npz').exists()
            else:
                pending.extend(ids)
        loader = DataLoader(Subset(dataset, pending), batch_size=1, shuffle=False, num_workers=workers,
            collate_fn=custom_collate_fn_temporal, pin_memory=True,
            **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))
        cache = FeatureCache(network, dataset)
        active = None
        for index, batch in zip(pending, loader):
            scene, scene_frame = dataset.keyframes[index]
            if active is None:
                active, records, arrays = scene, [], {}
                old_meta = read(OLD / 'scenes' / split / (scene + '.json'))
                with np.load(OLD / 'scenes' / split / (scene + '.npz')) as z:
                    old_metrics = z['metrics']
                assert old_meta['indices'] == by_scene[scene]
            assert active == scene
            batch = to_cuda(batch)
            with torch.inference_mode():
                out = forward(network, batch)
                row, arr = frame(network, out, batch, old_metrics[len(records)])
            token = dataset.base.scene_infos[scene][scene_frame]['token']
            assert old_meta['records'][len(records)]['sample_token'] == token
            records.append(dict(index=index, token=token, **row))
            for k, v in arr.items():
                arrays.setdefault(k, []).append(v)
            processed += 1
            if len(records) == len(by_scene[scene]):
                path = RUN / 'scenes' / split / (scene + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.npz.tmp').open('wb') as stream:
                    np.savez_compressed(stream, **{k: np.stack(v) for k, v in arrays.items()})
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                write(path, dict(status='complete', split=split, scene=scene,
                    indices=by_scene[scene], checkpoint_sha256=protocol['checkpoint_sha256'], records=records))
                done = len(list((RUN / 'scenes' / split).glob('*.json')))
                elapsed = time.monotonic() - start
                write(RUN / 'progress.json', dict(status='running', pid=os.getpid(), split=split,
                    completed_scenes_in_split=done, expected=protocol['expected'], processed=processed,
                    fps=processed / elapsed, seconds=elapsed, updated_at=datetime.now().astimezone().isoformat()))
                print(f'{split}: {done}/{len(by_scene)} scenes, {processed} frames, {processed/elapsed:.2f} frames/s', flush=True)
                active = None
            del out, batch, row, arr
        assert active is None
        cache.close()
        del loader, cache, dataset
        write(RUN / (split + '_complete.json'), dict(status='complete', **protocol['expected'][split]))
    write(RUN / 'progress.json', dict(status='complete', seconds=time.monotonic() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('pilot', 'run'), default='run')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = Config.fromfile(small.CONFIG)
        protocol = initialize(cfg)
        network = small.build(cfg)
        try:
            if not (RUN / 'pilot.json').exists():
                pilot(cfg, network)
            if args.action == 'run':
                run(cfg, network, protocol, args.workers)
        except BaseException as e:
            write(RUN / 'failure.json', dict(error=repr(e), time=datetime.now().astimezone().isoformat()))
            raise


if __name__ == '__main__':
    main()
