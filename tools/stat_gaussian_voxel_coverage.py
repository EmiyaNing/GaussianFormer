"""Full train/val census: how many Gaussians cover missed/detected foreground."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.stat_boundary_gaussians import CONFIG, CHECKPOINT, build, forward, to_cuda, read, write, sha
from tools.probe_gaussian_scale_supervision import layout
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.gaussian_voxel_coverage import extension, COUNT_NAMES

RUN = ROOT / 'work_dirs/gaussian_voxel_coverage_full'
OLD = ROOT / 'work_dirs/gaussian_scale_supervision_full'
GROUPS = ('miss', 'detected_occupied')
COUNT_CAP = 1024
FIELDS = (*COUNT_NAMES, 'nearest_mahalanobis', 'nearest_euclidean',
          'dominant_mahalanobis', 'dominant_euclidean', 'kernel_weighted_mahalanobis',
          'contribution_weighted_mahalanobis', 'contribution_weighted_euclidean',
          'effective_contributors', 'kernel_mass', 'opacity_mass', 'max_foreground',
          'empty_score', 'foreground_empty_ratio', 'semantic_amplitude', 'class_concentration',
          'dominant_share', 'share_0to1', 'share_1to2', 'share_2to3', 'share_over3')
FLAGS = ('no_native_support', 'no_1sigma', 'no_2sigma', 'no_3sigma',
         'no_positive_winning_class', 'no_1pct_contributor', 'no_10pct_contributor',
         'dominant_center_empty', 'dominant_center_visible_empty', 'majority_evidence_over3sigma')
HISTOGRAMS = {
    'nearest_mahalanobis': np.linspace(0, 15, 151),
    'dominant_mahalanobis': np.linspace(0, 15, 151),
    'kernel_weighted_mahalanobis': np.linspace(0, 15, 151),
    'contribution_weighted_mahalanobis': np.linspace(0, 15, 151),
    'effective_contributors': np.linspace(0, 100, 501),
    'dominant_share': np.linspace(0, 1.000001, 101),
}


def initialize():
    cfg = Config.fromfile(CONFIG)
    previous = read(OLD / 'protocol.json')
    assert sha(CHECKPOINT) == previous['checkpoint_sha256']
    assert read(OLD / 'complete.json')['status'] == 'complete'
    files = [Path(__file__), *sorted((ROOT / 'tools/gaussian_voxel_coverage').glob('*.*'))]
    base = read(ROOT / 'work_dirs/boundary_gaussian_full/protocol.json')
    for file, digest in base['source_sha256'].items():
        assert sha(ROOT / file) == digest, file
    annotations = {s: sha(cfg[s + '_dataset_config'].imageset) for s in ('train', 'val')}
    assert annotations == base['annotation_sha256']
    protocol = dict(version='voxel_coverage_v1', config=str(CONFIG.relative_to(ROOT)),
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=sha(CHECKPOINT),
        expected=previous['expected'], annotation_sha256=annotations,
        groups=GROUPS, count_names=COUNT_NAMES, fields=FIELDS, flags=FLAGS,
        count_histogram_cap=COUNT_CAP,
        float_histogram_edges={k:v.tolist() for k,v in HISTOGRAMS.items()},
        native_support='Production PyTorch integer centers/radii; alpha*exp(power)>0 in actual FP32',
        targets='Every valid GT foreground voxel within camera mask; detected means any predicted nonempty class, regardless of semantic correctness',
        radial='Mahalanobis distance sqrt(-2*power); <=1, (1,2], (2,3], >3, within native support',
        contribution='Individual contribution to highest rendered foreground logit, including on misses; no GT class used to choose it',
        effective_contributors='1/sum((individual contribution / native maximum foreground logit)^2), defined only for positive accumulated evidence',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        inherited_source_sha256=base['source_sha256'],
        seed=928, preprocessing='Frozen eval, identical deterministic history feature cache to previous full census',
        sampling='None: all 34,149 labeled train/val keyframes')
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == read_protocol_json(protocol), 'Source/protocol changed'
    else:
        write(RUN / 'protocol.json', protocol)
        cfg.dump(RUN / 'resolved_config.py')
        import shutil
        for p in files:
            dst = RUN / 'sources' / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
    return cfg, protocol


def read_protocol_json(value):
    import json
    return json.loads(json.dumps(value))


def measure(network, out, batch, subset=None):
    head = network.head
    _, values = layout(head, out['gaussian'])
    means, scales, inverse, alpha, semantic, centers, radii = values
    if subset is not None:
        means, scales, inverse, alpha, semantic, centers, radii = [v[subset].contiguous() for v in values]
    agg = head.aggregator
    xyz = out['sampled_xyz'][0].contiguous()
    labels = batch['occ_label'].flatten().long()
    camera = batch['occ_cam_mask'].flatten().bool()
    targets = torch.where(camera & (labels >= 0) & (labels < 17))[0]
    lookup = torch.full((len(labels),), -1, device=labels.device, dtype=torch.int32)
    lookup[targets] = torch.arange(len(targets), device=labels.device, dtype=torch.int32)
    z = out['pred_occ'][-1][0, :, targets]
    maximum, winner = z[:17].max(0)
    result = extension().coverage(means, inverse, alpha, semantic, centers, radii,
        xyz, lookup, winner.int().contiguous(), maximum.contiguous(), agg.H, agg.W, agg.D)
    info = dict(targets=targets, xyz=xyz, labels=labels, camera=camera,
        maximum=maximum, total=z[:17].sum(0), empty=z[17],
        pred=out['final_occ'].flatten()[targets], winner=winner,
        means=means, inverse=inverse, alpha=alpha, semantic=semantic, centers=centers, radii=radii)
    return result, info


def derived(result, info, shape):
    counts, mass, nearest, packed = result
    owner = 0xffffffff - (packed & 0xffffffff)
    owned = packed > 0
    safe_owner = owner.clamp(0, len(info['means']) - 1)
    delta = info['means'][safe_owner] - info['xyz'][info['targets']]
    inv = info['inverse'][safe_owner]
    power = -.5 * (inv[:, :3] * delta.square()).sum(1) - (
        inv[:, 3:] * delta[:, [0, 1, 0]] * delta[:, [1, 2, 2]]).sum(1)
    d = (-2 * power).clamp_min(0).sqrt()
    d = d.masked_fill(~owned, float('nan'))
    euclidean = delta.square().sum(1).sqrt().masked_fill(~owned, float('nan'))
    center = info['centers'][safe_owner].long()
    flat = (center[:, 0] * shape[1] + center[:, 1]) * shape[2] + center[:, 2]
    center_empty = owned & (info['labels'][flat] == 17)
    center_visible_empty = center_empty & info['camera'][flat]
    positive = mass[:, 2] > 0
    ratio = lambda x, y: (x.double() / y.double()).masked_fill(y <= 0, float('nan'))
    shares = ratio(mass[:, 7:11], mass[:, 2, None])
    effective = mass[:, 3].reciprocal().masked_fill(~positive | (mass[:, 3] <= 0), float('nan'))
    dominant = (packed >> 32).int().view(torch.float32)
    fields = torch.cat((counts.float(), nearest, d[:, None], euclidean[:, None],
        ratio(mass[:, 4], mass[:, 0])[:, None],
        ratio(mass[:, 5], mass[:, 2])[:, None], ratio(mass[:, 6], mass[:, 2])[:, None],
        effective[:, None], mass[:, :2], info['maximum'][:, None], info['empty'][:, None],
        ratio(info['maximum'], info['empty'])[:, None],
        ratio(info['total'], mass[:, 1])[:, None], ratio(info['maximum'], info['total'])[:, None],
        ratio(dominant, mass[:, 2])[:, None], shares), 1)
    assert fields.shape[1] == len(FIELDS), fields.shape
    flags = torch.stack((counts[:, 0] == 0, counts[:, 1] == 0, counts[:, 2] == 0, counts[:, 3] == 0,
        counts[:, 4] == 0, counts[:, 5] == 0, counts[:, 6] == 0,
        center_empty, center_visible_empty, shares[:, 3] >= .5), 1)
    return fields, flags


def accumulator(device='cuda'):
    return dict(voxels=torch.zeros(2, dtype=torch.int64, device=device),
        count_hist=torch.zeros((2, len(COUNT_NAMES), COUNT_CAP + 2), dtype=torch.int64, device=device),
        count_max=torch.zeros((2, len(COUNT_NAMES)), dtype=torch.int64, device=device),
        field_n=torch.zeros((2, len(FIELDS)), dtype=torch.int64, device=device),
        field_sum=torch.zeros((2, len(FIELDS)), dtype=torch.float64, device=device),
        flags=torch.zeros((2, len(FLAGS)), dtype=torch.int64, device=device),
        class_range_voxels=torch.zeros((2, 17, 4), dtype=torch.int64, device=device),
        class_range_n=torch.zeros((2, 17, 4, len(FIELDS)), dtype=torch.int64, device=device),
        class_range_sum=torch.zeros((2, 17, 4, len(FIELDS)), dtype=torch.float64, device=device),
        count_distance_joint=torch.zeros((2, 8, 5), dtype=torch.int64, device=device),
        **{'hist_' + k: torch.zeros((2, len(v) + 1), dtype=torch.int64, device=device) for k, v in HISTOGRAMS.items()})


def collect(acc, result, info, shape):
    counts, mass, nearest, packed = result
    fields, flags = derived(result, info, shape)
    groups = (info['pred'] != 17).long()
    labels = info['labels'][info['targets']]
    ranges = torch.bucketize(info['xyz'][info['targets'], :2].square().sum(1).sqrt(),
                            torch.tensor([10., 20., 30.], device=labels.device), right=True)
    strata = (groups * 17 + labels) * 4 + ranges
    acc['voxels'] += torch.bincount(groups, minlength=2)
    bins = counts.long().clamp_max(COUNT_CAP + 1)
    keys = ((groups[:, None] * len(COUNT_NAMES) + torch.arange(len(COUNT_NAMES), device=groups.device)) * (COUNT_CAP + 2) + bins)
    acc['count_hist'].view(-1).add_(torch.bincount(keys.flatten(), minlength=acc['count_hist'].numel()))
    finite = torch.isfinite(fields)
    acc['field_n'].index_add_(0, groups, finite.long())
    acc['field_sum'].index_add_(0, groups, fields.masked_fill(~finite, 0).double())
    acc['flags'].index_add_(0, groups, flags.long())
    acc['class_range_voxels'].view(-1).add_(torch.bincount(strata, minlength=2*17*4))
    acc['class_range_n'].view(2*17*4, -1).index_add_(0, strata, finite.long())
    acc['class_range_sum'].view(2*17*4, -1).index_add_(0, strata, fields.masked_fill(~finite, 0).double())
    for i in range(2):
        if (groups == i).any():
            acc['count_max'][i] = torch.maximum(acc['count_max'][i], counts[groups == i].max(0).values.long())
    for key, edges in HISTOGRAMS.items():
        value = fields[:, FIELDS.index(key)]
        valid = torch.isfinite(value)
        bins = torch.bucketize(value[valid].contiguous(), torch.tensor(edges, device=groups.device), right=True)
        index = groups[valid] * (len(edges) + 1) + bins
        acc['hist_' + key].view(-1).add_(torch.bincount(index, minlength=acc['hist_' + key].numel()))
    cb = torch.bucketize(counts[:, 2].contiguous(), torch.tensor([1, 2, 4, 8, 16, 32, 64], device=groups.device), right=True)
    db = torch.bucketize(fields[:, FIELDS.index('dominant_mahalanobis')].contiguous(),
                        torch.tensor([1., 2., 3.], device=groups.device), right=False)
    db[packed == 0] = 4
    keys = (groups * 8 + cb) * 5 + db
    acc['count_distance_joint'].view(-1).add_(torch.bincount(keys, minlength=2*8*5))
    target = info['maximum']
    active = target > 1e-8
    relative = ((mass[:, 2] - target).abs() / target.clamp_min(1e-8))[active]
    assert not ((counts[:, 1] > counts[:, 2]) | (counts[:, 2] > counts[:, 3]) | (counts[:, 3] > counts[:, 0])).any()
    assert torch.isfinite(mass).all()
    return dict(voxels=torch.bincount(groups, minlength=2).cpu().tolist(),
        maximum_winner_mass_relative_error=float(relative.max()) if len(relative) else 0.,
        zero_native_support=int((counts[:, 0] == 0).sum()),
        zero_winning_contribution=int((counts[:, 4] == 0).sum()))


def pilot():
    from tools.gaussian_voxel_coverage.audit import synthetic_audit, reference_audit
    from tools.stat_gaussian_support_regions import native_mass
    from dataset.utils import custom_collate_fn_temporal
    cfg, protocol = initialize()
    synthetic = synthetic_audit()
    network = build(cfg)
    audits = []
    for split in ('val', 'train'):
        dataset = EvaluationFrames(cfg, split)
        cache = FeatureCache(network, dataset)
        for index in (0, 7):
            batch = to_cuda(custom_collate_fn_temporal([dataset[index]]))
            with torch.inference_mode():
                out = forward(network, batch)
                result, info = measure(network, out, batch)
                shape = (network.head.aggregator.H, network.head.aggregator.W, network.head.aggregator.D)
                acc = accumulator()
                audit = collect(acc, result, info, shape)
                target = info['targets']
                production = native_mass(network.head, out)
                torch.testing.assert_close(result[1][:, :2].T, production[:, target], rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(result[1][:, 2], info['maximum'], rtol=1e-4, atol=1e-5)
                small, si = measure(network, out, batch, torch.arange(64, device='cuda'))
                reference_audit(small, si, shape)
                old = read(OLD / 'scenes' / split / (dataset.keyframes[index][0] + '.json'))
                record = next(r for r in old['records'] if r['index'] == index)
                confusion = np.asarray(record['confusion'][0])
                expected = [int(confusion[:17, 17].sum()), int(confusion[:17, :17].sum())]
                assert audit['voxels'] == expected
                audits.append(dict(split=split, index=index, all_gaussians=76800,
                    native_renderer_mass_parity=True, independent_64_gaussian_reference=True,
                    previous_frame_outcomes_exact=True, **audit))
            del batch, out, result, info, small, si, acc, production
        cache.close()
    write(RUN / 'pilot.json', dict(status='passed', synthetic=synthetic, audits=audits))
    print('Pilot passed: independent synthetic/reference and four real frames.', flush=True)


def run(workers):
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Subset
    cfg, protocol = initialize()
    assert read(RUN / 'pilot.json')['status'] == 'passed'
    network = build(cfg)
    shape = (network.head.aggregator.H, network.head.aggregator.W, network.head.aggregator.D)
    start = time.perf_counter()
    processed = completed = 0
    for split in ('val', 'train'):
        dataset = EvaluationFrames(cfg, split)
        by_scene = {}
        for index, (scene, frame) in enumerate(dataset.keyframes):
            by_scene.setdefault(scene, []).append(index)
        pending = []
        for scene, ids in by_scene.items():
            path = RUN / 'scenes' / split / (scene + '.json')
            if path.exists():
                prior = read(path)
                assert prior['indices'] == ids and prior['checkpoint_sha256'] == protocol['checkpoint_sha256']
                assert path.with_suffix('.npz').exists()
                completed += 1
            else:
                pending.extend(ids)
        loader = DataLoader(Subset(dataset, pending), batch_size=1, shuffle=False, num_workers=workers,
            collate_fn=custom_collate_fn_temporal, pin_memory=True,
            **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))
        cache = FeatureCache(network, dataset)
        active = None
        for index, batch in zip(pending, loader):
            scene, frame = dataset.keyframes[index]
            if active is None:
                active = scene
                acc = accumulator()
                records = []
                old = read(OLD / 'scenes' / split / (scene + '.json'))
                old_records = {r['index']: r for r in old['records']}
            assert active == scene
            batch = to_cuda(batch)
            with torch.inference_mode():
                out = forward(network, batch)
                result, info = measure(network, out, batch)
                audit = collect(acc, result, info, shape)
                confusion = np.asarray(old_records[index]['confusion'][0])
                expected = [int(confusion[:17, 17].sum()), int(confusion[:17, :17].sum())]
                assert audit['voxels'] == expected, (split, index)
                assert audit['maximum_winner_mass_relative_error'] < 5e-4
            records.append(dict(index=index, scene_frame=int(frame),
                sample_token=old_records[index]['sample_token'], **audit))
            processed += 1
            if len(records) == len(by_scene[scene]):
                path = RUN / 'scenes' / split / (scene + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                arrays = {k:v.cpu().numpy() for k,v in acc.items()}
                with path.with_suffix('.npz.tmp').open('wb') as f:
                    np.savez_compressed(f, **arrays)
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                write(path, dict(status='complete', split=split, scene_token=scene,
                    scene_name=old['scene_name'], indices=by_scene[scene], frames=len(records),
                    records=records, checkpoint_sha256=protocol['checkpoint_sha256']))
                completed += 1
                elapsed = time.perf_counter() - start
                write(RUN / 'progress.json', dict(status='running', completed_scenes=completed,
                    total_scenes=850, split=split, last_scene=old['scene_name'],
                    frames_this_process=processed, seconds=elapsed, frames_per_second=processed/elapsed,
                    updated_at=datetime.now().astimezone().isoformat()))
                if completed % 20 == 0:
                    print(f'{completed}/850 scenes; {processed} frames; {processed/elapsed:.2f} frames/s', flush=True)
                active = None
                del acc, arrays
            del batch, out, result, info
        assert active is None
        cache.close()
        del loader, dataset, cache
    elapsed = time.perf_counter() - start
    write(RUN / 'execution_complete.json', dict(status='complete', frames_this_process=processed,
        scenes=completed, seconds=elapsed, checkpoint_sha256=protocol['checkpoint_sha256']))
    write(RUN / 'progress.json', dict(status='complete', completed_scenes=completed,
        total_scenes=850, frames_this_process=processed, seconds=elapsed))
    print('Full voxel coverage census complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=['pilot', 'run'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    pilot() if args.action == 'pilot' else run(args.workers)
