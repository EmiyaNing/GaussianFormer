"""Full frozen census of query load, Gaussian core coverage and redundancy.

python3.10 tools/stat_query_budget.py --action pilot
python3.10 tools/stat_query_budget.py --action run --workers 3
"""
import argparse
from datetime import datetime
import json
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
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.probe_gaussian_scale_supervision import layout
from tools.query_budget_stats import extension

RUN = ROOT / 'work_dirs/query_budget_full'
OLD = ROOT / 'work_dirs/gaussian_voxel_coverage_full'
CAUSAL = ROOT / 'work_dirs/gaussian_scale_supervision_full'
Q, CHILDREN, COARSE_CHILDREN = 600, 128, 64
FIELDS = (
    'gt_foreground', 'gt_no_core', 'gt_own_core', 'gt_neighbor_only_core',
    'gt_miss', 'gt_correct', 'gt_wrong_class', 'gt_no_core_correct', 'gt_no_core_detected',
    'gt_semantic_boundary', 'gt_empty_boundary', 'gt_components', 'gt_classes',
    'gt_coarse_distance_sum', 'gt_horizontal_range_sum', 'gt_core_count_sum',
    'core_coverage_all_fg', 'core_shared_other_query', 'core_exclusive',
    'core_gs_touches_fg', 'native_gs_touches_fg', 'gs_camera_native', 'gs_camera_core',
    'dominant_fg_voxels', 'dominant_fp_voxels',
    'amplitude_mean', 'opacity_mean', 'effective_amplitude_mean', 'scale_mean', 'scale_max_mean',
    'center_horizontal_range_mean', 'coarse_spread_mean',
    'gt_span_x', 'gt_span_y', 'gt_span_z',
    'no_core_correct_dom_n', 'no_core_correct_dom_amplitude_sum', 'no_core_correct_dom_opacity_sum',
    'core_correct_dom_n', 'core_correct_dom_amplitude_sum', 'core_correct_dom_opacity_sum',
    'dominant_fp_amplitude_sum', 'dominant_fp_opacity_sum',
    'no_core_correct_dom_mahalanobis_sum', 'core_correct_dom_mahalanobis_sum',
    'dom_no_core_correct_from_same_query',
)
STRAT_FIELDS = ('foreground', 'no_core', 'miss', 'no_core_correct', 'own_core')


class QueryFrames(EvaluationFrames):
    def __getitem__(self, index):
        from scipy.ndimage import label, generate_binary_structure
        row = super().__getitem__(index)
        y = np.asarray(row['occ_label'])
        camera = np.asarray(row['occ_cam_mask'], dtype=bool)
        fg = camera & (y >= 0) & (y < 17)
        valid = camera & (y >= 0) & (y < 18)
        components, _ = label(fg, structure=generate_binary_structure(3, 1))
        boundary = np.zeros_like(y, dtype=np.uint8)
        for axis in range(3):
            a = [slice(None)] * 3
            b = a.copy()
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            sem = valid[a] & valid[b] & fg[a] & fg[b] & (y[a] != y[b])
            edge = valid[a] & valid[b] & (fg[a] != fg[b])
            boundary[a] |= sem.astype(np.uint8) | (edge.astype(np.uint8) << 1)
            boundary[b] |= sem.astype(np.uint8) | (edge.astype(np.uint8) << 1)
        row['query_components'] = components.astype(np.int32)
        row['query_boundaries'] = boundary
        return row


def initialize():
    cfg = Config.fromfile(CONFIG)
    old = read(OLD / 'protocol.json')
    assert read(OLD / 'complete.json')['status'] == 'complete'
    assert sha(CHECKPOINT) == old['checkpoint_sha256']
    assert {s: sha(cfg[s + '_dataset_config'].imageset) for s in ('train', 'val')} == old['annotation_sha256']
    for file, digest in old['inherited_source_sha256'].items():
        assert sha(ROOT / file) == digest, file
    files = [Path(__file__), *sorted((ROOT / 'tools/query_budget_stats').glob('*.*'))]
    protocol = dict(version='query_budget_v1', config=str(CONFIG.relative_to(ROOT)),
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=old['checkpoint_sha256'],
        annotation_sha256=old['annotation_sha256'], expected=old['expected'],
        queries=Q, children=CHILDREN, coarse_children=COARSE_CHILDREN, fields=FIELDS,
        sampling='None: all 34,149 labeled train/val keyframes; frozen eval, camera mask',
        assignment='Each camera-valid foreground voxel chooses its nearest stage-5 Gaussian center by Euclidean distance; query=flat_index//64. No final semantic scores in assignment.',
        core='Positive FP32 opacity*kernel, native production candidate cube and Mahalanobis distance<=1; distinct query bitset, not sum of Gaussians',
        components='Number of global 6-connected camera-valid foreground components represented among the query assigned voxels; not object instances',
        boundaries='Six-neighbor valid foreground class transitions and foreground/empty transitions; unknown adjacency excluded',
        strata='Active-query load decile within each frame x GT class x horizontal range [0,10,20,30,inf] x coarse distance [0,.4,.8,1.6,inf]',
        stratum_fields=STRAT_FIELDS, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        baseline_protocol_sha256=sha(OLD / 'protocol.json'))
    protocol = json.loads(json.dumps(protocol))
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol, 'Source/protocol changed; use a new run directory'
    else:
        import shutil
        write(RUN / 'protocol.json', protocol)
        cfg.dump(RUN / 'resolved_config.py')
        for p in files:
            dst = RUN / 'sources' / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
    return cfg, protocol


def frame_statistics(network, out, batch, audit_reference=False):
    from loss.opus_set_loss import _nearest_indices
    _, values = layout(network.head, out['gaussian'])
    means, scales, inverse, alpha, semantic, centers, radii = values
    assert len(means) == Q * CHILDREN
    coarse = out['gaussians'][-2].means[0].contiguous()
    assert len(coarse) == Q * COARSE_CHILDREN
    xyz = out['sampled_xyz'][0].contiguous()
    labels = batch['occ_label'].flatten().long()
    camera = batch['occ_cam_mask'].flatten().bool().contiguous()
    valid = camera & (labels >= 0) & (labels < 18)
    targets = torch.where(valid & (labels < 17))[0]
    gt = xyz[targets]
    nearest = _nearest_indices(gt, coarse, 1024)
    assigned = torch.div(nearest, COARSE_CHILDREN, rounding_mode='floor').long()
    distance = (gt - coarse[nearest]).norm(dim=1)
    lookup = torch.full((len(labels),), -1, device='cuda', dtype=torch.int32)
    lookup[targets] = torch.arange(len(targets), device='cuda', dtype=torch.int32)
    winner = out['pred_occ'][-1][0, :17].argmax(0).int().contiguous()
    shape = (network.head.aggregator.H, network.head.aggregator.W, network.head.aggregator.D)
    result = extension().census(means, inverse, alpha, semantic, centers, radii, xyz,
        labels.int().contiguous(), camera, lookup, winner, assigned.int().contiguous(), Q, CHILDREN, *shape)
    counts, bits, packed, gs, own, query_counts, qc = result
    pred = out['final_occ'].flatten().long()
    p, y = pred[targets], labels[targets]
    no_core, correct, miss = counts[:, 1] == 0, p == y, p == 17
    own = own.bool()
    features = torch.zeros((Q, len(FIELDS)), device='cuda', dtype=torch.float64)
    def add(name, value, groups=assigned):
        features[:, FIELDS.index(name)].index_add_(0, groups, value.double())
    def put(name, value):
        features[:, FIELDS.index(name)] = value.double()
    ones = torch.ones_like(assigned)
    boundary = batch['query_boundaries'].flatten()[targets]
    for name, value in (
        ('gt_foreground', ones), ('gt_no_core', no_core), ('gt_own_core', own),
        ('gt_neighbor_only_core', ~no_core & ~own), ('gt_miss', miss),
        ('gt_correct', correct), ('gt_wrong_class', ~correct & ~miss),
        ('gt_no_core_correct', no_core & correct), ('gt_no_core_detected', no_core & ~miss),
        ('gt_semantic_boundary', (boundary & 1) != 0), ('gt_empty_boundary', (boundary & 2) != 0),
        ('gt_coarse_distance_sum', distance), ('gt_horizontal_range_sum', gt[:, :2].norm(dim=1)),
        ('gt_core_count_sum', counts[:, 1])):
        add(name, value)
    class_counts = torch.bincount(assigned * 17 + y, minlength=Q * 17).reshape(Q, 17)
    put('gt_classes', (class_counts > 0).sum(1))
    components = batch['query_components'].flatten()[targets].long()
    width = max(int(components.max()) + 1, 1) if len(components) else 1
    pairs = torch.unique(assigned * width + components)
    put('gt_components', torch.bincount(pairs // width, minlength=Q))
    low = torch.full((Q, 3), float('inf'), device='cuda')
    high = torch.full_like(low, -float('inf'))
    indices = assigned[:, None].expand(-1, 3)
    low.scatter_reduce_(0, indices, gt, reduce='amin', include_self=True)
    high.scatter_reduce_(0, indices, gt, reduce='amax', include_self=True)
    span = (high - low).masked_fill(~torch.isfinite(high - low), 0)
    for axis in range(3):
        put(('gt_span_x', 'gt_span_y', 'gt_span_z')[axis], span[:, axis])
    for j, name in enumerate(('core_coverage_all_fg', 'core_shared_other_query', 'core_exclusive')):
        put(name, qc[:, j])
    for j, name in ((0, 'gs_camera_native'), (1, 'native_gs_touches_fg'),
                    (2, 'core_gs_touches_fg'), (3, 'gs_camera_core')):
        put(name, (gs[:, j] > 0).reshape(Q, CHILDREN).sum(1))
    amplitude = semantic.sum(1)
    for name, data in (('amplitude_mean', amplitude), ('opacity_mean', alpha),
        ('effective_amplitude_mean', alpha * amplitude), ('scale_mean', scales.mean(1)),
        ('scale_max_mean', scales.max(1).values), ('center_horizontal_range_mean', means[:, :2].norm(dim=1))):
        put(name, data.reshape(Q, CHILDREN).mean(1))
    put('coarse_spread_mean', coarse.reshape(Q, COARSE_CHILDREN, 3).std(1).mean(1))
    owners = (0xffffffff - (packed & 0xffffffff)).long()
    has_owner = packed > 0
    fg_owner = owners[targets]
    fg_has = has_owner[targets]
    safe = fg_owner.clamp(0, len(means) - 1)
    domq = safe // CHILDREN
    add('dominant_fg_voxels', ones[fg_has], domq[fg_has])
    delta = means[safe] - gt
    inv = inverse[safe]
    power = -.5 * (inv[:, :3] * delta.square()).sum(1) - (
        inv[:, 3:] * delta[:, [0, 1, 0]] * delta[:, [1, 2, 2]]).sum(1)
    mahal = (-2 * power).clamp_min(0).sqrt()
    for prefix, mask in (('no_core_correct', no_core & correct & fg_has),
                         ('core_correct', ~no_core & correct & fg_has)):
        add(prefix + '_dom_n', mask)
        add(prefix + '_dom_amplitude_sum', amplitude[safe] * mask)
        add(prefix + '_dom_opacity_sum', alpha[safe] * mask)
        add(prefix + '_dom_mahalanobis_sum', mahal * mask)
    add('dom_no_core_correct_from_same_query', no_core & correct & fg_has & (domq == assigned))
    fp = valid & (labels == 17) & (pred != 17)
    fp_owned = fp & has_owner
    fpg = owners[fp_owned]
    add('dominant_fp_voxels', torch.ones_like(fpg), fpg // CHILDREN)
    add('dominant_fp_amplitude_sum', amplitude[fpg], fpg // CHILDREN)
    add('dominant_fp_opacity_sum', alpha[fpg], fpg // CHILDREN)
    # Deciles are defined among active queries; empty assignments remain group 0.
    load = features[:, FIELDS.index('gt_foreground')]
    active = torch.where(load > 0)[0]
    order = active[torch.argsort(load[active], stable=True)]
    decile = torch.zeros(Q, device='cuda', dtype=torch.long)
    decile[order] = torch.div(torch.arange(len(order), device='cuda') * 10,
                             max(len(order), 1), rounding_mode='floor') + 1
    ranges = torch.bucketize(gt[:, :2].norm(dim=1), torch.tensor([10., 20., 30.], device='cuda'), right=True)
    errors = torch.bucketize(distance, torch.tensor([.4, .8, 1.6], device='cuda'), right=True)
    key = ((decile[assigned] * 17 + y) * 4 + ranges) * 4 + errors
    strata = torch.zeros((11 * 17 * 4 * 4, len(STRAT_FIELDS)), device='cuda', dtype=torch.int64)
    strata.index_add_(0, key, torch.stack((ones, no_core, miss, no_core & correct, own), 1).long())
    confusion = torch.bincount(labels[valid] * 18 + pred[valid], minlength=18 * 18).reshape(18, 18)
    assert torch.equal((features[:, FIELDS.index('gt_own_core')] +
        features[:, FIELDS.index('gt_neighbor_only_core')] + features[:, FIELDS.index('gt_no_core')]), load)
    assert torch.equal(qc[:, 0], qc[:, 1] + qc[:, 2])
    assert torch.equal(query_counts > 0, counts[:, 1] > 0)
    assert not (own & no_core).any()
    if audit_reference:
        from tools.stat_gaussian_voxel_coverage import measure
        old, info = measure(network, out, batch)
        torch.testing.assert_close(counts, old[0][:, [0, 1, 3]], rtol=0, atol=0)
        assert torch.equal(packed[targets], old[3])
        # Independent direct enumeration for several complete queries.
        reference_queries = (0, 1, 137, 599)
        for q in reference_queries:
            lo, hi = q * CHILDREN, (q + 1) * CHILDREN
            small, _ = measure(network, out, batch, torch.arange(lo, hi, device='cuda'))
            expected = small[0][:, 1] > 0
            actual = ((bits[:, q // 64] >> (q % 64)) & 1).bool()
            assert torch.equal(expected, actual), q
            assert int(expected.sum()) == int(qc[q, 0])
    audit = dict(voxels=[int(miss.sum()), int((~miss).sum())], no_core=int(no_core.sum()),
        foreground=int(len(targets)), false_positive=int(fp.sum()), unattributed_fp=int((fp & ~has_owner).sum()),
        distinct_query_coverage_sum=int(query_counts.sum()),
        counts_sum=counts.long().sum(0).cpu().tolist())
    return dict(metrics=features.float().cpu().numpy(), class_counts=class_counts.int().cpu().numpy(),
                strata=strata.reshape(11, 17, 4, 4, -1).cpu().numpy(),
                confusion=confusion.cpu().numpy(), audit=audit)


def pilot():
    from dataset.utils import custom_collate_fn_temporal
    cfg, protocol = initialize()
    network = build(cfg)
    audits = []
    for split in ('val', 'train'):
        dataset = QueryFrames(cfg, split)
        cache = FeatureCache(network, dataset)
        for index in (0, 7):
            batch = to_cuda(custom_collate_fn_temporal([dataset[index]]))
            with torch.inference_mode():
                out = forward(network, batch)
                row = frame_statistics(network, out, batch, audit_reference=True)
            old = read(OLD / 'scenes' / split / (dataset.keyframes[index][0] + '.json'))
            record = next(r for r in old['records'] if r['index'] == index)
            assert row['audit']['voxels'] == record['voxels']
            previous = read(CAUSAL / 'scenes' / split / (dataset.keyframes[index][0] + '.json'))
            prec = next(r for r in previous['records'] if r['index'] == index)
            np.testing.assert_array_equal(row['confusion'], prec['confusion'][0])
            assert row['audit']['unattributed_fp'] == 0
            audits.append(dict(split=split, index=index, reference_query_bits=True,
                reference_core_counts=True, reference_foreground_owners=True,
                previous_confusion_exact=True, **row['audit']))
            del batch, out, row
        cache.close()
    write(RUN / 'pilot.json', dict(status='passed', audits=audits))
    print('Four-frame query census pilot passed.', flush=True)


def run(workers):
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Subset
    cfg, protocol = initialize()
    assert read(RUN / 'pilot.json')['status'] == 'passed'
    network = build(cfg)
    start, processed, completed = time.perf_counter(), 0, 0
    total_done = 0
    for split in ('val', 'train'):
        dataset = QueryFrames(cfg, split)
        by_scene = {}
        for index, (scene, _) in enumerate(dataset.keyframes):
            by_scene.setdefault(scene, []).append(index)
        pending = []
        for scene, indices in by_scene.items():
            path = RUN / 'scenes' / split / (scene + '.json')
            if path.exists():
                prior = read(path)
                assert prior['indices'] == indices and prior['checkpoint_sha256'] == protocol['checkpoint_sha256']
                assert path.with_suffix('.npz').exists()
                completed += 1
                total_done += len(indices)
            else:
                pending.extend(indices)
        loader = DataLoader(Subset(dataset, pending), batch_size=1, shuffle=False, num_workers=workers,
            collate_fn=custom_collate_fn_temporal, pin_memory=True,
            **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))
        cache = FeatureCache(network, dataset)
        active_scene = None
        for index, batch in zip(pending, loader):
            scene, frame = dataset.keyframes[index]
            if active_scene is None:
                active_scene = scene
                records, metrics, classes = [], [], []
                strata = np.zeros((11, 17, 4, 4, len(STRAT_FIELDS)), dtype=np.int64)
                old = read(OLD / 'scenes' / split / (scene + '.json'))
                old_records = {r['index']: r for r in old['records']}
                reference = read(CAUSAL / 'scenes' / split / (scene + '.json'))
                reference_records = {r['index']: r for r in reference['records']}
            assert active_scene == scene
            batch = to_cuda(batch)
            with torch.inference_mode():
                out = forward(network, batch)
                row = frame_statistics(network, out, batch)
            assert row['audit']['voxels'] == old_records[index]['voxels'], (split, index)
            np.testing.assert_array_equal(row['confusion'], reference_records[index]['confusion'][0])
            assert row['audit']['unattributed_fp'] == 0
            records.append(dict(index=index, scene_frame=int(frame), sample_token=old_records[index]['sample_token'], **row['audit']))
            metrics.append(row['metrics'])
            classes.append(row['class_counts'])
            strata += row['strata']
            processed += 1
            total_done += 1
            if len(records) == len(by_scene[scene]):
                arrays = dict(metrics=np.stack(metrics), class_counts=np.stack(classes), strata=strata)
                with np.load((OLD / 'scenes' / split / (scene + '.npz'))) as z:
                    assert sum(r['no_core'] for r in records) == int(z['flags'][:, 1].sum())
                    np.testing.assert_array_equal(np.asarray([r['counts_sum'] for r in records]).sum(0),
                        z['field_sum'][:, [0, 1, 3]].sum(0).astype(np.int64))
                path = RUN / 'scenes' / split / (scene + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.npz.tmp').open('wb') as f:
                    np.savez_compressed(f, **arrays)
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                write(path, dict(status='complete', split=split, scene_token=scene, scene_name=old['scene_name'],
                    indices=by_scene[scene], frames=len(records), records=records,
                    checkpoint_sha256=protocol['checkpoint_sha256']))
                completed += 1
                elapsed = time.perf_counter() - start
                fps = processed / elapsed
                write(RUN / 'progress.json', dict(status='running', completed_scenes=completed, total_scenes=850,
                    completed_frames=total_done, total_frames=34149, split=split, last_scene=old['scene_name'],
                    frames_this_process=processed, seconds=elapsed, frames_per_second=fps,
                    remaining_minutes=(34149-total_done)/max(fps, 1e-6)/60,
                    updated_at=datetime.now().astimezone().isoformat()))
                if completed % 10 == 0:
                    print(f'{completed}/850 scenes; {total_done}/34149 frames; {fps:.2f} frames/s', flush=True)
                active_scene = None
                del arrays, metrics, classes, strata
            del batch, out, row
        assert active_scene is None
        cache.close()
        del loader, dataset, cache
    elapsed = time.perf_counter() - start
    write(RUN / 'execution_complete.json', dict(status='complete', frames=total_done, scenes=completed,
        seconds=elapsed, checkpoint_sha256=protocol['checkpoint_sha256']))
    write(RUN / 'progress.json', dict(status='complete', completed_scenes=completed, total_scenes=850,
        completed_frames=total_done, total_frames=34149, seconds=elapsed))
    print('Full query census complete.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['pilot', 'run'], required=True)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    pilot() if args.action == 'pilot' else run(args.workers)
