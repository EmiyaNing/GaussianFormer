#!/usr/bin/env python3
"""Teacher-query oracle substitution with a trainable Student occupancy decoder.

The frozen SparseWorld forecast and future-born slots are never replaced.
Teacher uses target-frame RGB (privileged); all arms train the identical
query-to-occupancy refiner/decoder on the same data and update schedule.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset, Subset
from mmengine import Config
from mmseg.models import build_segmentor

from dataset.utils import custom_collate_fn_temporal
from eval_teacher_aligned import _metric
from loss import OPENOCC_LOSS
from model.future_image_joint import FutureImageJointRefiner
from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher, file_sha256
from model.teacher_query_injection import ARMS, inject_teacher_queries
from tools.diagnose_future_image_joint import (HORIZONS, metric_row, rows_metrics,
                                               save_json, cache_loader)
from tools.diagnose_teacher_alignment import add_metric, loader_for, seed_all, tree_to


def source_paths(root, manifest, split):
    paths = [root / split / f'{index:06d}' for index in manifest['indices']]
    if any(not (path / 'complete.json').exists() for path in paths):
        raise FileNotFoundError(f'Incomplete Student cache for {split}')
    return paths


@torch.no_grad()
def match_teacher(student_state, teacher_state, pc_range, max_distance_m):
    """One-to-one position-only matching, restricted to persistent slots."""
    born = student_state['born_mask'].bool()
    if born.shape != (student_state['query'].shape[1],):
        raise ValueError('Unexpected future-born mask shape')
    if bool(born[:720].any()) or not bool(born[720:].all()):
        raise RuntimeError('Persistent and future-born slot ordering changed')
    if teacher_state['query_features'].shape[1] != 720:
        raise ValueError('Expected 720 Teacher queries')
    source_normalized = student_state['normalized_points'][0, :720].cuda().float()
    lower = source_normalized.new_tensor(pc_range[:3])
    extent = source_normalized.new_tensor(pc_range[3:]) - lower
    source_metric = source_normalized * extent + lower
    teacher_points = teacher_state['query_points_metric'][0].float()
    distance = torch.cdist(source_metric.mean(1), teacher_points.mean(1)).cpu().numpy()
    rows, cols = linear_sum_assignment(distance)
    keep = distance[rows, cols] <= max_distance_m
    student_rows = torch.as_tensor(rows[keep], device='cuda', dtype=torch.long)
    teacher_rows = torch.as_tensor(cols[keep], device='cuda', dtype=torch.long)
    if not len(student_rows):
        raise RuntimeError('No Teacher queries matched persistent Student queries')
    # Refined point slots are not guaranteed to have the same ordering.
    # Preserve the Student local point index by taking the nearest Teacher
    # point within each matched query. Reuse is allowed; it is not a 48x48
    # bijection and must be reported as such.
    local = torch.cdist(source_metric[student_rows],
                        teacher_points[teacher_rows]).argmin(-1)
    mapped = teacher_points[teacher_rows].gather(
        1, local[..., None].expand(-1, -1, 3))
    return dict(student_rows=student_rows.cpu(), teacher_rows=teacher_rows.cpu(),
                points_for_student=mapped.cpu(),
                center_distance_m=torch.as_tensor(distance[rows[keep], cols[keep]]).float(),
                local_point_distance_m=torch.cdist(source_metric[student_rows],
                                                  teacher_points[teacher_rows]).min(-1).values.cpu().float())


@torch.no_grad()
def build_teacher_cache(teacher, loader, manifest, student_paths, pc_range,
                        directory, split, max_distance_m):
    paths = [directory / split / f'{index:06d}.pth' for index in manifest['indices']]
    missing = [i for i, path in enumerate(paths) if not path.exists()]
    if not missing:
        return paths
    subset = DataLoader(Subset(loader.dataset, missing), batch_size=1,
                        num_workers=loader.num_workers, pin_memory=True,
                        collate_fn=custom_collate_fn_temporal)
    (directory / split).mkdir(parents=True, exist_ok=True)
    for position, data in zip(missing, subset):
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        source = torch.load(student_paths[position] / 'meta.pth', map_location='cpu')
        if source['index'] != manifest['indices'][position] or source['scene'] != manifest['scenes'][position]:
            raise RuntimeError('Teacher and Student cache samples are not paired')
        states = {}
        for h in HORIZONS:
            target = teacher.extract_horizon(imgs, data, h)
            match = match_teacher(source['states'][h], target, pc_range, max_distance_m)
            states[h] = dict(**{key: tree_to(value, 'cpu') for key, value in target.items()},
                             **match)
            print(f'[teacher cache {split}] {position+1}/{len(paths)} horizon={h} '
                  f'matched={len(match["student_rows"])}', flush=True) if position == missing[0] else None
            del target, match
        record = dict(index=source['index'], scene=source['scene'], states=states)
        temporary = paths[position].with_suffix('.pth.tmp')
        torch.save(record, temporary)
        temporary.replace(paths[position])
        if (position + 1) % 16 == 0 or position + 1 == len(paths):
            print(f'[teacher cache {split}] {position+1}/{len(paths)}', flush=True)
        del source, states, record, imgs, data
    return paths


class PairedSamples(Dataset):
    def __init__(self, student_paths, teacher_paths):
        self.student_paths = student_paths
        self.teacher_paths = teacher_paths

    def __len__(self):
        return len(self.student_paths)

    def __getitem__(self, index):
        student = torch.load(self.student_paths[index] / 'meta.pth', map_location='cpu')
        teacher = torch.load(self.teacher_paths[index], map_location='cpu')
        if student['index'] != teacher['index'] or student['scene'] != teacher['scene']:
            raise RuntimeError('Teacher/Student sample pairing changed')
        return dict(student=student, teacher=teacher)


def one_item(items):
    return items[0]


def paired_loader(student_paths, teacher_paths, workers, schedule=None):
    dataset = PairedSamples(student_paths, teacher_paths)
    if schedule is not None:
        dataset = Subset(dataset, schedule)
    options = dict(batch_size=1, num_workers=workers, collate_fn=one_item,
                   pin_memory=False)
    if workers:
        options['prefetch_factor'] = 1
    return DataLoader(dataset, **options)


def forward_horizon(refiner, sample, horizon, arm):
    state = sample['student']['states'][horizon]
    teacher = sample['teacher']['states'][horizon]
    query, points = inject_teacher_queries(
        tree_to(state, 'cuda'), tree_to(teacher, 'cuda'), arm, refiner.pc_range)
    return refiner(query, points)


def sample_loss(refiner, sample, objective, arm, residual_weight):
    values = []
    for h in HORIZONS:
        predictions = forward_horizon(refiner, sample, h, arm)
        loss = 0.
        for stage, prediction in enumerate(predictions):
            cls, pts = objective._set_loss(
                prediction['points'][0], prediction['logits'][0],
                sample['student']['metas']['future_occ_labels'][0, h-1].cuda().long())
            weight = 1. if stage == len(predictions)-1 else .25
            loss = loss + weight * (objective.lambda_future_cls * cls +
                                   objective.lambda_future_pts * pts +
                                   residual_weight * prediction['feature_delta'].square().mean())
        loss = loss / ((1. + .25 * (len(predictions)-1)) * len(HORIZONS))
        loss.backward()
        values.append(float(loss.detach()))
        del predictions, loss
    return sum(values)


@torch.no_grad()
def evaluate(student, refiner, student_paths, teacher_paths, cfg, arm, workers,
             destination, confirmation_indices):
    refiner.eval()
    rows, displacement = [], {h: [] for h in HORIZONS}
    for index, sample in enumerate(paired_loader(student_paths, teacher_paths, workers)):
        sample_rows = {}
        for h in HORIZONS:
            prediction = forward_horizon(refiner, sample, h, arm)[-1]
            metric = _metric(cfg)
            add_metric(metric, student, prediction['points'], prediction['logits'],
                       sample['student']['metas'], h)
            sample_rows[h] = metric_row(metric)
            original = refiner.world(sample['student']['states'][h]['normalized_points'].cuda()).flatten(1, 2)
            displacement[h].append(float((prediction['points'] - original).norm(dim=-1).mean()))
        rows.append(sample_rows)
        if (index + 1) % 64 == 0 or index + 1 == len(student_paths):
            print(f'[eval] {arm}: {index+1}/{len(student_paths)}', flush=True)
    np.savez_compressed(destination,
                        hist=np.stack([[row[h]['hist'] for h in HORIZONS] for row in rows]),
                        intersection=np.asarray([[row[h]['intersection'] for h in HORIZONS] for row in rows]),
                        union=np.asarray([[row[h]['union'] for h in HORIZONS] for row in rows]))
    return dict(metrics=rows_metrics(rows, cfg),
                confirmation=rows_metrics(rows, cfg, confirmation_indices) if confirmation_indices else None,
                mean_point_displacement_m={f'{h*.5:g}s': float(np.mean(v)) for h, v in displacement.items()},
                confusion_file=destination.name)


@torch.no_grad()
def teacher_native(student, student_paths, teacher_paths, cfg, confirmation_indices, destination,
                   workers):
    rows, diagnostics = [], {h: [] for h in HORIZONS}
    for index, sample in enumerate(paired_loader(student_paths, teacher_paths, workers)):
        item = {}
        for h in HORIZONS:
            state = sample['teacher']['states'][h]
            metric = _metric(cfg)
            add_metric(metric, student, state['query_points_metric'].cuda().flatten(1, 2),
                       state['point_logits'].cuda().flatten(1, 2),
                       sample['student']['metas'], h)
            item[h] = metric_row(metric)
            diagnostics[h].append(dict(matched=len(state['student_rows']),
                                       distance=float(state['center_distance_m'].mean()),
                                       local_point_distance=float(state['local_point_distance_m'].mean())))
        rows.append(item)
        if (index + 1) % 64 == 0 or index + 1 == len(student_paths):
            print(f'[teacher native] {index+1}/{len(student_paths)}', flush=True)
    np.savez_compressed(destination,
                        hist=np.stack([[row[h]['hist'] for h in HORIZONS] for row in rows]),
                        intersection=np.asarray([[row[h]['intersection'] for h in HORIZONS] for row in rows]),
                        union=np.asarray([[row[h]['union'] for h in HORIZONS] for row in rows]))
    return dict(metrics=rows_metrics(rows, cfg),
                confirmation=rows_metrics(rows, cfg, confirmation_indices) if confirmation_indices else None,
                match_diagnostics={f'{h*.5:g}s': {key: float(np.mean([v[key] for v in values]))
                                   for key in values[0]} for h, values in diagnostics.items()},
                confusion_file=destination.name)


def main(args):
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    if report_path.exists() and not args.resume:
        raise FileExistsError(f'Use a new output directory or --resume: {output}')
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.data_config)
    strict = Config.fromfile('config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    precedent = json.loads(Path(args.previous_report).read_text())
    if precedent['args']['sample_seed'] != args.sample_seed:
        raise RuntimeError('Sample seed differs from prior H4 experiment')
    train_manifest, val_manifest = precedent['training'], precedent['validation']
    if len(train_manifest['indices']) < args.train_samples or len(val_manifest['indices']) < args.val_samples:
        raise ValueError('Requested samples exceed the paired H4 cache')
    train_manifest = {**train_manifest, 'indices': train_manifest['indices'][:args.train_samples],
                      'scenes': train_manifest['scenes'][:args.train_samples]}
    val_manifest = {**val_manifest, 'indices': val_manifest['indices'][:args.val_samples],
                    'scenes': val_manifest['scenes'][:args.val_samples]}
    source_root = Path(args.student_cache_dir)
    train_sources = source_paths(source_root, train_manifest, 'train')
    val_sources = source_paths(source_root, val_manifest, 'val')
    for split, expected in [('train', train_manifest), ('val', val_manifest)]:
        _, actual = loader_for(cfg, split, len(expected['indices']), 0, args.sample_seed)
        if actual['indices'] != expected['indices'] or actual['scenes'] != expected['scenes']:
            raise RuntimeError(f'Dataset ordering changed from H4 for {split}')
    teacher_digest = file_sha256(cfg.teacher_checkpoint)
    if teacher_digest != cfg.teacher_checkpoint_sha256:
        raise RuntimeError('Teacher checkpoint SHA256 mismatch')
    if file_sha256(cfg.student_init_checkpoint) != cfg.student_init_checkpoint_sha256:
        raise RuntimeError('Student checkpoint SHA256 mismatch')
    cache_root = Path(args.teacher_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    contract = dict(version=1, teacher_sha256=teacher_digest,
                    student_sha256=cfg.student_init_checkpoint_sha256,
                    sample_seed=args.sample_seed,
                    max_match_distance_m=args.max_match_distance_m,
                    train_indices=train_manifest['indices'], val_indices=val_manifest['indices'])
    if (cache_root / 'contract.json').exists():
        if json.loads((cache_root / 'contract.json').read_text()) != contract:
            raise RuntimeError('Incompatible Teacher cache contract')
    else:
        save_json(contract, cache_root / 'contract.json')
    train_loader, _ = loader_for(cfg, 'train', args.train_samples, args.workers, args.sample_seed)
    val_loader, _ = loader_for(cfg, 'val', args.val_samples, args.workers, args.sample_seed)
    if not set(train_manifest['scenes']).isdisjoint(val_manifest['scenes']):
        raise RuntimeError('Train/validation scene overlap')
    missing = any(not (cache_root / split / f'{index:06d}.pth').exists()
                  for split, manifest in [('train', train_manifest), ('val', val_manifest)]
                  for index in manifest['indices'])
    teacher = MultiHorizonFrozenOPUSV1Teacher(
        cfg.teacher_config, cfg.teacher_checkpoint, 'cuda',
        history_frames=cfg.teacher_history_frames,
        expected_sha256=cfg.teacher_checkpoint_sha256, amp=False) if missing else None
    train_teacher_paths = build_teacher_cache(teacher, train_loader, train_manifest, train_sources,
                                               cfg.pc_range, cache_root, 'train', args.max_match_distance_m)
    val_teacher_paths = build_teacher_cache(teacher, val_loader, val_manifest, val_sources,
                                             cfg.pc_range, cache_root, 'val', args.max_match_distance_m)
    cache_peak_cuda_memory_bytes = torch.cuda.max_memory_allocated()
    del teacher, train_loader, val_loader
    torch.cuda.empty_cache()
    student = build_segmentor(strict.model)
    student.load_state_dict(torch.load(cfg.student_init_checkpoint, map_location='cpu')['state_dict'], strict=True)
    student.set_epoch(100)
    student.requires_grad_(False).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    old_scenes = set(json.loads(Path(args.old_h4_report).read_text())['validation']['scenes'])
    confirmation_indices = [i for i, scene in enumerate(val_manifest['scenes']) if scene not in old_scenes]
    baseline_rows = [torch.load(path / 'meta.pth', map_location='cpu')['baseline'] for path in val_sources]
    baseline = rows_metrics(baseline_rows, cfg)
    baseline_confirmation = rows_metrics(baseline_rows, cfg, confirmation_indices) if confirmation_indices else None
    objective = OPENOCC_LOSS.build(copy.deepcopy(cfg.base_sparseworld_loss)).cuda()
    factory = lambda: FutureImageJointRefiner(student, args.rounds, args.max_step_m).cuda()
    with torch.no_grad():
        first = PairedSamples(train_sources, train_teacher_paths)[0]
        refiner = factory().eval()
        original = forward_horizon(refiner, first, 2, 'task_only')[-1]
        torch.testing.assert_close(original['points'], student._world(
            first['student']['states'][2]['normalized_points'].cuda()).flatten(1, 2),
            rtol=0, atol=1e-5)
        torch.testing.assert_close(original['logits'],
            first['student']['states'][2]['logits'].cuda().flatten(1, 2), rtol=0, atol=0)
        assert len(first['teacher']['states'][2]['student_rows']) > 0
        for arm in args.arms:
            prediction = forward_horizon(refiner, first, 2, arm)[-1]
            assert prediction['points'].shape[1] == 840 * 48
        del refiner, first, original, prediction
    if args.resume and report_path.exists():
        report = json.loads(report_path.read_text())
        for key in ('train_samples','val_samples','updates','rounds','max_step_m',
                    'lr','residual_weight','sample_seed','training_seeds','arms',
                    'max_match_distance_m'):
            if report['args'][key] != vars(args)[key]:
                raise RuntimeError(f'Resume protocol mismatch: {key}')
    else:
        teacher_result = teacher_native(student, val_sources, val_teacher_paths, cfg,
                                        confirmation_indices, output / 'teacher_native_confusions.npz',
                                        args.workers)
        report = dict(args=vars(args), student_checkpoint_sha256=cfg.student_init_checkpoint_sha256,
                      teacher_checkpoint_sha256=teacher_digest,
                      teacher_input='eight frames ending at each target timestamp; native future ego',
                      student_input='current plus four past frames; cached frozen rollout',
                      train_manifest=train_manifest, val_manifest=val_manifest,
                      unique_train_scenes=len(set(train_manifest['scenes'])),
                      unique_val_scenes=len(set(val_manifest['scenes'])),
                      confirmation_indices=confirmation_indices,
                      confirmation_scene_count=len({val_manifest['scenes'][i] for i in confirmation_indices}),
                      baseline=baseline, baseline_confirmation=baseline_confirmation,
                      cache_peak_cuda_memory_bytes=cache_peak_cuda_memory_bytes,
                      teacher_native=teacher_result,
                      protocol=dict(arms=list(args.arms), matching=f'position-only 720x720 Hungarian; {args.max_match_distance_m:g} m gate',
                                    local_point_mapping='each Student point independently selects nearest Teacher point',
                                    future_born='all slots retained, neither features nor input positions overwritten',
                                    trainable='shared occupancy refiner: position encoder, spatial attention, FFN, fusion, semantic/regression/velocity heads',
                                    frozen='Teacher, original Student forecast, image backbone/FPN',
                                    separate_horizons=True, input_privileged=True,
                                    precision='FP32; TF32 disabled'), runs={})
        save_json(report, report_path)
    print('[preflight] baseline parity and future-born invariance passed', flush=True)
    for seed in args.training_seeds:
        report['runs'].setdefault(str(seed), {})
        for arm in args.arms:
            if arm in report['runs'][str(seed)]:
                print('[resume completed arm]', seed, arm, flush=True)
                continue
            seed_all(seed)
            refiner = factory().train()
            refiner.reader.requires_grad_(False)
            params = [p for p in refiner.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
            rng = np.random.RandomState(seed)
            schedule = []
            while len(schedule) < args.updates:
                schedule.extend(rng.permutation(len(train_sources)).tolist())
            start_step, history = 0, []
            checkpoint_path = output / f'{seed}_{arm}_latest.pth'
            if args.resume and checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                refiner.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_step = checkpoint['step']
                history = checkpoint.get('history', [])
                print('[resume step]', seed, arm, start_step, flush=True)
            for step, sample in enumerate(paired_loader(train_sources, train_teacher_paths,
                                                        args.workers, schedule[start_step:args.updates]),
                                          start=start_step):
                optimizer.zero_grad(set_to_none=True)
                value = sample_loss(refiner, sample, objective, arm, args.residual_weight)
                norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
                optimizer.step()
                if step < 3 or (step + 1) % 128 == 0:
                    history.append(dict(step=step+1, loss=value, grad_norm=float(norm)))
                    print('[train]', seed, arm, step+1, round(value, 6), flush=True)
                if (step + 1) % 512 == 0 or step + 1 == args.updates:
                    torch.save(dict(state_dict=refiner.state_dict(), optimizer=optimizer.state_dict(),
                                    step=step+1, arm=arm, seed=seed, history=history,
                                    args=vars(args)), checkpoint_path)
            result = evaluate(student, refiner, val_sources, val_teacher_paths, cfg,
                              arm, args.workers, output / f'{seed}_{arm}_confusions.npz',
                              confirmation_indices)
            result.update(training=history, trainable_parameters=sum(p.numel() for p in params))
            report['runs'][str(seed)][arm] = result
            report['elapsed_seconds'] = time.perf_counter() - started
            report['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
            save_json(report, report_path)
            print('[arm complete]', seed, arm,
                  {h: round(v['mIoU'], 4) for h, v in result['metrics'].items()}, flush=True)
            del refiner, optimizer, params
    print('[complete]', report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    parser.add_argument('--previous-report', default='out/future_alignment_hypotheses/H4_joint_512x256/report.json')
    parser.add_argument('--old-h4-report', default='out/future_alignment_hypotheses/H4_R2_decoder128/report.json')
    parser.add_argument('--student-cache-dir', default='out/future_alignment_hypotheses/H4_joint_cache')
    parser.add_argument('--teacher-cache-dir', default='out/future_alignment_hypotheses/teacher_query_cache_512x256')
    parser.add_argument('--output-dir', default='out/future_alignment_hypotheses/teacher_query_replacement_512x256')
    parser.add_argument('--train-samples', type=int, default=512)
    parser.add_argument('--val-samples', type=int, default=256)
    parser.add_argument('--updates', type=int, default=2048)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--max-step-m', type=float, default=4.)
    parser.add_argument('--max-match-distance-m', type=float, default=4.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--residual-weight', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[20260926,20260927,20260928])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
