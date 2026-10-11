#!/usr/bin/env python3
"""Frozen SparseWorld + isolated adapter: which future teacher signals are learnable?

The source model and calibrated readout are never updated. Teacher perception
has already been cached from real future images; no future images enter the
adapter's forward path. All outputs are written beneath a new experiment root.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmengine import Config
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import model  # noqa: F401  register original segmentor
import loss  # noqa: F401  register original loss
from eval_teacher_aligned import _metric
from model.multi_horizon_teacher import file_sha256
from tools.analyze_future2current_regions import DYNAMIC, STATIC
from tools.diagnose_future_image_joint import metric_row, rows_metrics
from tools.diagnose_teacher_alignment import add_metric, seed_all
from tools.experiment_future2current import SharedReadout, baseline, paired_states
from tools.experiment_future2current_identity import (
    ResidualAdapter, TripleCache, loader, save_json as strict_save_json, schedule_for,
)
from tools.experiment_future_response_distillation import build_student, paths_for


HORIZONS = (2, 4, 6)
STRATA = ('all', 'dynamic', 'static', 'other', 'teacher_helpful')


def save_json(path, value):
    """Represent undefined tiny-split metrics as null, never NaN in JSON."""
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item
    strict_save_json(path, clean(value))


def resolve_paths(root, pilot, split, manifest):
    full = paths_for(root / 'teacher_features', split, manifest)
    early = paths_for(pilot / 'teacher_features', split, manifest)
    selected = [p if p.exists() else q for p, q in zip(full, early)]
    absent = [str(p) for p in selected if not p.exists()]
    if absent:
        raise FileNotFoundError(f'Missing {len(absent)} teacher caches; first: {absent[0]}')
    return selected


def dataset_for(manifest, source_root, identity_root, pilot_root, split, count):
    part = {key: value[:count] if isinstance(value, list) else value
            for key, value in manifest[split].items()}
    sources = paths_for(source_root, split, part)
    targets = paths_for(identity_root / 'full_gt_targets', split, part)
    teachers = resolve_paths(identity_root, pilot_root, split, part)
    for family, paths in [('source', sources), ('target', targets)]:
        absent = [str(p) for p in paths if not p.exists()]
        if absent:
            raise FileNotFoundError(f'Missing {len(absent)} {family} caches; first: {absent[0]}')
    return TripleCache(sources, teachers, targets), part


def point_logits(readout, features):
    return readout(features).reshape(-1, 17)


def teacher_feature_loss(states, features, shuffle=False):
    losses = []
    for h in HORIZONS:
        valid = states[h]['valid']
        if not valid.any():
            continue
        target = states[h]['teacher'][:, valid]
        if shuffle and int(valid.sum()) > 1:
            target = target[:, torch.randperm(int(valid.sum()), device=target.device)]
        pred = features[h][:, valid]
        losses.append(F.smooth_l1_loss(F.layer_norm(pred, (256,)),
                                       F.layer_norm(target, (256,))) +
                      (1 - F.cosine_similarity(pred, target, dim=-1)).mean())
    return torch.stack(losses).mean() if losses else features[HORIZONS[0]].sum() * 0


def task_loss(readout, features, targets, max_points):
    loss_value = features[HORIZONS[0]].sum() * 0
    for h in HORIZONS:
        logits = point_logits(readout, features[h])
        labels = targets['labels'][h].cuda().long()
        if len(labels) != len(logits):
            raise RuntimeError(f'Fixed-point labels/logits differ at {h / 2:g}s')
        selected = (torch.randperm(len(labels), device='cuda')[:max_points]
                    if len(labels) > max_points else torch.arange(len(labels), device='cuda'))
        loss_value = loss_value + F.cross_entropy(logits[selected], labels[selected]) / len(HORIZONS)
    return loss_value


def feature_distance(pred, teacher):
    return 1 - F.cosine_similarity(pred, teacher, dim=-1)


def masks_for_query(labels, n_queries, valid, base_logits, teacher_logits):
    labels = labels.reshape(n_queries, 48)
    ids_dynamic = torch.tensor(DYNAMIC, device=labels.device)
    ids_static = torch.tensor(STATIC, device=labels.device)
    dynamic = torch.isin(labels, ids_dynamic).float().mean(-1) > .5
    static = torch.isin(labels, ids_static).float().mean(-1) > .5
    baseline_ce = F.cross_entropy(base_logits.reshape(n_queries * 48, 17),
                                  labels.flatten(), reduction='none').reshape(n_queries, 48).mean(-1)
    teacher_ce = F.cross_entropy(teacher_logits.reshape(n_queries * 48, 17),
                                 labels.flatten(), reduction='none').reshape(n_queries, 48).mean(-1)
    helpful = (teacher_ce + .05 < baseline_ce)
    return {'all': valid, 'dynamic': valid & dynamic, 'static': valid & static,
            'other': valid & ~(dynamic | static), 'teacher_helpful': valid & helpful}


def empty_strata():
    return {f'{h / 2:g}s': {name: dict(queries=0, feature_error=0.,
                                      point_ce=0., point_accuracy=0.,
                                      teacher_ce=0., baseline_ce=0.)
                         for name in STRATA} for h in HORIZONS}


def finalize_strata(sums):
    result = {}
    for horizon, rows in sums.items():
        result[horizon] = {}
        for name, row in rows.items():
            count = row['queries']
            result[horizon][name] = dict(queries=count, **{
                key: (value / count if count else None)
                for key, value in row.items() if key != 'queries'})
    return result


@torch.no_grad()
def evaluate(reference, readout, dataset, cfg, adapter, destination):
    adapter.eval()
    rows = {'baseline': [], 'teacher': [], 'student': []}
    strata = empty_strata()
    for index, (source, teacher, targets) in enumerate(loader(dataset, 0)):
        original = baseline(reference, source)
        states = paired_states(reference, source, teacher, original,
                               HORIZONS, 4, 4.0)
        sample = {side: {} for side in rows}
        for h in HORIZONS:
            state = states[h]
            n_queries = state['student'].shape[1]
            predicted = adapter(state['student'])
            features = dict(baseline=state['student'], teacher=state['teacher'],
                            student=predicted)
            logits = {name: readout(value).reshape(1, -1, 17)
                      for name, value in features.items()}
            labels = targets['labels'][h].cuda().long()
            if len(labels) != n_queries * 48:
                raise RuntimeError('Point-level target geometry changed')
            masks = masks_for_query(labels, n_queries, state['valid'],
                                    logits['baseline'][0], logits['teacher'][0])
            teacher_point_ce = F.cross_entropy(logits['teacher'][0], labels,
                                               reduction='none').reshape(n_queries, 48)
            baseline_point_ce = F.cross_entropy(logits['baseline'][0], labels,
                                                reduction='none').reshape(n_queries, 48)
            point_ce = F.cross_entropy(logits['student'][0], labels,
                                       reduction='none').reshape(n_queries, 48)
            point_correct = (logits['student'][0].argmax(-1) == labels).reshape(n_queries, 48)
            error = feature_distance(predicted[0], state['teacher'][0])
            horizon = f'{h / 2:g}s'
            for name, mask in masks.items():
                count = int(mask.sum())
                if not count:
                    continue
                entry = strata[horizon][name]
                entry['queries'] += count
                entry['feature_error'] += float(error[mask].sum())
                entry['point_ce'] += float(point_ce[mask].mean(-1).sum())
                entry['point_accuracy'] += float(point_correct[mask].float().mean(-1).sum())
                entry['teacher_ce'] += float(teacher_point_ce[mask].mean(-1).sum())
                entry['baseline_ce'] += float(baseline_point_ce[mask].mean(-1).sum())
            for side in rows:
                metric = _metric(cfg)
                add_metric(metric, reference, state['points'], logits[side], source['metas'], h)
                sample[side][h] = metric_row(metric)
        for side in rows:
            rows[side].append(sample[side])
        if (index + 1) % 64 == 0 or index + 1 == len(dataset):
            print(f'[evaluate {destination.stem}] {index + 1}/{len(dataset)}', flush=True)
    arrays = {f'{side}_hist': np.stack([[row[h]['hist'] for h in HORIZONS]
                                        for row in side_rows])
              for side, side_rows in rows.items()}
    np.savez_compressed(destination, **arrays)
    return dict(metrics={side: rows_metrics(side_rows, cfg)
                         for side, side_rows in rows.items()},
                strata=finalize_strata(strata), rows=str(destination))


def train(reference, readout, dataset, args, arm, seed):
    seed_all(seed)
    adapter = ResidualAdapter().cuda().train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=.01)
    order = schedule_for(len(dataset), args.updates, seed)
    for step, (source, teacher, targets) in enumerate(loader(dataset, args.workers, order), 1):
        with torch.no_grad():
            original = baseline(reference, source)
            states = paired_states(reference, source, teacher, original,
                                   HORIZONS, 4, 4.0)
        features = {h: adapter(states[h]['student']) for h in HORIZONS}
        task = task_loss(readout, features, targets, args.max_points)
        if arm == 'task_only':
            feature = task.new_zeros(())
        else:
            feature = teacher_feature_loss(states, features,
                                           shuffle=(arm == 'teacher_shuffled'))
        objective = task + args.feature_weight * feature
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % 128 == 0 or step == args.updates:
            print(f'[{arm} seed={seed}] {step}/{args.updates} '
                  f'task={task.item():.4f} feature={feature.item():.4f}', flush=True)
    return adapter.eval()


def report_settings(args, manifest, cfg):
    return dict(config=args.config, checkpoint=cfg.student_checkpoint,
                checkpoint_sha256=cfg.student_checkpoint_sha256,
                source_manifest=str(args.identity_root / 'manifest.json'),
                train_indices=manifest['train']['indices'],
                val_indices=manifest['val']['indices'],
                train_scenes=len(set(manifest['train']['scenes'])),
                val_scenes=len(set(manifest['val']['scenes'])),
                arms=args.arms, seeds=args.seeds, updates=args.updates,
                feature_weight=args.feature_weight, lr=args.lr,
                max_points=args.max_points,
                shared_readout=str(args.identity_root / 'shared_readout.pth'),
                readout_protocol='frozen shared 48-point semantic probe; not original SparseWorld head')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/opus/future_predictable_information.py')
    parser.add_argument('--output-dir')
    parser.add_argument('--train-samples', type=int)
    parser.add_argument('--val-samples', type=int)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--arms', nargs='+', choices=('task_only', 'teacher_all', 'teacher_shuffled'))
    parser.add_argument('--smoke', action='store_true', help='2 train/2 val, 2 updates, one seed')
    options = parser.parse_args()
    cfg = Config.fromfile(options.config)
    args = argparse.Namespace(
        config=options.config,
        identity_root=Path(cfg.predictability_identity_root),
        pilot_root=Path(cfg.predictability_pilot_root),
        output_dir=Path(options.output_dir or cfg.predictability_root),
        train_samples=options.train_samples or cfg.predictability_train_samples,
        val_samples=options.val_samples or cfg.predictability_val_samples,
        updates=options.updates or cfg.predictability_updates,
        workers=cfg.predictability_workers if options.workers is None else options.workers,
        seeds=options.seeds or cfg.predictability_seeds,
        arms=options.arms or cfg.predictability_arms,
        lr=cfg.predictability_lr,
        feature_weight=cfg.predictability_feature_weight,
        max_points=cfg.predictability_max_points,
    )
    if options.smoke:
        args.train_samples, args.val_samples, args.updates = 2, 2, 2
        args.seeds, args.arms = [cfg.predictability_seeds[0]], ['task_only']
    if file_sha256(cfg.student_checkpoint) != cfg.student_checkpoint_sha256:
        raise RuntimeError('SparseWorld checkpoint SHA256 mismatch')
    full_manifest = json.loads((args.identity_root / 'manifest.json').read_text())
    if full_manifest['checkpoint_sha256'] != cfg.student_checkpoint_sha256:
        raise RuntimeError('Cached source checkpoint SHA256 mismatch')
    source_root = Path(full_manifest['source_cache'])
    train_set, train_manifest = dataset_for(full_manifest, source_root, args.identity_root,
                                            args.pilot_root, 'train', args.train_samples)
    val_set, val_manifest = dataset_for(full_manifest, source_root, args.identity_root,
                                        args.pilot_root, 'val', args.val_samples)
    if set(train_manifest['scenes']) & set(val_manifest['scenes']):
        raise RuntimeError('Training and validation scenes overlap')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = report_settings(args, dict(train=train_manifest, val=val_manifest), cfg)
    settings_path = args.output_dir / 'settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise RuntimeError('Output directory already has different settings')
    else:
        save_json(settings_path, settings)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    strict = Config.fromfile(cfg.student_model_config)
    reference = build_student(cfg, strict)
    readout = SharedReadout(cfg.probe_readout_hidden).cuda()
    readout.load_state_dict(torch.load(args.identity_root / 'shared_readout.pth',
                                      map_location='cpu', weights_only=False))
    readout.eval().requires_grad_(False)
    reference.eval().requires_grad_(False)
    started = time.monotonic()
    report_path = args.output_dir / 'results.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        status='in_progress', settings=str(settings_path), arms={})
    if 'frozen' not in report:
        identity = nn.Identity().cuda()
        report['frozen'] = evaluate(reference, readout, val_set, cfg, identity,
                                    args.output_dir / 'frozen.npz')
        save_json(report_path, report)
    for arm in args.arms:
        report['arms'].setdefault(arm, {})
        for seed in args.seeds:
            if str(seed) in report['arms'][arm]:
                continue
            adapter = train(reference, readout, train_set, args, arm, seed)
            checkpoint = args.output_dir / f'{arm}_seed{seed}.pth'
            torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, checkpoint)
            result = evaluate(reference, readout, val_set, cfg, adapter,
                              args.output_dir / f'{arm}_seed{seed}.npz')
            result['checkpoint'] = str(checkpoint)
            report['arms'][arm][str(seed)] = result
            report['elapsed_seconds'] = round(time.monotonic() - started, 2)
            save_json(report_path, report)
    expected = set(cfg.predictability_arms)
    report['status'] = ('complete' if not options.smoke and
                        expected.issubset(report['arms']) and
                        all({str(seed) for seed in cfg.predictability_seeds}.issubset(
                            report['arms'][arm]) for arm in expected) else 'partial')
    save_json(report_path, report)
    print(f"[{report['status']}] {report_path}", flush=True)


if __name__ == '__main__':
    main()
