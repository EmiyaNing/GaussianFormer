#!/usr/bin/env python3
"""Full-size same-feature and learned-adapter future2current experiment.

The original SparseWorld module is unmodified.  All GT nearest-neighbor
targets are computed against the *full* occupied voxel set, then cached.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F
from mmengine import Config
from torch.utils.data import DataLoader, Dataset, Subset

import model  # noqa: F401
import loss  # noqa: F401
from dataset.utils import custom_collate_fn_temporal
from eval_teacher_aligned import _metric
from model.multi_horizon_teacher import file_sha256
from tools.diagnose_future_image_joint import metric_row, rows_metrics
from tools.diagnose_teacher_alignment import add_metric, loader_for, seed_all, tree_to
from tools.experiment_future_response_distillation import build_student, paths_for, prepare_student
from tools.experiment_future2current import (
    SharedReadout, baseline, forecast_features, paired_states, perceive_future,
    feature_loss,
)


class ResidualAdapter(nn.Module):
    """Identity at initialization; learns only a student-feature correction."""
    def __init__(self, dims=256):
        super().__init__()
        self.delta = nn.Sequential(nn.LayerNorm(dims), nn.Linear(dims, dims),
                                   nn.GELU(), nn.Linear(dims, dims))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, features):
        return features + self.delta(features)


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf8')
    tmp.replace(path)


def schedule_for(count, updates, seed):
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < updates:
        result.extend(rng.permutation(count).tolist())
    return result[:updates]


def loader(dataset, workers, schedule=None):
    if schedule is not None:
        dataset = Subset(dataset, schedule)
    return DataLoader(dataset, batch_size=1, num_workers=workers,
                      collate_fn=lambda rows: rows[0], pin_memory=False)


@torch.no_grad()
def prepare_future_teacher(reference, images, manifest, sources, root,
                           old_root, split, horizons):
    new_paths = paths_for(root / 'teacher_features', split, manifest)
    old_paths = paths_for(old_root, split, manifest)
    chosen = [old if old.exists() else new for old, new in zip(old_paths, new_paths)]
    missing = [i for i, path in enumerate(chosen) if not path.exists()]
    if not missing:
        return chosen
    (root / 'teacher_features' / split).mkdir(parents=True, exist_ok=True)
    selected = DataLoader(Subset(images.dataset, missing), batch_size=1,
                          num_workers=images.num_workers,
                          collate_fn=custom_collate_fn_temporal, pin_memory=True)
    for position, data in zip(missing, selected):
        source = torch.load(sources[position], map_location='cpu', weights_only=False)
        if source['index'] != manifest['indices'][position]:
            raise RuntimeError('Frozen source/image identity mismatch')
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        states = {h: perceive_future(reference, data, imgs, source, h)
                  for h in horizons}
        record = dict(index=source['index'], scene=source['scene'], states=states)
        tmp = new_paths[position].with_suffix('.pth.tmp')
        torch.save(record, tmp)
        tmp.replace(new_paths[position])
        if (position + 1) % 32 == 0 or position + 1 == len(chosen):
            print(f'[future teacher {split}] {position + 1}/{len(chosen)}', flush=True)
        del source, data, imgs, states, record
    return chosen


def target_paths(root, split, manifest):
    return paths_for(root / 'full_gt_targets', split, manifest)


@torch.no_grad()
def prepare_full_targets(reference, sources, manifest, root, split, horizons):
    paths = target_paths(root, split, manifest)
    missing = [i for i, path in enumerate(paths) if not path.exists()]
    if not missing:
        return paths
    (root / 'full_gt_targets' / split).mkdir(parents=True, exist_ok=True)
    lower = np.array([-40., -40., -1.], dtype=np.float64)
    for position in missing:
        source = torch.load(sources[position], map_location='cpu', weights_only=False)
        original = baseline(reference, source)
        labels_by_h = {}
        for h in horizons:
            dense = source['metas']['future_occ_labels'][0, h - 1].numpy()
            occupied = np.argwhere(dense != 17)
            if not len(occupied):
                raise RuntimeError(f'Empty occupied GT: {split} index={source["index"]} h={h}')
            centers = lower + (occupied.astype(np.float64) + .5) * .4
            point_cloud = original['points'][h - 1][0].cpu().numpy()
            _, nearest = cKDTree(centers).query(point_cloud, k=1, workers=1)
            nearest_voxel = occupied[nearest]
            classes = dense[nearest_voxel[:, 0], nearest_voxel[:, 1],
                            nearest_voxel[:, 2]].astype(np.uint8)
            labels_by_h[h] = torch.from_numpy(classes)
        record = dict(index=source['index'], scene=source['scene'], labels=labels_by_h)
        tmp = paths[position].with_suffix('.pth.tmp')
        torch.save(record, tmp)
        tmp.replace(paths[position])
        if (position + 1) % 32 == 0 or position + 1 == len(paths):
            print(f'[full GT targets {split}] {position + 1}/{len(paths)}', flush=True)
        del source, original, record
    return paths


class TripleCache(Dataset):
    def __init__(self, sources, teachers, targets):
        if len(sources) != len(teachers) or len(sources) != len(targets):
            raise ValueError('Paired cache lengths differ')
        self.sources, self.teachers, self.targets = sources, teachers, targets

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        records = [torch.load(path, map_location='cpu', weights_only=False)
                   for path in (self.sources[i], self.teachers[i], self.targets[i])]
        if len({(row['index'], row['scene']) for row in records}) != 1:
            raise RuntimeError('Source, Teacher and GT target sample identity differ')
        return tuple(records)


def classification_loss(readout, states, targets, horizons, max_points,
                        sides=('student',)):
    total = next(readout.parameters()).new_zeros(())
    for h in horizons:
        labels = targets['labels'][h].cuda().long()
        size = states[h]['points'].shape[1]
        if len(labels) != size:
            raise RuntimeError(f'Full-GT target length differs from fixed geometry at h={h}')
        selected = (torch.randperm(size, device='cuda')[:max_points]
                    if size > max_points else torch.arange(size, device='cuda'))
        for side in sides:
            logits = readout(states[h][side]).flatten(1, 2)[0]
            total = total + F.cross_entropy(logits[selected], labels[selected]) / (
                len(horizons) * len(sides))
    return total


def adapt_states(states, adapter, horizons):
    if adapter is None:
        return states
    for h in horizons:
        states[h]['student'] = adapter(states[h]['student'])
    return states


def semantic_kl_3s(readout, states, temperature):
    """Distill task-relevant class distributions only at the 3-second horizon."""
    state = states[6]
    if not state['valid'].any():
        return state['student'].sum() * 0
    selected = state['valid']
    student_logits = readout(state['student'][:, selected])
    with torch.no_grad():
        teacher_logits = readout(state['teacher'][:, selected])
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    divergence = F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                          teacher_probs, reduction='none').sum(-1).mean()
    return divergence * temperature ** 2


def magnitude_match(shuffled_loss, correct_loss):
    """Match scalar loss strength while keeping only shuffled-target gradients."""
    return shuffled_loss * (correct_loss.detach() /
                            shuffled_loss.detach().clamp_min(1e-8))


def save_rows(path, rows, horizons, feature_distances):
    np.savez_compressed(
        path,
        hist=np.stack([[row[h]['hist'] for h in horizons] for row in rows]),
        intersection=np.asarray([[row[h]['intersection'] for h in horizons] for row in rows]),
        union=np.asarray([[row[h]['union'] for h in horizons] for row in rows]),
        feature_distance=np.asarray(feature_distances, dtype=np.float32))


@torch.no_grad()
def evaluate(reference, readout, dataset, cfg, args, destination,
             student=None, adapter=None, oracle=False):
    sides = ('student', 'teacher') if oracle else ('student',)
    records = {side: [] for side in sides}
    distances = []
    distances_by_h = {h: [] for h in args.horizons}
    matched_by_h = {h: [] for h in args.horizons}
    if student is not None:
        student.eval()
    if adapter is not None:
        adapter.eval()
    for index, (source, teacher, target) in enumerate(loader(dataset, args.workers)):
        original = baseline(reference, source)
        predicted = (None if student is None else
                     forecast_features(student, source, original['trace']))
        states = paired_states(reference, source, teacher, original,
                               args.horizons, args.neighbors, args.radius, predicted)
        adapt_states(states, adapter, args.horizons)
        distances.append(float(feature_loss(states, args.horizons)))
        for h in args.horizons:
            distances_by_h[h].append(float(feature_loss(states, (h,))))
            matched_by_h[h].append(float(states[h]['valid'].float().mean()))
        for side in sides:
            row = {}
            for h in args.horizons:
                metric = _metric(cfg)
                logits = readout(states[h][side]).flatten(1, 2)
                add_metric(metric, reference, states[h]['points'], logits,
                           source['metas'], h)
                row[h] = metric_row(metric)
            records[side].append(row)
        if (index + 1) % 64 == 0 or index + 1 == len(dataset):
            print(f'[evaluate {destination.stem}] {index + 1}/{len(dataset)}', flush=True)
    result = dict(metrics={}, feature_distance=float(np.mean(distances)),
                  feature_distance_by_h={f'{h/2:g}s': float(np.mean(distances_by_h[h]))
                                         for h in args.horizons},
                  matched_fraction={f'{h/2:g}s': float(np.mean(matched_by_h[h]))
                                    for h in args.horizons},
                  rows={})
    for side in sides:
        side_path = destination.with_name(f'{destination.stem}_{side}.npz')
        save_rows(side_path, records[side], args.horizons, distances)
        result['metrics'][side] = rows_metrics(records[side], cfg)
        result['rows'][side] = str(side_path)
    return result


def calibrate_readout(reference, readout, train, args, root):
    path = root / 'shared_readout.pth'
    if path.exists() and not args.force_readout:
        readout.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
        readout.eval().requires_grad_(False)
        return
    seed_all(args.sample_seed)
    readout.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=args.readout_lr,
                                   weight_decay=.01)
    schedule = schedule_for(len(train), args.readout_updates, args.sample_seed)
    for update, (source, teacher, targets) in enumerate(
            loader(train, args.workers, schedule), 1):
        original = baseline(reference, source)
        states = paired_states(reference, source, teacher, original,
                               args.horizons, args.neighbors, args.radius)
        objective = classification_loss(readout, states, targets, args.horizons,
                                        args.max_points, ('student', 'teacher'))
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if update % 128 == 0 or update == args.readout_updates:
            print(f'[shared readout] {update}/{args.readout_updates} '
                  f'loss={objective.item():.4f}', flush=True)
    torch.save({k: v.detach().cpu() for k, v in readout.state_dict().items()}, path)
    readout.eval().requires_grad_(False)


def train_arm(reference, student, readout, train, val, cfg, args,
              root, arm, seed, reevaluate=False):
    result_path = root / f'{arm}_seed{seed}.json'
    ckpt_path = root / f'{arm}_seed{seed}.pth'
    if result_path.exists() and not reevaluate:
        return json.loads(result_path.read_text(encoding='utf8'))
    seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    student.load_state_dict(weights, strict=True)
    student.eval().requires_grad_(False)
    for name in ('ego_attention', 'position'):
        getattr(student, name).requires_grad_(True)
    use_adapter = arm.startswith('adapter_')
    adapter = ResidualAdapter().cuda() if use_adapter else None
    if reevaluate:
        saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        for name in ('ego_attention', 'position'):
            getattr(student, name).load_state_dict(saved[name], strict=True)
        if adapter is not None:
            adapter.load_state_dict(saved['adapter'], strict=True)
    else:
        param_groups = [dict(params=[p for p in student.parameters() if p.requires_grad],
                             lr=args.forecast_lr)]
        if adapter is not None:
            param_groups.append(dict(params=list(adapter.parameters()), lr=args.adapter_lr))
        optimizer = torch.optim.AdamW(param_groups, weight_decay=.01)
        trainable = [p for group in param_groups for p in group['params']]
        schedule = schedule_for(len(train), args.forecast_updates, seed)
        for update, (source, teacher, targets) in enumerate(
                loader(train, args.workers, schedule), 1):
            original = baseline(reference, source)
            predicted = forecast_features(student, source, original['trace'])
            states = paired_states(reference, source, teacher, original,
                                   args.horizons, args.neighbors, args.radius,
                                   predicted)
            adapt_states(states, adapter, args.horizons)
            matched_control = arm.endswith('shuffled_matched')
            correct_target_loss = None
            if matched_control:
                with torch.no_grad():
                    correct_target_loss = (
                        semantic_kl_3s(readout, states, args.semantic_temperature)
                        if arm.startswith('semantic_kd_3s') else
                        feature_loss(states, args.horizons))
            if 'shuffled' in arm:
                # Randomize only valid matched slots.  A one-slot cyclic shift
                # can preserve nearby content when query IDs have spatial order.
                for h in args.horizons:
                    valid = torch.nonzero(states[h]['valid'], as_tuple=False).flatten()
                    if len(valid) > 1:
                        permutation = valid[torch.randperm(len(valid), device=valid.device)]
                        target = states[h]['teacher'].clone()
                        target[:, valid] = states[h]['teacher'][:, permutation]
                        states[h]['teacher'] = target
            task = classification_loss(readout, states, targets, args.horizons,
                                       args.max_points)
            if arm.startswith('semantic_kd_3s'):
                align = semantic_kl_3s(readout, states, args.semantic_temperature)
                weight = args.semantic_weight
            elif arm.endswith('align') or 'shuffled' in arm:
                align = feature_loss(states, args.horizons)
                weight = args.feature_weight
            else:
                align = task.new_zeros(())
                weight = 0.
            if matched_control:
                align = magnitude_match(align, correct_target_loss)
            objective = task + weight * align
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
            optimizer.step()
            if update % 128 == 0 or update == args.forecast_updates:
                print(f'[{arm} seed={seed}] {update}/{args.forecast_updates} '
                      f'task={task.item():.4f} feature={align.item():.4f}', flush=True)
        checkpoint = {name: {k: v.detach().cpu() for k, v in
                            getattr(student, name).state_dict().items()}
                      for name in ('ego_attention', 'position')}
        if adapter is not None:
            checkpoint['adapter'] = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
        torch.save(checkpoint, ckpt_path)
    student.eval().requires_grad_(False)
    if adapter is not None:
        adapter.eval().requires_grad_(False)
    result = evaluate(reference, readout, val, cfg, args,
                      root / f'{arm}_seed{seed}.npz', student, adapter)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    if reevaluate and result_path.exists():
        previous = json.loads(result_path.read_text(encoding='utf8'))
        peak = previous.get('peak_allocated_gib', peak)
    result.update(arm=arm, seed=seed, checkpoint=str(ckpt_path),
                  peak_allocated_gib=peak)
    save_json(result_path, result)
    return result


def _miou_from_hist(hist):
    diagonal = np.diag(hist).astype(np.float64)
    denominator = hist.sum(0) + hist.sum(1) - diagonal
    iou = np.divide(diagonal[:17], denominator[:17],
                    out=np.full(17, np.nan), where=denominator[:17] > 0)
    return float(np.nanmean(iou) * 100)


def scene_bootstrap(root, arms, seeds, scenes, horizons, repeats, sample_seed):
    by_scene = {}
    for position, scene in enumerate(scenes):
        by_scene.setdefault(scene, []).append(position)
    groups = list(by_scene.values())
    rng = np.random.default_rng(sample_seed + 700)
    arrays = {}
    for arm in arms:
        for seed in seeds:
            path = root / f'{arm}_seed{seed}_student.npz'
            with np.load(path) as data:
                sample_hist = data['hist'].astype(np.int64)
                arrays[(arm, seed)] = np.stack(
                    [sample_hist[positions].sum(0) for positions in groups])
    comparisons = [('raw_align', 'task_only'),
                   ('raw_align', 'raw_shuffled'),
                   ('adapter_align', 'adapter_task'),
                   ('adapter_align', 'adapter_shuffled'),
                   ('semantic_kd_3s', 'task_only'),
                   ('semantic_kd_3s', 'semantic_kd_3s_shuffled'),
                   ('raw_align', 'raw_shuffled_matched'),
                   ('adapter_align', 'adapter_shuffled_matched'),
                   ('semantic_kd_3s', 'semantic_kd_3s_shuffled_matched')]
    output = {}
    for h_index, h in enumerate(horizons):
        key = f'{h/2:g}s'
        output[key] = {}
        for treatment, control in comparisons:
            if treatment not in arms or control not in arms:
                continue
            delta = []
            for _ in range(repeats):
                sampled_scenes = rng.integers(0, len(groups), size=len(groups))
                per_seed = []
                for seed in seeds:
                    a = arrays[(treatment, seed)][sampled_scenes, h_index].sum(0)
                    b = arrays[(control, seed)][sampled_scenes, h_index].sum(0)
                    per_seed.append(_miou_from_hist(a) - _miou_from_hist(b))
                delta.append(np.mean(per_seed))
            output[key][f'{treatment}_minus_{control}'] = dict(
                lower_pp=float(np.quantile(delta, .025)),
                upper_pp=float(np.quantile(delta, .975)),
                bootstrap_repeats=repeats)
    return output


def summarize(oracle, runs, args, root, val_scenes, final=False):
    comparison = {}
    for h in args.horizons:
        key = f'{h/2:g}s'
        comparison[key] = dict(
            oracle_teacher_minus_student_pp=(
                oracle['metrics']['teacher'][key]['mIoU'] -
                oracle['metrics']['student'][key]['mIoU']),
            arms={arm: [row['metrics']['student'][key]['mIoU'] for row in values]
                  for arm, values in runs.items()})
    report = dict(status='complete' if final else 'in_progress',
                  oracle=oracle, arms=runs, comparison=comparison,
                  settings=vars(args), time_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    if final:
        report['scene_bootstrap'] = scene_bootstrap(
            root, list(runs), args.seeds, val_scenes, args.horizons,
            args.bootstrap_repeats, args.sample_seed)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/opus/future2current_feature_identity.py')
    parser.add_argument('--output-dir', default='out/future2current/feature_identity_2048x512')
    parser.add_argument('--source-cache')
    parser.add_argument('--train-samples', type=int)
    parser.add_argument('--val-samples', type=int)
    parser.add_argument('--readout-updates', type=int)
    parser.add_argument('--forecast-updates', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--targets-only', action='store_true',
                        help='Build full-GT labels independently of future-image teacher cache')
    parser.add_argument('--oracle-only', action='store_true')
    parser.add_argument('--force-readout', action='store_true')
    parser.add_argument('--reevaluate', action='store_true')
    options = parser.parse_args()
    cfg = Config.fromfile(options.config)
    args = argparse.Namespace(
        checkpoint=cfg.student_checkpoint,
        train_samples=options.train_samples or cfg.identity_train_samples,
        val_samples=options.val_samples or cfg.identity_val_samples,
        readout_updates=options.readout_updates or cfg.identity_readout_updates,
        forecast_updates=options.forecast_updates or cfg.identity_forecast_updates,
        workers=cfg.identity_workers if options.workers is None else options.workers,
        seeds=options.seeds or cfg.identity_train_seeds,
        arms=options.arms or cfg.identity_arms,
        horizons=tuple(cfg.identity_horizons),
        sample_seed=cfg.identity_sample_seed,
        max_points=cfg.identity_max_prediction_points,
        readout_lr=cfg.identity_readout_lr,
        forecast_lr=cfg.identity_forecast_lr,
        adapter_lr=cfg.identity_adapter_lr,
        feature_weight=cfg.identity_feature_weight,
        semantic_weight=cfg.identity_semantic_weight,
        semantic_temperature=cfg.identity_semantic_temperature,
        neighbors=cfg.identity_teacher_neighbor_count,
        radius=cfg.identity_match_radius_m,
        oracle_gate_pp=cfg.identity_oracle_gate_pp,
        bootstrap_repeats=cfg.identity_bootstrap_repeats,
        force_readout=options.force_readout)
    for arm in args.arms:
        if arm not in cfg.identity_arms:
            raise ValueError(f'Unknown experiment arm: {arm}')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(args.sample_seed)
    if file_sha256(args.checkpoint) != cfg.student_checkpoint_sha256:
        raise RuntimeError('Pretrained SparseWorld checkpoint SHA256 mismatch')
    root = Path(options.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(options.source_cache or cfg.identity_source_cache)
    pilot_root = Path(cfg.identity_pilot_teacher_cache)
    strict = Config.fromfile(cfg.student_model_config)
    reference = build_student(cfg, strict)
    datasets, manifests = {}, {}
    for split, count in (('train', args.train_samples), ('val', args.val_samples)):
        images, manifest = loader_for(cfg, split, count, args.workers,
                                      args.sample_seed)
        manifests[split] = manifest
        sources = paths_for(source_root, split, manifest)
        if any(not path.exists() for path in sources):
            sources = prepare_student(reference, images, manifest, source_root, split)
        if options.targets_only:
            prepare_full_targets(reference, sources, manifest, root, split, args.horizons)
            continue
        teachers = prepare_future_teacher(reference, images, manifest, sources,
                                          root, pilot_root, split, args.horizons)
        targets = prepare_full_targets(reference, sources, manifest, root,
                                       split, args.horizons)
        datasets[split] = TripleCache(sources, teachers, targets)
    if options.targets_only:
        print('[full GT targets complete]', root, flush=True)
        return
    if set(manifests['train']['scenes']) & set(manifests['val']['scenes']):
        raise RuntimeError('Train and validation scenes overlap')
    manifest_path = root / 'manifest.json'
    manifest = dict(config=options.config, source_cache=str(source_root),
                    checkpoint=args.checkpoint,
                    checkpoint_sha256=cfg.student_checkpoint_sha256,
                    train=manifests['train'], val=manifests['val'],
                    horizons=args.horizons)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf8')) != json.loads(json.dumps(manifest)):
            raise RuntimeError('Output directory belongs to a different sample split or checkpoint')
    else:
        save_json(manifest_path, manifest)
    if options.prepare_only:
        print('[preparation complete]', root, flush=True)
        return
    sample = torch.load(datasets['train'].sources[0], map_location='cpu', weights_only=False)
    trace = baseline(reference, sample)['trace']
    predicted = forecast_features(reference, sample, trace)
    for step in range(1, 7):
        expected = 720 + sum(reference.future_queries[:step])
        if predicted[step].shape[1] != expected:
            raise RuntimeError('Future-born query count changed')
        torch.testing.assert_close(predicted[step], trace[step - 1]['output_features'],
                                   atol=1e-5, rtol=1e-5)
    del sample, trace, predicted
    readout = SharedReadout(cfg.probe_readout_hidden).cuda()
    calibrate_readout(reference, readout, datasets['train'], args, root)
    readout.eval().requires_grad_(False)
    oracle_path = root / 'frozen_oracle.json'
    if oracle_path.exists() and not options.reevaluate and not options.force_readout:
        oracle = json.loads(oracle_path.read_text(encoding='utf8'))
    else:
        oracle = evaluate(reference, readout, datasets['val'], cfg, args,
                          root / 'frozen_oracle.npz', oracle=True)
        save_json(oracle_path, oracle)
    report_path = root / 'results.json'
    runs = {}
    save_json(report_path, summarize(oracle, runs, args, root,
                                     manifests['val']['scenes']))
    if options.oracle_only:
        return
    positive = [oracle['metrics']['teacher'][f'{h/2:g}s']['mIoU'] -
                oracle['metrics']['student'][f'{h/2:g}s']['mIoU']
                for h in (4, 6)]
    if max(positive) < args.oracle_gate_pp:
        report = summarize(oracle, runs, args, root, manifests['val']['scenes'])
        report['status'] = 'oracle_gate_failed'
        report['reason'] = 'Future-image features lack the predefined shared-readout mIoU advantage'
        save_json(report_path, report)
        print('[oracle gate failed] stop before distillation', positive, flush=True)
        return
    student = build_student(cfg, strict)
    for arm in args.arms:
        runs[arm] = []
        for seed in args.seeds:
            run = train_arm(reference, student, readout, datasets['train'],
                            datasets['val'], cfg, args, root, arm, seed,
                            reevaluate=options.reevaluate)
            runs[arm].append(run)
            save_json(report_path, summarize(oracle, runs, args, root,
                                             manifests['val']['scenes']))
    final = set(args.arms) == set(cfg.identity_arms) and len(args.seeds) == len(cfg.identity_train_seeds)
    save_json(report_path, summarize(oracle, runs, args, root,
                                     manifests['val']['scenes'], final=final))


if __name__ == '__main__':
    main()
