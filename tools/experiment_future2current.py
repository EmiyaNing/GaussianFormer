#!/usr/bin/env python3
"""Same-architecture future-perception to current-forecast feature probe.

The intervention is intentionally local to this experiment.  It never edits
SparseWorld's production segmentor or its checkpoint.
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
from torch import nn
import torch.nn.functional as F
from mmengine import Config
from torch.utils.data import DataLoader, Dataset, Subset

import model  # noqa: F401
import loss  # noqa: F401
from dataset.utils import custom_collate_fn_temporal
from eval_teacher_aligned import _metric
from model.multi_horizon_teacher import file_sha256
from tools.diagnose_teacher_alignment import add_metric, loader_for, seed_all, tree_to
from tools.experiment_future_response_distillation import build_student, paths_for, prepare_student


def _frame(relative):
    return -relative if relative < 0 else (0 if relative == 0 else 6 + relative)


def _future_window(data, imgs, horizon):
    """Five frames ending at t+h; project every camera into the t+h ego frame."""
    frames = [_frame(r) for r in range(horizon, horizon - 5, -1)]
    indices = [6 * frame + cam for frame in frames for cam in range(6)]
    selected = dict(data)
    selected['projection_mat'] = data['projection_mat'][:, indices].clone().float()
    selected['image_wh'] = data['image_wh'][:, indices]
    transforms = data['future_ego_to_current'].float()
    current_from_target = transforms[:, horizon - 1]
    for slot, relative in enumerate(range(horizon, horizon - 5, -1)):
        if relative <= 0:
            frame_from_target = current_from_target
        else:
            frame_from_target = torch.linalg.inv(transforms[:, relative - 1]) @ current_from_target
        begin = 6 * slot
        selected['projection_mat'][:, begin:begin + 6] = (
            selected['projection_mat'][:, begin:begin + 6] @ frame_from_target[:, None])
    return imgs[:, indices].contiguous(), selected


@torch.no_grad()
def perceive_future(student, data, imgs, source, horizon):
    selected_imgs, metas = _future_window(data, imgs, horizon)
    state = dict(imgs=selected_imgs, metas=metas, points=None)
    state.update(student.extract_img_feat(**state))
    state.update(student.lifter(**state))
    ego = source['ego'].cuda()
    scale = (torch.tanh(student.points_scale(ego)) + 1.) * .35 + .8
    state['query_points'] = state['query_points'] * scale
    state.update(student.encoder(**state))
    final = state['representation'][-1]
    current = state['query_stamps'] == 0
    return dict(features=final['query_features'][:, current].half().cpu(),
                centers=student._world(final['query_points'][:, current].float()).mean(2).half().cpu())


def cache_paths(root, split, manifest):
    return paths_for(root / 'teacher_features', split, manifest)


@torch.no_grad()
def prepare_teacher(student, loader, manifest, root, split, source_root, horizons):
    paths = cache_paths(root, split, manifest)
    sources = paths_for(source_root, split, manifest)
    missing = [i for i, path in enumerate(paths) if not path.exists()]
    if not missing:
        return paths
    (root / 'teacher_features' / split).mkdir(parents=True, exist_ok=True)
    selected_loader = DataLoader(Subset(loader.dataset, missing), batch_size=1,
                                 num_workers=loader.num_workers,
                                 collate_fn=custom_collate_fn_temporal, pin_memory=True)
    for position, data in zip(missing, selected_loader):
        source = torch.load(sources[position], map_location='cpu', weights_only=False)
        if source['index'] != manifest['indices'][position]:
            raise RuntimeError('Source cache and selected image sample differ')
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        states = {h: perceive_future(student, data, imgs, source, h) for h in horizons}
        record = dict(index=source['index'], scene=source['scene'], states=states)
        tmp = paths[position].with_suffix('.pth.tmp')
        torch.save(record, tmp)
        tmp.replace(paths[position])
        if (position + 1) % 16 == 0 or position + 1 == len(paths):
            print(f'[future perception cache {split}] {position + 1}/{len(paths)}', flush=True)
        del source, data, imgs, states, record
    return paths


class PairCache(Dataset):
    def __init__(self, source_paths, teacher_paths):
        self.sources = source_paths
        self.teachers = teacher_paths

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        source = torch.load(self.sources[i], map_location='cpu', weights_only=False)
        teacher = torch.load(self.teachers[i], map_location='cpu', weights_only=False)
        if (source['index'], source['scene']) != (teacher['index'], teacher['scene']):
            raise RuntimeError('Paired cache identity mismatch')
        return source, teacher


def pair_loader(dataset, workers, schedule=None):
    if schedule is not None:
        dataset = Subset(dataset, schedule)
    return DataLoader(dataset, batch_size=1, num_workers=workers,
                      collate_fn=lambda rows: rows[0], pin_memory=False)


class SharedReadout(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, hidden),
                                 nn.GELU(), nn.Linear(hidden, 48 * 17))
        # CE calibration must cross the renderer's 0.25-0.35 score thresholds
        # early; the original future head's near-empty prior is unsuitable here.
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features).view(features.shape[0], features.shape[1], 48, 17)


@torch.no_grad()
def baseline(student, source):
    encoded = tree_to(source['encoded'], 'cuda')
    stamps = source['stamps'].cuda()
    current = stamps == 0
    positions = encoded['query_points'][:, current]
    state = dict(features=encoded['query_features'][:, current], positions=positions,
                 timestamp=positions.new_zeros(*positions.shape[:-1], 1),
                 ego=source['ego'].cuda(), all_features=encoded['query_features'],
                 all_points=encoded['query_points'], stamps=stamps,
                 slot_ids=torch.nonzero(current, as_tuple=False).flatten(),
                 completed_steps=0)
    return student._forecast_rollout(state, 6, collect_trace=True)


def forecast_features(student, source, trace):
    """Train feature recurrence; original future-born queries and geometry stay fixed."""
    encoded = tree_to(source['encoded'], 'cuda')
    stamps = source['stamps'].cuda()
    features = encoded['query_features'][:, stamps == 0]
    timestamp = encoded['query_points'][:, stamps == 0, :, :1].new_zeros(
        1, features.shape[1], student.num_refines, 1)
    ego = source['ego'].cuda()
    outputs = {}
    for step in range(6):
        fixed = trace[step]
        fused = student.ego_attention(ego, features, fixed['input_positions'])
        born = stamps == step + 1
        features = torch.cat([features, encoded['query_features'][:, born]], 1)
        timestamp = torch.cat([timestamp, timestamp.new_full(
            (1, int(born.sum()), student.num_refines, 1), .5)], 1)
        position_code = student.position(torch.cat([
            fixed['interaction_positions'], timestamp], -1).flatten(2))
        features = features + fused.expand(-1, features.shape[1], -1) + position_code
        outputs[step + 1] = features
    return outputs


def teacher_at_student(student, teacher_state, student_positions, neighbors, radius):
    """Same future-ego positions, spatial kNN rather than query-index pairing."""
    target = teacher_state['features'].cuda().float()[0]
    centers = teacher_state['centers'].cuda().float()[0]
    student_centers = student._world(student_positions.float()).mean(2)[0]
    dist = torch.cdist(student_centers[None], centers[None])[0]
    nearest_dist, nearest = dist.topk(min(neighbors, len(centers)), largest=False)
    weights = (nearest_dist + .1).reciprocal()
    weights = weights / weights.sum(-1, keepdim=True)
    pooled = (target[nearest] * weights[..., None]).sum(1)
    valid = nearest_dist[:, 0] <= radius
    return pooled[None], valid


def paired_states(student, source, teacher, original, horizons, neighbors, radius,
                  trainable=None, shuffle=False):
    result = {}
    for h in horizons:
        trace = original['trace'][h - 1]
        student_features = (trace['output_features'] if trainable is None
                            else trainable[h])
        teacher_features, valid = teacher_at_student(
            student, teacher['states'][h], trace['output_positions'], neighbors, radius)
        if shuffle:
            # Negative control: identical feature statistics but wrong spatial identity.
            teacher_features = teacher_features.roll(1, dims=1)
        # No teacher data exist outside radius; retain the student's value there.
        merged = torch.where(valid[None, :, None], teacher_features,
                             student_features.detach())
        result[h] = dict(student=student_features, teacher=merged,
                         valid=valid, points=original['points'][h - 1])
    return result


def target_bce(points, logits, labels, max_points=4096, max_targets=4096):
    """Native-style nearest occupied-voxel class, on bounded point subsets."""
    points = points[0]
    logits = logits[0]
    lower = points.new_tensor([-40., -40., -1.])
    occupied = (labels != 17).nonzero()
    if not len(occupied):
        return logits.sum() * 0
    if len(occupied) > max_targets:
        occupied = occupied[torch.randperm(len(occupied), device=points.device)[:max_targets]]
    target_points = lower + (occupied.float() + .5) * .4
    target_classes = labels[occupied[:, 0], occupied[:, 1], occupied[:, 2]].long()
    if len(points) > max_points:
        selected = torch.randperm(len(points), device=points.device)[:max_points]
        points, logits = points[selected], logits[selected]
    with torch.no_grad():
        nearest = torch.cdist(points[None].float(), target_points[None].float())[0].argmin(-1)
    return F.cross_entropy(logits, target_classes[nearest])


def occupancy_loss(readout, states, source, horizons, side):
    total = next(readout.parameters()).new_zeros(())
    for h in horizons:
        state = states[h]
        logits = readout(state[side]).flatten(1, 2)
        labels = source['metas']['future_occ_labels'][0, h - 1].cuda()
        total = total + target_bce(state['points'], logits, labels) / len(horizons)
    return total


def feature_loss(states, horizons):
    losses = []
    for h in horizons:
        s = states[h]
        if s['valid'].any():
            sf = s['student'][:, s['valid']]
            tf = s['teacher'][:, s['valid']]
            losses.append(F.smooth_l1_loss(F.layer_norm(sf, (256,)),
                                           F.layer_norm(tf, (256,))) +
                          (1 - F.cosine_similarity(sf, tf, -1)).mean())
    if not losses:
        return states[horizons[0]]['student'].sum() * 0
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(reference, readout, dataset, cfg, args, label, forecast_model=None):
    bank = {side: {h: _metric(cfg) for h in args.horizons}
            for side in ('student', 'teacher')}
    matched = {h: [] for h in args.horizons}
    feature_distances = []
    total = len(dataset)
    for index, (source, teacher) in enumerate(pair_loader(dataset, args.workers)):
        original = baseline(reference, source)
        trained = (None if forecast_model is None else
                   forecast_features(forecast_model, source, original['trace']))
        states = paired_states(reference, source, teacher, original, args.horizons,
                               args.neighbors, args.radius, trained)
        feature_distances.append(float(feature_loss(states, args.horizons)))
        for h in args.horizons:
            matched[h].append(float(states[h]['valid'].float().mean()))
            for side in bank:
                logits = readout(states[h][side]).flatten(1, 2)
                add_metric(bank[side][h], reference, states[h]['points'],
                           logits, source['metas'], h)
        if (index + 1) % 32 == 0 or index + 1 == total:
            print(f'[evaluate {label}] {index + 1}/{total}', flush=True)
    metrics = {side: {f'{h/2:g}s': bank[side][h].results()
                      for h in args.horizons} for side in bank}
    return dict(metrics=metrics,
                feature_distance=float(np.mean(feature_distances)),
                matched_fraction={f'{h/2:g}s': float(np.mean(matched[h]))
                                  for h in args.horizons})


def readout_train(student, readout, dataset, args, root):
    path = root / 'readout.pth'
    if path.exists() and not args.force_readout:
        readout.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
        return
    readout.train()
    opt = torch.optim.AdamW(readout.parameters(), lr=args.readout_lr, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    schedule = rng.integers(0, len(dataset), size=args.readout_updates).tolist()
    for update, (source, teacher) in enumerate(pair_loader(dataset, args.workers, schedule), 1):
        original = baseline(student, source)
        states = paired_states(student, source, teacher, original, args.horizons,
                               args.neighbors, args.radius)
        opt.zero_grad(set_to_none=True)
        objective = (occupancy_loss(readout, states, source, args.horizons, 'student') +
                     occupancy_loss(readout, states, source, args.horizons, 'teacher')) / 2
        objective.backward()
        opt.step()
        if update % 64 == 0 or update == args.readout_updates:
            print(f'[readout] {update}/{args.readout_updates} loss={objective.item():.4f}', flush=True)
    torch.save({k: v.cpu() for k, v in readout.state_dict().items()}, path)
    readout.eval().requires_grad_(False)


def train_arm(reference, student, readout, train, val, cfg, args, root, arm, seed,
              reevaluate=False):
    result_path = root / f'{arm}_seed{seed}.json'
    path = root / f'{arm}_seed{seed}.pth'
    if result_path.exists() and not reevaluate:
        return json.loads(result_path.read_text(encoding='utf8'))
    seed_all(seed)
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    student.load_state_dict(weights, strict=True)
    student.eval().requires_grad_(False)
    if reevaluate:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for name in ('ego_attention', 'position'):
            getattr(student, name).load_state_dict(saved[name], strict=True)
        result = evaluate(reference, readout, val, cfg, args,
                          f'{arm}_seed{seed}', forecast_model=student)
        result.update(checkpoint=str(path), arm=arm, seed=seed)
        result_path.write_text(json.dumps(result, indent=2), encoding='utf8')
        return result
    for name in ('ego_attention', 'position'):
        getattr(student, name).requires_grad_(True)
    parameters = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    rng = np.random.default_rng(seed)
    schedule = rng.integers(0, len(train), size=args.forecast_updates).tolist()
    for update, (source, teacher) in enumerate(pair_loader(train, args.workers, schedule), 1):
        original = baseline(reference, source)
        predicted = forecast_features(student, source, original['trace'])
        states = paired_states(reference, source, teacher, original, args.horizons,
                               args.neighbors, args.radius, predicted,
                               shuffle=(arm == 'shuffled'))
        task = occupancy_loss(readout, states, source, args.horizons, 'student')
        alignment = feature_loss(states, args.horizons) if arm != 'task_only' else task.new_zeros(())
        objective = task + (args.feature_weight * alignment if arm != 'task_only' else 0)
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        opt.step()
        if update % 128 == 0 or update == args.forecast_updates:
            print(f'[{arm} seed={seed}] {update}/{args.forecast_updates} '
                  f'task={task.item():.4f} align={alignment.item():.4f}', flush=True)
    torch.save({name: {k: v.detach().cpu() for k, v in getattr(student, name).state_dict().items()}
                for name in ('ego_attention', 'position')}, path)
    student.eval().requires_grad_(False)
    result = evaluate(reference, readout, val, cfg, args,
                      f'{arm}_seed{seed}', forecast_model=student)
    result.update(checkpoint=str(path), arm=arm, seed=seed)
    result_path.write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def summary(oracle, arms, args):
    def miou(row, side, key):
        result = row['metrics'][side][key]
        return float(result.get('miou', result.get('mIoU', float('nan'))))
    output = dict(oracle=oracle, arms=arms, comparison={})
    for key in [f'{h/2:g}s' for h in args.horizons]:
        output['comparison'][key] = dict(
            teacher_minus_student_pp=miou(oracle, 'teacher', key) - miou(oracle, 'student', key),
            arms={arm: [miou(run, 'student', key) for run in runs]
                  for arm, runs in arms.items()})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/opus/future2current_shared_readout.py')
    parser.add_argument('--output-dir', default='out/future2current/2048x512')
    parser.add_argument('--source-cache', default='out/future_distillation_response_based/2048x512/source_cache')
    parser.add_argument('--train-samples', type=int)
    parser.add_argument('--val-samples', type=int)
    parser.add_argument('--readout-updates', type=int)
    parser.add_argument('--forecast-updates', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--arms', nargs='+', choices=('task_only', 'aligned', 'shuffled'),
                        default=['task_only', 'aligned', 'shuffled'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--oracle-only', action='store_true')
    parser.add_argument('--force-readout', action='store_true')
    parser.add_argument('--reevaluate', action='store_true',
                        help='Re-evaluate saved arm checkpoints without training')
    options = parser.parse_args()
    cfg = Config.fromfile(options.config)
    args = argparse.Namespace(
        checkpoint=cfg.student_checkpoint,
        train_samples=options.train_samples or cfg.probe_train_samples,
        val_samples=options.val_samples or cfg.probe_val_samples,
        readout_updates=options.readout_updates or cfg.probe_readout_updates,
        forecast_updates=options.forecast_updates or cfg.probe_forecast_updates,
        workers=cfg.probe_workers if options.workers is None else options.workers,
        seeds=options.seeds or cfg.probe_train_seeds,
        seed=cfg.probe_sample_seed, horizons=tuple(cfg.probe_horizons),
        neighbors=cfg.probe_teacher_neighbor_count,
        radius=cfg.probe_match_radius_m,
        feature_weight=cfg.probe_feature_weight, lr=cfg.probe_lr,
        readout_lr=cfg.probe_readout_lr,
        force_readout=options.force_readout)
    root = Path(options.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if file_sha256(args.checkpoint) != cfg.student_checkpoint_sha256:
        raise RuntimeError('Strict SparseWorld checkpoint SHA256 mismatch')
    seed_all(args.seed)
    strict = Config.fromfile(cfg.student_model_config)
    student = build_student(cfg, strict)
    source_root = Path(options.source_cache)
    paired = {}
    manifests = {}
    for split, count in (('train', args.train_samples), ('val', args.val_samples)):
        loader, manifest = loader_for(cfg, split, count, args.workers, args.seed)
        sources = paths_for(source_root, split, manifest)
        if any(not path.exists() for path in sources):
            sources = prepare_student(student, loader, manifest, source_root, split)
        teachers = prepare_teacher(student, loader, manifest, root, split, source_root,
                                   args.horizons)
        paired[split] = PairCache(sources, teachers)
        manifests[split] = manifest
    manifest_path = root / 'manifest.json'
    experiment_manifest = dict(
        config=options.config, checkpoint=args.checkpoint,
        checkpoint_sha256=cfg.student_checkpoint_sha256,
        train=manifests['train'], val=manifests['val'],
        horizons=args.horizons, sample_seed=args.seed)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf8'))
        current = json.loads(json.dumps(experiment_manifest))
        if previous != current:
            raise RuntimeError('Output directory belongs to a different sample split or checkpoint')
    else:
        manifest_path.write_text(json.dumps(experiment_manifest, indent=2), encoding='utf8')
    if options.prepare_only:
        return
    # Forward parity ensures the custom differentiable feature path still
    # reproduces the unmodified six-step SparseWorld feature recurrence.
    sample = torch.load(paired['train'].sources[0], map_location='cpu', weights_only=False)
    original = baseline(student, sample)
    predicted = forecast_features(student, sample, original['trace'])
    for step in range(1, 7):
        expected_queries = 720 + sum(student.future_queries[:step])
        if predicted[step].shape[1] != expected_queries:
            raise RuntimeError(f'future-born queries were not retained at step {step}')
        torch.testing.assert_close(predicted[step], original['trace'][step - 1]['output_features'],
                                   atol=1e-5, rtol=1e-5)
    del sample, original, predicted
    readout = SharedReadout(cfg.probe_readout_hidden).cuda()
    readout_train(student, readout, paired['train'], args, root)
    readout.eval().requires_grad_(False)
    student.eval().requires_grad_(False)
    oracle = evaluate(student, readout, paired['val'], cfg, args, 'frozen_oracle')
    report_path = root / 'results.json'
    previous_arms = {}
    if options.oracle_only and report_path.exists() and not options.force_readout:
        previous_arms = json.loads(report_path.read_text(encoding='utf8')).get('arms', {})
    result = summary(oracle, previous_arms, args)
    result['settings'] = vars(args)
    result['time_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    report_path.write_text(json.dumps(result, indent=2), encoding='utf8')
    if options.oracle_only:
        return
    reference = student
    student = build_student(cfg, strict)
    arms = {}
    for arm in options.arms:
        arms[arm] = []
        for seed in args.seeds:
            run = train_arm(reference, student, readout, paired['train'], paired['val'], cfg,
                            args, root, arm, seed, reevaluate=options.reevaluate)
            arms[arm].append(run)
            result = summary(oracle, arms, args)
            result['settings'] = vars(args)
            result['time_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            report_path.write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
