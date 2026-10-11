#!/usr/bin/env python3
"""Controlled future-feature diagnosis; no changes to production modules.

Calibration uses training scenes only. Evaluation interventions replace
matched features/logits at each horizon independently, keeping Student points
and all unmatched/future-born slots. They are privileged-information probes,
not deployable forecasts. Cached pilot inputs come only from frozen perception.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mmengine import Config
from mmseg.models import build_segmentor

import model
import model.segmentor.sparseworld_teacher_aligned_segmentor
import loss.sparseworld_teacher_alignment_loss
from loss import OPENOCC_LOSS
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher, file_sha256
from model.sparse_latent_dynamics import HungarianSparseMatcher
from eval_teacher_aligned import _metric, _json_safe
from train_teacher_aligned import _load_student_initial, _audit_freeze


def tree_to(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    return value


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader_for(cfg, split, count, workers, seed):
    dc = copy.deepcopy(cfg.train_dataset_config if split == 'train'
                       else cfg.val_dataset_config)
    # Deterministic evaluation augmentation; load future RGB for teacher only.
    dc.pipeline = [copy.deepcopy(x) for x in cfg.train_pipeline
                   if x['type'] != 'PhotoMetricDistortionMultiViewImage']
    dc.phase = 'val'
    dc.history_frames = cfg.teacher_history_frames
    dataset = OPENOCC_DATASET.build(dc)
    by_scene = {}
    for index, (scene, _) in enumerate(dataset.keyframes):
        by_scene.setdefault(scene, []).append(index)
    rng = random.Random(seed)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    indices = []
    depth = 0
    while len(indices) < min(count, len(dataset)):
        for scene in scenes:
            candidates = by_scene[scene]
            # Randomized within scene, one sample/scene before second sample.
            if depth == 0:
                rng.shuffle(candidates)
            if depth < len(candidates):
                indices.append(candidates[depth])
                if len(indices) >= min(count, len(dataset)):
                    break
        depth += 1
    selected_scenes = [dataset.keyframes[i][0] for i in indices]
    return DataLoader(Subset(dataset, indices), batch_size=1,
                      num_workers=workers, collate_fn=custom_collate_fn_temporal,
                      pin_memory=True), dict(indices=indices,
                      scenes=selected_scenes, eligible_samples=len(dataset))


def build_student(cfg, checkpoint=None):
    instance = build_segmentor(cfg.model)
    # All existing weights load from checkpoint; new layers initialize in ctor.
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu')['state_dict']
        instance.load_state_dict(state, strict=True)
    else:
        _load_student_initial(instance, cfg.student_init_checkpoint,
                              cfg.student_init_checkpoint_sha256)
        instance.initialize_future_from_pretrained()
    instance.set_epoch(0)
    return instance.cuda().eval()


@contextmanager
def shared_projection(instance, enabled):
    """For pilot D, projection affects both recurrence and task decoding."""
    if not enabled:
        yield
        return
    instance._diagnostic_raw_deltas = []
    def project(_module, args, delta):
        instance._diagnostic_raw_deltas.append(delta)
        mixed = args[0]
        return instance.student_projection(mixed + delta) - mixed
    handle = instance.future_feature_mlp.register_forward_hook(project)
    try:
        yield
    finally:
        handle.remove()


def rollout(instance, cache, shared=False):
    encoded = tree_to(cache['encoded'], 'cuda')
    metas = tree_to(cache['metas'], 'cuda')
    ego = instance.future_plan_head(metas['temporal_ego_states'].float()
                                    .reshape(1, -1)).unsqueeze(1)
    with shared_projection(instance, shared):
        result = instance._teacher_aligned_rollout(dict(
            representation=[encoded], query_stamps=cache['stamps'].cuda(),
            future_ego_feature=ego))
    if shared:
        for state, raw_delta in zip(result['query_states'],
                                    instance._diagnostic_raw_deltas):
            state['aligned_features'] = state['query_features']
            state['feature_delta'] = raw_delta
        instance._diagnostic_raw_deltas = []
    return dict(future_pred_points=result['points'],
                future_pred_logits=result['logits'],
                future_query_states=result['query_states'],
                pred_traj=torch.cat(result['trajectories'], 1), metas=metas,
                teacher_states=tree_to(cache['teacher'], 'cuda'))


def match(matcher, student, teacher):
    def pack(state):
        return dict(points_metric=state['query_points_metric'],
                    semantic_logits=state['semantic_logits'],
                    query_feats=state['query_features'])
    matches = matcher(pack(student), pack(teacher))
    return matches.prediction[0], matches.target[0]


@torch.no_grad()
def add_metric(metric, instance, points, logits, metas, horizon):
    dense = instance.head.rasterizer(points.float(), logits.float(), 48)
    dense = dense.view(1, *instance.head.rasterizer.grid_shape)[0].cpu().numpy()
    key = 'occ_label' if horizon == 0 else 'future_occ_labels'
    if horizon == 0:
        labels = metas[key][0]
        camera, lidar = metas['occ_cam_mask'][0], metas['occ_lidar_mask'][0]
    else:
        labels = metas[key][0, horizon - 1]
        camera = metas['future_occ_cam_masks'][0, horizon - 1]
        lidar = metas['future_occ_lidar_masks'][0, horizon - 1]
    metric.add_batch(dense, labels.cpu().numpy(), lidar.cpu().numpy(),
                     camera.cpu().numpy())


def metric_bank(cfg, names):
    return {name: {h: _metric(cfg) for h in (2, 4, 6)} for name in names}


def bank_results(bank):
    return {name: {f'{h*.5:g}s': metric.results()
                   for h, metric in horizons.items()}
            for name, horizons in bank.items()}


@torch.no_grad()
def cache_samples(instance, teacher, loader, label):
    caches = []
    for i, data in enumerate(loader):
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        teacher_states = teacher.extract_all(imgs, data)
        # Exactly the five Student frames; no teacher-only RGB enters perception.
        student_metas = dict(data)
        student_metas['projection_mat'] = data['projection_mat'][:, :30]
        student_metas['image_wh'] = data['image_wh'][:, :30]
        result = instance._frozen_perception(imgs[:, :30], student_metas)
        retain = {k: data[k] for k in (
            'sample_idx', 'scene_token', 'temporal_ego_states', 'temporal_trajs',
            'occ_label', 'occ_cam_mask', 'occ_lidar_mask', 'future_occ_labels',
            'future_occ_cam_masks', 'future_occ_lidar_masks')}
        # Compact labels in host memory; restore long dtype before set losses.
        for key in ('occ_label', 'future_occ_labels'):
            retain[key] = retain[key].to(torch.uint8)
        caches.append(tree_to(dict(encoded=result['representation'][-1],
                                  stamps=result['query_stamps'], metas=retain,
                                  teacher=teacher_states), 'cpu'))
        print(f'[cache {label}] {i+1}/{len(loader)}', flush=True)
        del data, imgs, result, teacher_states
    return caches


@torch.no_grad()
def fit_adapters(instance, calibration, matcher, ridge):
    accum = {h: [torch.zeros(257, 257, device='cuda', dtype=torch.float64),
                  torch.zeros(257, 256, device='cuda', dtype=torch.float64), 0]
             for h in (2, 4, 6)}
    for cache in calibration:
        out = rollout(instance, cache)
        for h in accum:
            s, t = out['future_query_states'][h-1], out['teacher_states'][h-1]
            si, ti = match(matcher, s, t)
            x = t['query_features'][0, ti].double()
            y = s['query_features'][0, si].double()
            x = torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)
            accum[h][0].add_(x.T @ x)
            accum[h][1].add_(x.T @ y)
            accum[h][2] += len(x)
    adapters = {}
    for h, (xx, xy, count) in accum.items():
        penalty = torch.eye(257, device='cuda', dtype=torch.float64)
        penalty[-1, -1] = 0.
        scale = xx.diag()[:-1].mean().clamp_min(1.)
        adapters[h] = torch.linalg.solve(xx + ridge * scale * penalty, xy).float()
        print('[adapter]', h, 'training pairs', count, flush=True)
    return adapters


def feature_dist(x, y):
    return (F.smooth_l1_loss(x.float(), y.float(), beta=1.) +
            (1. - F.cosine_similarity(x.float(), y.float(), dim=-1)).mean())


@torch.no_grad()
def diagnose(instance, baseline, caches, adapters, matcher, cfg):
    names = ['baseline', 'trained', 'teacher_native', 'student_projected_decode',
             'teacher_raw_feature', 'teacher_calibrated_feature',
             'teacher_shuffled_feature', 'teacher_matched_logits']
    bank = metric_bank(cfg, names)
    statistics = {h: [] for h in (2, 4, 6)}
    for i, cache in enumerate(caches):
        out, base = rollout(instance, cache), rollout(baseline, cache)
        metas = out['metas']
        for h in statistics:
            s, t = out['future_query_states'][h-1], out['teacher_states'][h-1]
            si, ti = match(matcher, s, t)
            feature, target = s['query_features'], t['query_features']
            points, logits = out['future_pred_points'][h-1], s['point_logits']
            add_metric(bank['baseline'][h], instance,
                       base['future_pred_points'][h-1],
                       base['future_pred_logits'][h-1], metas, h)
            add_metric(bank['trained'][h], instance, points,
                       logits.flatten(1, 2), metas, h)
            add_metric(bank['teacher_native'][h], instance,
                       t['query_points_metric'].flatten(1, 2),
                       t['point_logits'].flatten(1, 2), metas, h)
            add_metric(bank['student_projected_decode'][h], instance, points,
                       instance.semantic(s['aligned_features']).view(
                           1, -1, 17), metas, h)
            x = target[0, ti]
            calibrated = torch.cat([x, torch.ones(len(x), 1, device=x.device)],
                                   1) @ adapters[h]
            for name, replacement in (
                    ('teacher_raw_feature', x),
                    ('teacher_calibrated_feature', calibrated),
                    ('teacher_shuffled_feature', calibrated.roll(1, 0))):
                modified = feature.clone()
                modified[0, si] = replacement
                new_logits = instance.semantic(modified).view(1, -1, 17)
                add_metric(bank[name][h], instance, points, new_logits, metas, h)
            replaced_logits = logits.clone()
            # Refined point order is unordered too. Associate Teacher logits
            # with each Student point by local nearest geometry, not index.
            local_nearest = torch.cdist(
                s['query_points_metric'][0, si].float(),
                t['query_points_metric'][0, ti].float()).argmin(-1)
            teacher_local_logits = t['point_logits'][0, ti].gather(
                1, local_nearest[..., None].expand(-1, -1, 17))
            replaced_logits[0, si] = teacher_local_logits
            add_metric(bank['teacher_matched_logits'][h], instance, points,
                       replaced_logits.flatten(1, 2), metas, h)
            row = dict(matched=len(si), total=feature.shape[1],
                       feature_raw=float(feature_dist(feature[0, si], x)),
                       feature_projected=float(feature_dist(
                           s['aligned_features'][0, si], x)),
                       feature_calibrated=float(feature_dist(
                           feature[0, si], calibrated)),
                       center_l2_m=float((s['query_points_metric'][0, si].mean(1)
                           - t['query_points_metric'][0, ti].mean(1)).norm(
                               dim=-1).mean()))
            for label, mask in (('persistent', s['persistent_mask']),
                                ('future_born', s['future_born_mask'])):
                total = int(mask.sum())
                n = int(mask[si].sum())
                row[label + '_matched_ratio'] = n / max(total, 1)
            statistics[h].append(row)
        print(f'[diagnose] {i+1}/{len(caches)}', flush=True)
    summary = {str(h): {k: float(np.mean([row[k] for row in rows]))
                        for k in rows[0]} for h, rows in statistics.items()}
    return dict(metrics=bank_results(bank), alignment=summary)


def gradient_probe(instance, cache, objective):
    instance.eval()
    out = rollout(instance, cache)
    out['metas']['future_occ_labels'] = out['metas']['future_occ_labels'].long()
    out['global_iter'] = objective.alignment_warmup_iters
    task, _ = objective._task_loss(out)
    total, _ = objective(out)
    alignment = total - task
    parameters = [(n, p) for n, p in instance.named_parameters()
                  if p.requires_grad]
    g_task = torch.autograd.grad(task, [p for _, p in parameters],
                                 retain_graph=True, allow_unused=True)
    g_align = torch.autograd.grad(alignment, [p for _, p in parameters],
                                  allow_unused=True)
    norms = {}
    for name, a, b in zip([n for n, _ in parameters], g_task, g_align):
        group = name.split('.')[0]
        v = norms.setdefault(group, [0., 0., 0.])
        if a is not None: v[0] += float(a.float().square().sum())
        if b is not None: v[1] += float(b.float().square().sum())
        if a is not None and b is not None:
            v[2] += float((a.float() * b.float()).sum())
    return {name: dict(task_grad_norm=a**.5, alignment_grad_norm=b**.5,
                       cosine=dot / (a*b)**.5 if a*b > 0 else None)
            for name, (a, b, dot) in norms.items()}


def pilot_loss_config(cfg, variant):
    """Keep KL-off and full objectives identical except semantic weight."""
    lc = copy.deepcopy(cfg.loss)
    lc.pop('_delete_', None)
    lc.alignment_warmup_iters = 0
    if variant in ('A_task_only', 'B_feature_only', 'D_shared_feature'):
        lc.geometry_weight = lc.semantic_weight = 0.
    elif variant == 'E_no_semantic_kl':
        lc.semantic_weight = 0.
    elif variant != 'C_full':
        raise ValueError('unknown pilot variant: ' + variant)
    if variant == 'A_task_only':
        lc.feature_weight = 0.
    return lc


def run_pilots(instance, calibration, validation, cfg, updates, seed, output_dir,
               variants=('A_task_only', 'B_feature_only', 'C_full',
                         'D_shared_feature')):
    initial = tree_to(instance.state_dict(), 'cpu')
    results = {}
    for variant in variants:
        seed_all(seed)
        instance.load_state_dict(initial, strict=True)
        instance.train()
        shared = variant.startswith('D')
        # Full alignment from update 1; same budget/learning rates for all arms.
        lc = pilot_loss_config(cfg, variant)
        objective = OPENOCC_LOSS.build(lc).cuda()
        groups = []
        for name, parameter in instance.named_parameters():
            if parameter.requires_grad:
                lr = (cfg.optimizer.optimizer.lr if name.startswith(
                    ('future_feature_mlp.', 'student_projection.'))
                      else cfg.optimizer.existing_future_lr)
                groups.append(dict(params=[parameter], lr=lr,
                    weight_decay=0. if name.startswith('student_projection.')
                    else cfg.optimizer.optimizer.weight_decay))
        optimizer = torch.optim.AdamW(groups)
        history = []
        for step in range(updates):
            cache = calibration[step % len(calibration)]
            out = rollout(instance, cache, shared=shared)
            out['metas']['future_occ_labels'] = out['metas'][
                'future_occ_labels'].long()
            if variant == 'A_task_only':
                value, details = objective._task_loss(out)
                residual = sum(weight * state['feature_delta'].square().mean()
                               for weight, state in zip(
                                   objective.horizon_weights,
                                   out['future_query_states']))
                value = value + objective.residual_weight * residual / sum(
                    objective.horizon_weights)
            else:
                value, details = objective(out)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in instance.parameters() if p.requires_grad],
                cfg.grad_max_norm, error_if_nonfinite=True)
            optimizer.step()
            history.append(dict(step=step+1, loss=float(value.detach()),
                                grad_norm=float(norm)))
            if (step+1) % 16 == 0 or step+1 == updates:
                print('[pilot]', variant, history[-1], flush=True)
        instance.eval()
        bank = metric_bank(cfg, [variant])
        with torch.no_grad():
            for cache in validation:
                out = rollout(instance, cache, shared=shared)
                for h in (2, 4, 6):
                    add_metric(bank[variant][h], instance,
                               out['future_pred_points'][h-1],
                               out['future_pred_logits'][h-1], out['metas'], h)
        results[variant] = dict(metrics=bank_results(bank)[variant],
                                training=history, loss_config=dict(lc))
        torch.save(dict(state_dict=instance.state_dict(), variant=variant,
                        updates=updates, seed=seed,
                        shared_projection=shared),
                   output_dir / (variant + '.pth'))
    instance.load_state_dict(initial, strict=True)
    instance.eval()
    return results


def main(args):
    started = time.perf_counter()
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg = Config.fromfile(args.config)
    # All compared branches, teacher included, execute in FP32.
    cfg.amp = False
    trained = build_student(cfg, args.checkpoint)
    baseline = build_student(cfg)
    from train_teacher_aligned import FROZEN_PREFIXES
    original = baseline.state_dict()
    for name, value in trained.state_dict().items():
        if name.startswith(FROZEN_PREFIXES) and not torch.equal(
                value, original[name]):
            raise RuntimeError('frozen perception differs at ' + name)
    del original
    train_loader, train_manifest = loader_for(
        cfg, 'train', args.calibration_samples, args.workers, args.seed)
    val_loader, val_manifest = loader_for(
        cfg, 'val', args.validation_samples, args.workers, args.seed)
    if set(train_manifest['scenes']) & set(val_manifest['scenes']):
        raise RuntimeError('train/validation scenes overlap')
    teacher = MultiHorizonFrozenOPUSV1Teacher(
        cfg.teacher_config, cfg.teacher_checkpoint, 'cuda',
        history_frames=cfg.teacher_history_frames,
        expected_sha256=cfg.teacher_checkpoint_sha256, amp=False)
    torch.cuda.reset_peak_memory_stats()
    calibration = cache_samples(baseline, teacher, train_loader, 'train')
    validation = cache_samples(baseline, teacher, val_loader, 'val')
    del teacher
    # Verify the zero-MLP baseline exactly reproduces the legacy rollout.
    with torch.no_grad():
        cache = calibration[0]
        encoded = tree_to(cache['encoded'], 'cuda')
        stamps = cache['stamps'].cuda()
        persistent = stamps == 0
        positions = encoded['query_points'][:, persistent]
        ego = baseline.plan_head(cache['metas']['temporal_ego_states'].cuda()
                                 .float().reshape(1, -1)).unsqueeze(1)
        legacy = baseline._forecast_rollout(dict(
            features=encoded['query_features'][:, persistent],
            positions=positions, timestamp=positions.new_zeros(
                *positions.shape[:-1], 1), ego=ego,
            all_features=encoded['query_features'],
            all_points=encoded['query_points'], stamps=stamps,
            slot_ids=persistent.nonzero().flatten(), completed_steps=0), 6)
        current = rollout(baseline, cache)
        for actual, expected in zip(current['future_pred_points'], legacy['points']):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        for actual, expected in zip(current['future_pred_logits'], legacy['logits']):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        print('[audit] frozen tensors unchanged; zero-MLP legacy parity exact',
              flush=True)
    matcher = HungarianSparseMatcher(**cfg.loss.matcher)
    adapters = fit_adapters(trained, calibration, matcher, args.ridge)
    report = dict(
        checkpoint=args.checkpoint, checkpoint_sha256=file_sha256(args.checkpoint),
        seed=args.seed, precision='FP32; TF32 disabled',
        calibration=train_manifest, validation=val_manifest,
        calibration_tokens=[c['metas']['sample_idx'] for c in calibration],
        validation_tokens=[c['metas']['sample_idx'] for c in validation],
        intervention_scope='independent horizon, fixed student geometry; '
                           'unmatched and future-born queries retained',
        pilot_scope='cached frozen perception, deterministic augmentation, '
                    'FP32, no warmup, identical finite training samples; '
                    'screening only, not a 16-epoch reproduction')
    report['diagnosis'] = diagnose(trained, baseline, validation,
                                   adapters, matcher, cfg)
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    report['gradient_probes'] = [gradient_probe(trained, cache, objective)
                                  for cache in calibration[:args.gradient_samples]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Persist expensive evaluation even if a later pilot fails.
    output.write_text(json.dumps(_json_safe(report), indent=2), encoding='utf-8')
    if args.pilot_updates:
        # All pilots start from pretrained SparseWorld plus zero-output MLP.
        report['pilots'] = run_pilots(baseline, calibration, validation, cfg,
                                      args.pilot_updates, args.seed,
                                      output.parent)
    report.update(elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                  freeze_audit=_audit_freeze(trained))
    output.write_text(json.dumps(_json_safe(report), indent=2), encoding='utf-8')
    print('[complete]', output, 'elapsed', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py'))
    parser.add_argument('--checkpoint', default=(
        'out/sparseworld_teacher_alignment/16e/epoch_16.pth'))
    parser.add_argument('--output', default=(
        'out/sparseworld_teacher_alignment/diagnosis/report.json'))
    parser.add_argument('--calibration-samples', type=int, default=16)
    parser.add_argument('--validation-samples', type=int, default=32)
    parser.add_argument('--pilot-updates', type=int, default=64)
    parser.add_argument('--gradient-samples', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--ridge', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=20260926)
    main(parser.parse_args())
