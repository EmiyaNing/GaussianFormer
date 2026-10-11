#!/usr/bin/env python3
"""Paired 2048/512 response-level OPUS -> strict SparseWorld experiment.

All inputs and frozen perception outputs are cached per sample.  The only
trainable parameters are the original six SparseWorld forecast submodules.
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
import torch.nn.functional as F
from mmengine import Config
from mmseg.models import build_segmentor
from torch.utils.data import DataLoader, Dataset, Subset

import loss  # noqa: F401 - register losses
import model  # noqa: F401 - register segmentors
from dataset.utils import custom_collate_fn_temporal
from eval_teacher_aligned import _metric
from loss import OPENOCC_LOSS
from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher, file_sha256
from model.response_distillation_forecast import forecast_with_position_gradient
from tools.diagnose_teacher_alignment import add_metric, loader_for, seed_all, tree_to
from tools.diagnose_future_image_joint import HORIZONS, metric_row, rows_metrics, save_json

ARMS = ('task_only', 'semantic', 'geometry', 'joint')
TRAINABLE = ('ego_attention', 'position', 'reg', 'velocity', 'semantic', 'traj')


def build_student(cfg, strict):
    student = build_segmentor(strict.model)
    weights = torch.load(cfg.student_init_checkpoint, map_location='cpu')['state_dict']
    student.load_state_dict(weights, strict=True)
    student.set_epoch(100)
    student.cuda().eval().requires_grad_(False)
    return student


def paths_for(root, split, manifest):
    return [root / split / f'{index:06d}.pth' for index in manifest['indices']]


@torch.no_grad()
def prepare_student(student, loader, manifest, root, split):
    paths = paths_for(root, split, manifest)
    missing = [position for position, path in enumerate(paths) if not path.exists()]
    if not missing:
        return paths
    (root / split).mkdir(parents=True, exist_ok=True)
    source_loader = DataLoader(Subset(loader.dataset, missing), batch_size=1,
                               num_workers=loader.num_workers,
                               collate_fn=custom_collate_fn_temporal, pin_memory=True)
    for position, data in zip(missing, source_loader):
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        metas = dict(data, projection_mat=data['projection_mat'][:, :30],
                     image_wh=data['image_wh'][:, :30])
        result = dict(imgs=imgs[:, :30], metas=metas, points=None)
        result.update(student.extract_img_feat(**result))
        result.update(student.lifter(**result))
        ego_state = metas['temporal_ego_states'].to(result['query_points']).reshape(1, -1)
        ego = student.plan_head(ego_state).unsqueeze(1)
        scale = (torch.tanh(student.points_scale(ego)) + 1.) * .35 + .8
        result['query_points'] = result['query_points'] * scale
        result.update(student.encoder(**result))
        encoded = result['representation'][-1]
        stored = dict(query_features=encoded['query_features'].detach().cpu(),
                      query_points=encoded['query_points'].detach().cpu())
        retain = {key: data[key].detach().cpu() for key in (
            'future_occ_labels', 'future_occ_cam_masks',
            'future_occ_lidar_masks', 'temporal_trajs')}
        retain['future_occ_labels'] = retain['future_occ_labels'].to(torch.uint8)
        record = dict(index=manifest['indices'][position],
                      scene=manifest['scenes'][position], encoded=stored,
                      stamps=result['query_stamps'].detach().cpu(),
                      ego=ego.detach().cpu(), metas=retain)
        temp = paths[position].with_suffix('.pth.tmp')
        torch.save(record, temp)
        temp.replace(paths[position])
        if (position + 1) % 32 == 0 or position + 1 == len(paths):
            print(f'[student cache {split}] {position+1}/{len(paths)}', flush=True)
        del data, imgs, metas, result, encoded, record
    return paths


@torch.no_grad()
def prepare_teacher(teacher, loader, manifest, root, old_root, split):
    new_paths = paths_for(root, split, manifest)
    old_paths = paths_for(old_root, split, manifest)
    paths = [old if old.exists() else new for old, new in zip(old_paths, new_paths)]
    missing = [position for position, path in enumerate(paths) if not path.exists()]
    if not missing:
        return paths
    (root / split).mkdir(parents=True, exist_ok=True)
    teacher_loader = DataLoader(Subset(loader.dataset, missing), batch_size=1,
                                num_workers=loader.num_workers,
                                collate_fn=custom_collate_fn_temporal, pin_memory=True)
    for position, data in zip(missing, teacher_loader):
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        states = {}
        for h in HORIZONS:
            output = teacher.extract_horizon(imgs, data, h)
            states[h] = {key: output[key].cpu() for key in
                         ('query_points_metric', 'point_logits')}
        record = dict(index=manifest['indices'][position],
                      scene=manifest['scenes'][position], states=states)
        temp = new_paths[position].with_suffix('.pth.tmp')
        torch.save(record, temp)
        temp.replace(new_paths[position])
        if (position + 1) % 16 == 0 or position + 1 == len(paths):
            print(f'[teacher cache {split}] {position+1}/{len(paths)}', flush=True)
        del data, imgs, states, record
    return paths


class PairedCache(Dataset):
    def __init__(self, sources, teachers):
        self.sources, self.teachers = sources, teachers

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, position):
        source = torch.load(self.sources[position], map_location='cpu')
        teacher = torch.load(self.teachers[position], map_location='cpu')
        if source['index'] != teacher['index'] or source['scene'] != teacher['scene']:
            raise RuntimeError('Teacher and Student sample mismatch')
        return source, teacher


def paired_loader(sources, teachers, workers, schedule=None):
    dataset = PairedCache(sources, teachers)
    if schedule is not None:
        dataset = Subset(dataset, schedule)
    return DataLoader(dataset, batch_size=1, num_workers=workers,
                      collate_fn=lambda rows: rows[0], pin_memory=False)


def rollout(student, source):
    encoded = tree_to(source['encoded'], 'cuda')
    return forecast_with_position_gradient(
        student, encoded, source['stamps'].cuda(), source['ego'].cuda())


@torch.no_grad()
def parity_and_freeze(student, source):
    student.eval()
    encoded = tree_to(source['encoded'], 'cuda')
    stamps = source['stamps'].cuda()
    ego = source['ego'].cuda()
    current = stamps == 0
    positions = encoded['query_points'][:, current].detach()
    original = student._forecast_rollout(dict(
        features=encoded['query_features'][:, current], positions=positions,
        timestamp=positions.new_zeros(*positions.shape[:-1], 1), ego=ego,
        all_features=encoded['query_features'],
        all_points=encoded['query_points'], stamps=stamps,
        slot_ids=torch.nonzero(current, as_tuple=False).flatten(),
        completed_steps=0), student.future_steps)
    changed = forecast_with_position_gradient(student, encoded, stamps, ego)
    for step in range(student.future_steps):
        torch.testing.assert_close(changed['points'][step], original['points'][step], atol=1e-5, rtol=0)
        torch.testing.assert_close(changed['logits'][step], original['logits'][step], atol=1e-5, rtol=0)
    torch.testing.assert_close(changed['trajectories'], torch.cat(original['trajectories'], 1), atol=1e-5, rtol=0)
    assert [p.shape[1] for p in changed['points']] == [
        (720 + sum(student.future_queries[:step+1])) * student.num_refines
        for step in range(student.future_steps)]


def configure_trainable(student):
    student.eval().requires_grad_(False)
    for name in TRAINABLE:
        getattr(student, name).requires_grad_(True).train()
    names = [name for name, p in student.named_parameters() if p.requires_grad]
    assert names and all(name.split('.')[0] in TRAINABLE for name in names)
    return [p for p in student.parameters() if p.requires_grad], names


def task_loss(student, prediction, source, objective):
    total = prediction['points'][0].new_zeros(())
    details = {}
    for h in HORIZONS:
        labels = source['metas']['future_occ_labels'][0, h-1].cuda().long()
        cls, geo = objective._set_loss(prediction['points'][h-1][0],
                                       prediction['logits'][h-1][0], labels)
        total = total + (objective.lambda_future_cls * cls +
                         objective.lambda_future_pts * geo) / len(HORIZONS)
        details[f'{h*.5:g}s_task_cls'] = float(cls.detach())
        details[f'{h*.5:g}s_task_geo'] = float(geo.detach())
    trajectory = F.mse_loss(prediction['trajectories'],
                            source['metas']['temporal_trajs'].cuda()[:, :6])
    total = total + objective.lambda_traj * trajectory
    details['trajectory'] = float(trajectory.detach())
    return total, details


def teacher_candidates(state, labels, threshold, limit, use_gt_gate):
    points = state['query_points_metric'].cuda().reshape(-1, 3).float()
    logits = state['point_logits'].cuda().reshape(-1, 17).float()
    probs = logits.sigmoid()
    confidence, classes = probs.max(-1)
    keep = confidence >= threshold
    if use_gt_gate:
        voxel = torch.floor((points - points.new_tensor([-40., -40., -1.])) / .4).long()
        inside = ((voxel >= 0) & (voxel < voxel.new_tensor([200, 200, 16]))).all(-1)
        valid = torch.nonzero(inside, as_tuple=False).flatten()
        observed = torch.full_like(classes, 17)
        observed[valid] = labels[voxel[valid, 0], voxel[valid, 1], voxel[valid, 2]]
        keep &= observed == classes
    selected = torch.nonzero(keep, as_tuple=False).flatten()
    if len(selected) > limit:
        selected = selected[confidence[selected].topk(limit).indices]
    return points[selected], logits[selected], int(keep.sum())


def nearest_student(teacher_points, student_points, chunk=128):
    """Detached nearest-point assignment, bounded to 128 x ~40k per chunk."""
    indices, distances = [], []
    with torch.no_grad():
        for block in teacher_points.split(chunk):
            distance = torch.cdist(block[None], student_points[None])[0]
            minimum, index = distance.min(-1)
            indices.append(index)
            distances.append(minimum)
    return torch.cat(indices), torch.cat(distances)


def response_loss(prediction, teacher, source, arm, args):
    value = prediction['points'][0].new_zeros(())
    stats = {}
    for h in HORIZONS:
        labels = source['metas']['future_occ_labels'][0, h-1].cuda().long()
        target_points, target_logits, candidates = teacher_candidates(
            teacher['states'][h], labels, args.teacher_confidence,
            args.max_teacher_points, args.gt_gate)
        if not len(target_points):
            stats[f'{h*.5:g}s_matches'] = 0
            continue
        student_points = prediction['points'][h-1][0]
        student_logits = prediction['logits'][h-1][0]
        nearest, distance = nearest_student(target_points, student_points.detach())
        match = distance <= args.match_radius
        stats[f'{h*.5:g}s_candidates'] = candidates
        stats[f'{h*.5:g}s_matches'] = int(match.sum())
        stats[f'{h*.5:g}s_match_distance_m'] = float(distance[match].mean()) if match.any() else None
        if not match.any():
            continue
        idx = nearest[match]
        if arm in ('geometry', 'joint'):
            geometry = F.smooth_l1_loss(student_points[idx], target_points[match],
                                        beta=.2, reduction='none').sum(-1).mean()
            value = value + args.geometry_weight * geometry / len(HORIZONS)
        if arm in ('semantic', 'joint'):
            temp = args.temperature
            soft = (target_logits[match] / temp).sigmoid()
            semantic = F.binary_cross_entropy_with_logits(
                student_logits[idx] / temp, soft, reduction='mean') * temp * temp
            value = value + args.semantic_weight * semantic / len(HORIZONS)
    return value, stats


@torch.no_grad()
def evaluate(student, sources, teachers, cfg, workers, destination,
             teacher_native=False):
    student.eval()
    rows = []
    for i, (source, teacher) in enumerate(paired_loader(sources, teachers, workers)):
        prediction = None if teacher_native else rollout(student, source)
        row = {}
        for h in HORIZONS:
            state = teacher['states'][h]
            points = (state['query_points_metric'].cuda().flatten(1, 2)
                      if teacher_native else prediction['points'][h-1])
            logits = (state['point_logits'].cuda().flatten(1, 2)
                      if teacher_native else prediction['logits'][h-1])
            metric = _metric(cfg)
            add_metric(metric, student, points, logits, source['metas'], h)
            row[h] = metric_row(metric)
        rows.append(row)
        if (i+1) % 64 == 0 or i+1 == len(sources):
            print(f'[eval {destination.stem}] {i+1}/{len(sources)}', flush=True)
    np.savez_compressed(destination,
                        hist=np.stack([[row[h]['hist'] for h in HORIZONS] for row in rows]),
                        intersection=np.asarray([[row[h]['intersection'] for h in HORIZONS] for row in rows]),
                        union=np.asarray([[row[h]['union'] for h in HORIZONS] for row in rows]))
    return rows_metrics(rows, cfg)


def main(args):
    started = time.time()
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.data_config)
    strict = Config.fromfile(args.strict_config)
    for checkpoint, expected in ((cfg.student_init_checkpoint, cfg.student_init_checkpoint_sha256),
                                 (cfg.teacher_checkpoint, cfg.teacher_checkpoint_sha256)):
        if file_sha256(checkpoint) != expected:
            raise RuntimeError(f'Checkpoint checksum mismatch: {checkpoint}')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_loader, train_manifest = loader_for(cfg, 'train', args.train_samples,
                                              args.workers, args.sample_seed)
    val_loader, val_manifest = loader_for(cfg, 'val', args.val_samples,
                                          args.workers, args.sample_seed)
    if len(train_manifest['indices']) != args.train_samples or len(val_manifest['indices']) != args.val_samples:
        raise RuntimeError('Not enough eligible samples')
    if set(train_manifest['scenes']) & set(val_manifest['scenes']):
        raise RuntimeError('Train/validation scene leakage')
    manifest = dict(args=vars(args), train=train_manifest, validation=val_manifest,
                    student_checkpoint_sha256=cfg.student_init_checkpoint_sha256,
                    teacher_checkpoint_sha256=cfg.teacher_checkpoint_sha256,
                    teacher_uses_future_images=True,
                    inference_uses_future_images=False,
                    trainable_modules=list(TRAINABLE),
                    frozen_modules=['img_backbone', 'img_neck', 'lifter', 'encoder',
                                    'head', 'plan_head', 'points_scale'])
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        for key in ('train', 'validation', 'student_checkpoint_sha256', 'teacher_checkpoint_sha256'):
            if prior[key] != manifest[key]:
                raise RuntimeError(f'Incompatible resumed experiment: {key}')
        for key in ('data_config', 'strict_config', 'train_samples', 'val_samples',
                    'updates', 'sample_seed', 'lr', 'temperature',
                    'teacher_confidence', 'max_teacher_points', 'match_radius',
                    'semantic_weight', 'geometry_weight', 'warmup_updates',
                    'gt_gate'):
            if prior['args'][key] != vars(args)[key]:
                raise RuntimeError(f'Incompatible resumed experiment option: {key}')
    else:
        save_json(manifest, manifest_path)
    source_root = output / 'source_cache'
    train_sources = paths_for(source_root, 'train', train_manifest)
    val_sources = paths_for(source_root, 'val', val_manifest)
    student = build_student(cfg, strict)
    if not args.skip_prepare:
        train_sources = prepare_student(student, train_loader, train_manifest, source_root, 'train')
        val_sources = prepare_student(student, val_loader, val_manifest, source_root, 'val')
    if any(not path.exists() for path in train_sources + val_sources):
        raise RuntimeError('Incomplete Student cache')
    parity_and_freeze(student, torch.load(train_sources[0], map_location='cpu'))
    del student
    torch.cuda.empty_cache()
    old_teacher = Path(args.old_teacher_cache)
    teacher_root = output / 'teacher_cache'
    train_teachers = [old if old.exists() else new for old, new in zip(
        paths_for(old_teacher, 'train', train_manifest),
        paths_for(teacher_root, 'train', train_manifest))]
    val_teachers = [old if old.exists() else new for old, new in zip(
        paths_for(old_teacher, 'val', val_manifest),
        paths_for(teacher_root, 'val', val_manifest))]
    if not args.skip_prepare and any(not p.exists() for p in train_teachers + val_teachers):
        teacher = MultiHorizonFrozenOPUSV1Teacher(
            cfg.teacher_config, cfg.teacher_checkpoint, 'cuda',
            history_frames=cfg.teacher_history_frames,
            expected_sha256=cfg.teacher_checkpoint_sha256, amp=False)
        train_teachers = prepare_teacher(teacher, train_loader, train_manifest,
                                          teacher_root, old_teacher, 'train')
        val_teachers = prepare_teacher(teacher, val_loader, val_manifest,
                                        teacher_root, old_teacher, 'val')
        del teacher
        torch.cuda.empty_cache()
    if any(not path.exists() for path in train_teachers + val_teachers):
        raise RuntimeError('Incomplete Teacher cache')
    del train_loader, val_loader
    if args.prepare_only:
        print('[prepare complete]', output, flush=True)
        return
    report_path = output / 'report.json'
    report = (json.loads(report_path.read_text()) if report_path.exists()
              else dict(manifest='manifest.json', runs={}, elapsed_seconds=0))
    objective = OPENOCC_LOSS.build(copy.deepcopy(cfg.base_sparseworld_loss)).cuda()
    if 'teacher_native' not in report:
        student = build_student(cfg, strict)
        report['teacher_native'] = evaluate(student, val_sources, val_teachers, cfg,
                                            args.workers, output / 'teacher_native.npz', True)
        report['pretrained_student'] = evaluate(student, val_sources, val_teachers, cfg,
                                                args.workers, output / 'pretrained_student.npz')
        save_json(report, report_path)
        del student
        torch.cuda.empty_cache()
    for seed in args.training_seeds:
        report['runs'].setdefault(str(seed), {})
        for arm in args.arms:
            if arm in report['runs'][str(seed)]:
                continue
            seed_all(seed)
            student = build_student(cfg, strict)
            params, trainable_names = configure_trainable(student)
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
            schedule = []
            rng = np.random.RandomState(seed)
            while len(schedule) < args.updates:
                schedule.extend(rng.permutation(args.train_samples).tolist())
            checkpoint = output / f'{seed}_{arm}_latest.pth'
            history, start = [], 0
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu')
                student.load_state_dict(saved['model'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.cuda()
                start = saved['step']
                history = saved['history']
            for step, (source, teacher) in enumerate(paired_loader(
                    train_sources, train_teachers, args.workers,
                    schedule[start:args.updates]), start=start):
                optimizer.zero_grad(set_to_none=True)
                prediction = rollout(student, source)
                task, details = task_loss(student, prediction, source, objective)
                kd, stats = response_loss(prediction, teacher, source, arm, args) if arm != 'task_only' else (task.new_zeros(()), {})
                kd_scale = min(1., (step + 1) / args.warmup_updates)
                value = task + kd_scale * kd
                if not torch.isfinite(value):
                    raise RuntimeError(f'Non-finite loss at seed={seed} arm={arm} step={step}')
                value.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
                optimizer.step()
                if step < 3 or (step+1) % 64 == 0:
                    row = dict(step=step+1, total=float(value.detach()), task=float(task.detach()),
                               kd=float(kd.detach()), grad_norm=float(norm), **stats)
                    history.append(row)
                    print('[train]', seed, arm, step+1, row, flush=True)
                if (step+1) % 256 == 0 or step+1 == args.updates:
                    torch.save(dict(model={key: value.detach().cpu()
                                           for key, value in student.state_dict().items()},
                                    optimizer=optimizer.state_dict(), step=step+1,
                                    history=history, seed=seed, arm=arm), checkpoint)
                del prediction, task, kd, value
            metrics = evaluate(student, val_sources, val_teachers, cfg, args.workers,
                               output / f'{seed}_{arm}.npz')
            report['runs'][str(seed)][arm] = dict(metrics=metrics, history=history,
                                                  trainable_parameter_count=sum(p.numel() for p in params),
                                                  trainable_parameter_names=trainable_names)
            report['elapsed_seconds'] = time.time() - started
            report['peak_cuda_memory_gib'] = max(
                report.get('peak_cuda_memory_gib', 0.),
                torch.cuda.max_memory_allocated() / 2**30)
            save_json(report, report_path)
            print('[arm complete]', seed, arm,
                  {h: round(v['mIoU'], 4) for h, v in metrics.items()}, flush=True)
            del student, optimizer
            torch.cuda.empty_cache()
    print('[complete]', report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    parser.add_argument('--strict-config', default='config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    parser.add_argument('--old-teacher-cache', default='out/future_alignment_hypotheses/teacher_query_cache_512x256')
    parser.add_argument('--output-dir', default='out/future_distillation_response_based/2048x512')
    parser.add_argument('--train-samples', type=int, default=2048)
    parser.add_argument('--val-samples', type=int, default=512)
    parser.add_argument('--updates', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[20260926])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--temperature', type=float, default=2.)
    parser.add_argument('--teacher-confidence', type=float, default=.55)
    parser.add_argument('--max-teacher-points', type=int, default=2048)
    parser.add_argument('--match-radius', type=float, default=1.2)
    parser.add_argument('--semantic-weight', type=float, default=.2)
    parser.add_argument('--geometry-weight', type=float, default=.1)
    parser.add_argument('--warmup-updates', type=int, default=256)
    parser.add_argument('--gt-gate', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--skip-prepare', action='store_true')
    main(parser.parse_args())
