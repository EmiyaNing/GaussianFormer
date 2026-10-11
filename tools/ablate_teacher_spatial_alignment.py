#!/usr/bin/env python3
"""Small paired screen: semantic Hungarian vs position-only Teacher matching.

This is an isolated experiment tool. It does not change the production model.
All query slots (including future-born queries) keep occupancy supervision.
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
from mmengine import Config

from loss import OPENOCC_LOSS
from eval_teacher_aligned import _json_safe
from tools.diagnose_teacher_alignment import (
    seed_all, build_student, loader_for, cache_samples, rollout,
    metric_bank, bank_results, add_metric, tree_to)
from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher, file_sha256
from train_teacher_aligned import _audit_freeze


def _objective_config(cfg, mode):
    loss_cfg = copy.deepcopy(cfg.loss)
    loss_cfg.pop('_delete_', None)
    loss_cfg.alignment_warmup_iters = 0
    if mode in ('spatial_only', 'spatial_shuffled'):
        matcher = dict(loss_cfg.matcher)
        matcher['semantic_weight'] = 0.
        matcher['feature_weight'] = 0.
        loss_cfg.matcher = matcher
    return loss_cfg


def _permute_teacher_content(out):
    # Keep teacher locations fixed, but associate each location with another
    # query's semantic and feature target. The position-only matcher therefore
    # selects exactly the same geometric pairing as the unshuffled arm.
    for state in out['teacher_states']:
        state['query_features'] = state['query_features'].roll(1, dims=1)
        state['semantic_logits'] = state['semantic_logits'].roll(1, dims=1)


def train_arm(instance, calibration, validation, cfg, updates, seed,
              output_dir, mode):
    initial = tree_to(instance.state_dict(), 'cpu')
    seed_all(seed)
    instance.load_state_dict(initial, strict=True)
    instance.train()
    loss_cfg = _objective_config(cfg, mode)
    objective = OPENOCC_LOSS.build(loss_cfg).cuda()
    groups = []
    for name, parameter in instance.named_parameters():
        if parameter.requires_grad:
            lr = (cfg.optimizer.optimizer.lr if name.startswith(
                ('future_feature_mlp.', 'student_projection.'))
                  else cfg.optimizer.existing_future_lr)
            groups.append(dict(
                params=[parameter], lr=lr,
                weight_decay=(0. if name.startswith('student_projection.')
                              else cfg.optimizer.optimizer.weight_decay)))
    optimizer = torch.optim.AdamW(groups)
    history = []
    for step in range(updates):
        cache = calibration[step % len(calibration)]
        out = rollout(instance, cache)
        out['metas']['future_occ_labels'] = out['metas'][
            'future_occ_labels'].long()
        if mode == 'spatial_shuffled':
            _permute_teacher_content(out)
        if mode == 'task_only':
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
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in instance.parameters() if p.requires_grad],
            cfg.grad_max_norm, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 128 == 0 or step + 1 == updates:
            history.append(dict(step=step + 1, loss=float(value.detach()),
                                grad_norm=float(grad_norm)))
            print('[train]', seed, mode, history[-1], flush=True)

    instance.eval()
    bank = metric_bank(cfg, [mode])
    with torch.no_grad():
        for cache in validation:
            out = rollout(instance, cache)
            for horizon in (2, 4, 6):
                add_metric(bank[mode][horizon], instance,
                           out['future_pred_points'][horizon - 1],
                           out['future_pred_logits'][horizon - 1],
                           out['metas'], horizon)
    result = dict(metrics=bank_results(bank)[mode], training=history,
                  loss_config=dict(loss_cfg))
    torch.save(dict(state_dict=instance.state_dict(), mode=mode,
                    updates=updates, seed=seed),
               output_dir / f'{mode}.pth')
    instance.load_state_dict(initial, strict=True)
    instance.eval()
    return result


def summarize(runs):
    names = ('task_only', 'hungarian', 'spatial_only', 'spatial_shuffled')
    summary = {}
    for horizon in ('1s', '2s', '3s'):
        summary[horizon] = {}
        for metric in ('mIoU', 'OccIoU'):
            values = {
                name: [runs[seed][name]['metrics'][horizon][metric]
                       for seed in runs]
                for name in names
            }
            mean = {name: float(np.mean(items))
                    for name, items in values.items()}
            summary[horizon][metric] = dict(
                arms={name: dict(mean=mean[name],
                                 std=float(np.std(
                                     values[name],
                                     ddof=1 if len(values[name]) > 1 else 0)))
                      for name in names},
                spatial_minus_hungarian=[
                    float(runs[seed]['spatial_only']['metrics'][horizon][metric]
                          - runs[seed]['hungarian']['metrics'][horizon][metric])
                    for seed in runs],
                spatial_minus_shuffled=[
                    float(runs[seed]['spatial_only']['metrics'][horizon][metric]
                          - runs[seed]['spatial_shuffled']['metrics'][horizon][metric])
                    for seed in runs])
    return summary


def main(args):
    if min(args.train_samples, args.val_samples, args.updates) < 1:
        raise ValueError('sample and update counts must be positive')
    started = time.perf_counter()
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg = Config.fromfile(args.config)
    cfg.amp = False
    student = build_student(cfg)
    baseline = build_student(cfg)
    train_loader, train_manifest = loader_for(
        cfg, 'train', args.train_samples, args.workers, args.sample_seed)
    val_loader, val_manifest = loader_for(
        cfg, 'val', args.val_samples, args.workers, args.sample_seed)
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
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        args=vars(args), checkpoint=cfg.student_init_checkpoint,
        checkpoint_sha256=file_sha256(cfg.student_init_checkpoint),
        teacher_checkpoint=cfg.teacher_checkpoint,
        teacher_sha256=cfg.teacher_checkpoint_sha256,
        training=train_manifest, validation=val_manifest,
        precision='FP32; TF32 disabled; batch size 1',
        initialization='same pretrained SparseWorld and zero-output future branch '
                       'for every arm and seed',
        comparison='default semantic/position Hungarian vs position-only '
                   'Hungarian vs position-only with shuffled feature/semantic '
                   'targets; all occupancy query slots remain supervised',
        scope='short cached-feature mechanism screen; not the full common-grid '
              'training experiment or a 16-epoch reproduction')
    report['runs'] = {}
    modes = ('task_only', 'hungarian', 'spatial_only', 'spatial_shuffled')
    for seed in args.training_seeds:
        seed_dir = output / f'seed_{seed}'
        seed_dir.mkdir(exist_ok=True)
        report['runs'][str(seed)] = {}
        for mode in modes:
            report['runs'][str(seed)][mode] = train_arm(
                student, calibration, validation, cfg, args.updates, seed,
                seed_dir, mode)
            if all(set(run) == set(modes) for run in report['runs'].values()):
                report['partial_summary'] = summarize(report['runs'])
            (output / 'report.partial.json').write_text(
                json.dumps(_json_safe(report), indent=2, allow_nan=False),
                encoding='utf-8')
    report.update(summary=summarize(report['runs']),
                  elapsed_seconds=time.perf_counter() - started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                  freeze_audit=_audit_freeze(baseline))
    report.pop('partial_summary', None)
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(_json_safe(report), indent=2,
                                      allow_nan=False), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)
    print('[complete]', report_path, 'elapsed', report['elapsed_seconds'],
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py'))
    parser.add_argument('--train-samples', type=int, default=128)
    parser.add_argument('--val-samples', type=int, default=64)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--training-seeds', type=int, nargs='+',
                        default=[20260926, 20260927, 20260928])
    parser.add_argument('--output-dir', default=(
        'out/future_alignment_hypotheses/H2_spatial_alignment128'))
    main(parser.parse_args())
