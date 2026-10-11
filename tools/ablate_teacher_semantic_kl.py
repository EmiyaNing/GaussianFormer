#!/usr/bin/env python3
"""Paired semantic-KL ablation on fixed scenes, with repeated dropout seeds."""
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
    seed_all, build_student, loader_for, cache_samples,
    gradient_probe, pilot_loss_config, run_pilots)
from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher, file_sha256
from train_teacher_aligned import _audit_freeze


def main(args):
    if min(args.train_samples, args.val_samples, args.updates) < 1:
        raise ValueError('sample and update counts must be positive')
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError('training seeds must be unique')
    started = time.perf_counter()
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.config)
    cfg.amp = False
    # Match the previous diagnostic's constructor RNG consumption so the
    # pretrained + zero-MLP initialization is reproducible against its pilots.
    trained = build_student(cfg, args.checkpoint)
    baseline = build_student(cfg)
    train_loader, train_manifest = loader_for(
        cfg, 'train', args.train_samples, args.workers, args.sample_seed)
    val_loader, val_manifest = loader_for(
        cfg, 'val', args.val_samples, args.workers, args.sample_seed)
    assert not set(train_manifest['scenes']) & set(val_manifest['scenes'])
    teacher = MultiHorizonFrozenOPUSV1Teacher(
        cfg.teacher_config, cfg.teacher_checkpoint, 'cuda',
        history_frames=cfg.teacher_history_frames,
        expected_sha256=cfg.teacher_checkpoint_sha256, amp=False)
    torch.cuda.reset_peak_memory_stats()
    calibration = cache_samples(baseline, teacher, train_loader, 'train')
    validation = cache_samples(baseline, teacher, val_loader, 'val')
    del teacher
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    full = pilot_loss_config(cfg, 'C_full')
    off = pilot_loss_config(cfg, 'E_no_semantic_kl')
    differing = [k for k in full if full[k] != off[k]]
    assert differing == ['semantic_weight'], differing
    report = dict(
        sample_seed=args.sample_seed, training_seeds=args.training_seeds,
        calibration=train_manifest, validation=val_manifest,
        calibration_tokens=[c['metas']['sample_idx'] for c in calibration],
        validation_tokens=[c['metas']['sample_idx'] for c in validation],
        updates_per_arm=args.updates,
        precision='FP32; TF32 disabled; no warmup; batch size 1',
        initialization='identical pretrained SparseWorld + zero-output MLP '
                       'for every arm and seed; seeds vary dropout RNG only',
        baseline_checkpoint=cfg.student_init_checkpoint,
        baseline_sha256=file_sha256(cfg.student_init_checkpoint),
        teacher_sha256=cfg.teacher_checkpoint_sha256,
        trained_checkpoint=args.checkpoint,
        trained_sha256=file_sha256(args.checkpoint),
        comparison='C_full versus E_no_semantic_kl changes semantic_weight '
                   'only; semantic matcher cost remains unchanged',
        scope=f'short cached-feature screen; {len(calibration)} training scenes; '
              'not a complete 16-epoch or independent-initialization study',
        objectives=dict(full=dict(full), no_kl=dict(off)))
    report['gradient_probes'] = {}
    for name, lc in [('full', full), ('no_kl', off)]:
        lc = copy.deepcopy(lc)
        objective = OPENOCC_LOSS.build(lc).cuda()
        report['gradient_probes'][name] = [
            gradient_probe(trained, cache, objective)
            for cache in calibration[:args.gradient_samples]]
    del trained
    report['runs'] = {}
    for seed in args.training_seeds:
        directory = output.parent / ('seed_' + str(seed))
        directory.mkdir(exist_ok=True)
        report['runs'][str(seed)] = run_pilots(
            baseline, calibration, validation, cfg, args.updates, seed,
            directory, variants=('A_task_only', 'C_full', 'E_no_semantic_kl'))
        output.write_text(json.dumps(_json_safe(report), indent=2,
                                     allow_nan=False), encoding='utf-8')
    summary = {}
    for horizon in ('1s', '2s', '3s'):
        summary[horizon] = {}
        for metric in ('mIoU', 'OccIoU'):
            values = {variant: [r[variant]['metrics'][horizon][metric]
                               for r in report['runs'].values()]
                      for variant in ('A_task_only', 'C_full',
                                      'E_no_semantic_kl')}
            delta = np.array(values['E_no_semantic_kl']) - values['C_full']
            summary[horizon][metric] = dict(
                arms={k: dict(mean=float(np.mean(v)),
                              std=float(np.std(v, ddof=1)) if len(v)>1 else 0.)
                      for k,v in values.items()},
                no_kl_minus_full=delta.tolist(),
                paired_delta_mean=float(delta.mean()))
    report.update(summary=summary,
                  elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                  freeze_audit=_audit_freeze(baseline))
    output.write_text(json.dumps(_json_safe(report), indent=2,
                                 allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('[complete]', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py'))
    parser.add_argument('--checkpoint', default=(
        'out/sparseworld_teacher_alignment/16e/epoch_16.pth'))
    parser.add_argument('--output', default=(
        'out/sparseworld_teacher_alignment/semantic_kl_ablation/report.json'))
    parser.add_argument('--train-samples', type=int, default=16)
    parser.add_argument('--val-samples', type=int, default=64)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--gradient-samples', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--training-seeds', type=int, nargs='+',
                        default=[20260926, 20260927, 20260928])
    main(parser.parse_args())
