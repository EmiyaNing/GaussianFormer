#!/usr/bin/env python3
"""Measure query coverage for semantic vs position-only Hungarian matching."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config

from eval_teacher_aligned import _json_safe
from tools.diagnose_teacher_alignment import (
    seed_all, build_student, loader_for, cache_samples, rollout, match)
from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher
from model.sparse_latent_dynamics import HungarianSparseMatcher


def coverage_for(caches, student, matchers, split):
    horizons = (2, 4, 6)  # 1s, 2s, 3s outputs
    values = {name: {h: [] for h in horizons} for name in matchers}
    for index, cache in enumerate(caches):
        output = rollout(student, cache)
        for horizon in horizons:
            state = output['future_query_states'][horizon - 1]
            target = output['teacher_states'][horizon - 1]
            total = int(state['query_features'].shape[1])
            for name, matcher in matchers.items():
                prediction, target_index = match(matcher, state, target)
                values[name][horizon].append(dict(
                    matched=int(prediction.numel()),
                    student_queries=total,
                    ratio=float(prediction.numel() / max(total, 1))))
        if (index + 1) % 16 == 0 or index + 1 == len(caches):
            print(f'[{split} coverage] {index+1}/{len(caches)}', flush=True)
    summary = {}
    for name, by_horizon in values.items():
        summary[name] = {}
        for horizon, rows in by_horizon.items():
            summary[name][f'{horizon * .5:g}s'] = dict(
                mean_matched=float(np.mean([row['matched'] for row in rows])),
                mean_student_queries=float(np.mean(
                    [row['student_queries'] for row in rows])),
                mean_ratio=float(np.mean([row['ratio'] for row in rows])),
                std_ratio=float(np.std([row['ratio'] for row in rows], ddof=1))
                if len(rows) > 1 else 0.)
    return summary


def main(args):
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.config)
    cfg.amp = False
    student = build_student(cfg)
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
    train_cache = cache_samples(student, teacher, train_loader, 'train')
    val_cache = cache_samples(student, teacher, val_loader, 'val')
    del teacher
    matcher_cfg = dict(cfg.loss.matcher)
    position_cfg = dict(matcher_cfg)
    position_cfg['semantic_weight'] = 0.
    position_cfg['feature_weight'] = 0.
    matchers = dict(
        semantic_position_hungarian=HungarianSparseMatcher(**matcher_cfg),
        position_only_hungarian=HungarianSparseMatcher(**position_cfg))
    result = dict(
        protocol='same pretrained SparseWorld rollout and same Teacher targets; '
                 'this audit only compares retained matching coverage, not task accuracy',
        training_manifest=train_manifest, validation_manifest=val_manifest,
        train=coverage_for(train_cache, student, matchers, 'train'),
        validation=coverage_for(val_cache, student, matchers, 'val'))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(result), indent=2,
                                 allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print('[complete]', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py'))
    parser.add_argument('--train-samples', type=int, default=128)
    parser.add_argument('--val-samples', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--output', default=(
        'out/future_alignment_hypotheses/H2_position_only128_20260926/coverage.json'))
    main(parser.parse_args())
