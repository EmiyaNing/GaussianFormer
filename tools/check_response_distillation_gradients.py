#!/usr/bin/env python3
"""Audit which original SparseWorld forecast branches receive KD gradients."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mmengine import Config

from tools.experiment_future_response_distillation import (
    build_student, configure_trainable, PairedCache, response_loss, rollout)


def group_norms(student):
    return {name: float(sum((p.grad.detach().square().sum() for p in
                             getattr(student, name).parameters() if p.grad is not None),
                            torch.zeros((), device='cuda')).sqrt())
            for name in ('ego_attention', 'position', 'reg', 'velocity', 'semantic', 'traj')}


def main(args):
    cfg = Config.fromfile('config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    strict = Config.fromfile('config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    root = Path(args.output_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    index = manifest['train']['indices'][0]
    source = root / 'source_cache' / 'train' / f'{index:06d}.pth'
    old_teacher = Path('out/future_alignment_hypotheses/teacher_query_cache_512x256') / 'train' / f'{index:06d}.pth'
    teacher = old_teacher if old_teacher.exists() else root / 'teacher_cache' / 'train' / f'{index:06d}.pth'
    student = build_student(cfg, strict)
    configure_trainable(student)
    sample, target = PairedCache([source], [teacher])[0]
    result = {}
    for arm in ('geometry', 'semantic', 'joint'):
        student.zero_grad(set_to_none=True)
        prediction = rollout(student, sample)
        kd, stats = response_loss(prediction, target, sample, arm, args)
        kd.backward()
        result[arm] = dict(kd=float(kd.detach()), grad_norms=group_norms(student),
                           matches={k: v for k, v in stats.items() if k.endswith('matches')})
    frozen_grad = [name for name, param in student.named_parameters()
                   if name.split('.')[0] not in ('ego_attention', 'position',
                     'reg', 'velocity', 'semantic', 'traj') and param.grad is not None]
    result['frozen_parameter_grads'] = frozen_grad
    (root / 'gradient_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if frozen_grad or result['geometry']['grad_norms']['reg'] <= 0 or result['semantic']['grad_norms']['semantic'] <= 0:
        raise RuntimeError('Unexpected response-distillation gradient path')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='out/future_distillation_response_based/2048x512')
    parser.add_argument('--teacher-confidence', type=float, default=.55)
    parser.add_argument('--max-teacher-points', type=int, default=2048)
    parser.add_argument('--match-radius', type=float, default=1.2)
    parser.add_argument('--semantic-weight', type=float, default=.2)
    parser.add_argument('--geometry-weight', type=float, default=.1)
    parser.add_argument('--temperature', type=float, default=2.)
    parser.add_argument('--gt-gate', action='store_true', default=True)
    main(parser.parse_args())
