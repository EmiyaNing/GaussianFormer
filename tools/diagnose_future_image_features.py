#!/usr/bin/env python3
"""Does reading actual future RGB at SparseWorld future queries help?

This privileged diagnostic is not a deployable forecaster. Original model,
OPUS reader and query proposals stay frozen. Only an identical small fusion
adapter is trained in each controlled arm, on disjoint training scenes.
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
from mmseg.models import build_segmentor

from tools.diagnose_teacher_alignment import (
    seed_all, tree_to, loader_for, add_metric, metric_bank, bank_results)
from eval_teacher_aligned import _metric, _json_safe
from loss import OPENOCC_LOSS
from model.future_image_probe import (
    FrozenOPUSImageReader, FutureImageFusionProbe, decoded_offset,
    probe_prediction)
from model.multi_horizon_teacher import file_sha256

HORIZONS = (2, 4, 6)
ARMS = ('query_only', 'current_image', 'future_image')


def select_extra(state, arm, intervention='normal'):
    if arm == 'query_only' or intervention == 'zero':
        return torch.zeros_like(state['future_extra'])
    extra = state['current_extra'] if arm == 'current_image' else state['future_extra']
    if intervention == 'shuffled':
        # Permute only visible slots; keep the visibility pattern unchanged.
        extra = extra.clone()
        indices = torch.nonzero(state['future_visible'][0]).flatten()
        extra[:, indices] = extra[:, indices.roll(max(1, len(indices) // 2))]
    return extra


@torch.no_grad()
def cache_split(student, reader, loader, cfg, label):
    caches = []
    baseline = metric_bank(cfg, ['baseline'])
    current_metric = _metric(cfg)
    visibility = {h: [] for h in HORIZONS}
    for index, data in enumerate(loader):
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        sm = dict(data, projection_mat=data['projection_mat'][:, :30],
                  image_wh=data['image_wh'][:, :30])
        output = student(imgs=imgs[:, :30], metas=sm,
                         forecast_snapshot_step=HORIZONS)
        states = {}
        current_maps = [f[:, :6].contiguous() for f in output['ms_img_feats']]
        for h in HORIZONS:
            snap = output['forecast_snapshots'][h]
            query, normalized_points = snap['features'], snap['positions']
            points = student._world(normalized_points)
            logits = snap['semantic']
            assert query.shape[1] == 720 + sum(student.future_queries[:h])
            # Appended future projections already use their native future ego.
            packed_index = cfg.teacher_history_frames + h
            begin = packed_index * 6
            fm = dict(projection_mat=data['projection_mat'][:, begin:begin+6],
                      image_wh=data['image_wh'][:, begin:begin+6])
            future_maps = student.extract_img_feat(
                imgs[:, begin:begin+6])['ms_img_feats']
            future_extra, future_visible = reader(
                normalized_points, query, future_maps, fm)
            # Current-image control reads the SAME physical future query
            # locations using GT ego transform, not unwarped wrong geometry.
            cm = dict(projection_mat=data['projection_mat'][:, :6] @
                      data['future_ego_to_current'][:, h-1, None],
                      image_wh=data['image_wh'][:, :6])
            current_extra, current_visible = reader(
                normalized_points, query, current_maps, cm)
            states[h] = dict(query=query, points=points,
                             logits=logits.flatten(1, 2),
                             offset=decoded_offset(student, query, logits),
                             future_extra=future_extra,
                             current_extra=current_extra,
                             future_visible=future_visible,
                             current_visible=current_visible)
            # Checkpoint routing can reassign timestamps to arbitrary slots.
            born = snap['stamps'][snap['slot_ids']] > 0
            visibility[h].append(dict(
                all=float(future_visible.float().mean()),
                persistent=float(future_visible[:, ~born].float().mean()),
                born=float(future_visible[:, born].float().mean()),
                current_image=float(current_visible.float().mean())))
            add_metric(baseline['baseline'][h], student,
                       points.flatten(1, 2), logits.flatten(1, 2), data, h)
            del future_maps
        current_metric.add_batch(
            output['final_occ_grid'][0].cpu().numpy(),
            data['occ_label'][0].cpu().numpy(),
            data['occ_lidar_mask'][0].cpu().numpy(),
            data['occ_cam_mask'][0].cpu().numpy())
        keep = ('future_occ_labels', 'future_occ_cam_masks',
                'future_occ_lidar_masks', 'sample_idx', 'scene_token')
        metas = {k: data[k] for k in keep}
        metas['future_occ_labels'] = metas['future_occ_labels'].to(torch.uint8)
        caches.append(tree_to(dict(states=states, metas=metas), 'cpu'))
        del output, imgs, data, current_maps
        if (index + 1) % 8 == 0 or index + 1 == len(loader):
            print(f'[cache {label}] {index+1}/{len(loader)}', flush=True)
    return caches, dict(
        future=bank_results(baseline)['baseline'], current=current_metric.results(),
        visibility={str(h): {k: float(np.mean([v[k] for v in values]))
                            for k in values[0]} for h, values in visibility.items()})


@torch.no_grad()
def evaluate(student, probe, caches, cfg, arm, mode, intervention='normal'):
    bank = metric_bank(cfg, [arm])
    for cache in caches:
        data = tree_to(cache, 'cuda')
        for h in HORIZONS:
            state = data['states'][h]
            points, logits, _ = probe_prediction(
                student, probe, state, select_extra(state, arm, intervention), mode)
            add_metric(bank[arm][h], student, points, logits, data['metas'], h)
    return bank_results(bank)[arm]


def train_arm(student, probe, caches, objective, arm, mode, args):
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=.01)
    rng = np.random.RandomState(args.active_seed)
    schedule = []
    while len(schedule) < args.updates:
        schedule.extend(rng.permutation(len(caches)).tolist())
    history = []
    probe.train()
    for step, index in enumerate(schedule[:args.updates]):
        data = tree_to(caches[index], 'cuda')
        value = 0.
        for h in HORIZONS:
            state = data['states'][h]
            points, logits, delta = probe_prediction(
                student, probe, state, select_extra(state, arm), mode)
            cls, pts = objective._set_loss(
                points[0], logits[0], data['metas']['future_occ_labels'][0, h-1].long())
            value = value + (objective.lambda_future_cls * cls +
                             objective.lambda_future_pts * pts +
                             args.residual_weight * delta.square().mean()) / len(HORIZONS)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step+1, loss=float(value.detach()), grad_norm=float(norm)))
        if (step + 1) % 64 == 0 or step + 1 == args.updates:
            print('[train]', mode, args.active_seed, arm, history[-1], flush=True)
    probe.eval()
    return history


def save_report(report, path):
    path.write_text(json.dumps(_json_safe(report), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding='utf-8')


def main(args):
    started = time.perf_counter()
    if min(args.train_samples, args.val_samples, args.updates) < 1:
        raise ValueError('sample/update counts must be positive')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'report.json').exists():
        raise FileExistsError('Use a new output directory to preserve prior results')
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.data_config)
    model_cfg = Config.fromfile('config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    student = build_segmentor(model_cfg.model)
    checkpoint = cfg.student_init_checkpoint
    digest = file_sha256(checkpoint)
    assert digest == cfg.student_init_checkpoint_sha256
    student.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
    student.set_epoch(100)
    student.requires_grad_(False).cuda().eval()
    reader = FrozenOPUSImageReader(student.encoder.layers[-1]).cuda()
    initial_student = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    train_loader, train_manifest = loader_for(
        cfg, 'train', args.train_samples, args.workers, args.sample_seed)
    val_loader, val_manifest = loader_for(
        cfg, 'val', args.val_samples, args.workers, args.sample_seed)
    assert not set(train_manifest['scenes']) & set(val_manifest['scenes'])
    train, train_baseline = cache_split(student, reader, train_loader, cfg, 'train')
    validation, baseline = cache_split(student, reader, val_loader, cfg, 'val')
    del reader
    report = dict(
        args=vars(args).copy(), checkpoint=checkpoint, checkpoint_sha256=digest,
        training=train_manifest, validation=val_manifest,
        train_tokens=[c['metas']['sample_idx'] for c in train],
        val_tokens=[c['metas']['sample_idx'] for c in validation],
        baseline=baseline, training_baseline=train_baseline,
        protocol=dict(
            precision='FP32, TF32 disabled', trainable='fusion adapter only',
            horizons_seconds=[1, 2, 3], query_counts=[840, 960, 1040],
            future_reader='pretrained last OPUS sampling/mixing; one RGB timestamp; '
                          'sampled features repeated in mixer temporal slots',
            recurrence='independent intervention at each horizon; original rollout fixed',
            current_image_geometry='GT future-ego-to-current transform; privileged control',
            future_rgb='real future six-camera RGB; not a deployable forecasting result',
            joint='redecode offset difference around frozen baseline pre-refinement centre'),
        runs={})
    objective = OPENOCC_LOSS.build(copy.deepcopy(cfg.base_sparseworld_loss)).cuda()
    # Exact no-op and usable task gradients are prerequisites for training.
    probe = FutureImageFusionProbe().cuda()
    one = tree_to(train[0], 'cuda')
    for mode in args.modes:
        for h in HORIZONS:
            state = one['states'][h]
            points, logits, _ = probe_prediction(student, probe, state,
                                                 state['future_extra'], mode)
            torch.testing.assert_close(points, state['points'].flatten(1, 2), rtol=0, atol=1e-5)
            torch.testing.assert_close(logits, state['logits'], rtol=0, atol=1e-6)
    report['zero_adapter_baseline_check'] = True
    save_report(report, output_dir / 'report.json')
    for mode in args.modes:
        report['runs'][mode] = {}
        for seed in args.training_seeds:
            args.active_seed = seed
            seed_all(seed)
            probe = FutureImageFusionProbe().cuda()
            initial = copy.deepcopy(probe.state_dict())
            report['runs'][mode][str(seed)] = {}
            for arm in ARMS:
                seed_all(seed)
                probe.load_state_dict(initial)
                history = train_arm(student, probe, train, objective, arm, mode, args)
                result = dict(training=history,
                              metrics=evaluate(student, probe, validation, cfg, arm, mode))
                if arm == 'future_image':
                    result['shuffled'] = evaluate(student, probe, validation, cfg, arm, mode, 'shuffled')
                    result['zero_image'] = evaluate(student, probe, validation, cfg, arm, mode, 'zero')
                filename = f'{mode}_{seed}_{arm}.pth'
                torch.save(dict(state_dict=probe.state_dict(), mode=mode, seed=seed,
                                arm=arm, updates=args.updates, baseline_sha256=digest),
                           output_dir / filename)
                result['checkpoint'] = filename
                report['runs'][mode][str(seed)][arm] = result
                save_report(report, output_dir / 'report.json')
                print('[eval]', mode, seed, arm,
                      {h: round(v['mIoU'], 4) for h, v in result['metrics'].items()}, flush=True)
    summary = {}
    for mode, runs in report['runs'].items():
        summary[mode] = {}
        for h in ('1s', '2s', '3s'):
            summary[mode][h] = {}
            for metric in ('mIoU', 'OccIoU'):
                values = {arm: [r[arm]['metrics'][h][metric] for r in runs.values()] for arm in ARMS}
                values.update({key: [r['future_image'][key][h][metric] for r in runs.values()]
                               for key in ('shuffled', 'zero_image')})
                summary[mode][h][metric] = dict(
                    arms={arm: dict(mean=float(np.mean(v)),
                                    std=float(np.std(v, ddof=1)) if len(v)>1 else 0.)
                          for arm, v in values.items()},
                    future_minus_query=(np.array(values['future_image'])-values['query_only']).tolist(),
                    future_minus_current=(np.array(values['future_image'])-values['current_image']).tolist())
    changed = [k for k, v in student.state_dict().items()
               if not torch.equal(v.cpu(), initial_student[k])]
    assert not changed, changed
    report.update(summary=summary, frozen_student_changed_tensors=changed,
                  trainable_parameters=sum(p.numel() for p in probe.parameters()),
                  elapsed_seconds=time.perf_counter()-started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    save_report(report, output_dir / 'report.json')
    print('[complete]', output_dir / 'report.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config', default='config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    p.add_argument('--output-dir', default='out/sparseworld_teacher_alignment/future_image_probe')
    p.add_argument('--train-samples', type=int, default=64)
    p.add_argument('--val-samples', type=int, default=64)
    p.add_argument('--updates', type=int, default=256)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--sample-seed', type=int, default=20260926)
    p.add_argument('--training-seeds', nargs='+', type=int, default=[20260926, 20260927])
    p.add_argument('--modes', nargs='+', choices=['semantic', 'joint'], default=['semantic', 'joint'])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--residual-weight', type=float, default=1e-3)
    main(p.parse_args())
