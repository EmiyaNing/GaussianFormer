#!/usr/bin/env python3
"""Trainable one-frame OPUS-style reader probe for privileged future RGB.

The SparseWorld predictor and occupancy decoder are frozen. Raw FPN maps are
cached once per sample; only a one-frame OPUS sampler/mixer and a zero-output
fusion adapter are trained. This tests whether adapting the feature-reading
interface unlocks information from real current/future images.
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
    AdaptiveOPUSImageReader, FutureImageFusionProbe)
from model.multi_horizon_teacher import file_sha256

HORIZONS = (2, 4, 6)
ARMS = ('query_only', 'current_image', 'future_image', 'future_shuffled')


def cpu_maps(features):
    return [feature.detach().to(dtype=torch.float16, device='cpu').contiguous()
            for feature in features]


def shuffle_visible(extra, visible):
    """Move read features to other visible query slots, preserving support."""
    if extra.shape[0] != 1:
        raise ValueError('The diagnostic uses batch size one')
    indices = torch.nonzero(visible[0], as_tuple=False).flatten()
    if indices.numel() < 2:
        return extra
    permutation = torch.arange(extra.shape[1], device=extra.device)
    permutation[indices] = indices.roll(1)
    return extra.index_select(1, permutation)


def state_extra(reader, cache, state, arm, spatial_shift_m=(0., 0.)):
    if arm == 'query_only':
        query = state['query'].cuda(non_blocking=True)
        return torch.zeros_like(query), state['future_visible']
    if arm == 'current_image':
        maps, metas = cache['current_maps'], state['current_metas']
    else:
        maps, metas = state['future_maps'], state['future_metas']
    sample_points = state['normalized_points'].cuda(non_blocking=True).clone()
    pc_range = reader.sampling.pc_range
    sample_points[..., 0] += spatial_shift_m[0] / (pc_range[3] - pc_range[0])
    sample_points[..., 1] += spatial_shift_m[1] / (pc_range[4] - pc_range[1])
    query = state['query'].to(device=sample_points.device, non_blocking=True)
    extra, visible = reader(sample_points, query,
                            [x.cuda(non_blocking=True).float() for x in maps],
                            tree_to(metas, 'cuda'))
    if arm == 'future_shuffled':
        extra = shuffle_visible(extra, visible)
    return extra, visible


@torch.no_grad()
def cache_split(student, loader, cfg, label):
    caches = []
    baseline = metric_bank(cfg, ['baseline'])
    current_metric = _metric(cfg)
    visibility = {h: [] for h in HORIZONS}
    for index, data in enumerate(loader):
        data = tree_to(data, 'cuda')
        imgs = data.pop('img')
        student_metas = dict(data, projection_mat=data['projection_mat'][:, :30],
                             image_wh=data['image_wh'][:, :30])
        output = student(imgs=imgs[:, :30], metas=student_metas,
                         forecast_snapshot_step=HORIZONS)
        current_maps = cpu_maps([f[:, :6] for f in output['ms_img_feats']])
        states = {}
        for h in HORIZONS:
            snap = output['forecast_snapshots'][h]
            query, normalized_points = snap['features'], snap['positions']
            points = student._world(normalized_points)
            logits = snap['semantic']
            assert query.shape[1] == 720 + sum(student.future_queries[:h])
            packed_index = cfg.teacher_history_frames + h
            begin = packed_index * 6
            future_maps = student.extract_img_feat(
                imgs[:, begin:begin + 6])['ms_img_feats']
            future_metas = dict(
                projection_mat=data['projection_mat'][:, begin:begin + 6],
                image_wh=data['image_wh'][:, begin:begin + 6])
            current_metas = dict(
                projection_mat=data['projection_mat'][:, :6] @
                    data['future_ego_to_current'][:, h - 1, None],
                image_wh=data['image_wh'][:, :6])
            states[h] = dict(
                query=query.detach().cpu(),
                normalized_points=normalized_points.detach().cpu(),
                points=points.detach().cpu(),
                logits=logits.detach().cpu(),
                future_maps=cpu_maps(future_maps),
                future_metas={k: v.detach().cpu() for k, v in future_metas.items()},
                current_metas={k: v.detach().cpu() for k, v in current_metas.items()},
                future_visible=torch.zeros(query.shape[:2], dtype=torch.bool))
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
        metas = tree_to({k: data[k] for k in keep}, 'cpu')
        metas['future_occ_labels'] = metas['future_occ_labels'].to(torch.uint8)
        cache = dict(states=states, current_maps=current_maps, metas=metas)
        caches.append(cache)
        del output, imgs, data, student_metas
        if (index + 1) % 8 == 0 or index + 1 == len(loader):
            print(f'[cache {label}] {index + 1}/{len(loader)}', flush=True)

    # Measure visibility with the exact one-frame reader after its outputs have
    # been cached. This is diagnostic only; no gradients are retained.
    reader = AdaptiveOPUSImageReader(student.encoder.layers[-1]).cuda().eval()
    for cache in caches:
        for h in HORIZONS:
            state = cache['states'][h]
            _, fv = reader(state['normalized_points'].cuda(), state['query'].cuda(),
                           [x.cuda().float() for x in state['future_maps']],
                           tree_to(state['future_metas'], 'cuda'))
            _, cv = reader(state['normalized_points'].cuda(), state['query'].cuda(),
                           [x.cuda().float() for x in cache['current_maps']],
                           tree_to(state['current_metas'], 'cuda'))
            state['future_visible'] = fv.cpu()
            visibility[h].append(dict(future=float(fv.float().mean()),
                                      current=float(cv.float().mean())))
    return caches, dict(
        future=bank_results(baseline)['baseline'], current=current_metric.results(),
        visibility={str(h): {k: float(np.mean([v[k] for v in values]))
                             for k in values[0]} for h, values in visibility.items()})


def predict(student, reader, fusion, cache, state, arm, spatial_shift_m=(0., 0.)):
    extra, visible = state_extra(reader, cache, state, arm, spatial_shift_m)
    features, delta = fusion(state['query'].cuda(), extra)
    logits = student.semantic(features).view(
        features.shape[0], features.shape[1], student.num_refines, 17)
    return state['points'].cuda().flatten(1, 2), logits.flatten(1, 2), delta, visible


@torch.no_grad()
def evaluate(student, reader, fusion, caches, cfg, arm, intervention=None,
             spatial_shift_m=(0., 0.)):
    bank = metric_bank(cfg, [arm])
    visible_values = []
    for cache in caches:
        data = tree_to(cache['metas'], 'cuda')
        for h in HORIZONS:
            state = cache['states'][h]
            use_arm = arm
            if intervention == 'zero':
                extra = torch.zeros_like(state['query'].cuda())
                visible = state['future_visible'].cuda()
                features, _ = fusion(state['query'].cuda(), extra)
                logits = student.semantic(features).view(
                    1, features.shape[1], student.num_refines, 17).flatten(1, 2)
                points = state['points'].cuda().flatten(1, 2)
            elif intervention == 'shuffled':
                use_arm = 'future_shuffled'
                points, logits, _, _ = predict(student, reader, fusion, cache, state, use_arm)
            else:
                points, logits, _, visible = predict(
                    student, reader, fusion, cache, state, use_arm, spatial_shift_m)
                if arm == 'future_image' and intervention is None:
                    visible_values.append(float(visible.float().mean()))
            add_metric(bank[arm][h], student, points, logits, data, h)
    result = bank_results(bank)[arm]
    if visible_values:
        result['query_coverage'] = float(np.mean(visible_values))
    return result


def train_arm(student, reader, fusion, caches, objective, arm, args, seed):
    device = next(student.parameters()).device
    reader.to(device)
    fusion.to(device)
    decoder_parameters = list(student.semantic.parameters()) if args.adapt_decoder else []
    misplaced = [name for name, value in list(reader.named_parameters()) +
                 list(fusion.named_parameters()) +
                 [(f'semantic.{name}', value) for name, value in
                  student.semantic.named_parameters() if args.adapt_decoder]
                 if value.device != device]
    if misplaced:
        raise RuntimeError(f'{arm} adapter parameters are not all on {device}: {misplaced[:8]}')
    parameters = list(reader.parameters()) + list(fusion.parameters()) + decoder_parameters
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr, weight_decay=.01)
    rng = np.random.RandomState(seed)
    schedule = []
    while len(schedule) < args.updates:
        schedule.extend(rng.permutation(len(caches)).tolist())
    history = []
    reader.train(); fusion.train()
    student.semantic.train(args.adapt_decoder)
    for step, index in enumerate(schedule[:args.updates]):
        cache = caches[index]
        data = tree_to(cache['metas'], 'cuda')
        value = 0.
        for h in HORIZONS:
            state = cache['states'][h]
            points, logits, delta, _ = predict(student, reader, fusion, cache, state, arm)
            cls, pts = objective._set_loss(
                points[0], logits[0], data['future_occ_labels'][0, h - 1].long())
            value = value + (objective.lambda_future_cls * cls +
                             objective.lambda_future_pts * pts +
                             args.residual_weight * delta.square().mean()) / len(HORIZONS)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, 5.,
            error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step + 1, loss=float(value.detach()),
                            grad_norm=float(norm)))
        if (step + 1) % 128 == 0 or step + 1 == args.updates:
            print('[train]', seed, arm, history[-1], flush=True)
    reader.eval(); fusion.eval(); student.semantic.eval()
    return history


def save_report(report, path):
    path.write_text(json.dumps(_json_safe(report), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding='utf-8')


def main(args):
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    if report_path.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    seed_all(args.sample_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(args.data_config)
    model_cfg = Config.fromfile(
        'config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py')
    student = build_segmentor(model_cfg.model)
    checkpoint = cfg.student_init_checkpoint
    digest = file_sha256(checkpoint)
    assert digest == cfg.student_init_checkpoint_sha256
    student.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
    student.set_epoch(100)
    student.requires_grad_(False).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    layer = student.encoder.layers[-1]
    base_reader = AdaptiveOPUSImageReader(layer).cuda()
    # Fail fast before building large caches if temporal weight folding is not
    # algebraically equivalent to repeated copies of the same sampled frame.
    with torch.no_grad():
        b, q, g, p, d = 1, 3, base_reader.mixing.n_groups, base_reader.sampling.num_points, 256 // 4
        sample = torch.randn(b, q, g, p, d, device='cuda')
        old_sample = sample.repeat(1, 1, 1, layer.sampling.num_frames, 1)
        query = torch.randn(b, q, 256, device='cuda')
        old_value = layer.mixing(old_sample, query)
        new_value = base_reader.mixing(sample, query)
        torch.testing.assert_close(old_value, new_value, rtol=2e-5, atol=2e-5)
        fusion_check = FutureImageFusionProbe().cuda().eval()
        fusion_query, fusion_delta = fusion_check(query, torch.randn_like(query))
        torch.testing.assert_close(fusion_query, query, rtol=0, atol=0)
        torch.testing.assert_close(fusion_delta, torch.zeros_like(fusion_delta), rtol=0, atol=0)
    del base_reader, fusion_check
    train_loader, train_manifest = loader_for(
        cfg, 'train', args.train_samples, args.workers, args.sample_seed)
    val_loader, val_manifest = loader_for(
        cfg, 'val', args.val_samples, args.workers, args.sample_seed)
    assert not set(train_manifest['scenes']) & set(val_manifest['scenes'])
    train, train_baseline = cache_split(student, train_loader, cfg, 'train')
    validation, baseline = cache_split(student, val_loader, cfg, 'val')
    report = dict(
        args=vars(args).copy(), checkpoint=checkpoint, checkpoint_sha256=digest,
        training=train_manifest, validation=val_manifest,
        train_tokens=[c['metas']['sample_idx'] for c in train],
        val_tokens=[c['metas']['sample_idx'] for c in validation],
        baseline=baseline, training_baseline=train_baseline,
        protocol=dict(precision='FP32; TF32 disabled',
                      trainable='single-frame OPUS sampling, mixing, and zero-output fusion; '
                                + ('plus shared semantic decoder' if args.adapt_decoder else
                                   'SparseWorld and occupancy decoder frozen'),
                      reader='five-frame point weights folded by summing equivalent repeated-frame coefficients',
                      horizons_seconds=[1, 2, 3], query_counts=[840, 960, 1040],
                      future_rgb='actual future six-camera RGB; privileged positive control, not deployable',
                      current_geometry='future query positions projected into current ego using GT transform',
                      spatial_shuffle='training-time query-slot permutation among visible slots'),
        single_frame_fold_equivalence=True,
        zero_output_fusion_check=True, runs={})
    save_report(report, report_path)
    objective = OPENOCC_LOSS.build(copy.deepcopy(cfg.base_sparseworld_loss)).cuda()
    semantic_initial = {k: v.detach().cpu().clone()
                        for k, v in student.semantic.state_dict().items()}
    student.semantic.requires_grad_(args.adapt_decoder)
    for seed in args.training_seeds:
        report['runs'][str(seed)] = {}
        for arm in ARMS:
            seed_all(seed)
            student.semantic.load_state_dict(semantic_initial, strict=True)
            student.semantic.requires_grad_(args.adapt_decoder)
            reader = AdaptiveOPUSImageReader(layer).cuda()
            fusion = FutureImageFusionProbe().cuda()
            initial_reader = copy.deepcopy(reader.state_dict())
            initial_fusion = copy.deepcopy(fusion.state_dict())
            history = train_arm(student, reader, fusion, train, objective,
                                arm, args, seed)
            checkpoint_name = f'{seed}_{arm}.pth'
            torch.save(dict(
                reader={k: v.detach().cpu() for k, v in reader.state_dict().items()},
                fusion={k: v.detach().cpu() for k, v in fusion.state_dict().items()},
                semantic={k: v.detach().cpu()
                          for k, v in student.semantic.state_dict().items()},
                seed=seed, arm=arm, updates=args.updates,
                baseline_sha256=digest), output / checkpoint_name)
            result = dict(training=history,
                          checkpoint=checkpoint_name,
                          metrics=evaluate(student, reader, fusion, validation, cfg, arm))
            if arm == 'future_image':
                result['shuffled'] = evaluate(
                    student, reader, fusion, validation, cfg, arm, 'shuffled')
                result['zero_image'] = evaluate(
                    student, reader, fusion, validation, cfg, arm, 'zero')
            result['initial_state_identical_to_control'] = True
            result['initial_reader_tensors'] = len(initial_reader)
            result['initial_fusion_tensors'] = len(initial_fusion)
            report['runs'][str(seed)][arm] = result
            save_report(report, report_path)
            if arm == 'future_image':
                result['spatial_shifts'] = {}
                for magnitude in (0.4, 1.0, 2.0):
                    for axis, sign in ((0, 1.), (0, -1.), (1, 1.), (1, -1.)):
                        shift = [0., 0.]
                        shift[axis] = sign * magnitude
                        direction = ('x' if axis == 0 else 'y') + ('+' if sign > 0 else '-')
                        key = f'{direction}_{magnitude:g}m'
                        result['spatial_shifts'][key] = evaluate(
                            student, reader, fusion, validation, cfg, arm,
                            spatial_shift_m=tuple(shift))
            save_report(report, report_path)
            print('[eval]', seed, arm,
                  {h: round(result['metrics'][h]['mIoU'], 4)
                   for h in ('1s', '2s', '3s')}, flush=True)
            del reader, fusion
    summary = {}
    for h in ('1s', '2s', '3s'):
        summary[h] = {}
        for metric in ('mIoU', 'OccIoU'):
            values = {arm: [r[arm]['metrics'][h][metric]
                           for r in report['runs'].values()] for arm in ARMS}
            values['future_shuffled_eval'] = [
                r['future_image']['shuffled'][h][metric]
                for r in report['runs'].values()]
            values['future_zero_eval'] = [
                r['future_image']['zero_image'][h][metric]
                for r in report['runs'].values()]
            summary[h][metric] = dict(
                arms={name: dict(mean=float(np.mean(vals)),
                                 std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.)
                      for name, vals in values.items()},
                future_minus_query=(np.asarray(values['future_image']) -
                                    np.asarray(values['query_only'])).tolist(),
                future_minus_current=(np.asarray(values['future_image']) -
                                      np.asarray(values['current_image'])).tolist(),
                future_minus_train_shuffled=(np.asarray(values['future_image']) -
                                             np.asarray(values['future_shuffled'])).tolist())
    report.update(summary=summary, elapsed_seconds=time.perf_counter() - started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    save_report(report, report_path)
    print('[complete]', report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    parser.add_argument('--output-dir', default='out/future_alignment_hypotheses/H4_adaptive')
    parser.add_argument('--train-samples', type=int, default=128)
    parser.add_argument('--val-samples', type=int, default=64)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--training-seeds', nargs='+', type=int, default=[20260926, 20260927, 20260928])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--residual-weight', type=float, default=1e-3)
    parser.add_argument('--adapt-decoder', action='store_true',
                        help='R2: also adapt the same semantic decoder in every arm')
    main(parser.parse_args())
