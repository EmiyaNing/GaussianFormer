"""Frozen D/E evaluator extended to diagnose every training epoch."""
import math
from pathlib import Path
import time

import numpy as np
import torch

from loss import OPENOCC_LOSS
from .. import train as runner
from ..ab_first_epoch.diagnostics import classification_statistics
from ..conditional_first_epoch.diagnostics import conditional_confusion
from ..fullspace_diagnostics import add_evaluation_batch, attribute_record, gradient_probe, partition_histograms, region_metrics
from ..metrics import CLASS_NAMES, HorizonMetrics
from ..train_fullspace import save_diagnostic
from ..semantic_occupancy_first_epoch.model import joint_log_probabilities, predict_factorized
from ..semantic_occupancy_first_epoch.protocol import E


@torch.no_grad()
def conditional_statistics(logits, target, camera, cfg, index):
    labels, visible = target.flatten(1).long(), camera.flatten(1).bool()
    valid = (labels >= 0) & (labels < 18)
    occupied = valid & (labels != 17)
    weights = logits.new_tensor(cfg.loss.occupancy_loss.manual_class_weight).float()
    auxiliary_weights = weights[:17] * logits.new_tensor(cfg.loss.occupancy_loss.get('conditional_class_multipliers', [1.] * 17))
    coefficient = cfg.loss.current_weight * cfg.loss.occupancy_loss.conditional_semantic_weight
    safe = labels.masked_fill(~valid, 0)
    sample_weights = weights[safe] * valid
    base_factor = cfg.loss.current_weight * cfg.loss.occupancy_loss.multi_loss_weights.loss_voxel_ce_weight
    probabilities = logits.float().softmax(1)
    correct = probabilities.gather(1, safe[:, None]).squeeze(1)
    factors = base_factor * sample_weights / sample_weights.sum().clamp_min(1e-30)
    base_norm_sq = (factors.square() * (probabilities.square().sum(1) - 2 * correct + 1).clamp_min(0)).sum()
    selected = logits.transpose(1, 2)[occupied][:, :17].float()
    targets = labels[occupied]
    lp = selected.log_softmax(1)
    p = lp.exp()
    pt = p.gather(1, targets[:, None]).squeeze(1)
    ce = -lp.gather(1, targets[:, None]).squeeze(1)
    selected_weights = auxiliary_weights[targets]
    denominator = selected_weights.sum().clamp_min(1e-30)
    auxiliary_factors = coefficient * selected_weights / denominator
    aux_norm_sq = (auxiliary_factors.square() * (p.square().sum(1) - 2 * pt + 1).clamp_min(0)).sum()
    selected_base = probabilities.transpose(1, 2)[occupied][:, :17]
    dot = (factors[occupied] * auxiliary_factors * ((selected_base * p).sum(1) - correct[occupied] - pt + 1)).sum()
    buckets = {}
    for name, region in (('camera', visible), ('noncamera', ~visible)):
        mask = region[occupied]
        buckets[name] = dict(gt_occupied_voxels=int(mask.sum()),
            loss_contribution=float((coefficient * ce * selected_weights)[mask].sum() / denominator),
            mean_correct_conditional_probability=float(pt[mask].mean()) if mask.any() else 0.)
    classes = []
    for c, name in enumerate(CLASS_NAMES[:17]):
        mask = targets == c
        classes.append(dict(class_name=name, gt_voxels=int(mask.sum()),
            auxiliary_weight=float(auxiliary_weights[c]),
            auxiliary_loss_contribution=float((ce * auxiliary_factors)[mask].sum()),
            correct_conditional_probability=float(pt[mask].mean()) if mask.any() else 0.))
    return dict(index=index, auxiliary_weight=float(coefficient), gt_occupied_voxels=int(occupied.sum()),
        label_weight_sum=float(selected_weights.sum()), conditional_semantic_loss=float((ce * auxiliary_factors).sum()),
        direct_empty_logit_gradient=0., auxiliary_logit_gradient_norm=float(aux_norm_sq.sqrt()),
        base_18class_ce_logit_gradient_norm=float(base_norm_sq.sqrt()),
        auxiliary_to_base_ce_logit_gradient_norm_ratio=float((aux_norm_sq / base_norm_sq.clamp_min(1e-30)).sqrt()),
        auxiliary_base_ce_logit_gradient_cosine=float(dot / (base_norm_sq * aux_norm_sq).sqrt().clamp_min(1e-30)),
        buckets=buckets, classes=classes,
        scope='Analytical gradients at the joint 18-class output, excluding shared parameter gradients; E occupancy gate is separately parameterized.')


def scene_name(dataset, index):
    scene_index, _ = dataset.keyframes[index]
    return str(scene_index)


def save_scene_counts(path, scenes, sample_counts):
    names = sorted(scenes)
    np.savez_compressed(path, scenes=np.asarray(names), confusion=np.stack([scenes[name] for name in names]),
                        sample_counts=np.asarray([sample_counts[name] for name in names], dtype=np.int64))


def decode(logits, cfg):
    return predict_factorized(logits, cfg.model.gaussian_head.occupancy_threshold) if cfg.route_arm == E else logits.argmax(1)


@torch.no_grad()
def evaluate(model, dataset, cfg, directory, epoch, limit=None):
    if cfg.loss.mask_type != 'none' or cfg.val_loader.batch_size != 1:
        raise ValueError('diagnostics require full-space supervision and batch-one validation')
    indices = np.linspace(0, len(dataset) - 1, cfg.epoch_diagnostics.attribute_samples, dtype=int).tolist()
    gradient_indices = set(indices[:cfg.epoch_diagnostics.gradient_samples])
    metric = HorizonMetrics()
    visibility = {region: np.zeros((7, 18, 18), dtype=np.int64) for region in ('full', 'camera', 'noncamera')}
    conditional = {region: np.zeros((17, 17), dtype=np.int64) for region in visibility}
    scenes, sample_counts, alternative = {}, {}, np.zeros((7, 18, 18), dtype=np.int64)
    attributes, gradients, probes, actual_indices, classification, semantics, gate_records = [], [], {}, [], [], [], []
    captured, capture_active = [], False
    original_render = model.gaussian_head.render_state
    rng = runner.capture_rng()

    def record_render(gaussian, xyz):
        logits = original_render(gaussian, xyz)
        if capture_active:
            captured.append((gaussian, logits))
        return logits

    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    model.gaussian_head.render_state = record_render
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    started = time.monotonic()
    try:
        for index, data in enumerate(runner.loader(dataset, cfg.val_loader)):
            capture_active = index in indices
            captured.clear()
            images, metas = runner.move_batch(data, 'cuda')
            result = model(imgs=images, metas=metas)
            add_evaluation_batch(metric, visibility, result, metas)
            name = scene_name(dataset, index)
            scenes.setdefault(name, np.zeros((7, 18, 18), dtype=np.int64))
            sample_counts[name] = sample_counts.get(name, 0) + 1
            grids = [result['final_occ_grid']] + result['future_occ_grids']
            for h, grid in enumerate(grids):
                target = metas['occ_label'] if h == 0 else metas['future_occ_labels'][:, h - 1]
                cam = metas['occ_cam_mask'] if h == 0 else metas['future_occ_cam_masks'][:, h - 1]
                scenes[name][h] += partition_histograms(grid, target, cam)['full']
                if cfg.route_arm == E:
                    alternative[h] += partition_histograms(result['joint_argmax_occ_grids'][h], target, cam)['full']
            current_logits = result['gaussian_logits_by_horizon'][0]
            for region, hist in conditional_confusion(current_logits, metas['occ_label'], metas['occ_cam_mask']).items():
                conditional[region] += hist
            if capture_active:
                if len(captured) != 7:
                    raise ValueError('all seven horizons must be exposed to diagnostics')
                actual_indices.append(index)
                classification.append(classification_statistics(current_logits, metas['occ_label'], metas['occ_cam_mask'], cfg, index))
                semantics.append(conditional_statistics(current_logits, metas['occ_label'], metas['occ_cam_mask'], cfg, index))
                current = model.pts_bbox_head.ind_stamps_all == 0
                points_list = [result['all_refine_pts'][-1][:, current]] + result['forecast_points_list']
                classes_list = [result['all_cls_scores'][-1][:, current]] + result['forecast_semantics_list']
                features_list = [result['query_feat'][:, current]] + result['forecast_features_list']
                xyz = metas['occ_xyz'].float().flatten(1, 3)
                trained_horizon = max(1, min(model.curr_epoch - model.finetune_epoch + 1, 6))
                for h, (gaussian, logits) in enumerate(tuple(captured)):
                    points, classes, features = points_list[h], classes_list[h], features_list[h]
                    attributes.append(attribute_record(index, h, points, classes, features, gaussian, model.gaussian_head))
                    target = metas['occ_label'] if h == 0 else metas['future_occ_labels'][:, h - 1]
                    cam = metas['occ_cam_mask'] if h == 0 else metas['future_occ_cam_masks'][:, h - 1]
                    modes = dict(default=decode(logits, cfg).view_as(target),
                                 point_rasterizer=model.pts_bbox_head.get_occ(dict(cls_scores=classes, refine_pts=points)))
                    if cfg.route_arm == E:
                        gate = torch.logsumexp(logits[:, :17], 1) - logits[:, 17]
                        modes['joint_18class_argmax'] = logits.argmax(1).view_as(target)
                        if h == 0:
                            foreground = target.flatten(1) != 17
                            gate_records.append(dict(index=index, foreground_gate_mean=float(gate[foreground].mean()),
                                free_gate_mean=float(gate[~foreground].mean()),
                                evidence_mean=float(gaussian.occupancy_evidence.mean()),
                                threshold=cfg.model.gaussian_head.occupancy_threshold))
                    for delta in (-2., -1., 1., 2.):
                        if cfg.route_arm == E:
                            changed = joint_log_probabilities(logits[:, :17], gate + delta)
                        else:
                            changed = logits.clone()
                            changed[:, 17] -= delta
                        modes[f'empty_bias_delta_{delta:+g}'] = decode(changed, cfg).view_as(target)
                    modes['fixed_scale_0_2'] = decode(original_render(gaussian._replace(scales=torch.full_like(gaussian.scales, .2)), xyz), cfg).view_as(target)
                    modes['opacity_times_0_5'] = decode(original_render(gaussian._replace(opacities=gaussian.opacities * .5), xyz), cfg).view_as(target)
                    for mode, prediction in modes.items():
                        for region, hist in partition_histograms(prediction, target, cam).items():
                            key = f'h{h}/{mode}/{region}'
                            probes.setdefault(key, np.zeros((18, 18), dtype=np.int64))
                            probes[key] += hist
                    supervised = h == 0 or (not model.pretrain and h <= trained_horizon and h in model.gaussian_loss_horizons)
                    if index in gradient_indices and supervised:
                        model.gaussian_head.render_state = original_render
                        gradients.append(dict(index=index, **gradient_probe(model, cfg, result, gaussian, xyz, metas, h, loss_fn)))
                        model.gaussian_head.render_state = record_render
            if (index + 1) % 128 == 0:
                print(f'EVAL+ROUTE_DIAG arm={cfg.route_arm} epoch={epoch} samples={metric.samples}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s', flush=True)
            if limit is not None and index + 1 >= limit:
                break
    finally:
        model.gaussian_head.render_state = original_render
        captured.clear()
        runner.restore_rng(rng)
        model.memory.clear()
        model.memory_queue.clear()
    if limit is None and metric.samples != 1024:
        raise ValueError('formal evaluation requires all 1024 samples')
    if not np.array_equal(sum(scenes.values()), metric.confusion):
        raise ValueError('scene counts differ from formal confusion')
    if not np.array_equal(visibility['full'], visibility['camera'] + visibility['noncamera']):
        raise ValueError('visibility partition failed')
    if not np.array_equal(conditional['full'].sum(1), metric.confusion[0].sum(1)[:17]):
        raise ValueError('conditional diagnosis omitted GT occupied voxels')
    formal = dict(metric.results(), epoch=epoch, evaluation_seconds=time.monotonic() - started)
    runner.atomic_json(directory / f'epoch_{epoch:02d}_metrics.json', formal)
    np.savez_compressed(directory / f'epoch_{epoch:02d}_confusion.npz', confusion=metric.confusion)
    save_scene_counts(directory / f'epoch_{epoch:02d}_scene_confusion.npz', scenes, sample_counts)
    output = directory / 'epoch_diagnostics_smoke' if limit is not None else Path(cfg.report_directory) / 'epoch_diagnostics'
    payload = dict(epoch=epoch, num_samples=metric.samples, is_smoke=limit is not None, voxel_supervision_mask='none',
        config_sha256=runner.digest(cfg.filename), class_names=CLASS_NAMES, attribute_sample_indices=actual_indices,
        supervised_gaussian_horizons=[0] + ([] if model.pretrain else [h for h in model.gaussian_loss_horizons if 0 < h <= trained_horizon]),
        visibility={f'h{h}/{region}': region_metrics(visibility[region][h]) for h in range(7) for region in visibility},
        attributes=attributes, gradient_probes=gradients, render_probes={key: region_metrics(hist) for key, hist in probes.items()},
        selection_mean_miou=formal['selection_mean_miou'], route_arm=cfg.route_arm,
        decision_rule='independent occupancy sigmoid > 0.5, then foreground argmax' if cfg.route_arm == E else '18-class argmax',
        _visibility_arrays=visibility)
    save_diagnostic(output, epoch, payload, formal)
    runner.atomic_json(output / f'epoch_{epoch:02d}_classification_components.json', dict(epoch=epoch,
        num_samples=len(classification), formal_sample_count=metric.samples, is_smoke=limit is not None,
        records=classification, configuration=cfg.loss.occupancy_loss.to_dict(), terms_included=['weighted_18class_joint_ce']))
    runner.atomic_json(output / f'epoch_{epoch:02d}_conditional_semantics.json', dict(epoch=epoch,
        num_samples=len(semantics), formal_sample_count=metric.samples, is_smoke=limit is not None, records=semantics,
        formal_current_conditional_class_accuracy_percent=100 * float(np.trace(conditional['full'])) / max(int(conditional['full'].sum()), 1),
        formal_current_conditional_macro_accuracy_percent=100 * float(np.mean(np.divide(conditional['full'].diagonal(), conditional['full'].sum(1), out=np.zeros(17), where=conditional['full'].sum(1)>0)))))
    np.savez_compressed(output / f'epoch_{epoch:02d}_conditional_semantic_confusion.npz', **conditional)
    if cfg.route_arm == E:
        np.savez_compressed(output / f'epoch_{epoch:02d}_joint_argmax_confusion.npz', confusion=alternative)
        runner.atomic_json(output / f'epoch_{epoch:02d}_occupancy_gate.json', dict(records=gate_records,
            threshold=.5, threshold_selected_before_training=True, same_logits_alternative=[region_metrics(hist) for hist in alternative],
            gradient_scope='The occupancy branch is independent of semantic logits at fixed shared features and geometry.'))
    print(f'ROUTE_DIAGNOSTIC_COMPLETE arm={cfg.route_arm} epoch={epoch} samples={metric.samples}', flush=True)
    return formal

