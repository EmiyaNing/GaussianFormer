"""Add conditional semantics to the unchanged epoch-one full-space diagnosis."""
from pathlib import Path

import numpy as np
import torch

from ..ab_first_epoch.diagnostics import evaluate as baseline_evaluate
from ..train import atomic_json


@torch.no_grad()
def conditional_statistics(logits, target, camera, cfg, index):
    labels, visible = target.flatten(1).long(), camera.flatten(1).bool()
    valid = (labels >= 0) & (labels < 18)
    occupied = valid & (labels != 17)
    weights = logits.new_tensor(cfg.loss.occupancy_loss.manual_class_weight).float()
    coefficient = cfg.loss.current_weight * cfg.loss.occupancy_loss.conditional_semantic_weight
    safe = labels.masked_fill(~valid, 0)
    all_weights = weights[safe] * valid
    base_factor = cfg.loss.current_weight * cfg.loss.occupancy_loss.multi_loss_weights.loss_voxel_ce_weight
    probabilities = logits.float().softmax(1)
    correct = probabilities.gather(1, safe[:, None]).squeeze(1)
    factors = base_factor * all_weights / all_weights.sum().clamp_min(torch.finfo(torch.float32).tiny)
    base_norm_sq = (factors.square() * (probabilities.square().sum(1) - 2 * correct + 1).clamp_min(0)).sum()
    selected = logits.transpose(1, 2)[occupied][:, :17].float()
    targets = labels[occupied]
    log_prob = selected.log_softmax(1)
    conditional_prob = log_prob.exp()
    pt = conditional_prob.gather(1, targets[:, None]).squeeze(1)
    ce = -log_prob.gather(1, targets[:, None]).squeeze(1)
    selected_weights = weights[targets]
    denominator = selected_weights.sum().clamp_min(torch.finfo(torch.float32).tiny)
    auxiliary_factors = coefficient * selected_weights / denominator
    aux_norm_sq = (auxiliary_factors.square() * (conditional_prob.square().sum(1) - 2 * pt + 1).clamp_min(0)).sum()
    selected_base = probabilities.transpose(1, 2)[occupied][:, :17]
    dot = (factors[occupied] * auxiliary_factors *
           ((selected_base * conditional_prob).sum(1) - correct[occupied] - pt + 1)).sum()
    norm_product = (base_norm_sq * aux_norm_sq).sqrt()
    buckets = {}
    for name, region in (('camera', visible), ('noncamera', ~visible)):
        mask = region[occupied]
        count = int(mask.sum())
        buckets[name] = dict(gt_occupied_voxels=count,
                             loss_contribution=float((coefficient * ce * selected_weights)[mask].sum() / denominator),
                             mean_correct_conditional_probability=float(pt[mask].mean()) if count else 0.)
    return dict(index=index, auxiliary_weight=float(coefficient), gt_occupied_voxels=int(occupied.sum()),
                label_weight_sum=float(selected_weights.sum()),
                conditional_semantic_loss=float(coefficient * (ce * selected_weights).sum() / denominator),
                direct_empty_logit_gradient=0., auxiliary_logit_gradient_norm=float(aux_norm_sq.sqrt()),
                base_18class_ce_logit_gradient_norm=float(base_norm_sq.sqrt()),
                auxiliary_to_base_ce_logit_gradient_norm_ratio=float((aux_norm_sq / base_norm_sq.clamp_min(1e-30)).sqrt()),
                auxiliary_base_ce_logit_gradient_cosine=float(dot / norm_product.clamp_min(1e-30)),
                buckets=buckets,
                scope='Current rendered logits; analytical classification gradients; excludes Lovasz/set/trajectory and shared parameter gradients.')


def conditional_confusion(logits, target, camera):
    labels = target.flatten(1).long()
    prediction = logits[:, :17].argmax(1)
    valid = (labels >= 0) & (labels < 17)
    visible = camera.flatten(1).bool()
    full = torch.bincount(labels[valid] * 17 + prediction[valid], minlength=289).reshape(17, 17).cpu().numpy()
    mask = valid & visible
    camera_hist = torch.bincount(labels[mask] * 17 + prediction[mask], minlength=289).reshape(17, 17).cpu().numpy()
    return dict(full=full, camera=camera_hist, noncamera=full - camera_hist)


def evaluate(model, dataset, cfg, directory, epoch, limit=None):
    if epoch != 1 and limit is None:
        return baseline_evaluate(model, dataset, cfg, directory, epoch, limit)
    indices = set(np.linspace(0, len(dataset) - 1, cfg.epoch_diagnostics.attribute_samples, dtype=int).tolist())
    records, cursor = [], 0
    matrices = {name: np.zeros((17, 17), dtype=np.int64) for name in ('full', 'camera', 'noncamera')}

    @torch.no_grad()
    def observe(module, args, kwargs, result):
        nonlocal cursor
        index = cursor
        cursor += 1
        metas = kwargs['metas']
        logits = result['gaussian_logits_by_horizon'][0]
        for name, hist in conditional_confusion(logits, metas['occ_label'], metas['occ_cam_mask']).items():
            matrices[name] += hist
        if index in indices:
            records.append(conditional_statistics(logits, metas['occ_label'], metas['occ_cam_mask'], cfg, index))

    handle = model.register_forward_hook(observe, with_kwargs=True)
    try:
        formal = baseline_evaluate(model, dataset, cfg, directory, epoch, limit)
    finally:
        handle.remove()
    output = directory / 'epoch_diagnostics_smoke' if limit is not None else Path(cfg.report_directory) / 'epoch_diagnostics'
    if (not np.array_equal(matrices['full'], matrices['camera'] + matrices['noncamera']) or
            not np.array_equal(matrices['full'].sum(1), formal['horizons'][0]['class_gt_count'][:17])):
        raise ValueError('conditional confusion does not include exactly all GT-occupied voxels')
    np.savez_compressed(output / f'epoch_{epoch:02d}_conditional_semantic_confusion.npz', **matrices)
    atomic_json(output / f'epoch_{epoch:02d}_conditional_semantics.json', dict(
        epoch=epoch, num_samples=len(records), formal_sample_count=formal['num_samples'], is_smoke=limit is not None,
        records=records, formal_current_GT_occupied_voxels=int(matrices['full'].sum()),
        formal_current_conditional_class_accuracy_percent=100 * float(np.trace(matrices['full'])) / max(int(matrices['full'].sum()), 1),
        note='Conditional prediction uses foreground argmax for every GT-occupied voxel, including those predicted free by the 18-class head. B did not save this full-set metric.'))
    # Clarify the legacy diagnostic: it reports B's 18-class CE, auxiliary separately.
    import json
    path = output / f'epoch_{epoch:02d}_classification_components.json'
    classification = json.loads(path.read_text())
    classification['terms_included'] = ['weighted_18class_ce']
    classification['auxiliary_report'] = f'epoch_{epoch:02d}_conditional_semantics.json'
    atomic_json(path, classification)
    print(f'CONDITIONAL_DIAGNOSTIC epoch={epoch} formal_samples={formal["num_samples"]} probe_samples={len(records)}', flush=True)
    return formal
