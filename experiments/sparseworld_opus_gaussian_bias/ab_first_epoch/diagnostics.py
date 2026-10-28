"""Add classification-loss and signed empty-bias gradients to existing probes."""
import numpy as np
import torch
from pathlib import Path

from .. import train as runner
from ..train_fullspace import evaluate_with_diagnostics
from .focal_loss import classification_terms


@torch.no_grad()
def classification_statistics(logits, target, camera, cfg, index):
    gamma = float(cfg.loss.occupancy_loss.get('focal_gamma', 0.))
    weights = logits.new_tensor(cfg.loss.occupancy_loss.manual_class_weight)
    labels = target.flatten(1)
    visible = camera.flatten(1).bool()
    ce, modulation, sample_weights, valid = classification_terms(logits, labels, weights, gamma)
    pt = (-ce).exp()
    empty_probability = logits.float().softmax(1)[:, 17]
    denominator = sample_weights.sum().clamp_min(torch.finfo(sample_weights.dtype).tiny)
    coefficient = cfg.loss.occupancy_loss.multi_loss_weights.loss_voxel_ce_weight * cfg.loss.current_weight
    # d focal / d logits = this factor times d CE / d logits.
    factor = (torch.ones_like(pt) if gamma == 0 else
              (1 - pt).pow(gamma) + gamma * pt * (1 - pt).pow(gamma - 1) * ce)
    empty_target = labels == 17
    bias_gradient = coefficient * sample_weights * factor * (empty_target.float() - empty_probability) / denominator
    rows = {}
    for region, region_mask in (('camera', visible), ('noncamera', ~visible)):
        for group, occupied_mask in (('free', empty_target), ('occupied', ~empty_target)):
            mask = valid & region_mask & occupied_mask
            count = int(mask.sum())
            rows[f'{region}/{group}'] = dict(
                voxels=count, label_weight_sum=float(sample_weights[mask].sum()),
                classification_loss_contribution=float(coefficient * (ce * modulation * sample_weights)[mask].sum() / denominator),
                ce_counterfactual_loss_contribution=float(coefficient * (ce * sample_weights)[mask].sum() / denominator),
                signed_bias_gradient=float(bias_gradient[mask].sum()),
                mean_correct_class_probability=float(pt[mask].mean()) if count else 0.,
                correct_class_probability_over_0_9_percent=100 * float((pt[mask] > .9).float().mean()) if count else 0.)
    return dict(index=index, objective='softmax_focal' if gamma else 'weighted_ce', gamma=gamma,
                empty_class_relative_weight=float(weights[17]), denominator=float(denominator),
                current_classification_loss=float(coefficient * (ce * modulation * sample_weights).sum() / denominator),
                signed_bias_gradient=float(bias_gradient.sum()), buckets=rows,
                scope='Current rendered-logit classification term only; excludes Lovasz, set and trajectory losses. Positive dL/db suppresses bias under gradient descent; negative increases it.')


def evaluate(model, dataset, cfg, directory, epoch, limit=None):
    if epoch != 1 and limit is None:
        return runner.evaluate_original(model, dataset, cfg, directory, epoch, limit)
    indices = set(np.linspace(0, len(dataset) - 1, cfg.epoch_diagnostics.attribute_samples, dtype=int).tolist())
    records, cursor = [], 0

    def observe(module, args, kwargs, result):
        nonlocal cursor
        index = cursor
        cursor += 1
        if index in indices:
            metas = kwargs['metas']
            records.append(classification_statistics(result['gaussian_logits_by_horizon'][0],
                metas['occ_label'], metas['occ_cam_mask'], cfg, index))

    handle = model.register_forward_hook(observe, with_kwargs=True)
    try:
        formal = evaluate_with_diagnostics(model, dataset, cfg, directory, epoch, limit)
    finally:
        handle.remove()
    output = (directory / 'epoch_diagnostics_smoke' if limit is not None else
              Path(cfg.report_directory) / 'epoch_diagnostics')
    runner.atomic_json(output / f'epoch_{epoch:02d}_classification_components.json', dict(
        epoch=epoch, num_samples=len(records), formal_sample_count=formal['num_samples'],
        is_smoke=limit is not None, records=records, configuration=cfg.loss.occupancy_loss.to_dict(),
        note='Fixed-sample analytical gradients at identical rendered outputs; never update model parameters.'))
    print(f'CLASSIFICATION_DIAGNOSTIC epoch={epoch} samples={len(records)}', flush=True)
    return formal
