"""Semantic Gaussian auxiliary loss for current/future SparseWorld points."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class SparseWorldGaussianPointOccupancyLoss(nn.Module):
    """Nearest occupied-voxel focal supervision for Gaussian residual logits."""

    def __init__(self, weight=1., empty_label=17, focal_gamma=2.,
                 focal_alpha=.25, class_weights=None, chunk_size=1024,
                 warmup_iters=0, horizon_weights=(1., 1., 1., 1., 1., 1., 1.),
                 enable_during_pretrain=False):
        super().__init__()
        if len(horizon_weights) != 7:
            raise ValueError('horizon_weights must contain current plus six futures')
        self.weight = float(weight)
        self.empty_label = int(empty_label)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.chunk_size = int(chunk_size)
        self.warmup_iters = int(warmup_iters)
        self.horizon_weights = tuple(float(value) for value in horizon_weights)
        self.enable_during_pretrain = bool(enable_during_pretrain)
        self.register_buffer('class_weights', torch.as_tensor(
            class_weights if class_weights is not None else [],
            dtype=torch.float32), persistent=False)

    def _loss_one(self, logits, points, valid_masks, labels, xyz):
        total = logits.sum() * 0.
        valid_batches = 0
        valid_predictions = 0
        valid_targets = 0
        flat_labels = labels.to(points.device).long().flatten(1)
        flat_xyz = xyz.to(points.device).flatten(1, -2)
        for batch_index in range(points.shape[0]):
            prediction_mask = valid_masks[batch_index].bool()
            target_mask = flat_labels[batch_index] != self.empty_label
            if not prediction_mask.any() or not target_mask.any():
                continue
            predicted_points = points[batch_index, prediction_mask]
            target_points = flat_xyz[batch_index, target_mask]
            assignment = _nearest_indices(
                predicted_points, target_points, self.chunk_size)
            target = flat_labels[batch_index, target_mask][assignment]
            prediction = logits[batch_index, prediction_mask]
            target_one_hot = F.one_hot(
                target, num_classes=prediction.shape[-1]).to(prediction.dtype)
            probability = prediction.sigmoid()
            ce = F.binary_cross_entropy_with_logits(
                prediction, target_one_hot, reduction='none')
            pt = (probability * target_one_hot +
                  (1. - probability) * (1. - target_one_hot))
            alpha = (self.focal_alpha * target_one_hot +
                     (1. - self.focal_alpha) * (1. - target_one_hot))
            focal = alpha * (1. - pt).pow(self.focal_gamma) * ce
            if self.class_weights.numel():
                focal = focal * self.class_weights.to(prediction)[target].unsqueeze(-1)
            total = total + focal.sum(dim=-1).mean()
            valid_batches += 1
            valid_predictions += int(prediction_mask.sum())
            valid_targets += int(target_mask.sum())
        return (total / max(valid_batches, 1), valid_batches,
                valid_predictions, valid_targets)

    def forward(self, inputs):
        current_logits = inputs['current_gaussian_point_logits']
        zero = current_logits.sum() * 0.
        pretrain = bool(inputs.get('sparseworld_pretrain', False))
        if pretrain and not self.enable_during_pretrain:
            return zero, {'gaussian_loss_ramp': current_logits.new_zeros(()),
                          'gaussian_active_horizons': current_logits.new_zeros(())}

        metas = inputs['metas']
        xyz = metas['occ_xyz']
        current = self._loss_one(
            current_logits, inputs['current_gaussian_point_points'],
            inputs['current_gaussian_point_valid_mask'], metas['occ_label'], xyz)
        weighted = self.horizon_weights[0] * current[0]
        normalizer = self.horizon_weights[0] if current[1] else 0.
        metrics = {
            'h0_loss': current[0].detach(),
            'h0_valid_predictions': current_logits.new_tensor(current[2]),
            'h0_valid_targets': current_logits.new_tensor(current[3]),
        }

        future_logits = inputs.get('future_gaussian_point_logits', ())
        future_points = inputs.get('future_gaussian_point_points', ())
        future_masks = inputs.get('future_gaussian_point_valid_masks', ())
        future_horizons = inputs.get('future_gaussian_horizons', ())
        for logits, points, masks, horizon in zip(
                future_logits, future_points, future_masks, future_horizons):
            horizon = int(horizon)
            if horizon < 1 or horizon > 6:
                raise ValueError(f'invalid future Gaussian horizon {horizon}')
            value = self._loss_one(
                logits, points, masks,
                metas['future_occ_labels'][:, horizon - 1], xyz)
            horizon_weight = self.horizon_weights[horizon]
            weighted = weighted + horizon_weight * value[0]
            if value[1]:
                normalizer += horizon_weight
            metrics[f'h{horizon}_loss'] = value[0].detach()
            metrics[f'h{horizon}_valid_predictions'] = logits.new_tensor(value[2])
            metrics[f'h{horizon}_valid_targets'] = logits.new_tensor(value[3])

        ramp = (1. if self.warmup_iters <= 0 else min(
            1., float(inputs.get('global_iter', 0)) / self.warmup_iters))
        loss = self.weight * ramp * weighted / max(normalizer, 1e-12)
        metrics['gaussian_loss_ramp'] = current_logits.new_tensor(ramp)
        metrics['gaussian_active_horizons'] = current_logits.new_tensor(
            1 + len(future_horizons))
        scales = inputs.get('gaussian_scales')
        opacities = inputs.get('gaussian_opacities')
        residual = inputs.get('gaussian_residual')
        if scales is not None:
            metrics['scale_mean'] = scales.detach().mean()
        if opacities is not None:
            metrics['opacity_mean'] = opacities.detach().mean()
        if residual is not None:
            metrics['residual_abs_mean'] = residual.detach().abs().mean()
        return loss, metrics
