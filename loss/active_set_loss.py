"""P2 T2.3/T2.6: occupancy loss over the sampled active set.

Plan references:

* ``plans/sparse_render_plans/P2_sparse_renderer_completion.md`` task T2.3 --
  replace "predicted points only" supervision with an active-set loss whose
  sources (boundary / uncertain / empty) can be enabled one rung at a time.
* task T2.6 -- "prove the sampled loss is an unbiased estimator of the dense
  loss, or give an upper bound on what is missed".

Unbiasedness is implemented, not asserted.  If each location is drawn
independently from the dense grid with probability ``p_i``, then

    L_hat = sum_i  l_i / (V * p_i)

is an unbiased estimator of the mean dense loss ``(1/V) sum_all l``.  That mode
is ``normalizer='importance'`` and its expectation is checked numerically by
Monte-Carlo in ``tests/test_active_set_loss.py``.

**Scope of the claim.** Unbiasedness requires two things the caller must supply:
independent draws (or correct inclusion probabilities), and probabilities that
match the actual mixture.  A deterministic source is not a probability sample:
predicted points and boundary voxels are *deliberately* biased toward objects.
So the unbiasedness result here covers a properly weighted sample; using it for
the full active set additionally requires the sampler to return probabilities
consistent with its source mixture, and that remains a P2 task.  The default is
therefore ``normalizer='count'`` (a plain, admittedly biased sample mean), and
``'importance'`` is opt-in so a silently biased objective is never the default.


This module is deliberately model-agnostic: it consumes already-gathered
logits/labels at active locations.  Producing those logits requires the
operator change in P2/T2.1 (today the CUDA op only evaluates the Gaussian field
at Gaussian centres), so this loss is ready before the operator is.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from model.ops.active_sampling import (
    SOURCE_BOUNDARY, SOURCE_EMPTY, SOURCE_PREDICTED, SOURCE_UNCERTAIN,
    SOURCE_NAMES)

# Default rung weights: the plan's ladder adds sources rather than balancing
# them, so every enabled source starts at weight 1.  Overriding this is how the
# "hard empty negatives" rung is emphasised.
DEFAULT_SOURCE_WEIGHTS = {
    SOURCE_PREDICTED: 1.0,
    SOURCE_BOUNDARY: 1.0,
    SOURCE_UNCERTAIN: 1.0,
    SOURCE_EMPTY: 1.0,
}


def active_set_focal_loss(logits, targets, focal_gamma=2.0, focal_alpha=0.25,
                          class_weights=None, location_weights=None):
    """Sigmoid focal loss per active location.

    Args:
        logits: ``[Q, C]`` raw logits.
        targets: ``[Q]`` integer targets in ``[0, C-1]``.
        focal_gamma: focal focusing parameter.
        focal_alpha: focal balancing parameter.
        class_weights: optional ``[C]`` per-class multipliers.
        location_weights: optional ``[Q]`` per-location multipliers.

    Returns:
        ``[Q]`` per-location loss (already summed over classes).
    """
    if logits.dim() != 2:
        raise ValueError('logits must be [Q, C]')
    if targets.shape != logits.shape[:1]:
        raise ValueError('targets must be [Q]')
    if targets.numel() == 0:
        return logits.new_zeros(0)
    target_one_hot = F.one_hot(targets.long(),
                               num_classes=logits.shape[-1]).to(logits.dtype)
    probability = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target_one_hot,
                                            reduction='none')
    pt = probability * target_one_hot + (1. - probability) * (1. - target_one_hot)
    alpha = focal_alpha * target_one_hot + (1. - focal_alpha) * (1. - target_one_hot)
    focal = alpha * (1. - pt).pow(focal_gamma) * ce
    per_location = focal.sum(dim=-1)
    if class_weights is not None:
        per_location = per_location * class_weights.to(logits)[targets.long()]
    if location_weights is not None:
        per_location = per_location * location_weights.to(logits)
    return per_location


@OPENOCC_LOSS.register_module()
class ActiveSetOccupancyLoss(nn.Module):
    """Occupancy classification loss on the sampled active set.

    Two target conventions are supported, which matters because the current
    SparseWorld head emits 17 sigmoid class channels with no explicit empty
    channel:

    * ``logits.shape[-1] == num_classes`` -- only occupied locations are
      supervised (targets equal to ``empty_label`` are dropped), matching the
      existing point loss;
    * ``logits.shape[-1] == num_classes + 1`` -- the last channel is the empty
      logit, so empty and boundary samples are supervised too.  This is the
      convention the active-set rungs need.
    """

    def __init__(self, weight=1., num_classes=17, empty_label=17,
                 focal_gamma=2., focal_alpha=.25, class_weights=None,
                 source_weights=None, normalizer='count',
                 warmup_iters=0, enable_during_pretrain=False):
        super().__init__()
        if normalizer not in ('importance', 'count'):
            raise ValueError("normalizer must be 'importance' or 'count'")
        self.weight = float(weight)
        self.num_classes = int(num_classes)
        self.empty_label = int(empty_label)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.normalizer = normalizer
        self.warmup_iters = int(warmup_iters)
        self.enable_during_pretrain = bool(enable_during_pretrain)
        weights = dict(DEFAULT_SOURCE_WEIGHTS)
        if source_weights:
            weights.update({int(key): float(value)
                            for key, value in source_weights.items()})
        self.source_weights = weights
        self.register_buffer(
            'class_weights',
            torch.as_tensor(class_weights if class_weights is not None else [],
                            dtype=torch.float32),
            persistent=False)

    def forward(self, inputs, logits=None, labels=None, source=None,
                sample_prob=None, dense_voxel_count=None, global_iter=None):
        """Compute the loss.

        Called either as ``forward(inputs)`` (registry convention, reading
        ``active_logits`` / ``active_labels`` / ``active_source`` /
        ``active_sample_prob`` from ``inputs``) or with explicit tensors.
        """
        if logits is None:
            logits = inputs['active_logits']
            labels = inputs['active_labels']
            source = inputs.get('active_source')
            sample_prob = inputs.get('active_sample_prob')
            dense_voxel_count = inputs.get('dense_voxel_count',
                                           dense_voxel_count)
            global_iter = inputs.get('global_iter', global_iter)
        if global_iter is None:
            global_iter = 0

        has_empty_column = logits.shape[-1] == self.num_classes + 1
        zero = logits.sum() * 0.

        # Restrict to locations this convention can score.
        if has_empty_column:
            targets = labels.long().clamp(max=self.num_classes)
            keep = torch.ones_like(targets, dtype=torch.bool)
        else:
            targets = labels.long()
            keep = targets != self.empty_label
        if source is not None:
            source = source.long()

        if not bool(keep.any()):
            return zero, {'active_set_loss': zero, 'active_set_locations': zero}

        scored_logits = logits[keep]
        scored_targets = targets[keep]
        scored_source = source[keep] if source is not None else None
        scored_prob = sample_prob[keep] if sample_prob is not None else None

        location_weights = None
        if scored_source is not None:
            weight_table = logits.new_tensor(
                [self.source_weights.get(index, 1.0)
                 for index in range(max(self.source_weights) + 1)])
            location_weights = weight_table[scored_source]

        per_location = active_set_focal_loss(
            scored_logits, scored_targets, focal_gamma=self.focal_gamma,
            focal_alpha=self.focal_alpha,
            class_weights=self.class_weights if self.class_weights.numel() else None,
            location_weights=location_weights)

        # Importance weighting: sampling probabilities must be strictly
        # positive; a zero would mean an unreachable location, which is a
        # caller bug rather than something to silently clamp.
        if self.normalizer == 'importance':
            if scored_prob is None:
                raise ValueError(
                    "normalizer='importance' requires active_sample_prob")
            if bool((scored_prob <= 0).any()):
                raise ValueError('active_sample_prob must be strictly positive')
            if dense_voxel_count is None:
                raise ValueError(
                    "normalizer='importance' requires dense_voxel_count")
            # L_hat = (1 / (n * V)) * sum_j l_j / p_j estimates the mean dense
            # loss (1/V) sum_all l.  Dropping the 1/n factor would estimate
            # n * mean(l) instead -- the first version of this did exactly that.
            locations = max(int(per_location.numel()), 1)
            total = (per_location / scored_prob).sum() / (
                float(locations) * float(dense_voxel_count))
        else:
            total = per_location.sum() / max(int(per_location.numel()), 1)

        ramp = 1. if self.warmup_iters <= 0 else min(
            1., float(global_iter) / self.warmup_iters)
        loss = self.weight * ramp * total

        metrics = {'active_set_loss': loss.detach(),
                   'active_set_loss_ramp': logits.new_tensor(ramp),
                   'active_set_locations': logits.new_tensor(
                       float(per_location.numel()))}
        if scored_source is not None:
            for code, name in SOURCE_NAMES.items():
                mask = scored_source == code
                if bool(mask.any()):
                    metrics[f'active_set_{name}_loss'] = \
                        per_location[mask].mean().detach()
                    metrics[f'active_set_{name}_count'] = logits.new_tensor(
                        float(int(mask.sum().item())))
        return loss, metrics
