"""Direct supervision for the rendered occupied-vs-empty decision margin."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .multi_loss import MultiLoss


@OPENOCC_LOSS.register_module()
class OccupancyMarginLoss(torch.nn.Module):
    """Train the exact margin whose sign determines binary occupancy.

    For raw rendered voxel logits ``[B,C,V]``, the margin is the largest
    foreground logit minus the empty logit.  Its zero crossing is identical
    to the normal argmax occupied/empty decision at inference.
    """

    def __init__(self, empty_label=17, bce_weight=1.0, dice_weight=1.0,
                 weight=1.0, max_positive_weight=50.0, eps=1e-6):
        super().__init__()
        self.empty_label = int(empty_label)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.weight = float(weight)
        self.max_positive_weight = float(max_positive_weight)
        self.eps = float(eps)

    def forward(self, inputs):
        logits = inputs['pred_occ'][-1]
        if logits.ndim != 3 or logits.shape[1] != self.empty_label + 1:
            raise ValueError('pred_occ final layer must be [B, C, V] with empty last')
        labels = inputs['sampled_label'].reshape(logits.shape[0], -1).long()
        if labels.shape[1] != logits.shape[2]:
            raise ValueError('sampled_label and rendered logits have different voxel counts')
        margin = logits[:, :self.empty_label].amax(dim=1) - logits[:, self.empty_label]
        occupied = (labels != self.empty_label).to(margin.dtype)
        mask = inputs.get('occ_mask')
        if mask is not None:
            valid = mask.reshape_as(labels).bool()
            margin = margin[valid]
            occupied = occupied[valid]
        else:
            margin = margin.reshape(-1)
            occupied = occupied.reshape(-1)
        if margin.numel() == 0:
            raise ValueError('OccupancyMarginLoss received no supervised voxels')
        positive_count = occupied.sum().detach()
        negative_count = occupied.numel() - positive_count
        positive_weight = (negative_count / positive_count.clamp_min(1)).clamp(
            min=1.0, max=self.max_positive_weight)
        bce = F.binary_cross_entropy_with_logits(
            margin, occupied, pos_weight=positive_weight)
        probability = margin.sigmoid()
        intersection = (probability * occupied).sum()
        dice = 1.0 - (2.0 * intersection + self.eps) / (
            probability.sum() + occupied.sum() + self.eps)
        loss = self.weight * (self.bce_weight * bce + self.dice_weight * dice)
        return loss, {
            'bce': bce.detach(),
            'positive_weight': positive_weight.detach(),
            'dice': dice.detach(),
            'predicted_occupied_fraction': (margin > 0).float().mean().detach(),
            'target_occupied_fraction': occupied.mean().detach(),
        }


@OPENOCC_LOSS.register_module()
class MultiLossWithMargin(MultiLoss):
    """Append margin supervision without copying the base experiment losses."""

    def __init__(self, loss_cfgs, margin_cfg):
        super().__init__(list(loss_cfgs) + [margin_cfg])
