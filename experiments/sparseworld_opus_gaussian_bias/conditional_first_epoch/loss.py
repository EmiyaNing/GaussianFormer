"""B's unchanged full-space objective plus GT-occupied 17-class semantic CE."""
import math

import torch
import torch.nn.functional as F

from loss import OPENOCC_LOSS
from loss.occupancy_loss import OccupancyLoss


def conditional_semantic_ce(logits, labels, class_weights):
    """Weighted mean CE over GT classes 0..16; exclude free=17 and ignore=255."""
    if logits.ndim != 3 or logits.shape[1] != 18 or tuple(labels.shape) != (logits.shape[0], logits.shape[2]):
        raise ValueError('expected logits [B,18,V] and labels [B,V]')
    labels = labels.long()
    valid = labels != 255
    if ((labels[valid] < 0) | (labels[valid] >= 18)).any():
        raise ValueError('invalid target class')
    occupied = valid & (labels != 17)
    if not occupied.any():
        return logits[:, :17].sum() * 0.
    foreground = logits.transpose(1, 2)[occupied][:, :17].float()
    weights = class_weights[:17].to(foreground)
    return F.cross_entropy(foreground, labels[occupied], weight=weights, reduction='mean')


@OPENOCC_LOSS.register_module()
class FullspaceConditionalSemanticOccupancyLoss(OccupancyLoss):
    def __init__(self, conditional_semantic_weight=1., **kwargs):
        if (kwargs.get('use_focal_loss', False) or kwargs.get('ignore_empty', False) or
                kwargs.get('use_sem_geo_scal_loss', True) or kwargs.get('use_dice_loss', False) or
                not kwargs.get('lovasz_use_softmax', True) or
                kwargs.get('num_classes', 18) != 18 or kwargs.get('empty_label', 17) != 17):
            raise ValueError('C preserves B full-space softmax CE/Lovasz and adds only conditional semantic CE')
        super().__init__(**kwargs)
        self.conditional_semantic_weight = float(conditional_semantic_weight)
        if not math.isfinite(self.conditional_semantic_weight) or self.conditional_semantic_weight < 0:
            raise ValueError('conditional semantic weight must be finite and nonnegative')

    def loss_voxel(self, pred_occ, sampled_xyz, sampled_label, occ_mask=None,
                   gaussian_sem_preds=None, gaussian_sem_label=None):
        original = super().loss_voxel(pred_occ, sampled_xyz, sampled_label, occ_mask,
                                      gaussian_sem_preds, gaussian_sem_label)
        if self.conditional_semantic_weight == 0:
            return original
        if not pred_occ:
            raise ValueError('at least one decoder prediction is required')
        labels = sampled_label
        mask = None if occ_mask is None else occ_mask.flatten(1).bool()
        if mask is not None:
            labels = labels[mask][None]
        auxiliary = original * 0.
        for logits in pred_occ:
            if mask is not None:
                logits = logits.transpose(1, 2)[mask][None].transpose(1, 2)
            auxiliary = auxiliary + conditional_semantic_ce(logits, labels, self.class_weights)
        # lambda=1 is independent of the existing 18-class CE coefficient=10.
        return original + self.conditional_semantic_weight * auxiliary / len(pred_occ)
