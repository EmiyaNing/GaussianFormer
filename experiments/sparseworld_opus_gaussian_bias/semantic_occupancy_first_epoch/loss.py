import torch

from loss import OPENOCC_LOSS
from loss.occupancy_loss import OccupancyLoss
from ..conditional_first_epoch.loss import FullspaceConditionalSemanticOccupancyLoss, conditional_semantic_ce


@OPENOCC_LOSS.register_module()
class BalancedConditionalSemanticOccupancyLoss(FullspaceConditionalSemanticOccupancyLoss):
    """Change only C's auxiliary weights; full-space CE and Lovasz are untouched."""
    def __init__(self, conditional_class_multipliers, **kwargs):
        super().__init__(**kwargs)
        multipliers = torch.as_tensor(conditional_class_multipliers, dtype=torch.float32)
        if multipliers.shape != (17,) or not torch.isfinite(multipliers).all() or (multipliers <= 0).any():
            raise ValueError('seventeen finite positive auxiliary multipliers required')
        self.register_buffer('conditional_class_weights', self.class_weights[:17] * multipliers)

    def loss_voxel(self, pred_occ, sampled_xyz, sampled_label, occ_mask=None,
                   gaussian_sem_preds=None, gaussian_sem_label=None):
        base = OccupancyLoss.loss_voxel(self, pred_occ, sampled_xyz, sampled_label, occ_mask,
                                       gaussian_sem_preds, gaussian_sem_label)
        if self.conditional_semantic_weight == 0:
            return base
        labels = sampled_label
        mask = None if occ_mask is None else occ_mask.flatten(1).bool()
        if mask is not None:
            labels = labels[mask][None]
        auxiliary = base * 0.
        for logits in pred_occ:
            if mask is not None:
                logits = logits.transpose(1, 2)[mask][None].transpose(1, 2)
            auxiliary += conditional_semantic_ce(logits, labels, self.conditional_class_weights)
        return base + self.conditional_semantic_weight * auxiliary / len(pred_occ)

