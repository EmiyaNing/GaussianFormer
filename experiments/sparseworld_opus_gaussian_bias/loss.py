"""Original SparseWorld objective plus shared current/future voxel losses."""
import torch
from torch import nn

from loss import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class SparseWorldOPUSGaussianBiasLoss(nn.Module):
    def __init__(self, strict_loss, occupancy_loss, mask_type='camera',
                 current_weight=1., future_weight=.5):
        super().__init__()
        self.strict = OPENOCC_LOSS.build(strict_loss)
        self.occupancy = OPENOCC_LOSS.build(occupancy_loss)
        if mask_type not in ('camera', 'none'):
            raise ValueError('mask_type must be camera or none')
        self.mask_type = mask_type
        self.current_weight = current_weight
        self.future_weight = future_weight

    def forward(self, inputs):
        total, details = self.strict(inputs)
        metas = inputs['metas']
        for horizon, logits in inputs['gaussian_logits_by_horizon'].items():
            label = (metas['occ_label'] if horizon == 0 else
                     metas['future_occ_labels'][:, horizon - 1])
            mask = None
            if self.mask_type == 'camera':
                mask = (metas['occ_cam_mask'] if horizon == 0 else
                        metas['future_occ_cam_masks'][:, horizon - 1])
            if mask is not None and not mask.any():
                voxel = logits.sum() * 0.
            else:
                voxel = self.occupancy.loss_voxel(
                    [logits], metas['occ_xyz'].flatten(1, 3), label.flatten(1), mask)
            weight = self.current_weight if horizon == 0 else self.future_weight
            total = total + weight * voxel
            details[f'gaussian.h{horizon}.loss_voxel'] = voxel.detach()
        return total, details
