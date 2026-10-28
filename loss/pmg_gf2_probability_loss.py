"""PMG A0 loss coefficients with probability NLL/Lovasz and conditional NLL."""

import torch
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .occupancy_loss import OccupancyLoss
from .sparse_opus_gausisan_world_loss import SparseOpusGaussianWorldLoss


def conditional_probability_nll(conditional, labels, weights):
    occupied = (labels != 255) & (labels != 17)
    if not occupied.any():
        return conditional.sum() * 0
    values = conditional.transpose(1, 2)[occupied].float().clamp_min(1e-6).log()
    return F.nll_loss(values, labels[occupied].long(), weight=weights.to(values))


@OPENOCC_LOSS.register_module()
class PMGGF2ProbabilityVoxelLoss(OccupancyLoss):
    def __init__(self, conditional_class_multipliers, conditional_semantic_weight=1., **kwargs):
        if kwargs.get("lovasz_use_softmax", True):
            raise ValueError("GF2 requires direct probability NLL/Lovasz")
        super().__init__(**kwargs)
        multipliers = torch.as_tensor(conditional_class_multipliers, dtype=torch.float32)
        if multipliers.shape != (17,) or not torch.isfinite(multipliers).all() or (multipliers <= 0).any():
            raise ValueError("seventeen finite positive conditional multipliers required")
        self.register_buffer("conditional_class_weights", self.class_weights[:17] * multipliers)
        self.conditional_semantic_weight = float(conditional_semantic_weight)

    def distribution_loss(self, joint, conditional, xyz, labels, mask=None):
        original = self.loss_voxel([joint], xyz, labels, mask)
        if mask is not None:
            mask = mask.flatten(1).bool()
            labels = labels[mask][None]
            conditional = conditional.transpose(1, 2)[mask][None].transpose(1, 2)
        auxiliary = conditional_probability_nll(conditional, labels, self.conditional_class_weights)
        return original + self.conditional_semantic_weight * auxiliary


@OPENOCC_LOSS.register_module()
class PMGGF2ProbabilityWorldLoss(SparseOpusGaussianWorldLoss):
    def forward(self, inputs):
        total, details = self.strict(inputs)
        metas = inputs["metas"]
        for horizon, joint in inputs["gaussian_logits_by_horizon"].items():
            labels = metas["occ_label"] if horizon == 0 else metas["future_occ_labels"][:, horizon - 1]
            mask = None
            if self.mask_type == "camera":
                mask = metas["occ_cam_mask"] if horizon == 0 else metas["future_occ_cam_masks"][:, horizon - 1]
            if mask is not None and not mask.any():
                voxel = joint.sum() * 0
            else:
                voxel = self.occupancy.distribution_loss(
                    joint, inputs["gaussian_conditionals_by_horizon"][horizon],
                    metas["occ_xyz"].flatten(1, 3), labels.flatten(1), mask)
            weight = self.current_weight if horizon == 0 else self.future_weight
            total = total + weight * voxel
            details[f"gaussian.h{horizon}.loss_voxel"] = voxel.detach()
        return total, details
