"""D's original SparseWorld objective plus balanced conditional voxel CE.

The 18-class full-space term still supervises free voxels. The additional
17-class term uses only GT occupied voxels and normalizes by their summed
target weights, so it never directly differentiates the empty-class logit.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .occupancy_loss import OccupancyLoss


def conditional_semantic_ce(logits, labels, class_weights):
    """Weighted mean CE over GT labels 0..16; free=17 and ignore=255 excluded.

    ``logits`` has shape [batch, 18, voxels], ``labels`` [batch, voxels].
    Predictions that chose free still contribute when their GT is occupied.
    """
    if (
        logits.ndim != 3
        or logits.shape[1] != 18
        or tuple(labels.shape) != (logits.shape[0], logits.shape[2])
    ):
        raise ValueError("expected logits [B,18,V] and labels [B,V]")
    labels = labels.long()
    valid = labels != 255
    if ((labels[valid] < 0) | (labels[valid] >= 18)).any():
        raise ValueError("invalid target class")
    occupied = valid & (labels != 17)
    if not occupied.any():
        return logits[:, :17].sum() * 0.0
    foreground = logits.transpose(1, 2)[occupied][:, :17].float()
    weights = class_weights[:17].to(foreground)
    return F.cross_entropy(
        foreground, labels[occupied], weight=weights, reduction="mean"
    )


@OPENOCC_LOSS.register_module()
class SparseOpusGaussianVoxelLoss(OccupancyLoss):
    """Full-space CE/Lovasz plus D's frequency-balanced conditional semantic CE.

    The static configuration supplies multipliers clipped from inverse square
    root training frequencies. They affect the auxiliary term only. Lambda
    is independent of the original 18-class CE coefficient (10 in D).
    """

    def __init__(
        self, conditional_class_multipliers, conditional_semantic_weight=1.0, **kwargs
    ):
        if (
            kwargs.get("use_focal_loss", False)
            or kwargs.get("ignore_empty", False)
            or kwargs.get("use_sem_geo_scal_loss", True)
            or kwargs.get("use_dice_loss", False)
            or not kwargs.get("lovasz_use_softmax", True)
            or kwargs.get("num_classes", 18) != 18
            or kwargs.get("empty_label", 17) != 17
        ):
            raise ValueError(
                "D requires full-space 18-class softmax CE/Lovasz with free=17"
            )
        super().__init__(**kwargs)
        self.conditional_semantic_weight = float(conditional_semantic_weight)
        if (
            not math.isfinite(self.conditional_semantic_weight)
            or self.conditional_semantic_weight < 0
        ):
            raise ValueError(
                "conditional semantic weight must be finite and nonnegative"
            )
        multipliers = torch.as_tensor(
            conditional_class_multipliers, dtype=torch.float32
        )
        if (
            multipliers.shape != (17,)
            or not torch.isfinite(multipliers).all()
            or (multipliers <= 0).any()
        ):
            raise ValueError("seventeen finite positive auxiliary multipliers required")
        self.register_buffer(
            "conditional_class_weights", self.class_weights[:17] * multipliers
        )

    def loss_voxel(
        self,
        pred_occ,
        sampled_xyz,
        sampled_label,
        occ_mask=None,
        gaussian_sem_preds=None,
        gaussian_sem_label=None,
    ):
        original = super().loss_voxel(
            pred_occ,
            sampled_xyz,
            sampled_label,
            occ_mask,
            gaussian_sem_preds,
            gaussian_sem_label,
        )
        if self.conditional_semantic_weight == 0:
            return original
        labels = sampled_label
        mask = None if occ_mask is None else occ_mask.flatten(1).bool()
        if mask is not None:
            labels = labels[mask][None]
        auxiliary = original * 0.0
        for logits in pred_occ:
            if mask is not None:
                logits = logits.transpose(1, 2)[mask][None].transpose(1, 2)
            # Preserve D's accumulation graph as well as its weighted mean.
            auxiliary += conditional_semantic_ce(
                logits, labels, self.conditional_class_weights
            )
        return original + self.conditional_semantic_weight * auxiliary / len(pred_occ)


@OPENOCC_LOSS.register_module()
class SparseOpusGaussianWorldLoss(nn.Module):
    """Combine unchanged point/world/trajectory losses with rendered voxels."""

    def __init__(
        self,
        strict_loss,
        occupancy_loss,
        mask_type="camera",
        current_weight=1.0,
        future_weight=0.5,
    ):
        super().__init__()
        self.strict = OPENOCC_LOSS.build(strict_loss)
        self.occupancy = OPENOCC_LOSS.build(occupancy_loss)
        if mask_type not in ("camera", "none"):
            raise ValueError("mask_type must be camera or none")
        self.mask_type = mask_type
        self.current_weight = current_weight
        self.future_weight = future_weight

    def forward(self, inputs):
        total, details = self.strict(inputs)
        metas = inputs["metas"]
        for horizon, logits in inputs["gaussian_logits_by_horizon"].items():
            labels = (
                metas["occ_label"]
                if horizon == 0
                else metas["future_occ_labels"][:, horizon - 1]
            )
            mask = None
            if self.mask_type == "camera":
                mask = (
                    metas["occ_cam_mask"]
                    if horizon == 0
                    else metas["future_occ_cam_masks"][:, horizon - 1]
                )
            if mask is not None and not mask.any():
                voxel = logits.sum() * 0.0
            else:
                voxel = self.occupancy.loss_voxel(
                    [logits], metas["occ_xyz"].flatten(1, 3), labels.flatten(1), mask
                )
            weight = self.current_weight if horizon == 0 else self.future_weight
            total = total + weight * voxel
            details[f"gaussian.h{horizon}.loss_voxel"] = voxel.detach()
        return total, details
