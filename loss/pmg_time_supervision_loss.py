"""Dense-time diagnostic preserving original current/future total weights."""

from . import OPENOCC_LOSS
from .sparse_opus_gausisan_world_loss import SparseOpusGaussianWorldLoss


def future_horizon_weight(max_horizon, future_weight=0.5):
    """Original sparse {2,4,6} future weight divided among all active steps.

    At the first epoch only h=1 is unfolded and the original future voxel
    budget is zero, so dense-time voxel supervision also begins at epoch 2.
    All point/world/trajectory losses retain the original curriculum.
    """
    return future_weight * (max_horizon // 2) / max_horizon if max_horizon else 0.0


@OPENOCC_LOSS.register_module()
class PMGDenseTimeWorldLoss(SparseOpusGaussianWorldLoss):
    def forward(self, inputs):
        total, details = self.strict(inputs)
        metas = inputs["metas"]
        states = inputs["gaussian_logits_by_horizon"]
        maximum = max(states)
        if set(states) != set(range(maximum + 1)):
            raise ValueError("dense-time diagnosis requires every unfolded horizon")
        for horizon, logits in states.items():
            weight = (
                self.current_weight
                if horizon == 0
                else future_horizon_weight(maximum, self.future_weight)
            )
            if weight == 0:
                continue
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
            voxel = (
                logits.sum() * 0
                if mask is not None and not mask.any()
                else self.occupancy.loss_voxel(
                    [logits], metas["occ_xyz"].flatten(1, 3), labels.flatten(1), mask
                )
            )
            total = total + weight * voxel
            details[f"gaussian.h{horizon}.loss_voxel"] = voxel.detach()
        return total, details
