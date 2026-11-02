"""Local CE counterfactual targets for the shared Gaussian growth allocator.

Targets evaluate one proposed parent replacement + child with every other GS
fixed. They are a sampled local surrogate, not an oracle for final mIoU.
"""

import torch
from torch import nn
from torch.nn import functional as F

from model.utils.utils import get_rotation_matrix
from . import OPENOCC_LOSS


def gaussian_field(gaussian, points, pc_range, grid_size, scale_multiplier=3.0):
    """Evaluate additive semantic fields [B,M,Q,C], with exact integer AABB.

    The covariance convention matches GaussianHead: Sigma = R.T @ S^2 @ R.
    Support uses the renderer's truncation/ceil rules, not a radial cutoff.
    """
    rotation = get_rotation_matrix(gaussian.rotations.float())
    displacement = points - gaussian.means[..., None, :]
    local_displacement = torch.einsum("bmij,bmqj->bmqi", rotation, displacement)
    exponent = -0.5 * (
        local_displacement / gaussian.scales[..., None, :].clamp_min(1e-6)
    ).square().sum(-1)

    origin = points.new_tensor(pc_range[:3])
    center_cells = ((gaussian.means - origin) / grid_size).to(torch.int32)
    query_cells = ((points - origin) / grid_size).to(torch.int32)
    support_radius = torch.ceil(gaussian.scales * scale_multiplier / grid_size).to(
        torch.int32
    )
    supported = (
        (query_cells - center_cells[..., None, :]).abs() <= support_radius[..., None, :]
    ).all(-1)
    weight = exponent.exp() * gaussian.opacities * supported
    return weight[..., None] * gaussian.semantics[..., None, :]


def _neighborhood_indices(parent, child, pc_range, grid_size, grid_shape, radius):
    """Return sorted flat indices and a validity mask for deduplicated probes."""
    cell_count = grid_shape[0] * grid_shape[1] * grid_shape[2]
    axes = torch.arange(-radius, radius + 1, device=parent.means.device)
    offsets = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"), -1).reshape(
        -1, 3
    )

    centers = torch.stack([parent.means, child.means], 2)
    centers = ((centers - centers.new_tensor(pc_range[:3])) / grid_size).long()
    neighbors = centers[..., None, :] + offsets
    shape_tensor = neighbors.new_tensor(grid_shape)
    inside_grid = ((neighbors >= 0) & (neighbors < shape_tensor)).all(-1)

    flat_indices = (neighbors[..., 0] * grid_shape[1] + neighbors[..., 1]) * grid_shape[
        2
    ] + neighbors[..., 2]
    # Invalid entries sort to the end. Mask duplicates before clamping the
    # sentinel to an in-range index so subsequent gathers are always safe.
    flat_indices = (
        flat_indices.masked_fill(~inside_grid, cell_count).flatten(2).sort(-1).values
    )
    valid = flat_indices < cell_count
    valid[..., 1:] &= flat_indices[..., 1:] != flat_indices[..., :-1]
    return flat_indices.clamp_max(cell_count - 1), valid


def _voxel_cross_entropy(logits, labels, class_weight):
    return F.cross_entropy(
        logits.flatten(0, 2),
        labels.flatten(),
        weight=class_weight,
        reduction="none",
    ).reshape_as(labels)


def local_gain_targets(
    aux,
    sampled_xyz,
    sampled_label,
    mask=None,
    radius=2,
    class_weight=None,
    scale_multiplier=3.0,
):
    """Detached CE gain on the union of parent and child voxel neighborhoods."""
    with torch.no_grad():
        parent, updated, child = (
            aux[key] for key in ("probe_parent", "probe_updated", "probe_child")
        )
        batch_size, probe_count = parent.means.shape[:2]
        if not probe_count:
            return parent.means.new_empty(batch_size, 0)

        pc_range, grid_size = aux["pc_range"], aux["grid_size"]
        grid_shape = tuple(
            round((pc_range[axis + 3] - pc_range[axis]) / grid_size)
            for axis in range(3)
        )
        cell_count = grid_shape[0] * grid_shape[1] * grid_shape[2]
        if (
            sampled_xyz.shape[1] != cell_count
            or sampled_label.numel() != batch_size * cell_count
        ):
            raise ValueError(
                "V7 gain supervision requires the full fixed grid in xyz-major order"
            )

        indices, valid = _neighborhood_indices(
            parent, child, pc_range, grid_size, grid_shape, radius
        )
        batch_index = torch.arange(batch_size, device=indices.device)[:, None, None]
        points = sampled_xyz[batch_index, indices].float()
        labels = sampled_label.reshape(batch_size, -1)[batch_index, indices].long()
        valid &= (labels >= 0) & (labels < aux["base_logits"].shape[-1])
        if mask is not None:
            valid &= mask.reshape(batch_size, -1)[batch_index, indices].bool()

        base_logits = aux["base_logits"]
        if base_logits.dim() == 2:
            base_logits = base_logits[None]
        before = base_logits[batch_index, indices].float()
        difference = (
            gaussian_field(updated, points, pc_range, grid_size, scale_multiplier)
            + gaussian_field(child, points, pc_range, grid_size, scale_multiplier)
            - gaussian_field(parent, points, pc_range, grid_size, scale_multiplier)
        )
        # Foreground Gaussians contribute nothing to the empty/background logit.
        difference = F.pad(difference, (0, before.shape[-1] - difference.shape[-1]))
        safe_labels = labels.clamp(0, before.shape[-1] - 1)
        ce_before = _voxel_cross_entropy(before, safe_labels, class_weight)
        ce_after = _voxel_cross_entropy(before + difference, safe_labels, class_weight)
        return ((ce_before - ce_after) * valid).sum(-1) / valid.sum(-1).clamp_min(1)


@OPENOCC_LOSS.register_module()
class AdaptiveAllocationV7Loss(nn.Module):
    """Supervise gain scores and a short initial field-consistency warmup."""

    def __init__(
        self,
        weight=0.1,
        ranking_weight=1.0,
        regression_weight=1.0,
        ranking_temperature=0.5,
        teacher_radius=2,
        scale_multiplier=3.0,
        class_weight=None,
        stability_weight=0.05,
        stability_steps=200,
    ):
        super().__init__()
        self.weight = weight
        self.ranking_weight = ranking_weight
        self.regression_weight = regression_weight
        self.ranking_temperature = ranking_temperature
        self.teacher_radius = teacher_radius
        self.scale_multiplier = scale_multiplier
        self.stability_weight = stability_weight
        self.stability_steps = stability_steps
        self.register_buffer(
            "class_weight",
            (
                None
                if class_weight is None
                else torch.tensor(class_weight, dtype=torch.float32)
            ),
        )

    def _score_losses(self, score, gain, zero):
        if not score.shape[1]:
            return zero, zero

        centered_gain = gain - gain.mean(-1, keepdim=True)
        scale = centered_gain.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        target = (centered_gain / scale).clamp(-5, 5)
        regression = F.smooth_l1_loss(score, target)

        ranking = zero
        if score.shape[1] > 1:
            pair_count = max(1, score.shape[1] // 4)
            high_index = target.topk(pair_count, 1).indices
            low_index = target.topk(pair_count, 1, largest=False).indices
            gap = (
                target.gather(1, high_index) - target.gather(1, low_index)
            ).clamp_min(0)
            pair_loss = F.softplus(
                -(score.gather(1, high_index) - score.gather(1, low_index))
                / self.ranking_temperature
            )
            ranking = (pair_loss * gap).sum() / gap.sum().clamp_min(1e-6)
        return regression, ranking

    def _stability_loss(self, aux, zero):
        warmup = (
            max(0.0, 1 - aux["step"] / max(1, self.stability_steps))
            if self.stability_steps
            else 0.0
        )
        stability = zero
        if warmup and self.stability_weight:
            base_logits = aux["base_logits"]
            if base_logits.dim() == 2:
                base_logits = base_logits[None]
            output_logits = aux["output_logits"].transpose(1, 2)
            stability = (output_logits - base_logits).square().mean()
        return stability, warmup

    def _single_aux_loss(self, aux, inputs):
        if "base_logits" not in aux:
            raise ValueError(
                "GaussianHead must supply detached base_logits for V7 training"
            )
        gain = local_gain_targets(
            aux,
            inputs["sampled_xyz"],
            inputs["sampled_label"],
            inputs.get("occ_mask"),
            self.teacher_radius,
            self.class_weight,
            self.scale_multiplier,
        )
        score = aux["probe_score"].float()
        zero = score.sum() * 0
        regression, ranking = self._score_losses(score, gain, zero)
        stability, warmup = self._stability_loss(aux, zero)
        loss = (
            self.weight
            * (self.regression_weight * regression + self.ranking_weight * ranking)
            + self.stability_weight * warmup * stability
        )
        metrics = dict(
            loss_gain_regression=regression.detach(),
            loss_gain_ranking=ranking.detach(),
            loss_stability=stability.detach(),
            teacher_gain_mean=gain.mean() if gain.numel() else zero.detach(),
            teacher_positive_fraction=(
                (gain > 0).float().mean() if gain.numel() else zero.detach()
            ),
            parent_displacement=aux["parent_displacement"].detach(),
            child_opacity=aux["child_opacity"].detach(),
            input_gs=aux["input_count"],
            output_gs=aux["output_count"],
        )
        return loss, metrics

    def forward(self, inputs):
        aux_list = inputs.get("allocation_aux", [])
        if isinstance(aux_list, dict):
            aux_list = [aux_list]

        losses, metrics = [], {}
        for aux in aux_list:
            if aux.get("version") != 7:
                continue
            loss, metrics = self._single_aux_loss(aux, inputs)
            losses.append(loss)

        if not losses:
            # Keep the existing eval/missing-aux zero-loss and diagnostic flag.
            predictions = inputs.get("pred_occ", [])
            zero = predictions[-1].sum() * 0 if predictions else torch.tensor(0.0)
            return zero, dict(missing_allocation_aux=1.0)
        # Losses average over entries; metrics intentionally describe the final
        # V7 entry, matching the original aggregation behavior.
        return torch.stack(losses).mean(), metrics
