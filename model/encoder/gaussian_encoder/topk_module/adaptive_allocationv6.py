"""Memory-bounded, cost-aware adaptive Gaussian allocation.

V6 deliberately does not introduce explicit diagnostic heads.  It extends the
allocation state with spatial cues, makes the risky selector differentiable,
and materializes one cost-aware operation for each selected Gaussian.  Proxy
targets and auxiliary losses are consumed outside this module through
``get_allocation_aux``.
"""

from typing import Dict, List, Tuple

from mmengine.model import BaseModule
from mmengine.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import GaussianPrediction
from ....utils.safe_ops import safe_inverse_sigmoid


PASS_THROUGH = 0
CLONE_PARENT = 1
CLONE_CHILD = 2
SPLIT_CHILD = 3
ATTEN = 4
PADDING = 5

OP_CLONE = 0
OP_SPLIT = 1
OP_ATTEN = 2


def _opacity_to_anchor_logit(
    opacity: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    opacity = torch.where(
        torch.isfinite(opacity),
        opacity,
        torch.full_like(opacity, 0.5),
    )
    opacity = opacity.clamp(eps, 1.0 - eps)
    logit = torch.log(opacity / (1.0 - opacity))
    logit = torch.where(torch.isfinite(logit), logit, torch.zeros_like(logit))
    return logit.clamp(-9.21, 9.21)


def get_max_scale_axis(
    rotation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Return the world-space direction of each Gaussian's longest axis."""
    max_scale_dim = scale.argmax(dim=-1, keepdim=True)
    w, x, y, z = rotation.unbind(dim=-1)

    col_x = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], dim=-1)
    col_y = torch.stack([
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
    ], dim=-1)
    col_z = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ], dim=-1)

    all_cols = torch.stack([col_x, col_y, col_z], dim=-2)
    index = max_scale_dim.unsqueeze(-1).expand(*max_scale_dim.shape, 3)
    axis_dir = all_cols.gather(dim=-2, index=index).squeeze(-2)
    return F.normalize(axis_dir, dim=-1, eps=1e-6)


@MODELS.register_module()
class AdaptiveAllocationV6(BaseModule):
    """Cost-aware TopK clone/split/attenuation allocator.

    The public forward interface remains the three-tuple used by the existing
    Gaussian encoder.  Non-detached tensors needed by auxiliary losses can be
    retrieved with :meth:`get_allocation_aux` immediately after forward.
    """

    def __init__(
        self,
        feat_embed_dim=128,
        semantic_dim=17,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        hidden_dim=256,
        allocation_ratio=0.3,
        router_temperature=1.0,
        selector_temperature=0.1,
        risk_floor=0.1,
        m_min=0.05,
        opacity_floor=1e-4,
        neighbor_k=8,
        neighbor_radius=2.0,
        cost_grid_size=0.5,
        cost_scale_multiplier=3.0,
        cost_budget_ratio=1.3,
        cost_priority_power=0.5,
        cache_router_stats=True,
        **kwargs,
    ):
        super().__init__()
        if pc_range is None:
            raise ValueError("AdaptiveAllocationV6 requires pc_range.")
        if scale_range is None:
            raise ValueError("AdaptiveAllocationV6 requires scale_range.")
        if cost_budget_ratio < 1.0:
            raise ValueError("cost_budget_ratio must be at least 1.0.")

        self.feat_embed_dim = feat_embed_dim
        self.semantic_dim = semantic_dim
        self.pc_range = tuple(float(v) for v in pc_range)
        self.scale_range = tuple(float(v) for v in scale_range)
        self.allocation_ratio = float(allocation_ratio)
        self.router_temperature = float(router_temperature)
        self.selector_temperature = float(selector_temperature)
        self.risk_floor = float(risk_floor)
        self.m_min = float(m_min)
        self.opacity_floor = float(opacity_floor)
        self.neighbor_k = int(neighbor_k)
        self.neighbor_radius = float(neighbor_radius)
        self.cost_grid_size = float(cost_grid_size)
        self.cost_scale_multiplier = float(cost_scale_multiplier)
        self.cost_budget_ratio = float(cost_budget_ratio)
        self.cost_priority_power = float(cost_priority_power)
        self.cache_router_stats = cache_router_stats
        self.boundary_margin = 0.1

        if unit_xyz is not None:
            unit_prob = [
                unit_xyz[i] / (self.pc_range[i + 3] - self.pc_range[i])
                for i in range(3)
            ]
            self.unit_sigmoid = [4 * value for value in unit_prob]
        else:
            self.unit_sigmoid = [1.0, 1.0, 1.0]

        # geometry(6), semantics(3), xyz(3), opacity(1), local/boundary(7)
        state_input_dim = feat_embed_dim + 20
        self.allocation_state_encoder_v6 = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.risky_score_head = nn.Linear(hidden_dim, 1)
        self.risky_gate = nn.Linear(hidden_dim, 3)

        self.clone_dir_head = nn.Linear(hidden_dim, 3)
        self.clone_dist_head = nn.Linear(hidden_dim, 1)
        self.clone_scale_head = nn.Linear(hidden_dim, 1)
        self.clone_opacity_head = nn.Linear(hidden_dim, 1)
        self.clone_feature_head = nn.Linear(hidden_dim, feat_embed_dim)
        self.clone_semantic_head = nn.Linear(hidden_dim, semantic_dim)

        self.split_dir_head = nn.Linear(hidden_dim, 3)
        self.split_dist_head = nn.Linear(hidden_dim, 1)
        # Per-axis ratios allow the split outcome loss to reduce anisotropy.
        self.split_axis_scale_head = nn.Linear(hidden_dim, 3)
        self.split_opacity_head = nn.Linear(hidden_dim, 2)
        self.split_feature_head = nn.Linear(hidden_dim, 2 * feat_embed_dim)
        self.split_semantic_head = nn.Linear(hidden_dim, 2 * semantic_dim)

        self.atten_opacity_head = nn.Linear(hidden_dim, 1)

        self._allocation_aux = None
        self.latest_selected_mask = None
        self.latest_risky_score = None
        self.latest_risky_operation_prob = None
        self.latest_risky_operation_stats = None
        self.latest_output_candidate_labels = None

    @staticmethod
    def build_geometry_cue(
        scale: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mean_scale = scale.mean(dim=-1, keepdim=True)
        max_scale = scale.max(dim=-1, keepdim=True).values
        min_scale = scale.min(dim=-1, keepdim=True).values
        log_volume = torch.log(scale.prod(dim=-1, keepdim=True) + eps)
        anisotropy_ratio = max_scale / (min_scale + eps)
        return torch.cat([
            scale,
            mean_scale,
            log_volume,
            anisotropy_ratio,
        ], dim=-1)

    @staticmethod
    def build_semantic_cue(
        semantic: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        prob = semantic.softmax(dim=-1)
        entropy = -(prob * (prob + eps).log()).sum(dim=-1, keepdim=True)
        if semantic.shape[-1] > 1:
            top2_prob = prob.topk(k=2, dim=-1).values
            margin = top2_prob[..., 0:1] - top2_prob[..., 1:2]
        else:
            margin = prob
        confidence = prob.max(dim=-1, keepdim=True).values
        return torch.cat([entropy, margin, confidence], dim=-1)

    def _normalized_xyz_and_boundary(
        self,
        means: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pc_min = means.new_tensor(self.pc_range[:3])
        pc_max = means.new_tensor(self.pc_range[3:])
        xyz = ((means - pc_min) / (pc_max - pc_min).clamp_min(1e-6)).clamp(0, 1)
        boundary = torch.minimum(xyz, 1.0 - xyz).amin(dim=-1, keepdim=True)
        return xyz, boundary

    @staticmethod
    def _gather_neighbors(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(value.shape[0], device=value.device)[:, None, None]
        return value[batch, index.clamp_min(0)]

    def _radius_neighbor_indices(self, points: torch.Tensor) -> torch.Tensor:
        """Find a bounded local neighborhood without building an NxN tensor."""
        count = min(self.neighbor_k + 1, points.shape[1])
        if count <= 0:
            return torch.empty(
                *points.shape[:2], 0,
                dtype=torch.long,
                device=points.device,
            )

        if points.is_cuda:
            try:
                import frnn

                _, index, _, _ = frnn.frnn_grid_points(
                    points.float(),
                    points.float(),
                    K=count,
                    r=self.neighbor_radius,
                    return_nn=True,
                )
                return index.long()
            except (ImportError, RuntimeError):
                pass

        # This path is primarily for CPU tests.  Chunking caps peak memory.
        output = []
        chunk_size = 512
        for batch_id in range(points.shape[0]):
            batch_output = []
            reference = points[batch_id].float()
            for start in range(0, points.shape[1], chunk_size):
                query = reference[start:start + chunk_size]
                distance = torch.cdist(query, reference)
                distance = distance.masked_fill(
                    distance > self.neighbor_radius,
                    float("inf"),
                )
                values, index = distance.topk(count, largest=False, dim=-1)
                index = index.masked_fill(~torch.isfinite(values), -1)
                batch_output.append(index)
            output.append(torch.cat(batch_output, dim=0))
        return torch.stack(output, dim=0)

    def build_local_spatial_cue(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        semantics: torch.Tensor,
    ) -> torch.Tensor:
        """Build count/distance/overlap/semantic/density/coverage cues."""
        with torch.no_grad():
            work_means = means.detach().float()
            work_scales = scales.detach().float().clamp_min(1e-4)
            work_opacity = opacities.detach().float()
            work_semantic = semantics.detach().float().softmax(dim=-1)
            index = self._radius_neighbor_indices(work_means)

            if index.shape[-1] == 0:
                local = work_means.new_zeros(*work_means.shape[:2], 6)
            else:
                self_index = torch.arange(
                    work_means.shape[1],
                    device=work_means.device,
                )[None, :, None]
                valid = (index >= 0) & (index != self_index)
                valid_f = valid.float()

                neighbor_means = self._gather_neighbors(work_means, index)
                neighbor_scales = self._gather_neighbors(work_scales, index)
                neighbor_opacity = self._gather_neighbors(work_opacity, index)
                neighbor_semantic = self._gather_neighbors(work_semantic, index)

                delta = neighbor_means - work_means.unsqueeze(2)
                distance = delta.norm(dim=-1)
                count = valid_f.sum(dim=-1, keepdim=True)
                denominator = count.clamp_min(1.0)

                sigma2 = (
                    work_scales.unsqueeze(2).square()
                    + neighbor_scales.square()
                ).clamp_min(1e-6)
                mahalanobis = (delta.square() / sigma2).sum(dim=-1)
                overlap = torch.exp(-0.5 * mahalanobis) * valid_f
                semantic_agreement = (
                    work_semantic.unsqueeze(2) * neighbor_semantic
                ).sum(dim=-1) * valid_f
                density = overlap * neighbor_opacity.squeeze(-1)

                neighbor_count = count / max(float(self.neighbor_k), 1.0)
                mean_distance = (
                    (distance * valid_f).sum(dim=-1, keepdim=True)
                    / denominator
                    / max(self.neighbor_radius, 1e-6)
                )
                overlap_sum = overlap.sum(dim=-1, keepdim=True) / denominator
                agreement = (
                    semantic_agreement.sum(dim=-1, keepdim=True) / denominator
                )
                density_mass = density.sum(dim=-1, keepdim=True) / denominator
                coverage_deficit = torch.exp(-density_mass)
                local = torch.cat([
                    neighbor_count.clamp(0, 1),
                    mean_distance.clamp(0, 1),
                    overlap_sum.clamp(0, 1),
                    agreement.clamp(0, 1),
                    density_mass.clamp(0, 1),
                    coverage_deficit.clamp(0, 1),
                ], dim=-1)

            _, boundary = self._normalized_xyz_and_boundary(work_means)
            return torch.cat([local, boundary], dim=-1).to(means.dtype)

    def encode_allocation_state(
        self,
        instance_feature: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        semantics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        geometry_cue = self.build_geometry_cue(scales)
        semantic_cue = self.build_semantic_cue(semantics)
        xyz, _ = self._normalized_xyz_and_boundary(means)
        local_cue = self.build_local_spatial_cue(
            means,
            scales,
            opacities,
            semantics,
        )
        state = torch.cat([
            instance_feature,
            geometry_cue,
            semantic_cue,
            xyz,
            opacities,
            local_cue,
        ], dim=-1)
        return self.allocation_state_encoder_v6(state), local_cue

    def compute_risky_operation_prob(self, h: torch.Tensor) -> torch.Tensor:
        temperature = max(self.router_temperature, 1e-6)
        return (self.risky_gate(h) / temperature).softmax(dim=-1)

    def _predict_clone_geometry(
        self,
        h: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        direction = F.normalize(self.clone_dir_head(h), dim=-1, eps=1e-6)
        distance = torch.sigmoid(self.clone_dist_head(h)) * scales.mean(
            dim=-1,
            keepdim=True,
        )
        ratio = 0.7 + 0.3 * torch.sigmoid(self.clone_scale_head(h))
        child_means = self._clamp_means(means + distance * direction)
        child_scales = (scales * ratio).clamp(*self.scale_range)
        return child_means, child_scales

    def _predict_split_geometry(
        self,
        h: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        axis = get_max_scale_axis(rotations, scales)
        learned = F.normalize(self.split_dir_head(h), dim=-1, eps=1e-6)
        direction = F.normalize(axis + learned, dim=-1, eps=1e-6)
        distance = torch.sigmoid(self.split_dist_head(h)) * scales.mean(
            dim=-1,
            keepdim=True,
        )
        ratio = 0.45 + 0.30 * torch.sigmoid(self.split_axis_scale_head(h))
        offset = distance * direction
        child_1_means = self._clamp_means(means + offset)
        child_2_means = self._clamp_means(means - offset)
        child_scales = (scales * ratio).clamp(*self.scale_range)
        return child_1_means, child_2_means, child_scales

    def _clamp_means(self, means: torch.Tensor) -> torch.Tensor:
        margin = self.boundary_margin
        return torch.stack([
            means[..., 0].clamp(
                self.pc_range[0] + margin,
                self.pc_range[3] - margin,
            ),
            means[..., 1].clamp(
                self.pc_range[1] + margin,
                self.pc_range[4] - margin,
            ),
            means[..., 2].clamp(
                self.pc_range[2] + margin,
                self.pc_range[5] - margin,
            ),
        ], dim=-1)

    def estimate_tile_cost(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce localagg's clipped integer tile-box volume."""
        pc_min = means.new_tensor(self.pc_range[:3])
        pc_max = means.new_tensor(self.pc_range[3:])
        grid_shape = torch.round(
            (pc_max - pc_min) / self.cost_grid_size,
        ).long()
        center = ((means.detach() - pc_min) / self.cost_grid_size).long()
        radius = torch.ceil(
            scales.detach() * self.cost_scale_multiplier / self.cost_grid_size,
        ).long().clamp_min(1)
        rect_min = (center - radius).clamp_min(0)
        rect_max = center + radius + 1
        rect_min = torch.minimum(rect_min, grid_shape)
        rect_max = torch.minimum(rect_max.clamp_min(0), grid_shape)
        extent = (rect_max - rect_min).clamp_min(0)
        return extent.prod(dim=-1).to(means.dtype)

    def _operation_costs(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        clone_means: torch.Tensor,
        clone_scales: torch.Tensor,
        split_1_means: torch.Tensor,
        split_2_means: torch.Tensor,
        split_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parent = self.estimate_tile_cost(means, scales)
        clone = parent + self.estimate_tile_cost(clone_means, clone_scales)
        split = (
            self.estimate_tile_cost(split_1_means, split_scales)
            + self.estimate_tile_cost(split_2_means, split_scales)
        )
        operation = torch.stack([clone, split, parent], dim=-1)
        return parent, operation

    def _compute_cost_aware_topk(
        self,
        risk_score: torch.Tensor,
        operation_prob: torch.Tensor,
        operation_cost: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_cost = (operation_prob * operation_cost.detach()).sum(dim=-1)
        priority = risk_score / expected_cost.clamp_min(1.0).pow(
            self.cost_priority_power,
        )
        count = min(
            max(1, int(self.allocation_ratio * risk_score.shape[1])),
            risk_score.shape[1],
        )
        _, topk_index = torch.topk(priority, k=count, dim=1)
        hard_mask = torch.zeros_like(risk_score)
        hard_mask.scatter_(1, topk_index, 1.0)

        threshold = priority.gather(1, topk_index[:, -1:]).detach()
        soft_mask = torch.sigmoid(
            (priority - threshold) / max(self.selector_temperature, 1e-6),
        )
        selection_st = hard_mask + soft_mask - soft_mask.detach()
        return hard_mask.bool(), topk_index, selection_st, priority

    def _choose_operations_under_budget(
        self,
        selected_index: torch.Tensor,
        operation_prob: torch.Tensor,
        parent_cost: torch.Tensor,
        operation_cost: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        selected_prob = operation_prob[selected_index]
        selected_cost = operation_cost[selected_index]
        operation_id = selected_prob.argmax(dim=-1)

        base_cost = parent_cost.sum()
        budget = base_cost * self.cost_budget_ratio
        chosen_cost = selected_cost.gather(1, operation_id[:, None]).squeeze(1)
        output_cost = (
            base_cost
            - parent_cost[selected_index].sum()
            + chosen_cost.sum()
        )
        downgrade_count = 0

        if output_cost.detach() > budget.detach():
            cheapest_cost, cheapest_id = selected_cost.min(dim=-1)
            saving = chosen_cost - cheapest_cost
            saving_sorted, order = saving.sort(descending=True)
            cumulative = saving_sorted.cumsum(dim=0)
            excess = (output_cost - budget).detach().clamp_min(0)
            enough = torch.nonzero(cumulative >= excess, as_tuple=False)
            downgrade_count = (
                int(enough[0, 0].item()) + 1
                if enough.numel() > 0
                else selected_index.numel()
            )
            downgrade_local = order[:downgrade_count]
            operation_id = operation_id.clone()
            operation_id[downgrade_local] = cheapest_id[downgrade_local]

        return operation_id, downgrade_count, budget

    def _with_effective_opacity(
        self,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, GaussianPrediction]:
        if self.opacity_floor > 0:
            weight = weight.clamp_min(self.opacity_floor)
        opacity = (gaussian.opacities * weight).clamp(1e-8, 1.0 - 1e-8)
        weighted = GaussianPrediction(
            means=gaussian.means,
            scales=gaussian.scales,
            rotations=gaussian.rotations,
            opacities=opacity,
            semantics=gaussian.semantics,
        )
        weighted_anchor = anchor.clone()
        weighted_anchor[..., 10:11] = _opacity_to_anchor_logit(opacity)
        return weighted_anchor, weighted

    def _clone_branch(
        self,
        features: torch.Tensor,
        h: torch.Tensor,
        parent: GaussianPrediction,
        parent_anchor: torch.Tensor,
        child_means: torch.Tensor,
        child_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, GaussianPrediction]:
        opacity = parent.opacities * torch.sigmoid(self.clone_opacity_head(h))
        semantic_delta = self.clone_semantic_head(h)
        child = GaussianPrediction(
            means=child_means,
            scales=child_scales,
            rotations=parent.rotations,
            opacities=opacity,
            semantics=parent.semantics + semantic_delta,
        )
        child_anchor = self._build_anchor(
            child,
            semantic_anchor=parent_anchor[..., 11:] + semantic_delta,
        )
        child_feature = features + self.clone_feature_head(h)
        return child_feature, child_anchor, child

    def _split_branch(
        self,
        features: torch.Tensor,
        h: torch.Tensor,
        parent: GaussianPrediction,
        parent_anchor: torch.Tensor,
        child_1_means: torch.Tensor,
        child_2_means: torch.Tensor,
        child_scales: torch.Tensor,
    ):
        opacity_ratio = torch.sigmoid(self.split_opacity_head(h))
        semantic_delta = self.split_semantic_head(h)
        feature_delta = self.split_feature_head(h)
        semantic_dim = self.semantic_dim
        feature_dim = self.feat_embed_dim

        child_1 = GaussianPrediction(
            means=child_1_means,
            scales=child_scales,
            rotations=parent.rotations,
            opacities=parent.opacities * opacity_ratio[..., 0:1],
            semantics=parent.semantics + semantic_delta[..., :semantic_dim],
        )
        child_2 = GaussianPrediction(
            means=child_2_means,
            scales=child_scales,
            rotations=parent.rotations,
            opacities=parent.opacities * opacity_ratio[..., 1:2],
            semantics=parent.semantics + semantic_delta[..., semantic_dim:],
        )
        anchor_1 = self._build_anchor(
            child_1,
            semantic_anchor=(
                parent_anchor[..., 11:] + semantic_delta[..., :semantic_dim]
            ),
        )
        anchor_2 = self._build_anchor(
            child_2,
            semantic_anchor=(
                parent_anchor[..., 11:] + semantic_delta[..., semantic_dim:]
            ),
        )
        feature_1 = features + feature_delta[..., :feature_dim]
        feature_2 = features + feature_delta[..., feature_dim:]
        return feature_1, anchor_1, child_1, feature_2, anchor_2, child_2

    def _atten_branch(
        self,
        h: torch.Tensor,
        parent: GaussianPrediction,
        parent_anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, GaussianPrediction, torch.Tensor]:
        factor = self.m_min + (1 - self.m_min) * torch.sigmoid(
            self.atten_opacity_head(h),
        )
        attenuated = GaussianPrediction(
            means=parent.means,
            scales=parent.scales,
            rotations=parent.rotations,
            opacities=parent.opacities * factor,
            semantics=parent.semantics,
        )
        attenuated_anchor = parent_anchor.clone()
        attenuated_anchor[..., 10:11] = _opacity_to_anchor_logit(
            attenuated.opacities,
        )
        return attenuated_anchor, attenuated, factor

    @staticmethod
    def _sanitize_anchor_tail(anchor_tail: torch.Tensor) -> torch.Tensor:
        anchor_tail = torch.where(
            torch.isfinite(anchor_tail),
            anchor_tail,
            torch.zeros_like(anchor_tail),
        )
        return anchor_tail.clamp(min=-9.21, max=9.21)

    def _build_anchor(
        self,
        gaussian: GaussianPrediction,
        semantic_anchor: torch.Tensor = None,
    ) -> torch.Tensor:
        minimum, maximum = self.scale_range
        eps = 1e-4
        pc = self.pc_range
        xyz = torch.stack([
            (gaussian.means[..., 0] - pc[0]) / (pc[3] - pc[0]),
            (gaussian.means[..., 1] - pc[1]) / (pc[4] - pc[1]),
            (gaussian.means[..., 2] - pc[2]) / (pc[5] - pc[2]),
        ], dim=-1).clamp(eps, 1.0 - eps)
        scale = gaussian.scales.clamp(minimum + eps, maximum - eps)
        if semantic_anchor is None:
            semantic_anchor = gaussian.semantics
        return torch.cat([
            safe_inverse_sigmoid(xyz),
            torch.log((scale - minimum) / (maximum - scale + eps) + eps),
            gaussian.rotations,
            _opacity_to_anchor_logit(gaussian.opacities),
            self._sanitize_anchor_tail(semantic_anchor),
        ], dim=-1)

    def _sync_anchor_with_gaussian(
        self,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ) -> torch.Tensor:
        rebuilt = self._build_anchor(gaussian, semantic_anchor=anchor[..., 11:])
        if rebuilt.shape[-1] == anchor.shape[-1]:
            return rebuilt
        synced = anchor.clone()
        synced[..., :11] = rebuilt[..., :11]
        return synced

    def _sanitize_gaussian(
        self,
        gaussian: GaussianPrediction,
    ) -> GaussianPrediction:
        means = torch.where(
            torch.isfinite(gaussian.means),
            gaussian.means,
            torch.zeros_like(gaussian.means),
        )
        means = self._clamp_means(means)
        scales = torch.where(
            torch.isfinite(gaussian.scales),
            gaussian.scales,
            torch.full_like(gaussian.scales, self.scale_range[0]),
        ).clamp(*self.scale_range)
        rotations = torch.where(
            torch.isfinite(gaussian.rotations),
            gaussian.rotations,
            torch.zeros_like(gaussian.rotations),
        )
        norm = rotations.norm(dim=-1, keepdim=True)
        rotations = rotations / norm.clamp_min(1e-8)
        identity = torch.zeros_like(rotations)
        identity[..., 0] = 1.0
        bad = (norm < 1e-8) | ~torch.isfinite(rotations).all(
            dim=-1,
            keepdim=True,
        )
        rotations = torch.where(bad.expand_as(rotations), identity, rotations)
        opacities = torch.where(
            torch.isfinite(gaussian.opacities),
            gaussian.opacities,
            torch.zeros_like(gaussian.opacities),
        ).clamp(1e-8, 1.0 - 1e-8)
        semantics = torch.where(
            torch.isfinite(gaussian.semantics),
            gaussian.semantics,
            torch.zeros_like(gaussian.semantics),
        ).clamp(-30, 30)
        return GaussianPrediction(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            semantics=semantics,
        )

    @staticmethod
    def _make_gaussian(
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        semantics: torch.Tensor,
    ) -> GaussianPrediction:
        return GaussianPrediction(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            semantics=semantics,
        )

    def _pad_output(
        self,
        output: Dict[str, torch.Tensor],
        target_count: int,
    ) -> Dict[str, torch.Tensor]:
        pad_count = target_count - output["features"].shape[0]
        if pad_count <= 0:
            return output

        device = output["features"].device
        dtype = output["features"].dtype
        means = torch.tensor(
            [
                self.pc_range[0] + self.boundary_margin,
                self.pc_range[1] + self.boundary_margin,
                self.pc_range[2] + self.boundary_margin,
            ],
            device=device,
            dtype=dtype,
        ).expand(pad_count, 3)
        scales = torch.full(
            (pad_count, 3),
            self.scale_range[0],
            device=device,
            dtype=dtype,
        )
        rotations = torch.zeros(pad_count, 4, device=device, dtype=dtype)
        rotations[:, 0] = 1
        opacities = torch.full((pad_count, 1), 1e-8, device=device, dtype=dtype)
        semantics = torch.zeros(
            pad_count,
            self.semantic_dim,
            device=device,
            dtype=dtype,
        )
        gaussian = self._make_gaussian(
            means,
            scales,
            rotations,
            opacities,
            semantics,
        )
        anchor = torch.zeros(
            pad_count,
            output["anchors"].shape[-1],
            device=device,
            dtype=output["anchors"].dtype,
        )
        anchor = self._sync_anchor_with_gaussian(anchor, gaussian)

        output["features"] = torch.cat([
            output["features"],
            torch.zeros(pad_count, self.feat_embed_dim, device=device, dtype=dtype),
        ])
        output["anchors"] = torch.cat([output["anchors"], anchor])
        for key, value in (
            ("means", means),
            ("scales", scales),
            ("rotations", rotations),
            ("opacities", opacities),
            ("semantics", semantics),
        ):
            output[key] = torch.cat([output[key], value])
        output["labels"] = torch.cat([
            output["labels"],
            torch.full((pad_count,), PADDING, device=device, dtype=torch.long),
        ])
        output["parent_index"] = torch.cat([
            output["parent_index"],
            torch.full((pad_count,), -1, device=device, dtype=torch.long),
        ])
        return output

    def get_allocation_aux(self):
        return self._allocation_aux

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        gaussian = self._sanitize_gaussian(gaussian)
        batch_size, num_gaussians, _ = instance_feature.shape
        h, local_cue = self.encode_allocation_state(
            instance_feature,
            gaussian.means,
            gaussian.scales,
            gaussian.opacities,
            gaussian.semantics,
        )
        risk_logits = self.risky_score_head(h).squeeze(-1)
        risk_score = torch.sigmoid(risk_logits)
        operation_prob = self.compute_risky_operation_prob(h)

        clone_means, clone_scales = self._predict_clone_geometry(
            h,
            gaussian.means,
            gaussian.scales,
        )
        split_1_means, split_2_means, split_scales = (
            self._predict_split_geometry(
                h,
                gaussian.means,
                gaussian.scales,
                gaussian.rotations,
            )
        )
        parent_cost, operation_cost = self._operation_costs(
            gaussian.means,
            gaussian.scales,
            clone_means,
            clone_scales,
            split_1_means,
            split_2_means,
            split_scales,
        )
        selected_mask, topk_index, selection_st, priority = (
            self._compute_cost_aware_topk(
                risk_score,
                operation_prob,
                operation_cost,
            )
        )

        expected_operation_cost = (
            operation_prob * operation_cost.detach()
        ).sum(dim=-1)
        expected_output_cost = parent_cost.sum(dim=1) + (
            selection_st
            * (expected_operation_cost - parent_cost.detach())
        ).sum(dim=1)
        cost_budget = parent_cost.sum(dim=1) * self.cost_budget_ratio

        outputs: List[Dict[str, torch.Tensor]] = []
        records = []
        stats = []

        for batch_id in range(batch_size):
            selected_index = topk_index[batch_id]
            non_selected_index = torch.where(~selected_mask[batch_id])[0]
            operation_id, downgrade_count, hard_budget = (
                self._choose_operations_under_budget(
                    selected_index,
                    operation_prob[batch_id],
                    parent_cost[batch_id],
                    operation_cost[batch_id],
                )
            )

            chosen_prob = operation_prob[batch_id, selected_index].gather(
                1,
                operation_id[:, None],
            )
            operation_st = 1.0 + chosen_prob - chosen_prob.detach()
            selected_st = selection_st[batch_id, selected_index, None]
            selected_risk = risk_score[batch_id, selected_index, None]
            risk_weight = self.risk_floor + (1 - self.risk_floor) * selected_risk
            contribution_weight = risk_weight * selected_st * operation_st

            clone_local = torch.where(operation_id == OP_CLONE)[0]
            split_local = torch.where(operation_id == OP_SPLIT)[0]
            atten_local = torch.where(operation_id == OP_ATTEN)[0]
            clone_index = selected_index[clone_local]
            split_index = selected_index[split_local]
            atten_index = selected_index[atten_local]

            feature_parts = [instance_feature[batch_id, non_selected_index]]
            anchor_parts = [anchor[batch_id, non_selected_index]]
            means_parts = [gaussian.means[batch_id, non_selected_index]]
            scales_parts = [gaussian.scales[batch_id, non_selected_index]]
            rotation_parts = [gaussian.rotations[batch_id, non_selected_index]]
            opacity_parts = [gaussian.opacities[batch_id, non_selected_index]]
            semantic_parts = [gaussian.semantics[batch_id, non_selected_index]]
            label_parts = [torch.full(
                (non_selected_index.numel(),),
                PASS_THROUGH,
                device=anchor.device,
                dtype=torch.long,
            )]
            parent_parts = [non_selected_index]
            record = {
                "clone_index": clone_index,
                "split_index": split_index,
                "atten_index": atten_index,
            }

            if clone_index.numel() > 0:
                parent = self._make_gaussian(
                    gaussian.means[batch_id, clone_index],
                    gaussian.scales[batch_id, clone_index],
                    gaussian.rotations[batch_id, clone_index],
                    gaussian.opacities[batch_id, clone_index],
                    gaussian.semantics[batch_id, clone_index],
                )
                child_feature, child_anchor, child = self._clone_branch(
                    instance_feature[batch_id, clone_index],
                    h[batch_id, clone_index],
                    parent,
                    anchor[batch_id, clone_index],
                    clone_means[batch_id, clone_index],
                    clone_scales[batch_id, clone_index],
                )
                weight = contribution_weight[clone_local]
                parent_anchor, weighted_parent = self._with_effective_opacity(
                    anchor[batch_id, clone_index],
                    parent,
                    weight,
                )
                child_anchor, child = self._with_effective_opacity(
                    child_anchor,
                    child,
                    weight,
                )
                feature_parts.extend([
                    instance_feature[batch_id, clone_index],
                    child_feature,
                ])
                anchor_parts.extend([parent_anchor, child_anchor])
                means_parts.extend([weighted_parent.means, child.means])
                scales_parts.extend([weighted_parent.scales, child.scales])
                rotation_parts.extend([weighted_parent.rotations, child.rotations])
                opacity_parts.extend([weighted_parent.opacities, child.opacities])
                semantic_parts.extend([weighted_parent.semantics, child.semantics])
                label_parts.extend([
                    torch.full_like(clone_index, CLONE_PARENT),
                    torch.full_like(clone_index, CLONE_CHILD),
                ])
                parent_parts.extend([clone_index, clone_index])
                record["clone_parent"] = parent
                record["clone_child"] = child

            if split_index.numel() > 0:
                parent = self._make_gaussian(
                    gaussian.means[batch_id, split_index],
                    gaussian.scales[batch_id, split_index],
                    gaussian.rotations[batch_id, split_index],
                    gaussian.opacities[batch_id, split_index],
                    gaussian.semantics[batch_id, split_index],
                )
                (
                    feature_1,
                    anchor_1,
                    child_1,
                    feature_2,
                    anchor_2,
                    child_2,
                ) = self._split_branch(
                    instance_feature[batch_id, split_index],
                    h[batch_id, split_index],
                    parent,
                    anchor[batch_id, split_index],
                    split_1_means[batch_id, split_index],
                    split_2_means[batch_id, split_index],
                    split_scales[batch_id, split_index],
                )
                weight = contribution_weight[split_local]
                anchor_1, child_1 = self._with_effective_opacity(
                    anchor_1,
                    child_1,
                    weight,
                )
                anchor_2, child_2 = self._with_effective_opacity(
                    anchor_2,
                    child_2,
                    weight,
                )
                feature_parts.extend([feature_1, feature_2])
                anchor_parts.extend([anchor_1, anchor_2])
                means_parts.extend([child_1.means, child_2.means])
                scales_parts.extend([child_1.scales, child_2.scales])
                rotation_parts.extend([child_1.rotations, child_2.rotations])
                opacity_parts.extend([child_1.opacities, child_2.opacities])
                semantic_parts.extend([child_1.semantics, child_2.semantics])
                label_parts.extend([
                    torch.full_like(split_index, SPLIT_CHILD),
                    torch.full_like(split_index, SPLIT_CHILD),
                ])
                parent_parts.extend([split_index, split_index])
                record["split_parent"] = parent
                record["split_child_1"] = child_1
                record["split_child_2"] = child_2

            if atten_index.numel() > 0:
                parent = self._make_gaussian(
                    gaussian.means[batch_id, atten_index],
                    gaussian.scales[batch_id, atten_index],
                    gaussian.rotations[batch_id, atten_index],
                    gaussian.opacities[batch_id, atten_index],
                    gaussian.semantics[batch_id, atten_index],
                )
                atten_anchor, attenuated, factor = self._atten_branch(
                    h[batch_id, atten_index],
                    parent,
                    anchor[batch_id, atten_index],
                )
                atten_anchor, attenuated = self._with_effective_opacity(
                    atten_anchor,
                    attenuated,
                    contribution_weight[atten_local],
                )
                feature_parts.append(instance_feature[batch_id, atten_index])
                anchor_parts.append(atten_anchor)
                means_parts.append(attenuated.means)
                scales_parts.append(attenuated.scales)
                rotation_parts.append(attenuated.rotations)
                opacity_parts.append(attenuated.opacities)
                semantic_parts.append(attenuated.semantics)
                label_parts.append(torch.full_like(atten_index, ATTEN))
                parent_parts.append(atten_index)
                record["atten_parent"] = parent
                record["attenuated"] = attenuated
                record["atten_factor"] = factor

            output_gaussian = self._sanitize_gaussian(self._make_gaussian(
                torch.cat(means_parts, dim=0),
                torch.cat(scales_parts, dim=0),
                torch.cat(rotation_parts, dim=0),
                torch.cat(opacity_parts, dim=0),
                torch.cat(semantic_parts, dim=0),
            ))
            output_anchor = self._sync_anchor_with_gaussian(
                torch.cat(anchor_parts, dim=0),
                output_gaussian,
            )
            output = {
                "features": torch.cat(feature_parts, dim=0),
                "anchors": output_anchor,
                "means": output_gaussian.means,
                "scales": output_gaussian.scales,
                "rotations": output_gaussian.rotations,
                "opacities": output_gaussian.opacities,
                "semantics": output_gaussian.semantics,
                "labels": torch.cat(label_parts, dim=0),
                "parent_index": torch.cat(parent_parts, dim=0),
            }
            outputs.append(output)
            records.append(record)

            if self.cache_router_stats:
                output_cost = self.estimate_tile_cost(
                    output_gaussian.means,
                    output_gaussian.scales,
                ).sum()
                stats.append({
                    "input_gaussians": int(num_gaussians),
                    "output_gaussians": int(output_gaussian.means.shape[0]),
                    "selected_topk": int(selected_index.numel()),
                    "materialized_clone": int(clone_index.numel()),
                    "materialized_split": int(split_index.numel()),
                    "materialized_atten": int(atten_index.numel()),
                    "budget_downgrades": int(downgrade_count),
                    "estimated_r_input": float(parent_cost[batch_id].sum().detach().cpu()),
                    "estimated_r_output": float(output_cost.detach().cpu()),
                    "estimated_r_budget": float(hard_budget.detach().cpu()),
                    "materialized_ops_per_topk": 1.0,
                })

        target_count = max(output["features"].shape[0] for output in outputs)
        outputs = [self._pad_output(output, target_count) for output in outputs]
        result_gaussian = self._make_gaussian(
            torch.stack([output["means"] for output in outputs]),
            torch.stack([output["scales"] for output in outputs]),
            torch.stack([output["rotations"] for output in outputs]),
            torch.stack([output["opacities"] for output in outputs]),
            torch.stack([output["semantics"] for output in outputs]),
        )
        # Hard routing may leave a branch empty on one DDP rank.  A zero-valued
        # dependency keeps every branch parameter in the autograd graph.
        branch_guard = result_gaussian.opacities.new_zeros(())
        guarded_modules = (
            self.clone_opacity_head,
            self.clone_feature_head,
            self.clone_semantic_head,
            self.split_opacity_head,
            self.split_feature_head,
            self.split_semantic_head,
            self.atten_opacity_head,
        )
        for module in guarded_modules:
            for parameter in module.parameters():
                branch_guard = branch_guard + parameter.sum() * 0.0
        result_gaussian = result_gaussian._replace(
            opacities=result_gaussian.opacities + branch_guard,
        )
        result_anchor = torch.stack([output["anchors"] for output in outputs])
        result_feature = torch.stack([output["features"] for output in outputs])
        output_labels = torch.stack([output["labels"] for output in outputs])
        output_parent_index = torch.stack([
            output["parent_index"] for output in outputs
        ])

        output_cost = self.estimate_tile_cost(
            result_gaussian.means,
            result_gaussian.scales,
        ).sum(dim=1)
        self._allocation_aux = {
            "input_gaussian": gaussian,
            "risk_logits": risk_logits,
            "risk_score": risk_score,
            "risk_priority": priority,
            "operation_prob": operation_prob,
            "selected_mask": selected_mask,
            "selection_st": selection_st,
            "local_cue": local_cue.detach(),
            "parent_cost": parent_cost.detach(),
            "operation_cost": operation_cost.detach(),
            "expected_output_cost": expected_output_cost,
            "cost_budget": cost_budget.detach(),
            "estimated_r_input": parent_cost.sum(dim=1).detach(),
            "estimated_r_output": output_cost.detach(),
            "output_count": torch.tensor(
                [
                    int((output["labels"] != PADDING).sum().item())
                    for output in outputs
                ],
                device=anchor.device,
                dtype=parent_cost.dtype,
            ),
            "input_count": parent_cost.new_full(
                (batch_size,),
                float(num_gaussians),
            ),
            "output_labels": output_labels,
            "output_parent_index": output_parent_index,
            "records": records,
        }

        if self.cache_router_stats:
            self.latest_selected_mask = selected_mask.detach()
            self.latest_risky_score = risk_score.detach()
            self.latest_risky_operation_prob = operation_prob.detach()
            self.latest_output_candidate_labels = output_labels.detach()
            self.latest_risky_operation_stats = stats

        return result_anchor, result_gaussian, result_feature
