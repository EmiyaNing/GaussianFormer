"""
AdaptiveAllocationV5: TopK risky Gaussian selection with a soft risky-operation
candidate bank.

V5 intentionally has no keep operation inside the router. Non-TopK Gaussians
pass through unchanged. TopK Gaussians generate clone/split/attenuation
candidates, and clone/split/attenuation probabilities directly modulate each
candidate's opacity contribution.
"""

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


def _opacity_to_anchor_logit(opacity: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    opacity = torch.where(torch.isfinite(opacity), opacity, torch.full_like(opacity, 0.5))
    opacity = opacity.clamp(eps, 1.0 - eps)
    logit = torch.log(opacity / (1.0 - opacity))
    logit = torch.where(torch.isfinite(logit), logit, torch.zeros_like(logit))
    return logit.clamp(-9.21, 9.21)


def get_max_scale_axis(rotation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Get the world-space direction of the Gaussian's maximum scale axis."""
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
    idx = max_scale_dim.unsqueeze(-1).expand(-1, -1, -1, 3)
    axis_dir = all_cols.gather(dim=-2, index=idx).squeeze(-2)
    return F.normalize(axis_dir, dim=-1)


@MODELS.register_module()
class AdaptiveAllocationV5(BaseModule):
    """
    Select TopK risky Gaussians and route them softly among clone/split/atten.

    Forward returns the same three-tuple as existing densify modules:
        result_anchors, result_gaussian, result_features
    """

    def __init__(
        self,
        feat_embed_dim=128,
        semantic_dim=17,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        hidden_dim=256,
        allocation_ratio=0.4,
        router_temperature=1.0,
        m_min=0.05,
        opacity_floor=1e-3,
        cache_router_stats=True,
        **kwargs,
    ):
        super().__init__()
        self.feat_embed_dim = feat_embed_dim
        self.semantic_dim = semantic_dim
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.allocation_ratio = allocation_ratio
        self.router_temperature = router_temperature
        self.m_min = m_min
        self.opacity_floor = opacity_floor
        self.cache_router_stats = cache_router_stats
        self.boundary_margin = 0.1
        if self.pc_range is None:
            raise ValueError("AdaptiveAllocationV5 requires pc_range.")
        if self.scale_range is None:
            raise ValueError("AdaptiveAllocationV5 requires scale_range.")

        if unit_xyz is not None and pc_range is not None:
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            self.unit_sigmoid = [4 * unit_prob[i] for i in range(3)]
        else:
            self.unit_sigmoid = [1.0, 1.0, 1.0]

        geo_cue_dim = 6
        sem_cue_dim = 3
        state_input_dim = feat_embed_dim + geo_cue_dim + sem_cue_dim

        self.allocation_state_encoder = nn.Sequential(
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
        self.split_scale_head = nn.Linear(hidden_dim, 1)
        self.split_opacity_head = nn.Linear(hidden_dim, 2)
        self.split_feature_head = nn.Linear(hidden_dim, 2 * feat_embed_dim)
        self.split_semantic_head = nn.Linear(hidden_dim, 2 * semantic_dim)

        self.atten_opacity_head = nn.Linear(hidden_dim, 1)

        self.latest_selected_mask = None
        self.latest_risky_operation_prob = None
        self.latest_risky_operation_stats = None
        self.latest_output_candidate_labels = None

    @staticmethod
    def build_geometry_cue(scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean_scale = scale.mean(dim=-1, keepdim=True)
        max_scale = scale.max(dim=-1, keepdim=True).values
        min_scale = scale.min(dim=-1, keepdim=True).values
        log_volume = torch.log(scale.prod(dim=-1, keepdim=True) + eps)
        anisotropy_ratio = max_scale / (min_scale + eps)
        return torch.cat([scale, mean_scale, log_volume, anisotropy_ratio], dim=-1)

    @staticmethod
    def build_semantic_cue(semantic: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        prob = semantic.softmax(dim=-1)
        entropy = -(prob * (prob + eps).log()).sum(dim=-1, keepdim=True)
        top2_prob = prob.topk(k=2, dim=-1).values
        margin = top2_prob[..., 0:1] - top2_prob[..., 1:2]
        confidence = top2_prob[..., 0:1]
        return torch.cat([entropy, margin, confidence], dim=-1)

    def encode_allocation_state(
        self,
        instance_feature: torch.Tensor,
        scale: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        geometry_cue = self.build_geometry_cue(scale)
        semantic_cue = self.build_semantic_cue(semantic)
        z = torch.cat([instance_feature, geometry_cue, semantic_cue], dim=-1)
        return self.allocation_state_encoder(z)

    def compute_risky_topk(self, h: torch.Tensor, N: int):
        B = h.shape[0]
        device = h.device
        risky_score = torch.sigmoid(self.risky_score_head(h)).squeeze(-1)
        K = max(1, int(self.allocation_ratio * N))
        K = min(K, N)
        _, topk_idx = torch.topk(risky_score, k=K, dim=1)
        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        selected_mask.scatter_(1, topk_idx, True)
        return risky_score, selected_mask, topk_idx

    def compute_risky_operation_prob(self, h: torch.Tensor) -> torch.Tensor:
        temperature = max(float(self.router_temperature), 1e-6)
        return (self.risky_gate(h) / temperature).softmax(dim=-1)

    def _with_effective_opacity(
        self,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
        weight: torch.Tensor,
    ):
        if self.opacity_floor > 0:
            weight = weight.clamp_min(self.opacity_floor)
        eff_opacity = (gaussian.opacities * weight).clamp(1e-8, 1.0 - 1e-8)
        eff_anchor = anchor.clone()
        eff_anchor[:, 10:11] = _opacity_to_anchor_logit(eff_opacity)
        eff_gaussian = GaussianPrediction(
            means=gaussian.means,
            scales=gaussian.scales,
            rotations=gaussian.rotations,
            opacities=eff_opacity,
            semantics=gaussian.semantics,
        )
        return eff_anchor, eff_gaussian

    def _clone_branch(
        self,
        clone_feats: torch.Tensor,
        clone_h: torch.Tensor,
        clone_gaussian: GaussianPrediction,
        clone_anchor: torch.Tensor,
    ):
        clone_dir = F.normalize(self.clone_dir_head(clone_h), dim=-1)
        mean_scale = clone_gaussian.scales.mean(dim=-1, keepdim=True)
        clone_dist = torch.sigmoid(self.clone_dist_head(clone_h)) * mean_scale
        clone_scale_ratio = 0.7 + 0.3 * torch.sigmoid(self.clone_scale_head(clone_h))

        new_means = clone_gaussian.means + clone_dist * clone_dir
        m = self.boundary_margin
        pc = self.pc_range
        new_means = torch.stack([
            torch.clamp(new_means[:, 0], pc[0] + m, pc[3] - m),
            torch.clamp(new_means[:, 1], pc[1] + m, pc[4] - m),
            torch.clamp(new_means[:, 2], pc[2] + m, pc[5] - m),
        ], dim=-1)

        new_scales = torch.clamp(
            clone_gaussian.scales * clone_scale_ratio,
            self.scale_range[0],
            self.scale_range[1],
        )
        clone_opacity_ratio = torch.sigmoid(self.clone_opacity_head(clone_h))
        new_opacities = clone_gaussian.opacities * clone_opacity_ratio
        semantic_delta = self.clone_semantic_head(clone_h)
        new_semantics = clone_gaussian.semantics + semantic_delta
        c_proj = clone_feats + self.clone_feature_head(clone_h)

        c_gaussian = GaussianPrediction(
            means=new_means,
            scales=new_scales,
            rotations=clone_gaussian.rotations,
            opacities=new_opacities,
            semantics=new_semantics,
        )
        new_anchor = self._build_anchor(
            c_gaussian,
            semantic_anchor=clone_anchor[:, 11:] + semantic_delta,
        )
        return c_proj, new_anchor, c_gaussian

    def _split_branch(
        self,
        split_feats: torch.Tensor,
        split_h: torch.Tensor,
        split_gaussian: GaussianPrediction,
        split_anchor: torch.Tensor,
    ):
        parent_means = split_gaussian.means
        parent_scales = split_gaussian.scales
        parent_rots = split_gaussian.rotations
        parent_sems = split_gaussian.semantics
        parent_opas = split_gaussian.opacities

        axis_dir = get_max_scale_axis(parent_rots.unsqueeze(0), parent_scales.unsqueeze(0))
        axis_dir = axis_dir.squeeze(0)
        learned_dir = F.normalize(self.split_dir_head(split_h), dim=-1)
        split_dir = F.normalize(axis_dir + learned_dir, dim=-1)

        mean_scale = parent_scales.mean(dim=-1, keepdim=True)
        split_dist = torch.sigmoid(self.split_dist_head(split_h)) * mean_scale
        split_scale_ratio = 0.45 + 0.30 * torch.sigmoid(self.split_scale_head(split_h))

        offset = split_dist * split_dir
        c1_means = parent_means + offset
        c2_means = parent_means - offset

        m = self.boundary_margin
        pc = self.pc_range
        c1_means = torch.stack([
            torch.clamp(c1_means[:, 0], pc[0] + m, pc[3] - m),
            torch.clamp(c1_means[:, 1], pc[1] + m, pc[4] - m),
            torch.clamp(c1_means[:, 2], pc[2] + m, pc[5] - m),
        ], dim=-1)
        c2_means = torch.stack([
            torch.clamp(c2_means[:, 0], pc[0] + m, pc[3] - m),
            torch.clamp(c2_means[:, 1], pc[1] + m, pc[4] - m),
            torch.clamp(c2_means[:, 2], pc[2] + m, pc[5] - m),
        ], dim=-1)

        c1_scales = torch.clamp(
            parent_scales * split_scale_ratio,
            self.scale_range[0],
            self.scale_range[1],
        )
        c2_scales = torch.clamp(
            parent_scales * split_scale_ratio,
            self.scale_range[0],
            self.scale_range[1],
        )

        split_opacity = self.split_opacity_head(split_h)
        c1_opa = parent_opas * torch.sigmoid(split_opacity[:, 0:1])
        c2_opa = parent_opas * torch.sigmoid(split_opacity[:, 1:2])

        split_sem = self.split_semantic_head(split_h)
        sem_c = self.semantic_dim
        c1_sems = parent_sems + split_sem[:, :sem_c]
        c2_sems = parent_sems + split_sem[:, sem_c:]

        split_feat = self.split_feature_head(split_h)
        feat_c = self.feat_embed_dim
        c1_feat = split_feats + split_feat[:, :feat_c]
        c2_feat = split_feats + split_feat[:, feat_c:]

        c1_gaussian = GaussianPrediction(
            means=c1_means,
            scales=c1_scales,
            rotations=parent_rots,
            opacities=c1_opa,
            semantics=c1_sems,
        )
        c2_gaussian = GaussianPrediction(
            means=c2_means,
            scales=c2_scales,
            rotations=parent_rots,
            opacities=c2_opa,
            semantics=c2_sems,
        )

        c1_anchor = self._build_anchor(
            c1_gaussian,
            semantic_anchor=split_anchor[:, 11:] + split_sem[:, :sem_c],
        )
        c2_anchor = self._build_anchor(
            c2_gaussian,
            semantic_anchor=split_anchor[:, 11:] + split_sem[:, sem_c:],
        )
        return c1_feat, c1_anchor, c1_gaussian, c2_feat, c2_anchor, c2_gaussian

    def _atten_branch(
        self,
        atten_h: torch.Tensor,
        atten_gaussian: GaussianPrediction,
        atten_anchor: torch.Tensor,
    ):
        m = self.m_min + (1 - self.m_min) * torch.sigmoid(self.atten_opacity_head(atten_h))
        a_opacity = atten_gaussian.opacities * m
        a_gaussian = GaussianPrediction(
            means=atten_gaussian.means,
            scales=atten_gaussian.scales,
            rotations=atten_gaussian.rotations,
            opacities=a_opacity,
            semantics=atten_gaussian.semantics,
        )
        a_anchor = atten_anchor.clone()
        a_anchor[:, 10:11] = _opacity_to_anchor_logit(a_opacity)
        return a_anchor, a_gaussian

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
    ):
        sr0, sr1 = self.scale_range[0], self.scale_range[1]
        eps = 1e-4
        pc = self.pc_range
        xyz_unit = torch.stack([
            (gaussian.means[..., 0] - pc[0]) / (pc[3] - pc[0]),
            (gaussian.means[..., 1] - pc[1]) / (pc[4] - pc[1]),
            (gaussian.means[..., 2] - pc[2]) / (pc[5] - pc[2]),
        ], dim=-1)
        xyz_unit = xyz_unit.clamp(eps, 1.0 - eps)
        scale = gaussian.scales.clamp(sr0 + eps, sr1 - eps)
        anchor_xyz = safe_inverse_sigmoid(xyz_unit)
        anchor_scale = torch.log((scale - sr0) / (sr1 - scale + eps) + eps)
        anchor_opa = _opacity_to_anchor_logit(gaussian.opacities)
        if semantic_anchor is None:
            semantic_anchor = gaussian.semantics
        anchor_sem = self._sanitize_anchor_tail(semantic_anchor)
        return torch.cat([
            anchor_xyz,
            anchor_scale,
            gaussian.rotations,
            anchor_opa,
            anchor_sem,
        ], dim=-1)

    def _sync_anchor_with_gaussian(
        self,
        anchor: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ) -> torch.Tensor:
        """Keep anchor geometry aligned with the sanitized Gaussian state."""
        sr0, sr1 = self.scale_range[0], self.scale_range[1]
        eps = 1e-4
        pc = self.pc_range

        xyz_unit = torch.stack([
            (means[..., 0] - pc[0]) / (pc[3] - pc[0]),
            (means[..., 1] - pc[1]) / (pc[4] - pc[1]),
            (means[..., 2] - pc[2]) / (pc[5] - pc[2]),
        ], dim=-1)
        xyz_unit = xyz_unit.clamp(eps, 1.0 - eps)
        scale = scales.clamp(sr0 + eps, sr1 - eps)

        synced = anchor.clone()
        synced[..., 0:3] = safe_inverse_sigmoid(xyz_unit)
        synced[..., 3:6] = torch.log((scale - sr0) / (sr1 - scale + eps) + eps)
        synced[..., 6:10] = rotations
        synced[..., 10:11] = _opacity_to_anchor_logit(opacities)
        if synced.shape[-1] > 11:
            synced[..., 11:] = self._sanitize_anchor_tail(synced[..., 11:])
        return synced

    def _sanitize_gaussian_tensors(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ):
        sr0, sr1 = self.scale_range[0], self.scale_range[1]
        pc = self.pc_range
        m = self.boundary_margin

        means = torch.where(torch.isfinite(means), means, torch.zeros_like(means))
        means = torch.stack([
            torch.clamp(means[..., 0], pc[0] + m, pc[3] - m),
            torch.clamp(means[..., 1], pc[1] + m, pc[4] - m),
            torch.clamp(means[..., 2], pc[2] + m, pc[5] - m),
        ], dim=-1)

        scales = torch.where(
            torch.isfinite(scales),
            scales,
            torch.full_like(scales, sr0),
        )
        scales = scales.clamp(min=sr0, max=sr1)

        rotations = torch.where(torch.isfinite(rotations), rotations, torch.zeros_like(rotations))
        rot_norm = rotations.norm(dim=-1, keepdim=True)
        rotations = rotations / rot_norm.clamp_min(1e-8)
        bad_rot = (rot_norm < 1e-8) | (~torch.isfinite(rotations).all(dim=-1, keepdim=True))
        if bad_rot.any():
            identity = torch.zeros_like(rotations)
            identity[..., 0] = 1.0
            rotations = torch.where(bad_rot.expand_as(rotations), identity, rotations)

        opacities = torch.where(
            torch.isfinite(opacities),
            opacities,
            torch.zeros_like(opacities),
        )
        opacities = opacities.clamp(1e-8, 1.0 - 1e-8)
        return means, scales, rotations, opacities

    @staticmethod
    def _sanitize_semantics(semantics: torch.Tensor) -> torch.Tensor:
        semantics = torch.where(
            torch.isfinite(semantics),
            semantics,
            torch.zeros_like(semantics),
        )
        return semantics.clamp(min=-30.0, max=30.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        opacities = gaussian.opacities
        means = gaussian.means
        scales = gaussian.scales
        rotations = gaussian.rotations
        semantics = gaussian.semantics
        means, scales, rotations, opacities = self._sanitize_gaussian_tensors(
            means,
            scales,
            rotations,
            opacities,
        )
        semantics = self._sanitize_semantics(semantics)

        B, N, _ = instance_feature.shape
        h = self.encode_allocation_state(instance_feature, scales, semantics)
        risky_score, selected_mask, topk_idx = self.compute_risky_topk(h, N)
        risky_prob_all = self.compute_risky_operation_prob(h)

        result_features_list = []
        result_anchors_list = []
        result_opacities_list = []
        result_semantics_list = []
        result_rotations_list = []
        result_means_list = []
        result_scales_list = []
        output_labels_list = []
        stats = []

        for b in range(B):
            selected_idx = topk_idx[b]
            non_selected_mask = ~selected_mask[b]
            non_selected_idx = torch.where(non_selected_mask)[0]

            cur_feats = instance_feature[b]
            cur_anchor = anchor[b]
            cur_h = h[b]
            cur_prob = risky_prob_all[b]
            cur_means = means[b]
            cur_scales = scales[b]
            cur_rots = rotations[b]
            cur_opa = opacities[b]
            cur_sems = semantics[b]

            pass_feats = cur_feats[non_selected_idx]
            pass_anchor = cur_anchor[non_selected_idx]
            pass_means = cur_means[non_selected_idx]
            pass_scales = cur_scales[non_selected_idx]
            pass_rots = cur_rots[non_selected_idx]
            pass_opa = cur_opa[non_selected_idx]
            pass_sems = cur_sems[non_selected_idx]

            s_feats = cur_feats[selected_idx]
            s_anchor = cur_anchor[selected_idx]
            s_h = cur_h[selected_idx]
            s_prob = cur_prob[selected_idx]
            p_clone = s_prob[:, 0:1]
            p_split = s_prob[:, 1:2]
            p_atten = s_prob[:, 2:3]
            s_g = GaussianPrediction(
                means=cur_means[selected_idx],
                scales=cur_scales[selected_idx],
                rotations=cur_rots[selected_idx],
                opacities=cur_opa[selected_idx],
                semantics=cur_sems[selected_idx],
            )

            clone_child_feat, clone_child_anchor, clone_child_g = self._clone_branch(
                s_feats, s_h, s_g, s_anchor)
            split_c1_feat, split_c1_anchor, split_c1_g, split_c2_feat, split_c2_anchor, split_c2_g = \
                self._split_branch(s_feats, s_h, s_g, s_anchor)
            atten_anchor, atten_g = self._atten_branch(s_h, s_g, s_anchor)

            clone_parent_anchor, clone_parent_g = self._with_effective_opacity(
                s_anchor, s_g, p_clone)
            clone_child_anchor, clone_child_g = self._with_effective_opacity(
                clone_child_anchor, clone_child_g, p_clone)
            split_c1_anchor, split_c1_g = self._with_effective_opacity(
                split_c1_anchor, split_c1_g, p_split)
            split_c2_anchor, split_c2_g = self._with_effective_opacity(
                split_c2_anchor, split_c2_g, p_split)
            atten_anchor, atten_g = self._with_effective_opacity(
                atten_anchor, atten_g, p_atten)


            result_features_b = torch.cat([
                pass_feats,
                s_feats,
                clone_child_feat,
                split_c1_feat,
                split_c2_feat,
                s_feats,
            ], dim=0)
            result_anchor_b = torch.cat([
                pass_anchor,
                clone_parent_anchor,
                clone_child_anchor,
                split_c1_anchor,
                split_c2_anchor,
                atten_anchor,
            ], dim=0)
            result_opa_b = torch.cat([
                pass_opa,
                clone_parent_g.opacities,
                clone_child_g.opacities,
                split_c1_g.opacities,
                split_c2_g.opacities,
                atten_g.opacities,
            ], dim=0)
            result_sems_b = torch.cat([
                pass_sems,
                clone_parent_g.semantics,
                clone_child_g.semantics,
                split_c1_g.semantics,
                split_c2_g.semantics,
                atten_g.semantics,
            ], dim=0)
            result_sems_b = self._sanitize_semantics(result_sems_b)
            result_rots_b = torch.cat([
                pass_rots,
                clone_parent_g.rotations,
                clone_child_g.rotations,
                split_c1_g.rotations,
                split_c2_g.rotations,
                atten_g.rotations,
            ], dim=0)
            result_means_b = torch.cat([
                pass_means,
                clone_parent_g.means,
                clone_child_g.means,
                split_c1_g.means,
                split_c2_g.means,
                atten_g.means,
            ], dim=0)
            result_scales_b = torch.cat([
                pass_scales,
                clone_parent_g.scales,
                clone_child_g.scales,
                split_c1_g.scales,
                split_c2_g.scales,
                atten_g.scales,
            ], dim=0)
            result_means_b, result_scales_b, result_rots_b, result_opa_b = \
                self._sanitize_gaussian_tensors(
                    result_means_b,
                    result_scales_b,
                    result_rots_b,
                    result_opa_b,
                )
            result_anchor_b = self._sync_anchor_with_gaussian(
                result_anchor_b,
                result_means_b,
                result_scales_b,
                result_rots_b,
                result_opa_b,
            )

            labels_b = torch.cat([
                torch.full((non_selected_idx.numel(),), PASS_THROUGH, device=cur_feats.device, dtype=torch.long),
                torch.full((selected_idx.numel(),), CLONE_PARENT, device=cur_feats.device, dtype=torch.long),
                torch.full((selected_idx.numel(),), CLONE_CHILD, device=cur_feats.device, dtype=torch.long),
                torch.full((selected_idx.numel(),), SPLIT_CHILD, device=cur_feats.device, dtype=torch.long),
                torch.full((selected_idx.numel(),), SPLIT_CHILD, device=cur_feats.device, dtype=torch.long),
                torch.full((selected_idx.numel(),), ATTEN, device=cur_feats.device, dtype=torch.long),
            ], dim=0)

            result_features_list.append(result_features_b)
            result_anchors_list.append(result_anchor_b)
            result_opacities_list.append(result_opa_b)
            result_semantics_list.append(result_sems_b)
            result_rotations_list.append(result_rots_b)
            result_means_list.append(result_means_b)
            result_scales_list.append(result_scales_b)
            output_labels_list.append(labels_b)

            if self.cache_router_stats:
                stats.append({
                    'input_gaussians': int(N),
                    'selected_topk': int(selected_idx.numel()),
                    'non_topk_pass_through': int(non_selected_idx.numel()),
                    'expected_clone': float(p_clone.detach().sum().cpu()),
                    'expected_split': float(p_split.detach().sum().cpu()),
                    'expected_atten': float(p_atten.detach().sum().cpu()),
                    'topk_p_clone_mean': float(p_clone.detach().mean().cpu()),
                    'topk_p_split_mean': float(p_split.detach().mean().cpu()),
                    'topk_p_atten_mean': float(p_atten.detach().mean().cpu()),
                    'effective_opacity_clone_parent': float(clone_parent_g.opacities.detach().sum().cpu()),
                    'effective_opacity_clone_child': float(clone_child_g.opacities.detach().sum().cpu()),
                    'effective_opacity_split_child_1': float(split_c1_g.opacities.detach().sum().cpu()),
                    'effective_opacity_split_child_2': float(split_c2_g.opacities.detach().sum().cpu()),
                    'effective_opacity_atten': float(atten_g.opacities.detach().sum().cpu()),
                    'output_candidate_count': int(result_features_b.shape[0]),
                })

        result_features = torch.stack(result_features_list, dim=0)
        result_anchors = torch.stack(result_anchors_list, dim=0)
        result_opacities = torch.stack(result_opacities_list, dim=0)
        result_semantics = torch.stack(result_semantics_list, dim=0)
        result_rotations = torch.stack(result_rotations_list, dim=0)
        result_means = torch.stack(result_means_list, dim=0)
        result_scales = torch.stack(result_scales_list, dim=0)
        output_labels = torch.stack(output_labels_list, dim=0)

        if self.cache_router_stats:
            self.latest_selected_mask = selected_mask.detach()
            self.latest_risky_operation_prob = risky_prob_all.detach()
            self.latest_output_candidate_labels = output_labels.detach()
            self.latest_risky_operation_stats = stats

        result_gaussian = GaussianPrediction(
            means=result_means,
            scales=result_scales,
            rotations=result_rotations,
            opacities=result_opacities,
            semantics=result_semantics,
        )
        return result_anchors, result_gaussian, result_features
