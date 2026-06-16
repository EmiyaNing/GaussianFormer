"""
AdaptiveAllocationV3 module: Budget-constrained Hard Routing Adaptive Gaussian Allocation.

Based on allocationv3.md design:
1. Multi-cue allocation state encoding (geometry + semantic + feature)
2. Learned allocation score + budget-constrained TopK selection
3. 4-class hard operation routing (keep/clone/split/opacity-modulate)
4. 5N candidate bank with active mask for GPU-friendly computation

Core design principles:
- Budget-constrained TopK selection via allocation_ratio
- Hard routing consistency: train & inference both use hard TopK + hard argmax
- Operation-specific allocation: clone for coverage, split for purity
- 5N candidate bank with active mask ensures fixed-shape GPU computation

Output interface is consistent with AdaptiveAllocation (3-tuple) for encoder reuse.
"""

from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import GaussianPrediction
from ....utils.safe_ops import safe_sigmoid


# ---------------------------------------------------------------------------
#  Utility: get the world-space direction of Gaussian's maximum scale axis
# ---------------------------------------------------------------------------

def get_max_scale_axis(rotation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Get the world-space direction of the Gaussian's maximum scale axis.

    The Gaussian's covariance is R @ diag(s^2) @ R^T.
    The axis of maximum scale in world space is the column of the rotation
    matrix corresponding to the largest scale dimension.

    Args:
        rotation: [B, N, 4] quaternion [w, x, y, z]
        scale: [B, N, 3] scale factors (sx, sy, sz)

    Returns:
        axis_dir: [B, N, 3] normalized world-space direction vector
    """
    # Find the dimension with maximum scale per gaussian
    max_scale_dim = scale.argmax(dim=-1, keepdim=True)  # [B, N, 1]

    # Quaternion components
    w, x, y, z = rotation.unbind(dim=-1)  # each [B, N]

    # Rotation matrix columns (R converts local → world)
    # col_x = R @ [1, 0, 0]^T
    col_x = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], dim=-1)  # [B, N, 3]

    # col_y = R @ [0, 1, 0]^T
    col_y = torch.stack([
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
    ], dim=-1)  # [B, N, 3]

    # col_z = R @ [0, 0, 1]^T
    col_z = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ], dim=-1)  # [B, N, 3]

    # Stack columns and gather the one corresponding to max scale
    all_cols = torch.stack([col_x, col_y, col_z], dim=-2)  # [B, N, 3, 3]
    idx = max_scale_dim.unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, N, 1, 3]
    axis_dir = all_cols.gather(dim=-2, index=idx).squeeze(-2)  # [B, N, 3]

    return F.normalize(axis_dir, dim=-1)


# ---------------------------------------------------------------------------
#  Constants for operation routing
# ---------------------------------------------------------------------------

KEEP = 0
CLONE = 1
SPLIT = 2
ATTEN = 3  # opacity-modulate / attenuation


# ===========================================================================
#  Main Module
# ===========================================================================

@MODELS.register_module()
class AdaptiveAllocationV4(BaseModule):
    """
    Budget-constrained Hard Routing Adaptive Gaussian Allocation.

    This module:
    1. Encodes multi-cue states (geometry, semantic, feature) for each Gaussian.
    2. Predicts an allocation score and selects TopK via budget-constrained selection.
    3. Routes selected Gaussians to one of 4 operations: keep, clone, split, opacity-modulate.
    4. Generates a 5N candidate bank with active mask.
    5. Returns assembled (anchor, GaussianPrediction, features) consistent with
       the AdaptiveAllocation interface.

    Args:
        feat_embed_dim: Feature embedding dimension (default 128)
        semantic_dim: Number of semantic classes (default 17)
        pc_range: Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]
        scale_range: Scale clamping range [min, max]
        unit_xyz: Unit position shift for clone branch location refinement
        hidden_dim: Hidden dimension of the state encoder (default 256)
        allocation_ratio: Ratio of Gaussians to select for adaptive allocation (default 0.4)
        m_min: Minimum opacity modulation factor (default 0.05)
        use_straight_through: Whether to use straight-through gradient for operation (default True)
        **kwargs: Additional keyword arguments
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
        m_min=0.05,
        use_straight_through=True,
        **kwargs,
    ):
        super(AdaptiveAllocationV4, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.semantic_dim = semantic_dim
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.allocation_ratio = allocation_ratio
        self.m_min = m_min
        self.use_straight_through = use_straight_through

        # Safe margin from pc_range boundary to avoid float32 precision issues
        self.boundary_margin = 0.1

        # Pre-compute unit sigmoid factors for clone branch location shift
        if unit_xyz is not None and pc_range is not None:
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            unit_prob = [4 * unit_prob[i] for i in range(3)]
            self.unit_sigmoid = unit_prob
        else:
            self.unit_sigmoid = [1.0, 1.0, 1.0]

        # ------------------------------------------------------------------
        # Geometry cue dimension: scale(3) + mean_scale(1) + log_volume(1) + anisotropy_ratio(1) = 6
        # Semantic cue dimension: entropy(1) + margin(1) + confidence(1) = 3
        # ------------------------------------------------------------------
        geo_cue_dim = 6
        sem_cue_dim = 3
        state_input_dim = feat_embed_dim + geo_cue_dim + sem_cue_dim

        # ================================================================
        #  T1: Multi-cue Allocation State Encoder
        # ================================================================
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ================================================================
        #  T2: Allocation Score Predictor
        # ================================================================
        self.score_head = nn.Linear(hidden_dim, 1)

        # ================================================================
        #  T3: Operation Router (4-class: keep/clone/split/atten)
        # ================================================================
        self.operation_head = nn.Linear(hidden_dim, 4)

        # ================================================================
        #  T4: Clone Branch Heads (allocationv3.md §10)
        # ================================================================
        self.clone_dir_head = nn.Linear(hidden_dim, 3)
        self.clone_dist_head = nn.Linear(hidden_dim, 1)
        self.clone_scale_head = nn.Linear(hidden_dim, 1)
        self.clone_opacity_head = nn.Linear(hidden_dim, 1)
        self.clone_feature_head = nn.Linear(hidden_dim, feat_embed_dim)
        self.clone_semantic_head = nn.Linear(hidden_dim, semantic_dim)

        # ================================================================
        #  T5: Split Branch Heads (allocationv3.md §11)
        # ================================================================
        self.split_dir_head = nn.Linear(hidden_dim, 3)
        self.split_dist_head = nn.Linear(hidden_dim, 1)
        self.split_scale_head = nn.Linear(hidden_dim, 1)
        self.split_opacity_head = nn.Linear(hidden_dim, 2)  # 2 children
        self.split_feature_head = nn.Linear(hidden_dim, 2 * feat_embed_dim)
        self.split_semantic_head = nn.Linear(hidden_dim, 2 * semantic_dim)

        # ================================================================
        #  T6: Opacity-modulate Head (allocationv3.md §12)
        # ================================================================
        self.opacity_mod_head = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------
    #  Helper: Build Geometry Cues
    # ------------------------------------------------------------------

    @staticmethod
    def build_geometry_cue(scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Build geometry cues from Gaussian scales.

        Args:
            scale: [B, N, 3] Gaussian scale factors
            eps: Small constant for numerical stability

        Returns:
            geometry_cue: [B, N, 6] concatenated geometry features:
                [scale(3), mean_scale(1), log_volume(1), anisotropy_ratio(1)]
        """
        mean_scale = scale.mean(dim=-1, keepdim=True)                 # [B, N, 1]
        max_scale = scale.max(dim=-1, keepdim=True).values            # [B, N, 1]
        min_scale = scale.min(dim=-1, keepdim=True).values            # [B, N, 1]
        log_volume = torch.log(scale.prod(dim=-1, keepdim=True) + eps)  # [B, N, 1]
        anisotropy_ratio = max_scale / (min_scale + eps)              # [B, N, 1]

        geometry_cue = torch.cat([
            scale,          # [B, N, 3]
            mean_scale,     # [B, N, 1]
            log_volume,     # [B, N, 1]
            anisotropy_ratio,  # [B, N, 1]
        ], dim=-1)  # [B, N, 6]

        return geometry_cue

    # ------------------------------------------------------------------
    #  Helper: Build Semantic Cues
    # ------------------------------------------------------------------

    @staticmethod
    def build_semantic_cue(semantic: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Build semantic cues from Gaussian semantic logits.

        Args:
            semantic: [B, N, C] semantic logits
            eps: Small constant for numerical stability

        Returns:
            semantic_cue: [B, N, 3] concatenated semantic features:
                [entropy(1), margin(1), confidence(1)]
        """
        prob = semantic.softmax(dim=-1)                                  # [B, N, C]
        entropy = -(prob * (prob + eps).log()).sum(dim=-1, keepdim=True)  # [B, N, 1]
        top2_prob = prob.topk(k=2, dim=-1).values                         # [B, N, 2]
        margin = top2_prob[..., 0:1] - top2_prob[..., 1:2]               # [B, N, 1]
        confidence = top2_prob[..., 0:1]                                  # [B, N, 1]

        semantic_cue = torch.cat([
            entropy,
            margin,
            confidence,
        ], dim=-1)  # [B, N, 3]

        return semantic_cue

    # ------------------------------------------------------------------
    #  Helper: Encode Multi-cue Allocation State
    # ------------------------------------------------------------------

    def encode_allocation_state(
        self,
        instance_feature: torch.Tensor,
        scale: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode multi-cue allocation state for each Gaussian.

        Args:
            instance_feature: [B, N, E] instance features
            scale: [B, N, 3] Gaussian scales
            semantic: [B, N, C] semantic logits

        Returns:
            h: [B, N, hidden_dim] encoded allocation state
        """
        geometry_cue = self.build_geometry_cue(scale)       # [B, N, 6]
        semantic_cue = self.build_semantic_cue(semantic)    # [B, N, 3]

        z = torch.cat([instance_feature, geometry_cue, semantic_cue], dim=-1)
        h = self.state_encoder(z)  # [B, N, hidden_dim]

        return h

    # ------------------------------------------------------------------
    #  Helper: Compute Allocation Score and TopK Selection
    # ------------------------------------------------------------------

    def compute_allocation_topk(
        self,
        h: torch.Tensor,
        N: int,
    ):
        """
        Compute allocation scores and perform budget-constrained TopK selection.

        Args:
            h: [B, N, hidden_dim] encoded allocation state
            N: Number of Gaussians

        Returns:
            allocation_score: [B, N] sigmoid scores
            selected_mask: [B, N] boolean mask of selected Gaussians
            topk_idx: [B, K] indices of selected Gaussians
        """
        B = h.shape[0]
        device = h.device

        allocation_score = torch.sigmoid(self.score_head(h)).squeeze(-1)  # [B, N]
        K = max(1, int(self.allocation_ratio * N))
        K = min(K, N)

        _, topk_idx = torch.topk(allocation_score, k=K, dim=1)  # [B, K]
        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        selected_mask.scatter_(1, topk_idx, True)

        return allocation_score, selected_mask, topk_idx

    # ------------------------------------------------------------------
    #  Helper: Compute Operation Routing
    # ------------------------------------------------------------------

    def compute_operation_routing(
        self,
        h: torch.Tensor,
        selected_mask: torch.Tensor,
    ):
        """
        Compute 4-class operation routing with straight-through gradient.

        Args:
            h: [B, N, hidden_dim] encoded allocation state
            selected_mask: [B, N] boolean mask of selected Gaussians

        Returns:
            op_logits: [B, N, 4] operation logits
            op_prob: [B, N, 4] soft operation probabilities
            op_onehot: [B, N, 4] straight-through one-hot mask
            op_id: [B, N] hard operation IDs {0:keep, 1:clone, 2:split, 3:atten}
        """
        op_logits = self.operation_head(h)           # [B, N, 4]
        op_prob = op_logits.softmax(dim=-1)          # [B, N, 4]

        # Hard argmax for operation ID
        op_id = op_prob.argmax(dim=-1)               # [B, N]

        # Non-selected Gaussians are forced to KEEP
        op_id = torch.where(selected_mask, op_id, torch.full_like(op_id, KEEP))

        if self.training and self.use_straight_through:
            # Straight-through: forward = hard one-hot, backward = soft prob gradients
            op_onehot_hard = F.one_hot(op_id, num_classes=4).float()
            op_onehot = op_onehot_hard.detach() - op_prob.detach() + op_prob
        else:
            op_onehot = F.one_hot(op_id, num_classes=4).float()

        return op_logits, op_prob, op_onehot, op_id

    # ------------------------------------------------------------------
    #  Helper: Clone Branch (allocationv3.md §10)
    # ------------------------------------------------------------------

    def _clone_branch(
        self,
        clone_feats: torch.Tensor,
        clone_h: torch.Tensor,
        clone_gaussian: GaussianPrediction,
        clone_anchor: torch.Tensor,
    ):
        """
        Process clone-selected Gaussians. Original kept + 1 child appended.

        Args:
            clone_feats:    (N_c, E) instance features of clone gaussians
            clone_h:        (N_c, hidden_dim) allocation states
            clone_gaussian: GaussianPrediction fields
            clone_anchor:   (N_c, A) anchors

        Returns:
            c_proj:    (N_c, E) projected features for new gaussians
            c_anchor:  (N_c, A) anchor for new gaussians
            c_gaussian: GaussianPrediction for new gaussians
        """
        # ---- Clone direction and distance (allocationv3.md §10.3) ----
        clone_dir = F.normalize(self.clone_dir_head(clone_h), dim=-1)         # (N_c, 3)
        mean_scale = clone_gaussian.scales.mean(dim=-1, keepdim=True)          # (N_c, 1)
        clone_dist = torch.sigmoid(self.clone_dist_head(clone_h)) * mean_scale  # (N_c, 1)
        clone_scale_ratio = 0.7 + 0.3 * torch.sigmoid(self.clone_scale_head(clone_h))  # (N_c, 1)

        # ---- Child position (allocationv3.md §10.4) ----
        new_means = clone_gaussian.means + clone_dist * clone_dir

        # Boundary clamping
        m = self.boundary_margin
        pc = self.pc_range
        new_means_x = torch.clamp(new_means[:, 0], pc[0] + m, pc[3] - m)
        new_means_y = torch.clamp(new_means[:, 1], pc[1] + m, pc[4] - m)
        new_means_z = torch.clamp(new_means[:, 2], pc[2] + m, pc[5] - m)
        new_means = torch.stack([new_means_x, new_means_y, new_means_z], dim=-1)

        # ---- Child scales (shrunk) ----
        new_scales = clone_gaussian.scales * clone_scale_ratio
        new_scales = torch.clamp(new_scales, self.scale_range[0], self.scale_range[1])

        # ---- Child rotations (inherit) ----
        new_rotations = clone_gaussian.rotations

        # ---- Child opacity ----
        clone_opacity_ratio = torch.sigmoid(self.clone_opacity_head(clone_h))  # (N_c, 1)
        new_opacities = clone_gaussian.opacities * clone_opacity_ratio

        # ---- Child semantic / feature (residual adjustment) ----
        new_semantics = clone_gaussian.semantics + self.clone_semantic_head(clone_h)
        c_proj = clone_feats + self.clone_feature_head(clone_h)

        # ---- Anchor update ----
        new_xyz_anchor = safe_sigmoid(new_means)
        new_anchor_scale = clone_anchor[:, 3:6] * clone_scale_ratio
        new_anchor_rot = clone_anchor[:, 6:10]
        new_anchor_opa = clone_anchor[:, 10:11] * clone_opacity_ratio
        new_anchor_sem = clone_anchor[:, 11:] + self.clone_semantic_head(clone_h)

        c_anchor = torch.cat([
            new_xyz_anchor, new_anchor_scale, new_anchor_rot,
            new_anchor_opa, new_anchor_sem,
        ], dim=-1)

        c_gaussian = GaussianPrediction(
            means=new_means,
            scales=new_scales,
            rotations=new_rotations,
            opacities=new_opacities,
            semantics=new_semantics,
        )

        return c_proj, c_anchor, c_gaussian

    # ------------------------------------------------------------------
    #  Helper: Split Branch (allocationv3.md §11)
    # ------------------------------------------------------------------

    def _split_branch(
        self,
        split_feats: torch.Tensor,
        split_h: torch.Tensor,
        split_gaussian: GaussianPrediction,
        split_anchor: torch.Tensor,
    ):
        """
        Process split-selected Gaussians. Original replaced by 2 children.

        Args:
            split_feats:    (N_s, E) instance features
            split_h:        (N_s, hidden_dim) allocation states
            split_gaussian: GaussianPrediction fields
            split_anchor:   (N_s, A) anchors

        Returns:
            c1_feat, c1_anchor, c1_gaussian: child 1
            c2_feat, c2_anchor, c2_gaussian: child 2
        """
        parent_means = split_gaussian.means
        parent_scales = split_gaussian.scales
        parent_rots = split_gaussian.rotations
        parent_sems = split_gaussian.semantics
        parent_opas = split_gaussian.opacities

        # ---- Split direction (allocationv3.md §11.3) ----
        axis_dir = get_max_scale_axis(parent_rots.unsqueeze(0), parent_scales.unsqueeze(0))
        axis_dir = axis_dir.squeeze(0)  # (N_s, 3)
        learned_dir = F.normalize(self.split_dir_head(split_h), dim=-1)  # (N_s, 3)
        split_dir = F.normalize(axis_dir + learned_dir, dim=-1)          # (N_s, 3)

        # ---- Split distance and scale ratio (allocationv3.md §11.4) ----
        mean_scale = parent_scales.mean(dim=-1, keepdim=True)              # (N_s, 1)
        split_dist = torch.sigmoid(self.split_dist_head(split_h)) * mean_scale  # (N_s, 1)
        split_scale_ratio = 0.45 + 0.30 * torch.sigmoid(self.split_scale_head(split_h))  # (N_s, 1)

        # ---- Child positions (allocationv3.md §11.5) ----
        offset = split_dist * split_dir  # (N_s, 3)
        c1_means = parent_means + offset
        c2_means = parent_means - offset

        # Boundary clamping (non-inplace to preserve gradients)
        m = self.boundary_margin
        pc = self.pc_range
        c1_means_x = torch.clamp(c1_means[:, 0], pc[0] + m, pc[3] - m)
        c1_means_y = torch.clamp(c1_means[:, 1], pc[1] + m, pc[4] - m)
        c1_means_z = torch.clamp(c1_means[:, 2], pc[2] + m, pc[5] - m)
        c1_means = torch.stack([c1_means_x, c1_means_y, c1_means_z], dim=-1)

        c2_means_x = torch.clamp(c2_means[:, 0], pc[0] + m, pc[3] - m)
        c2_means_y = torch.clamp(c2_means[:, 1], pc[1] + m, pc[4] - m)
        c2_means_z = torch.clamp(c2_means[:, 2], pc[2] + m, pc[5] - m)
        c2_means = torch.stack([c2_means_x, c2_means_y, c2_means_z], dim=-1)

        # ---- Child scales (significantly smaller) ----
        c1_scales = parent_scales * split_scale_ratio
        c2_scales = parent_scales * split_scale_ratio
        c1_scales = torch.clamp(c1_scales, self.scale_range[0], self.scale_range[1])
        c2_scales = torch.clamp(c2_scales, self.scale_range[0], self.scale_range[1])

        # ---- Child rotations (inherit from parent) ----
        c1_rots = parent_rots
        c2_rots = parent_rots

        # ---- Child opacities ----
        split_opacity = self.split_opacity_head(split_h)  # (N_s, 2)
        c1_opa = parent_opas * torch.sigmoid(split_opacity[:, 0:1])
        c2_opa = parent_opas * torch.sigmoid(split_opacity[:, 1:2])

        # ---- Child semantics (residual) ----
        split_sem = self.split_semantic_head(split_h)  # (N_s, 2*C)
        sem_c = self.semantic_dim
        c1_sems = parent_sems + split_sem[:, :sem_c]
        c2_sems = parent_sems + split_sem[:, sem_c:]

        # ---- Child features (residual) ----
        split_feat = self.split_feature_head(split_h)  # (N_s, 2*E)
        feat_c = self.feat_embed_dim
        c1_feat = split_feats + split_feat[:, :feat_c]
        c2_feat = split_feats + split_feat[:, feat_c:]

        # ---- Build child gaussians ----
        c1_gaussian = GaussianPrediction(
            means=c1_means, scales=c1_scales,
            rotations=c1_rots, opacities=c1_opa, semantics=c1_sems,
        )
        c2_gaussian = GaussianPrediction(
            means=c2_means, scales=c2_scales,
            rotations=c2_rots, opacities=c2_opa, semantics=c2_sems,
        )

        # ---- Build child anchors (from scratch, matching adaptive_allocation.py style) ----
        sr0, sr1 = self.scale_range[0], self.scale_range[1]
        eps = 1e-8

        c1_anchor_xyz = safe_sigmoid(c1_means)
        c2_anchor_xyz = safe_sigmoid(c2_means)

        c1_anchor_scale = torch.log(
            (c1_scales - sr0) / (sr1 - c1_scales + eps) + eps)
        c2_anchor_scale = torch.log(
            (c2_scales - sr0) / (sr1 - c2_scales + eps) + eps)

        c1_anchor_opa = torch.log(c1_opa / (1 - c1_opa + eps) + eps)
        c2_anchor_opa = torch.log(c2_opa / (1 - c2_opa + eps) + eps)
        c1_anchor_sem = torch.log(c1_sems.clamp(eps, 1.0 - eps) / (1 - c1_sems.clamp(eps, 1.0 - eps) + eps) + eps)
        c2_anchor_sem = torch.log(c2_sems.clamp(eps, 1.0 - eps) / (1 - c2_sems.clamp(eps, 1.0 - eps) + eps) + eps)

        c1_anchor = torch.cat([
            c1_anchor_xyz, c1_anchor_scale, c1_rots,
            c1_anchor_opa, c1_anchor_sem,
        ], dim=-1)
        c2_anchor = torch.cat([
            c2_anchor_xyz, c2_anchor_scale, c2_rots,
            c2_anchor_opa, c2_anchor_sem,
        ], dim=-1)

        return c1_feat, c1_anchor, c1_gaussian, c2_feat, c2_anchor, c2_gaussian

    # ------------------------------------------------------------------
    #  Helper: Opacity-modulate Branch (allocationv3.md §12)
    # ------------------------------------------------------------------

    def _atten_branch(
        self,
        atten_h: torch.Tensor,
        atten_gaussian: GaussianPrediction,
        atten_anchor: torch.Tensor,
    ):
        """
        Process opacity-modulate selected Gaussians. Opacity is reduced.

        Args:
            atten_h:        (N_a, hidden_dim) allocation states
            atten_gaussian: GaussianPrediction fields
            atten_anchor:   (N_a, A) anchors

        Returns:
            a_anchor:  (N_a, A) anchor
            a_gaussian: GaussianPrediction with reduced opacity
        """
        # Modulation factor (allocationv3.md §12.2)
        m = self.m_min + (1 - self.m_min) * torch.sigmoid(self.opacity_mod_head(atten_h))  # (N_a, 1)

        # Attenuated Gaussian: only opacity changes
        a_means = atten_gaussian.means
        a_scales = atten_gaussian.scales
        a_rotations = atten_gaussian.rotations
        a_opacities = atten_gaussian.opacities * m
        a_semantics = atten_gaussian.semantics

        a_gaussian = GaussianPrediction(
            means=a_means, scales=a_scales,
            rotations=a_rotations, opacities=a_opacities,
            semantics=a_semantics,
        )

        # Anchor: only opacity changes
        a_anchor_opa = atten_anchor[:, 10:11] + torch.log(m / (1 - m + 1e-8) + 1e-8)
        a_anchor = torch.cat([
            atten_anchor[:, :10],
            a_anchor_opa,
            atten_anchor[:, 11:],
        ], dim=-1)

        # Feature: unchanged
        a_feat = None  # features are handled in forward

        return a_anchor, a_gaussian

    # ------------------------------------------------------------------
    #  Main Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Forward pass of AdaptiveAllocationV3.

        Args:
            instance_feature: (B, N, E) instance features
            anchor:           (B, N, A) anchor encodings
            gaussian:         GaussianPrediction with fields (B, N, *)

        Returns:
            result_anchors:   (B, M, A)  where M = N + N_clone + N_split
            result_gaussian:  GaussianPrediction (B, M, *)
            result_features:  (B, M, E)
        """
        opacities = gaussian.opacities  # B, N, 1
        means = gaussian.means          # B, N, 3
        scales = gaussian.scales        # B, N, 3
        rotations = gaussian.rotations  # B, N, 4
        semantics = gaussian.semantics  # B, N, C

        B, N, E = instance_feature.shape
        device = instance_feature.device

        # ================================================================
        #  Step 1: Multi-cue Allocation State Encoding
        # ================================================================
        h = self.encode_allocation_state(instance_feature, scales, semantics)  # [B, N, hidden_dim]

        # ================================================================
        #  Step 2: Allocation Score + Budget-constrained TopK
        # ================================================================
        allocation_score, selected_mask, topk_idx = self.compute_allocation_topk(h, N)

        # ================================================================
        #  Step 3: 4-class Operation Routing
        # ================================================================
        op_logits, op_prob, op_onehot, op_id = self.compute_operation_routing(h, selected_mask)

        # ================================================================
        #  Step 4: Per-batch Operation-specific Candidate Generation
        # ================================================================

        result_features_list = []
        result_anchors_list = []
        result_opacities_list = []
        result_semantics_list = []
        result_rotations_list = []
        result_means_list = []
        result_scales_list = []

        for b in range(B):
            cur_feats = instance_feature[b]      # (N, E)
            cur_opa = opacities[b]               # (N, 1)
            cur_means = means[b]                 # (N, 3)
            cur_scales = scales[b]               # (N, 3)
            cur_rots = rotations[b]              # (N, 4)
            cur_sems = semantics[b]              # (N, C)
            cur_anchor = anchor[b]               # (N, A)
            cur_h = h[b]                         # (N, hidden_dim)
            cur_selected = selected_mask[b]      # (N,)
            cur_op_id = op_id[b]                 # (N,)

            # Operation masks
            keep_mask = (cur_op_id == KEEP)
            clone_mask = (cur_op_id == CLONE)
            split_mask = (cur_op_id == SPLIT)
            atten_mask = (cur_op_id == ATTEN)

            # Indices for each operation
            clone_indices = torch.where(clone_mask)[0]   # (N_c,)
            split_indices = torch.where(split_mask)[0]   # (N_s,)
            atten_indices = torch.where(atten_mask)[0]   # (N_a,)

            # ---- Start with modifiable containers (all original values) ----
            mod_feats = cur_feats.clone()
            mod_anchor = cur_anchor.clone()
            mod_opa = cur_opa.clone()
            mod_sems = cur_sems.clone()
            mod_rots = cur_rots.clone()
            mod_means = cur_means.clone()
            mod_scales = cur_scales.clone()

            # Collect new gaussians to append
            new_feats_list = []
            new_anchor_list = []
            new_opa_list = []
            new_sems_list = []
            new_rots_list = []
            new_means_list = []
            new_scales_list = []

            # ---- Clone branch (allocationv3.md §10) ----
            if clone_mask.any():
                c_feats = cur_feats[clone_indices]           # (N_c, E)
                c_h = cur_h[clone_indices]                   # (N_c, hidden_dim)
                c_g = GaussianPrediction(
                    means=cur_means[clone_indices],
                    scales=cur_scales[clone_indices],
                    rotations=cur_rots[clone_indices],
                    opacities=cur_opa[clone_indices],
                    semantics=cur_sems[clone_indices],
                )
                c_anchor = cur_anchor[clone_indices]         # (N_c, A)

                c_proj, c_new_anchor, c_new_g = self._clone_branch(
                    c_feats, c_h, c_g, c_anchor)

                # Clone child appended at the end
                new_feats_list.append(c_proj)
                new_anchor_list.append(c_new_anchor)
                new_opa_list.append(c_new_g.opacities)
                new_sems_list.append(c_new_g.semantics)
                new_rots_list.append(c_new_g.rotations)
                new_means_list.append(c_new_g.means)
                new_scales_list.append(c_new_g.scales)

            # ---- Split branch (allocationv3.md §11) ----
            if split_mask.any():
                s_feats = cur_feats[split_indices]           # (N_s, E)
                s_h = cur_h[split_indices]                   # (N_s, hidden_dim)
                s_g = GaussianPrediction(
                    means=cur_means[split_indices],
                    scales=cur_scales[split_indices],
                    rotations=cur_rots[split_indices],
                    opacities=cur_opa[split_indices],
                    semantics=cur_sems[split_indices],
                )
                s_anchor = cur_anchor[split_indices]         # (N_s, A)

                c1_feat, c1_anchor, c1_g, c2_feat, c2_anchor, c2_g = \
                    self._split_branch(s_feats, s_h, s_g, s_anchor)

                # Child 1 replaces the original position
                mod_feats[split_indices] = c1_feat
                mod_anchor[split_indices] = c1_anchor
                mod_opa[split_indices] = c1_g.opacities
                mod_sems[split_indices] = c1_g.semantics
                mod_rots[split_indices] = c1_g.rotations
                mod_means[split_indices] = c1_g.means
                mod_scales[split_indices] = c1_g.scales

                # Child 2 appended at the end
                new_feats_list.append(c2_feat)
                new_anchor_list.append(c2_anchor)
                new_opa_list.append(c2_g.opacities)
                new_sems_list.append(c2_g.semantics)
                new_rots_list.append(c2_g.rotations)
                new_means_list.append(c2_g.means)
                new_scales_list.append(c2_g.scales)

            # ---- Opacity-modulate branch (allocationv3.md §12) ----
            if atten_mask.any():
                a_h = cur_h[atten_indices]                   # (N_a, hidden_dim)
                a_g = GaussianPrediction(
                    means=cur_means[atten_indices],
                    scales=cur_scales[atten_indices],
                    rotations=cur_rots[atten_indices],
                    opacities=cur_opa[atten_indices],
                    semantics=cur_sems[atten_indices],
                )
                a_anchor = cur_anchor[atten_indices]         # (N_a, A)

                a_new_anchor, a_new_g = self._atten_branch(a_h, a_g, a_anchor)

                # Replace original with attenuated version
                mod_anchor[atten_indices] = a_new_anchor
                mod_opa[atten_indices] = a_new_g.opacities
                # Other attributes unchanged

            # ---- Assemble results ----
            if len(new_feats_list) > 0:
                new_feats_cat = torch.cat(new_feats_list, dim=0)        # (N_new, E)
                new_anchor_cat = torch.cat(new_anchor_list, dim=0)      # (N_new, A)
                new_opa_cat = torch.cat(new_opa_list, dim=0)
                new_sems_cat = torch.cat(new_sems_list, dim=0)
                new_rots_cat = torch.cat(new_rots_list, dim=0)
                new_means_cat = torch.cat(new_means_list, dim=0)
                new_scales_cat = torch.cat(new_scales_list, dim=0)

                result_features_b = torch.cat([mod_feats, new_feats_cat], dim=0)
                result_anchor_b = torch.cat([mod_anchor, new_anchor_cat], dim=0)
                result_opa_b = torch.cat([mod_opa, new_opa_cat], dim=0)
                result_sems_b = torch.cat([mod_sems, new_sems_cat], dim=0)
                result_rots_b = torch.cat([mod_rots, new_rots_cat], dim=0)
                result_means_b = torch.cat([mod_means, new_means_cat], dim=0)
                result_scales_b = torch.cat([mod_scales, new_scales_cat], dim=0)
            else:
                result_features_b = mod_feats
                result_anchor_b = mod_anchor
                result_opa_b = mod_opa
                result_sems_b = mod_sems
                result_rots_b = mod_rots
                result_means_b = mod_means
                result_scales_b = mod_scales

            result_features_list.append(result_features_b)
            result_anchors_list.append(result_anchor_b)
            result_opacities_list.append(result_opa_b)
            result_semantics_list.append(result_sems_b)
            result_rotations_list.append(result_rots_b)
            result_means_list.append(result_means_b)
            result_scales_list.append(result_scales_b)

        # ================================================================
        #  Step 5: Stack across batch with padding for variable sizes
        # ================================================================

        sizes = [f.shape[0] for f in result_features_list]

        if len(set(sizes)) == 1:
            # All same size, simple stack
            result_features = torch.stack(result_features_list, dim=0)
            result_anchors = torch.stack(result_anchors_list, dim=0)
            result_opacities = torch.stack(result_opacities_list, dim=0)
            result_semantics = torch.stack(result_semantics_list, dim=0)
            result_rotations = torch.stack(result_rotations_list, dim=0)
            result_means = torch.stack(result_means_list, dim=0)
            result_scales = torch.stack(result_scales_list, dim=0)
        else:
            # Variable sizes: pad to max
            max_m = max(sizes)
            sr0 = self.scale_range[0]
            result_features = self._pad_and_stack(result_features_list, max_m, 0.0)
            result_anchors = self._pad_and_stack(result_anchors_list, max_m, 0.0)
            result_opacities = self._pad_and_stack(result_opacities_list, max_m, 0.0)
            result_semantics = self._pad_and_stack(result_semantics_list, max_m, 0.0)
            result_means = self._pad_and_stack(result_means_list, max_m, 0.0)
            result_scales = self._pad_and_stack(result_scales_list, max_m, sr0)
            result_rotations = self._pad_rotations(result_rotations_list, max_m)

        result_gaussian = GaussianPrediction(
            means=result_means,
            scales=result_scales,
            rotations=result_rotations,
            opacities=result_opacities,
            semantics=result_semantics,
        )

        # store routing info for auxiliary loss (accessed via encoder)
        self._last_op_prob = op_prob
        self._last_op_id = op_id

        return result_anchors, result_gaussian, result_features

    # ------------------------------------------------------------------
    #  Utility: Pad variable-length tensors for batch stacking
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_and_stack(tensor_list, max_size, fill_value=0.0):
        """
        Pad each tensor in the list to max_size along dim 0, then stack.

        Args:
            tensor_list: list of tensors with same ndim, different dim-0
            max_size: target size for dim 0
            fill_value: scalar value to fill padding with

        Returns:
            stacked tensor of shape (len(tensor_list), max_size, ...)
        """
        padded = []
        for t in tensor_list:
            if t.shape[0] < max_size:
                pad = torch.full(
                    (max_size - t.shape[0], *t.shape[1:]),
                    fill_value, device=t.device, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)

    @staticmethod
    def _pad_rotations(tensor_list, max_size):
        """
        Pad rotation quaternions with identity [1, 0, 0, 0] instead of zeros
        to avoid degenerate rotation matrices.

        Args:
            tensor_list: list of (N, 4) rotation tensors
            max_size: target N

        Returns:
            stacked tensor of shape (len(tensor_list), max_size, 4)
        """
        padded = []
        for t in tensor_list:
            if t.shape[0] < max_size:
                pad = torch.zeros(
                    (max_size - t.shape[0], t.shape[1]),
                    device=t.device, dtype=t.dtype)
                pad[:, 0] = 1.0  # identity quaternion
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)
