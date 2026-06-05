"""
AdaptiveAllocation module: adaptively decides keep/clone/split for each gaussian.

Unlike DensifyOnly which uses Top-K selection and uniform densify,
this module learns a 3-class decision factor [keep, clone, split] for EVERY
input gaussian, then routes each to its corresponding processing branch.

- keep:  gaussian passes through unchanged
- clone: original gaussian kept + 1 new refined copy appended (same as DensifyOnly)
- split: original gaussian discarded, 2 children generated via learned network
         child_1 replaces the original position, child_2 is appended
"""

from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn as nn, torch
import torch.nn.functional as F
from ..utils import GaussianPrediction
from ....utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class AdaptiveAllocation(BaseModule):
    """
    Adaptive gaussian allocation module with learned keep/clone/split decisions.

    Args:
        feat_embed_dim: feature embedding dimension (default 128)
        semantic_dim: number of semantic classes (default 17)
        pc_range: point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]
        scale_range: scale clamping range [min, max]
        unit_xyz: unit position shift for clone branch location refinement
        split_mode: "constrained" (children stay near parent) or "free" (learn freely)
        gumbel_tau: temperature for Gumbel-Softmax during training (default 1.0)
    """

    def __init__(
        self,
        feat_embed_dim=128,
        semantic_dim=17,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        split_mode="constrained",
        gumbel_tau=1.0,
        **kwargs,
    ):
        super(AdaptiveAllocation, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.semantic_dim = semantic_dim
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.split_mode = split_mode
        self.gumbel_tau = gumbel_tau

        assert split_mode in ("constrained", "free"), \
            f"split_mode must be 'constrained' or 'free', got {split_mode}"

        # Pre-compute unit sigmoid factors for clone branch location shift
        # (same as DensifyOnly)
        unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
        unit_prob = [4 * unit_prob[i] for i in range(3)]
        self.unit_sigmoid = unit_prob

        # Pre-compute pc_range span for split branch free-mode offsets
        self.pc_span = [
            pc_range[3] - pc_range[0],
            pc_range[4] - pc_range[1],
            pc_range[5] - pc_range[2],
        ]

        # ================================================================
        #  T1: Decision Network
        #  Learns P_keep, P_clone, P_split for each gaussian
        # ================================================================
        self.decision_net = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
            nn.Linear(feat_embed_dim, 3),
        )

        # ================================================================
        #  T3: Clone Branch Networks (mirrors DensifyOnly)
        # ================================================================
        self.clone_feature_proj = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
        )

        self.clone_location_shift = nn.Linear(feat_embed_dim, 3)
        self.clone_scale_shift = nn.Linear(feat_embed_dim, 3)
        self.clone_rotation_shift = nn.Linear(feat_embed_dim, 4)
        self.clone_semantic_shift = nn.Linear(feat_embed_dim, semantic_dim)
        self.clone_opacities_shift = nn.Linear(feat_embed_dim, 1)

        # ================================================================
        #  T4: Split Branch Networks
        #  Deeper feature projection + dual child prediction heads
        # ================================================================
        self.split_feature_proj = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
        )

        # Child 1 prediction heads (5 geometric + 1 feature transform)
        self.c1_location_shift = nn.Linear(feat_embed_dim, 3)
        self.c1_scale_shift = nn.Linear(feat_embed_dim, 3)
        self.c1_rotation_shift = nn.Linear(feat_embed_dim, 4)
        self.c1_semantic_shift = nn.Linear(feat_embed_dim, semantic_dim)
        self.c1_opacities_shift = nn.Linear(feat_embed_dim, 1)
        self.c1_feature_transform = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
        )

        # Child 2 prediction heads (5 geometric + 1 feature transform)
        self.c2_location_shift = nn.Linear(feat_embed_dim, 3)
        self.c2_scale_shift = nn.Linear(feat_embed_dim, 3)
        self.c2_rotation_shift = nn.Linear(feat_embed_dim, 4)
        self.c2_semantic_shift = nn.Linear(feat_embed_dim, semantic_dim)
        self.c2_opacities_shift = nn.Linear(feat_embed_dim, 1)
        self.c2_feature_transform = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    #  Helper: compute decision labels from instance features
    # ------------------------------------------------------------------

    def _compute_decisions(self, instance_feature: torch.Tensor):
        """
        Compute per-gaussian decision labels and probabilities.

        Args:
            instance_feature: (B, N, E)

        Returns:
            decision_label: (B, N) hard labels {0:keep, 1:clone, 2:split}
            decision_probs:  (B, N, 3) soft probabilities
        """
        logits = self.decision_net(instance_feature)  # (B, N, 3)

        if self.training:
            # Gumbel-Softmax with hard forward, soft backward
            decision_one_hot = F.gumbel_softmax(
                logits, tau=self.gumbel_tau, hard=True, dim=-1
            )  # (B, N, 3)
            decision_label = torch.argmax(decision_one_hot, dim=-1)  # (B, N)
            decision_probs = decision_one_hot  # one-hot but with gradients
        else:
            probs = F.softmax(logits, dim=-1)  # (B, N, 3)
            decision_label = torch.argmax(probs, dim=-1)  # (B, N)
            decision_probs = probs

        return decision_label, decision_probs

    # ------------------------------------------------------------------
    #  Helper: clone branch processing (mirrors DensifyOnly)
    # ------------------------------------------------------------------

    def _clone_branch(self, clone_feats, clone_gaussian, clone_anchor):
        """
        Process clone-selected gaussians. Same logic as DensifyOnly.

        Args:
            clone_feats:    (N_c, E)  instance features of clone gaussians
            clone_gaussian: GaussianPrediction fields, each (N_c, *)
            clone_anchor:   (N_c, A)  anchor of clone gaussians

        Returns:
            new_feats:   (N_c, E)  projected features for new gaussians
            new_anchor:  (N_c, A)  anchor for new gaussians
            new_gaussian: GaussianPrediction for new gaussians
        """
        # Feature projection
        projected = self.clone_feature_proj(clone_feats)  # (N_c, E)

        # ---- Location shift (restricted, same as DensifyOnly) ----
        delta_xyz_sigmoid = self.clone_location_shift(projected)  # (N_c, 3)
        delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1  # [-1, 1]
        mean_shift_out = torch.stack([
            delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
            delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
            delta_xyz_prob[..., 2] * self.unit_sigmoid[2],
        ], dim=-1)  # (N_c, 3)

        # ---- Other attribute shifts ----
        scale_shift_out = self.clone_scale_shift(projected)       # (N_c, 3)
        rots_shift_out = F.normalize(self.clone_rotation_shift(projected))  # (N_c, 4)
        sem_shift_out = self.clone_semantic_shift(projected)       # (N_c, S)
        opa_shift_out = self.clone_opacities_shift(projected)      # (N_c, 1)

        # ---- Residual refinement ----
        means_shift = (safe_sigmoid(mean_shift_out) - 0.5) * 2    # [-1, 1]
        scale_shift = safe_sigmoid(scale_shift_out)                # [0, 1]
        rots_shift = rots_shift_out
        sem_shift = safe_sigmoid(sem_shift_out)
        opa_shift = safe_sigmoid(opa_shift_out)

        # New means: shift relative to old scale, clamp to pc_range
        new_means = clone_gaussian.means + means_shift * clone_gaussian.scales
        new_means_x = torch.clamp(
            new_means[:, 0], self.pc_range[0] + 1e-6, self.pc_range[3] - 1e-6)
        new_means_y = torch.clamp(
            new_means[:, 1], self.pc_range[1] + 1e-6, self.pc_range[4] - 1e-6)
        new_means_z = torch.clamp(
            new_means[:, 2], self.pc_range[2] + 1e-6, self.pc_range[5] - 1e-6)
        new_means = torch.stack([new_means_x, new_means_y, new_means_z], dim=-1)
        new_xyz_anchor = safe_sigmoid(new_means)  # convert to anchor format

        # New scales: multiplicative residual, clamp to scale_range
        new_scales = clone_gaussian.scales * scale_shift
        new_scales = torch.clamp(new_scales, self.scale_range[0], self.scale_range[1])

        # New rotations / semantics / opacities: average blend
        new_rots = clone_gaussian.rotations / 2 + rots_shift / 2
        new_sems = clone_gaussian.semantics / 2 + sem_shift / 2
        new_opas = clone_gaussian.opacities / 2 + opa_shift / 2

        # ---- Anchor update ----
        new_anchor_scale = clone_anchor[:, 3:6] * scale_shift
        new_anchor_rot = clone_anchor[:, 6:10] / 2 + rots_shift_out / 2
        new_anchor_opa = clone_anchor[:, 10:11] / 2 + opa_shift_out / 2
        new_anchor_sem = clone_anchor[:, 11:] / 2 + sem_shift_out / 2

        new_anchor = torch.cat([
            new_xyz_anchor, new_anchor_scale, new_anchor_rot,
            new_anchor_opa, new_anchor_sem
        ], dim=-1)

        new_gaussian = GaussianPrediction(
            means=new_means,
            scales=new_scales,
            rotations=new_rots,
            opacities=new_opas,
            semantics=new_sems,
        )

        return projected, new_anchor, new_gaussian

    # ------------------------------------------------------------------
    #  Helper: split branch processing
    # ------------------------------------------------------------------

    def _split_branch(self, split_feats, split_gaussian, split_anchor):
        """
        Process split-selected gaussians. Generates 2 children,
        original parent is discarded.

        Args:
            split_feats:    (N_s, E)  instance features of split gaussians
            split_gaussian: GaussianPrediction fields, each (N_s, *)
            split_anchor:   (N_s, A)  anchor of split gaussians

        Returns:
            c1_feat:     (N_s, E)  child_1 instance feature
            c1_anchor:   (N_s, A)  child_1 anchor
            c1_gaussian: GaussianPrediction for child_1
            c2_feat:     (N_s, E)  child_2 instance feature
            c2_anchor:   (N_s, A)  child_2 anchor
            c2_gaussian: GaussianPrediction for child_2
        """
        device = split_feats.device
        parent_means = split_gaussian.means      # (N_s, 3)
        parent_scales = split_gaussian.scales     # (N_s, 3)
        parent_rots = split_gaussian.rotations    # (N_s, 4)
        parent_sems = split_gaussian.semantics    # (N_s, S)
        parent_opas = split_gaussian.opacities    # (N_s, 1)

        # ---- Deep feature projection ----
        projected = self.split_feature_proj(split_feats)  # (N_s, E)

        # ---- Child 1 predictions ----
        c1_loc = self.c1_location_shift(projected)       # (N_s, 3)
        c1_scale = self.c1_scale_shift(projected)         # (N_s, 3)
        c1_rot = self.c1_rotation_shift(projected)        # (N_s, 4)
        c1_sem = self.c1_semantic_shift(projected)        # (N_s, S)
        c1_opa = self.c1_opacities_shift(projected)       # (N_s, 1)
        c1_feat = self.c1_feature_transform(projected)    # (N_s, E)

        # ---- Child 2 predictions ----
        c2_loc = self.c2_location_shift(projected)       # (N_s, 3)
        c2_scale = self.c2_scale_shift(projected)         # (N_s, 3)
        c2_rot = self.c2_rotation_shift(projected)        # (N_s, 4)
        c2_sem = self.c2_semantic_shift(projected)        # (N_s, S)
        c2_opa = self.c2_opacities_shift(projected)       # (N_s, 1)
        c2_feat = self.c2_feature_transform(projected)    # (N_s, E)

        # ---- Compute child geometry based on split_mode ----
        if self.split_mode == "constrained":
            c1_means, c2_means, c1_scales, c2_scales, \
                c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas = \
                self._split_constrained(
                    parent_means, parent_scales, parent_rots,
                    parent_sems, parent_opas,
                    c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
                    c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
                    device)
        else:  # free
            c1_means, c2_means, c1_scales, c2_scales, \
                c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas = \
                self._split_free(
                    parent_means, parent_scales, parent_rots,
                    parent_sems, parent_opas,
                    c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
                    c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
                    device)

        # ---- Boundary clamping (prevent InverseSigmoid NaN) ----
        pc = self.pc_range
        c1_means_x = torch.clamp(c1_means[:, 0], pc[0] + 1e-6, pc[3] - 1e-6)
        c1_means_y = torch.clamp(c1_means[:, 1], pc[1] + 1e-6, pc[4] - 1e-6)
        c1_means_z = torch.clamp(c1_means[:, 2], pc[2] + 1e-6, pc[5] - 1e-6)
        c1_means = torch.stack([c1_means_x, c1_means_y, c1_means_z], dim=-1)

        c2_means_x = torch.clamp(c2_means[:, 0], pc[0] + 1e-6, pc[3] - 1e-6)
        c2_means_y = torch.clamp(c2_means[:, 1], pc[1] + 1e-6, pc[4] - 1e-6)
        c2_means_z = torch.clamp(c2_means[:, 2], pc[2] + 1e-6, pc[5] - 1e-6)
        c2_means = torch.stack([c2_means_x, c2_means_y, c2_means_z], dim=-1)

        c1_scales = torch.clamp(c1_scales, self.scale_range[0], self.scale_range[1])
        c2_scales = torch.clamp(c2_scales, self.scale_range[0], self.scale_range[1])

        eps = 1e-6
        c1_sems = torch.clamp(c1_sems, eps, 1.0 - eps)
        c2_sems = torch.clamp(c2_sems, eps, 1.0 - eps)
        c1_opas = torch.clamp(c1_opas, eps, 1.0 - eps)
        c2_opas = torch.clamp(c2_opas, eps, 1.0 - eps)

        # ---- Anchor conversion ----
        c1_anchor_xyz = safe_sigmoid(c1_means)
        c2_anchor_xyz = safe_sigmoid(c2_means)

        # For anchor scale/opa/sem: convert from physical to anchor (sigmoid) space
        sr0, sr1 = self.scale_range[0], self.scale_range[1]
        c1_anchor_scale = torch.log(
            (c1_scales - sr0) / (sr1 - c1_scales + 1e-8) + 1e-8)
        c2_anchor_scale = torch.log(
            (c2_scales - sr0) / (sr1 - c2_scales + 1e-8) + 1e-8)

        c1_anchor_opa = torch.log(c1_opas / (1 - c1_opas + 1e-8) + 1e-8)
        c2_anchor_opa = torch.log(c2_opas / (1 - c2_opas + 1e-8) + 1e-8)
        c1_anchor_sem = torch.log(c1_sems / (1 - c1_sems + 1e-8) + 1e-8)
        c2_anchor_sem = torch.log(c2_sems / (1 - c2_sems + 1e-8) + 1e-8)

        c1_anchor = torch.cat([
            c1_anchor_xyz, c1_anchor_scale, c1_rots,
            c1_anchor_opa, c1_anchor_sem
        ], dim=-1)

        c2_anchor = torch.cat([
            c2_anchor_xyz, c2_anchor_scale, c2_rots,
            c2_anchor_opa, c2_anchor_sem
        ], dim=-1)

        c1_gaussian = GaussianPrediction(
            means=c1_means, scales=c1_scales,
            rotations=c1_rots, opacities=c1_opas, semantics=c1_sems)
        c2_gaussian = GaussianPrediction(
            means=c2_means, scales=c2_scales,
            rotations=c2_rots, opacities=c2_opas, semantics=c2_sems)

        return c1_feat, c1_anchor, c1_gaussian, \
            c2_feat, c2_anchor, c2_gaussian

    # ------------------------------------------------------------------
    #  Split mode: constrained (children near parent along max-scale axis)
    # ------------------------------------------------------------------

    def _split_constrained(
        self,
        parent_means, parent_scales, parent_rots, parent_sems, parent_opas,
        c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
        c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
        device,
    ):
        """
        Constrained split: children placed symmetrically along the
        parent's maximum-scale axis, staying within/near the parent.
        """
        # Split axis: dimension with largest scale
        split_axis = torch.argmax(parent_scales, dim=-1)  # (N_s,)
        direction = F.one_hot(split_axis, num_classes=3).float()  # (N_s, 3)

        # Offset magnitude: learned from c1_loc, always positive
        offset_mag = F.softplus(c1_loc.sum(dim=-1, keepdim=True))  # (N_s, 1)

        # Child positions: symmetric along split axis
        c1_means = parent_means + direction * offset_mag * parent_scales
        c2_means = parent_means - direction * offset_mag * parent_scales

        # Child scales: inherit from parent, shrink to [0.3, 0.7]×
        c1_scale_factor = 0.3 + 0.4 * torch.sigmoid(c1_scale.sum(dim=-1, keepdim=True))
        c2_scale_factor = 0.3 + 0.4 * torch.sigmoid(c2_scale.sum(dim=-1, keepdim=True))
        c1_scales = parent_scales * c1_scale_factor
        c2_scales = parent_scales * c2_scale_factor

        # Child rotations: parent + small residual
        c1_rots = F.normalize(parent_rots + c1_rot * 0.1)
        c2_rots = F.normalize(parent_rots + c2_rot * 0.1)

        # Child semantics / opacity: parent + small tanh residual
        c1_sems = parent_sems + torch.tanh(c1_sem) * 0.1
        c2_sems = parent_sems + torch.tanh(c2_sem) * 0.1
        c1_opas = parent_opas + torch.tanh(c1_opa) * 0.1
        c2_opas = parent_opas + torch.tanh(c2_opa) * 0.1

        return c1_means, c2_means, c1_scales, c2_scales, \
            c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas

    # ------------------------------------------------------------------
    #  Split mode: free (children can be anywhere)
    # ------------------------------------------------------------------

    def _split_free(
        self,
        parent_means, parent_scales, parent_rots, parent_sems, parent_opas,
        c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
        c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
        device,
    ):
        """
        Free split: children positions and attributes learned without
        parent constraints.
        """
        pc_span = torch.tensor(self.pc_span, device=device, dtype=parent_means.dtype)

        # Child positions: free offset from parent, up to ±half scene span
        delta1 = torch.tanh(c1_loc) * pc_span * 0.5  # (N_s, 3)
        delta2 = torch.tanh(c2_loc) * pc_span * 0.5
        c1_means = parent_means + delta1
        c2_means = parent_means + delta2

        # Child scales: independent prediction
        c1_scales = F.softplus(c1_scale)
        c2_scales = F.softplus(c2_scale)

        # Child rotations: independent prediction
        c1_rots = F.normalize(c1_rot)
        c2_rots = F.normalize(c2_rot)

        # Child semantics / opacity: independent prediction (sigmoid-bounded)
        c1_sems = torch.sigmoid(c1_sem)
        c2_sems = torch.sigmoid(c2_sem)
        c1_opas = torch.sigmoid(c1_opa)
        c2_opas = torch.sigmoid(c2_opa)

        return c1_means, c2_means, c1_scales, c2_scales, \
            c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas

    # ------------------------------------------------------------------
    #  Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Forward pass of AdaptiveAllocation.

        Args:
            instance_feature: (B, N, E)
            anchor:           (B, N, A)
            gaussian:         GaussianPrediction with fields (B, N, *)

        Returns:
            result_anchor:   (B, M, A)  where M = N + N_clone + N_split
            result_gaussian: GaussianPrediction (B, M, *)
            result_features: (B, M, E)
        """
        opacities = gaussian.opacities  # B, N, 1
        means = gaussian.means
        scales = gaussian.scales
        rotations = gaussian.rotations
        semantics = gaussian.semantics

        batch_size = opacities.shape[0]

        # Results collected across batch items
        result_features_list = []
        result_anchors_list = []
        result_opacities_list = []
        result_semantics_list = []
        result_rotations_list = []
        result_means_list = []
        result_scales_list = []

        for b in range(batch_size):
            cur_feats = instance_feature[b]      # (N, E)
            cur_opa = opacities[b]               # (N, 1)
            cur_means = means[b]                 # (N, 3)
            cur_scales = scales[b]               # (N, 3)
            cur_rots = rotations[b]              # (N, 4)
            cur_sems = semantics[b]              # (N, S)
            cur_anchor = anchor[b]               # (N, A)

            # ---- T1: Learn decision factor ----
            decision_label, _ = self._compute_decisions(
                cur_feats.unsqueeze(0))  # (1, N, E) -> (1, N), (1, N, 3)
            decision_label = decision_label.squeeze(0)  # (N,)

            # ---- T2: Classify and route ----
            keep_mask = (decision_label == 0)
            clone_mask = (decision_label == 1)
            split_mask = (decision_label == 2)

            # Record split positions for in-place replacement
            split_indices = torch.where(split_mask)[0]  # (N_s,)

            # ---- Prepare output containers ----
            # Start with all original values (keep + clone originals + split to-be-replaced)
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

            # ---- T3: Clone branch ----
            if clone_mask.any():
                clone_idx = torch.where(clone_mask)[0]
                clone_feats = cur_feats[clone_idx]           # (N_c, E)

                clone_g = GaussianPrediction(
                    means=cur_means[clone_idx],
                    scales=cur_scales[clone_idx],
                    rotations=cur_rots[clone_idx],
                    opacities=cur_opa[clone_idx],
                    semantics=cur_sems[clone_idx],
                )
                clone_a = cur_anchor[clone_idx]              # (N_c, A)

                c_proj, c_anchor, c_g = self._clone_branch(
                    clone_feats, clone_g, clone_a)

                # Clone new gaussians appended at the end
                new_feats_list.append(c_proj)
                new_anchor_list.append(c_anchor)
                new_opa_list.append(c_g.opacities)
                new_sems_list.append(c_g.semantics)
                new_rots_list.append(c_g.rotations)
                new_means_list.append(c_g.means)
                new_scales_list.append(c_g.scales)

            # ---- T4: Split branch ----
            if split_mask.any():
                split_idx = torch.where(split_mask)[0]
                split_feats = cur_feats[split_idx]           # (N_s, E)

                split_g = GaussianPrediction(
                    means=cur_means[split_idx],
                    scales=cur_scales[split_idx],
                    rotations=cur_rots[split_idx],
                    opacities=cur_opa[split_idx],
                    semantics=cur_sems[split_idx],
                )
                split_a = cur_anchor[split_idx]              # (N_s, A)

                c1_feat, c1_anchor, c1_g, c2_feat, c2_anchor, c2_g = \
                    self._split_branch(split_feats, split_g, split_a)

                # ---- T5: In-place replacement with child_1 ----
                mod_feats[split_indices] = c1_feat
                mod_anchor[split_indices] = c1_anchor
                mod_opa[split_indices] = c1_g.opacities
                mod_sems[split_indices] = c1_g.semantics
                mod_rots[split_indices] = c1_g.rotations
                mod_means[split_indices] = c1_g.means
                mod_scales[split_indices] = c1_g.scales

                # Child_2 appended at the end
                new_feats_list.append(c2_feat)
                new_anchor_list.append(c2_anchor)
                new_opa_list.append(c2_g.opacities)
                new_sems_list.append(c2_g.semantics)
                new_rots_list.append(c2_g.rotations)
                new_means_list.append(c2_g.means)
                new_scales_list.append(c2_g.scales)

            # ---- T5: Assemble results ----
            if len(new_feats_list) > 0:
                new_feats_cat = torch.cat(new_feats_list, dim=0)      # (N_new, E)
                new_anchor_cat = torch.cat(new_anchor_list, dim=0)    # (N_new, A)
                new_opa_cat = torch.cat(new_opa_list, dim=0)
                new_sems_cat = torch.cat(new_sems_list, dim=0)
                new_rots_cat = torch.cat(new_rots_list, dim=0)
                new_means_cat = torch.cat(new_means_list, dim=0)
                new_scales_cat = torch.cat(new_scales_list, dim=0)

                result_feats = torch.cat([mod_feats, new_feats_cat], dim=0)
                result_anchor_b = torch.cat([mod_anchor, new_anchor_cat], dim=0)
                result_opa = torch.cat([mod_opa, new_opa_cat], dim=0)
                result_sems = torch.cat([mod_sems, new_sems_cat], dim=0)
                result_rots = torch.cat([mod_rots, new_rots_cat], dim=0)
                result_means_b = torch.cat([mod_means, new_means_cat], dim=0)
                result_scales_b = torch.cat([mod_scales, new_scales_cat], dim=0)
            else:
                result_feats = mod_feats
                result_anchor_b = mod_anchor
                result_opa = mod_opa
                result_sems = mod_sems
                result_rots = mod_rots
                result_means_b = mod_means
                result_scales_b = mod_scales

            result_features_list.append(result_feats)
            result_anchors_list.append(result_anchor_b)
            result_opacities_list.append(result_opa)
            result_semantics_list.append(result_sems)
            result_rotations_list.append(result_rots)
            result_means_list.append(result_means_b)
            result_scales_list.append(result_scales_b)

        # Stack across batch (padded if needed, but variable counts
        # within batch should be rare; if they occur, pad to max)
        # Note: different batch items may have different M.
        # We stack them as a list and return padded tensor if needed,
        # or rely on the framework to handle variable-length tensors.

        # For simplicity, we stack if all have same M, otherwise
        # we pad to max M
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
            result_features = self._pad_and_stack(result_features_list, max_m)
            result_anchors = self._pad_and_stack(result_anchors_list, max_m)
            result_opacities = self._pad_and_stack(result_opacities_list, max_m)
            result_semantics = self._pad_and_stack(result_semantics_list, max_m)
            result_rotations = self._pad_and_stack(result_rotations_list, max_m)
            result_means = self._pad_and_stack(result_means_list, max_m)
            result_scales = self._pad_and_stack(result_scales_list, max_m)

        result_gaussian = GaussianPrediction(
            means=result_means,
            scales=result_scales,
            rotations=result_rotations,
            opacities=result_opacities,
            semantics=result_semantics,
        )

        return result_anchors, result_gaussian, result_features

    # ------------------------------------------------------------------
    #  Utility: pad variable-length tensors to same size for stacking
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_and_stack(tensor_list, max_size):
        """
        Pad each tensor in the list to max_size along dim 0, then stack.

        Args:
            tensor_list: list of tensors with same ndim, different dim-0
            max_size: target size for dim 0

        Returns:
            stacked tensor of shape (len(tensor_list), max_size, ...)
        """
        padded = []
        for t in tensor_list:
            if t.shape[0] < max_size:
                pad = torch.zeros(
                    (max_size - t.shape[0], *t.shape[1:]),
                    device=t.device, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)
