"""
AdaptiveAllocationv2 module: fully differentiable adaptive gaussian allocation.

Improvements over V1 (AdaptiveAllocation):
1. TRAINING: Soft routing via Gumbel-Softmax(hard=False) — all branches are
   differentiable, gradients flow end-to-end from loss back to decision_net.
2. INFERENCE: Hard routing via argmax + TopK fallback for efficiency.
3. Decision network receives anchor geometric attributes (xyz/scale/rot/opacity)
   via a built-in lightweight anchor_embed encoder.
4. New gaussian opacity is automatically weighted by soft decision scores
   (implicit in soft routing, explicit in hard routing).

Routing logic per gaussian:
- keep:  gaussian passes through unchanged (or blended with split child1)
- clone: original gaussian kept + 1 new refined copy appended
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
class AdaptiveAllocationv2(BaseModule):
    """
    Adaptive gaussian allocation v2 — fully differentiable routing.

    Args:
        feat_embed_dim: feature embedding dimension (default 128)
        semantic_dim:   number of semantic classes (default 17)
        pc_range:       point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]
        scale_range:    scale clamping range [min, max]
        unit_xyz:       unit position shift for clone branch location refinement
        split_mode:     "constrained" (children stay near parent) or "free"
        gumbel_tau:     temperature for Gumbel-Softmax (default 1.0)
        topk_clone:     minimum clone count for inference TopK fallback (default 256)
        topk_split:     minimum split count for inference TopK fallback (default 256)
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
        topk_clone=256,
        topk_split=256,
        **kwargs,
    ):
        super(AdaptiveAllocationv2, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.semantic_dim = semantic_dim
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.split_mode = split_mode
        self.gumbel_tau = gumbel_tau
        self.topk_clone = topk_clone
        self.topk_split = topk_split

        assert split_mode in ("constrained", "free"), \
            f"split_mode must be 'constrained' or 'free', got {split_mode}"

        # Safe margin from pc_range boundary to avoid float32 precision issues
        self.boundary_margin = 0.1

        # Pre-compute unit sigmoid factors for clone branch location shift
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
        #  T3: Lightweight Anchor Embedding Encoder
        #  Encodes anchor geometric attributes (xyz/scale/rot/opacity)
        #  into a unified embedding, then fuses with instance_feature
        #  before feeding into decision_net.
        # ================================================================
        self.anchor_embed_dim = feat_embed_dim

        self.anchor_embed_xyz = nn.Sequential(
            nn.Linear(3, self.anchor_embed_dim),
            nn.GELU(),
        )
        self.anchor_embed_scale = nn.Sequential(
            nn.Linear(3, self.anchor_embed_dim),
            nn.GELU(),
        )
        self.anchor_embed_rot = nn.Sequential(
            nn.Linear(4, self.anchor_embed_dim),
            nn.GELU(),
        )
        self.anchor_embed_opa = nn.Sequential(
            nn.Linear(1, self.anchor_embed_dim),
            nn.GELU(),
        )

        # Fusion layer: concatenated [instance_feature | anchor_embed] → fused
        self.anchor_fusion = nn.Sequential(
            nn.Linear(self.anchor_embed_dim * 2, self.anchor_embed_dim),
            nn.LayerNorm(self.anchor_embed_dim),
            nn.GELU(),
        )

        # ================================================================
        #  T1: Decision Network (receives fused [instance_feature + anchor_embed])
        # ================================================================
        self.decision_net = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
            nn.Linear(feat_embed_dim, 3),
        )

        # ================================================================
        #  Clone Branch Networks (same as V1)
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
        #  Split Branch Networks (same as V1)
        # ================================================================
        self.split_feature_proj = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
        )

        # Child 1 prediction heads
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

        # Child 2 prediction heads
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

    # ==================================================================
    #  T3: Anchor Embedding Computation
    # ==================================================================

    def _compute_anchor_embed(self, anchor: torch.Tensor):
        """
        Encode anchor geometric attributes into a unified embedding.

        Anchor layout: [xyz(3), scale(3), rot(4), opacity(1), semantics(S)]

        Args:
            anchor: (..., A)  anchor tensor

        Returns:
            anchor_embed: (..., E)  encoded anchor embedding
        """
        xyz_embed = self.anchor_embed_xyz(anchor[..., :3])
        scale_embed = self.anchor_embed_scale(anchor[..., 3:6])
        rot_embed = self.anchor_embed_rot(anchor[..., 6:10])
        opa_embed = self.anchor_embed_opa(anchor[..., 10:11])
        return xyz_embed + scale_embed + rot_embed + opa_embed

    # ==================================================================
    #  T1: Decision Computation (training=soft, inference=hard)
    # ==================================================================

    def _compute_decisions(self, instance_feature: torch.Tensor, anchor: torch.Tensor):
        """
        Compute per-gaussian decision outputs.

        Training: returns soft continuous assignments from Gumbel-Softmax(hard=False).
        Inference: returns hard discrete labels from argmax(softmax(logits)).

        Args:
            instance_feature: (N, E)  instance features
            anchor:           (N, A)  anchor tensor

        Returns (training):
            soft_assign: (N, 3)  continuous soft assignment, fully differentiable
        Returns (inference):
            decision_label: (N,)   hard labels {0:keep, 1:clone, 2:split}
            soft_probs:     (N, 3) softmax probabilities (for opacity weighting)
            logits:         (N, 3) raw logits (for TopK fallback scoring)
        """
        # Fuse instance_feature with anchor_embed
        anchor_embed = self._compute_anchor_embed(anchor)              # (N, E)
        fused = torch.cat([instance_feature, anchor_embed], dim=-1)    # (N, 2E)
        fused = self.anchor_fusion(fused)                              # (N, E)
        logits = self.decision_net(fused)                              # (N, 3)

        if self.training:
            # Gumbel-Softmax with hard=False: continuous values, fully differentiable
            # Gumbel noise provides exploration during training
            soft_assign = F.gumbel_softmax(
                logits, tau=self.gumbel_tau, hard=False, dim=-1
            )  # (N, 3), values in [0,1], sum to 1
            return soft_assign
        else:
            probs = F.softmax(logits, dim=-1)                          # (N, 3)
            decision_label = torch.argmax(probs, dim=-1)               # (N,)
            return decision_label, probs, logits

    # ==================================================================
    #  Main Forward: dispatches to soft (training) or hard (inference) routing
    # ==================================================================

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Forward pass of AdaptiveAllocationv2.

        Args:
            instance_feature: (B, N, E)
            anchor:           (B, N, A)
            gaussian:         GaussianPrediction with fields (B, N, *)

        Returns:
            result_anchor:   (B, M, A)
            result_gaussian: GaussianPrediction (B, M, *)
            result_features: (B, M, E)
                where M = 2N (training) or N + N_clone + N_split (inference)
        """
        if self.training:
            return self._soft_routing_forward(instance_feature, anchor, gaussian)
        else:
            return self._hard_routing_forward(instance_feature, anchor, gaussian)

    # ==================================================================
    #  T1: Training Forward — Soft Routing (Fully Differentiable)
    # ==================================================================

    def _soft_routing_forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Training forward: soft routing with differentiable blending.

        All N gaussians go through clone and split branches in parallel.
        Output is a fixed-size tensor of 2N gaussians:
          - First N:  blended originals  (p_keep * orig + p_split * child1)
          - Last N:   blended appended   (p_clone * clone_new + p_split * child2)

        Args:
            instance_feature: (B, N, E)
            anchor:           (B, N, A)
            gaussian:         GaussianPrediction with fields (B, N, *)

        Returns:
            result_anchor:   (B, 2N, A)
            result_gaussian: GaussianPrediction (B, 2N, *)
            result_features: (B, 2N, E)
        """
        batch_size = instance_feature.shape[0]

        result_features_list = []
        result_anchors_list = []
        result_opacities_list = []
        result_semantics_list = []
        result_rotations_list = []
        result_means_list = []
        result_scales_list = []

        for b in range(batch_size):
            cur_feats = instance_feature[b]      # (N, E)
            cur_opa = gaussian.opacities[b]      # (N, 1)
            cur_means = gaussian.means[b]        # (N, 3)
            cur_scales = gaussian.scales[b]      # (N, 3)
            cur_rots = gaussian.rotations[b]     # (N, 4)
            cur_sems = gaussian.semantics[b]     # (N, S)
            cur_anchor = anchor[b]               # (N, A)

            # --- Step 1: Compute soft assignments ---
            soft_assign = self._compute_decisions(cur_feats, cur_anchor)  # (N, 3)
            p_keep  = soft_assign[:, 0]  # (N,)
            p_clone = soft_assign[:, 1]  # (N,)
            p_split = soft_assign[:, 2]  # (N,)

            # --- Step 2: Compute clone branch for ALL gaussians ---
            cur_g = GaussianPrediction(
                means=cur_means, scales=cur_scales,
                rotations=cur_rots, opacities=cur_opa, semantics=cur_sems,
            )
            clone_feat, clone_anchor, clone_g = self._clone_branch(
                cur_feats, cur_g, cur_anchor)  # all (N, *)

            # --- Step 3: Compute split branch for ALL gaussians ---
            c1_feat, c1_anchor, c1_g, c2_feat, c2_anchor, c2_g = \
                self._split_branch(cur_feats, cur_g, cur_anchor)  # all (N, *)

            # --- Step 4: Soft blending at ORIGINAL positions ---
            # keep_branch ⊕ split_child1
            w_keep  = p_keep.unsqueeze(-1)   # (N, 1)
            w_split = p_split.unsqueeze(-1)  # (N, 1)

            mod_feat   = w_keep * cur_feats  + w_split * c1_feat
            mod_anchor = w_keep * cur_anchor + w_split * c1_anchor
            mod_opa    = w_keep * cur_opa    + w_split * c1_g.opacities
            mod_sems   = w_keep * cur_sems   + w_split * c1_g.semantics
            mod_rots   = w_keep * cur_rots   + w_split * c1_g.rotations
            mod_means  = w_keep * cur_means  + w_split * c1_g.means
            mod_scales = w_keep * cur_scales + w_split * c1_g.scales

            # --- Step 5: Soft blending at APPENDED positions ---
            # clone_new ⊕ split_child2
            w_clone = p_clone.unsqueeze(-1)  # (N, 1)

            new_feat   = w_clone * clone_feat    + w_split * c2_feat
            new_anchor = w_clone * clone_anchor   + w_split * c2_anchor
            new_opa    = w_clone * clone_g.opacities + w_split * c2_g.opacities
            new_sems   = w_clone * clone_g.semantics + w_split * c2_g.semantics
            new_rots   = w_clone * clone_g.rotations + w_split * c2_g.rotations
            new_means  = w_clone * clone_g.means     + w_split * c2_g.means
            new_scales = w_clone * clone_g.scales    + w_split * c2_g.scales

            # --- Step 5.5: Clamp blended outputs to valid ranges ---
            # Soft blending with non-normalized weights (w_keep + w_split = 1-p_clone)
            # can produce excessively small scales when p_clone dominates, causing
            # downstream aggregator assertion failures (radii < 1).
            # We clamp scales/opacities/semantics to safe ranges while means are
            # clamped to pc_range to prevent out-of-bounds coordinates.
            m = self.boundary_margin
            pc = self.pc_range
            sr0, sr1 = self.scale_range[0], self.scale_range[1]
            eps = 1e-6

            mod_scales = torch.clamp(mod_scales, sr0, sr1)
            new_scales = torch.clamp(new_scales, sr0, sr1)
            mod_opa = torch.clamp(mod_opa, eps, 1.0 - eps)
            new_opa = torch.clamp(new_opa, eps, 1.0 - eps)
            mod_sems = torch.clamp(mod_sems, eps, 1.0 - eps)
            new_sems = torch.clamp(new_sems, eps, 1.0 - eps)

            mod_means_x = torch.clamp(mod_means[:, 0], pc[0] + m, pc[3] - m)
            mod_means_y = torch.clamp(mod_means[:, 1], pc[1] + m, pc[4] - m)
            mod_means_z = torch.clamp(mod_means[:, 2], pc[2] + m, pc[5] - m)
            mod_means = torch.stack([mod_means_x, mod_means_y, mod_means_z], dim=-1)

            new_means_x = torch.clamp(new_means[:, 0], pc[0] + m, pc[3] - m)
            new_means_y = torch.clamp(new_means[:, 1], pc[1] + m, pc[4] - m)
            new_means_z = torch.clamp(new_means[:, 2], pc[2] + m, pc[5] - m)
            new_means = torch.stack([new_means_x, new_means_y, new_means_z], dim=-1)

            # --- Step 6: Concatenate [modified_originals | appended_new] → (2N, *) ---
            result_feat   = torch.cat([mod_feat,   new_feat],   dim=0)
            result_anchor_b = torch.cat([mod_anchor, new_anchor], dim=0)
            result_opa    = torch.cat([mod_opa,    new_opa],    dim=0)
            result_sems   = torch.cat([mod_sems,   new_sems],   dim=0)
            result_rots   = torch.cat([mod_rots,   new_rots],   dim=0)
            result_means_b = torch.cat([mod_means,  new_means],  dim=0)
            result_scales_b = torch.cat([mod_scales, new_scales], dim=0)

            result_features_list.append(result_feat)
            result_anchors_list.append(result_anchor_b)
            result_opacities_list.append(result_opa)
            result_semantics_list.append(result_sems)
            result_rotations_list.append(result_rots)
            result_means_list.append(result_means_b)
            result_scales_list.append(result_scales_b)

        # Stack across batch: training always produces 2N, so sizes are uniform
        result_features = torch.stack(result_features_list, dim=0)
        result_anchors = torch.stack(result_anchors_list, dim=0)
        result_opacities = torch.stack(result_opacities_list, dim=0)
        result_semantics = torch.stack(result_semantics_list, dim=0)
        result_rotations = torch.stack(result_rotations_list, dim=0)
        result_means = torch.stack(result_means_list, dim=0)
        result_scales = torch.stack(result_scales_list, dim=0)

        result_gaussian = GaussianPrediction(
            means=result_means,
            scales=result_scales,
            rotations=result_rotations,
            opacities=result_opacities,
            semantics=result_semantics,
        )

        return result_anchors, result_gaussian, result_features

    # ==================================================================
    #  T1+T2+T4: Inference Forward — Hard Routing + TopK Fallback
    # ==================================================================

    def _hard_routing_forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Inference forward: hard routing via argmax + TopK fallback.

        Uses discrete decision labels for efficient routing. When clone/split
        counts fall below user-defined thresholds, TopK fallback forcibly
        selects additional gaussians with mutual exclusion.

        New gaussian opacities are explicitly weighted by soft decision scores.

        Args:
            instance_feature: (B, N, E)
            anchor:           (B, N, A)
            gaussian:         GaussianPrediction with fields (B, N, *)

        Returns:
            result_anchor:   (B, M, A)  where M = N + N_clone + N_split
            result_gaussian: GaussianPrediction (B, M, *)
            result_features: (B, M, E)
        """
        opacities = gaussian.opacities   # (B, N, 1)
        means = gaussian.means
        scales = gaussian.scales
        rotations = gaussian.rotations
        semantics = gaussian.semantics

        batch_size = opacities.shape[0]

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

            # ---- Compute decisions (inference: hard labels + soft probs + logits) ----
            decision_label, soft_probs, logits = self._compute_decisions(
                cur_feats, cur_anchor)  # (N,), (N,3), (N,3)

            # ---- T4: Extract soft scores for opacity weighting ----
            p_clone_scores = soft_probs[:, 1]  # (N,)
            p_split_scores = soft_probs[:, 2]  # (N,)

            # ---- T2: TopK fallback with mutual exclusion ----
            clone_mask, split_mask = self._ensure_min_densify(
                decision_label, logits, self.topk_clone, self.topk_split)

            # Record split positions for in-place replacement
            split_indices = torch.where(split_mask)[0]  # (N_s,)

            # ---- Prepare output containers ----
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

            # ---- Clone branch ----
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

                # T4: Explicit opacity weighting with P_clone score
                p_clone = p_clone_scores[clone_idx]          # (N_c,)
                c_g = c_g._replace(
                    opacities=c_g.opacities * p_clone.unsqueeze(-1))

                # Clone new gaussians appended at the end
                new_feats_list.append(c_proj)
                new_anchor_list.append(c_anchor)
                new_opa_list.append(c_g.opacities)
                new_sems_list.append(c_g.semantics)
                new_rots_list.append(c_g.rotations)
                new_means_list.append(c_g.means)
                new_scales_list.append(c_g.scales)

            # ---- Split branch ----
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

                # T4: Explicit opacity weighting with P_split score
                p_split = p_split_scores[split_idx]          # (N_s,)
                c1_g = c1_g._replace(
                    opacities=c1_g.opacities * p_split.unsqueeze(-1))
                c2_g = c2_g._replace(
                    opacities=c2_g.opacities * p_split.unsqueeze(-1))

                # Child_1 replaces original position
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

            # ---- Assemble results ----
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

        # Stack across batch (handle variable-length outputs via padding)
        sizes = [f.shape[0] for f in result_features_list]
        if len(set(sizes)) == 1:
            result_features = torch.stack(result_features_list, dim=0)
            result_anchors = torch.stack(result_anchors_list, dim=0)
            result_opacities = torch.stack(result_opacities_list, dim=0)
            result_semantics = torch.stack(result_semantics_list, dim=0)
            result_rotations = torch.stack(result_rotations_list, dim=0)
            result_means = torch.stack(result_means_list, dim=0)
            result_scales = torch.stack(result_scales_list, dim=0)
        else:
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

        return result_anchors, result_gaussian, result_features

    # ==================================================================
    #  T2: TopK Fallback with Mutual Exclusion (inference only)
    # ==================================================================

    def _ensure_min_densify(self, decision_label, logits, topk_clone, topk_split):
        """
        Ensure minimum clone/split count via TopK fallback with mutual exclusion.

        When the number of gaussians predicted as clone (or split) by decision_net
        falls below the user-defined threshold, this function forcibly selects
        additional gaussians with the highest clone (or split) logit scores.

        Clone selection takes priority: gaussians assigned to clone are excluded
        from subsequent split selection, guaranteeing mutual exclusion.

        Args:
            decision_label: (N,)  hard labels from argmax {0:keep, 1:clone, 2:split}
            logits:         (N, 3) raw decision_net output, used as confidence scores
            topk_clone:     int,   user-defined minimum clone count
            topk_split:     int,   user-defined minimum split count

        Returns:
            clone_mask: (N,) bool, guaranteed >= topk_clone True values
            split_mask: (N,) bool, guaranteed >= topk_split True values,
                          mutually exclusive with clone_mask
        """
        clone_mask = (decision_label == 1)
        split_mask = (decision_label == 2)

        n_clone = clone_mask.sum().item()
        n_split = split_mask.sum().item()

        need_clone = (n_clone < topk_clone)
        need_split = (n_split < topk_split)

        if not need_clone and not need_split:
            return clone_mask, split_mask

        # Use decision_net's own clone/split logits as confidence scores
        clone_score = logits[:, 1].clone()  # (N,), higher = more likely to clone
        split_score = logits[:, 2].clone()  # (N,), higher = more likely to split

        # Exclude gaussians already assigned to clone or split
        clone_score[clone_mask | split_mask] = -float('inf')

        # Clone has priority — fill shortage first
        if need_clone:
            shortage = topk_clone - n_clone
            _, topk_idx = torch.topk(clone_score, shortage, dim=-1)
            clone_mask[topk_idx] = True
            # Exclude newly-selected clone gaussians from split candidates
            clone_score[topk_idx] = -float('inf')
            split_score[topk_idx] = -float('inf')

        # Split fills its shortage from remaining (non-clone, non-split)
        if need_split:
            split_score[clone_mask | split_mask] = -float('inf')
            shortage = topk_split - n_split
            _, topk_idx = torch.topk(split_score, shortage, dim=-1)
            split_mask[topk_idx] = True

        return clone_mask, split_mask

    # ==================================================================
    #  Clone Branch (same as V1 AdaptiveAllocation._clone_branch)
    # ==================================================================

    def _clone_branch(self, clone_feats, clone_gaussian, clone_anchor):
        """
        Process clone-selected gaussians. Same logic as V1 DensifyOnly.

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

        # ---- Location shift (restricted, same as V1) ----
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
        m = self.boundary_margin
        new_means = clone_gaussian.means + means_shift * clone_gaussian.scales
        new_means_x = torch.clamp(
            new_means[:, 0], self.pc_range[0] + m, self.pc_range[3] - m)
        new_means_y = torch.clamp(
            new_means[:, 1], self.pc_range[1] + m, self.pc_range[4] - m)
        new_means_z = torch.clamp(
            new_means[:, 2], self.pc_range[2] + m, self.pc_range[5] - m)
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

    # ==================================================================
    #  Split Branch (same as V1 AdaptiveAllocation._split_branch)
    # ==================================================================

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

        # ---- Boundary clamping ----
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

    # ==================================================================
    #  Split Mode: Constrained (same as V1)
    # ==================================================================

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

    # ==================================================================
    #  Split Mode: Free (same as V1)
    # ==================================================================

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

    # ==================================================================
    #  Utility: pad variable-length tensors (same as V1)
    # ==================================================================

    @staticmethod
    def _pad_and_stack(tensor_list, max_size, fill_value=0.0):
        """
        Pad each tensor in the list to max_size along dim 0, then stack.

        Args:
            tensor_list: list of tensors with same ndim, different dim-0
            max_size:    target size for dim 0
            fill_value:  scalar value to fill padding with (default 0.0)

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
        to avoid degenerate rotation matrices in downstream processing.

        Args:
            tensor_list: list of (N, 4) rotation tensors
            max_size:    target N

        Returns:
            stacked tensor of shape (len(tensor_list), max_size, 4)
        """
        padded = []
        for t in tensor_list:
            if t.shape[0] < max_size:
                pad = torch.zeros(
                    (max_size - t.shape[0], t.shape[1]),
                    device=t.device, dtype=t.dtype)
                pad[:, 0] = 1.0  # identity quaternion: w=1, x=y=z=0
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)
