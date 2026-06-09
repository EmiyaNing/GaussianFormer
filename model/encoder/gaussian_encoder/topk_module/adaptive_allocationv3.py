"""
AdaptiveAllocationV3: Opacity-Weighted Candidate Routing with Training TopK Fallback.

Core design change from V2:
    V2: soft attribute blending → interpolates mean/scale/rot/semantic/feature
    V3: opacity-weighted candidate routing → only modulates opacity

Training:
    Build all 4N candidates (original, clone_new, split_child1, split_child2),
    apply Gumbel-Softmax + TopK-aware probability boosting, then use decision
    probabilities only as opacity weights. No cross-branch attribute blending.

Inference:
    Hard routing via argmax + TopK fallback with mutual exclusion.
    - keep:  output original
    - clone: output original + clone_new
    - split: output split_child1 + split_child2 (original discarded)

Key features:
    1. No cross-branch attribute interpolation
    2. Keep Gaussian geometry/semantics/feature unchanged
    3. Clone: original + clone_new in both training and inference
    4. Split: child1 + child2 in both training and inference
    5. Training TopK fallback via probability boosting (no extra loss)
    6. decision_net gradients flow through opacity contribution
"""

from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn as nn, torch
import torch.nn.functional as F
from ..utils import GaussianPrediction
from ....utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class AdaptiveAllocationV3(BaseModule):
    """
    Adaptive Gaussian Allocation V3 — opacity-weighted candidate routing.

    Args:
        feat_embed_dim:         feature embedding dimension (default 128)
        semantic_dim:           number of semantic classes (default 17)
        pc_range:               point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]
        scale_range:            scale clamping range [min, max]
        unit_xyz:               unit position shift for clone branch location refinement
        split_mode:             "constrained" (children stay near parent) or "free"
        gumbel_tau:             temperature for Gumbel-Softmax (default 1.0)
        topk_clone:             minimum clone count for TopK fallback (default 256)
        topk_split:             minimum split count for TopK fallback (default 256)
        use_train_topk_fallback: enable TopK-aware probability boosting in training
        train_topk_boost_alpha: boosting strength for training TopK fallback [0, 1]
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
        use_train_topk_fallback=True,
        train_topk_boost_alpha=0.5,
        **kwargs,
    ):
        super(AdaptiveAllocationV3, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.semantic_dim = semantic_dim
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.split_mode = split_mode
        self.gumbel_tau = gumbel_tau
        self.topk_clone = topk_clone
        self.topk_split = topk_split
        self.use_train_topk_fallback = use_train_topk_fallback
        self.train_topk_boost_alpha = train_topk_boost_alpha

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
        #  T6: Anchor Embedding Encoder
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

        # Fusion layer: [instance_feature | anchor_embed] → fused
        self.anchor_fusion = nn.Sequential(
            nn.Linear(self.anchor_embed_dim * 2, self.anchor_embed_dim),
            nn.LayerNorm(self.anchor_embed_dim),
            nn.GELU(),
        )

        # ================================================================
        #  Decision Network: outputs 3-class logits [keep, clone, split]
        # ================================================================
        self.decision_net = nn.Sequential(
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.LayerNorm(feat_embed_dim),
            nn.GELU(),
            nn.Linear(feat_embed_dim, 3),
        )

        # ================================================================
        #  Clone Branch Networks (same as V2)
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
        #  Split Branch Networks (same as V2)
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
    #  T6: Anchor Embedding Computation
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
    #  T3: Decision Computation (training=soft, inference=hard)
    # ==================================================================

    def _compute_decisions(self, instance_feature: torch.Tensor, anchor: torch.Tensor):
        """
        Compute per-gaussian decision outputs.

        Training: returns soft continuous assignments from Gumbel-Softmax(hard=False).
        Inference: returns hard discrete labels + softmax probs + raw logits.

        Args:
            instance_feature: (N, E)  instance features
            anchor:           (N, A)  anchor tensor

        Returns (training):
            soft_assign: (N, 3)  continuous soft assignment, fully differentiable
            logits:      (N, 3)  raw logits (for TopK probability boosting)
        Returns (inference):
            decision_label: (N,)   hard labels {0:keep, 1:clone, 2:split}
            soft_probs:     (N, 3) softmax probabilities (for opacity weighting)
            logits:         (N, 3) raw logits (for TopK fallback scoring)
        """
        # Fuse instance_feature with anchor_embed
        anchor_embed = self._compute_anchor_embed(anchor)            # (N, E)
        fused = torch.cat([instance_feature, anchor_embed], dim=-1)  # (N, 2E)
        fused = self.anchor_fusion(fused)                            # (N, E)
        logits = self.decision_net(fused)                            # (N, 3)

        if self.training:
            # Gumbel-Softmax with hard=False: continuous values, fully differentiable
            # Return logits alongside soft_assign so that training TopK boosting
            # uses the SAME logits without a duplicate no_grad forward.
            soft_assign = F.gumbel_softmax(
                logits, tau=self.gumbel_tau, hard=False, dim=-1
            )  # (N, 3), continuous values in [0,1], sum to 1
            return soft_assign, logits
        else:
            probs = F.softmax(logits, dim=-1)            # (N, 3)
            decision_label = torch.argmax(probs, dim=-1) # (N,)
            return decision_label, probs, logits

    # ==================================================================
    #  T3: Training TopK-aware Probability Boosting
    # ==================================================================

    def _training_topk_probability_boost(
        self,
        soft_assign: torch.Tensor,
        logits: torch.Tensor,
        topk_clone: int,
        topk_split: int,
    ):
        """
        Apply TopK-aware probability boosting during training.

        This modifies routing probabilities to ensure clone/split branches
        receive stable activation during training. Uses residual boosting
        instead of hard masking to preserve differentiability of the main
        gradient path.

        Clone has priority: clone TopK positions are excluded from split selection.

        Args:
            soft_assign: (N, 3)  raw soft assignments from Gumbel-Softmax
            logits:      (N, 3)  raw logits from decision_net (for scoring)
            topk_clone:  int     minimum clone candidates to boost
            topk_split:  int     minimum split candidates to boost

        Returns:
            soft_assign_final: (N, 3)  boosted and renormalized probabilities
        """
        p_keep  = soft_assign[..., 0]  # (N,)
        p_clone = soft_assign[..., 1]  # (N,)
        p_split = soft_assign[..., 2]  # (N,)

        N = p_keep.shape[0]

        clone_score = logits[..., 1]  # (N,)
        split_score = logits[..., 2]  # (N,)

        topk_clone = min(topk_clone, N)
        topk_split = min(topk_split, max(N - topk_clone, 0))

        clone_topk_mask = torch.zeros_like(p_clone, dtype=torch.bool)
        split_topk_mask = torch.zeros_like(p_split, dtype=torch.bool)

        # Step 1: clone TopK, clone has priority
        if topk_clone > 0:
            _, clone_idx = torch.topk(clone_score, k=topk_clone, dim=0)
            clone_topk_mask[clone_idx] = True

        # Step 2: split TopK from remaining candidates (exclude clone TopK)
        if topk_split > 0:
            split_score_masked = split_score.masked_fill(
                clone_topk_mask, -float("inf")
            )
            _, split_idx = torch.topk(split_score_masked, k=topk_split, dim=0)
            split_topk_mask[split_idx] = True

        # Step 3: residual probability boosting
        alpha = self.train_topk_boost_alpha

        p_clone_boosted = p_clone + alpha * clone_topk_mask.float() * (1.0 - p_clone)
        p_split_boosted = p_split + alpha * split_topk_mask.float() * (1.0 - p_split)

        # Step 4: renormalize probabilities
        eps = 1e-6
        p_sum = p_keep + p_clone_boosted + p_split_boosted + eps

        p_keep_final  = p_keep / p_sum
        p_clone_final = p_clone_boosted / p_sum
        p_split_final = p_split_boosted / p_sum

        soft_assign_final = torch.stack(
            [p_keep_final, p_clone_final, p_split_final],
            dim=-1
        )

        return soft_assign_final

    # ==================================================================
    #  T1: Build All Candidate Gaussians
    # ==================================================================

    def _build_all_candidates(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Build all candidate Gaussians for differentiable training.

        Constructs four candidate sets per original Gaussian:
            original:      original Gaussian (used by keep and clone)
            clone_new:     generated by clone branch
            split_child1:  generated by split branch
            split_child2:  generated by split branch

        No cross-branch attribute interpolation is performed.

        Args:
            instance_feature: (N, E)  instance features
            anchor:           (N, A)  anchor tensor
            gaussian:         GaussianPrediction with fields (N, *)

        Returns:
            candidates: dict with keys:
                "orig":  (orig_feat,  orig_anchor,  orig_g)
                "clone": (clone_feat, clone_anchor, clone_g)
                "c1":    (c1_feat,    c1_anchor,    c1_g)
                "c2":    (c2_feat,    c2_anchor,    c2_g)
        """
        # Original candidate — identity, kept as-is
        orig_feat = instance_feature
        orig_anchor = anchor
        orig_g = gaussian

        # Clone candidate
        clone_feat, clone_anchor, clone_g = self._clone_branch(
            instance_feature, gaussian, anchor
        )

        # Split candidates
        c1_feat, c1_anchor, c1_g, c2_feat, c2_anchor, c2_g = \
            self._split_branch(instance_feature, gaussian, anchor)

        return {
            "orig":  (orig_feat,  orig_anchor,  orig_g),
            "clone": (clone_feat, clone_anchor, clone_g),
            "c1":    (c1_feat,    c1_anchor,    c1_g),
            "c2":    (c2_feat,    c2_anchor,    c2_g),
        }

    # ==================================================================
    #  T2: Apply Opacity Weights
    # ==================================================================

    def _apply_opacity_weights(
        self,
        candidates: dict,
        soft_assign: torch.Tensor,
    ):
        """
        Apply routing probabilities to candidate Gaussian opacities only.

        Weight mapping:
            w_orig  = p_keep + p_clone  (clone retains original)
            w_clone = p_clone
            w_c1    = p_split
            w_c2    = p_split

        Args:
            candidates:  dict returned by _build_all_candidates
            soft_assign: (N, 3)  columns are [p_keep, p_clone, p_split]

        Returns:
            weighted_candidates: dict with same structure, opacities modulated
        """
        p_keep  = soft_assign[..., 0]  # (N,)
        p_clone = soft_assign[..., 1]  # (N,)
        p_split = soft_assign[..., 2]  # (N,)

        w_orig  = (p_keep + p_clone).unsqueeze(-1)  # (N, 1)
        w_clone = p_clone.unsqueeze(-1)              # (N, 1)
        w_c1    = p_split.unsqueeze(-1)              # (N, 1)
        w_c2    = p_split.unsqueeze(-1)              # (N, 1)

        orig_feat, orig_anchor, orig_g = candidates["orig"]
        clone_feat, clone_anchor, clone_g = candidates["clone"]
        c1_feat, c1_anchor, c1_g = candidates["c1"]
        c2_feat, c2_anchor, c2_g = candidates["c2"]

        # Modulate only opacity — all other attributes stay unchanged
        orig_g = orig_g._replace(
            opacities=orig_g.opacities * w_orig
        )
        clone_g = clone_g._replace(
            opacities=clone_g.opacities * w_clone
        )
        c1_g = c1_g._replace(
            opacities=c1_g.opacities * w_c1
        )
        c2_g = c2_g._replace(
            opacities=c2_g.opacities * w_c2
        )

        return {
            "orig":  (orig_feat,  orig_anchor,  orig_g),
            "clone": (clone_feat, clone_anchor, clone_g),
            "c1":    (c1_feat,    c1_anchor,    c1_g),
            "c2":    (c2_feat,    c2_anchor,    c2_g),
        }

    # ==================================================================
    #  T1: Concatenate All Candidates into 4N Output
    # ==================================================================

    def _concat_candidates(self, candidates: dict):
        """
        Concatenate weighted candidate Gaussians into 4N output tensors.

        Order: [original, clone_new, split_child1, split_child2]

        Args:
            candidates: dict with keys "orig", "clone", "c1", "c2"
                        each value is (feat, anchor, gaussian)

        Returns:
            result_anchor:   (4N, A)
            result_gaussian: GaussianPrediction (4N, *)
            result_feature:  (4N, E)
        """
        orig_feat, orig_anchor, orig_g = candidates["orig"]
        clone_feat, clone_anchor, clone_g = candidates["clone"]
        c1_feat, c1_anchor, c1_g = candidates["c1"]
        c2_feat, c2_anchor, c2_g = candidates["c2"]

        result_feature = torch.cat(
            [orig_feat, clone_feat, c1_feat, c2_feat],
            dim=0
        )

        result_anchor = torch.cat(
            [orig_anchor, clone_anchor, c1_anchor, c2_anchor],
            dim=0
        )

        result_gaussian = GaussianPrediction(
            means=torch.cat(
                [orig_g.means, clone_g.means, c1_g.means, c2_g.means],
                dim=0
            ),
            scales=torch.cat(
                [orig_g.scales, clone_g.scales, c1_g.scales, c2_g.scales],
                dim=0
            ),
            rotations=torch.cat(
                [orig_g.rotations, clone_g.rotations, c1_g.rotations, c2_g.rotations],
                dim=0
            ),
            opacities=torch.cat(
                [orig_g.opacities, clone_g.opacities, c1_g.opacities, c2_g.opacities],
                dim=0
            ),
            semantics=torch.cat(
                [orig_g.semantics, clone_g.semantics, c1_g.semantics, c2_g.semantics],
                dim=0
            ),
        )

        return result_anchor, result_gaussian, result_feature

    # ==================================================================
    #  Main Forward: dispatches to soft (training) or hard (inference)
    # ==================================================================

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Forward pass of AdaptiveAllocationV3.

        Args:
            instance_feature: (B, N, E)
            anchor:           (B, N, A)
            gaussian:         GaussianPrediction with fields (B, N, *)

        Returns:
            result_anchor:   (B, M, A)
            result_gaussian: GaussianPrediction (B, M, *)
            result_features: (B, M, E)
                where M = 4N (training) or N + N_clone + N_split (inference)
        """
        if self.training:
            return self._soft_candidate_forward(instance_feature, anchor, gaussian)
        else:
            return self._hard_routing_forward(instance_feature, anchor, gaussian)

    # ==================================================================
    #  T4+T5: Training Forward — Opacity-Weighted Candidate Routing
    # ==================================================================

    def _soft_candidate_forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Training forward with opacity-weighted candidate routing.

        Unlike V2, this method does NOT interpolate Gaussian attributes.
        It constructs all candidates and uses decision probabilities only
        to modulate opacity contribution.

        Training TopK fallback is applied before opacity weighting.

        Args:
            instance_feature: (B, N, E)
            anchor:           (B, N, A)
            gaussian:         GaussianPrediction with fields (B, N, *)

        Returns:
            result_anchor:   (B, 4N, A)
            result_gaussian: GaussianPrediction (B, 4N, *)
            result_features: (B, 4N, E)
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
            cur_anchor = anchor[b]               # (N, A)
            cur_opa = gaussian.opacities[b]      # (N, 1)
            cur_means = gaussian.means[b]        # (N, 3)
            cur_scales = gaussian.scales[b]      # (N, 3)
            cur_rots = gaussian.rotations[b]     # (N, 4)
            cur_sems = gaussian.semantics[b]     # (N, S)

            # --- Step 1: Compute soft assignments + logits from single forward ---
            soft_assign, logits = self._compute_decisions(
                cur_feats, cur_anchor)  # (N, 3), (N, 3)

            # --- Step 2: Training TopK-aware probability boosting ---
            if self.use_train_topk_fallback:
                soft_assign = self._training_topk_probability_boost(
                    soft_assign=soft_assign,
                    logits=logits,
                    topk_clone=self.topk_clone,
                    topk_split=self.topk_split,
                )

            # --- Step 3: Build all candidates ---
            cur_g = GaussianPrediction(
                means=cur_means,
                scales=cur_scales,
                rotations=cur_rots,
                opacities=cur_opa,
                semantics=cur_sems,
            )
            candidates = self._build_all_candidates(
                instance_feature=cur_feats,
                anchor=cur_anchor,
                gaussian=cur_g,
            )

            # --- Step 4: Apply opacity weights ---
            weighted_candidates = self._apply_opacity_weights(
                candidates=candidates,
                soft_assign=soft_assign,
            )

            # --- Step 5: Concatenate into 4N output ---
            result_anchor_b, result_g, result_feat = self._concat_candidates(
                weighted_candidates
            )

            result_features_list.append(result_feat)
            result_anchors_list.append(result_anchor_b)
            result_opacities_list.append(result_g.opacities)
            result_semantics_list.append(result_g.semantics)
            result_rotations_list.append(result_g.rotations)
            result_means_list.append(result_g.means)
            result_scales_list.append(result_g.scales)

        # Stack across batch: training always produces 4N, so sizes are uniform
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
    #  T4+T5: Inference Forward — Hard Routing + TopK Fallback
    # ==================================================================

    def _hard_routing_forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """
        Inference forward with hard routing.

        Semantics:
            keep  → original
            clone → original + clone_new
            split → split_child1 + split_child2 (original discarded)

        Uses TopK fallback to guarantee minimum clone/split counts.

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

            # ---- Step 1: Compute decisions (inference: hard labels + soft probs + logits) ----
            decision_label, soft_probs, logits = self._compute_decisions(
                cur_feats, cur_anchor)  # (N,), (N,3), (N,3)

            # ---- Step 2: TopK fallback with mutual exclusion ----
            clone_mask, split_mask = self._ensure_min_densify(
                decision_label, logits, self.topk_clone, self.topk_split)

            keep_mask = ~(clone_mask | split_mask)

            # ---- Collect output containers ----
            output_feats = []
            output_anchors = []
            output_gaussians = []

            # ---- Keep: output original ----
            if keep_mask.any():
                keep_idx = torch.where(keep_mask)[0]
                keep_feat = cur_feats[keep_idx]
                keep_anchor_b = cur_anchor[keep_idx]
                keep_g = GaussianPrediction(
                    means=cur_means[keep_idx],
                    scales=cur_scales[keep_idx],
                    rotations=cur_rots[keep_idx],
                    opacities=cur_opa[keep_idx],
                    semantics=cur_sems[keep_idx],
                )
                output_feats.append(keep_feat)
                output_anchors.append(keep_anchor_b)
                output_gaussians.append(keep_g)

            # ---- Clone: output original + clone_new ----
            if clone_mask.any():
                clone_idx = torch.where(clone_mask)[0]
                clone_feats = cur_feats[clone_idx]

                clone_g = GaussianPrediction(
                    means=cur_means[clone_idx],
                    scales=cur_scales[clone_idx],
                    rotations=cur_rots[clone_idx],
                    opacities=cur_opa[clone_idx],
                    semantics=cur_sems[clone_idx],
                )
                clone_a = cur_anchor[clone_idx]

                # original (kept for clone)
                output_feats.append(clone_feats)
                output_anchors.append(clone_a)
                output_gaussians.append(clone_g)

                # clone_new
                c_proj, c_anchor, c_g = self._clone_branch(
                    clone_feats, clone_g, clone_a)

                # Hard routing: no soft score opacity scaling on new gaussians.
                # clone_new opacity is used as-is from the clone branch output.
                output_feats.append(c_proj)
                output_anchors.append(c_anchor)
                output_gaussians.append(c_g)

            # ---- Split: output split_child1 + split_child2 ----
            if split_mask.any():
                split_idx = torch.where(split_mask)[0]
                split_feats = cur_feats[split_idx]

                split_g = GaussianPrediction(
                    means=cur_means[split_idx],
                    scales=cur_scales[split_idx],
                    rotations=cur_rots[split_idx],
                    opacities=cur_opa[split_idx],
                    semantics=cur_sems[split_idx],
                )
                split_a = cur_anchor[split_idx]

                c1_feat, c1_anchor, c1_g, c2_feat, c2_anchor, c2_g = \
                    self._split_branch(split_feats, split_g, split_a)

                # Hard routing: no soft score opacity scaling on new gaussians.
                # split children opacity is used as-is from the split branch output.
                output_feats.extend([c1_feat, c2_feat])
                output_anchors.extend([c1_anchor, c2_anchor])
                output_gaussians.extend([c1_g, c2_g])

            # ---- Assemble results ----
            result_feat = torch.cat(output_feats, dim=0)
            result_anchor_b = torch.cat(output_anchors, dim=0)

            result_features_list.append(result_feat)
            result_anchors_list.append(result_anchor_b)

            # Collect gaussian fields
            result_opacities_list.append(
                torch.cat([g.opacities for g in output_gaussians], dim=0))
            result_semantics_list.append(
                torch.cat([g.semantics for g in output_gaussians], dim=0))
            result_rotations_list.append(
                torch.cat([g.rotations for g in output_gaussians], dim=0))
            result_means_list.append(
                torch.cat([g.means for g in output_gaussians], dim=0))
            result_scales_list.append(
                torch.cat([g.scales for g in output_gaussians], dim=0))

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
    #  T5: TopK Fallback with Mutual Exclusion (inference only)
    # ==================================================================

    def _ensure_min_densify(self, decision_label, logits, topk_clone, topk_split):
        """
        Ensure minimum clone/split count via TopK fallback with mutual exclusion.

        Clone has priority and may override existing split assignments when
        the fallback shortage cannot be satisfied from keep candidates alone.
        This guarantees strict mutual exclusion even in edge cases.

        Args:
            decision_label: (N,)  hard labels {0:keep, 1:clone, 2:split}
            logits:         (N, 3) raw decision_net logits
            topk_clone:     int   minimum clone count
            topk_split:     int   minimum split count

        Returns:
            clone_mask: (N,) bool, >= topk_clone True values
            split_mask: (N,) bool, >= topk_split True values, exclusive with clone
        """
        N = decision_label.shape[0]

        clone_mask = (decision_label == 1)
        split_mask = (decision_label == 2)

        # ---- Clone fallback: clone has priority and may override split ----
        clone_shortage = max(topk_clone - clone_mask.sum().item(), 0)
        # Available candidates: everything except already-assigned clone
        clone_shortage = min(clone_shortage, N - clone_mask.sum().item())

        if clone_shortage > 0:
            clone_score = logits[:, 1].clone()
            clone_score[clone_mask] = -float("inf")
            clone_idx = torch.topk(clone_score, clone_shortage, dim=-1).indices
            # Explicit override: guarantee clone/split exclusiveness
            clone_mask[clone_idx] = True
            split_mask[clone_idx] = False

        # ---- Split fallback: select from non-clone candidates only ----
        split_shortage = max(topk_split - split_mask.sum().item(), 0)
        available_for_split = ~clone_mask
        split_shortage = min(split_shortage, available_for_split.sum().item())

        if split_shortage > 0:
            split_score = logits[:, 2].clone()
            split_score[~available_for_split] = -float("inf")
            split_idx = torch.topk(split_score, split_shortage, dim=-1).indices
            split_mask[split_idx] = True

        return clone_mask, split_mask

    # ==================================================================
    #  Clone Branch (same as V2)
    # ==================================================================

    def _clone_branch(self, clone_feats, clone_gaussian, clone_anchor):
        """
        Process clone-selected gaussians. Same logic as V2/DensifyOnly.

        Args:
            clone_feats:    (N_c, E)  instance features
            clone_gaussian: GaussianPrediction fields, each (N_c, *)
            clone_anchor:   (N_c, A)  anchor

        Returns:
            new_feats:   (N_c, E)
            new_anchor:  (N_c, A)
            new_gaussian: GaussianPrediction
        """
        projected = self.clone_feature_proj(clone_feats)  # (N_c, E)

        # Location shift (restricted)
        delta_xyz_sigmoid = self.clone_location_shift(projected)
        delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1  # [-1, 1]
        mean_shift_out = torch.stack([
            delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
            delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
            delta_xyz_prob[..., 2] * self.unit_sigmoid[2],
        ], dim=-1)

        # Other attribute shifts
        scale_shift_out = self.clone_scale_shift(projected)
        rots_shift_out = F.normalize(self.clone_rotation_shift(projected))
        sem_shift_out = self.clone_semantic_shift(projected)
        opa_shift_out = self.clone_opacities_shift(projected)

        # Residual refinement
        means_shift = (safe_sigmoid(mean_shift_out) - 0.5) * 2    # [-1, 1]
        scale_shift = safe_sigmoid(scale_shift_out)                # [0, 1]
        rots_shift = rots_shift_out
        sem_shift = safe_sigmoid(sem_shift_out)
        opa_shift = safe_sigmoid(opa_shift_out)

        # New means
        m = self.boundary_margin
        new_means = clone_gaussian.means + means_shift * clone_gaussian.scales
        new_means_x = torch.clamp(
            new_means[:, 0], self.pc_range[0] + m, self.pc_range[3] - m)
        new_means_y = torch.clamp(
            new_means[:, 1], self.pc_range[1] + m, self.pc_range[4] - m)
        new_means_z = torch.clamp(
            new_means[:, 2], self.pc_range[2] + m, self.pc_range[5] - m)
        new_means = torch.stack([new_means_x, new_means_y, new_means_z], dim=-1)
        new_xyz_anchor = safe_sigmoid(new_means)

        # New scales
        new_scales = clone_gaussian.scales * scale_shift
        new_scales = torch.clamp(new_scales, self.scale_range[0], self.scale_range[1])

        # New rotations / semantics / opacities: average blend
        # Normalize rotation to maintain unit quaternion constraint
        new_rots = F.normalize(clone_gaussian.rotations + rots_shift, dim=-1)
        new_sems = clone_gaussian.semantics / 2 + sem_shift / 2
        new_opas = clone_gaussian.opacities / 2 + opa_shift / 2

        # Anchor update
        new_anchor_scale = clone_anchor[:, 3:6] * scale_shift
        new_anchor_rot = F.normalize(clone_anchor[:, 6:10] + rots_shift_out, dim=-1)
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
    #  Split Branch (same as V2)
    # ==================================================================

    def _split_branch(self, split_feats, split_gaussian, split_anchor):
        """
        Process split-selected gaussians. Generates 2 children,
        original parent is discarded.

        Args:
            split_feats:    (N_s, E)
            split_gaussian: GaussianPrediction fields, each (N_s, *)
            split_anchor:   (N_s, A)

        Returns:
            c1_feat, c1_anchor, c1_gaussian, c2_feat, c2_anchor, c2_gaussian
        """
        device = split_feats.device
        parent_means = split_gaussian.means
        parent_scales = split_gaussian.scales
        parent_rots = split_gaussian.rotations
        parent_sems = split_gaussian.semantics
        parent_opas = split_gaussian.opacities

        projected = self.split_feature_proj(split_feats)

        # Child 1 predictions
        c1_loc = self.c1_location_shift(projected)
        c1_scale = self.c1_scale_shift(projected)
        c1_rot = self.c1_rotation_shift(projected)
        c1_sem = self.c1_semantic_shift(projected)
        c1_opa = self.c1_opacities_shift(projected)
        c1_feat = self.c1_feature_transform(projected)

        # Child 2 predictions
        c2_loc = self.c2_location_shift(projected)
        c2_scale = self.c2_scale_shift(projected)
        c2_rot = self.c2_rotation_shift(projected)
        c2_sem = self.c2_semantic_shift(projected)
        c2_opa = self.c2_opacities_shift(projected)
        c2_feat = self.c2_feature_transform(projected)

        # Compute child geometry based on split_mode
        if self.split_mode == "constrained":
            c1_means, c2_means, c1_scales, c2_scales, \
                c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas = \
                self._split_constrained(
                    parent_means, parent_scales, parent_rots,
                    parent_sems, parent_opas,
                    c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
                    c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
                    device)
        else:
            c1_means, c2_means, c1_scales, c2_scales, \
                c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas = \
                self._split_free(
                    parent_means, parent_scales, parent_rots,
                    parent_sems, parent_opas,
                    c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
                    c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
                    device)

        # Boundary clamping
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

        # Anchor conversion
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
    #  Split Mode: Constrained (same as V2)
    # ==================================================================

    def _split_constrained(
        self,
        parent_means, parent_scales, parent_rots, parent_sems, parent_opas,
        c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
        c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
        device,
    ):
        split_axis = torch.argmax(parent_scales, dim=-1)
        direction = F.one_hot(split_axis, num_classes=3).float()

        offset_mag = F.softplus(c1_loc.sum(dim=-1, keepdim=True))

        c1_means = parent_means + direction * offset_mag * parent_scales
        c2_means = parent_means - direction * offset_mag * parent_scales

        c1_scale_factor = 0.3 + 0.4 * torch.sigmoid(c1_scale.sum(dim=-1, keepdim=True))
        c2_scale_factor = 0.3 + 0.4 * torch.sigmoid(c2_scale.sum(dim=-1, keepdim=True))
        c1_scales = parent_scales * c1_scale_factor
        c2_scales = parent_scales * c2_scale_factor

        c1_rots = F.normalize(parent_rots + c1_rot * 0.1)
        c2_rots = F.normalize(parent_rots + c2_rot * 0.1)

        c1_sems = parent_sems + torch.tanh(c1_sem) * 0.1
        c2_sems = parent_sems + torch.tanh(c2_sem) * 0.1
        c1_opas = parent_opas + torch.tanh(c1_opa) * 0.1
        c2_opas = parent_opas + torch.tanh(c2_opa) * 0.1

        return c1_means, c2_means, c1_scales, c2_scales, \
            c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas

    # ==================================================================
    #  Split Mode: Free (same as V2)
    # ==================================================================

    def _split_free(
        self,
        parent_means, parent_scales, parent_rots, parent_sems, parent_opas,
        c1_loc, c1_scale, c1_rot, c1_sem, c1_opa,
        c2_loc, c2_scale, c2_rot, c2_sem, c2_opa,
        device,
    ):
        pc_span = torch.tensor(self.pc_span, device=device, dtype=parent_means.dtype)

        delta1 = torch.tanh(c1_loc) * pc_span * 0.5
        delta2 = torch.tanh(c2_loc) * pc_span * 0.5
        c1_means = parent_means + delta1
        c2_means = parent_means + delta2

        c1_scales = F.softplus(c1_scale)
        c2_scales = F.softplus(c2_scale)

        c1_rots = F.normalize(c1_rot)
        c2_rots = F.normalize(c2_rot)

        c1_sems = torch.sigmoid(c1_sem)
        c2_sems = torch.sigmoid(c2_sem)
        c1_opas = torch.sigmoid(c1_opa)
        c2_opas = torch.sigmoid(c2_opa)

        return c1_means, c2_means, c1_scales, c2_scales, \
            c1_rots, c2_rots, c1_sems, c2_sems, c1_opas, c2_opas

    # ==================================================================
    #  Utility: pad variable-length tensors (same as V2)
    # ==================================================================

    @staticmethod
    def _pad_and_stack(tensor_list, max_size, fill_value=0.0):
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
        padded = []
        for t in tensor_list:
            if t.shape[0] < max_size:
                pad = torch.zeros(
                    (max_size - t.shape[0], t.shape[1]),
                    device=t.device, dtype=t.dtype)
                pad[:, 0] = 1.0  # identity quaternion: w=1
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)
