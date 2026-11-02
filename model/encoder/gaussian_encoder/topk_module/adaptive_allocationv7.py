"""Fixed-budget shared Gaussian growth. No operation experts or opacity router.

Terminal use exposes additive replacement provenance for incremental rendering.
The optional training probes are geometry proposals, never extra rendered GS.
"""

import math

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import nn
from torch.nn import functional as F

from ..utils import GaussianPrediction
from .adaptive_allocationv6 import get_max_scale_axis

GAUSSIAN_FIELDS = ("means", "scales", "rotations", "opacities", "semantics")


def gather_gaussian(gaussian, index):
    """Gather the same [B,K] parent indices from all five Gaussian fields."""

    def gather(value):
        expanded_index = index[..., None].expand(-1, -1, value.shape[-1])
        return value.gather(1, expanded_index)

    return GaussianPrediction(
        **{name: gather(getattr(gaussian, name)) for name in GAUSSIAN_FIELDS}
    )


def concatenate_gaussians(*items):
    """Concatenate Gaussian sets along their parent/child dimension."""
    return GaussianPrediction(
        **{
            name: torch.cat([getattr(gaussian, name) for gaussian in items], 1)
            for name in GAUSSIAN_FIELDS
        }
    )


def inverse_softplus(value):
    value = value.clamp_min(1e-6)
    return value + torch.log(-torch.expm1(-value))


@MODELS.register_module()
class AdaptiveAllocationV7(BaseModule):
    """Select K parents, then predict one child and an optional parent residual.

    Features have shape [B,N,C]; each Gaussian field has shape [B,N,D]. The
    growth quota is applied before generation. Scores never multiply opacity.
    Set update_parent=False for the append-only ablation. Inference does not
    construct supervision probes.
    """

    def __init__(
        self,
        feat_embed_dim=128,
        semantic_dim=17,
        hidden_dim=128,
        pc_range=None,
        scale_range=None,
        grid_size=0.4,
        allocation_ratio=0.15,
        max_new_gaussians=None,
        selection_mode="learned",
        exploration_fraction=0.1,
        exploration_steps=1000,
        probe_count=128,
        update_parent=True,
        child_opacity_init=0.02,
        child_offset_init=0.25,
        parent_offset_limit=0.25,
        child_offset_limit=1.0,
        log_scale_limit=0.5,
        semantic_delta_limit=3.0,
        semantics_activation="softplus",
        **kwargs,
    ):
        super().__init__()
        if pc_range is None or scale_range is None:
            raise ValueError("AdaptiveAllocationV7 requires pc_range and scale_range")
        if not 0 <= allocation_ratio <= 1 or not 0 <= exploration_fraction <= 1:
            raise ValueError("Growth and exploration fractions must be in [0,1]")
        if selection_mode not in ("learned", "opacity", "random"):
            raise ValueError("selection_mode must be learned, opacity or random")
        if not 0 < child_opacity_init < 1 or child_offset_init <= 0:
            raise ValueError("Child opacity and spatial offset must start nonzero")
        if grid_size <= 0 or probe_count < 0 or exploration_steps < 0:
            raise ValueError("Invalid grid size, probe count or exploration duration")
        if max_new_gaussians is not None and max_new_gaussians < 0:
            raise ValueError("max_new_gaussians must be nonnegative")
        if semantics_activation not in ("softplus", "identity"):
            raise ValueError("V7 supports softplus or identity semantic fields")

        self.pc_range = tuple(pc_range)
        self.scale_range = tuple(scale_range)
        self.feat_embed_dim = feat_embed_dim
        self.semantic_dim = semantic_dim
        self.grid_size = float(grid_size)
        self.allocation_ratio = float(allocation_ratio)
        self.max_new_gaussians = max_new_gaussians
        self.selection_mode = selection_mode
        self.exploration_fraction = exploration_fraction
        self.exploration_steps = exploration_steps
        self.probe_count = int(probe_count)
        self.update_parent = bool(update_parent)
        self.child_opacity_init = child_opacity_init
        self.child_offset_init = child_offset_init
        self.parent_offset_limit = parent_offset_limit
        self.child_offset_limit = child_offset_limit
        self.log_scale_limit = log_scale_limit
        self.semantic_delta_limit = semantic_delta_limit
        self.semantics_activation = semantics_activation

        # Keep these module names/order stable for existing checkpoints and RNG.
        self.state_encoder = nn.Sequential(
            nn.Linear(feat_embed_dim + 10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gain_head = nn.Linear(hidden_dim, 1)
        # Each output packs xyz(3), log-scale(3), opacity(1), semantics(S).
        # Rotation is inherited; all selected parents share this generator.
        self.parameter_dim = 7 + semantic_dim
        self.generator = nn.Linear(
            hidden_dim, self.parameter_dim * (2 if update_parent else 1)
        )
        self.register_buffer("allocation_step", torch.zeros((), dtype=torch.long))
        self._allocation_aux = None
        self.init_weight()

    def init_weight(self):
        # GaussianOccEncoder invokes this after its generic Xavier pass.
        nn.init.normal_(self.gain_head.weight, std=0.001)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

    def _state(self, features, gaussian):
        """Encode features plus detached geometry and semantic statistics."""
        with torch.no_grad():
            minimum = gaussian.means.new_tensor(self.pc_range[:3])
            extent = gaussian.means.new_tensor(self.pc_range[3:]) - minimum
            normalized_xyz = 2 * (gaussian.means - minimum) / extent - 1
            probabilities = gaussian.semantics.float().softmax(-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
                -1, keepdim=True
            )
            top_probabilities = probabilities.topk(min(2, self.semantic_dim), -1).values
            margin = top_probabilities[..., :1] - top_probabilities[..., -1:]
            cues = torch.cat(
                [
                    normalized_xyz,
                    gaussian.scales.clamp_min(1e-6).log(),
                    gaussian.opacities,
                    entropy,
                    margin,
                    top_probabilities[..., :1],
                ],
                -1,
            )
        return self.state_encoder(torch.cat([features, cues.to(features.dtype)], -1))

    def _selection_priority(self, score, gaussian):
        if self.selection_mode == "opacity":
            return gaussian.opacities.squeeze(-1)
        if self.selection_mode == "random" and self.training:
            return torch.rand_like(score)
        if self.selection_mode == "random":
            # Repeatable, spatially uninformative control; no inference RNG.
            index = torch.arange(
                score.shape[1], device=score.device, dtype=torch.float64
            )
            priority = torch.frac(torch.sin(index * 12.9898 + 78.233) * 43758.5453)
            return priority[None].expand_as(score)
        return score

    def _select(self, score, gaussian, count, step):
        """Fill the exact quota with ranked parents and optional exploration."""
        # Compute priority before the empty-budget check to preserve RNG draws.
        priority = self._selection_priority(score, gaussian)
        if not count:
            return torch.empty(score.shape[0], 0, device=score.device, dtype=torch.long)

        fraction = self.exploration_fraction * max(
            0.0, 1.0 - step / max(1, self.exploration_steps)
        )
        explore_count = (
            int(count * fraction)
            if self.training
            and self.exploration_steps > 0
            and self.selection_mode == "learned"
            else 0
        )
        selected = priority.topk(count - explore_count, 1).indices
        if explore_count:
            random_priority = torch.rand_like(score).scatter(1, selected, -1.0)
            explored = random_priority.topk(explore_count, 1).indices
            selected = torch.cat([selected, explored], 1)
        return selected

    def _semantic(self, semantic, residual):
        delta = self.semantic_delta_limit * residual.tanh()
        if self.semantics_activation == "softplus":
            return F.softplus(inverse_softplus(semantic) + delta)
        return semantic + delta

    def _clamp_means(self, means):
        minimum = means.new_tensor(self.pc_range[:3]) + 1e-5
        maximum = means.new_tensor(self.pc_range[3:]) - 1e-5
        return torch.maximum(minimum, torch.minimum(maximum, means))

    def _predict_child(self, parent, residual):
        axis = get_max_scale_axis(parent.rotations, parent.scales)
        initial_offset = (
            self.child_offset_init * axis * parent.scales.amax(-1, keepdim=True)
        )
        learned_offset = (
            self.child_offset_limit * parent.scales * residual[..., :3].tanh()
        )
        return GaussianPrediction(
            means=self._clamp_means(parent.means + initial_offset + learned_offset),
            scales=(
                parent.scales
                * torch.exp(self.log_scale_limit * residual[..., 3:6].tanh())
            ).clamp(*self.scale_range),
            rotations=parent.rotations,
            opacities=parent.opacities
            * torch.sigmoid(
                residual[..., 6:7]
                + math.log(self.child_opacity_init / (1 - self.child_opacity_init))
            ),
            semantics=self._semantic(parent.semantics, residual[..., 7:]),
        )

    def _predict_parent(self, parent, residual):
        opacity_delta = residual[..., 6:7].tanh()
        # Exact identity at zero, including extreme parent opacities.
        opacity = parent.opacities + opacity_delta * torch.where(
            opacity_delta >= 0, 1 - parent.opacities, parent.opacities
        )
        return GaussianPrediction(
            means=self._clamp_means(
                parent.means
                + self.parent_offset_limit * parent.scales * residual[..., :3].tanh()
            ),
            scales=(
                parent.scales
                * torch.exp(self.log_scale_limit * residual[..., 3:6].tanh())
            ).clamp(*self.scale_range),
            rotations=parent.rotations,
            opacities=opacity,
            semantics=self._semantic(parent.semantics, residual[..., 7:]),
        )

    def _predict(self, hidden, parent):
        raw = self.generator(hidden)
        child = self._predict_child(parent, raw[..., -self.parameter_dim :])
        updated = (
            self._predict_parent(parent, raw[..., : self.parameter_dim])
            if self.update_parent
            else parent
        )
        return updated, child

    @staticmethod
    def _gather_hidden(hidden, index):
        return hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))

    def _replace_parents(self, gaussian, index, updated):
        fields = {}
        for name in GAUSSIAN_FIELDS:
            value = getattr(gaussian, name)
            expanded_index = index[..., None].expand(-1, -1, value.shape[-1])
            fields[name] = value.scatter(1, expanded_index, getattr(updated, name))
        return GaussianPrediction(**fields)

    def _current_step(self):
        # Read the CUDA buffer only once after construction/checkpoint loading.
        step = getattr(self, "_step_value", None)
        if step is None:
            step = int(self.allocation_step.cpu())
            self._step_value = step
        return step

    def _training_aux(
        self, hidden, score, gaussian, index, updated, child, parent, step
    ):
        """Propose detached teacher probes; only probe scores retain gradients."""
        count = hidden.shape[1]
        budget = index.shape[1]
        probe_count = min(self.probe_count, count)
        probe_index = torch.rand_like(score).topk(probe_count, 1).indices
        with torch.no_grad():
            probe_parent = gather_gaussian(gaussian, probe_index)
            probe_hidden = self._gather_hidden(hidden.detach(), probe_index)
            probe_updated, probe_child = self._predict(probe_hidden, probe_parent)

        return dict(
            version=7,
            needs_base_logits=True,
            probe_index=probe_index,
            probe_score=score.gather(1, probe_index),
            probe_parent=probe_parent,
            probe_updated=probe_updated,
            probe_child=probe_child,
            selected_index=index,
            step=step,
            input_count=count,
            output_count=count + budget,
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            parent_displacement=(
                (updated.means - parent.means).square().mean()
                if budget
                else score.sum() * 0
            ),
            child_opacity=child.opacities.mean() if budget else score.sum() * 0,
        )

    def _run(self, features, gaussian):
        self._allocation_aux = None
        _, count, _ = features.shape
        hidden = self._state(features, gaussian)
        score = self.gain_head(hidden).squeeze(-1)
        budget = int(count * self.allocation_ratio)
        if self.max_new_gaussians is not None:
            budget = min(budget, self.max_new_gaussians)

        training = self.training and torch.is_grad_enabled()
        step = self._current_step()
        index = self._select(score, gaussian, budget, step)
        parent = gather_gaussian(gaussian, index)
        selected_hidden = self._gather_hidden(hidden, index)
        updated, child = self._predict(selected_hidden, parent)
        parents = (
            self._replace_parents(gaussian, index, updated)
            if self.update_parent
            else gaussian
        )
        result = concatenate_gaussians(parents, child)

        if training:
            self._allocation_aux = self._training_aux(
                hidden, score, gaussian, index, updated, child, parent, step
            )
            self.allocation_step.add_(1)
            self._step_value = step + 1
        return result, index, updated, child, parent

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._step_value = None
        self._allocation_aux = None

    def get_allocation_aux(self):
        return self._allocation_aux

    def forward_terminal(self, features, anchor, gaussian):
        result, _, updated, child, parent = self._run(features, gaussian)
        return dict(
            gaussian=result,
            delta_gaussian=(
                concatenate_gaussians(updated, child) if self.update_parent else child
            ),
            replaced_gaussian=parent if self.update_parent else None,
        )

    def _build_anchor(self, gaussian):
        minimum = gaussian.means.new_tensor(self.pc_range[:3])
        maximum = gaussian.means.new_tensor(self.pc_range[3:])
        xyz = ((gaussian.means - minimum) / (maximum - minimum)).clamp(1e-6, 1 - 1e-6)
        scale = (
            (gaussian.scales - self.scale_range[0])
            / (self.scale_range[1] - self.scale_range[0])
        ).clamp(1e-6, 1 - 1e-6)
        semantic = (
            inverse_softplus(gaussian.semantics)
            if self.semantics_activation == "softplus"
            else gaussian.semantics
        )
        return torch.cat(
            [
                torch.logit(xyz),
                torch.logit(scale),
                gaussian.rotations,
                torch.logit(gaussian.opacities.clamp(1e-6, 1 - 1e-6)),
                semantic,
            ],
            -1,
        )

    def forward(self, features, anchor, gaussian):
        result, index, _, _, _ = self._run(features, gaussian)
        output_anchor = self._build_anchor(result)
        child_features = self._gather_hidden(features, index)
        return output_anchor, result, torch.cat([features, child_features], 1)
