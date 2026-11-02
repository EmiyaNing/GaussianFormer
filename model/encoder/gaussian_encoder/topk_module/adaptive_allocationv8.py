"""Learn parent selection while retaining the original DensifyOnly generator.

The quota is fixed, original parents are untouched, and the scorer receives
detached decoder features. Warmup is exactly the baseline generation path.
Only K children are rendered; supervision probes never enter the output.
"""

import torch
from mmengine.registry import MODELS
from torch import nn

from .adaptive_allocationv7 import gather_gaussian
from .topk_densify_v2 import DensifyOnly


@MODELS.register_module()
class AdaptiveAllocationV8(DensifyOnly):
    """Terminal fixed-budget growth with a separately supervised ranker.

    Warmup/transition durations count gradient-enabled training forwards per
    rank, including gradient accumulation microbatches. Evaluation uses the
    saved training step and the same selection mixture, without RNG or labels.
    The original generator's parameter names are checkpoint compatible.
    """

    def __init__(
        self,
        feat_embed_dim=128,
        semantic_dim=17,
        hidden_dim=128,
        topk_count=2560,
        grid_size=0.4,
        selection_mode="learned",
        selection_warmup_steps=1000,
        selection_transition_steps=1000,
        probe_count=128,
        probe_seed=61029,
        **kwargs,
    ):
        if topk_count < 0 or probe_count < 0 or grid_size <= 0:
            raise ValueError("Invalid allocation budget, probe count or grid size")
        if selection_warmup_steps < 0 or selection_transition_steps < 0:
            raise ValueError("Selection durations must be nonnegative")
        if selection_mode not in ("learned", "opacity"):
            raise ValueError("selection_mode must be learned or opacity")
        super().__init__(
            feat_embed_dim=feat_embed_dim,
            semantic_dim=semantic_dim,
            topk_count=topk_count,
            **kwargs,
        )
        self.grid_size = float(grid_size)
        self.selection_mode = selection_mode
        self.selection_warmup_steps = int(selection_warmup_steps)
        self.selection_transition_steps = int(selection_transition_steps)
        self.probe_count = int(probe_count)
        self.probe_seed = int(probe_seed)
        # Constructing the extra head must not change baseline initialization
        # or the global dropout stream. fork_rng here covers CPU construction.
        with torch.random.fork_rng(devices=[]):
            self.score_network = nn.Sequential(
                nn.Linear(feat_embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        self.register_buffer("allocation_step", torch.zeros((), dtype=torch.long))
        self._step_value = 0
        self._allocation_aux = None
        self.init_weight()

    def generic_init_parameters(self):
        """Let GaussianOccEncoder initialize precisely the baseline parameters."""
        return (
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("score_network.")
        )

    def allocation_score_parameters(self):
        return self.score_network.parameters()

    def init_weight(self):
        # This is also called after the encoder's generic Xavier pass. A local
        # generator keeps the original model's global RNG state intact.
        device = self.score_network[0].weight.device
        generator = torch.Generator(device=device).manual_seed(self.probe_seed)
        nn.init.xavier_uniform_(self.score_network[0].weight, generator=generator)
        nn.init.zeros_(self.score_network[0].bias)
        nn.init.ones_(self.score_network[1].weight)
        nn.init.zeros_(self.score_network[1].bias)
        nn.init.normal_(self.score_network[3].weight, std=0.001, generator=generator)
        nn.init.zeros_(self.score_network[3].bias)

    def _current_step(self):
        if self._step_value is None:
            self._step_value = int(self.allocation_step.cpu())
        return self._step_value

    def learned_fraction(self, step=None):
        if self.selection_mode == "opacity":
            return 0.0
        step = self._current_step() if step is None else step
        if step < self.selection_warmup_steps:
            return 0.0
        if not self.selection_transition_steps:
            return 1.0
        return min(
            1.0, (step - self.selection_warmup_steps) / self.selection_transition_steps
        )

    def _select(self, features, gaussian, budget, step):
        opacity = gaussian.opacities.squeeze(-1)
        learned_count = int(budget * self.learned_fraction(step))
        if not learned_count:
            return opacity.topk(budget, 1).indices
        # Hard top-k selection has no task gradient. Save autograd state only
        # for the small set of supervised probes, never all N ranking inputs.
        with torch.no_grad():
            priority = self.score_network(features.detach()).squeeze(-1)
            learned = priority.topk(learned_count, 1).indices
            remaining = opacity.scatter(1, learned, -torch.inf)
            original = remaining.topk(budget - learned_count, 1).indices
        return torch.cat([learned, original], 1)

    def _training_aux(self, features, gaussian, index, child, step):
        batch_size, count = features.shape[:2]
        probe_count = min(self.probe_count, count)
        generator = torch.Generator(device=features.device).manual_seed(
            self.probe_seed + step
        )
        priority = torch.rand(
            batch_size, count, device=features.device, generator=generator
        )
        probes = priority.topk(probe_count, 1).indices
        probe_features = features.detach().gather(
            1, probes[..., None].expand(-1, -1, features.shape[-1])
        )
        # Main CE/Lovasz trains the original child generator; this loss trains
        # only score_network. Neither its input nor the teacher reaches trunk.
        score = self.score_network(probe_features).squeeze(-1)
        with torch.no_grad():
            parent = gather_gaussian(gaussian, probes)
            local_index = torch.arange(probe_count, device=features.device)[
                None
            ].expand(batch_size, -1)
            unused_anchor = features.new_empty(batch_size, probe_count, 0)
            _, probe_child = super().forward(
                probe_features,
                unused_anchor,
                parent,
                terminal=True,
                selected_indices=local_index,
            )
        return dict(
            version=8,
            needs_base_logits=True,
            probe_index=probes,
            probe_score=score,
            probe_parent=parent,
            probe_updated=parent,
            probe_child=probe_child,
            selected_index=index,
            step=step,
            input_count=count,
            output_count=count + index.shape[1],
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            parent_displacement=score.detach().sum() * 0,
            child_opacity=(
                child.opacities.detach().mean()
                if index.shape[1]
                else score.detach().sum() * 0
            ),
            learned_fraction=self.learned_fraction(step),
        )

    def forward_terminal(self, features, anchor, gaussian):
        self._allocation_aux = None
        step = self._current_step()
        budget = min(self.topk_count, features.shape[1])
        index = self._select(features, gaussian, budget, step)
        result, child = super().forward(
            features, anchor, gaussian, terminal=True, selected_indices=index
        )
        if self.training and torch.is_grad_enabled():
            self._allocation_aux = self._training_aux(
                features, gaussian, index, child, step
            )
            self.allocation_step.add_(1)
            self._step_value = step + 1
        return dict(gaussian=result, delta_gaussian=child)

    def forward(self, features, anchor, gaussian):
        raise RuntimeError("AdaptiveAllocationV8 is terminal; use forward_terminal")

    def get_allocation_aux(self):
        return self._allocation_aux

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._step_value = None
        self._allocation_aux = None
