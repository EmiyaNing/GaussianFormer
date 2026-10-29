"""OPUS temporal decoder with query scores and a fixed final Gaussian budget."""
import torch
from torch import nn
from mmengine.registry import MODELS

from .opus_v1_gaussian_temporal_encoder import OPUSV1GaussianTemporalEncoder
from .gaussian_budget_gate import budget_sigmoid


@MODELS.register_module()
class OPUSV1GaussianBudgetEncoder(OPUSV1GaussianTemporalEncoder):
    def __init__(self, keep_budget=76800, budget_total_iters=28130,
                 budget_temperature_start=.5, budget_temperature_end=.15,
                 importance_hidden_dims=64, **kwargs):
        super().__init__(**kwargs)
        if keep_budget < 1 or budget_total_iters < 1:
            raise ValueError('Budget and schedule length must be positive')
        self.keep_budget = int(keep_budget)
        self.budget_total_iters = int(budget_total_iters)
        self.budget_temperature_start = float(budget_temperature_start)
        self.budget_temperature_end = float(budget_temperature_end)
        if min(self.budget_temperature_start, self.budget_temperature_end) <= 0:
            raise ValueError('Gate temperatures must be positive')
        dims = self.layers[-1].cls_branch[-1].in_features
        self.importance_branch = nn.Sequential(
            nn.Linear(dims, importance_hidden_dims), nn.ReLU(inplace=True),
            nn.Linear(importance_hidden_dims, 1))
        nn.init.zeros_(self.importance_branch[-1].weight)
        nn.init.zeros_(self.importance_branch[-1].bias)

    def forward(self, *args, global_iter=None, **kwargs):
        result = super().forward(*args, **kwargs)
        state = result['representation'][-1]
        raw_score = self.importance_branch(state['query_features']).squeeze(-1)
        query_score = raw_score.sigmoid()
        alpha = state['gaussian'].opacities.squeeze(-1)
        # log is monotone: this has exactly the ranking of score * opacity.
        log_priority = (query_score.clamp_min(1e-8).log()[..., None]
                        + alpha.clamp_min(1e-8).log()).flatten(1)
        valid = state['query_valid_mask'][..., None].expand_as(alpha).flatten(1)
        log_priority = log_priority.masked_fill(~valid, -torch.inf)
        iteration = (self.budget_total_iters if not self.training else 0) if global_iter is None else int(global_iter)
        fraction = min(max(iteration / max(self.budget_total_iters - 1, 1), 0.), 1.)
        temperature = self.budget_temperature_start + fraction * (
            self.budget_temperature_end - self.budget_temperature_start)
        gates = budget_sigmoid(log_priority, self.keep_budget, temperature)
        # Stable ties prefer lower candidate indices. Selection is used for
        # final Set supervision and inference; Occ trains the continuous gates.
        with torch.no_grad():
            indices = torch.argsort(log_priority, dim=-1, descending=True,
                                    stable=True)[:, :self.keep_budget]
            children = alpha.shape[-1]
            counts = torch.zeros_like(query_score, dtype=torch.long)
            counts.scatter_add_(1, indices // children, torch.ones_like(indices))
        result['allocation'] = dict(
            gates=gates, query_scores=query_score, log_priority=log_priority,
            selected_indices=indices, selected_per_query=counts,
            keep_budget=self.keep_budget, candidate_count=log_priority.shape[-1],
            children=alpha.shape[-1], temperature=temperature)
        return result
