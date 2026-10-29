"""Gradually sharpen a fixed-budget soft selection, with observable diagnostics."""
import torch
from torch import nn

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class GaussianBudgetLoss(nn.Module):
    def __init__(self, binary_weight=.1, total_iters=28130,
                 warmup_fraction=.1, ramp_fraction=.2):
        super().__init__()
        self.binary_weight = float(binary_weight)
        self.total_iters = int(total_iters)
        self.warmup_fraction = float(warmup_fraction)
        self.ramp_fraction = float(ramp_fraction)
        if min(self.binary_weight, self.warmup_fraction) < 0 or self.ramp_fraction <= 0 or self.total_iters <= 0:
            raise ValueError('Invalid binary gate schedule')

    def forward(self, inputs):
        allocation = inputs['allocation']
        p = allocation['gates']
        progress = int(inputs.get('global_iter', 0)) / max(self.total_iters - 1, 1)
        ramp = min(max((progress - self.warmup_fraction) / self.ramp_fraction, 0.), 1.)
        weight = self.binary_weight * ramp
        raw = (p * (1 - p)).mean()
        score = allocation['query_scores'].detach()
        counts = allocation['selected_per_query']
        top = counts.topk(max(counts.shape[-1] // 5, 1), dim=-1).values
        metrics = dict(
            binary_raw=raw.detach(), binary_weight=weight,
            temperature=allocation['temperature'],
            candidates=allocation['candidate_count'], hard_count=allocation['keep_budget'],
            soft_count=p.detach().sum(-1).mean(),
            budget_max_abs_error=(p.detach().sum(-1) - allocation['keep_budget']).abs().max(),
            intermediate_gate_fraction=((p.detach() > .1) & (p.detach() < .9)).float().mean(),
            query_score_mean=score.mean(), query_score_std=score.std(unbiased=False),
            hard_children_min=counts.min(), hard_children_max=counts.max(),
            hard_children_mean=counts.float().mean(),
            hard_children_std=counts.float().std(unbiased=False),
            zero_selected_query_fraction=(counts == 0).float().mean(),
            most_allocated_20pct_budget_share=top.sum(-1).float().mean() / allocation['keep_budget'])
        return weight * raw, metrics
