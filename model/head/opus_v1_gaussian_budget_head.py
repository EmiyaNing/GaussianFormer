"""Soft-budget training and exact-count inference through the native renderer."""
import torch
from mmengine.registry import MODELS

from .opus_v1_gaussian_trainable_bias_head import OPUSV1GaussianTrainableBiasHead
from model.encoder.gaussian_budget_gate import gather_candidates, gather_gaussian


@MODELS.register_module()
class OPUSV1GaussianBudgetHead(OPUSV1GaussianTrainableBiasHead):
    def forward(self, representation, allocation, metas=None, **kwargs):
        states = list(representation)
        last = dict(states[-1])
        original = last['gaussian']
        indices = allocation['selected_indices']
        selected = gather_gaussian(original, indices)
        selected_logits = gather_candidates(last['opus_logits'], indices)
        if self.training:
            gate = allocation['gates'].reshape(original.opacities.shape)
            last['gaussian'] = original._replace(opacities=original.opacities * gate)
        else:
            # Gather before covariance preparation and CUDA aggregation. The
            # selected GS retain their physical opacity; no query score is
            # multiplied a second time at inference.
            last['gaussian'] = selected
            last['query_points'] = gather_candidates(last['query_points'], indices)
            last['opus_logits'] = selected_logits
            last['query_valid_mask'] = torch.ones(
                (indices.shape[0], 1), device=indices.device, dtype=torch.bool)
        states[-1] = last
        result = super().forward(states, metas=metas, **kwargs)
        # Keep final Set supervision at N, even though soft training renders
        # 1.5N candidates. Discrete indices do not provide allocator gradients.
        result['opus_pred_points'][-1] = selected.means.flatten(1, 2)
        result['opus_pred_logits'][-1] = selected_logits.flatten(1, 2)
        result['opus_point_valid_masks'][-1] = torch.ones_like(indices, dtype=torch.bool)
        result['opus_points'] = result['opus_pred_points'][-1]
        result['opus_labels'] = result['opus_pred_logits'][-1].argmax(-1)
        result['opus_scores'] = result['opus_pred_logits'][-1].sigmoid().amax(-1)
        result['allocation'] = allocation
        return result
