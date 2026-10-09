"""OPUS-V1 Gaussian output rendered through GaussianFormer's local aggregate."""
import torch
from mmengine.registry import MODELS

from .gaussian_opus_head import GaussianOPUSHead


@MODELS.register_module()
class OPUSV1GaussianHead(GaussianOPUSHead):
    """Return both differentiable Gaussian occupancy and OPUS set outputs."""

    def __init__(self, empty_logit_offset=0.0, **kwargs):
        super().__init__(**kwargs)
        # Evaluation-only calibration of the final empty-vs-occupied decision.
        # The default preserves the original GaussianFormer rendering path and
        # every training-time occupancy logit/loss.  A nonzero value must be
        # selected on training data, never on the evaluation split.
        self.empty_logit_offset = float(empty_logit_offset)

    def forward(self, representation, metas=None, **kwargs):
        result = super().forward(representation, metas=metas, **kwargs)
        if not self.training and self.empty_logit_offset:
            logits = result['pred_occ'][-1]
            foreground_score, foreground_class = logits[:, :self.empty_label].max(dim=1)
            empty_score = logits[:, self.empty_label] - self.empty_logit_offset
            result['final_occ'] = torch.where(
                foreground_score > empty_score,
                foreground_class,
                torch.full_like(foreground_class, self.empty_label))
        pred_points, pred_logits, valid_masks = [], [], []
        for state in representation:
            gaussian = state['gaussian']
            batch, queries, children = state['query_points'].shape[:3]
            pred_points.append(gaussian.means.reshape(batch, queries * children, 3))
            pred_logits.append(state['opus_logits'].reshape(batch, queries * children, -1))
            valid_query = state.get('query_valid_mask')
            if valid_query is None:
                valid_query = torch.ones((batch, queries), device=gaussian.means.device,
                                         dtype=torch.bool)
            valid_masks.append(valid_query.unsqueeze(-1).expand(-1, -1, children).reshape(
                batch, queries * children))
        result.update({
            'opus_pred_points': pred_points,
            'opus_pred_logits': pred_logits,
            'opus_point_valid_masks': valid_masks,
            'opus_points': pred_points[-1],
            'opus_labels': pred_logits[-1].argmax(dim=-1),
            'opus_scores': pred_logits[-1].sigmoid().amax(dim=-1),
        })
        return result
