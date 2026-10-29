"""Isolated experiment head: checkpoint renders, diagnose all layers at eval."""
import torch
from torch.utils.checkpoint import checkpoint
from mmengine.registry import MODELS
from .opus_v1_gaussian_trainable_bias_head import OPUSV1GaussianTrainableBiasHead


@MODELS.register_module()
class OPUSAllLayerExperimentHead(OPUSV1GaussianTrainableBiasHead):
    def __init__(self, render_checkpoint=True, render_all_eval=False, **kwargs):
        super().__init__(**kwargs)
        self.render_checkpoint = bool(render_checkpoint)
        self.render_all_eval = bool(render_all_eval)

    def _render_batched(self, *arguments):
        render = super()._render_batched
        if self.training and torch.is_grad_enabled() and self.render_checkpoint:
            return checkpoint(render, *arguments, use_reentrant=False, preserve_rng_state=False)
        return render(*arguments)

    def forward(self, representation, metas=None, **kwargs):
        result = super().forward(representation, metas=metas, **kwargs)
        if not self.training and self.render_all_eval:
            predictions = []
            for gaussian in result['gaussians'][:-1]:
                args = self.prepare_gaussian_args(gaussian)
                logits = self._render_batched(result['sampled_xyz'], *args)
                predictions.append(torch.cat((logits[:, :self.empty_label],
                    logits[:, self.empty_label:] - self.empty_bias), dim=1))
            result['pred_occ'] = predictions + result['pred_occ']
            result['supervised_stage_indices'] = list(range(len(representation)))
        return result
