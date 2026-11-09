"""Learn a global empty-logit bias for the additive Gaussian renderer."""

import math

import torch
from mmengine.registry import MODELS

from .gaussian_head import GaussianHead


@MODELS.register_module()
class GaussianTrainableBiasHead(GaussianHead):
    """Use the same learned empty correction in every supervised decoder.

    Positive bias favors occupied predictions. Initialization at zero preserves
    an existing checkpoint's output. The rendered parent logits are left intact
    so incremental densification cannot apply the bias more than once.
    """

    def __init__(self, initial_empty_bias=0.0, **kwargs):
        if not math.isfinite(initial_empty_bias):
            raise ValueError('initial_empty_bias must be finite')
        super().__init__(**kwargs)
        if self.use_localaggprob:
            raise ValueError('Learned empty-logit bias requires additive logits')
        if self.empty_label != self.num_classes - 1:
            raise ValueError('The empty class must be the last output channel')
        self.empty_bias = torch.nn.Parameter(torch.tensor(float(initial_empty_bias)))

    def forward(self, representation, metas=None, **kwargs):
        return_unbiased = kwargs.pop('return_unbiased_prediction', False)
        result = super().forward(representation, metas=metas, **kwargs)
        if return_unbiased:
            result['unbiased_final_occ'] = result['final_occ']
        result['pred_occ'] = [torch.cat(
            (logits[:, :self.empty_label], logits[:, self.empty_label:] - self.empty_bias),
            dim=1) for logits in result['pred_occ']]
        result['final_occ'] = result['pred_occ'][-1].argmax(dim=1)
        return result
