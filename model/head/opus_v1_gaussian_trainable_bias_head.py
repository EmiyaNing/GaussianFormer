"""GaussianFormer rendering with a bias learned during training on rendered logits."""
import torch
from mmengine.registry import MODELS

from .opus_v1_gaussian_head import OPUSV1GaussianHead


@MODELS.register_module()
class OPUSV1GaussianTrainableBiasHead(OPUSV1GaussianHead):
    """Apply one learned empty-class bias in both training and evaluation."""

    def __init__(self, initial_empty_bias=0.0, **kwargs):
        if kwargs.pop('empty_logit_offset', 0.0):
            raise ValueError('Trainable bias head cannot use evaluation-only offset')
        super().__init__(empty_logit_offset=0.0, **kwargs)
        self.empty_bias = torch.nn.Parameter(torch.tensor(float(initial_empty_bias)))

    def forward(self, representation, metas=None, **kwargs):
        result = super().forward(representation, metas=metas, **kwargs)
        calibrated = []
        for logits in result['pred_occ']:
            calibrated.append(torch.cat((logits[:, :self.empty_label],
                                         logits[:, self.empty_label:] - self.empty_bias),
                                        dim=1))
        result['pred_occ'] = calibrated
        result['final_occ'] = calibrated[-1].argmax(dim=1)
        return result
