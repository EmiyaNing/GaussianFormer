"""OPUS-V1 Gaussian encoder with positive renderer semantics only."""
import torch.nn.functional as F
from mmengine.registry import MODELS

from .opus_v1_gaussian_encoder import OPUSV1GaussianEncoder


@MODELS.register_module()
class OPUSV1GaussianSoftplusEncoder(OPUSV1GaussianEncoder):
    """Keep OPUS raw logits for set loss; softplus only Gaussian semantics."""

    def _make_gaussian(self, features, points, logits, valid, stage):
        gaussian = super()._make_gaussian(features, points, logits, valid, stage)
        return gaussian._replace(semantics=F.softplus(logits))
