"""OPUS-v1 with a trainable post-decoder adapter and a frozen base model.

P1 needs a genuinely controlled comparison: every OPUS-v1 parameter must stay
fixed while only the candidate adapter learns.  ``requires_grad_(False)``
alone is insufficient for image backbones because BatchNorm running statistics
would still change in train mode, so this class also keeps the frozen modules
in evaluation mode after each ``train()`` call.
"""
from __future__ import annotations

from mmseg.models import SEGMENTORS

from .opus_segmentor import OPUSSegmentor


@SEGMENTORS.register_module()
class FrozenOPUSAdapterSegmentor(OPUSSegmentor):
    """Freeze OPUS-v1 and expose only ``head.gaussian`` for P1 training.

    The head must be an ``OPUSGaussianResidualHead`` (or its CUDA subclass).
    This explicit contract prevents a configuration typo from silently
    fine-tuning the backbone and invalidating the ablation.
    """

    def __init__(self, freeze_opus_backbone=False, **kwargs):
        self.freeze_opus_backbone = bool(freeze_opus_backbone)
        super().__init__(**kwargs)
        if self.freeze_opus_backbone:
            if not hasattr(self.head, 'gaussian'):
                raise TypeError(
                    'FrozenOPUSAdapterSegmentor requires a Gaussian residual '
                    'head exposing head.gaussian')
            self.requires_grad_(False)
            self.head.gaussian.requires_grad_(True)
            self._frozen_modules = (
                self.img_backbone, self.img_neck, self.lifter, self.encoder)
        else:
            self._frozen_modules = ()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_opus_backbone:
            # Do not update BatchNorm/dropout state in the checkpointed OPUS
            # path.  The adapter itself remains in the requested mode.
            for module in self._frozen_modules:
                module.eval()
        return self
