"""Reuse the local counterfactual teacher for an isolated parent ranker."""

import torch

from . import OPENOCC_LOSS
from .adaptive_allocation_v7_loss import AdaptiveAllocationV7Loss


@OPENOCC_LOSS.register_module()
class AdaptiveAllocationV8Loss(AdaptiveAllocationV7Loss):
    """No field-consistency penalty: the generator remains DensifyOnly."""

    def __init__(self, **kwargs):
        kwargs.update(stability_weight=0.0, stability_steps=0)
        super().__init__(**kwargs)

    def forward(self, inputs):
        auxiliary = inputs.get("allocation_aux", [])
        if isinstance(auxiliary, dict):
            auxiliary = [auxiliary]
        losses, metrics = [], {}
        for aux in auxiliary:
            if aux.get("version") == 8:
                loss, metrics = self._single_aux_loss(aux, inputs)
                metrics["learned_fraction"] = aux["learned_fraction"]
                losses.append(loss)
        if not losses:
            predictions = inputs.get("pred_occ", [])
            zero = predictions[-1].sum() * 0 if predictions else torch.tensor(0.0)
            return zero, dict(missing_allocation_aux=1.0)
        return torch.stack(losses).mean(), metrics
