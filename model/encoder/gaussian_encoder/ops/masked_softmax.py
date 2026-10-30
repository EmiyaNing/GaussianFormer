"""Query/group masked softmax with one saved probability tensor."""

import torch
from torch.autograd.function import once_differentiable

from . import deformable_aggregation_ext


class MaskedWeightSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, mask):
        if not hasattr(deformable_aggregation_ext, 'masked_weight_softmax_forward'):
            raise RuntimeError('Rebuild DAF for fused_weight_softmax=True: cd '
                               'model/encoder/gaussian_encoder/ops && '
                               'python3.10 setup.py build_ext --inplace')
        probabilities = deformable_aggregation_ext.masked_weight_softmax_forward(
            logits.contiguous().float(), mask.contiguous())
        # DAF saves this same allocation. No separate softmax output, float
        # mask multiplier or dropout/visibility mask is needed for backward.
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        probabilities, = ctx.saved_tensors
        return deformable_aggregation_ext.masked_weight_softmax_backward(
            probabilities, grad.contiguous().float()), None


def normalize_weights(logits, mask, fused=False):
    """Normalize [B,N,K,V,L,G] over K*V*L, zeroing fully masked groups."""
    if fused:
        return MaskedWeightSoftmax.apply(logits, mask)
    valid = mask.any(dim=(2, 3, 4), keepdim=True)
    safe = logits.masked_fill(~mask, -torch.inf).masked_fill(~valid, 0.)
    probabilities = safe.flatten(2, 4).softmax(dim=-2).reshape_as(logits)
    return probabilities.masked_fill(~valid, 0.)
