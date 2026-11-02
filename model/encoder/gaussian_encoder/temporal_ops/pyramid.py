"""Pack each image time once, before caching and before either decoder."""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from . import extension


@lru_cache(maxsize=16)
def _metadata(spatial_shapes, device):
    starts = [0]
    for height, width in spatial_shapes[:-1]:
        starts.append(starts[-1] + height * width)
    return (torch.tensor(spatial_shapes, device=device, dtype=torch.int32),
            torch.tensor(starts, device=device, dtype=torch.int32))


class _PackPyramid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *levels):
        ctx.sizes = [list(level.shape) for level in levels]
        ctx.strides = [list(level.stride()) for level in levels]
        # A permutation has no value-dependent derivative: retain only host
        # shape/stride metadata, not another set of FPN activations.
        return extension().pack_pyramid_forward(list(levels))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        gradients = extension().pack_pyramid_backward(
            gradient.contiguous(), ctx.sizes, ctx.strides)
        return tuple(gradient if needed else None
                     for gradient, needed in zip(gradients, ctx.needs_input_grad))


def pack_pyramid(levels):
    packed = _PackPyramid.apply(*levels)
    spatial = tuple(tuple(level.shape[-2:]) for level in levels)
    shapes, starts = _metadata(spatial, packed.device)
    return packed, shapes, starts
