"""Fused DAF without global broadcast logits or probability matrices."""
import torch
from torch.autograd.function import once_differentiable
from . import deformable_aggregation_ext as ext


class NormalizedSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, shape, starts, locations, visible, dropout, query, camera):
        if not hasattr(ext, 'normalized_sampling_forward'):
            raise RuntimeError('Rebuild the DAF extension for fuse_normalized_sampling=True')
        tensors = (features.contiguous().float(),shape.contiguous().int(),starts.contiguous().int(),
                   locations.contiguous().float(),visible.contiguous(),dropout.contiguous(),
                   query.contiguous().float(),camera.contiguous().float())
        output, statistics = ext.normalized_sampling_forward(*tensors)
        ctx.save_for_backward(*tensors, output, statistics)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        df, dloc, dq, dcam = ext.normalized_sampling_backward(*ctx.saved_tensors, grad.float().contiguous())
        return df, None, None, dloc, None, None, dq, dcam
