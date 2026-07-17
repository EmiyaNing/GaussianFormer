"""Official OPUS MSMV sampling extension and compatible fallback.

The C++/CUDA kernels in ``csrc`` are vendored from the official OPUS-V1
repository.  The fallback has the identical channel-last tensor contract and
is useful for CPU CI; production training must build the extension.
"""
import torch
import torch.nn.functional as F

try:
    from ._msmv_sampling_cuda import (
        _ms_deform_attn_cuda_c2345_forward,
        _ms_deform_attn_cuda_c2345_backward,
    )
    MSMV_CUDA = True
except ImportError:
    MSMV_CUDA = False


def msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights):
    if len(mlvl_feats) != scale_weights.shape[-1]:
        raise ValueError('scale_weights must have one entry per FPN level')
    batch, queries, points, _ = sampling_locations.shape
    channels = mlvl_feats[0].shape[-1]
    output = mlvl_feats[0].new_zeros(batch, channels, queries, points)
    grid = sampling_locations.mul(2.0).sub(1.0).unsqueeze(3)
    for level, feature in enumerate(mlvl_feats):
        # [B, N, H, W, C] -> [B, C, N, H, W], matching the official fallback.
        feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        sampled = F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros',
                                align_corners=True).squeeze(-1)
        output += sampled * scale_weights[..., level].view(batch, 1, queries, points)
    return output.permute(0, 2, 1, 3)


class _MSMVSamplingC2345(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights)
        return _ms_deform_attn_cuda_c2345_forward(
            feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        (feat_c2, feat_c3, feat_c4, feat_c5,
         sampling_locations, scale_weights) = ctx.saved_tensors
        # pybind converts the official C++ ``std::vector<Tensor>`` result to a
        # Python list.  torch.autograd.Function requires one positional return
        # value per forward input, so a list is interpreted as a single result
        # on current PyTorch versions (and triggers "expected 6, got 1").
        gradients = _ms_deform_attn_cuda_c2345_backward(
            grad_output.contiguous(), feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights)
        if len(gradients) != 6:
            raise RuntimeError(
                'MSMV C2345 backward must return gradients for all 6 inputs; '
                f'got {len(gradients)}')
        return tuple(gradients)


def msmv_sampling(mlvl_feats, sampling_locations, scale_weights):
    if len(mlvl_feats) == 4 and MSMV_CUDA and sampling_locations.is_cuda:
        # The official C2345 kernel is implemented for float32 tensors.  The
        # local runner enables AMP globally, unlike the original detector's
        # out-fp32 feature boundary, so make that boundary explicit here.
        # These casts remain in the autograd graph and gradients are cast back
        # to the feature dtype by PyTorch during backward.
        return _MSMVSamplingC2345.apply(
            *(feature.float() for feature in mlvl_feats),
            sampling_locations.float(), scale_weights.float())
    return msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights)
