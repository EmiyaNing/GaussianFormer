"""Repository-local CUDA aggregation with compact autograd state.

Build explicitly from the repository root:
    python3 model/ops/sparse_localagg/setup.py build_ext --inplace
"""
import torch
from torch import nn
from torch.autograd.function import once_differentiable

try:
    from . import _C
except ImportError:
    _C = None


class _CompactAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts, points_int, means, means_int, opacity, semantics,
                radii, covariance, H, W, D):
        if _C is None:
            raise RuntimeError('Build the repository-local sparse localagg extension: '
                               'python3 model/ops/sparse_localagg/setup.py build_ext --inplace')
        save_backward = any(ctx.needs_input_grad)
        logits, offsets, keys = _C.forward(
            pts, points_int, means, means_int, opacity, semantics, radii,
            covariance, H, W, D, save_backward)
        ctx.grid = H, W, D
        if save_backward:
            ctx.save_for_backward(offsets, keys, pts, points_int, means,
                                  opacity, semantics, covariance)
        return logits

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        offsets, keys, pts, points_int, means, opacity, semantics, covariance = ctx.saved_tensors
        means_grad, opacity_grad, semantics_grad, covariance_grad = _C.backward(
            offsets, keys, pts, points_int, means, opacity, semantics,
            covariance, gradient.contiguous(), *ctx.grid)
        return (None, None, means_grad, None, opacity_grad, semantics_grad,
                None, covariance_grad, None, None, None)


class CompactLocalAggregator(nn.Module):
    """Same single-scene contract and checkpoint buffer as LocalAggregator."""
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H, self.W, self.D = H, W, D
        self.grid_size = grid_size
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float32)[None])

    def forward(self, pts, means, opacity, semantics, scales, covariance):
        if pts.shape[0] != 1 or means.shape[0] != 1 or pts.requires_grad:
            raise ValueError('compact localagg requires one scene and fixed sample positions')
        if not pts.is_cuda or pts.dtype != torch.float32:
            raise ValueError('compact localagg requires CUDA float32 inputs')
        pts, means, semantics = pts[0], means[0], semantics[0]
        opacity, scales, covariance = opacity[0].reshape(-1), scales[0].detach(), covariance[0]
        if semantics.shape != (means.shape[0], 18):
            raise ValueError('compact localagg requires 18 semantic channels')
        points_int = ((pts - self.pc_min) / self.grid_size).int()
        means_int = ((means.detach() - self.pc_min) / self.grid_size).int()
        shape = points_int.new_tensor([self.H, self.W, self.D])
        if not (((points_int >= 0) & (points_int < shape)).all() and
                ((means_int >= 0) & (means_int < shape)).all()):
            raise ValueError('sample points and Gaussian centers must be inside the grid')
        radii = torch.ceil(scales.amax(-1) * self.scale_multiplier / self.grid_size).int()
        if not (radii >= 1).all():
            raise ValueError('Gaussian radii must be positive')
        covariance = covariance.flatten(1)[:, [0, 4, 8, 1, 5, 2]]
        tensors = (pts, points_int, means, means_int, opacity, semantics, radii, covariance)
        return _CompactAggregate.apply(*(tensor.contiguous() for tensor in tensors),
                                       self.H, self.W, self.D)
