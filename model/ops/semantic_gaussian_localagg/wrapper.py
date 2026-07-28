"""Autograd wrapper for the Phase-B semantic Gaussian CUDA local aggregator.

The operator uses a bounded world-space grid to find sparse point--Gaussian
pairs.  It intentionally returns no point-coordinate gradient: a Gaussian
center is strictly the corresponding OPUS final point, while the auxiliary
Gaussian loss only learns scale, opacity and semantic residual attributes.
"""
import math

import torch

try:
    from . import _C
    CUDA_LOCALAGG_AVAILABLE = True
except ImportError:
    _C = None
    CUDA_LOCALAGG_AVAILABLE = False


def _grid_shape(pc_range, cell_size):
    """Return the finite dense-bin grid covering ``[pc_min, pc_max]``."""
    return tuple(int(math.ceil((pc_range[axis + 3] - pc_range[axis]) / cell_size))
                 for axis in range(3))


class _SemanticGaussianLocalAgg(torch.autograd.Function):
    """Bridge the compiled forward/backward while preserving PyTorch autograd."""
    @staticmethod
    def forward(ctx, points, scales, opacities, semantic_seed, valid_mask,
                pc_min, grid_x, grid_y, grid_z, cell_size, support_sigma, scale_eps,
                denom_eps, include_self, max_scale):
        # A query scans enough cells to retrieve every Gaussian whose largest
        # permitted support overlaps it.  This makes binning a candidate
        # acceleration structure rather than an approximation.
        neighbour_cells = int(math.ceil(support_sigma * max_scale / cell_size))
        residual, denominator = _C.forward(
            points, scales, opacities, semantic_seed, valid_mask, pc_min,
            grid_x, grid_y, grid_z, float(cell_size), float(support_sigma),
            float(scale_eps), float(denom_eps), neighbour_cells, bool(include_self))
        ctx.grid_shape = (grid_x, grid_y, grid_z)
        ctx.params = (float(cell_size), float(support_sigma), float(scale_eps),
                      float(denom_eps), neighbour_cells, bool(include_self))
        ctx.save_for_backward(points, scales, opacities, semantic_seed,
                              valid_mask, pc_min, residual, denominator)
        return residual

    @staticmethod
    def backward(ctx, grad_residual):
        (points, scales, opacities, semantic_seed, valid_mask, pc_min,
         residual, denominator) = ctx.saved_tensors
        grid_x, grid_y, grid_z = ctx.grid_shape
        (cell_size, support_sigma, scale_eps, denom_eps,
         neighbour_cells, include_self) = ctx.params
        grad_scales, grad_opacities, grad_seed = _C.backward(
            grad_residual.contiguous(), points, scales, opacities, semantic_seed,
            valid_mask, pc_min, residual, denominator, grid_x, grid_y, grid_z,
            cell_size, support_sigma, scale_eps, denom_eps, neighbour_cells,
            include_self)
        # points are an anchored geometry input for this branch; pc bounds,
        # bin parameters and the validity mask are non-differentiable.
        return (None, grad_scales, grad_opacities, grad_seed, None, None, None,
                None, None, None, None, None, None, None, None)


def semantic_gaussian_localagg(points, scales, opacities, semantic_seed,
                               valid_mask, pc_range, cell_size,
                               support_sigma=3.0, scale_eps=1e-4,
                               denom_eps=1e-6, include_self=True,
                               max_scale=None):
    """Aggregate point-anchored semantic Gaussians with a fused CUDA operator.

    Args:
        points: OPUS final points in metres, ``[B,G,3]`` float32 CUDA.
        scales: Positive axis-aligned Gaussian scales, ``[B,G,3]`` float32.
        opacities: Sigmoid Gaussian opacity, ``[B,G,1]`` or ``[B,G]`` float32.
        semantic_seed: Per-Gaussian residual logit seed, ``[B,G,C]`` float32.
        valid_mask: Valid Gaussian/query indicator, ``[B,G]`` bool CUDA.
        pc_range: Six world-coordinate floats ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
        cell_size: World-space bin edge length in metres.
        support_sigma: Per-axis Gaussian support cutoff in scale units.
        scale_eps: Lower bound used by the Gaussian kernel.
        denom_eps: Denominator stabilizer for normalized aggregation.
        include_self: Whether a point's own Gaussian participates.
        max_scale: Static scale upper bound. Production heads pass it to avoid
            a device-to-host maximum reduction in the forward hot path.

    Returns:
        Point-aligned semantic residual logits, ``[B,G,C]`` float32 CUDA.
    """
    if not CUDA_LOCALAGG_AVAILABLE:
        raise RuntimeError(
            'semantic_gaussian_localagg CUDA extension is not installed. Run '\
            '`pip install -e model/ops/semantic_gaussian_localagg` in a CUDA environment.')
    tensors = (points, scales, opacities, semantic_seed, valid_mask)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError('semantic_gaussian_localagg requires CUDA tensors')
    if points.dtype != torch.float32 or scales.dtype != torch.float32 or \
            opacities.dtype != torch.float32 or semantic_seed.dtype != torch.float32:
        raise TypeError('semantic_gaussian_localagg currently requires float32 attributes')
    if valid_mask.dtype != torch.bool:
        raise TypeError('semantic_gaussian_localagg requires a bool valid_mask')
    if points.ndim != 3 or points.shape[-1] != 3 or scales.shape != points.shape:
        raise ValueError('points and scales must both have shape [B,G,3]')
    if semantic_seed.shape[:2] != points.shape[:2]:
        raise ValueError('semantic_seed must have shape [B,G,C]')
    if valid_mask.shape != points.shape[:2]:
        raise ValueError('valid_mask must have shape [B,G]')
    if opacities.shape == points.shape[:2] + (1,):
        opacities = opacities.squeeze(-1)
    if opacities.shape != points.shape[:2]:
        raise ValueError('opacities must have shape [B,G] or [B,G,1]')
    if cell_size <= 0 or support_sigma <= 0 or scale_eps <= 0 or denom_eps <= 0:
        raise ValueError('cell_size, support_sigma, scale_eps and denom_eps must be positive')
    if max_scale is None:
        # Useful for direct debugging calls only. The production CUDA head
        # supplies its configured bound and therefore does not synchronize.
        max_scale = float(scales.detach().amax())
    if max_scale <= 0:
        raise ValueError('max_scale must be positive')
    pc_min = points.new_tensor(pc_range[:3], dtype=torch.float32)
    grid_x, grid_y, grid_z = _grid_shape(pc_range, cell_size)
    return _SemanticGaussianLocalAgg.apply(
        points.contiguous(), scales.contiguous(), opacities.contiguous(), semantic_seed.contiguous(),
        valid_mask.contiguous(), pc_min, grid_x, grid_y, grid_z, cell_size, support_sigma,
        scale_eps, denom_eps, include_self, max_scale)
