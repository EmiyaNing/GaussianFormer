"""Python autograd wrapper for the S2GO-style block dense G2V operator.

Mirrors ``model/head/localagg``'s ``LocalAggregator`` calling convention so the
two dense operators are interchangeable, and adds a backward so the operator can
be used in a training path (needed to measure training-level memory against an
optimised dense baseline rather than the naive one).
"""
from __future__ import annotations

import torch

try:
    from . import block_g2v as _C
    CUDA_BLOCK_G2V_AVAILABLE = True
except ImportError:
    _C = None
    CUDA_BLOCK_G2V_AVAILABLE = False


def tile_edge():
    """Tile edge in voxels that this extension was **compiled** with.

    ``None`` when the extension is unavailable.  A tile-size sweep must assert
    this matches the size it asked for, otherwise a broken ``-D`` plumbing would
    silently produce three identical rows that look like a design-space study.
    """
    if not CUDA_BLOCK_G2V_AVAILABLE:
        return None
    return int(_C.tile_edge())


class _BlockG2V(torch.autograd.Function):
    """Bridge the compiled forward/backward while preserving autograd."""

    @staticmethod
    def forward(ctx, pts, points_int, means3D, means3D_int, opacities,
                semantics, radii, cov3D, grid_x, grid_y, grid_z):
        out, sorted_keys, sorted_ids = _C.forward(
            pts, points_int, means3D, means3D_int, opacities, semantics,
            radii, cov3D, grid_x, grid_y, grid_z)
        ctx.grid = (grid_x, grid_y, grid_z)
        ctx.save_for_backward(pts, points_int, means3D, means3D_int, opacities,
                              semantics, radii, cov3D, sorted_keys, sorted_ids)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (pts, points_int, means3D, means3D_int, opacities, semantics, radii,
         cov3D, sorted_keys, sorted_ids) = ctx.saved_tensors
        grid_x, grid_y, grid_z = ctx.grid
        grad_means, grad_opacity, grad_semantic, grad_cov = _C.backward(
            grad_out.contiguous(), pts, points_int, means3D, means3D_int,
            opacities, semantics, radii, cov3D, sorted_keys, sorted_ids,
            grid_x, grid_y, grid_z)
        # Query points, integer bins, radii and the validity data are not
        # differentiable here, matching the naive dense operator.
        return (None, None, grad_means, None, grad_opacity, grad_semantic,
                None, grad_cov, None, None, None)


def block_g2v_forward(pts, points_int, means3D, means3D_int, opacities,
                      semantics, radii, cov3D, grid):
    """Render Gaussians to a dense voxel grid with 4x4x4 tile sharing.

    Args:
        pts: ``[N, 3]`` float32 voxel-centre coordinates in row-major order
            ``(x * grid_y + y) * grid_z + z``.
        points_int: ``[N, 3]`` int32 integer voxel coordinates of ``pts``.
        means3D: ``[P, 3]`` float32 Gaussian centres.
        means3D_int: ``[P, 3]`` int32 integer voxel coordinates of the centres.
        opacities: ``[P]`` float32.
        semantics: ``[P, 18]`` float32 (the dense path is fixed to 18 channels).
        radii: ``[P]`` int32 per-Gaussian voxel radius.
        cov3D: ``[P, 6]`` float32 packed inverse covariance
            ``(xx, yy, zz, xy, yz, xz)``.
        grid: ``(grid_x, grid_y, grid_z)``.

    Returns:
        ``[N, 18]`` float32 dense logits.
    """
    if not CUDA_BLOCK_G2V_AVAILABLE:
        raise RuntimeError(
            'block_g2v extension is not installed. Run '
            '`pip install -e model/ops/block_g2v` in a CUDA environment.')
    grid_x, grid_y, grid_z = (int(v) for v in grid)
    if pts.shape[0] != grid_x * grid_y * grid_z:
        raise ValueError(
            'pts must contain every voxel centre in row-major order '
            f'(expected {grid_x * grid_y * grid_z}, got {pts.shape[0]})')
    return _BlockG2V.apply(pts.contiguous(), points_int.contiguous(),
                           means3D.contiguous(), means3D_int.contiguous(),
                           opacities.contiguous(), semantics.contiguous(),
                           radii.contiguous(), cov3D.contiguous(),
                           grid_x, grid_y, grid_z)
