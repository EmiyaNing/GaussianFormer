#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opacities,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics
        )
        return logits

    @staticmethod # todo
    def backward(ctx, out_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opacities, semantics = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics,
            out_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opacity_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opacity_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads


def _local_aggregate_inverse_no_grad(
    pts,
    points_int,
    means3D,
    means3D_int,
    opacity,
    occupancy_gt,
    cov3D,
    radii,
    H, W, D
):
    """
    Inverse render function without gradient computation.
    Used when gradients are not needed for inverse rendering.
    """
    # Restructure arguments the way that the C++ lib expects them
    args = (
        pts,
        points_int,
        means3D,
        means3D_int,
        opacity,
        occupancy_gt,
        cov3D,
        radii,
        H, W, D
    )
    # Invoke C++/CUDA inverse renderer without gradient tracking
    with torch.no_grad():
        gaussian_semantic_mask = _C.local_aggregate_inverse(*args)
    
    return gaussian_semantic_mask


class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, inv_softmax=False):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.inv_softmax = inv_softmax

    def forward(
        self, 
        pts,
        means3D, 
        opacities, 
        semantics, 
        scales, 
        cov3D): 

        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opacities = opacities.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # Invoke C++/CUDA rasterization routine
        logits = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        if not self.inv_softmax:
            return logits # n, c
        else:
            assert False

    def inverse_render(
        self,
        pts,
        means3D, 
        opacities, 
        occupancy_gt, 
        scales, 
        cov3D):
        """
        Inverse rendering function for Gaussian semantic mask generation
        
        Args:
            pts: Sampled point coordinates (1, N, 3)
            means3D: Gaussian means (1, G, 3)
            opacities: Gaussian opacities (1, G)
            occupancy_gt: Ground truth occupancy (1, N, C)
            scales: Gaussian scales (1, G, 3)
            cov3D: Gaussian covariance matrices (1, G, 3, 3)
            
        Returns:
            gaussian_semantic_mask: Semantic mask for each Gaussian (G, C)
        """
        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        means3D = means3D.squeeze(0)
        opacities = opacities.squeeze(0)
        occupancy_gt = occupancy_gt.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        # Ensure all tensors are on the same device and have correct data types
        device = pts.device
        dtype = pts.dtype
        
        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int32)
        # Clamp points_int to valid range to prevent CUDA out-of-bounds access
        points_int = torch.clamp(points_int, min=0)
        points_int[:, 0] = torch.clamp(points_int[:, 0], max=self.H-1)
        points_int[:, 1] = torch.clamp(points_int[:, 1], max=self.W-1)
        points_int[:, 2] = torch.clamp(points_int[:, 2], max=self.D-1)
        
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int32)
        # Clamp means3D_int to valid range
        means3D_int = torch.clamp(means3D_int, min=0)
        means3D_int[:, 0] = torch.clamp(means3D_int[:, 0], max=self.H-1)
        means3D_int[:, 1] = torch.clamp(means3D_int[:, 1], max=self.W-1)
        means3D_int[:, 2] = torch.clamp(means3D_int[:, 2], max=self.D-1)
        
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int32)
        # Ensure radii are at least 1 and not too large
        radii = torch.clamp(radii, min=1, max=min(self.H, self.W, self.D))
        
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # Ensure all tensors are contiguous and on the correct device
        pts = pts.contiguous().to(device=device, dtype=dtype)
        points_int = points_int.contiguous().to(device=device, dtype=torch.int32)
        means3D = means3D.contiguous().to(device=device, dtype=dtype)
        means3D_int = means3D_int.contiguous().to(device=device, dtype=torch.int32)
        opacities = opacities.contiguous().to(device=device, dtype=dtype)
        occupancy_gt = occupancy_gt.contiguous().to(device=device, dtype=dtype)
        cov3D = cov3D.contiguous().to(device=device, dtype=dtype)
        radii = radii.contiguous().to(device=device, dtype=torch.int32)

        # Invoke C++/CUDA inverse rendering routine without gradient computation
        gaussian_semantic_mask = _local_aggregate_inverse_no_grad(
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            occupancy_gt,
            cov3D,
            radii,
            self.H, self.W, self.D
        )

        return gaussian_semantic_mask  # G, C