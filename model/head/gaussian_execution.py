"""Algebraic helpers for the opt-in React execution path."""

import torch

from ..utils.utils import get_rotation_matrix


def inverse_gaussian_covariance(scales, rotations):
    """Invert (S R)^T(S R) without a matrix inverse or a CPU transfer."""
    rotation = get_rotation_matrix(rotations)
    inverse_factor = rotation / scales[..., :, None]
    return inverse_factor.transpose(-1, -2) @ inverse_factor


def fixed_gaussian_weights(points, mean, scales, rotations, scale_multiplier,
                           pc_min, grid_size, grid_shape):
    """Exact additive background field, including the renderer's AABB support."""
    inverse = inverse_gaussian_covariance(scales, rotations)[0, 0]
    delta = points - mean[0, 0]
    weights = torch.exp(-0.5 * (delta @ inverse * delta).sum(-1))
    # The current Occ3D background covers the whole grid. Keeping this mask
    # also preserves behavior for smaller backgrounds or arbitrary query sets.
    coords = ((points - pc_min) / grid_size).to(torch.int32)
    center = ((mean[0, 0] - pc_min) / grid_size).to(torch.int32)
    radius = torch.ceil(scales[0, 0] * scale_multiplier / grid_size).to(torch.int32)
    lower = (center - radius).clamp_min(0)
    upper = torch.minimum(center + radius + 1, coords.new_tensor(grid_shape))
    inside = ((coords >= lower) & (coords < upper)).all(-1)
    return weights * inside.to(weights.dtype)
