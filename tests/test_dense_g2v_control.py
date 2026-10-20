"""Tests for the P3 dense-G2V control.

Uses a small voxel grid so the dense operator is cheap to run.  The point is to
verify the *plumbing* (shape, warm start, gradients, sampling-back), because a
silently wrong control would make the whole model-level dense-vs-sparse
comparison meaningless.
"""
from __future__ import annotations

import pytest
import torch

from model.head.ablations.dense_g2v_residual import DenseG2VResidualAggregator
from model.head.point_anchored_semantic_gaussian import PointAnchoredSemanticGaussianCUDA

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason='requires CUDA')

PC_RANGE = (-8.0, -8.0, -8.0, 8.0, 8.0, 8.0)
GRID = (16, 16, 16)
GRID_SIZE = 1.0


def build(dims=32, children=3, classes=4, device='cuda', seed=0):
    torch.manual_seed(seed)
    module = DenseG2VResidualAggregator(
        embed_dims=dims, num_children=children, num_classes=classes,
        pc_range=PC_RANGE, scale_range=((.1, .1, .1), (.6, .6, .6)),
        initial_scale=.35, initial_opacity=.02, num_neighbors=8,
        include_self=True, query_chunk_size=256,
        grid_shape=list(GRID), grid_size=GRID_SIZE, scale_multiplier=3.0)
    return module.to(device)


def scene(batch=1, queries=2, children=3, device='cuda'):
    token = torch.randn(batch, queries, 32, device=device)
    points = (torch.rand(batch, queries, children, 3, device=device) * 8.0 - 4.0)
    query_valid = torch.ones(batch, queries, dtype=torch.bool, device=device)
    return token, points, query_valid


def test_parameter_count_matches_the_sparse_adapter():
    dense = build(device='cpu')
    sparse = PointAnchoredSemanticGaussianCUDA(
        embed_dims=32, num_children=3, num_classes=4, pc_range=PC_RANGE,
        scale_range=((.1, .1, .1), (.6, .6, .6)))
    assert sum(p.numel() for p in dense.parameters()) == \
        sum(p.numel() for p in sparse.parameters())


def test_voxel_centre_buffer_is_not_persistent():
    dense = build(device='cpu')
    assert dense.voxel_centres.shape == (1, GRID[0] * GRID[1] * GRID[2], 3)
    # A 7 MiB buffer must not be written into every checkpoint.
    assert 'voxel_centres' not in dense.state_dict()


@requires_cuda
def test_zero_initialised_residual_equals_the_base_classifier():
    """Warm start must be exact: a zero residual head gives an all-zero field."""
    module = build(device='cuda')
    token, points, query_valid = scene()
    with torch.no_grad():
        out = module(token, points, query_valid)
    assert out['residual'].shape == (1, 6, 4)
    assert float(out['residual'].abs().max()) == 0.0


@requires_cuda
def test_forward_shapes_and_finiteness():
    module = build(device='cuda')
    token, points, query_valid = scene()
    with torch.no_grad():
        out = module(token, points, query_valid)
    for key in ('points', 'scales', 'opacities', 'residual', 'valid_mask'):
        assert key in out, key
    assert out['points'].shape == (1, 6, 3)
    assert out['residual'].shape == (1, 6, 4)
    assert torch.isfinite(out['residual']).all()


@requires_cuda
def test_residual_is_nonzero_once_the_head_is_perturbed():
    """The sampled dense field must actually depend on the Gaussian attributes."""
    module = build(device='cuda')
    with torch.no_grad():
        module.residual_head.weight.normal_(0, 0.5)
        module.residual_head.bias.normal_(0, 0.5)
    token, points, query_valid = scene()
    with torch.no_grad():
        out = module(token, points, query_valid)
    assert float(out['residual'].abs().max()) > 0.0


@requires_cuda
def test_gradients_reach_scale_opacity_and_seed_heads():
    module = build(device='cuda')
    token, points, query_valid = scene()
    out = module(token, points, query_valid)
    out['residual'].sum().backward()
    for name in ('scale_head', 'opacity_head', 'residual_head'):
        head = getattr(module, name)
        assert head.weight.grad is not None, name
        assert float(head.weight.grad.abs().sum()) >= 0.0


@requires_cuda
def test_points_outside_the_grid_are_clamped_not_crashing():
    module = build(device='cuda')
    token, points, query_valid = scene()
    points[0, 0, 0] = torch.tensor([500.0, -500.0, 500.0], device='cuda')
    with torch.no_grad():
        out = module(token, points, query_valid)
    assert torch.isfinite(out['residual']).all()
