"""Numerical checks against dense references, independent of the mini data."""
import torch
import torch.nn.functional as F

from loss.gaussian_responsibility_loss import (
    allocation_targets, match_gradient_norm, scatter_sum, sparse_support,
)
from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.utils.utils import get_rotation_matrix
from tools.center_support_common import kernel_weights


def test_sparse_support_matches_dense_rotated_gaussian_reference():
    torch.manual_seed(928)
    xyz = torch.cartesian_prod(torch.arange(4), torch.arange(4), torch.arange(4)).float() * .4 + .2
    means = torch.tensor([[.21, .59, .99], [1.2, 1.0, .4]])
    scales = torch.tensor([[.08, .3, .17], [.4, .2, .11]])
    rotations = torch.randn(2, 4)
    opacity = torch.tensor([[.2], [.8]])
    gaussian = GaussianPrediction(means, scales, rotations, opacity, torch.zeros(2, 17))
    row, col, weight = sparse_support(gaussian, xyz, torch.zeros(len(xyz), dtype=torch.long),
        pc_min=(0, 0, 0), shape=(4, 4, 4), chunk_size=1)
    transform = torch.diag_embed(scales) @ get_rotation_matrix(rotations)
    inv = torch.linalg.inv(transform.transpose(-1, -2) @ transform)
    dense = kernel_weights(xyz, means, opacity, scales, inv, (0, 0, 0))
    dense *= torch.linalg.det(inv).sqrt()[None] / (2 * torch.pi) ** 1.5
    result = torch.zeros_like(dense)
    result[col, row] = weight
    torch.testing.assert_close(result, dense, rtol=2e-5, atol=2e-6)


def test_region_targets_preserve_missing_voxels_and_unused_gaussians():
    row, col = torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])
    weight = torch.tensor([1., 3., 2.])
    y, nearest, guard = torch.tensor([2, 4, 5]), torch.tensor([4, 2, 8]), torch.tensor([0, 0, 1])
    target, diag, state = allocation_targets(row, col, weight, y, nearest, guard)
    expected = torch.zeros(3, 17)
    expected[0, 2], expected[0, 4] = .2, .8
    expected[1, 2], expected[1, 5] = 3 / 7, 4 / 7
    expected[2, 8] = 1
    torch.testing.assert_close(target, expected)
    assert diag['guard_voxels'] == 1
    torch.testing.assert_close(scatter_sum(state['plan'], state['columns'], 3), torch.ones(3))


def test_sparse_unbalanced_transport_matches_dense_scaling_solution():
    kernel = torch.tensor([[.7, .8, .6], [.3, .2, .4]], dtype=torch.float64)
    row, col = torch.cartesian_prod(torch.arange(2), torch.arange(3)).T
    y, nearest = torch.tensor([0, 1, 2]), torch.tensor([0, 1])
    target, diagnostic, state = allocation_targets(row, col, kernel.flatten(), y,
        nearest, torch.tensor([0, 0, 0]), transport=True, iterations=64)
    a = kernel.sum(-1).sqrt(); a *= 3 / a.sum()
    u, v = torch.ones(2, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    for _ in range(64):
        u = (a / (kernel @ v)) ** .5
        v = (1 / (kernel.T @ u)) ** (10 / 11)
    expected = u[:, None] * kernel * v[None]
    torch.testing.assert_close(state['plan'].reshape(2, 3), expected)
    torch.testing.assert_close(target[:, :3], expected / expected.sum(-1, keepdim=True))
    assert diagnostic['transport_fixed_point_log_residual'] < 1e-9
    assert not torch.allclose(state['mass'], state['base_mass'])


def test_gradient_matching_preserves_proposed_direction_and_reference_norm():
    torch.manual_seed(928)
    z = torch.randn(11, 17, requires_grad=True)
    label = torch.arange(11) % 17
    reference = F.binary_cross_entropy_with_logits(z, F.one_hot(label, 17).float(), reduction='sum') / 11
    proposed = F.cross_entropy(z, label)
    original_direction = torch.autograd.grad(proposed, z, retain_graph=True)[0]
    matched, scale, _, _ = match_gradient_norm(reference, proposed, z)
    actual = torch.autograd.grad(matched, z, retain_graph=True)[0]
    ref = torch.autograd.grad(reference, z)[0]
    torch.testing.assert_close(actual.norm(), ref.norm())
    torch.testing.assert_close(actual, original_direction * scale)
    assert not scale.requires_grad
