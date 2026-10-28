"""Probability objective, invalid-child masking and actual CUDA source gradients."""

import pytest
import torch
import torch.nn.functional as F

from loss.pmg_gf2_probability_loss import conditional_probability_nll


def test_joint_empty_nll_has_no_direct_semantic_gradient():
    raw = torch.randn(2, 17, 5, requires_grad=True)
    occupancy_logit = torch.randn(2, 5, requires_grad=True)
    conditional, occupied = raw.softmax(1), occupancy_logit.sigmoid()
    joint = torch.cat((conditional * occupied[:, None], (1 - occupied)[:, None]), 1)
    loss = F.nll_loss(joint.log(), torch.full((2, 5), 17))
    semantic_gradient, geometry_gradient = torch.autograd.grad(loss, (raw, occupancy_logit))
    torch.testing.assert_close(semantic_gradient, torch.zeros_like(raw))
    assert geometry_gradient.abs().sum() > 0
    torch.testing.assert_close(joint.sum(1), torch.ones_like(occupied))


def test_conditional_nll_matches_weighted_ce_and_ignores_free():
    raw = torch.randn(2, 17, 5, requires_grad=True)
    labels = torch.tensor([[0, 17, 2, 255, 3], [16, 1, 17, 255, 8]])
    weights = torch.linspace(.5, 3, 17)
    occupied = (labels != 255) & (labels != 17)
    actual = conditional_probability_nll(raw.softmax(1), labels, weights)
    expected = F.cross_entropy(raw.transpose(1, 2)[occupied], labels[occupied], weight=weights)
    torch.testing.assert_close(actual, expected)
    actual_gradient, = torch.autograd.grad(actual, raw, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, raw)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA operator validation")
def test_batched_renderer_reference_gradients_and_invalid_child_filter():
    from model.head.pmg_gf2_probability_head import PMGGF2ProbabilityRenderer
    from model.encoder.gaussian_encoder.utils import GaussianPrediction

    renderer = PMGGF2ProbabilityRenderer(max_render_scale=.5, apply_loss_type="fixed_5",
        cuda_kwargs=dict(scale_multiplier=3, H=16, W=16, D=16, pc_min=[0., 0., 0.], grid_size=.5)).cuda()
    xyz = torch.tensor([[[2.21, 2.13, 2.17], [2.70, 2.25, 2.10], [6., 6., 6.]]], device="cuda").repeat(2, 1, 1)
    means = torch.tensor([[[2., 2., 2.], [2.5, 2.2, 2.3], [6., 6., 6.]]], device="cuda").repeat(2, 1, 1).requires_grad_()
    scales = torch.full_like(means, .5, requires_grad=True)
    rotations = torch.zeros(2, 3, 4, device="cuda"); rotations[..., 0] = 1
    opacity = torch.tensor([[[.3], [.8], [0.]]], device="cuda").repeat(2, 1, 1).requires_grad_()
    raw = torch.randn(2, 3, 17, device="cuda", requires_grad=True)
    state = GaussianPrediction(*(value[:, None] for value in (means, scales, rotations, opacity, raw)))
    joint, conditional = renderer.render_distribution(state, xyz)
    delta = xyz[:, :, None] - means[:, None, :2]
    kernel = torch.exp(-.5 * (delta / scales[:, None, :2]).square().sum(-1))
    # All first two samples are inside both support boxes. Third has none.
    supported = torch.tensor([1., 1., 0.], device="cuda")[None, :, None]
    kernel = kernel * supported
    weight = kernel * opacity[:, None, :2, 0] / scales[:, None, :2].prod(-1) / (2 * torch.pi)**1.5
    q = (weight @ raw[:, :2].softmax(-1)) / weight.sum(-1, keepdim=True).clamp_min(1e-9)
    q = torch.where(supported.bool(), q, torch.full_like(q, 1 / 17))
    occupied = 1 - (1 - kernel).prod(-1)
    expected = torch.cat((q * occupied[..., None], (1 - occupied)[..., None]), -1).transpose(1, 2)
    torch.testing.assert_close(joint, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(conditional, q.transpose(1, 2), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(joint.sum(1), torch.ones_like(occupied))
    assert torch.equal(joint[:, :, -1].argmax(1), torch.full((2,), 17, device="cuda"))
    payload = torch.linspace(-.5, .7, joint.numel(), device="cuda").reshape_as(joint)
    actual_gradients = torch.autograd.grad((joint * payload).sum(), (means, scales, opacity, raw), retain_graph=True)
    reference_gradients = torch.autograd.grad((expected * payload).sum(), (means, scales, opacity, raw))
    for actual, reference in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-4)
        torch.testing.assert_close(actual[:, 2], torch.zeros_like(actual[:, 2]))
