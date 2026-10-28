"""Check factor separation and fixed displacement budgets in oracle diagnostics."""

from collections import namedtuple

import numpy as np
import torch

from tools.pmg_world_stage3.interventions import bounded_xy_displacement, make_variants


def test_l2_bound_and_no_z_correction():
    means = np.array([[0., 0., 2.], [1., 1., 3.]])
    targets = np.array([[3., 4., 9.], [1., 1., 8.]])
    delta = bounded_xy_displacement(means, targets, .8)
    np.testing.assert_allclose(delta[0], [.48, .64])
    np.testing.assert_array_equal(delta[1], [0., 0.])


def test_position_and_semantics_are_independent_and_births_unchanged():
    State = namedtuple("State", "means scales rotations opacities semantics")
    state = State(torch.zeros(1, 2, 3, 3), torch.ones(1, 2, 3, 3) * .2,
                  torch.tensor([1., 0., 0., 0.]).expand(1, 2, 3, 4),
                  torch.ones(1, 2, 3, 1), torch.arange(1., 103.).reshape(1, 2, 3, 17))
    mapping = dict(indices=np.array([0, 2]), targets=np.array([[1., 1., 0.], [-1., 0., 0.]]), labels=np.array([4, 7]))
    variants, checks = make_variants(state, mapping, current_queries=1, seed=17)
    assert torch.equal(variants["D11_r08"].means, variants["D10_r08"].means)
    assert torch.equal(variants["D11_r08"].semantics, variants["D01"].semantics)
    assert torch.equal(variants["D01"].means, state.means)
    torch.testing.assert_close(variants["D01"].semantics.sum(-1), state.semantics.sum(-1))
    for changed in variants.values():
        assert torch.equal(changed.means[:, 1:], state.means[:, 1:])
        assert torch.equal(changed.semantics[:, 1:], state.semantics[:, 1:])
        assert torch.equal(changed.means[..., 2], state.means[..., 2])
    assert checks["max_displacement_r08"] <= .80001
