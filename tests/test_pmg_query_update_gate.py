"""Functional initialization, gradient and selectivity checks for the new gate."""

import torch

from model.segmentor.pmg_query_update_gate_world import QueryUpdateGate


def test_zero_gate_preserves_native_function_and_input_gradients():
    torch.manual_seed(17)
    rng = torch.get_rng_state().clone()
    gate = QueryUpdateGate()
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in gate.parameters()) == 514
    inputs = [torch.randn(*shape, requires_grad=True) for shape in
              ((2, 7, 256), (2, 1, 256), (2, 7, 256))]
    weight = torch.randn(2, 7, 256)
    native = inputs[0] + inputs[1] + inputs[2]
    native_grad = torch.autograd.grad((native * weight).sum(), inputs)
    result, scale = gate.update(*inputs, normalized_time=1/6)
    assert torch.equal(native, result)
    assert torch.equal(scale, torch.ones_like(scale))
    gradients = torch.autograd.grad((result * weight).sum(), inputs + list(gate.parameters()))
    for expected, actual in zip(native_grad, gradients[:3]):
        assert torch.equal(expected, actual)
    assert all(torch.isfinite(x).all() and x.norm() > 0 for x in gradients[3:])


def test_query_dependent_scaling_and_unit_intervention():
    gate = QueryUpdateGate()
    state = torch.zeros(1, 2, 256)
    state[0, 0, 0], state[0, 1, 0] = 1., -1.
    with torch.no_grad():
        gate.linear.weight[0, 0] = .1
    fused, position = torch.ones(1, 1, 256), torch.zeros_like(state)
    result, scale = gate.update(state, fused, position, .5)
    assert .5 < scale.min() < 1 < scale.max() < 1.5
    assert not torch.equal(result, state + fused + position)
    original, units = gate.update(state, fused, position, .5, force_unit=True)
    assert torch.equal(original, state + fused + position)
    assert torch.equal(units, torch.ones_like(units))
