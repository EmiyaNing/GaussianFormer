"""Check native equivalence, the actual residual bound and preserved rollout."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from model.segmentor.pmg_native_state_adapter_world import (
    NativeStateAdapter, PMGNativeStateAdapterWorld)


def test_zero_adapter_preserves_native_function_and_input_gradients():
    torch.manual_seed(17)
    rng = torch.get_rng_state().clone()
    adapter = NativeStateAdapter()
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 24896
    inputs = [torch.randn(*shape, requires_grad=True) for shape in
              ((2, 7, 256), (2, 1, 256), (2, 7, 256))]
    weight = torch.randn(2, 7, 256)
    native = inputs[0] + inputs[1] + inputs[2]
    native_grad = torch.autograd.grad((native * weight).sum(), inputs)
    result, ratio = adapter.update(*inputs, normalized_time=1 / 6)
    assert torch.equal(native, result)
    assert torch.equal(ratio, torch.zeros_like(ratio))
    gradients = torch.autograd.grad((result * weight).sum(),
                                    inputs + list(adapter.parameters()))
    for expected, actual in zip(native_grad, gradients[:3]):
        assert torch.equal(expected, actual)
    assert all(torch.isfinite(value).all() for value in gradients)
    # A zero up-projection deliberately delays down-layer learning until the
    # up-projection has moved. Both output tensors already receive gradients.
    assert all(torch.count_nonzero(value) == 0 for value in gradients[3:5])
    assert all(value.norm() > 0 for value in gradients[5:])


def test_vector_residual_bound_direction_and_zero_intervention():
    adapter = NativeStateAdapter()
    state = torch.randn(2, 7, 256)
    fused, position = torch.randn(2, 1, 256), torch.randn_like(state)
    with torch.no_grad():
        adapter.up.weight.normal_(std=3.)
        adapter.up.bias.normal_(std=3.)
    increment = fused + position
    delta = adapter(state, increment, .5)
    assert torch.all(delta.norm(dim=-1) <= .1 * increment.norm(dim=-1) + 1e-6)
    # A vector innovation has useful directions unavailable to scalar gating.
    cosine = torch.nn.functional.cosine_similarity(delta, increment, dim=-1)
    assert torch.all(cosine.abs() < .95)
    result, ratio = adapter.update(state, fused, position, .5)
    assert ratio.max() <= .1 + 1e-7
    assert not torch.equal(result, state + fused + position)
    recorder = SimpleNamespace(native_state_adapter=adapter, last_adapter_statistics=[])
    PMGNativeStateAdapterWorld._record_adapter(recorder, 3, ratio)
    json.dumps(recorder.last_adapter_statistics, allow_nan=False)
    assert recorder.last_adapter_statistics[0]["query_evaluations"] == 14
    original, zeros = adapter.update(state, fused, position, .5, force_zero=True)
    assert torch.equal(original, state + fused + position)
    assert torch.equal(zeros, torch.zeros_like(zeros))


def test_zero_increment_and_gradients_through_frozen_readout():
    adapter = NativeStateAdapter()
    with torch.no_grad():
        adapter.up.weight.normal_(std=.01)
    state = torch.randn(1, 4, 256, requires_grad=True)
    zero = torch.zeros_like(state)
    result, ratio = adapter.update(state, zero, zero, .5)
    assert torch.equal(result, state)
    assert ratio.count_nonzero() == 0
    assert all(torch.isfinite(value).all() for value in
               adapter.last_diagnostics.values())
    # Frozen native weights still transmit a gradient to the adapted state.
    readout = torch.nn.Linear(256, 17).requires_grad_(False)
    result, _ = adapter.update(state, torch.ones_like(state), zero, .5)
    readout(result).square().sum().backward()
    assert all(value.grad is None for value in readout.parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               and value.grad.norm() > 0 for value in adapter.parameters())


def test_forecast_loop_keeps_native_birth_detach_and_output_logic():
    root = Path(__file__).resolve().parents[1]

    def loop(relative_path):
        tree = ast.parse((root / relative_path).read_text())
        forecast = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "_forecast")
        return next(node for node in forecast.body if isinstance(node, ast.For))

    original = loop("model/segmentor/sparseworld_strict.py")
    adapted = loop("model/segmentor/pmg_native_state_adapter_world.py")
    normalized = []
    for node in adapted.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            if [item.id for item in node.targets[0].elts] == ["features", "ratio"]:
                normalized.append(ast.parse("features = features + fused + position").body[0])
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            function = node.value.func
            if isinstance(function, ast.Attribute) and function.attr == "_record_adapter":
                continue
        normalized.append(node)
    adapted.body = normalized
    assert ast.dump(adapted, include_attributes=False) == ast.dump(original, include_attributes=False)
