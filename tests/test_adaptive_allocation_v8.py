"""Baseline parity, gradient isolation, deterministic selection and resume."""

from copy import deepcopy

import pytest
import torch

from loss.adaptive_allocation_v8_loss import AdaptiveAllocationV8Loss
from model.encoder.gaussian_encoder.topk_module.adaptive_allocationv8 import (
    AdaptiveAllocationV8,
)
from model.encoder.gaussian_encoder.topk_module.topk_densify_v2 import DensifyOnly
from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.encoder.gaussian_encoder.gaussian_encoder import GaussianOccEncoder
from model.utils.allocation_grad_clip import clip_model_grad_norm_


def inputs(device="cpu", count=23):
    generator = torch.Generator(device=device).manual_seed(12)
    rand = lambda width: torch.rand(2, count, width, device=device, generator=generator)
    gaussian = GaussianPrediction(
        means=rand(3) * 0.8 + 0.2,
        scales=rand(3) * 0.2 + 0.15,
        rotations=torch.nn.functional.normalize(rand(4), dim=-1),
        opacities=rand(1),
        semantics=rand(3) + 0.1,
    )
    gaussian = gaussian._replace(
        **{
            name: getattr(gaussian, name).requires_grad_()
            for name in ("means", "scales", "rotations", "opacities", "semantics")
        }
    )
    features = rand(8).requires_grad_()
    return features, rand(14), gaussian


def options():
    return dict(
        feat_embed_dim=8,
        semantic_dim=3,
        topk_count=7,
        pc_range=[0, 0, 0, 1.6, 1.6, 1.6],
        scale_range=[0.08, 0.64],
        unit_xyz=[4, 4, 1],
    )


def initialized_allocator(kind, device):
    settings = dict(type=kind, **options())
    if kind == "AdaptiveAllocationV8":
        settings["probe_count"] = 5
    encoder = GaussianOccEncoder(
        anchor_encoder=None,
        norm_layer=None,
        ffn=None,
        deformable_model=None,
        refine_layer=None,
        densify_layer=settings,
        operation_order=["densify"],
    ).to(device)
    encoder.init_weights()
    return encoder.layers[0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_baseline_initialization_rng_outputs_and_all_task_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(92)
    baseline = initialized_allocator("DensifyOnly", device)
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    torch.manual_seed(92)
    candidate = initialized_allocator("AdaptiveAllocationV8", device)
    assert torch.equal(rng, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    f, a, g = inputs(device)
    expected, _ = baseline(f, a, g, terminal=True)
    state = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    actual = candidate.forward_terminal(f, a, g)["gaussian"]
    assert torch.equal(
        state, torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    )
    for x, y in zip(expected[:5], actual[:5]):
        assert torch.equal(x, y)
    variables_a = (f,) + tuple(g[:5]) + tuple(baseline.parameters())
    variables_b = (
        (f,)
        + tuple(g[:5])
        + tuple(
            p
            for n, p in candidate.named_parameters()
            if not n.startswith("score_network.")
        )
    )
    loss_a = sum(value.square().sum() for value in expected[:5])
    loss_b = sum(value.square().sum() for value in actual[:5])
    ga = torch.autograd.grad(loss_a, variables_a, retain_graph=True)
    gb = torch.autograd.grad(loss_b, variables_b)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_teacher_scores_cannot_backpropagate_to_features_or_child_generator():
    module = AdaptiveAllocationV8(**options(), probe_count=9)
    f, a, g = inputs()
    module.forward_terminal(f, a, g)
    aux = module.get_allocation_aux()
    aux["probe_score"].square().sum().backward()
    assert f.grad is None
    assert all(value.grad is None for value in g[:5])
    for name, p in module.named_parameters():
        if name.startswith("score_network."):
            assert p.grad is not None and torch.isfinite(p.grad).all()
        else:
            assert p.grad is None
    assert all(not value.requires_grad for value in aux["probe_child"][:5])


def test_warmup_transition_quota_eval_determinism_and_resume():
    module = AdaptiveAllocationV8(
        **options(),
        probe_count=5,
        selection_warmup_steps=2,
        selection_transition_steps=2
    )
    f, a, g = inputs()
    original = g.opacities.squeeze(-1).topk(7, 1).indices
    for step in range(6):
        out = module.forward_terminal(f, a, g)
        aux = module.get_allocation_aux()
        assert aux["step"] == step
        assert aux["learned_fraction"] == [0, 0, 0, 0.5, 1, 1][step]
        if step <= 2:
            assert torch.equal(aux["selected_index"], original)
        assert all(row.unique().numel() == 7 for row in aux["selected_index"])
        for name in ("means", "scales", "rotations", "opacities", "semantics"):
            value = getattr(out["gaussian"], name)
            assert value.shape[1] == 30
            assert torch.equal(value[:, :23], getattr(g, name))
    clone = AdaptiveAllocationV8(
        **options(),
        probe_count=5,
        selection_warmup_steps=2,
        selection_transition_steps=2
    )
    clone.load_state_dict(deepcopy(module.state_dict()), strict=True)
    module.eval()
    clone.eval()
    rng = torch.get_rng_state()
    first = module.forward_terminal(f, a, g)["gaussian"]
    second = clone.forward_terminal(f, a, g)["gaussian"]
    for x, y in zip(first[:5], second[:5]):
        assert torch.equal(x, y)
    assert torch.equal(rng, torch.get_rng_state())
    assert module._current_step() == clone._current_step() == 6
    assert module.get_allocation_aux() is None


@pytest.mark.parametrize("count,budget,probes", [(3, 7, 5), (0, 7, 0), (23, 0, 0)])
def test_small_empty_and_zero_budget(count, budget, probes):
    settings = options()
    settings["topk_count"] = budget
    module = AdaptiveAllocationV8(**settings, probe_count=probes)
    f, a, g = inputs(count=count)
    result = module.forward_terminal(f, a, g)
    assert result["gaussian"].means.shape[1] == count + min(count, budget)


def test_real_local_teacher_loss_is_finite_and_scorer_only():
    module = AdaptiveAllocationV8(**options(), probe_count=5)
    f, a, g = inputs()
    module.forward_terminal(f, a, g)
    aux = module.get_allocation_aux()
    axis = torch.arange(4) * 0.4 + 0.2
    xyz = (
        torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
        .reshape(1, -1, 3)
        .repeat(2, 1, 1)
    )
    aux.update(base_logits=torch.zeros(2, 64, 4), output_logits=torch.zeros(2, 4, 64))
    loss, metrics = AdaptiveAllocationV8Loss()(
        dict(
            allocation_aux=[aux],
            sampled_xyz=xyz,
            sampled_label=torch.zeros(2, 64, dtype=torch.long),
        )
    )
    assert torch.isfinite(loss) and "missing_allocation_aux" not in metrics
    loss.backward()
    assert f.grad is None
    assert all(
        p.grad is None
        for n, p in module.named_parameters()
        if not n.startswith("score_network.")
    )


@pytest.mark.parametrize("separate", [False, True])
def test_gradient_clipping_isolates_score_norm_and_preserves_default(separate):
    module = AdaptiveAllocationV8(**options())
    reference = deepcopy(module)
    for name, parameter in module.named_parameters():
        parameter.grad = torch.full_like(
            parameter, 10 if name.startswith("score_network.") else 1
        )
    for name, parameter in reference.named_parameters():
        parameter.grad = torch.full_like(
            parameter, 10 if name.startswith("score_network.") else 1
        )
    target = [
        p
        for n, p in reference.named_parameters()
        if not separate or not n.startswith("score_network.")
    ]
    expected_norm = torch.nn.utils.clip_grad_norm_(target, 1)
    actual_norm = clip_model_grad_norm_(module, 1, separate_allocation_score=separate)
    torch.testing.assert_close(actual_norm, expected_norm, atol=0, rtol=0)
    for name, parameter in module.named_parameters():
        if not separate or not name.startswith("score_network."):
            assert torch.equal(
                parameter.grad, dict(reference.named_parameters())[name].grad
            )
    if separate:
        norm = torch.stack(
            [p.grad.norm() for p in module.allocation_score_parameters()]
        ).norm()
        assert float(norm) <= 1.000001
