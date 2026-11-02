from copy import deepcopy
import pytest
import torch
from torch.nn import functional as F

from model.encoder.gaussian_encoder.topk_module.adaptive_allocationv7 import (
    AdaptiveAllocationV7,
    gather_gaussian,
)
from model.encoder.gaussian_encoder.utils import GaussianPrediction
from loss.adaptive_allocation_v7_loss import (
    gaussian_field,
    local_gain_targets,
    AdaptiveAllocationV7Loss,
)


def example(batch=2, count=13, device="cpu", classes=3):
    torch.manual_seed(28)
    g = GaussianPrediction(
        means=torch.rand(batch, count, 3, device=device) * 0.7 + 0.2,
        scales=torch.rand(batch, count, 3, device=device) * 0.15 + 0.15,
        rotations=F.normalize(torch.randn(batch, count, 4, device=device), dim=-1),
        opacities=torch.rand(batch, count, 1, device=device) * 0.6 + 0.2,
        semantics=torch.rand(batch, count, classes, device=device) + 0.1,
    )
    features = torch.randn(batch, count, 8, device=device, requires_grad=True)
    anchor = torch.zeros(batch, count, 11 + classes, device=device)
    return features, anchor, g


def allocator(**kw):
    return AdaptiveAllocationV7(
        feat_embed_dim=8,
        semantic_dim=3,
        hidden_dim=12,
        pc_range=[0, 0, 0, 1.2, 1.2, 1.2],
        scale_range=[0.08, 0.6],
        grid_size=0.4,
        **kw
    )


@pytest.mark.parametrize("ratio", [0.0, 0.15, 0.6, 1.0])
@pytest.mark.parametrize("update", [False, True])
def test_exact_budget_both_modes_initialization_and_finite_backward(ratio, update):
    features, anchor, g = example()
    m = allocator(
        allocation_ratio=ratio,
        update_parent=update,
        probe_count=5,
        exploration_fraction=0,
    )
    a, out, f = m(features, anchor, g)
    k = int(13 * ratio)
    assert out.means.shape == (2, 13 + k, 3) and a.shape == (2, 13 + k, 14)
    assert f.shape == (2, 13 + k, 8)
    for name in ("means", "scales", "rotations", "opacities", "semantics"):
        torch.testing.assert_close(
            getattr(out, name)[:, :13], getattr(g, name), atol=2e-7, rtol=1e-6
        )
    if k:
        index = m.get_allocation_aux()["selected_index"]
        parent = gather_gaussian(g, index)
        torch.testing.assert_close(out.opacities[:, 13:], parent.opacities * 0.02)
        assert (out.means[:, 13:] - parent.means).norm(dim=-1).min() > 0
    loss = sum(
        getattr(out, name).sum()
        for name in ("means", "scales", "opacities", "semantics")
    )
    loss = loss + m.get_allocation_aux()["probe_score"].square().sum()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    assert all(torch.isfinite(value).all() for value in out[:5])
    m.eval()
    rng = torch.get_rng_state()
    step = m.allocation_step.clone()
    first = m(features, anchor, g)[1]
    second = m(features, anchor, g)[1]
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(
        step, m.allocation_step
    )
    assert m.get_allocation_aux() is None
    assert torch.equal(first.means, second.means)


def test_exploration_unique_cap_and_checkpoint_step_resume():
    f, a, g = example(count=31)
    m = allocator(
        allocation_ratio=0.6,
        max_new_gaussians=12,
        probe_count=0,
        exploration_fraction=1.0,
        exploration_steps=3,
    )
    for step in range(4):
        m(f, a, g)
        index = m.get_allocation_aux()["selected_index"]
        assert index.shape == (2, 12)
        assert all(row.unique().numel() == 12 for row in index)
    other = allocator(
        allocation_ratio=0.6,
        max_new_gaussians=12,
        probe_count=0,
        exploration_fraction=1.0,
        exploration_steps=3,
    )
    other.load_state_dict(m.state_dict(), strict=True)
    m(f, a, g)
    other(f, a, g)
    assert other.get_allocation_aux()["step"] == 4
    assert torch.equal(
        m.get_allocation_aux()["selected_index"],
        other.get_allocation_aux()["selected_index"],
    )


def test_parent_updates_bounded_and_anchor_semantic_roundtrip():
    f, a, g = example()
    m = allocator(allocation_ratio=0.6, probe_count=3)
    with torch.no_grad():
        m.generator.weight.normal_(0, 0.2)
    anchor, out, _ = m(f, a, g)
    assert (out.means >= 0).all() and (out.means < 1.2).all()
    assert (out.scales >= 0.08).all() and (out.scales <= 0.6).all()
    assert (out.opacities > 0).all() and (out.opacities < 1).all()
    assert (out.semantics > 0).all()
    torch.testing.assert_close(torch.sigmoid(anchor[..., :3]) * 1.2, out.means)
    torch.testing.assert_close(
        0.08 + 0.52 * torch.sigmoid(anchor[..., 3:6]), out.scales
    )
    torch.testing.assert_close(torch.sigmoid(anchor[..., 10:11]), out.opacities)
    torch.testing.assert_close(F.softplus(anchor[..., 11:]), out.semantics)


def test_frozen_module_statistics_stay_fixed_but_allocator_and_head_train():
    from model.segmentor.bev_segmentor import BEVSegmentor

    m = BEVSegmentor.__new__(BEVSegmentor)
    torch.nn.Module.__init__(m)
    m.freeze_img_backbone = m.freeze_img_neck = m.freeze_lifter = True
    m.freeze_encoder_anchor_encoder = m.freeze_encoder_except_densify = True
    m.frozen_modules_eval = True
    m.img_backbone = torch.nn.BatchNorm1d(3)
    m.img_neck = torch.nn.BatchNorm1d(3)
    m.lifter = torch.nn.BatchNorm1d(3)
    m.head = torch.nn.Identity()
    m.encoder = torch.nn.Module()
    m.encoder.anchor_encoder = torch.nn.BatchNorm1d(3)
    m.encoder.operation_order = ["norm", "densify"]
    m.encoder.layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(3), allocator()])
    m.train()
    assert not any(
        x.training
        for x in [
            m.img_backbone,
            m.img_neck,
            m.lifter,
            m.encoder.anchor_encoder,
            m.encoder.layers[0],
        ]
    )
    assert m.encoder.layers[1].training and m.head.training
    m.eval()
    m.train()
    assert not m.lifter.training and m.encoder.layers[1].training


def test_counterfactual_gain_matches_full_grid_ce_and_mask_and_has_no_target_grad():
    f, a, g = example()
    m = allocator(allocation_ratio=0.6, probe_count=7)
    with torch.no_grad():
        m.generator.weight.normal_(0, 0.01)
    m(f, a, g)
    aux = m.get_allocation_aux()
    xyz = torch.cartesian_prod(*[torch.arange(3) * 0.4 + 0.2] * 3)[None].expand(
        2, -1, -1
    )
    labels = torch.randint(0, 4, (2, 27))
    mask = torch.rand(2, 27) > 0.3
    base = gaussian_field(
        g, xyz[:, None].expand(-1, 13, -1, -1), aux["pc_range"], 0.4
    ).sum(1)
    base = F.pad(base, (0, 1))
    base[..., -1] = 0.7
    aux["base_logits"] = base
    actual = local_gain_targets(aux, xyz, labels, mask, radius=2)
    assert not actual.requires_grad
    before = F.cross_entropy(base.transpose(1, 2), labels, reduction="none")
    points = xyz[:, None].expand(-1, 7, -1, -1)
    delta = (
        gaussian_field(aux["probe_updated"], points, aux["pc_range"], 0.4)
        + gaussian_field(aux["probe_child"], points, aux["pc_range"], 0.4)
        - gaussian_field(aux["probe_parent"], points, aux["pc_range"], 0.4)
    )
    after = base[:, None] + F.pad(delta, (0, 1))
    ce = F.cross_entropy(
        after.flatten(0, 1).transpose(1, 2),
        labels[:, None].expand(-1, 7, -1).flatten(0, 1),
        reduction="none",
    ).reshape(2, 7, 27)
    expected = ((before[:, None] - ce) * mask[:, None]).sum(-1) / mask.sum(-1)[:, None]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-4)
    aux["output_logits"] = base.transpose(1, 2)
    loss, metrics = AdaptiveAllocationV7Loss()(
        dict(allocation_aux=[aux], sampled_xyz=xyz, sampled_label=labels, occ_mask=mask)
    )
    loss.backward()
    assert torch.isfinite(loss) and m.gain_head.weight.grad.abs().sum() > 0
    empty = local_gain_targets(aux, xyz, labels, torch.zeros_like(mask))
    assert torch.equal(empty, torch.zeros_like(empty))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_teacher_field_matches_rotated_anisotropic_native_renderer_and_replacement_gradients():
    from model.head.gaussian_head import GaussianHead

    torch.manual_seed(13)
    f, a, g = example(batch=1, device="cuda", classes=17)
    m = AdaptiveAllocationV7(
        feat_embed_dim=8,
        semantic_dim=17,
        hidden_dim=12,
        allocation_ratio=0.6,
        pc_range=[0, 0, 0, 1.2, 1.2, 1.2],
        scale_range=[0.08, 0.6],
        probe_count=5,
        exploration_fraction=0,
    ).cuda()
    with torch.no_grad():
        m.generator.weight.normal_(0, 0.01)
    rep = m.forward_terminal(f, a, g)
    rep["densify_parent_index"] = 0
    h = (
        GaussianHead(
            apply_loss_type="all",
            num_classes=18,
            with_empty=True,
            empty_args=dict(mean=[0.5, 0.5, 0.5], scale=[10, 10, 10]),
            use_localagg_react=True,
            analytic_covariance=True,
            separate_empty=True,
            incremental_densify=True,
            cache_render_grid=True,
            react_renderer="tiled",
            cuda_kwargs=dict(
                H=3,
                W=3,
                D=3,
                pc_min=[0, 0, 0],
                grid_size=0.4,
                scale_multiplier=3,
                validate_inputs=False,
            ),
        )
        .cuda()
        .train()
    )
    xyz = torch.cartesian_prod(*[torch.arange(3, device="cuda") * 0.4 + 0.2] * 3)[None]
    pure = h._render(g, xyz, h._react_context(xyz), include_empty=False)
    expected = gaussian_field(
        g, xyz[:, None].expand(-1, 13, -1, -1), [0, 0, 0, 1.2, 1.2, 1.2], 0.4
    ).sum(1)[0]
    torch.testing.assert_close(pure, F.pad(expected, (0, 1)), atol=2e-6, rtol=2e-6)
    meta = dict(
        occ_xyz=xyz.reshape(1, 3, 3, 3, 3),
        occ_label=torch.zeros(1, 3, 3, 3, device="cuda", dtype=torch.long),
    )
    first = h([dict(gaussian=g), rep], metas=meta)["pred_occ"][-1]
    params = tuple(m.parameters())
    probe = torch.randn_like(first)
    ga = torch.autograd.grad(first, params, probe, allow_unused=True, retain_graph=True)
    h.incremental_densify = False
    second = h([dict(gaussian=g), rep], metas=meta)["pred_occ"][-1]
    gb = torch.autograd.grad(second, params, probe, allow_unused=True)
    torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-6)
    for x, y in zip(ga, gb):
        if x is None:
            assert y is None
        else:
            torch.testing.assert_close(x, y, atol=5e-5, rtol=2e-4)
