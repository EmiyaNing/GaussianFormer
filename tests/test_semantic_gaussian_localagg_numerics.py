"""P0.1 numerical audit of the Phase-B semantic Gaussian CUDA local aggregator.

Plan reference: ``plans/sparse_render_plans/P0_correctness_and_cost_audit.md``
task T0.1.

The CUDA operator only accepts float32, so the audit is layered:

* :func:`test_reference_gradcheck_float64` proves the *reference* implements a
  differentiable function and that autograd's gradient is the analytic one.
* :func:`test_cuda_forward_matches_reference` and
  :func:`test_cuda_backward_matches_reference_autograd` compare the CUDA op
  against that reference in float32.
* :func:`test_cuda_backward_matches_finite_difference` validates the
  hand-written CUDA backward directly against central differences of the CUDA
  forward, so a shared mistake in reference and kernel cannot hide.

Nothing here is imported by training code.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarks.reference.g2q_bruteforce import (
    ellipsoid_reference, g2q_bruteforce, g2q_capped_reference, g2q_cuda,
    neighbour_cells)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason='requires CUDA')

PC_RANGE = (-10.0, -10.0, -5.0, 10.0, 10.0, 5.0)

# Float32 accumulation in the CUDA kernel sums over a different order than the
# reference's matmul, so comparisons are absolute-plus-relative.  The absolute
# floor matters because a query whose support is nearly empty produces a
# residual close to zero, where a pure relative error is meaningless.
FORWARD_ATOL = 1e-5
FORWARD_RTOL = 1e-4


def make_scene(count=64, batch=1, classes=5, seed=0, dtype=torch.float32,
               device='cuda', scale_range=(0.2, 0.8), pc_range=PC_RANGE,
               invalid_fraction=0.0, isolated=False):
    """Build a reproducible synthetic Gaussian scene inside ``pc_range``."""
    generator = torch.Generator(device=device).manual_seed(seed)
    low = torch.tensor(pc_range[:3], device=device, dtype=dtype)
    high = torch.tensor(pc_range[3:], device=device, dtype=dtype)
    points = low + (high - low) * torch.rand(
        batch, count, 3, generator=generator, device=device, dtype=dtype)
    if isolated:
        # Spread points far apart along a line so no support overlaps.
        points = torch.zeros(batch, count, 3, device=device, dtype=dtype)
        points[..., 0] = torch.arange(count, device=device, dtype=dtype) * 50.0
    lo, hi = scale_range
    scales = lo + (hi - lo) * torch.rand(
        batch, count, 3, generator=generator, device=device, dtype=dtype)
    opacities = torch.rand(
        batch, count, generator=generator, device=device, dtype=dtype)
    semantic_seed = torch.randn(
        batch, count, classes, generator=generator, device=device, dtype=dtype)
    valid = torch.ones(batch, count, dtype=torch.bool, device=device)
    if invalid_fraction > 0:
        drop = torch.rand(batch, count, generator=generator, device=device) \
            < invalid_fraction
        valid &= ~drop
    return points, scales, opacities, semantic_seed, valid


def report(actual, expected, label):
    """Print and return the max absolute / max relative (floored) error."""
    difference = (actual - expected).abs()
    absolute = difference.max().item()
    relative = (difference / expected.abs().clamp_min(1e-2)).max().item()
    print(f'  {label}: max abs err = {absolute:.3e}, max rel err (floor 1e-2) '
          f'= {relative:.3e}')
    return absolute, relative


def assert_matches(actual, expected, label):
    absolute, relative = report(actual, expected, label)
    assert torch.allclose(actual, expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL), (
        f'{label}: max abs err {absolute:.3e} / max rel err {relative:.3e} '
        f'exceeds rtol={FORWARD_RTOL}, atol={FORWARD_ATOL}')


# --------------------------------------------------------------------------
# The reference itself
# --------------------------------------------------------------------------

def test_reference_gradcheck_float64():
    """The reference must be a smooth, correctly differentiated function."""
    torch.manual_seed(0)
    points, scales, opacities, seed, _ = make_scene(
        count=10, classes=3, seed=1, dtype=torch.float64, device='cpu')
    closed = torch.ones(1, 10, dtype=torch.bool)

    def function(p, s, o, k):
        return g2q_bruteforce(p, s, o, k, closed, query_chunk=4)

    assert torch.autograd.gradcheck(
        function,
        (points.requires_grad_(), scales.requires_grad_(),
         opacities.requires_grad_(), seed.requires_grad_()),
        eps=1e-6, atol=1e-6, rtol=1e-4)


def test_reference_self_term_is_the_dominant_neighbour_when_isolated():
    """With ``include_self`` the self Gaussian always participates.

    An isolated point therefore returns (approximately) its own seed: the only
    term in the normalized average is ``opacity * seed / opacity``.  This is
    the expected behaviour, and it is why "no neighbour" has to be tested with
    ``include_self=False``.
    """
    points, scales, opacities, seed, valid = make_scene(
        count=6, classes=4, seed=4, isolated=True, device='cpu')
    out = g2q_bruteforce(points, scales, opacities, seed, valid)
    expected = opacities.unsqueeze(-1) * seed / (opacities.unsqueeze(-1) + 1e-6)
    assert torch.allclose(out, expected, atol=1e-5)
    assert not torch.allclose(out, torch.zeros_like(out), atol=1e-3)


def test_reference_no_neighbour_returns_zero_without_self():
    points, scales, opacities, seed, valid = make_scene(
        count=6, classes=4, seed=4, isolated=True, device='cpu')
    out = g2q_bruteforce(points, scales, opacities, seed, valid,
                         include_self=False)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12)


def test_reference_flags_invalid_queries_as_zero():
    points, scales, opacities, seed, valid = make_scene(
        count=16, classes=3, seed=5, invalid_fraction=0.5, device='cpu')
    out = g2q_bruteforce(points, scales, opacities, seed, valid)
    assert torch.allclose(out[~valid], torch.zeros_like(out[~valid]), atol=1e-12)


# --------------------------------------------------------------------------
# CUDA forward vs reference
# --------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('count', [1, 2, 17, 256])
def test_cuda_forward_matches_reference(seed, count):
    """Forward error must stay in float32 rounding territory."""
    torch.backends.cuda.matmul.allow_tf32 = False
    points, scales, opacities, seed_t, valid = make_scene(
        count=count, classes=7, seed=seed)
    reference = g2q_bruteforce(points, scales, opacities, seed_t, valid)
    actual = g2q_cuda(points, scales, opacities, seed_t, valid, PC_RANGE,
                      max_scale=0.8)
    assert_matches(actual, reference, f'forward seed={seed} G={count}')


@requires_cuda
def test_cuda_forward_matches_reference_batch_and_determinism():
    points, scales, opacities, seed_t, valid = make_scene(
        count=64, batch=3, classes=7, seed=7, invalid_fraction=0.25)
    reference = g2q_bruteforce(points, scales, opacities, seed_t, valid)
    first = g2q_cuda(points, scales, opacities, seed_t, valid, PC_RANGE,
                     max_scale=0.8)
    assert_matches(first, reference, 'forward batch=3')

    # 100 repeated runs must be bit-identical (the forward path has no
    # atomics, so any drift would indicate an uninitialized read or a race).
    for _ in range(100):
        again = g2q_cuda(points, scales, opacities, seed_t, valid, PC_RANGE,
                         max_scale=0.8)
        assert torch.equal(again, first), 'forward is not deterministic'


# --------------------------------------------------------------------------
# CUDA backward vs reference autograd and finite differences
# --------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize('dense', [False, True])
def test_cuda_backward_matches_reference_autograd(dense):
    """CUDA backward must equal autograd through the exact reference.

    Both a sparse and a dense scene are checked.  The sparse random scene is
    the realistic one but its ``grad_scales`` are tiny, so a wrong scale
    gradient could pass unnoticed; the dense cluster makes all three gradients
    large enough that the comparison is meaningful.
    """
    if dense:
        pc_range = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
        base = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1)
                             for k in (0, 1)], dtype=torch.float32,
                            device='cuda')
        points = (base * 0.3).unsqueeze(0)
        scales = (torch.full((1, 8, 3), 0.5, device='cuda')
                  * torch.tensor([0.9, 1.0, 1.1], device='cuda'))
        opacities = torch.linspace(0.3, 0.9, 8, device='cuda').unsqueeze(0)
        generator = torch.Generator(device='cuda').manual_seed(11)
        seed_t = torch.randn(1, 8, 3, generator=generator, device='cuda')
        valid = torch.ones(1, 8, dtype=torch.bool, device='cuda')
        label = 'dense'
    else:
        pc_range = PC_RANGE
        points, scales, opacities, seed_t, valid = make_scene(
            count=128, classes=7, seed=11)
        label = 'sparse'
    upstream = torch.randn_like(seed_t)

    ref_scales = scales.clone().requires_grad_()
    ref_opacities = opacities.clone().requires_grad_()
    ref_seed = seed_t.clone().requires_grad_()
    g2q_bruteforce(points, ref_scales, ref_opacities, ref_seed, valid) \
        .backward(upstream)

    cuda_scales = scales.clone().requires_grad_()
    cuda_opacities = opacities.clone().requires_grad_()
    cuda_seed = seed_t.clone().requires_grad_()
    g2q_cuda(points, cuda_scales, cuda_opacities, cuda_seed, valid, pc_range,
             max_scale=0.8).backward(upstream)

    for name, got, want in (
            ('grad_scales', cuda_scales.grad, ref_scales.grad),
            ('grad_opacities', cuda_opacities.grad, ref_opacities.grad),
            ('grad_seeds', cuda_seed.grad, ref_seed.grad)):
        absolute, relative = report(got, want, f'{label} {name}')
        assert absolute < 1e-4, f'{label} {name} abs error {absolute:.3e} too large'
        if dense:
            # A vacuous comparison would pass even if the kernel were wrong.
            assert got.abs().max().item() > 1e-4, (
                f'{label} {name} is degenerate (all near zero); the comparison '
                'would not detect a wrong gradient')


@requires_cuda
def test_cuda_backward_matches_finite_difference():
    """Central differences of the CUDA forward vs the CUDA backward kernel.

    Two details make this measurable in float32:

    * the scene is *dense* (a small ``pc_range``) so that non-self neighbours
      exist and the scale gradient is not identically zero;
    * the scalar objective is a **mean**, not a sum.  A sum over a few dozen
      O(1) outputs is ~O(10), whose float32 spacing (~1e-6) swamps the
      ~1e-7 loss change produced by perturbing an opacity whose gradient is
      ~1e-4 -- which is exactly how the first version of this check failed.

    Every parameter entry is probed, and the pass criterion is aggregate:
    the largest absolute deviation must be small relative to the largest
    analytic gradient, so near-zero entries cannot dominate through a relative
    error.
    """
    dense_pc = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    # Deterministic 2x2x2 cluster with 0.3 m spacing.  Every pair sits far
    # inside the support box, so perturbing a scale cannot flip a neighbour
    # in or out of the support set.  That matters because the box cutoff is a
    # hard indicator: finite differences would see the jump, while the
    # analytic gradient (like any autograd derivative of a piecewise-constant
    # mask) does not.  Random scenes put many pairs exactly on that boundary
    # and therefore cannot be used for a scale gradient check.
    spacing, sigma = 0.3, 3.0
    base = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1)
                         for k in (0, 1)], dtype=torch.float32, device='cuda')
    points = (base * spacing).unsqueeze(0)
    scales = (torch.full((1, 8, 3), 0.5, device='cuda')
              * torch.tensor([0.9, 1.0, 1.1], device='cuda'))
    opacities = torch.linspace(0.3, 0.9, 8, device='cuda').unsqueeze(0)
    generator = torch.Generator(device='cuda').manual_seed(13)
    seed_t = torch.randn(1, 8, 3, generator=generator, device='cuda')
    valid = torch.ones(1, 8, dtype=torch.bool, device='cuda')

    # Self-check the precondition that makes the finite differences valid.
    relative = ((points[:, :, None, :] - points[:, None, :, :])
                / scales[:, None, :, :]).abs()
    worst_relative = relative.max().item()
    assert worst_relative < 0.5 * sigma, (
        f'cluster is too close to the support boundary ({worst_relative:.3f} '
        f'vs cutoff {sigma}); finite differences would straddle the cutoff')
    print(f'  support margin: max |r/s| = {worst_relative:.3f} vs cutoff {sigma}')

    upstream = torch.randn_like(seed_t)

    cuda_scales = scales.clone().requires_grad_()
    cuda_opacities = opacities.clone().requires_grad_()
    cuda_seed = seed_t.clone().requires_grad_()
    # The objective must match the finite-difference objective exactly.  Using
    # ``.backward(upstream)`` here would mean an implicit *sum*, while the
    # finite differences below differentiate a *mean* -- a factor of
    # ``out.numel()`` that the first version of this check got wrong.
    (g2q_cuda(points, cuda_scales, cuda_opacities, cuda_seed, valid, dense_pc,
              max_scale=0.8) * upstream).mean().backward()

    def loss(scales_variant, opacity_variant, seed_variant):
        out = g2q_cuda(points, scales_variant, opacity_variant, seed_variant,
                       valid, dense_pc, max_scale=0.8)
        return (out * upstream).mean()

    def numeric_gradient(tensor, step, kind):
        numeric = torch.zeros_like(tensor)
        flat = numeric.flatten()
        for flat_index in range(tensor.numel()):
            index = tuple(int(v) for v in torch.unravel_index(
                torch.tensor(flat_index), tensor.shape))
            plus, minus = tensor.clone(), tensor.clone()
            plus[index] += step
            minus[index] -= step
            if kind == 'scale':
                value = (loss(plus, opacities, seed_t)
                         - loss(minus, opacities, seed_t)) / (2 * step)
            elif kind == 'opacity':
                value = (loss(scales, plus, seed_t)
                         - loss(scales, minus, seed_t)) / (2 * step)
            else:
                value = (loss(scales, opacities, plus)
                         - loss(scales, opacities, minus)) / (2 * step)
            flat[flat_index] = value
        return numeric

    # Steps are chosen per parameter type: opacity gradients are ~1e-4, so a
    # larger step is needed to lift the loss difference above float32 spacing.
    for kind, tensor, analytic, step in (
            ('scale', scales, cuda_scales.grad, 1e-3),
            ('opacity', opacities, cuda_opacities.grad, 5e-3),
            ('seed', seed_t, cuda_seed.grad, 1e-3)):
        numeric = numeric_gradient(tensor, step, kind)
        deviation = (numeric - analytic).abs().max().item()
        reference = analytic.abs().max().item()
        print(f'  {kind}: |analytic|max={reference:.3e} '
              f'max|numeric-analytic|={deviation:.3e}')
        if kind == 'scale':
            assert reference > 1e-6, (
                'the finite-difference scene produced an all-zero scale '
                'gradient, so this check would be vacuous')
        assert deviation < 0.05 * reference + 1e-5, (
            f'{kind} finite-difference mismatch: deviation {deviation:.3e} '
            f'vs |analytic|max {reference:.3e}')


@requires_cuda
def test_cuda_returns_no_point_gradient():
    """Documented contract: Gaussian centres are an anchored, non-diff input."""
    points, scales, opacities, seed_t, valid = make_scene(
        count=32, classes=3, seed=17)
    tracked = points.clone().requires_grad_()
    g2q_cuda(tracked, scales, opacities, seed_t, valid, PC_RANGE, max_scale=0.8) \
        .sum().backward()
    assert tracked.grad is None, (
        'point coordinates unexpectedly receive a gradient; update the operator '
        'contract and the plan if this is intentional')


# --------------------------------------------------------------------------
# Support-domain semantics
# --------------------------------------------------------------------------

@requires_cuda
def test_support_cutoff_is_a_per_axis_box_not_an_ellipsoid():
    """A neighbour at the box corner ``(3s, 3s, 0)`` must be included.

    With ``include_self=False`` and a single neighbour the two support-domain
    definitions differ maximally: the box keeps the neighbour (so the query
    returns the neighbour's seed) while the ellipsoid drops it (so the query
    returns exactly zero).
    """
    count, classes, sigma, scale = 2, 3, 3.0, 0.5
    points = torch.zeros(1, count, 3, device='cuda')
    points[0, 1, 0] = sigma * scale   # exactly on the box corner in x ...
    points[0, 1, 1] = sigma * scale   # ... and in y, so ||r/s||^2 = 18 > 9
    scales = torch.full((1, count, 3), scale, device='cuda')
    opacities = torch.full((1, count), 0.5, device='cuda')
    seed = torch.zeros(1, count, classes, device='cuda')
    seed[0, 1] = torch.tensor([1.0, -2.0, 0.5], device='cuda')
    valid = torch.ones(1, count, dtype=torch.bool, device='cuda')

    box = g2q_bruteforce(points, scales, opacities, seed, valid,
                         support_sigma=sigma, include_self=False)
    ellipsoid = ellipsoid_reference(points, scales, opacities, seed, valid,
                                    support_sigma=sigma, include_self=False)

    # The box keeps the corner neighbour, so the query returns its seed scaled
    # by the single-term normalization w / (w + denom_eps).  With
    # w = 0.5 * exp(-9) = 6.17e-5 and denom_eps = 1e-6 that factor is ~0.984,
    # so this is a non-zero result proportional to the neighbour's seed.
    weight = 0.5 * math.exp(-0.5 * 2 * sigma ** 2)
    expected = seed[0, 1] * (weight / (weight + 1e-6))
    assert torch.allclose(box[0, 0], expected, atol=1e-6), \
        'box reference did not keep the corner neighbour'
    assert box[0, 0].abs().max().item() > 1e-3, 'box result should be non-zero'
    assert torch.allclose(ellipsoid[0, 0], torch.zeros(classes, device='cuda'),
                          atol=1e-6), 'ellipsoid reference should drop it'

    actual = g2q_cuda(points, scales, opacities, seed, valid, PC_RANGE,
                      support_sigma=sigma, max_scale=scale, include_self=False)
    assert_matches(actual, box, 'corner neighbour (box semantics)')


@requires_cuda
def test_support_boundary_is_inclusive():
    """``|r_d| == support_sigma * s_d`` is included (the kernel uses ``<=``)."""
    sigma, scale, classes = 3.0, 0.4, 2
    points = torch.zeros(1, 2, 3, device='cuda')
    points[0, 1, 0] = sigma * scale            # exactly on the boundary
    scales = torch.full((1, 2, 3), scale, device='cuda')
    opacities = torch.full((1, 2), 1.0, device='cuda')
    seed = torch.ones(1, 2, classes, device='cuda')
    valid = torch.ones(1, 2, dtype=torch.bool, device='cuda')

    out = g2q_cuda(points, scales, opacities, seed, valid, PC_RANGE,
                   support_sigma=sigma, max_scale=scale)
    # Both weights are 1.0 (self) and exp(-4.5) (boundary), seeds are equal, so
    # the normalized result is 1.0 either way; use distinct seeds instead.
    seed[0, 1] = 0.0
    out = g2q_cuda(points, scales, opacities, seed, valid, PC_RANGE,
                   support_sigma=sigma, max_scale=scale)
    expected = 1.0 / (1.0 + math.exp(-0.5 * sigma ** 2))
    assert abs(out[0, 0, 0].item() - expected) < 1e-5

    # Nudge just outside the box: the neighbour must drop out entirely.
    points[0, 1, 0] = sigma * scale + 1e-3
    out_far = g2q_cuda(points, scales, opacities, seed, valid, PC_RANGE,
                       support_sigma=sigma, max_scale=scale)
    assert abs(out_far[0, 0, 0].item() - 1.0) < 1e-5


@requires_cuda
@pytest.mark.parametrize('cell_size', [0.6, 1.2, 1.8, 2.4])
@pytest.mark.parametrize('shift', [0.0, 0.3, 0.9])
def test_binning_is_exact_under_pc_range_shifts(cell_size, shift):
    """Cell size and grid origin must not change the result (binning is exact)."""
    points, scales, opacities, seed_t, valid = make_scene(
        count=200, classes=4, seed=19)
    points = points + shift
    reference = g2q_bruteforce(points, scales, opacities, seed_t, valid)
    pc_range = tuple(x + shift for x in PC_RANGE)
    actual = g2q_cuda(points, scales, opacities, seed_t, valid, pc_range,
                      cell_size=cell_size, max_scale=0.8)
    assert_matches(actual, reference, f'cell_size={cell_size} shift={shift}')


@requires_cuda
def test_points_exactly_on_cell_boundaries():
    """Points landing exactly on cell edges must still aggregate correctly."""
    cell_size, scale, classes = 1.0, 0.5, 2
    grid = torch.arange(8, device='cuda', dtype=torch.float32) * 0.5
    points = torch.zeros(1, 8, 3, device='cuda')
    points[0, 1:, 0] = grid[1:]
    scales = torch.full((1, 8, 3), scale, device='cuda')
    opacities = torch.full((1, 8), 0.7, device='cuda')
    seed = torch.ones(1, 8, classes, device='cuda')
    valid = torch.ones(1, 8, dtype=torch.bool, device='cuda')
    reference = g2q_bruteforce(points, scales, opacities, seed, valid)
    actual = g2q_cuda(points, scales, opacities, seed, valid, PC_RANGE,
                      cell_size=cell_size, max_scale=scale)
    assert_matches(actual, reference, 'cell-boundary points')


@requires_cuda
def test_out_of_range_points_are_handled():
    """Points outside ``pc_range`` are clamped into border cells; results hold."""
    points, scales, opacities, seed_t, valid = make_scene(
        count=128, classes=3, seed=23)
    points[0, :8] = torch.tensor([[-50.0, 0.0, 0.0]], device='cuda')
    points[0, 8:16] = torch.tensor([[50.0, 0.0, 0.0]], device='cuda')
    reference = g2q_bruteforce(points, scales, opacities, seed_t, valid)
    actual = g2q_cuda(points, scales, opacities, seed_t, valid, PC_RANGE,
                      max_scale=0.8)
    assert_matches(actual, reference, 'out-of-range points')


@requires_cuda
def test_include_self_toggle():
    points, scales, opacities, seed_t, valid = make_scene(
        count=64, classes=3, seed=29)
    for include_self in (True, False):
        reference = g2q_bruteforce(points, scales, opacities, seed_t, valid,
                                   include_self=include_self)
        actual = g2q_cuda(points, scales, opacities, seed_t, valid, PC_RANGE,
                          include_self=include_self, max_scale=0.8)
        assert_matches(actual, reference, f'include_self={include_self}')


# --------------------------------------------------------------------------
# Candidate-completeness and dtype contracts
# --------------------------------------------------------------------------

@requires_cuda
def test_binning_completeness_requires_adequate_max_scale():
    """An undersized ``max_scale`` silently drops true neighbours.

    This is the silent-failure mode the plan flags: the cell scan radius is
    ``ceil(support_sigma * max_scale / cell_size)``, so a caller that
    under-reports ``max_scale`` loses pairs without any error being raised.
    Seeds are made distinct here; with identical seeds a dropped neighbour
    would not change the normalized average.
    """
    cell_size, scale, classes, sigma = 0.5, 0.8, 2, 3.0
    points = torch.zeros(1, 2, 3, device='cuda')
    points[0, 1, 0] = 1.5            # inside support (|r| = 1.5 <= 2.4)
    scales = torch.full((1, 2, 3), scale, device='cuda')
    opacities = torch.full((1, 2), 1.0, device='cuda')
    seed = torch.zeros(1, 2, classes, device='cuda')
    seed[0, 0] = 1.0                 # self seed
    seed[0, 1] = 0.0                 # neighbour seed
    valid = torch.ones(1, 2, dtype=torch.bool, device='cuda')

    reference = g2q_bruteforce(points, scales, opacities, seed, valid,
                               support_sigma=sigma)
    good = g2q_cuda(points, scales, opacities, seed, valid, PC_RANGE,
                    cell_size=cell_size, support_sigma=sigma, max_scale=scale)
    assert_matches(good, reference, 'adequate max_scale')
    assert neighbour_cells(sigma, scale, cell_size) == 5

    bad = g2q_cuda(points, scales, opacities, seed, valid, PC_RANGE,
                   cell_size=cell_size, support_sigma=sigma, max_scale=0.1)
    assert neighbour_cells(sigma, 0.1, cell_size) == 1
    print(f'  undersized max_scale: query0 = {bad[0, 0, 0].item():.6f}, '
          f'reference = {reference[0, 0, 0].item():.6f}')
    assert not torch.allclose(bad, reference, atol=1e-3), (
        'undersized max_scale did not drop the neighbour; the risk model is wrong')


@requires_cuda
def test_cuda_rejects_non_float32():
    """AMP/FP16/BF16 are unsupported; the operator only accepts float32."""
    points, scales, opacities, seed_t, valid = make_scene(
        count=8, classes=3, seed=31)
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        with pytest.raises((TypeError, RuntimeError)):
            g2q_cuda(points.to(dtype), scales.to(dtype), opacities.to(dtype),
                     seed_t.to(dtype), valid, PC_RANGE, max_scale=0.8)


# --------------------------------------------------------------------------
# Density cap (opt-in, max_neighbours > 0)
# --------------------------------------------------------------------------

def _capped_cuda(points, scales, opacities, seed_t, valid, max_neighbours,
                 cell_size=1.8, support_sigma=3.0, max_scale=0.8):
    from model.ops.semantic_gaussian_localagg import semantic_gaussian_localagg
    return semantic_gaussian_localagg(
        points.contiguous(), scales.contiguous(), opacities.contiguous(),
        seed_t.contiguous(), valid.contiguous(), PC_RANGE, cell_size,
        support_sigma, 1e-4, 1e-6, True, max_scale, max_neighbours)


@requires_cuda
def test_density_cap_is_inert_when_not_binding():
    """A cap above the maximum accepted degree must reproduce the full result."""
    points, scales, opacities, seed_t, valid = make_scene(
        count=128, classes=4, seed=41)
    uncapped = g2q_cuda(points, scales, opacities, seed_t, valid, PC_RANGE,
                        max_scale=0.8)
    capped = _capped_cuda(points, scales, opacities, seed_t, valid,
                          max_neighbours=100000)
    assert_matches(capped, uncapped, 'cap=100000 (non-binding)')


@requires_cuda
@pytest.mark.parametrize('cap', [1, 4, 16, 64])
def test_density_cap_forward_and_backward_match_capped_reference(cap):
    """Capped forward and gradient must match the capped shell reference.

    The reference is single-scene (``[G, 3]``) while the CUDA op is batched
    (``[B, G, 3]``), so the batch dimension is squeezed for the reference.
    """
    batched = make_scene(count=96, classes=4, seed=43)
    points, scales, opacities, seed_t, valid = batched
    flat_points, flat_scales, flat_opacities, flat_seed, flat_valid = (
        tensor[0] for tensor in batched)

    reference = g2q_capped_reference(
        flat_points, flat_scales, flat_opacities, flat_seed, flat_valid,
        PC_RANGE, 1.8, 3.0, 0.8, cap)
    actual = _capped_cuda(points, scales, opacities, seed_t, valid, cap)
    assert_matches(actual, reference.unsqueeze(0), f'capped forward cap={cap}')

    upstream = torch.randn_like(seed_t)
    ref_scales = flat_scales.clone().requires_grad_()
    ref_opacities = flat_opacities.clone().requires_grad_()
    ref_seed = flat_seed.clone().requires_grad_()
    g2q_capped_reference(flat_points, ref_scales, ref_opacities, ref_seed,
                         flat_valid, PC_RANGE, 1.8, 3.0, 0.8, cap) \
        .backward(upstream[0])

    cuda_scales = scales.clone().requires_grad_()
    cuda_opacities = opacities.clone().requires_grad_()
    cuda_seed = seed_t.clone().requires_grad_()
    _capped_cuda(points, cuda_scales, cuda_opacities, cuda_seed, valid, cap) \
        .backward(upstream)

    for name, got, want in (
            ('grad_scales', cuda_scales.grad[0], ref_scales.grad),
            ('grad_opacities', cuda_opacities.grad[0], ref_opacities.grad),
            ('grad_seeds', cuda_seed.grad[0], ref_seed.grad)):
        absolute, _ = report(got, want, f'cap={cap} {name}')
        assert absolute < 1e-4, f'cap={cap} {name} abs error {absolute:.3e}'


@requires_cuda
def test_density_cap_actually_truncates_on_clustered_points():
    """On clustered data a small cap must change the result and stay exact."""
    generator = torch.Generator(device='cuda').manual_seed(47)
    count, classes = 256, 3
    centres = torch.randn(4, 3, generator=generator, device='cuda') * 0.5
    assignment = torch.randint(4, (count,), generator=generator, device='cuda')
    flat_points = centres[assignment] + torch.randn(
        count, 3, generator=generator, device='cuda') * 0.05
    flat_scales = torch.full((count, 3), 0.2, device='cuda')
    flat_opacities = torch.rand(count, generator=generator, device='cuda')
    flat_seed = torch.randn(count, classes, generator=generator, device='cuda')
    flat_valid = torch.ones(count, dtype=torch.bool, device='cuda')

    points = flat_points.unsqueeze(0)
    scales = flat_scales.unsqueeze(0)
    opacities = flat_opacities.unsqueeze(0)
    seed_t = flat_seed.unsqueeze(0)
    valid = flat_valid.unsqueeze(0)

    uncapped = _capped_cuda(points, scales, opacities, seed_t, valid, 0)
    capped = _capped_cuda(points, scales, opacities, seed_t, valid, 8)
    reference = g2q_capped_reference(
        flat_points, flat_scales, flat_opacities, flat_seed, flat_valid,
        PC_RANGE, 1.8, 3.0, 0.2, 8)

    assert_matches(capped, reference.unsqueeze(0), 'clustered cap=8')
    difference = (capped - uncapped).abs().max().item()
    print(f'  clustered cap=8 vs uncapped: max abs difference = {difference:.4f}')
    assert difference > 1e-3, (
        'the cap did not truncate anything on a clustered scene, so the '
        'latency experiment would be vacuous')


@requires_cuda
def test_density_cap_rejects_negative():
    points, scales, opacities, seed_t, valid = make_scene(
        count=8, classes=3, seed=53)
    with pytest.raises(ValueError):
        _capped_cuda(points, scales, opacities, seed_t, valid, max_neighbours=-1)


def test_production_head_max_scale_bounds_learned_scales():
    """The head's static ``max_scale`` really upper-bounds its own outputs.

    This is what makes the cell scan exact in production: ``max_scale`` comes
    from the configured ``scale_range`` upper bound, not from the data, so it
    never under-reports and never forces a device synchronization.
    """
    from model.head.point_anchored_semantic_gaussian import (
        PointAnchoredSemanticGaussianCUDA)

    torch.manual_seed(0)
    head = PointAnchoredSemanticGaussianCUDA(
        embed_dims=16, num_children=4, num_classes=5, pc_range=PC_RANGE,
        scale_range=((0.1, 0.1, 0.1), (0.6, 0.6, 0.6)), cell_size=1.8)
    token = torch.randn(2, 6, 16)
    points = torch.rand(2, 6, 4, 3) * 4.0 - 2.0
    query_valid = torch.ones(2, 6, dtype=torch.bool)
    attributes = head._predict_attributes(token, points, query_valid)
    assert attributes['scales'].max().item() <= head.max_scale + 1e-6
    assert head.max_scale == pytest.approx(0.6)
