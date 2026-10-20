"""Correctness tests for the S2GO-style block dense G2V forward.

The block kernel must reproduce the repository's naive dense ``localagg``
exactly: both aggregate every Gaussian whose voxel AABB covers a voxel, so the
only difference may be floating-point summation order.  Verifying this is what
makes the later speed comparison meaningful -- a fast-but-wrong kernel would
"win" for the wrong reason.
"""
from __future__ import annotations

import pytest
import torch

from model.ops.block_g2v import CUDA_BLOCK_G2V_AVAILABLE, block_g2v_forward

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not (CUDA_AVAILABLE and CUDA_BLOCK_G2V_AVAILABLE),
    reason='requires CUDA and the block_g2v extension')

# The grid must exactly tile pc_range: 16x16x8 voxels of size 1.0
# means x,y in [-8, 8) and z in [-8, 0).
PC_RANGE = (-8.0, -8.0, -8.0, 8.0, 8.0, 0.0)
GRID = (16, 16, 8)
GRID_SIZE = 1.0
CHANNELS = 18


def build_inputs(gaussians=64, seed=0, scale=2.0, device='cuda'):
    generator = torch.Generator(device=device).manual_seed(seed)
    low = torch.tensor(PC_RANGE[:3], device=device)
    high = torch.tensor(PC_RANGE[3:], device=device)
    # Gaussian centres, kept inside the grid.
    means = low + (high - low) * torch.rand(
        gaussians, 3, generator=generator, device=device)
    scales = torch.full((gaussians, 3), scale, device=device)
    opacities = torch.rand(gaussians, generator=generator, device=device) * 0.8 + 0.1
    semantics = torch.randn(gaussians, CHANNELS, generator=generator, device=device)
    covariance = torch.zeros(gaussians, 3, 3, device=device)
    covariance[:, 0, 0] = 1.0 / scale ** 2
    covariance[:, 1, 1] = 1.0 / scale ** 2
    covariance[:, 2, 2] = 1.0 / scale ** 2

    gx, gy, gz = GRID
    xs = (torch.arange(gx, device=device, dtype=torch.float32) + 0.5) * GRID_SIZE + low[0]
    ys = (torch.arange(gy, device=device, dtype=torch.float32) + 0.5) * GRID_SIZE + low[1]
    zs = (torch.arange(gz, device=device, dtype=torch.float32) + 0.5) * GRID_SIZE + low[2]
    pts = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1).reshape(-1, 3)
    return means, scales, opacities, semantics, covariance, pts


def to_localagg_arguments(means, scales, opacities, semantics, covariance, pts,
                          scale_multiplier=3.0):
    low = torch.tensor(PC_RANGE[:3], device=pts.device)
    points_int = ((pts - low) / GRID_SIZE).to(torch.int32)
    means_int = ((means.detach() - low) / GRID_SIZE).to(torch.int32)
    radii = torch.ceil(
        scales.max(dim=-1)[0] * scale_multiplier / GRID_SIZE).to(torch.int32).clamp_min(1)
    cov6 = covariance.flatten(1)[:, [0, 4, 8, 1, 5, 2]].contiguous()
    return points_int, means_int, radii, cov6


@requires_cuda
def test_block_g2v_matches_localagg():
    from model.head.localagg.local_aggregate import LocalAggregator

    means, scales, opacities, semantics, covariance, pts = build_inputs()
    points_int, means_int, radii, cov6 = to_localagg_arguments(
        means, scales, opacities, semantics, covariance, pts)

    aggregator = LocalAggregator(
        scale_multiplier=3.0, H=GRID[0], W=GRID[1], D=GRID[2],
        pc_min=(PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]),
        grid_size=GRID_SIZE).cuda()
    with torch.no_grad():
        reference = aggregator(
            pts.unsqueeze(0), means.unsqueeze(0), opacities.unsqueeze(0),
            semantics.unsqueeze(0), scales.unsqueeze(0), covariance.unsqueeze(0))
        actual = block_g2v_forward(pts, points_int, means, means_int, opacities,
                                   semantics, radii, cov6, GRID)

    assert actual.shape == reference.shape
    difference = (actual - reference).abs().max().item()
    print(f'  block vs localagg: max abs difference = {difference:.6e}')
    assert difference < 1e-4, f'max abs difference {difference:.3e} too large'


@requires_cuda
@pytest.mark.parametrize('scale', [1.0, 2.0, 0.5])
def test_block_g2v_matches_localagg_across_scales(scale):
    """Different scales change the tile occupancy and the union list sizes."""
    means, scales, opacities, semantics, covariance, pts = build_inputs(
        gaussians=96, seed=3, scale=scale)
    points_int, means_int, radii, cov6 = to_localagg_arguments(
        means, scales, opacities, semantics, covariance, pts)
    from model.head.localagg.local_aggregate import LocalAggregator

    aggregator = LocalAggregator(
        scale_multiplier=3.0, H=GRID[0], W=GRID[1], D=GRID[2],
        pc_min=(PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]),
        grid_size=GRID_SIZE).cuda()
    with torch.no_grad():
        reference = aggregator(
            pts.unsqueeze(0), means.unsqueeze(0), opacities.unsqueeze(0),
            semantics.unsqueeze(0), scales.unsqueeze(0), covariance.unsqueeze(0))
        actual = block_g2v_forward(pts, points_int, means, means_int, opacities,
                                   semantics, radii, cov6, GRID)
    difference = (actual - reference).abs().max().item()
    print(f'  scale={scale}: max abs difference = {difference:.6e}')
    assert difference < 1e-4


@requires_cuda
def test_block_g2v_handles_zero_gaussians():
    """No Gaussians must produce an all-zero field, not crash on an empty scan.

    Note a Gaussian with radius 0 still covers its own voxel, so the degenerate
    case has to be P == 0 rather than a zero radius.
    """
    means, scales, opacities, semantics, covariance, pts = build_inputs(gaussians=4, seed=7)
    points_int, means_int, radii, cov6 = to_localagg_arguments(
        means, scales, opacities, semantics, covariance, pts)
    empty = torch.zeros(0, device='cuda')
    with torch.no_grad():
        out = block_g2v_forward(pts, points_int, torch.zeros(0, 3, device='cuda'),
                                torch.zeros(0, 3, dtype=torch.int32, device='cuda'),
                                empty, torch.zeros(0, CHANNELS, device='cuda'),
                                torch.zeros(0, dtype=torch.int32, device='cuda'),
                                torch.zeros(0, 6, device='cuda'), GRID)
    assert out.shape == (pts.shape[0], CHANNELS)
    assert torch.all(out == 0)


@requires_cuda
def test_block_g2v_rejects_incomplete_voxel_grid():
    means, scales, opacities, semantics, covariance, pts = build_inputs(gaussians=4)
    points_int, means_int, radii, cov6 = to_localagg_arguments(
        means, scales, opacities, semantics, covariance, pts)
    with pytest.raises(ValueError):
        block_g2v_forward(pts[:-1], points_int[:-1], means, means_int, opacities,
                          semantics, radii, cov6, GRID)


@requires_cuda
def test_block_g2v_backward_matches_localagg():
    """Gradients must match the naive dense operator that defines the function.

    Both operators compute the same sum-over-covering-Gaussians, so their
    gradients must agree; a fast-but-wrong backward would silently corrupt any
    training comparison built on it.
    """
    from model.head.localagg.local_aggregate import LocalAggregator

    means, scales, opacities, semantics, covariance, pts = build_inputs(
        gaussians=64, seed=11, scale=2.0)
    points_int, means_int, radii, cov6 = to_localagg_arguments(
        means, scales, opacities, semantics, covariance, pts)

    generator = torch.Generator(device='cuda').manual_seed(5)
    upstream = torch.randn(pts.shape[0], CHANNELS, generator=generator, device='cuda')

    # --- naive dense reference ---
    ref_means = means.clone().requires_grad_()
    ref_opacity = opacities.clone().requires_grad_()
    ref_semantics = semantics.clone().requires_grad_()
    ref_cov = covariance.clone().requires_grad_()
    aggregator = LocalAggregator(
        scale_multiplier=3.0, H=GRID[0], W=GRID[1], D=GRID[2],
        pc_min=(PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]),
        grid_size=GRID_SIZE).cuda()
    aggregator(pts.unsqueeze(0), ref_means.unsqueeze(0), ref_opacity.unsqueeze(0),
               ref_semantics.unsqueeze(0), scales.unsqueeze(0),
               ref_cov.unsqueeze(0)).backward(upstream)

    # --- block operator ---
    blk_means = means.clone().requires_grad_()
    blk_opacity = opacities.clone().requires_grad_()
    blk_semantics = semantics.clone().requires_grad_()
    blk_cov = covariance.clone().requires_grad_()
    block_g2v_forward(pts, points_int, blk_means, means_int, blk_opacity,
                      blk_semantics, radii, blk_cov.flatten(1)[:, [0, 4, 8, 1, 5, 2]].contiguous(),
                      GRID).backward(upstream)

    for name, got, want in (
            ('grad_means', blk_means.grad, ref_means.grad),
            ('grad_opacity', blk_opacity.grad, ref_opacity.grad),
            ('grad_semantics', blk_semantics.grad, ref_semantics.grad),
            ('grad_cov',
             blk_cov.grad.flatten(1)[:, [0, 4, 8, 1, 5, 2]],
             ref_cov.grad.flatten(1)[:, [0, 4, 8, 1, 5, 2]])):
        difference = (got - want).abs().max().item()
        reference = want.abs().max().item()
        # Gradients are summed with atomicAdd in a different order in each
        # kernel, so float32 rounding sets the tolerance; compare relative to
        # the gradient magnitude rather than with a bare absolute threshold.
        print(f'  {name}: max abs diff = {difference:.3e} (|ref|max {reference:.3e})')
        assert difference < 1e-4 * max(reference, 1.0) + 1e-5, \
            f'{name} differs by {difference:.3e} (|ref|max {reference:.3e})'


@requires_cuda
def test_block_g2v_backward_matches_finite_difference():
    """Independent check: central differences of the CUDA forward.

    The ``localagg`` comparison pins the backward against the reference
    implementation; this check pins it against a different method entirely, so a
    shared mistake in both CUDA backwards cannot hide.  Probes are restricted to
    the largest-magnitude gradient entries because float32 central differences
    have limited signal-to-noise on near-zero entries.
    """
    means, scales, opacities, semantics, covariance, pts = build_inputs(
        gaussians=24, seed=17, scale=2.0)
    points_int, means_int, radii, cov6 = to_localagg_arguments(
        means, scales, opacities, semantics, covariance, pts)

    upstream = torch.randn(pts.shape[0], CHANNELS, device='cuda')
    blk_means = means.clone().requires_grad_()
    blk_opacity = opacities.clone().requires_grad_()
    blk_semantics = semantics.clone().requires_grad_()
    blk_cov = cov6.clone().requires_grad_()
    loss = (block_g2v_forward(pts, points_int, blk_means, means_int, blk_opacity,
                              blk_semantics, radii, blk_cov, GRID) * upstream).sum()
    loss.backward()

    def objective(m, o, s, c):
        # Mean, not sum: a sum over 2048 x 18 float32 outputs has spacing that
        # swamps the change produced by perturbing one parameter.
        return (block_g2v_forward(pts, points_int, m, means_int, o, s, radii, c, GRID)
                * upstream).mean()

    cases = (
        ('means', means, blk_means.grad, 1e-3),
        ('opacity', opacities, blk_opacity.grad, 5e-3),
        ('semantics', semantics, blk_semantics.grad, 1e-3),
        ('cov', cov6, blk_cov.grad, 1e-3),
    )
    for label, base, gradient, step in cases:
        flat = gradient.abs().flatten()
        probes = flat.topk(min(3, flat.numel())).indices.tolist()
        for flat_index in probes:
            index = tuple(int(v) for v in
                          torch.unravel_index(torch.tensor(flat_index), gradient.shape))
            plus, minus = base.clone(), base.clone()
            plus[index] += step
            minus[index] -= step
            if label == 'means':
                numeric = (objective(plus, opacities, semantics, cov6)
                           - objective(minus, opacities, semantics, cov6)) / (2 * step)
            elif label == 'opacity':
                numeric = (objective(means, plus, semantics, cov6)
                           - objective(means, minus, semantics, cov6)) / (2 * step)
            elif label == 'semantics':
                numeric = (objective(means, opacities, plus, cov6)
                           - objective(means, opacities, minus, cov6)) / (2 * step)
            else:
                numeric = (objective(means, opacities, semantics, plus)
                           - objective(means, opacities, semantics, minus)) / (2 * step)
            # The analytic gradient is of the SUM; rescale to the mean objective.
            analytic = gradient[index].item() / (pts.shape[0] * CHANNELS)
            reference = abs(analytic)
            relative = abs(numeric.item() - analytic) / max(reference, 1e-8)
            print(f'  {label}{index}: numeric={numeric.item():+.6e} '
                  f'analytic={analytic:+.6e} rel={relative:.3e}')
            assert relative < 5e-2, f'{label}{index} relative error {relative:.3e}'
