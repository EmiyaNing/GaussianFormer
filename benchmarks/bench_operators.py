"""P0.2 operator cost matrix for the sparse-render plan.

Plan reference: ``plans/sparse_render_plans/P0_correctness_and_cost_audit.md``
task T0.2.

Usage examples::

    python3.10 benchmarks/bench_operators.py --op g2q --dist uniform --gaussians 76800
    python3.10 benchmarks/bench_operators.py --op g2v --dist clustered --gaussians 76800
    python3.10 benchmarks/bench_operators.py --op rasterizer --gaussians 76800
    python3.10 benchmarks/bench_operators.py --op pairs --dist ground_band --gaussians 76800

Every measurement reports p50/p95 latency, peak allocated and reserved memory,
and the candidate-pair / local-degree statistics that explain the latency.

Timing notes
------------
* Latency uses CUDA events with a warm-up, reported as p50 and p95 rather than
  a mean, because the clustered distributions have a heavy tail.
* Backward latency is reported as ``forward+backward`` minus ``forward``: the
  two loops are timed identically and the difference is clearly labelled as
  derived, never as an independently measured quantity.
* Cross-hardware comparisons are not produced here.  A single RTX 4090 is used
  and the report marks every number as locally measured.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.data.point_distributions import (  # noqa: E402
    GRID_SHAPE, GRID_SIZE, NUM_CLASSES_G2Q, NUM_CLASSES_G2V, PC_RANGE,
    make_cloud, make_voxel_centres)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def percentile(values, fraction):
    if not values:
        return float('nan')
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bench(fn, warmup, iters):
    """Time ``fn`` with CUDA events, returning per-iteration milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def measure(fn, warmup, iters):
    """Return latency percentiles plus peak allocated/reserved memory (MiB)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = bench(fn, warmup, iters)
    return {
        'p50_ms': percentile(times, 0.50),
        'p95_ms': percentile(times, 0.95),
        'min_ms': min(times),
        'max_ms': max(times),
        'peak_alloc_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20,
    }


def summarize(result):
    return {key: round(value, 4) if isinstance(value, float) else value
            for key, value in result.items()}


# ---------------------------------------------------------------------------
# Pair counting (the quantity the plan claims explains the cost)
# ---------------------------------------------------------------------------

def g2q_pair_statistics(points, valid, pc_range, cell_size, support_sigma,
                        max_scale, grid_shape=None):
    """Exact candidate-pair count and local-degree distribution for the G2Q op.

    This reproduces the CUDA kernel's candidate set exactly: a query in cell
    ``c`` scans the ``(2k+1)^3`` cells around ``c`` for
    ``k = ceil(support_sigma * max_scale / cell_size)``, skipping cells outside
    the cell grid.  A 3D box filter (with zero padding) over the per-cell valid
    counts gives every window sum at once, so this is O(cells) instead of
    O(G^2).
    """
    device = points.device
    low = points.new_tensor(pc_range[:3])
    high = points.new_tensor(pc_range[3:])
    cells = ((high - low) / cell_size).ceil().long()
    gx, gy, gz = (int(v) for v in cells)
    coord = ((points - low) / cell_size).floor().long()
    coord = torch.maximum(torch.minimum(coord, cells - 1),
                          torch.zeros_like(coord))
    flat = (coord[:, 0] * gy + coord[:, 1]) * gz + coord[:, 2]

    counts = torch.zeros(gx * gy * gz, device=device, dtype=torch.float32)
    counts.index_add_(0, flat[valid], torch.ones(
        int(valid.sum().item()), device=device))
    window = F.conv3d(
        counts.view(1, 1, gx, gy, gz),
        torch.ones(1, 1, 2 * _neighbour_cells(support_sigma, max_scale, cell_size) + 1,
                   2 * _neighbour_cells(support_sigma, max_scale, cell_size) + 1,
                   2 * _neighbour_cells(support_sigma, max_scale, cell_size) + 1,
                   device=device),
        padding=_neighbour_cells(support_sigma, max_scale, cell_size))
    degree = window.view(-1)[flat]
    degree = torch.where(valid, degree, torch.zeros_like(degree))
    return {
        'candidate_pairs': int(degree.sum().item()),
        'degree_mean': degree.mean().item(),
        'degree_p50': percentile(degree.tolist(), 0.50),
        'degree_p95': percentile(degree.tolist(), 0.95),
        'degree_p99': percentile(degree.tolist(), 0.99),
        'degree_max': degree.max().item(),
        'cell_grid': [gx, gy, gz],
    }


def _neighbour_cells(support_sigma, max_scale, cell_size):
    import math
    return int(math.ceil(support_sigma * max_scale / cell_size))


def g2v_pair_statistics(points, scales, pc_range, grid_size, grid_shape,
                        scale_multiplier):
    """``L = sum_g |support_cube(g) intersect voxel_grid|`` for dense G2V.

    This is the quantity the parent analysis names as the true G2V bottleneck.
    Each Gaussian covers an integer voxel box of radius
    ``ceil(scale_multiplier * max_scale / grid_size)``; the count is the volume
    of that box clipped to the grid.
    """
    device = points.device
    low = points.new_tensor(pc_range[:3])
    shape = torch.tensor(grid_shape, device=device)
    centre = ((points - low) / grid_size).floor().long()
    import math
    radii = torch.ceil(scales.max(dim=-1).values * scale_multiplier / grid_size) \
        .long().clamp_min(0)
    lo = (centre - radii.unsqueeze(-1)).clamp_min(0)
    hi = (centre + radii.unsqueeze(-1) + 1).clamp_max(shape - 1)
    extent = (hi - lo).clamp_min(0)
    per_gaussian = extent.prod(dim=-1)
    return {
        'L_pairs': int(per_gaussian.sum().item()),
        'L_mean': per_gaussian.float().mean().item(),
        'L_max': int(per_gaussian.max().item()),
        'radius_mean': radii.float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def g2v_block_union_statistics(points, scales, pc_range, grid_size, grid_shape,
                              scale_multiplier, block=4):
    """Analytic bound on what S2GO-style 4x4x4 blocking can save in G2V.

    The repository's dense ``localagg`` is a naive per-voxel scan: each voxel
    walks its own list of Gaussians straight from global memory.  S2GO-style
    blocking partitions the grid into ``block^3`` voxel tiles and loads the
    *union* of the tile's Gaussians once, sharing it across the tile's voxels.

    The best-case memory-traffic ratio is therefore

        naive traffic  proportional to  L      = sum_g |voxels(g)|
        blocked traffic proportional to U      = sum_g |blocks(g)|

    because each ``(gaussian, block)`` pair is loaded once in the blocked
    scheme, whereas the naive scheme loads the Gaussian once per covered voxel.
    ``L / U`` bounds the achievable traffic reduction; it says nothing about
    wall-clock gain, which also depends on cache behaviour, occupancy and the
    per-pair channel loop.  Reporting it keeps the latency claim honest without
    writing the kernel.
    """
    device = points.device
    low = points.new_tensor(pc_range[:3])
    shape = torch.tensor(grid_shape, device=device)
    import math
    centre = ((points - low) / grid_size).floor().long()
    radii = torch.ceil(scales.max(dim=-1).values * scale_multiplier / grid_size) \
        .long().clamp_min(0)

    voxel_lo = (centre - radii.unsqueeze(-1)).clamp_min(0)
    voxel_hi = (centre + radii.unsqueeze(-1) + 1).clamp_max(shape - 1)
    voxels = (voxel_hi - voxel_lo).clamp_min(0).prod(dim=-1)

    block_shape = (shape + block - 1) // block
    block_lo = (voxel_lo // block).clamp_max(block_shape - 1)
    block_hi = ((voxel_hi + block - 1) // block).clamp_max(block_shape)
    blocks = (block_hi - block_lo).clamp_min(0).prod(dim=-1)

    total_voxels = int(voxels.sum().item())
    total_blocks = int(blocks.sum().item())
    return {
        'L_pairs': total_voxels,
        'block_union_pairs': total_blocks,
        'traffic_reduction_bound': round(total_voxels / max(total_blocks, 1), 3),
        'block': int(block),
        'block_grid': [int(v) for v in block_shape],
        'mean_voxels_per_gaussian': round(float(voxels.float().mean()), 2),
        'mean_blocks_per_gaussian': round(float(blocks.float().mean()), 2),
    }


def run_blockunion(args):
    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale,
                       real_points_path=args.real_points,
                       classes=args.classes, device=args.device, seed=args.seed)
    stats = g2v_block_union_statistics(
        cloud.points, cloud.scales, PC_RANGE, GRID_SIZE, GRID_SHAPE,
        args.scale_multiplier, block=args.block)
    stats.update({'op': 'blockunion', 'dist': args.dist,
                  'gaussians': args.gaussians, 'scale': args.scale,
                  'max_scale': round(cloud.max_scale, 6)})
    return stats


def run_g2q(args):
    from model.ops.semantic_gaussian_localagg import semantic_gaussian_localagg

    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale,
                       real_points_path=args.real_points,
                       classes=args.classes, device=args.device,
                       seed=args.seed)
    batch = args.batch
    count = args.gaussians
    points = cloud.points.unsqueeze(0).repeat(batch, 1, 1).contiguous()
    scales = cloud.scales.unsqueeze(0).repeat(batch, 1, 1).contiguous()
    opacities = cloud.opacities.unsqueeze(0).repeat(batch, 1).contiguous()
    seed = cloud.seed.unsqueeze(0).repeat(batch, 1, 1).contiguous()
    valid = torch.ones(batch, count, dtype=torch.bool, device=args.device)
    upstream = torch.randn_like(seed)

    # Inference-shaped forward (no autograd graph).
    def forward():
        with torch.no_grad():
            return semantic_gaussian_localagg(
                points, scales, opacities, seed, valid, PC_RANGE, args.cell,
                args.support, 1e-4, 1e-6, True, cloud.max_scale,
                args.max_neighbours)

    # Training-shaped step: only the attribute tensors need a gradient; the
    # operator deliberately returns none for the point coordinates.
    grad_scales = cloud.scales.unsqueeze(0).repeat(batch, 1, 1).clone().requires_grad_()
    grad_opacities = cloud.opacities.unsqueeze(0).repeat(batch, 1).clone().requires_grad_()
    grad_seed = cloud.seed.unsqueeze(0).repeat(batch, 1, 1).clone().requires_grad_()

    def forward_backward():
        out = semantic_gaussian_localagg(
            points, grad_scales, grad_opacities, grad_seed, valid, PC_RANGE,
            args.cell, args.support, 1e-4, 1e-6, True, cloud.max_scale,
            args.max_neighbours)
        out.backward(upstream)
        for tensor in (grad_scales, grad_opacities, grad_seed):
            tensor.grad = None
        return out

    forward_stats = measure(forward, args.warmup, args.iters)
    step_stats = measure(forward_backward, args.warmup, args.iters)
    pairs = g2q_pair_statistics(cloud.points, valid[0], PC_RANGE, args.cell,
                                args.support, cloud.max_scale, GRID_SHAPE)

    result = {
        'op': 'g2q', 'dist': args.dist, 'gaussians': count, 'batch': batch,
        'classes': args.classes, 'cell_size': args.cell,
        'support_sigma': args.support, 'scale': args.scale,
        'max_scale': round(cloud.max_scale, 6),
        'max_neighbours': args.max_neighbours,
        'forward': summarize(forward_stats),
        'forward_backward': summarize(step_stats),
        'backward_derived': {
            'p50_ms': round(step_stats['p50_ms'] - forward_stats['p50_ms'], 4),
            'p95_ms': round(step_stats['p95_ms'] - forward_stats['p95_ms'], 4),
            'note': 'derived as forward+backward minus forward',
        },
        'pairs': {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in pairs.items()},
    }
    return result


def run_g2v(args):
    from model.head.localagg.local_aggregate import LocalAggregator

    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale,
                       real_points_path=args.real_points,
                       classes=NUM_CLASSES_G2V, device=args.device, seed=args.seed)
    aggregator = LocalAggregator(
        scale_multiplier=args.scale_multiplier, H=GRID_SHAPE[0],
        W=GRID_SHAPE[1], D=GRID_SHAPE[2],
        pc_min=(PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]), grid_size=GRID_SIZE)
    # ``pc_min`` is a registered buffer created on CPU; the module must be
    # moved so the voxel-centre indexing arithmetic stays on one device.
    aggregator = aggregator.to(args.device)
    voxels = make_voxel_centres(GRID_SHAPE, PC_RANGE, GRID_SIZE, args.device)

    means = cloud.points.unsqueeze(0).contiguous()
    # ``GaussianHead`` passes ``origi_opa.reshape(bs, g)``, i.e. [B, G].  The
    # released CUDA backward returns a flat [B*G] opacity gradient, so the
    # [B, G, 1] form is rejected by autograd's shape check.
    opacity = cloud.opacities.unsqueeze(0).contiguous()
    semantics = cloud.seed.unsqueeze(0).contiguous()
    scales = cloud.scales.unsqueeze(0).contiguous()
    covariance = torch.zeros(cloud.points.shape[0], 3, 3, device=args.device)
    covariance[:, 0, 0] = cloud.scales[:, 0] ** 2
    covariance[:, 1, 1] = cloud.scales[:, 1] ** 2
    covariance[:, 2, 2] = cloud.scales[:, 2] ** 2
    covariance = covariance.unsqueeze(0).contiguous()
    # The dense G2V output is per *voxel*, not per Gaussian: [V, C].
    upstream = torch.randn(voxels.shape[1], NUM_CLASSES_G2V,
                           device=args.device)

    def forward():
        with torch.no_grad():
            return aggregator(voxels, means, opacity, semantics, scales, covariance)

    grad_means = means.clone().requires_grad_()
    grad_opacity = opacity.clone().requires_grad_()
    grad_semantics = semantics.clone().requires_grad_()
    grad_covariance = covariance.clone().requires_grad_()

    def forward_backward():
        out = aggregator(voxels, grad_means, grad_opacity, grad_semantics,
                         scales, grad_covariance)
        out.backward(upstream)
        for tensor in (grad_means, grad_opacity, grad_semantics, grad_covariance):
            tensor.grad = None
        return out

    forward_stats = measure(forward, args.warmup, args.iters)
    step_stats = measure(forward_backward, args.warmup, args.iters)
    pairs = g2v_pair_statistics(cloud.points, cloud.scales, PC_RANGE,
                                GRID_SIZE, GRID_SHAPE, args.scale_multiplier)

    return {
        'op': 'g2v', 'dist': args.dist, 'gaussians': args.gaussians, 'batch': 1,
        'classes': NUM_CLASSES_G2V, 'grid_shape': list(GRID_SHAPE),
        'voxels': int(torch.tensor(GRID_SHAPE).prod().item()),
        'scale': args.scale, 'scale_multiplier': args.scale_multiplier,
        'forward': summarize(forward_stats),
        'forward_backward': summarize(step_stats),
        'backward_derived': {
            'p50_ms': round(step_stats['p50_ms'] - forward_stats['p50_ms'], 4),
            'p95_ms': round(step_stats['p95_ms'] - forward_stats['p95_ms'], 4),
            'note': 'derived as forward+backward minus forward',
        },
        'pairs': {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in pairs.items()},
    }


def run_rasterizer(args):
    from model.head.opus_rasterizer import OPUSRasterizer

    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale,
                       real_points_path=args.real_points,
                       classes=NUM_CLASSES_G2Q, device=args.device, seed=args.seed)
    rasterizer = OPUSRasterizer(PC_RANGE, GRID_SIZE, GRID_SHAPE, empty_label=17,
                                score_threshold=0.5)
    points = cloud.points.unsqueeze(0).contiguous()
    logits = cloud.seed.unsqueeze(0).contiguous()
    results = {}
    for label, padding in (('padding_off', False), ('padding_on', True)):
        stats = measure(lambda p=padding: rasterizer(points, logits, padding=p),
                        args.warmup, args.iters)
        results[label] = summarize(stats)
    results.update({
        'op': 'rasterizer', 'dist': args.dist, 'gaussians': args.gaussians,
        'classes': NUM_CLASSES_G2Q, 'grid_shape': list(GRID_SHAPE),
        'voxels': int(torch.tensor(GRID_SHAPE).prod().item()),
        'note': 'padding_on enables the 3D morphological closing',
    })
    return results


def run_knn_gather(args):
    """KNN-8 gather baseline: a different neighbour rule, same kernel algebra."""
    from mmcv.ops import knn

    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale,
                       real_points_path=args.real_points,
                       classes=args.classes, device=args.device, seed=args.seed)
    points = cloud.points.contiguous()
    scales = cloud.scales.contiguous().clamp_min(1e-4)
    opacities = cloud.opacities.contiguous()
    seed = cloud.seed.contiguous()
    upstream = torch.randn_like(seed)
    neighbours = args.neighbours

    def forward():
        index = knn(neighbours, points[None], points[None])[0].transpose(0, 1)
        neighbour_points = points[index]
        neighbour_scales = scales[index]
        delta = (points[:, None] - neighbour_points) / neighbour_scales
        kernel = torch.exp(-0.5 * delta.square().sum(dim=-1))
        weight = opacities[index] * kernel
        numerator = (weight[..., None] * seed[index]).sum(dim=1)
        return numerator / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

    grad_scales = cloud.scales.clone().requires_grad_()
    grad_opacities = cloud.opacities.clone().requires_grad_()
    grad_seed = cloud.seed.clone().requires_grad_()

    def forward_backward():
        index = knn(neighbours, points[None], points[None])[0].transpose(0, 1)
        neighbour_points = points[index]
        neighbour_scales = grad_scales[index]
        delta = (points[:, None] - neighbour_points) / neighbour_scales
        kernel = torch.exp(-0.5 * delta.square().sum(dim=-1))
        weight = grad_opacities[index] * kernel
        numerator = (weight[..., None] * grad_seed[index]).sum(dim=1)
        out = numerator / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        out.backward(upstream)
        for tensor in (grad_scales, grad_opacities, grad_seed):
            tensor.grad = None
        return out

    forward_stats = measure(forward, args.warmup, args.iters)
    step_stats = measure(forward_backward, args.warmup, args.iters)
    return {
        'op': 'knn_gather', 'dist': args.dist, 'gaussians': args.gaussians,
        'classes': args.classes, 'neighbours': neighbours,
        'forward': summarize(forward_stats),
        'forward_backward': summarize(step_stats),
        'backward_derived': {
            'p50_ms': round(step_stats['p50_ms'] - forward_stats['p50_ms'], 4),
            'note': 'derived as forward+backward minus forward',
        },
        'note': 'fixed-k neighbour rule, so this is a cost-structure baseline '
                'and not an equivalent operator',
    }


def run_pairs(args):
    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale,
                       real_points_path=args.real_points,
                       classes=args.classes, device=args.device, seed=args.seed)
    valid = torch.ones(args.gaussians, dtype=torch.bool, device=args.device)
    g2q = g2q_pair_statistics(cloud.points, valid, PC_RANGE, args.cell,
                              args.support, cloud.max_scale, GRID_SHAPE)
    g2v = g2v_pair_statistics(cloud.points, cloud.scales, PC_RANGE, GRID_SIZE,
                              GRID_SHAPE, args.scale_multiplier)
    return {
        'op': 'pairs', 'dist': args.dist, 'gaussians': args.gaussians,
        'cell_size': args.cell, 'support_sigma': args.support,
        'scale': args.scale, 'max_scale': round(cloud.max_scale, 6),
        'g2q': {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in g2q.items()},
        'g2v': {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in g2v.items()},
    }


def g2q_accepted_degree(points, valid, scales, opacities, pc_range, support_sigma,
                        scale_eps=1e-4, sample=512, seed=0):
    """Sampled count of neighbours that actually pass the support-box test.

    ``g2q_pair_statistics`` counts *candidates* scanned by the cell window,
    which is set by ``max_scale`` alone.  The kernel's real inner-loop work is
    driven by neighbours that also pass the per-axis support check, so two
    configurations with identical candidate counts can differ several-fold in
    latency.  A uniform sample of queries against all Gaussians measures that
    directly in O(sample * G) instead of O(G^2).
    """
    device = points.device
    count = points.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = min(sample, count)
    queries = torch.randperm(count, generator=generator, device=device)[:sample]
    query_points = points[queries]
    neighbour_scales = scales.clamp_min(scale_eps)
    delta = query_points[:, None, :] - points[None, :, :]
    relative = (delta / neighbour_scales[None, :, :]).abs()
    inside = (relative <= support_sigma).all(dim=-1)
    inside &= valid[None, :] & valid[queries, None]
    degree = inside.sum(dim=-1).float()
    return {
        'accepted_degree_mean': degree.mean().item(),
        'accepted_degree_p50': percentile(degree.tolist(), 0.50),
        'accepted_degree_p95': percentile(degree.tolist(), 0.95),
        'accepted_degree_max': degree.max().item(),
        'accepted_sample': sample,
    }


def run_degree(args):
    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale,
                       real_points_path=args.real_points,
                       classes=args.classes, device=args.device, seed=args.seed)
    valid = torch.ones(args.gaussians, dtype=torch.bool, device=args.device)
    candidates = g2q_pair_statistics(cloud.points, valid, PC_RANGE, args.cell,
                                     args.support, cloud.max_scale, GRID_SHAPE)
    accepted = g2q_accepted_degree(cloud.points, valid, cloud.scales,
                                   cloud.opacities, PC_RANGE, args.support)
    ratio = (accepted['accepted_degree_mean'] / max(candidates['degree_mean'], 1e-9))
    return {
        'op': 'degree', 'dist': args.dist, 'gaussians': args.gaussians,
        'scale': args.scale, 'support_sigma': args.support,
        'cell_size': args.cell, 'max_scale': round(cloud.max_scale, 6),
        'candidate_degree_mean': round(candidates['degree_mean'], 3),
        'accepted_degree_mean': round(accepted['accepted_degree_mean'], 3),
        'accepted_to_candidate': round(ratio, 4),
        'accepted_degree_p95': round(accepted['accepted_degree_p95'], 1),
        'accepted_degree_max': round(accepted['accepted_degree_max'], 1),
        'candidate_pairs': candidates['candidate_pairs'],
    }


OPERATIONS = {
    'g2q': run_g2q,
    'g2v': run_g2v,
    'rasterizer': run_rasterizer,
    'knn_gather': run_knn_gather,
    'pairs': run_pairs,
    'degree': run_degree,
    'blockunion': run_blockunion,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--op', required=True, choices=sorted(OPERATIONS))
    parser.add_argument('--dist', default='uniform')
    parser.add_argument('--real-points',
                        default='benchmarks/results/real_points.pt',
                        help='point cloud used when --dist real')
    parser.add_argument('--gaussians', type=int, default=76800)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--classes', type=int, default=NUM_CLASSES_G2Q)
    parser.add_argument('--scale', type=float, default=0.35)
    parser.add_argument('--scale-multiplier', type=float, default=3.0)
    parser.add_argument('--support', type=float, default=3.0)
    parser.add_argument('--cell', type=float, default=1.8)
    parser.add_argument('--neighbours', type=int, default=8)
    parser.add_argument('--block', type=int, default=4, help='S2GO-style voxel block edge')
    parser.add_argument('--max-neighbours', type=int, default=0,
                        help='G2Q density cap; 0 disables it')
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--json', default=None)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() and args.device == 'cuda':
        raise SystemExit('CUDA is not available')
    result = OPERATIONS[args.op](args)
    result.setdefault('device', torch.cuda.get_device_name(0))
    payload = json.dumps(result, indent=2, sort_keys=False)
    if not args.quiet:
        print(payload)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    main()
