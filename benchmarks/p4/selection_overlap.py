"""Do A2-k8 and A5-cap8 select the same neighbours?

P1-6 section 4.2.1 concluded that, at a matched neighbour count, A5's thin-class
deficit must come from either its **weight profile** (uniform 1/k vs a learned
distance-decaying kernel) or its **selection rule** (k nearest by Euclidean
distance vs the k nearest *cell shells* within the support box).  Splitting those
two apart properly needs a new operator variant, but the confound can be bounded
cheaply: if the two selections mostly coincide, the matched-k comparison is
essentially a weighting test; if they barely overlap, selection dominates and the
attribution stays open.

This replicates the CUDA cap's selection for a sample of queries -- shells in
increasing Chebyshev radius, lexicographic offsets within a shell, out-of-grid
cells skipped, ascending index within a cell, and only neighbours passing
validity plus the per-axis box test count toward the cap -- and compares it with
the k Euclidean-nearest set.  It deliberately does NOT build the reference's
``[G, G]`` selection matrix, which would need ~1.2 GB at G = 34,560.

    python3.10 benchmarks/p4/selection_overlap.py
    python3.10 benchmarks/p4/selection_overlap.py --queries 400 --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.bench_operators import PC_RANGE, make_cloud  # noqa: E402
from benchmarks.common_paths import display  # noqa: E402
from benchmarks.reference.g2q_bruteforce import (  # noqa: E402
    cell_coordinates, neighbour_cells)


def shell_selection(coord, members, query, points, scales, radius, cell_size,
                    support_sigma, pc_range, grid_shape, k):
    """Replicate the CUDA cap's selection order for one query."""
    base = tuple(int(v) for v in coord[query])
    accepted = []
    for shell in range(radius + 1):
        if len(accepted) >= k:
            break
        offsets = [(dx, dy, dz)
                   for dx in range(-shell, shell + 1)
                   for dy in range(-shell, shell + 1)
                   for dz in range(-shell, shell + 1)
                   if max(abs(dx), abs(dy), abs(dz)) == shell]
        for dx, dy, dz in offsets:
            cell = (base[0] + dx, base[1] + dy, base[2] + dz)
            if not (0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]
                    and 0 <= cell[2] < grid_shape[2]):
                continue
            for other in members.get(cell, ()):
                if other == query:
                    continue                      # include_self handled below
                delta = (points[query] - points[other]).abs()
                if bool((delta <= support_sigma * scales[other]).all()):
                    accepted.append(other)
                    if len(accepted) >= k:
                        break
            if len(accepted) >= k:
                break
    return accepted


def scale_sweep(points, queries, k, scales):
    """ESS/k of the Gaussian weight profile over the k-NN set, per scale.

    The A2-gauss control must manipulate the weight profile by a *meaningful*
    amount, otherwise a null result would only show that a nearly-uniform
    Gaussian equals a uniform mean.  The criterion used to pick its scale is
    **relative concentration**: A5's kernel has ESS/accepted = 0.278 over its own
    accepted neighbours, so the control should have ESS/k near 0.278 too.

    Every ESS is asserted to be <= k: normalised weights cannot be more
    concentrated than uniform, so a larger value means the computation
    underflowed (which it did, before the log-sum-exp stabilisation here).
    """
    rows = []
    for scale in scales:
        ess_values, shares = [], []
        for query in queries.tolist():
            distance = (points - points[query]).norm(dim=-1)
            distance[query] = float('inf')
            knn = torch.topk(distance, k, largest=False).indices
            delta = points[knn] - points[query]
            log_kernel = -0.5 * (delta / scale).square().sum(dim=-1)
            log_kernel = log_kernel - log_kernel.max()
            kernel = torch.exp(log_kernel)
            weights = kernel / kernel.sum()
            value = float(1.0 / weights.square().sum())
            assert value <= k + 1e-6, (
                f'ESS {value} > k for scale {scale}: the kernel underflowed and '
                'the normalisation is broken')
            ess_values.append(value)
            shares.append(float(weights[0]))
        rows.append({'scale': scale,
                     'ess': round(sum(ess_values) / len(ess_values), 4),
                     'ess_over_k': round(sum(ess_values) / len(ess_values) / k, 4),
                     'nearest_weight_share': round(sum(shares) / len(shares), 4)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', default='real')
    parser.add_argument('--gaussians', type=int, default=34560)
    parser.add_argument('--scale', type=float, default=0.53)
    parser.add_argument('--cell', type=float, default=1.8)
    parser.add_argument('--support', type=float, default=3.0)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--queries', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--real-points', default='benchmarks/results/real_points.pt')
    parser.add_argument('--scale-sweep', action='store_true',
                        help='report ESS/k of the Gaussian profile versus scale, '
                             'to pick the A2-gauss control scale by relative '
                             'concentration rather than by absolute width')
    parser.add_argument('--json', default='benchmarks/results/p4_selection_overlap.json')
    args = parser.parse_args(argv)

    cloud = make_cloud(args.dist, args.gaussians, scale=args.scale, device='cpu',
                       real_points_path=args.real_points, seed=args.seed)
    points = cloud.points.float()
    if args.scale_sweep:
        generator = torch.Generator().manual_seed(args.seed)
        sampled = torch.randperm(points.shape[0], generator=generator)[:args.queries]
        scales = [0.05, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.35, 0.53]
        rows = scale_sweep(points, sampled, args.k, scales)
        print(f"ESS of the Gaussian weight profile over the {args.k}-NN set "
              f"(uniform = {args.k}); target ESS/k ~ 0.278 to match A5:")
        print(f"{'scale':>7} {'ESS':>8} {'ESS/k':>8} {'nearest share':>14}")
        for row in rows:
            print(f"{row['scale']:7.2f} {row['ess']:8.2f} {row['ess_over_k']:8.3f} "
                  f"{row['nearest_weight_share']:13.1%}")
        best = min(rows, key=lambda r: abs(r['ess_over_k'] - 0.278))
        print(f"\nclosest to A5's relative concentration: scale "
              f"{best['scale']} (ESS/k {best['ess_over_k']:.3f})")
        target = ROOT / args.json
        target.write_text(json.dumps({'scale_sweep': rows,
                                      'target_ess_over_k': 0.278,
                                      'chosen_scale': best['scale']}, indent=1),
                          encoding='utf-8')
        print(f'wrote {display(target)}')
        return
    scales = cloud.scales.clamp_min(1e-4).float()
    coord, grid_shape = cell_coordinates(points, PC_RANGE, args.cell)
    grid_shape = tuple(int(v) for v in grid_shape)
    members = {}
    for index in range(points.shape[0]):
        key = tuple(int(v) for v in coord[index])
        members.setdefault(key, []).append(index)
    radius = neighbour_cells(args.support, float(cloud.max_scale), args.cell)

    generator = torch.Generator().manual_seed(args.seed)
    queries = torch.randperm(points.shape[0], generator=generator)[:args.queries]

    overlaps, knn_distances, shell_distances = [], [], []
    # Pre-registration for the A2-gauss control: how strong is the weighting
    # manipulation?  The control uses a fixed scale of 0.35 (A5's *initialisation*
    # value, chosen so the constant is not tuned on the outcome), while A5 learned
    # 0.53.  If the Gaussian weights were nearly uniform over the k-NN set the
    # manipulation would be too weak for a null result to mean anything, so the
    # effective sample size of the weight profile is reported for both scales.
    weight_ess = {scale: [] for scale in (0.35, 0.53, 1.0)}
    nearest_weight_share = {scale: [] for scale in (0.35, 0.53, 1.0)}
    for query in queries.tolist():
        distance = (points - points[query]).norm(dim=-1)
        distance[query] = float('inf')
        knn = torch.topk(distance, args.k, largest=False).indices.tolist()
        shell = shell_selection(coord, members, query, points, scales, radius,
                                args.cell, args.support, PC_RANGE, grid_shape, args.k)
        if not shell:
            continue
        overlaps.append(len(set(knn) & set(shell)) / max(len(shell), 1))
        for scale in weight_ess:
            delta = points[knn] - points[query]
            # Stable ESS: factor out the largest log-kernel before exponentiating.
            # Without this, a query whose neighbours are all far relative to the
            # scale underflows every kernel to zero; dividing by a clamped sum
            # then destroys the normalisation and yields an ESS above k, which is
            # impossible for normalised weights (it happened, and produced 1e34).
            log_kernel = -0.5 * (delta / scale).square().sum(dim=-1)
            log_kernel = log_kernel - log_kernel.max()
            kernel = torch.exp(log_kernel)
            normalised = kernel / kernel.sum()
            weight_ess[scale].append(float(1.0 / normalised.square().sum()))
            nearest_weight_share[scale].append(float(normalised[0]))
        knn_distances.append(float(distance[knn].mean()))
        shell_distances.append(float(distance[shell].mean()))

    mean_overlap = sum(overlaps) / len(overlaps)
    result = {
        'dist': args.dist, 'gaussians': args.gaussians, 'scale': args.scale,
        'cell': args.cell, 'support': args.support, 'k': args.k,
        'queries_analysed': len(overlaps),
        'mean_selection_overlap': round(mean_overlap, 4),
        'overlap_p05': round(float(torch.quantile(torch.tensor(overlaps), 0.05)), 4),
        'overlap_p50': round(float(torch.quantile(torch.tensor(overlaps), 0.50)), 4),
        'overlap_p95': round(float(torch.quantile(torch.tensor(overlaps), 0.95)), 4),
        'mean_distance_knn_m': round(sum(knn_distances) / len(knn_distances), 4),
        'mean_distance_shell_m': round(sum(shell_distances) / len(shell_distances), 4),
        'gaussian_weight_ess': {str(k): round(sum(v) / len(v), 3)
                                for k, v in weight_ess.items()},
        'gaussian_nearest_weight_share': {str(k): round(sum(v) / len(v), 3)
                                          for k, v in nearest_weight_share.items()},
        'uniform_weight_ess': float(args.k),
    }
    print(f"k = {args.k}, {result['queries_analysed']} queries on {args.dist} points "
          f"(cell {args.cell} m, support {args.support}, scale {args.scale})")
    print(f"  selection overlap: mean {result['mean_selection_overlap']:.3f} "
          f"(p05 {result['overlap_p05']:.3f}, p50 {result['overlap_p50']:.3f}, "
          f"p95 {result['overlap_p95']:.3f})")
    print(f"  mean neighbour distance: k-NN {result['mean_distance_knn_m']:.3f} m vs "
          f"shell {result['mean_distance_shell_m']:.3f} m")

    print('  weighting manipulation strength over the SAME k-NN set '
          '(uniform weights would give ESS = k):')
    for scale, value in result['gaussian_weight_ess'].items():
        share = result['gaussian_nearest_weight_share'][scale]
        print(f'    scale {scale}: weight ESS {value:.2f} of {args.k}, '
              f'nearest neighbour carries {share:.1%} of the weight')

    if mean_overlap > 0.8:
        print('\nVERDICT: the two selection rules pick almost the same neighbours, '
              'so the matched-k comparison in P1-6 section 4.2.1 is essentially a '
              'WEIGHTING test, not a selection test.')
    elif mean_overlap > 0.5:
        print('\nVERDICT: the selections overlap substantially but not fully, so '
              'the matched-k comparison mixes weighting with selection; both '
              'explanations stay open.')
    else:
        print('\nVERDICT: the two rules select largely DIFFERENT neighbours, so '
              'selection dominates and the matched-k attribution stays open -- a '
              'k-NN-plus-Gaussian-weights variant would be needed.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')


if __name__ == '__main__':
    main()
