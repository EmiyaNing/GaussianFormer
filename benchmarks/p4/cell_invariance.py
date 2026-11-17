"""Is the accepted neighbour set independent of ``cell_size``?

Plan reference: robustness probe (report ``P4-8``, section 2.1.1).

``cell_size`` is a candidate-lookup structure, not a model hyper-parameter: the
operator scans ``(2k+1)^3`` cells with ``k = ceil(support_sigma * max_scale /
cell)`` and then admits a candidate only if it passes the per-axis support box
test.  Since ``k * cell >= support_sigma * max_scale`` and two points at
distance ``d`` can differ in cell index by at most ``ceil(d / cell)``, the window
always covers the box -- so the *admitted* set must be identical for every cell
size, and any accuracy difference between cell variants would be a bug rather
than "robustness to the bin size".

That claim is analytic; this script checks it empirically on the real point
positions from ``benchmarks/results/real_points.pt`` at the same workpoint as
``benchmarks/results/p4_ess.json`` (uniform scale 0.53, support 3.0), so the
accepted-degree number here and the ``accepted_mean`` recorded there must agree.
It also runs a *heterogeneous* scale row (``scale_jitter``), because uniform
scales put every neighbour at the same relative distance and are a weak probe of
the box boundary.

CPU-only by design: it must not compete with training for the single GPU.

    python3.10 benchmarks/p4/cell_invariance.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.bench_operators import PC_RANGE  # noqa: E402
from benchmarks.common_paths import display  # noqa: E402
from benchmarks.data.point_distributions import (  # noqa: E402
    load_real_points, make_cloud)
from benchmarks.reference.g2q_bruteforce import (  # noqa: E402
    cell_coordinates, neighbour_cells)

CELLS = (0.6, 1.2, 1.8, 2.4, 3.6)


def brute_force_masks(points, scales, queries, support_sigma, scale_eps,
                      chunk=16):
    """``[len(queries), G]`` bool: neighbours passing the per-axis box test.

    Chunked over *queries* (not points) so each step is one large broadcast
    instead of ~1200 tiny ones per query.
    """
    neighbour_scales = scales.clamp_min(scale_eps).unsqueeze(0)   # [1, G, 3]
    out = torch.zeros(queries.shape[0], points.shape[0], dtype=torch.bool)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        delta = (points[None, :, :] - points[queries[start:stop]][:, None, :]).abs()
        relative = delta / neighbour_scales
        out[start:stop] = (relative <= support_sigma).all(dim=-1)
    return out


def window_mask(points, scales, query, cell, support_sigma, max_scale,
                scale_eps, members=None, coord=None, shape=None):
    """Bool mask over all points: admitted through the cell window + box test.

    ``members`` / ``coord`` / ``shape`` may be passed in to avoid recomputing the
    spatial hash for every query.
    """
    if coord is None:
        coord_t, shape_t = cell_coordinates(points, PC_RANGE, cell)
        shape = tuple(int(v) for v in shape_t)
        coord = [tuple(int(v) for v in row) for row in coord_t]
    if members is None:
        members = {}
        for index, key in enumerate(coord):
            members.setdefault(key, []).append(index)
    radius = neighbour_cells(support_sigma, max_scale, cell)
    base = coord[query]
    mask = torch.zeros(points.shape[0], dtype=torch.bool)
    neighbour_scales = scales.clamp_min(scale_eps)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                key = (base[0] + dx, base[1] + dy, base[2] + dz)
                if not (0 <= key[0] < shape[0] and 0 <= key[1] < shape[1]
                        and 0 <= key[2] < shape[2]):
                    continue
                candidates = members.get(key)
                if not candidates:
                    continue
                index = torch.tensor(candidates, dtype=torch.long)
                delta = (points[index] - points[query]).abs()
                relative = delta / neighbour_scales[index]
                mask[index[(relative <= support_sigma).all(dim=-1)]] = True
    return mask, radius


def run_row(cloud, queries, support_sigma, scale_eps=1e-4):
    """Compare every cell size against the brute-force set for each query."""
    points, scales = cloud.points, cloud.scales
    reference = brute_force_masks(points, scales, queries, support_sigma,
                                 scale_eps)
    degrees = reference.sum(dim=-1).float()
    cells = {}
    for cell in CELLS:
        coord_t, shape_t = cell_coordinates(points, PC_RANGE, cell)
        shape = tuple(int(v) for v in shape_t)
        coord = [tuple(int(v) for v in row) for row in coord_t]
        members = {}
        for index, key in enumerate(coord):
            members.setdefault(key, []).append(index)
        accepted, missing, extra, mismatched, max_symdiff = [], 0, 0, 0, 0
        radius = None
        for position, q in enumerate(queries):
            q = int(q)
            mask, radius = window_mask(points, scales, q, cell, support_sigma,
                                       cloud.max_scale, scale_eps,
                                       members=members, coord=coord, shape=shape)
            ref = reference[position]
            accepted.append(int(mask.sum()))
            miss = int((ref & ~mask).sum())
            ext = int((mask & ~ref).sum())
            missing += miss
            extra += ext
            if miss or ext:
                mismatched += 1
            max_symdiff = max(max_symdiff, miss + ext)
        deg = torch.tensor(accepted, dtype=torch.float32)
        cells[f'{cell:g}'] = {
            'k': int(radius),
            'window_cells': int((2 * radius + 1) ** 3),
            'window_radius_m': round(radius * cell, 6),
            'box_radius_m': round(support_sigma * cloud.max_scale, 6),
            'window_covers_box': bool(radius * cell + 1e-9 >=
                                      support_sigma * cloud.max_scale),
            'accepted_mean': round(float(deg.mean()), 3),
            'accepted_p50': round(float(deg.median()), 1),
            'missing_total': missing,
            'extra_total': extra,
            'mismatched_queries': mismatched,
            'max_symmetric_difference': max_symdiff,
        }
    return {
        'scale': cloud.scales.max().item(),
        'max_scale': round(cloud.max_scale, 6),
        'support_sigma': support_sigma,
        'sample': len(queries),
        'brute_force_accepted_mean': round(float(degrees.mean()), 6),
        'brute_force_accepted_p50': round(float(degrees.median()), 1),
        'brute_force_accepted_p95': round(float(degrees.quantile(0.95)), 1),
        'cells': cells,
    }


def cap_selection_scan(real_points, region, points_cap, caps, cells,
                       support_sigma, scale, seed=0):
    """How cell-dependent is the *capped* selection (a different question).

    The admitted set is cell-independent, but with ``max_neighbours > 0`` the
    operator keeps only the first ``cap`` neighbours found by visiting cell
    shells in increasing Chebyshev radius.  That order depends on the bin size,
    so the *selected* subset is expected to be cell-dependent -- this measures
    it, so the invariant in P4-8 section 2.1.1 is not extended to capped runs.

    A dense local region is used because a cap only binds where the local
    neighbour count exceeds it.  Callers keep ``points_cap`` modest: the
    reference returns a ``[G, G]`` bool matrix.
    """
    from benchmarks.reference.g2q_bruteforce import shell_cap_selection
    pool = load_real_points(real_points)
    keep = (pool[:, 0].abs() < region) & (pool[:, 1].abs() < region)
    points = pool[keep].contiguous()
    if points.shape[0] > points_cap:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(points.shape[0], generator=generator)[:points_cap]
        points = points[order].contiguous()
    scales = torch.full_like(points, scale)
    valid = torch.ones(points.shape[0], dtype=torch.bool)
    out = {'region_m': region, 'points': int(points.shape[0]),
           'support_sigma': support_sigma, 'max_scale': float(scale), 'caps': {}}
    for cap in caps:
        reference, scans = shell_cap_selection(
            points, scales, valid, PC_RANGE, 1.8, support_sigma, 1e-4, scale, cap)
        entry = {'reference_cell': 1.8, 'scans': int(scans), 'cells': {}}
        for cell in cells:
            got, scans_cell = shell_cap_selection(
                points, scales, valid, PC_RANGE, cell, support_sigma, 1e-4,
                scale, cap)
            identical = int((reference == got).all(dim=1).sum())
            jaccard = []
            for index in range(points.shape[0]):
                left, right = reference[index], got[index]
                inter = int((left & right).sum())
                union = int(left.sum()) + int(right.sum()) - inter
                jaccard.append(inter / max(union, 1))
            entry['cells'][f'{cell:g}'] = {
                'identical_rows': identical,
                'identical_fraction': round(identical / points.shape[0], 4),
                'mean_jaccard': round(sum(jaccard) / len(jaccard), 4),
                'scans': int(scans_cell),
            }
        out['caps'][str(cap)] = entry
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-points',
                        default='benchmarks/results/real_points.pt')
    # 34560 = the per-sample Gaussian count the adapter actually consumes, so
    # the brute-force mean must reproduce p4_ess.json's accepted_mean exactly
    # and the two artifacts cross-validate each other.
    parser.add_argument('--gaussians', type=int, default=34560)
    parser.add_argument('--scales', nargs='*', type=float, default=[0.53])
    parser.add_argument('--scale-jitters', nargs='*', type=float, default=[0.0,
                                                                         0.3])
    parser.add_argument('--support', type=float, default=3.0)
    parser.add_argument('--sample', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json',
                        default='benchmarks/results/p4_cell_invariance.json')
    # Opt-in: the cap-selection scan is the slow part (~2 min) and answers a
    # *different* question, so the admitted-set check stays fast by default.
    parser.add_argument('--cap-scan', action='store_true',
                        help='also measure how cell-dependent the capped '
                             'selection is (expected: strongly)')
    parser.add_argument('--cap-caps', nargs='*', default=[8, 32])
    parser.add_argument('--cap-cells', nargs='*', default=[0.6, 1.2, 2.4, 3.6])
    parser.add_argument('--cap-points', type=int, default=6000,
                        help='cap on the dense-region point count; the '
                             'reference materialises a [G, G] bool matrix')
    parser.add_argument('--cap-region', type=float, default=6.0,
                        help='keep points with |x|,|y| below this (metres) so '
                             'the local density makes the cap bind')
    args = parser.parse_args(argv)

    # The query sample must match ``g2q_accepted_degree``'s (randperm seed 0) so
    # the brute-force mean is comparable with p4_ess.json's accepted_mean.
    rows = []
    for scale in args.scales:
        for jitter in args.scale_jitters:
            cloud = make_cloud('real', args.gaussians, scale=scale,
                               scale_jitter=jitter, real_points_path=args.real_points,
                               device='cpu', seed=args.seed)
            generator = torch.Generator().manual_seed(args.seed)
            count = cloud.points.shape[0]
            queries = torch.randperm(count, generator=generator)[
                :min(args.sample, count)]
            row = run_row(cloud, queries, args.support)
            row['scale_jitter'] = jitter
            row['scale_requested'] = scale
            rows.append(row)
            print(f"scale={scale:.2f} jitter={jitter:.2f} "
                  f"max_scale={row['max_scale']:.4f} "
                  f"brute-force accepted={row['brute_force_accepted_mean']:.3f}")
            for cell, info in row['cells'].items():
                flag = 'OK' if (info['missing_total'] == 0
                                and info['extra_total'] == 0) else 'BUG'
                print(f"  cell={cell:>4s} k={info['k']} "
                      f"window={info['window_radius_m']:.2f}m "
                      f"accepted={info['accepted_mean']:8.3f} "
                      f"missing={info['missing_total']} extra={info['extra_total']} "
                      f"{flag}")

    payload = {
        'script': 'benchmarks/p4/cell_invariance.py',
        'real_points': args.real_points,
        'gaussians': args.gaussians,
        'sample': args.sample,
        'seed': args.seed,
        'device': 'cpu',
        'claim': 'the admitted neighbour set is identical for every cell size; '
                 'cell_size changes cost only',
        'rows': rows,
    }
    if args.cap_scan:
        scan = cap_selection_scan(args.real_points, args.cap_region,
                                  args.cap_points,
                                  [int(c) for c in args.cap_caps],
                                  [float(c) for c in args.cap_cells],
                                  args.support, args.scales[0], seed=args.seed)
        payload['cap_selection'] = scan
        payload['cap_selection_claim'] = (
            'with max_neighbours > 0 the SELECTED subset IS cell-dependent '
            '(shell order changes with the bin size); the admitted-set '
            'invariant does not extend to capped runs')
        print(f"\ncap-selection scan ({scan['points']} points, "
              f"|x|,|y| < {scan['region_m']} m):")
        for cap, entry in scan['caps'].items():
            for cell, info in entry['cells'].items():
                print(f"  cap={cap:>3s} cell={cell:>4s}: "
                      f"identical rows {info['identical_rows']}"
                      f"/{scan['points']} ({info['identical_fraction']:.3f}), "
                      f"mean Jaccard {info['mean_jaccard']:.3f}")
    out_path = ROOT / args.json
    out_path.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(f'\nwrote {display(out_path)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
