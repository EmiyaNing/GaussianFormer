"""P4 head-to-head: naive dense G2V vs S2GO-style block dense G2V.

Answers the question raised in
``analysis/sparse_render/results/P4-2_block_g2v_bound.md``: the analytic bound
says block sharing can cut Gaussian *load counts* by ~22x, but how much of that
is realisable in wall-clock time?

Both operators are timed on the same inputs, cycling operators **within each
repetition** so that the shared GPU's drift cannot be attributed to an operator
(measured p50 spread across repetitions is reported too).

    python3.10 benchmarks/p3/bench_dense_g2v.py --gaussians 76800 --reps 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.data.point_distributions import (  # noqa: E402
    GRID_SHAPE, GRID_SIZE, PC_RANGE, make_voxel_centres)


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def build(gaussians, points_path, scale, seed, device='cuda'):
    payload = torch.load(ROOT / points_path, map_location='cpu')
    pool = payload['points'].float()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    order = torch.randperm(pool.shape[0], generator=generator)[:gaussians]
    means = pool[order].to(device)
    scales = torch.full((gaussians, 3), scale, device=device)
    opacities = torch.rand(gaussians, generator=torch.Generator(device=device).manual_seed(seed),
                           device=device) * 0.8 + 0.1
    semantics = torch.randn(gaussians, 18,
                            generator=torch.Generator(device=device).manual_seed(seed + 1),
                            device=device)
    covariance = torch.zeros(gaussians, 3, 3, device=device)
    covariance[:, 0, 0] = 1.0 / scale ** 2
    covariance[:, 1, 1] = 1.0 / scale ** 2
    covariance[:, 2, 2] = 1.0 / scale ** 2

    lower = means.new_tensor(PC_RANGE[:3])
    points_int = ((make_voxel_centres(GRID_SHAPE, PC_RANGE, GRID_SIZE).squeeze(0)
                   - lower) / GRID_SIZE).to(torch.int32).contiguous()
    means_int = ((means - lower) / GRID_SIZE).to(torch.int32).contiguous()
    radii = torch.ceil(scales.max(dim=-1)[0] * 3.0 / GRID_SIZE) \
        .to(torch.int32).clamp_min(1).contiguous()
    cov6 = covariance.flatten(1)[:, [0, 4, 8, 1, 5, 2]].contiguous()
    return means, scales, opacities, semantics, covariance, points_int, means_int, radii, cov6


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gaussians', type=int, default=76800)
    parser.add_argument('--points', default='benchmarks/results/real_points.pt')
    parser.add_argument('--scale', type=float, default=0.35)
    parser.add_argument('--reps', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', default='benchmarks/results/p4_block_vs_naive.json')
    # Tile-size sweep support.  ``--tile`` labels the block operator; the actual
    # tile edge is a compile-time constant of the extension, so the sweep driver
    # (benchmarks/p4/sweep_tile_size.py) rebuilds the .so between invocations,
    # VERIFIES the build via ``block_g2v.tile_edge()`` and only then passes the
    # matching value here.  The label is therefore trustworthy, but it is still
    # a label -- the verified value lives in the sweep's own JSON.
    parser.add_argument('--tile', type=int, default=4,
                        help='tile edge the loaded block extension was built with')
    parser.add_argument('--only', choices=('both', 'naive', 'block'), default='both',
                        help='restrict timing to one operator (sweep mode)')
    args = parser.parse_args(argv)

    from model.head.localagg.local_aggregate import LocalAggregator
    from model.ops.block_g2v import block_g2v_forward

    (means, scales, opacities, semantics, covariance, points_int, means_int,
     radii, cov6) = build(args.gaussians, args.points, args.scale, args.seed)
    voxels = make_voxel_centres(GRID_SHAPE, PC_RANGE, GRID_SIZE)

    aggregator = LocalAggregator(
        scale_multiplier=3.0, H=GRID_SHAPE[0], W=GRID_SHAPE[1], D=GRID_SHAPE[2],
        pc_min=(PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]),
        grid_size=GRID_SIZE).cuda()

    def run_naive():
        with torch.no_grad():
            return aggregator(voxels, means.unsqueeze(0), opacities.unsqueeze(0),
                              semantics.unsqueeze(0), scales.unsqueeze(0),
                              covariance.unsqueeze(0))

    def run_block():
        return block_g2v_forward(voxels.squeeze(0), points_int, means, means_int,
                                 opacities, semantics, radii, cov6, GRID_SHAPE)

    block_name = f'block_{args.tile}x{args.tile}x{args.tile}'
    operators = {}
    if args.only in ('both', 'naive'):
        operators['naive_localagg'] = run_naive
    if args.only in ('both', 'block'):
        operators[block_name] = run_block
    for _ in range(args.warmup):
        for fn in operators.values():
            fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = {name: [] for name in operators}
    peaks = {name: 0.0 for name in operators}
    for _ in range(args.reps):
        for name, fn in operators.items():
            torch.cuda.reset_peak_memory_stats()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end))
            peaks[name] = max(peaks[name],
                              torch.cuda.max_memory_allocated() / 2 ** 20)

    rows = []
    for name, times in samples.items():
        rows.append({'operator': name,
                     'p50_ms': round(percentile(times, 0.50), 3),
                     'p95_ms': round(percentile(times, 0.95), 3),
                     'min_ms': round(min(times), 3),
                     'max_ms': round(max(times), 3),
                     'peak_alloc_mib': round(peaks[name], 1)})
        print(f"{name:16s} p50={rows[-1]['p50_ms']:8.3f} ms  "
              f"p95={rows[-1]['p95_ms']:8.3f}  min={rows[-1]['min_ms']:8.3f}  "
              f"peak={rows[-1]['peak_alloc_mib']:7.1f} MiB")

    naive = next((r['p50_ms'] for r in rows if r['operator'] == 'naive_localagg'), None)
    block = next((r['p50_ms'] for r in rows if r['operator'] == block_name), None)
    speedup = None
    if naive is not None and block is not None:
        speedup = naive / block
        print(f'\nblock / naive = {block / naive:.3f}  (speedup {speedup:.2f}x)')
        print(f'analytic load-count bound from P4-2 was ~22x')

    payload = {'gaussians': args.gaussians, 'scale': args.scale,
               'reps': args.reps, 'grid_shape': list(GRID_SHAPE),
               'tile_edge': args.tile,
               'rows': rows, 'speedup_p50': None if speedup is None else round(speedup, 3)}
    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(f'wrote {target.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
