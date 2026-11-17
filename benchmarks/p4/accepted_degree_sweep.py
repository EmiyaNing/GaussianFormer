"""Measure the neighbour count that actually drives the sparse G2Q inner loop.

``benchmarks/p0_matrix_sweeps.jsonl`` records ``candidate_pairs`` — the number of
Gaussians the *cell window* scans — but that number is set by ``max_scale``
alone and is **identical** for support_sigma 2.0/3.0/4.0 even though latency
changes 1.72 -> 2.59 -> 3.87 ms on the same input.  The missing explanatory
variable is the count of candidates that also pass the per-axis support-box
test; only those reach the full Gaussian evaluation (and only those the density
cap can cut).

``g2q_accepted_degree`` already computes it in ``O(sample * G)``; it was simply
never run over the sweep grid, so every cost table in this repo lacked it.  This
script fills that gap and reports the ratio, which is the factor by which the
candidate count over- or under-states the real work.

The measurement is a **count**, not a timing, so it is immune to the GPU
contention that makes absolute latencies in this environment unusable.  It does
allocate a ``sample x G x 3`` temporary and is therefore still queued behind
training rather than run concurrently (the host's OOM path is broken).

    python3.10 benchmarks/p4/accepted_degree_sweep.py
    python3.10 benchmarks/p4/accepted_degree_sweep.py --gaussians 76800 --sample 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _display(path):
    """Path for messages: repo-relative when possible, absolute otherwise."""
    try:
        return Path(path).relative_to(ROOT)
    except ValueError:
        return Path(path)
sys.path.insert(0, str(ROOT))

from benchmarks.bench_operators import (  # noqa: E402
    GRID_SHAPE, PC_RANGE, g2q_accepted_degree, g2q_pair_statistics, make_cloud)

# The sweep grid from P0 T0.2, so the two tables join on the same axes.
CELLS = (0.6, 1.2, 1.8, 2.4)
SCALES = (0.1, 0.35, 0.6, 0.8)
SUPPORTS = (2.0, 3.0, 4.0)
DISTRIBUTIONS = ('uniform', 'clustered')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gaussians', type=int, default=76800)
    parser.add_argument('--sample', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--real-points', default='benchmarks/results/real_points.pt',
                        help='only used when a distribution is "real"')
    parser.add_argument('--json', default='benchmarks/results/p4_accepted_degree.json')
    parser.add_argument('--device', default='cuda',
                        help='cpu allows a fast end-to-end dry run of this script '
                             'without the GPU; the measurement is a count, so the '
                             'cpu result differs only by point-cloud sampling cost')
    parser.add_argument('--quick', action='store_true',
                        help='only the support axis, which is where the candidate '
                             'count is provably blind')
    args = parser.parse_args(argv)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')

    jobs = []
    if args.quick:
        for support in SUPPORTS:
            for dist in DISTRIBUTIONS:
                jobs.append(dict(dist=dist, cell=1.8, scale=0.35, support=support))
    else:
        for dist in DISTRIBUTIONS:
            for cell in CELLS:
                jobs.append(dict(dist=dist, cell=cell, scale=0.35, support=3.0))
            for scale in SCALES:
                jobs.append(dict(dist=dist, cell=1.8, scale=scale, support=3.0))
            for support in SUPPORTS:
                jobs.append(dict(dist=dist, cell=1.8, scale=0.35, support=support))

    rows = []
    for job in jobs:
        cloud = make_cloud(job['dist'], args.gaussians, scale=job['scale'],
                           real_points_path=args.real_points, device=args.device,
                           seed=args.seed)
        valid = torch.ones(args.gaussians, dtype=torch.bool, device=args.device)
        candidates = g2q_pair_statistics(cloud.points, valid, PC_RANGE, job['cell'],
                                         job['support'], cloud.max_scale, GRID_SHAPE)
        accepted = g2q_accepted_degree(cloud.points, valid, cloud.scales,
                                       cloud.opacities, PC_RANGE, job['support'],
                                       sample=args.sample, seed=args.seed)
        ratio = accepted['accepted_degree_mean'] / max(candidates['degree_mean'], 1e-9)
        row = dict(job, gaussians=args.gaussians, sample=args.sample,
                   candidate_pairs=candidates['candidate_pairs'],
                   candidate_degree_mean=round(candidates['degree_mean'], 3),
                   accepted_degree_mean=round(accepted['accepted_degree_mean'], 3),
                   accepted_to_candidate=round(ratio, 4),
                   accepted_degree_p95=round(accepted['accepted_degree_p95'], 1),
                   accepted_degree_max=round(accepted['accepted_degree_max'], 1))
        rows.append(row)
        print(f"dist={row['dist']:12s} cell={row['cell']:.1f} scale={row['scale']:.2f} "
              f"support={row['support']:.1f}  candidates={row['candidate_degree_mean']:9.1f} "
              f"accepted={row['accepted_degree_mean']:9.1f} "
              f"accepted/candidate={row['accepted_to_candidate']:.4f} "
              f"p95={row['accepted_degree_p95']:.0f} max={row['accepted_degree_max']:.0f}",
              flush=True)

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'gaussians': args.gaussians,
                                  'sample': args.sample, 'rows': rows}, indent=1),
                      encoding='utf-8')
    print(f'wrote {_display(target)}')


if __name__ == '__main__':
    main()
