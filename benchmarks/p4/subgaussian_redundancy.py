"""Sub-Gaussian redundancy diagnostic: plan P4.2 item 3.

The plan asks, verbatim:

    同一查询下各子高斯占用体素的**去重比例（unique voxel ratio）**与两两距离
    （检验子高斯是否冗余）

That is a property of the model's own query geometry: each query emits 48
children, and if those children land in the same voxel the query is spending
capacity on duplicates rather than on covering distinct occupied locations.

Input is the **grouped** dump from ``benchmarks/p4/dump_real_points.py --grouped``
(shape ``[S, Q*R, 3]`` plus a validity mask), because the original flat dump
applied a boolean mask that destroyed the grouping: of 4 x 34,560 points only
138,124 survive, so the 48-child boundaries cannot be recovered afterwards.

Two voxel resolutions are reported, because they answer different questions:

* **0.4 m** -- the evaluation voxel.  "Do two children of this query claim the
  same occupancy cell?" is the task-level question.
* **1.8 m** -- the operator's own bin edge (``cell_size``).  This is the scale at
  which the aggregation treats two children as interchangeable.

CPU-only; needs the grouped dump, not a GPU.

    python3.10 benchmarks/p4/subgaussian_redundancy.py
    python3.10 benchmarks/p4/subgaussian_redundancy.py --voxels 0.4 1.8 3.6
"""
from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402


def percentile(values, fraction):
    if not values.numel():
        return float('nan')
    return float(torch.quantile(values.float(), fraction))


def unique_voxel_ratio(children, voxel_size, pc_lower):
    """Distinct occupied voxels per valid child, one value per query."""
    index = torch.floor((children - pc_lower) / voxel_size).long()
    unique = torch.unique(index, dim=0).shape[0]
    return unique / children.shape[0]


def pairwise_distances(children):
    """Upper-triangle pairwise distances, in metres."""
    delta = children[:, None, :] - children[None, :, :]
    distance = delta.norm(dim=-1)
    rows, cols = torch.triu_indices(children.shape[0], children.shape[0], offset=1)
    if rows.numel() == 0:
        return torch.zeros(0)
    return distance[rows, cols]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dumped', default='benchmarks/results/real_points_grouped.pt')
    parser.add_argument('--voxels', nargs='*', type=float, default=[0.4, 1.8])
    parser.add_argument('--max-queries', type=int, default=4000,
                        help='cap on queries analysed per sample (the loop is '
                             'O(R^2) per query)')
    parser.add_argument('--json', default='benchmarks/results/p4_subgaussian_redundancy.json')
    args = parser.parse_args(argv)

    path = ROOT / args.dumped
    if not path.exists():
        raise SystemExit(
            f'grouped dump missing: {path}\n'
            'run: python3.10 benchmarks/p4/dump_real_points.py --grouped \\\n'
            '       --config config/opus/ablations/p1_a5_gaussian.py \\\n'
            '       --checkpoint out/p1_a5_gaussian/epoch_40.pth --samples 4 \\\n'
            '       --out benchmarks/results/real_points.pt')
    payload = torch.load(path, map_location='cpu')
    points = payload['points'].float()          # [S, Q*R, 3]
    valid = payload['valid'].bool()             # [S, Q*R]
    children = int(payload['children_per_query'])
    pc_lower = torch.tensor(payload['pc_range'][:3], dtype=torch.float32)
    queries = points.shape[1] // children
    print(f'grouped dump: {tuple(points.shape)} -> {queries} queries x {children} '
          f'children per sample, {int(valid.sum())} valid children')

    rows = []
    for voxel in args.voxels:
        ratios, all_distances = [], []
        for sample in range(points.shape[0]):
            sample_points = points[sample].view(queries, children, 3)
            sample_valid = valid[sample].view(queries, children)
            analysed = 0
            for query in range(queries):
                keep = sample_valid[query]
                count = int(keep.sum())
                if count < 2:
                    continue
                kids = sample_points[query][keep]
                ratios.append(unique_voxel_ratio(kids, voxel, pc_lower))
                all_distances.append(pairwise_distances(kids))
                analysed += 1
                if analysed >= args.max_queries:
                    break
        ratio_tensor = torch.tensor(ratios)
        distance_tensor = (torch.cat(all_distances) if all_distances
                           else torch.zeros(0))
        row = {
            'voxel_size': voxel,
            'queries_analysed': len(ratios),
            'unique_voxel_ratio_mean': round(float(ratio_tensor.mean()), 4),
            'unique_voxel_ratio_p05': round(percentile(ratio_tensor, 0.05), 4),
            'unique_voxel_ratio_p50': round(percentile(ratio_tensor, 0.50), 4),
            'unique_voxel_ratio_p95': round(percentile(ratio_tensor, 0.95), 4),
            'fully_redundant_query_fraction': round(
                float((ratio_tensor <= 1.0 / children + 1e-9).float().mean()), 4),
            'child_distance_mean_m': round(float(distance_tensor.mean()), 4),
            'child_distance_p50_m': round(percentile(distance_tensor, 0.50), 4),
            'child_distance_p95_m': round(percentile(distance_tensor, 0.95), 4),
            'child_distance_max_m': round(float(distance_tensor.max()), 4),
        }
        rows.append(row)
        print(f"\nvoxel {voxel} m over {row['queries_analysed']} queries:")
        print(f"  unique-voxel ratio   mean {row['unique_voxel_ratio_mean']:.3f} "
              f"(p05 {row['unique_voxel_ratio_p05']:.3f}, "
              f"p50 {row['unique_voxel_ratio_p50']:.3f}, "
              f"p95 {row['unique_voxel_ratio_p95']:.3f})")
        print(f"  queries fully redundant: {row['fully_redundant_query_fraction']:.1%}")
        print(f"  child distance (m)   mean {row['child_distance_mean_m']:.3f}, "
              f"p50 {row['child_distance_p50_m']:.3f}, "
              f"p95 {row['child_distance_p95_m']:.3f}, "
              f"max {row['child_distance_max_m']:.3f}")

    # The interpretation the plan is after: a ratio near 1 means every child
    # covers a distinct voxel (no redundancy); near 0 means they collapse onto
    # the same location (capacity wasted).
    fine = rows[0] if rows else None
    if fine:
        ratio = fine['unique_voxel_ratio_mean']
        if ratio > 0.9:
            print(f"\nVERDICT: at {fine['voxel_size']} m the children are nearly all "
                  f'distinct (ratio {ratio:.3f}) -- sub-Gaussians are NOT redundant '
                  'at the task voxel resolution.')
        elif ratio > 0.5:
            print(f"\nVERDICT: at {fine['voxel_size']} m the children are partly "
                  f'redundant (ratio {ratio:.3f}).')
        else:
            print(f"\nVERDICT: at {fine['voxel_size']} m the children collapse onto "
                  f'few distinct voxels (ratio {ratio:.3f}) -- query capacity is '
                  'largely spent on duplicates.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'dumped': args.dumped, 'children_per_query': children,
        'queries': queries, 'samples': int(points.shape[0]),
        'valid_children': int(valid.sum()),
        'rows': rows,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')


if __name__ == '__main__':
    main()
