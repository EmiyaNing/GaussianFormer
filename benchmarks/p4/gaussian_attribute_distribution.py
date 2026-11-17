"""Gaussian attribute distribution and semantic aliasing: plan P4.2 items 2 and 4.

The plan asks, verbatim:

    item 2: 高斯的尺度/不透明度/协方差分布（是否退化成近似各向同性？旋转是否真的被用上？）
    item 4: 语义混叠、未使用高斯比例、类别一致的支持域

Both need the model's *realised* per-child attributes, not just its checkpoint.
``gaussian_attribute_stats.py`` could only read the scale/opacity **bias and
weight rows** from the checkpoint and therefore answered "capacity and usage in
the weights" -- it explicitly could not measure the realised per-axis spread.
It turns out no model change is needed for that: the segmentor's result dict
already carries ``gaussian_scales``, ``gaussian_opacities`` and the fused
``current_gaussian_point_logits``, which ``dump_real_points.py --grouped`` now
saves.  This script uses them.

What it measures:

* **realised scale per axis** -- mean/p05/p50/p95 for each axis separately, so a
  degenerate (isotropic) geometry is visible directly rather than inferred;
* **realised anisotropy** -- per-child ``max(scale)/min(scale)`` across the three
  axes, and the same quantity measured on the *base* scales from the checkpoint
  as the "no input dependence" reference;
* **opacity distribution**;
* **unused Gaussian ratio** -- the fraction of children masked out;
* **semantic aliasing / class-consistent support** -- among the children of one
  query that land in the *same* voxel, how often do they disagree on their top-1
  class?  Aliasing means the query spends capacity on two conflicting claims at
  one location; agreement means its support region is class-consistent.

CPU-only; needs the grouped dump (one GPU forward).

    python3.10 benchmarks/p4/gaussian_attribute_distribution.py
    python3.10 benchmarks/p4/gaussian_attribute_distribution.py --voxels 0.4 1.8
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


def quantiles(values):
    if not values.numel():
        return {name: float('nan') for name in ('mean', 'p05', 'p50', 'p95')}
    values = values.float()
    return {
        'mean': round(float(values.mean()), 4),
        'p05': round(float(torch.quantile(values, 0.05)), 4),
        'p50': round(float(torch.quantile(values, 0.50)), 4),
        'p95': round(float(torch.quantile(values, 0.95)), 4),
    }


def anisotropy(scale_children):
    """max/min across the three axes, per child (1.0 == perfectly isotropic)."""
    low = scale_children.min(dim=-1).values.clamp_min(1e-6)
    high = scale_children.max(dim=-1).values
    return high / low


def aliasing_statistics(children, logits, voxel_size, pc_lower):
    """Do same-voxel children of one query disagree on their top-1 class?

    Returns ``(aliased_voxels, multi_child_voxels, conflicting_children,
    total_children_in_multi)``.  A query whose children in one voxel all predict
    the same class has a *class-consistent support region*; disagreement is
    semantic aliasing.
    """
    index = torch.floor((children - pc_lower) / voxel_size).long()
    top1 = logits.argmax(dim=-1)
    # Group by voxel key without a Python dict per query: sort by key.
    keys = (index[:, 0] * 100003 + index[:, 1] * 1009 + index[:, 2])
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    sorted_top1 = top1[order]
    aliased = multi = conflicting = children_in_multi = 0
    start = 0
    while start < sorted_keys.numel():
        stop = start + 1
        while stop < sorted_keys.numel() and sorted_keys[stop] == sorted_keys[start]:
            stop += 1
        group = sorted_top1[start:stop]
        if group.numel() > 1:
            multi += 1
            children_in_multi += int(group.numel())
            majority = int(torch.mode(group).values)
            if int((group != majority).sum()) > 0:
                aliased += 1
                conflicting += int((group != majority).sum())
        start = stop
    return aliased, multi, conflicting, children_in_multi


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dumped', default='benchmarks/results/real_points_grouped.pt')
    parser.add_argument('--voxels', nargs='*', type=float, default=[0.4, 1.8])
    parser.add_argument('--base-attributes',
                        default='benchmarks/results/p4_gaussian_attributes.json')
    parser.add_argument('--json',
                        default='benchmarks/results/p4_attribute_distribution.json')
    args = parser.parse_args(argv)

    path = ROOT / args.dumped
    if not path.exists():
        raise SystemExit(
            f'grouped dump missing: {path}\n'
            'run: python3.10 tools/run_p4_subgaussian.sh  (or the dump command '
            'in its comments)')
    payload = torch.load(path, map_location='cpu')
    points = payload['points'].float()
    valid = payload['valid'].bool()
    children = int(payload['children_per_query'])
    pc_lower = torch.tensor(payload['pc_range'][:3], dtype=torch.float32)
    queries = points.shape[1] // children

    result = {
        'dumped': display(path),
        'samples': int(points.shape[0]), 'queries': queries,
        'children_per_query': children,
        'valid_children': int(valid.sum()),
        'total_children': int(valid.numel()),
        'unused_gaussian_ratio': round(1.0 - float(valid.float().mean()), 4),
    }
    print(f'grouped dump {tuple(points.shape)}: {queries} queries x {children} '
          f'children, {result["valid_children"]}/{result["total_children"]} valid')
    print(f"unused Gaussian ratio (item 4): {result['unused_gaussian_ratio']:.4f}")

    # --- item 2: realised scale / opacity distributions -------------------
    if 'scales' in payload:
        scales = payload['scales'].float()[valid]
        per_axis = {axis: quantiles(scales[:, axis]) for axis in range(3)}
        realised = anisotropy(scales)
        result['scale_per_axis'] = per_axis
        result['scale_anisotropy'] = quantiles(realised)
        realised_ratio = per_axis[2]['mean'] / max(per_axis[0]['mean'], 1e-9)
        result['scale_realised_axis_ratio'] = round(realised_ratio, 4)
        print('\nrealised scale per axis (item 2):')
        for axis, stats in per_axis.items():
            print(f'  axis {axis}: mean {stats["mean"]:.4f} '
                  f'(p05 {stats["p05"]:.4f}, p50 {stats["p50"]:.4f}, '
                  f'p95 {stats["p95"]:.4f})')
        print(f'  realised max/min across axes: mean '
              f'{result["scale_anisotropy"]["mean"]:.4f}, '
              f'p95 {result["scale_anisotropy"]["p95"]:.4f} '
              f'(1.0 = perfectly isotropic per child)')

        base = None
        base_path = ROOT / args.base_attributes
        if base_path.exists():
            base = json.loads(base_path.read_text(encoding='utf-8'))
            entry = base['rows'][0] if base.get('rows') else None
            if entry:
                result['checkpoint_base_scales'] = entry['base_scales_per_axis']
                base_ratio = (max(entry['base_scales_per_axis'])
                              / min(entry['base_scales_per_axis']))
                result['checkpoint_base_axis_ratio'] = round(base_ratio, 4)
                print(f"  checkpoint base scales (input-independent): "
                      f"{entry['base_scales_per_axis']} -> axis ratio {base_ratio:.4f}")
                if result['scale_realised_axis_ratio'] > base_ratio * 1.5:
                    print('  => the realised geometry is MORE anisotropic than the '
                          'isotropic base point, so the input-dependent term does '
                          'break axis symmetry.')
                else:
                    print('  => the realised geometry stays close to the isotropic '
                          'base point; per-axis differentiation is not being used.')
    else:
        print('\n[scales absent from the dump: re-dump with the current script]')

    if 'opacities' in payload:
        opacity = payload['opacities'].float()[valid].reshape(-1)
        result['opacity'] = quantiles(opacity)
        print(f"\nrealised opacity (item 2): mean {result['opacity']['mean']:.4f} "
              f"(p05 {result['opacity']['p05']:.4f}, p50 {result['opacity']['p50']:.4f}, "
              f"p95 {result['opacity']['p95']:.4f})")

    # --- item 4: semantic aliasing / class-consistent support -------------
    if 'logits' in payload:
        logits = payload['logits'].float()[valid]
        print('\nsemantic aliasing (item 4): do same-voxel children of one query '
              'agree on their top-1 class?')
        rows = []
        for voxel in args.voxels:
            aliased = multi = conflicting = in_multi = 0
            for sample in range(points.shape[0]):
                sample_points = points[sample].view(queries, children, 3)
                sample_valid = valid[sample].view(queries, children)
                sample_logits = payload['logits'].float()[sample].view(queries, children, -1)
                for query in range(queries):
                    keep = sample_valid[query]
                    if int(keep.sum()) < 2:
                        continue
                    a, m, c, t = aliasing_statistics(
                        sample_points[query][keep], sample_logits[query][keep],
                        voxel, pc_lower)
                    aliased += a
                    multi += m
                    conflicting += c
                    in_multi += t
            row = {
                'voxel_size': voxel,
                'multi_child_voxels': multi,
                'aliased_voxels': aliased,
                'aliased_voxel_fraction': round(aliased / multi, 4) if multi else None,
                'conflicting_child_fraction': (round(conflicting / in_multi, 4)
                                               if in_multi else None),
                'class_consistent_support_fraction': (round(1 - aliased / multi, 4)
                                                      if multi else None),
            }
            rows.append(row)
            print(f"  voxel {voxel} m: {multi} voxels hold >1 child; "
                  f"{aliased} disagree ({row['aliased_voxel_fraction']:.1%}), "
                  f"conflicting children {row['conflicting_child_fraction']:.1%}; "
                  f"class-consistent support {row['class_consistent_support_fraction']:.1%}")
        result['aliasing'] = rows
    else:
        print('\n[logits absent from the dump: re-dump with the current script]')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')


if __name__ == '__main__':
    main()
