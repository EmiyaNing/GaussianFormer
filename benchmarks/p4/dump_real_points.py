"""P4.1 enabler: dump the real point distribution produced by a trained model.

Plan reference: ``plans/sparse_render_plans/P4_paper_level_analysis.md`` task
P4.1 ("复杂度与真实负载") and the P0 report's own limitation: the cost matrix
was measured on *synthetic* distributions.

The P0 matrix and the P0 density-cap sweep both used synthetic uniform /
ground-band / clustered points.  P3-1 then hinted the real distribution is much
denser than any of them (the model-level cap saving of ~137 ms exceeds the
synthetic operator-level saving of ~49 ms).  This script extracts the actual
points the adapter consumes, so the operator cost matrix can be re-measured on
the real workload instead of a proxy.

    python3.10 benchmarks/p4/dump_real_points.py \
      --config config/opus/ablations/p1_a5_gaussian.py \
      --checkpoint out/p1_a5_gaussian/epoch_40.pth --samples 4 \
      --out benchmarks/results/real_points.pt
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--grouped', action='store_true',
                        help='also save the (sample, query*child) structure and '
                             'the valid mask, so sub-Gaussian grouping survives. '
                             'The flat dump applies a boolean mask that destroys '
                             'the grouping needed for the redundancy diagnostic '
                             '(P4.2 item 3): 138,124 of 4x34,560 points survive, '
                             'so boundaries cannot be recovered afterwards.')
    parser.add_argument('--out', default='benchmarks/results/real_points.pt')
    parser.add_argument('--json', default=None)
    args = parser.parse_args(argv)

    from benchmarks.p3.profile_model import build_model, get_val_loader

    config, model = build_model(args.config, args.checkpoint)
    loader = get_val_loader(config)

    collected = []
    grouped_points, grouped_valid = [], []
    grouped_scales, grouped_opacities, grouped_logits = [], [], []
    with torch.no_grad():
        for index, data in enumerate(loader):
            if index >= args.samples:
                break
            moved = {key: (value.cuda() if isinstance(value, torch.Tensor) else value)
                     for key, value in data.items()}
            images = moved.pop('img')
            result = model(imgs=images, metas=moved)
            # OPUS uses these canonical output keys.  The fallback preserves
            # read-only access to legacy SparseWorld pilot checkpoints.
            points = result.get('gaussian_point_points',
                                result.get('current_gaussian_point_points'))
            valid = result.get('gaussian_point_valid_mask',
                               result.get('current_gaussian_point_valid_mask'))
            if points is None or valid is None:
                raise KeyError('model did not emit Gaussian point coordinates and mask')
            for batch in range(points.shape[0]):
                keep = valid[batch].bool()
                if bool(keep.any()):
                    collected.append(points[batch][keep].detach().cpu())
                if args.grouped:
                    grouped_points.append(points[batch].detach().cpu())
                    grouped_valid.append(keep.detach().cpu())
                    # The result dict already carries the adapter's own geometry
                    # (``gaussian_scales``/``gaussian_opacities``) and the fused
                    # per-child logits, so the plan's P4.2 items 2 and 4 can be
                    # measured from this dump without any model change.
                    for key, bucket in (
                            ('gaussian_scales', grouped_scales),
                            ('gaussian_opacities', grouped_opacities),
                            ('gaussian_point_logits', grouped_logits),
                            ('current_gaussian_point_logits', grouped_logits)):
                        if key in result:
                            bucket.append(result[key][batch].detach().cpu())

    if not collected:
        raise SystemExit('no points collected')

    cloud = torch.cat(collected, dim=0)
    adapter_cfg = config.model.get('semantic_gaussian', config.model.get('head', {}))
    lower = cloud.new_tensor(adapter_cfg['pc_range'][:3])
    upper = cloud.new_tensor(adapter_cfg['pc_range'][3:])
    inside = ((cloud >= lower) & (cloud < upper)).all(dim=-1)

    payload = {
        'points': cloud[inside],
        'pc_range': list(adapter_cfg['pc_range']),
        'config': args.config,
        'checkpoint': args.checkpoint,
    }
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)

    if args.grouped and grouped_points:
        children = int(adapter_cfg.get('num_children',
            adapter_cfg.get('point_multipliers', [1])[-1]))
        grouped_payload = {
            'points': torch.stack(grouped_points),          # [S, Q*R, 3]
            'valid': torch.stack(grouped_valid),            # [S, Q*R]
            'children_per_query': children,
            'pc_range': list(adapter_cfg['pc_range']),
            'config': args.config, 'checkpoint': args.checkpoint,
        }
        if grouped_scales:
            grouped_payload['scales'] = torch.stack(grouped_scales)      # [S,Q*R,3]
        if grouped_opacities:
            grouped_payload['opacities'] = torch.stack(grouped_opacities)  # [S,Q*R]
        if grouped_logits:
            grouped_payload['logits'] = torch.stack(grouped_logits)      # [S,Q*R,C]
        grouped_path = target.with_name(target.stem + '_grouped.pt')
        torch.save(grouped_payload, grouped_path)
        print(f'wrote {display(grouped_path)} '
              f'({tuple(grouped_payload["points"].shape)} = '
              f'[{len(grouped_points)} samples, queries*{children}])')

    summary = {
        'samples': len(collected),
        'points_total': int(cloud.shape[0]),
        'points_inside_range': int(inside.sum().item()),
        'per_sample_mean': round(float(cloud.shape[0]) / len(collected), 1),
        'out': str(target.relative_to(ROOT)),
    }
    # A coarse occupancy summary so the report can describe the real geometry.
    inside_points = cloud[inside]
    summary['z_percentiles'] = {
        f'p{q}': round(float(torch.quantile(inside_points[:, 2].float(), q / 100)), 3)
        for q in (5, 25, 50, 75, 95)
    }
    print(json.dumps(summary, indent=1))
    if args.json:
        (ROOT / args.json).write_text(json.dumps(summary, indent=1),
                                      encoding='utf-8')


if __name__ == '__main__':
    main()
