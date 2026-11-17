"""How much smaller is A4's neighbourhood than A5's? (P1-7's disclosed confound)

P1-7 shows that removing the *per-point* geometry (A4) recovers A5's deficit.  But
``freeze_geometry`` changes two coupled things: it removes the input-dependent
term **and** it changes the scale the model settles on -- A4's shared scale is
0.405 (measured from its checkpoint) while A5's realised per-point mean is 0.538
(``p4_gaussian_attributes.json``).  Since the support box half-width is
``support_sigma * scale``, A4 aggregates a smaller neighbourhood, and P1-6 4.2
showed that *smaller* neighbourhoods hurt thin classes.  The confound therefore
runs **against** P1-7's conclusion, which is why it must be quantified rather
than hand-waved.

This script measures the accepted neighbour count on the same real points for
both scales, using the same counting helper and sampling convention as
``p4_accepted_degree.json`` (512 queries, seed 0).  A5's per-point scale spread is
not recoverable from the flat dump, so both arms use a constant scale: the point
is the *box size* effect, which is what the confound is about, and A5's realised
mean is used for its arm.

    python3.10 benchmarks/p1/a4_field_size_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

DUMP = 'benchmarks/results/real_points.pt'
SUPPORT_SIGMA = 3.0
# A4's shared scale, read from out/a4_fixed_gaussian/epoch_40.pth (bias-only head).
A4_SCALE = 0.4051
# A5's realised per-point mean scale, from p4_gaussian_attributes.json.
A5_SCALE = 0.5381


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', default=DUMP)
    parser.add_argument('--support', type=float, default=SUPPORT_SIGMA)
    parser.add_argument('--sample', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json',
                        default='benchmarks/results/p1_a4_field_size.json')
    args = parser.parse_args(argv)

    import torch  # noqa: PLC0415
    from benchmarks.bench_operators import g2q_accepted_degree  # noqa: PLC0415

    path = ROOT / args.dump
    if not path.exists():
        raise SystemExit(f'missing point dump {args.dump}; run '
                         f'benchmarks/p4/dump_real_points.py first')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    points = payload['points'].float()
    pc_range = list(payload['pc_range'])
    valid = torch.ones(points.shape[0], dtype=torch.bool)
    print(f'{points.shape[0]} real points; support_sigma={args.support}; '
          f'sample={args.sample} seed={args.seed}')

    rows = []
    for label, scale in (('A4 global (0.4051)', A4_SCALE),
                         ('A5 realised mean (0.5381)', A5_SCALE)):
        scales = torch.full_like(points, float(scale))
        opacities = torch.ones(points.shape[0])
        out = g2q_accepted_degree(points, valid, scales, opacities, pc_range,
                                  args.support, sample=args.sample, seed=args.seed)
        half_width = args.support * scale
        row = {'label': label, 'scale': scale,
               'box_half_width_m': round(half_width, 4),
               'accepted_degree_mean': round(out['accepted_degree_mean'], 3),
               'accepted_degree_p95': round(out['accepted_degree_p95'], 1),
               'accepted_degree_max': round(out['accepted_degree_max'], 1)}
        rows.append(row)
        print(f"  {label:26s} half-width {half_width:.3f} m  "
              f"accepted {out['accepted_degree_mean']:.3f} "
              f"(p95 {out['accepted_degree_p95']:.0f})")

    a4, a5 = rows
    ratio = (a5['accepted_degree_mean'] / a4['accepted_degree_mean']
             if a4['accepted_degree_mean'] else float('nan'))
    width_ratio = a4['box_half_width_m'] / a5['box_half_width_m']
    print(f'\nhalf-width ratio A4/A5 = {width_ratio:.3f} '
          f'(-{(1 - width_ratio) * 100:.1f}%)')
    print(f'accepted-degree ratio A5/A4 = {ratio:.2f}x  '
          f'(cube-scaling from the box volume would predict '
          f'{1 / width_ratio ** 3:.2f}x)')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'script': 'benchmarks/p1/a4_field_size_probe.py',
        'dump': args.dump, 'points': int(points.shape[0]),
        'support_sigma': args.support, 'sample': args.sample, 'seed': args.seed,
        'rows': rows,
        'half_width_ratio': round(width_ratio, 4),
        'accepted_ratio_a5_over_a4': round(ratio, 3) if ratio == ratio else None,
        'note': 'A5 per-point scale spread is not recoverable from the flat dump; '
                'both arms use a constant scale, so this measures the box-size '
                'effect the confound is about.',
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
