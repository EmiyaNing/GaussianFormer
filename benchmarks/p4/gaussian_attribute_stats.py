"""Gaussian attribute statistics from a trained checkpoint (plan P4.2, item 2).

The plan's P4.2 asks, verbatim:

    高斯的尺度/不透明度/协方差分布（是否退化成近似各向同性？旋转是否真的被用上？）

That is a *property of the trained adapter*, not of the evaluation metrics, and
it can be answered in large part **without a forward pass**: the A5 adapter
produces its geometry through

    scales = scale_min + (scale_max - scale_min) * sigmoid(scale_head(hidden))
    opacity = sigmoid(opacity_head(hidden))

so the head's bias is the input-independent per-axis base scale, and the head's
weight rows determine how much the scale can vary with the input.  Reading those
from the checkpoint answers three concrete questions:

* **Is it nearly isotropic?**  Compare the three implied base scales from
  ``scale_head.bias``; the head is initialised with a *zero* weight and a bias
  equal to ``logit(initial_scale)`` for all three axes, so any anisotropy at the
  base level had to be learned.
* **How much anisotropy could it express?**  The three weight rows' pairwise
  cosine similarity and relative gap bound how differently the axes can move
  with the input.  Identical rows mean the only anisotropy available is the
  (constant) bias difference.
* **Did opacity move off its initialisation?**  ``logit(0.02)`` is the init.

Limitations, stated because they bound what the numbers mean: this sees the
*capacity and base point* of the parameterisation, not its realised distribution
over the data (that needs the query tokens from a forward pass).  The realised
*means* are in the training logs, which this script also reads when available, so
the two views can be put side by side.

    python3.10 benchmarks/p4/gaussian_attribute_stats.py
    python3.10 benchmarks/p4/gaussian_attribute_stats.py --checkpoint out/a4_fixed_gaussian/epoch_40.pth
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

# A5's configuration (config/opus/sparse_gaussian_world.py).
SCALE_MIN, SCALE_MAX = 0.1, 0.6
INITIAL_SCALE = 0.35
INITIAL_OPACITY = 0.02


def implied_scale(bias):
    return SCALE_MIN + (SCALE_MAX - SCALE_MIN) * torch.sigmoid(bias)


def describe(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location='cpu')
    state = payload.get('state_dict', payload)
    prefix = 'semantic_gaussian.'
    scale_w = state[prefix + 'scale_head.weight'].float()
    scale_b = state[prefix + 'scale_head.bias'].float()
    opacity_w = state[prefix + 'opacity_head.weight'].float()
    opacity_b = state[prefix + 'opacity_head.bias'].float()

    base_scales = implied_scale(scale_b)
    rows = scale_w / scale_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = (rows @ rows.T)
    pairwise = [cosine[0, 1].item(), cosine[0, 2].item(), cosine[1, 2].item()]
    # Relative gap of each row from the mean row: how differently the axes can
    # move for the same input.  Zero would mean the head cannot be anisotropic
    # beyond whatever the bias already encodes.
    mean_row = scale_w.mean(dim=0, keepdim=True)
    row_gap = ((scale_w - mean_row).norm(dim=1) / scale_w.norm(dim=1).clamp_min(1e-12))
    # The adapter sets ``scale_prob = (initial_scale - scale_min) /
    # (scale_max - scale_min)`` and biases the head to ``logit(scale_prob)``, so
    # the *scale* initialisation is 0.5 in probability space, NOT logit(0.35).
    # Getting this wrong makes "how far did the bias move from init" meaningless.
    init_scale_prob = (INITIAL_SCALE - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)
    init_bias = math.log(init_scale_prob / (1 - init_scale_prob))
    init_opacity_bias = math.log(INITIAL_OPACITY / (1 - INITIAL_OPACITY))

    return {
        'checkpoint': str(checkpoint.relative_to(ROOT)),
        'base_scales_per_axis': [round(float(v), 4) for v in base_scales],
        'base_scale_isotropy_ratio': round(float(base_scales.max() / base_scales.min()), 4),
        'base_scale_spread': round(float(base_scales.max() - base_scales.min()), 4),
        'scale_bias_delta_from_init': [round(float(v) - init_bias, 4) for v in scale_b],
        'scale_weight_row_cosines': [round(v, 6) for v in pairwise],
        'scale_weight_row_gap': [round(float(v), 4) for v in row_gap],
        'scale_weight_norm': round(float(scale_w.norm()), 4),
        'scale_weight_abs_mean': round(float(scale_w.abs().mean()), 6),
        'opacity': round(float(torch.sigmoid(opacity_b)), 5),
        'opacity_bias_delta_from_init': round(float(opacity_b) - init_opacity_bias, 4),
        'opacity_weight_norm': round(float(opacity_w.norm()), 4),
    }


def realised_means(log_path: Path):
    """Last logged mean scale/opacity, which is the *realised* distribution."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding='utf-8', errors='replace')
    out = {}
    for key in ('scale_mean', 'opacity_mean', 'residual_abs_mean'):
        values = re.findall(rf'{key}: current=([-+0-9.eE]+)', text)
        if values:
            out[key] = round(float(values[-1]), 4)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='*', default=[
        'out/p1_a5_gaussian/epoch_40.pth',
        'out/a5_s43/epoch_40.pth',
    ])
    parser.add_argument('--json', default='benchmarks/results/p4_gaussian_attributes.json')
    args = parser.parse_args(argv)

    rows = []
    for name in args.checkpoints:
        path = ROOT / name
        if not path.exists():
            print(f'skip (missing): {name}')
            continue
        row = describe(path)
        log = path.parent / '20260914_210517.log'
        # The trainer writes one timestamped log per run; take the newest.
        candidates = sorted(path.parent.glob('2026*.log'))
        if candidates:
            row['realised'] = realised_means(candidates[-1])
        rows.append(row)
        print(f"\n=== {row['checkpoint']}")
        print(f"  base scale per axis   : {row['base_scales_per_axis']}  "
              f"(max/min = {row['base_scale_isotropy_ratio']}, spread {row['base_scale_spread']})")
        print(f"  scale bias vs init    : {row['scale_bias_delta_from_init']}")
        print(f"  scale row cosines     : {row['scale_weight_row_cosines']}")
        print(f"  scale row gap         : {row['scale_weight_row_gap']}  "
              f"(0 = the axes cannot move independently)")
        print(f"  scale weight norm/|mean|: {row['scale_weight_norm']} / {row['scale_weight_abs_mean']}")
        print(f"  opacity               : {row['opacity']}  "
              f"(bias delta from init {row['opacity_bias_delta_from_init']})")
        if row.get('realised'):
            realised = row['realised'].get('scale_mean')
            base = sum(row['base_scales_per_axis']) / 3
            print(f"  realised (training log): {row['realised']}")
            if realised:
                # Decompose the movement away from the isotropic initialisation:
                # zero weight at init means the bias moved base away from
                # INITIAL_SCALE, and the weights moved it further to the realised
                # mean.  Attributing the weights' share of the *gap from init* is
                # the meaningful split; a share of the realised value is not.
                bias_move = base - INITIAL_SCALE
                weight_move = realised - base
                total = bias_move + weight_move
                share = weight_move / total if total else float('nan')
                print(f"  -> init {INITIAL_SCALE} -> base(bias) {base:.4f} -> realised "
                      f"{realised:.4f}: bias moved it {bias_move:+.4f}, the weights "
                      f"{weight_move:+.4f} ({share:.0%} of the total movement)")

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'rows': rows}, indent=1), encoding='utf-8')
    print(f'\nwrote {target.relative_to(ROOT)}')

    for row in rows:
        ratio = row['base_scale_isotropy_ratio']
        gap = max(row['scale_weight_row_gap'])
        if ratio < 1.05 and gap < 0.05:
            print(f"\n{row['checkpoint']}: base scales differ by <5% and the weight rows "
                  'are near-identical, so the learned geometry is effectively '
                  'ISOTROPIC -- the axis-aligned anisotropy A5 is supposed to '
                  'provide is not being used.')
        elif ratio < 1.05:
            print(f"\n{row['checkpoint']}: the INPUT-INDEPENDENT base point stayed at its "
                  f'isotropic initialisation (per-axis spread {ratio:.4f} = '
                  f'{(ratio - 1) * 100:.1f}%), but the weight rows differ substantially '
                  f'(max row gap {gap:.2f}), so any per-axis geometry is produced '
                  'entirely by the input-dependent term.  This is capacity and usage '
                  'in the weights, NOT the realised per-axis spread, which needs a '
                  'forward pass to measure.')
        else:
            print(f"\n{row['checkpoint']}: base scale ratio {ratio}, max row gap "
                  f'{gap}: the parameterisation does express per-axis geometry.')


if __name__ == '__main__':
    main()
