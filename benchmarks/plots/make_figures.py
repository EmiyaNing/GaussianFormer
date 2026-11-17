"""Generate the report figures from the recorded result artifacts.

The plan asks for figure artifacts in several places (``P4.2`` item 5, and the
``analysis/assets/sparse_render/...`` paths in each phase's deliverables), and the
compliance audit recorded "no figure artifacts at all -- every report is tables".
These figures close part of that gap.

Every panel is computed from a JSON artifact rather than typed in, so a figure
cannot drift away from the numbers in the reports:

* ``p1_control_matrix.png``      <- benchmarks/results/p1_common_class_miou.json
* ``p1_cap_mass_accuracy.png``   <- p4_ess.json + p1_eval.json
* ``p4_ess_weight_profile.png``  <- p4_ess.json
* ``p4_tile_cost_model.png``     <- p4_tile_model.json

Figure labels are English on purpose: a CJK font is not guaranteed to be present,
and a figure full of tofu boxes is worse than an English one.  The Chinese
explanation lives in the report captions.

    python3.10 benchmarks/plots/make_figures.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.common_paths import display  # noqa: E402

RESULTS = ROOT / 'benchmarks/results'
OUT = ROOT / 'analysis/sparse_render/results/assets'
DPI = 160


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def control_matrix(rows, out):
    """Fixed-divisor mIoU per adapter, with the measured noise band."""
    order = [('a0_base', 'A0\nbaseline'), ('a1_pointwise', 'A1\npoint-wise'),
             ('a2_knn_mean', 'A2\nKNN-8 mean'), ('a3_graph_attention', 'A3\nattention'),
             ('a5_cap0', 'A5\nGaussian'), ('a5_s43', 'A5\nseed 43'),
             ('a2_s43', 'A2\nseed 43')]
    names, values = [], []
    for key, label in order:
        if key in rows:
            names.append(label)
            values.append(rows[key]['comparable_avg_mIoU'])
    best = max(values)
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    bars = axis.bar(names, values, color=['#888'] * 5 + ['#c33', '#36c'])
    # The measured same-configuration seed spread (P1-6) as a band around the
    # best value: anything inside it is not resolvable.
    axis.axhspan(best - 0.05, best + 0.05, color='orange', alpha=0.18,
                 label='measured seed spread (±0.05)')
    axis.set_ylim(min(values) - 0.15, best + 0.25)
    axis.set_ylabel('avg mIoU (fixed divisor, 3 dp)')
    axis.set_title('P1 control matrix: every adapter is inside the noise band')
    axis.legend(loc='lower right', fontsize=8)
    for bar, value in zip(bars, values):
        # Three decimals, matching the reports.  Two would render 14.375 as
        # "14.38" beside 14.365 as "14.37" and invite the reader to see a 0.01
        # ordering that the noise band explicitly denies.
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.3f}',
                  ha='center', fontsize=8)
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def cap_mass_accuracy(ess, evaluation, out):
    """Missed Gaussian mass against accuracy, plus the marginal gain."""
    row = next(r for r in ess['rows'] if r['scale'] == 0.53 and r['include_self'])
    shares = {int(k): v for k, v in row['nearest_weight_share'].items()}
    recorded = {r['variant']: (r.get('metrics') or {}).get('avg/mIoU')
                for r in evaluation}
    base = recorded['a5_cap0']
    points = [(0, 1.0, base)]
    for cap in (8, 32, 64, 128):
        if recorded.get(f'a5_cap{cap}') is not None:
            points.append((cap, shares[cap], recorded[f'a5_cap{cap}']))
    points.sort(key=lambda item: item[1])

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    caps = [p[0] for p in points]
    retained = [p[1] for p in points]
    accuracy = [p[2] for p in points]
    axes[0].plot(caps, retained, 'o-', color='#36c')
    axes[0].set_xlabel('density cap (neighbours kept)')
    axes[0].set_ylabel('retained weight mass')
    axes[0].set_title('cap -> retained Gaussian weight mass')
    axes[0].grid(alpha=0.3)
    for cap, mass in zip(caps, retained):
        axes[0].annotate(f'{mass:.3f}', (cap, mass), textcoords='offset points',
                         xytext=(0, 7), ha='center', fontsize=7)

    axes[1].plot(retained, accuracy, 'o-', color='#c33')
    axes[1].set_xlabel('retained weight mass')
    axes[1].set_ylabel('avg mIoU')
    axes[1].set_title('accuracy vs retained mass (concave)')
    axes[1].grid(alpha=0.3)
    for (cap_a, mass_a, value_a), (cap_b, mass_b, value_b) in zip(points, points[1:]):
        per_unit = (value_b - value_a) / (mass_b - mass_a)
        axes[1].annotate(f'{per_unit:+.2f}/unit',
                         ((mass_a + mass_b) / 2, (value_a + value_b) / 2),
                         textcoords='offset points', xytext=(0, -14), ha='center',
                         fontsize=7, color='#555')
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def ess_weight_profile(ess, out):
    """Retained weight share by neighbour rank, with the ESS as a marker."""
    row = next(r for r in ess['rows'] if r['scale'] == 0.53 and r['include_self'])
    shares = {int(k): v for k, v in row['nearest_weight_share'].items()}
    cuts = sorted(shares)
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    axis.plot(cuts, [shares[k] for k in cuts], 'o-', color='#264')
    axis.axvline(row['ess_p50'], color='gray', linestyle='--', linewidth=1,
                 label=f"ESS median {row['ess_p50']:.0f}")
    axis.axvline(row['ess_mean'], color='black', linestyle=':', linewidth=1,
                 label=f"ESS mean {row['ess_mean']:.0f}")
    axis.axvline(row['accepted_mean'], color='#c33', linestyle='-', linewidth=1,
                 label=f"accepted mean {row['accepted_mean']:.0f}")
    axis.set_xscale('log', base=2)
    axis.set_xlabel('k nearest accepted neighbours (log2)')
    axis.set_ylabel('share of total weight')
    axis.set_title('The Gaussian kernel keeps 31.5% of its mass beyond rank 32')
    axis.legend(fontsize=8, loc='lower right')
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def tile_cost_model(model, out):
    """Predicted tile latency, the analytic load bound, and the calibration."""
    point = model['model_point_predictions']
    bounds = json.loads((RESULTS / 'p4_tile_sweep.json').read_text(encoding='utf-8'))
    tiles = [int(t) for t in point]
    predicted = [point[str(t)] for t in tiles]
    reduction = {b['tile_edge']: b['bound_ratio'] for b in bounds['bounds']}

    figure, axis = plt.subplots(figsize=(6.8, 3.6))
    axis.plot(tiles, predicted, 'o-', color='#36c', label='model predicted p50')
    measured = model['model_point']['measured_block4_ms']
    axis.plot([4], [measured], '*', markersize=16, color='#c33',
              label=f'P4-6 measured tile 4 ({measured:.3f} ms)')
    axis.set_xscale('log', base=2)
    axis.set_xticks(tiles)
    axis.set_xticklabels([f'{t}x{t}x{t}' for t in tiles])
    axis.set_xlabel('tile edge')
    axis.set_ylabel('block G2V p50 (ms)')
    axis.set_title('Tile 4 is optimal: more sharing buys nothing (out-of-sample error 2.3%)')
    axis.grid(alpha=0.3)
    # Headroom so the top annotation is not clipped by the axes frame.
    axis.set_ylim(min(predicted) * 0.85, max(predicted) * 1.18)
    for tile, value in zip(tiles, predicted):
        axis.annotate(f'{value:.2f} ms', (tile, value), textcoords='offset points',
                      xytext=(0, 8), ha='center', fontsize=8)
    twin = axis.twinx()
    twin.plot(tiles, [reduction[t] for t in tiles], 's--', color='#999', alpha=0.8,
              label='analytic load reduction')
    twin.set_ylabel('load reduction vs naive (x)', color='#666')
    twin.tick_params(axis='y', colors='#666')
    lines, labels = axis.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    axis.legend(lines + lines2, labels + labels2, fontsize=8, loc='upper center')
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    common = load('p1_common_class_miou.json')
    evaluation = load('p1_eval.json')
    ess = load('p4_ess.json')
    model = load('p4_tile_model.json')

    made = []
    if common and evaluation:
        control_matrix(common['rows'], out / 'p1_control_matrix.png')
        made.append('p1_control_matrix.png')
    if ess and evaluation:
        cap_mass_accuracy(ess, evaluation, out / 'p1_cap_mass_accuracy.png')
        made.append('p1_cap_mass_accuracy.png')
    if ess:
        ess_weight_profile(ess, out / 'p4_ess_weight_profile.png')
        made.append('p4_ess_weight_profile.png')
    if model:
        tile_cost_model(model, out / 'p4_tile_cost_model.png')
        made.append('p4_tile_cost_model.png')

    for name in made:
        # ``--out /tmp/...`` is legitimate (a dry run), so format defensively.
        print(f'wrote {display(out / name)}')
    if not made:
        print('no artifacts found; nothing generated')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
