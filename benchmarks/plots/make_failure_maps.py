"""Render the spatial failure cases: plan P4.2 item 5 (partial).

The plan asks for

    可视化：被漏掉的边界、空域误检、细长物体

The confusion matrices in the other diagnostics can say *how much* the model
misses or hallucinates but not *where*.  With ``DSH_EVAL_SPATIAL_SAMPLES`` set the
evaluator records the flat voxel indices of each scene's true positives, false
positives and false negatives, which is enough to rasterise a bird's-eye view.

**Scope, stated because it bounds the claim**: this is a 2D BEV projection of the
0.4 m grid, i.e. a voxel counts as occupied if any z-cell in its column is.  It
locates errors in the ground plane, which is what "空域误检" and "被漏掉的边界"
need; the plan's "细长物体" really wants a 3D view or a zoomed crop, and this
figure cannot supply that.

Two panels per variant, so the reader can compare a baseline with an adapter:

    python3.10 benchmarks/plots/make_failure_maps.py \
        --variants a0=out/p1_persample/a0_base.json \
                   a5=out/p1_persample/a5_cap0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

def column_flags(entry, key):
    """BEV boolean map: does the column contain any voxel of this category?

    Each panel marks ONE category, so no precedence rule is needed.  A single map
    with priority encoding would hide a true positive behind a false positive in
    the same column; separate maps cannot.
    """
    shape = tuple(entry['grid_shape'])
    assert len(shape) == 3, shape
    depth, height, width = shape
    canvas = np.zeros((height, width), dtype=bool)
    indices = np.asarray(entry.get(key, []), dtype=np.int64)
    if indices.size == 0:
        return canvas
    rest = indices // depth
    y = rest % height
    x = rest // height
    canvas[y, x] = True
    return canvas


def summarise(payload):
    """Per-horizon FP/FN voxel counts, pooled over the captured scenes."""
    totals = {}
    for entry in payload.get('spatial', []):
        horizon = entry['horizon']
        bucket = totals.setdefault(horizon, {'tp': 0, 'fp': 0, 'fn': 0, 'scenes': 0})
        bucket['tp'] += len(entry.get('tp', []))
        bucket['fp'] += len(entry.get('fp', []))
        bucket['fn'] += len(entry.get('fn', []))
        bucket['scenes'] += 1
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', required=True,
                        help='name=path pairs pointing at per-sample dumps')
    parser.add_argument('--horizon', type=int, default=4,
                        help='horizon to render (4 s by default: where adapters '
                             'differ most in the P1 per-horizon tables)')
    parser.add_argument('--scene', type=int, default=0,
                        help='which captured scene to render')
    parser.add_argument('--out', default='analysis/sparse_render/results/assets')
    args = parser.parse_args(argv)

    loaded = []
    for spec in args.variants:
        name, _, path = spec.partition('=')
        target = Path(path)
        if not target.is_absolute():
            target = ROOT / path
        if not target.exists():
            print(f'skip {name}: {path} missing')
            continue
        payload = json.loads(target.read_text(encoding='utf-8'))
        if not payload.get('spatial'):
            print(f'skip {name}: no spatial capture '
                  '(set DSH_EVAL_SPATIAL_SAMPLES before evaluating)')
            continue
        loaded.append((name, payload))
    if not loaded:
        raise SystemExit('no variant had spatial data; nothing to render')

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    for name, payload in loaded:
        entries = [e for e in payload['spatial']
                   if e['horizon'] == args.horizon and e['sample'] == args.scene]
        if not entries:
            print(f'skip {name}: no scene {args.scene} at {args.horizon} s')
            continue
        entry = entries[0]
        fp_map = column_flags(entry, 'fp')
        fn_map = column_flags(entry, 'fn')
        tp_map = column_flags(entry, 'tp')

        figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
        for axis, (canvas, title, colour) in zip(axes, (
                (fp_map, 'false positives (hallucinated space)', '#d62728'),
                (fn_map, 'false negatives (missed space)', '#1f77b4'))):
            # TP in light grey as context, so the reader sees what was right.
            axis.imshow(np.where(tp_map, 0.55, 1.0), cmap='gray', vmin=0, vmax=1,
                        interpolation='nearest')
            overlay = np.zeros((*canvas.shape, 4))
            overlay[canvas] = matplotlib.colors.to_rgba(colour)
            axis.imshow(overlay, interpolation='nearest')
            axis.set_title(f'{title}\n{int(canvas.sum())} BEV columns', fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        counts = {key: len(entry.get(key, [])) for key in ('tp', 'fp', 'fn')}
        precision = counts['tp'] / max(counts['tp'] + counts['fp'], 1)
        recall = counts['tp'] / max(counts['tp'] + counts['fn'], 1)
        figure.suptitle(f'{name}  horizon {args.horizon} s, scene {args.scene}: '
                        f'TP {counts["tp"]}, FP {counts["fp"]}, FN {counts["fn"]} '
                        f'-> P={precision:.2f} R={recall:.2f}  (BEV over z; grey = TP)',
                        fontsize=9)
        figure.tight_layout()
        target = out / f'p4_failure_map_{name}_h{args.horizon}_s{args.scene}.png'
        figure.savefig(target, dpi=160)
        plt.close(figure)
        print(f'wrote {display(target)}')

    # Numeric summary so the report can quote counts, not just a picture.
    for name, payload in loaded:
        totals = summarise(payload)
        print(f'{name}:')
        for horizon in sorted(totals):
            bucket = totals[horizon]
            precision = bucket['tp'] / max(bucket['tp'] + bucket['fp'], 1)
            recall = bucket['tp'] / max(bucket['tp'] + bucket['fn'], 1)
            print(f"  {horizon} s: TP {bucket['tp']}, FP {bucket['fp']}, "
                  f"FN {bucket['fn']} over {bucket['scenes']} scenes -> "
                  f"precision {precision:.3f}, recall {recall:.3f}")


if __name__ == '__main__':
    main()
