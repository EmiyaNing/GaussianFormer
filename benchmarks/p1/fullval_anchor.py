"""Verify the published SparseWorldStrict anchors on the FULL val split (P3 T3.1).

The plan makes this a hard prerequisite:

    SparseWorldStrict 的发布目标为 0/2/4/6 horizon、无 camera/lidar mask 下
    Avg mIoU 13.20、Avg occupied IoU 22.03。主实验开始前必须先复现这两个数字，
    否则 A0 基线不可信。

Every P1/P4 result in this project was produced on a *reduced* val split, and
``tools/make_reduced_split.py`` keeps ``scene_order[:N]`` -- the first N scenes in
**sorted order**, not a random sample (val: 4 of 150 scenes, 161 keyframes, 113
evaluated samples).  The anchors were therefore never checked, and the P1 A0
number (14.23 mIoU / 20.17 IoU) sits on the wrong side of *both* anchors
(+1.03 mIoU, -1.86 IoU).

This script re-runs the same models on all 150 scenes / 6019 keyframes and
reports the anchors side by side.  It is deliberately a separate output file:
p1_eval.json holds reduced-split rows and T3.1 forbids mixing protocols in one
table.

Two things this settles at once:

* whether the evaluation protocol matches the published one at all (an external
  comparison is invalid if it does not), and
* whether the P1 A5-vs-A2 gap survives a **37x larger** eval sample -- the
  current 113 samples are the main source of the "differences are within noise"
  caveat.

    python3.10 benchmarks/p1/fullval_anchor.py --list
    python3.10 benchmarks/p1/fullval_anchor.py --variants a0_base
"""
from __future__ import annotations


import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

from benchmarks.p1.eval_matrix import parse_metrics  # noqa: E402

# Published anchors for 0/2/4/6 horizon, no camera/lidar mask (plan T3.1).
ANCHOR_MIOU = 13.20
ANCHOR_IOU = 22.03

# name -> (config, checkpoint, description)
VARIANTS = {
    'a0_base': ('config/opus/ablations/p4_fullval_a0.py',
                'ckpts/sparseworld_strict_epoch_56.pth',
                'A0: released SparseWorldStrict, no adapter'),
    'a2_knn_mean': ('config/opus/ablations/p4_fullval_a2.py',
                    'out/a2_knn_mean/epoch_40.pth',
                    'A2: KNN-8 mean residual adapter (P1 control)'),
    'a5_cap0': ('config/opus/ablations/p4_fullval_a5.py',
                'out/p1_a5_gaussian/epoch_40.pth',
                'A5: Gaussian adapter, cap=0'),
    'a4_fixed_gaussian': ('config/opus/ablations/p4_fullval_a4.py',
                          'out/a4_fixed_gaussian/epoch_40.pth',
                          'A4: Gaussian kernel, global (not per-point) geometry'),
}


def run_variant(name, config, checkpoint, timeout):
    work_dir = ROOT / 'out' / f'p1_fullval_{name}'
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'eval.py', '--py-config', config,
               '--resume-from', checkpoint, '--work-dir', str(work_dir)]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                               text=True, timeout=timeout)
    log = completed.stdout + '\n' + completed.stderr
    (work_dir / 'eval.log').write_text(log, encoding='utf-8')
    metrics = parse_metrics(log)
    row = {'variant': name, 'config': config, 'checkpoint': checkpoint,
           'changes': VARIANTS[name][2], 'returncode': completed.returncode,
           'metrics': metrics}
    if not metrics:
        tail = [line for line in log.strip().splitlines() if line.strip()][-3:]
        row['error'] = 'no metrics parsed; log tail: ' + ' | '.join(tail)[-400:]
    return row


def anchor_verdict(base, reduced_base, anchor_miou=ANCHOR_MIOU,
                   anchor_iou=ANCHOR_IOU, tolerance=0.1):
    """Which pre-registered branch the full-val A0 result falls in.

    * ``'reproduces'`` -- within ``tolerance`` of both published anchors, so the
      earlier gap was the reduced subset;
    * ``'protocol'`` -- within ``tolerance`` of the *reduced-split* A0 instead, so
      the sample set is not what differs; the protocol (class/mask convention)
      is, and the observed shape (mIoU higher, IoU lower than the anchor) says so;
    * ``'neither'`` -- moved away from both, so nothing can be concluded yet.
    """
    if base.get('avg/mIoU') is None or base.get('avg/IoU') is None:
        return 'neither'
    if (abs(base['avg/mIoU'] - anchor_miou) <= tolerance
            and abs(base['avg/IoU'] - anchor_iou) <= tolerance):
        return 'reproduces'
    if (reduced_base.get('avg/mIoU') is not None
            and reduced_base.get('avg/IoU') is not None
            and abs(base['avg/mIoU'] - reduced_base['avg/mIoU']) <= tolerance
            and abs(base['avg/IoU'] - reduced_base['avg/IoU']) <= tolerance):
        return 'protocol'
    return 'neither'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='*', default=None)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--json', default='benchmarks/results/p1_fullval.json')
    parser.add_argument('--reduced', default='benchmarks/results/p1_eval.json',
                        help='reduced-split results to print side by side')
    args = parser.parse_args(argv)

    if args.list:
        for name, (config, checkpoint, changes) in VARIANTS.items():
            print(f'{name:20s} {config:46s} <- {checkpoint:44s} {changes}')
        return

    names = args.variants or list(VARIANTS)
    out_path = ROOT / args.json
    existing = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            existing = []
    rows = [row for row in existing if row.get('variant') not in set(names)]

    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f'unknown variant {name!r}; see --list')
        config, checkpoint, _ = VARIANTS[name]
        if not (ROOT / checkpoint).exists():
            print(f'--- {name}: SKIPPED, checkpoint missing ({checkpoint})',
                  flush=True)
            continue
        print(f'--- {name}: {config} <- {checkpoint}', flush=True)
        row = run_variant(name, config, checkpoint, args.timeout)
        rows.append(row)
        metrics = row.get('metrics') or {}
        if metrics:
            print(f"    samples={metrics.get('num_samples')} "
                  f"avg/mIoU={metrics.get('avg/mIoU')} "
                  f"avg/IoU={metrics.get('avg/IoU')}", flush=True)
        else:
            print(f"    FAILED: {row.get('error', 'unknown')[:200]}", flush=True)
        out_path.write_text(json.dumps(sorted(rows, key=lambda r: r['variant']),
                                       indent=1), encoding='utf-8')

    # --- side-by-side with the reduced split, and with the anchors --------
    reduced = {}
    reduced_path = ROOT / args.reduced
    if reduced_path.exists():
        for row in json.loads(reduced_path.read_text(encoding='utf-8')):
            reduced[row['variant']] = row.get('metrics') or {}

    print('\n| variant | split | samples | avg/mIoU | avg/IoU |')
    print('| --- | --- | ---: | ---: | ---: |')
    for row in sorted(rows, key=lambda r: r['variant']):
        metrics = row.get('metrics') or {}
        print(f"| {row['variant']} | full val | {metrics.get('num_samples')} | "
              f"{metrics.get('avg/mIoU')} | {metrics.get('avg/IoU')} |")
    for name, metrics in sorted(reduced.items()):
        if metrics:
            print(f"| {name} | reduced val | {metrics.get('num_samples')} | "
                  f"{metrics.get('avg/mIoU')} | {metrics.get('avg/IoU')} |")
    print(f'| **published anchor** | full val | — | **{ANCHOR_MIOU}** | '
          f'**{ANCHOR_IOU}** |')

    base = next((r.get('metrics') or {} for r in rows if r['variant'] == 'a0_base'), {})
    if base.get('avg/mIoU') is not None:
        print(f"\nA0 vs anchor: mIoU {base['avg/mIoU'] - ANCHOR_MIOU:+.2f}, "
              f"IoU {base['avg/IoU'] - ANCHOR_IOU:+.2f}")
        branch = anchor_verdict(base, reduced.get('a0_base') or {})
        # The pre-registered second branch is "full val stays at the reduced-split
        # numbers", which is a sharper criterion than "not near the anchor": it
        # says the *protocol* (not the subset) is what differs, and it points at
        # the class/mask convention.  Print which branch materialised instead of
        # leaving that inference to the reader.
        if branch == 'reproduces':
            print('VERDICT (pre-registered branch 1): protocol reproduces the '
                  'published anchors; external comparison is licit and the '
                  'reduced split was the cause of the earlier gap.')
        elif branch == 'protocol':
            print('VERDICT (pre-registered branch 2): full val stays at the '
                  'reduced-split level, so the *protocol* differs from the '
                  'published one -- not the sample set.  Note the observed shape '
                  'is mIoU HIGHER and IoU LOWER than the anchor, i.e. both '
                  'directions disagree; that points at the IoU/class-mask '
                  'convention and must be explained before any cross-paper '
                  'comparison.')
        else:
            print('VERDICT: neither branch -- the full-val numbers moved away from '
                  'both the anchor and the reduced split.  Report the movement '
                  'and the sample count before drawing any conclusion.')
    print(f'\nwrote {display(out_path)}')


if __name__ == '__main__':
    main()
