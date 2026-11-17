"""P4.3 accuracy-side robustness probe for the sparse neighbourhood operator.

The plan's P4.3 asks whether the method breaks when conditions leave the
training configuration.  P0's sweeps answered the **cost** half of that question
(cell size, scale, support, batch); nothing has measured the **accuracy** half.
This probe does, for the two knobs that are inference-time only:

* ``cell_size``  — the operator's candidate-lookup bin edge (trained at 1.8 m);
* ``support_sigma`` — the reach multiplier, i.e. ``support_sigma * scale``
  (trained at 3.0).

Both are constructor arguments of the adapter, so a config override changes them
without touching a single weight or re-running training.  That makes any delta
here a clean measurement of the operator's domain shift, and it is deliberately
*not* the "change the evaluation voxel resolution" experiment: the GT comes from
``LoadOccupancyOcc3D(grid_size=.4)``, so re-voxelising the ground truth would
need new pipeline code and is recorded as a capability gap instead.

Metrics are parsed with the same parser the P1 matrix uses, so the numbers are
directly comparable with the P1 control matrix.

    python3.10 benchmarks/p4/robustness_probe.py --list
    python3.10 benchmarks/p4/robustness_probe.py
    python3.10 benchmarks/p4/robustness_probe.py --variants baseline cell0p9
    python3.10 benchmarks/p4/robustness_probe.py --reparse   # rebuild the JSON
                                                             # from stored logs,
                                                             # no GPU needed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _display(path):
    """Path for messages: repo-relative when possible, absolute otherwise."""
    try:
        return Path(path).relative_to(ROOT)
    except ValueError:
        return Path(path)
sys.path.insert(0, str(ROOT))

from benchmarks.p1.eval_matrix import parse_metrics  # noqa: E402

# The trained A5 adapter; identical for every variant, which is the whole point.
CHECKPOINT = 'out/p1_a5_gaussian/epoch_40.pth'

# Measured on 2026-09-15 (P4-8 section 8.1): two evaluations of this same frozen
# checkpoint with the same command line differed by 0.01 avg / up to 0.03 per
# horizon, with per-class differences up to 0.32 and *no* divisor change.  A knob
# delta at or below this floor is not a resolved effect, so the summary says so
# instead of leaving the pre-registered wording to be read as "detected".
SAME_CONFIG_FLOOR = 0.03

# name -> (config, what it changes relative to the trained configuration)
VARIANTS = {
    'baseline': ('config/opus/ablations/p1_a5_gaussian.py',
                 'trained configuration (cell 1.8, support 3.0)'),
    'cell0p9': ('config/opus/ablations/p4_rob_cell0p9.py',
                'cell 0.9 (half the trained bin edge)'),
    'cell3p6': ('config/opus/ablations/p4_rob_cell3p6.py',
                'cell 3.6 (double the trained bin edge)'),
    'support2': ('config/opus/ablations/p4_rob_support2.py',
                 'support_sigma 2.0 (smaller reach than trained)'),
    'support4': ('config/opus/ablations/p4_rob_support4.py',
                 'support_sigma 4.0 (larger reach than trained)'),
}


def run_variant(name, config, timeout):
    work_dir = ROOT / 'out' / f'p4_rob_{name}'
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'eval.py', '--py-config', config,
               '--resume-from', CHECKPOINT, '--work-dir', str(work_dir)]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                               text=True, timeout=timeout)
    log = completed.stdout + '\n' + completed.stderr
    (work_dir / 'eval.log').write_text(log, encoding='utf-8')
    metrics = parse_metrics(log)
    row = {'variant': name, 'config': config, 'changes': VARIANTS[name][1],
           'checkpoint': CHECKPOINT, 'returncode': completed.returncode,
           'metrics': metrics}
    if not metrics:
        tail = [line for line in log.strip().splitlines() if line.strip()][-3:]
        row['error'] = 'no metrics parsed; log tail: ' + ' | '.join(tail)[-400:]
    return row


def delta_verdicts(rows, base, tolerance=0.05, noise=None):
    """Per-axis worst |delta| and the pre-registered verdict.

    ``cell`` variants are a consistency check: the admitted neighbour set does not
    depend on the bin size, so an identical metric is the expected outcome and
    anything above the seed noise is a bug.  ``support`` variants change the
    support box itself, so a delta there is real model behaviour.  Returns
    ``{axis: (worst_abs_delta, verdict, n_variants)}``.

    ``noise`` is the measured *same-configuration* repeat floor (see
    ``SAME_CONFIG_FLOOR``).  Passing it qualifies the verdict, because "support
    deltas are real model behaviour" says what a difference would *mean*, not
    that one was detected: at or below the floor no effect is resolved.
    """
    out = {}
    for axis in ('cell', 'support'):
        group = [row for row in rows if axis in str(row.get('variant', ''))]
        values = [(row['variant'], (row.get('metrics') or {}).get('avg/mIoU'))
                  for row in group]
        values = [value for _, value in values if value is not None]
        if not values:
            continue
        worst = max(abs(value - base) for value in values)
        if axis == 'cell':
            verdict = 'CONSISTENT' if worst <= tolerance else 'DIFFERS'
        else:
            verdict = 'REAL DELTA'
        if noise is not None and worst <= noise:
            verdict += (f' but no effect resolved (<= {noise:.2f} same-config '
                        'repeat floor)')
        out[axis] = (worst, verdict, len(values))
    return out


def build_row(name, metrics, source):
    row = {'variant': name, 'config': VARIANTS[name][0], 'changes': VARIANTS[name][1],
           'checkpoint': CHECKPOINT, 'returncode': None, 'metrics': metrics}
    if source is not None:
        row['reparsed_from'] = str(source)
    if not metrics:
        row['error'] = 'no metrics parsed from the stored log'
    return row


def reparse_rows(names, root=None):
    """Rebuild rows from the ``eval.log`` files already on disk (no GPU).

    ``run_variant`` writes its full stdout+stderr to
    ``out/p4_rob_<name>/eval.log`` before parsing it, so the metrics are
    recoverable without touching the GPU.  That matters on this host: the card is
    usually occupied by the next training stage, and NVML is broken, so an
    accidental overwrite of the JSON would otherwise force a re-evaluation that
    cannot run until the queue finishes -- or, worse, tempt a concurrent run that
    makes the *training* stage OOM.

    Refuses if any requested variant has no log, so this can never quietly
    produce a partial artifact.
    """
    root = Path(root) if root is not None else ROOT
    rows, missing = [], []
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f'unknown variant {name!r}; see --list')
        log_path = root / 'out' / f'p4_rob_{name}' / 'eval.log'
        if not log_path.exists():
            missing.append(str(log_path))
            continue
        log = log_path.read_text(encoding='utf-8', errors='replace')
        try:
            source = log_path.relative_to(root)
        except ValueError:
            source = log_path
        rows.append(build_row(name, parse_metrics(log), source))
    if missing:
        raise SystemExit('cannot reparse, no stored eval log for: '
                         + ', '.join(missing))
    return rows


def print_summary(rows, out_path):
    """Per-variant deltas plus the pre-registered reading, for both modes."""
    by_name = {row['variant']: row.get('metrics') or {} for row in rows}
    base = by_name.get('baseline', {}).get('avg/mIoU')
    if base is None:
        return
    print('\n| variant | avg/mIoU | delta vs trained |')
    print('| --- | ---: | ---: |')
    for row in sorted(rows, key=lambda r: r.get('variant', '')):
        value = (row.get('metrics') or {}).get('avg/mIoU')
        delta = f'{value - base:+.2f}' if value is not None else 'FAILED'
        print(f"| {row['variant']} | {value if value is not None else 'FAILED'} "
              f"| {delta} |")

    # Pre-registered interpretation, printed next to the numbers so the log
    # cannot be read the wrong way later -- the report is written from it.
    # ``cell_size`` is only a candidate-lookup structure: the window always
    # covers the support box, so the admitted neighbour set (and therefore
    # the metric) must match the trained value up to float accumulation
    # order.  A material cell difference is a *bug*, not evidence of
    # robustness.  ``support_sigma`` changes the box itself, so its
    # differences are real model behaviour.
    print('\nPre-registered reading (P4-8 section 2.1.1):')
    print('  * cell variants are a CONSISTENCY CHECK: identical metrics are '
          'the expected outcome')
    print('    (the admitted set is cell-independent; a difference means an '
          'operator bug or an')
    print('    uncontrolled variable, NOT "robustness to the bin size").')
    print('  * support variants change the support box itself, so their '
          'deltas are REAL model behaviour.')
    for label, (worst, verdict, count) in delta_verdicts(
            rows, base, noise=SAME_CONFIG_FLOOR).items():
        print(f'  {label}: worst |delta| = {worst:.2f} mIoU '
              f'({count} variants) -> {verdict}')
        if label == 'cell' and verdict == 'DIFFERS':
            print('    WARNING: a cell variant moved by more than the seed '
                  'noise; investigate before reporting this as robustness.')
    print(f'\nwrote {_display(out_path)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='*', default=None)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--reparse', action='store_true',
                        help='rebuild the JSON from stored eval logs; no GPU')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--json', default='benchmarks/results/p4_robustness.json')
    args = parser.parse_args(argv)

    if args.list:
        for name, (config, changes) in VARIANTS.items():
            print(f'{name:10s} {config:52s} {changes}')
        return

    names = args.variants or list(VARIANTS)
    out_path = ROOT / args.json

    if args.reparse:
        rows = reparse_rows(names)
        ordered = sorted(rows, key=lambda row: row.get('variant', ''))
        out_path.write_text(json.dumps(ordered, indent=1), encoding='utf-8')
        print_summary(rows, out_path)
        return

    if not (ROOT / CHECKPOINT).exists():
        raise SystemExit(f'frozen adapter checkpoint missing: {CHECKPOINT}')

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
        config = VARIANTS[name][0]
        print(f'--- {name}: {config}  [{VARIANTS[name][1]}]', flush=True)
        row = run_variant(name, config, args.timeout)
        rows.append(row)
        metrics = row.get('metrics') or {}
        if metrics:
            print(f"    avg/mIoU={metrics.get('avg/mIoU')} "
                  f"avg/boundaryIoU={metrics.get('avg/boundaryIoU')} "
                  f"samples={metrics.get('num_samples')}", flush=True)
        else:
            print(f"    FAILED: {row.get('error', 'unknown')[:200]}", flush=True)
        ordered = sorted(rows, key=lambda row: row.get('variant', ''))
        out_path.write_text(json.dumps(ordered, indent=1), encoding='utf-8')

    # Summary against the trained configuration, so the report can quote it.
    print_summary(rows, out_path)


if __name__ == '__main__':
    main()
