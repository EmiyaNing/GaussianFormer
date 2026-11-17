"""P1 evaluation matrix: run ``eval.py`` per variant and collect the metrics.

Plan reference: ``plans/sparse_render_plans/P1_frozen_backbone_semantic_validation.md``.

Writes one JSON row per variant (``benchmarks/results/p1_eval.json``) with the
SparseWorldStrict ``avg/mIoU`` and ``avg/IoU`` plus every per-horizon number,
so the P1 report is generated from recorded data rather than transcription.

    python3.10 benchmarks/p1/eval_matrix.py --variants a0_base a5_cap0 a5_cap32
    python3.10 benchmarks/p1/eval_matrix.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

# name -> (config, checkpoint)
ADAPTER_CKPT = 'out/p1_a5_gaussian/epoch_40.pth'
VARIANTS = {
    'a0_base': ('config/opus/ablations/p1_a0_base.py',
                'ckpts/sparseworld_strict_epoch_56.pth'),
    'a5_cap0': ('config/opus/ablations/p1_a5_gaussian.py', ADAPTER_CKPT),
    'a5_cap8': ('config/opus/ablations/p1_a5_cap8.py', ADAPTER_CKPT),
    'a5_cap32': ('config/opus/ablations/p1_a5_cap32.py', ADAPTER_CKPT),
    'a5_cap64': ('config/opus/ablations/p1_a5_cap64.py', ADAPTER_CKPT),
    'a5_cap128': ('config/opus/ablations/p1_a5_cap128.py', ADAPTER_CKPT),
    'a1_pointwise': ('config/opus/ablations/p1_a1_pointwise.py',
                     'out/a1_pointwise/epoch_40.pth'),
    'a3_graph_attention': ('config/opus/ablations/p1_a3_graph_attention.py',
                           'out/a3_graph_attention/epoch_40.pth'),
    'a2_knn_mean': ('config/opus/ablations/p1_a2_knn_mean.py',
                    'out/a2_knn_mean/epoch_40.pth'),
    'a2_knn_max': ('config/opus/ablations/p1_a2_knn_max.py',
                   'out/a2_knn_max/epoch_40.pth'),
    # Multi-seed repeats of the decisive pair (A5 Gaussian vs A2 KNN control).
    # The single-seed difference is ~0.05 mIoU, which is inside the likely
    # noise floor, so the plan's ">= 3 seeds, mean +- std" requirement is what
    # decides the P1 gate.
    # A2': KNN with A5's learned receptive field (~215 neighbours).  P1's central
    # comparison (A5 vs A2-k8) confounds the weighting with the neighbour count,
    # so this is the receptive-field- and compute-matched control.
    'a2_knn_k215': ('config/opus/ablations/p1_a2_knn_k215.py',
                    'out/a2_knn_k215/epoch_40.pth'),
    # The weighting-vs-selection control: A2's exact k-NN selection with a fixed
    # isotropic Gaussian weight profile instead of uniform.  Closes P1-6 4.2.1.
    'a2_knn_gaussian8': ('config/opus/ablations/p1_a2_knn_gaussian8.py',
                         'out/a2_knn_gaussian8/epoch_40.pth'),
    # A2'': effective-field match (ESS of A5's weighting is 59.3 at scale 0.53).
    'a2_knn_k59': ('config/opus/ablations/p1_a2_knn_k59.py',
                   'out/a2_knn_k59/epoch_40.pth'),
    # A4: the Gaussian kernel with fixed isotropic geometry.  Plan T1.1 lists it
    # and P1-4 recorded it as "not run".  It completes the *chain*
    # A2 --(kernel form)--> A4 --(learnable geometry)--> A5; it is not a 2x2
    # cell, because a uniform-weight arm that still uses the geometry does not
    # exist (uniform weights make the geometry irrelevant).
    'a4_fixed_gaussian': ('config/opus/ablations/p1_a4_fixed_gaussian.py',
                          'out/a4_fixed_gaussian/epoch_40.pth'),
    'a5_s43': ('config/opus/ablations/p1_a5_gaussian.py', 'out/a5_s43/epoch_40.pth'),
    'a5_s44': ('config/opus/ablations/p1_a5_gaussian.py', 'out/a5_s44/epoch_40.pth'),
    'a2_s43': ('config/opus/ablations/p1_a2_knn_mean.py', 'out/a2_s43/epoch_40.pth'),
    'a2_s44': ('config/opus/ablations/p1_a2_knn_mean.py', 'out/a2_s44/epoch_40.pth'),
    # Is the cap's accuracy cost a property of the cap, or of the bin size?
    # The cap keeps the first K neighbours found along cell shells, so at K=8
    # cell 0.6 keeps only 0.1% of the same neighbours as cell 1.8.  These two
    # re-evaluate the frozen A5 at cap=8 with the bin edge bracketed around the
    # trained value (1.8); P1-1's "-0.26 mIoU at cap=8" was measured at 1.8 only.
    'a5_cap8_cell0p9': ('config/opus/ablations/p4_cap8_cell0p9.py',
                        ADAPTER_CKPT),
    'a5_cap8_cell3p6': ('config/opus/ablations/p4_cap8_cell3p6.py',
                        ADAPTER_CKPT),
}

# The cap-cost curve (P1-1) rests on a single training seed, and the divisor
# artifact proved that single-run comparisons in this project are treacherous
# (P1-5).  Evaluating the same caps on the other two A5 checkpoints gives the
# curve a seed dimension for ~1 minute of GPU per point.  Only the checkpoint
# changes: the configs are the same files the seed-42 points used, so any
# difference is the seed, not a knob.
for _seed in ('s43', 's44'):
    for _cap in (8, 32, 64, 128):
        VARIANTS[f'a5_cap{_cap}_{_seed}'] = (
            f'config/opus/ablations/p1_a5_cap{_cap}.py',
            f'out/a5_{_seed}/epoch_40.pth')

NUMBER = r'([-+]?\d+\.?\d*)'


def parse_metrics(text):
    """Extract the SparseWorldStrict metric dict from an eval log."""
    metrics = {}
    # The average precision/recall keys are definition-gated.  An earlier
    # evaluator computed them with ``nanmean`` over horizons, whose divisor can
    # depend on the variant (see P1-5's failure class, one level down); logs from
    # that version carry no ``avg/occupiedPrecision_pooled`` flag, so refusing
    # them here is what stops a superseded number from entering the results file.
    if re.search(r"'avg/occupiedPrecision_pooled':\s*True", text):
        for key in ('avg/occupiedPrecision', 'avg/occupiedRecall'):
            match = re.search(rf"'{re.escape(key)}':\s*{NUMBER}", text)
            if match:
                metrics[key] = float(match.group(1))
        metrics['avg/occupiedPrecision_pooled'] = True
    else:
        for key in ('avg/occupiedPrecision', 'avg/occupiedRecall'):
            metrics.pop(key, None)
    for key in ('avg/mIoU', 'avg/IoU', 'avg/boundaryIoU', 'num_samples',
                'boundary_samples_per_horizon'):
        match = re.search(rf"'{re.escape(key)}':\s*{NUMBER}", text)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if '.' in value else int(value)
    # The mIoU divisor (P1-5): carried through so a run's class-divisor can be
    # compared with another run's *from the results file*, without re-parsing
    # per-class IoUs.  ``..._uniform_across_horizons`` is the one-glance flag.
    for key in ('avg/mIoU_divisor_min', 'avg/mIoU_divisor_max'):
        match = re.search(rf"'{re.escape(key)}':\s*(\d+)", text)
        if match:
            metrics[key] = int(match.group(1))
    match = re.search(r"'avg/mIoU_divisor_uniform_across_horizons':\s*(True|False)",
                      text)
    if match:
        metrics['avg/mIoU_divisor_uniform_across_horizons'] = match.group(1) == 'True'
    for horizon in (0, 2, 4, 6):
        match = re.search(rf"'{horizon}s/mIoU_finite_classes':\s*(\d+)", text)
        if match:
            metrics[f'{horizon}s/mIoU_finite_classes'] = int(match.group(1))
    for horizon in (0, 2, 4, 6):
        for suffix in ('mIoU', 'IoU', 'boundaryIoU', 'occupiedPrecision',
                       'occupiedRecall'):
            match = re.search(rf"'{horizon}s/{suffix}':\s*{NUMBER}", text)
            if match:
                metrics[f'{horizon}s/{suffix}'] = float(match.group(1))
    return metrics


def run_variant(name, config, checkpoint, timeout):
    work_dir = ROOT / 'out' / f'p1_eval_{name}'
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'eval.py', '--py-config', config,
               '--resume-from', checkpoint, '--work-dir', str(work_dir)]
    if not (ROOT / checkpoint).exists():
        return {'variant': name, 'config': config, 'checkpoint': checkpoint,
                'error': f'checkpoint not found: {checkpoint}'}
    # Opt-in per-sample dumps: set DSH_EVAL_PER_SAMPLE_DIR to a directory and each
    # variant writes its own file, which is what bootstrap_ci.py consumes.  One
    # shared path would make every variant overwrite the previous one.
    env = dict(os.environ)
    dump_dir = env.get('DSH_EVAL_PER_SAMPLE_DIR')
    if dump_dir:
        target = Path(dump_dir)
        if not target.is_absolute():
            target = ROOT / target
        target.mkdir(parents=True, exist_ok=True)
        env['DSH_EVAL_PER_SAMPLE'] = str(target / f'{name}.json')
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                               text=True, timeout=timeout, env=env)
    log = completed.stdout + '\n' + completed.stderr
    log_path = work_dir / 'eval.log'
    log_path.write_text(log, encoding='utf-8')
    metrics = parse_metrics(log)
    row = {'variant': name, 'config': config, 'checkpoint': checkpoint,
           'returncode': completed.returncode, 'metrics': metrics}
    if not metrics:
        tail = [line for line in log.strip().splitlines() if line.strip()][-3:]
        row['error'] = 'no metrics parsed; log tail: ' + ' | '.join(tail)[-400:]
    return row


def reparse(args):
    """Refresh every recorded row's metrics from its saved ``eval.log``.

    Adding a key to ``parse_metrics`` must not silently apply only to runs made
    afterwards: the earlier rows would then be quietly missing the metric, and a
    table built from them would look like a real result of "not measured".
    This re-reads the logs so all rows carry the same key set.
    """
    out_path = ROOT / args.out
    rows = json.loads(out_path.read_text(encoding='utf-8'))
    changed = 0
    for row in rows:
        log_path = ROOT / 'out' / f"p1_eval_{row['variant']}" / 'eval.log'
        if not log_path.exists():
            print(f"  {row['variant']}: no log, unchanged")
            continue
        metrics = parse_metrics(log_path.read_text(encoding='utf-8', errors='replace'))
        if not metrics:
            print(f"  {row['variant']}: log parsed to nothing, unchanged")
            continue
        before = set((row.get('metrics') or {}).keys())
        row['metrics'] = metrics
        added = sorted(set(metrics) - before)
        changed += 1
        print(f"  {row['variant']}: {len(metrics)} keys"
              + (f", added {added}" if added else ""))
    out_path.write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print(f'reparsed {changed}/{len(rows)} rows -> {_display(out_path)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='*', default=None)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--out', default='benchmarks/results/p1_eval.json')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--reparse', action='store_true',
                        help='re-extract metrics from the recorded eval logs '
                             'instead of running the GPU; use after adding a '
                             'new parsed key so earlier runs are not lost')
    args = parser.parse_args(argv)

    if args.list:
        for name in VARIANTS:
            print(name)
        return

    if args.reparse:
        reparse(args)
        return

    names = args.variants or list(VARIANTS)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with previously recorded variants: a later invocation must never
    # drop the results of an earlier one.
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
        config, checkpoint = VARIANTS[name]
        print(f'--- {name}: {config} <- {checkpoint}', flush=True)
        row = run_variant(name, config, checkpoint, args.timeout)
        rows.append(row)
        metrics = row.get('metrics') or {}
        if metrics:
            print(f"    avg/mIoU={metrics.get('avg/mIoU')} "
                  f"avg/IoU={metrics.get('avg/IoU')} "
                  f"samples={metrics.get('num_samples')}", flush=True)
        else:
            print(f"    FAILED: {row.get('error', 'unknown')[:200]}", flush=True)
        ordered = sorted(rows, key=lambda row: row.get('variant', ''))
        out_path.write_text(json.dumps(ordered, indent=1), encoding='utf-8')

    print(f'\nwrote {_display(out_path)}')


if __name__ == '__main__':
    main()
