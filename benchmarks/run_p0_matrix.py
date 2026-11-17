"""Run the P0.2 cost matrix and append every measurement as JSONL.

Plan reference: ``plans/sparse_render_plans/P0_correctness_and_cost_audit.md``
task T0.2.

Each row is written as it completes, so an interrupted run still leaves a
usable partial result file.  Use ``--only`` to run a subset.

    python3.10 benchmarks/run_p0_matrix.py --out benchmarks/results/p0_matrix.jsonl
    python3.10 benchmarks/run_p0_matrix.py --only g2q_core
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / 'benchmarks' / 'bench_operators.py'

DISTRIBUTIONS = ('uniform', 'ground_band', 'clustered', 'surface_mix')
GAUSSIAN_COUNTS = (9000, 34560, 76800)
LARGE = 76800


def run(op, **kwargs):
    """Invoke ``bench_operators.py`` and return its parsed JSON result."""
    tmp = ROOT / 'benchmarks' / 'results' / '_last.json'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    command = [sys.executable, str(BENCH), '--op', op, '--quiet', '--json', str(tmp)]
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               cwd=str(ROOT))
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()
        return {'op': op, 'error': tail[-1] if tail else 'unknown failure',
                'params': kwargs}
    if not tmp.exists():
        return {'op': op, 'error': 'benchmark produced no JSON output',
                'params': kwargs}
    return json.loads(tmp.read_text(encoding='utf-8'))


def build_jobs(only=None):
    """Yield ``(group, op, params)`` triples for the whole matrix."""
    jobs = []

    def add(group, op, **params):
        if only and group not in only:
            return
        jobs.append((group, op, params))

    # 1. Core table: distribution x Gaussian count, for every operator.
    for dist in DISTRIBUTIONS:
        for count in GAUSSIAN_COUNTS:
            add('core', 'g2q', dist=dist, gaussians=count)
            add('core', 'g2v', dist=dist, gaussians=count)
            add('core', 'knn_gather', dist=dist, gaussians=count)
            add('core', 'rasterizer', dist=dist, gaussians=count)
            add('core', 'pairs', dist=dist, gaussians=count)

    # 2. Cell size: the bounding parameter on G2Q's candidate scan.
    for cell in (0.6, 1.2, 1.8, 2.4):
        for dist in ('uniform', 'clustered'):
            add('cell', 'g2q', dist=dist, gaussians=LARGE, cell=cell)

    # 3. Scale / support: bounds the support box and the scan radius.
    for scale in (0.1, 0.35, 0.8):
        for dist in DISTRIBUTIONS:
            add('scale', 'g2q', dist=dist, gaussians=LARGE, scale=scale)
            add('scale', 'pairs', dist=dist, gaussians=LARGE, scale=scale)
    for support in (2.0, 3.0, 4.0):
        for dist in ('uniform', 'clustered'):
            add('support', 'g2q', dist=dist, gaussians=LARGE, support=support)

    # 4. Batch: does G2Q stay linear in batch?
    for batch in (1, 2, 4):
        add('batch', 'g2q', dist='uniform', gaussians=34560, batch=batch)
        add('batch', 'g2q', dist='clustered', gaussians=34560, batch=batch)

    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='benchmarks/results/p0_matrix.jsonl')
    parser.add_argument('--only', nargs='*', default=None,
                        help='subset of groups: core cell scale support batch')
    args = parser.parse_args(argv)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(set(args.only) if args.only else None)
    print(f'{len(jobs)} jobs -> {out_path}', flush=True)

    with out_path.open('w', encoding='utf-8') as handle:
        for index, (group, op, params) in enumerate(jobs, 1):
            started = time.time()
            result = run(op, **params)
            result['group'] = group
            result['params'] = params
            handle.write(json.dumps(result, sort_keys=True) + '\n')
            handle.flush()
            status = 'ERROR' if 'error' in result else 'ok'
            keys = []
            if 'forward' in result:
                keys.append(f"fwd={result['forward']['p50_ms']:.4f}ms")
                keys.append(f"mem={result['forward']['peak_alloc_mib']:.1f}MiB")
            if 'pairs' in result and 'candidate_pairs' in result.get('pairs', {}):
                keys.append(f"pairs={result['pairs']['candidate_pairs']}")
            if 'error' in result:
                keys.append(result['error'][:120])
            print(f'[{index}/{len(jobs)}] {group} {op} {params} -> {status} '
                  f'{" ".join(keys)} ({time.time() - started:.1f}s)', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
