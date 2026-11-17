"""Holdout test for the three-term tile cost model (report P4-10).

P4-10 showed that the two-parameter model (``U*l + S*s``) over-predicts every tile
and that the error grows with the tile edge.  The kernel explains why: the inner
loop cheaply rejects a (voxel, union-entry) pair with a per-axis AABB test before
the full evaluation, so a scan has two prices.  The accepted count is exactly
``L`` for every tile (double counting: each Gaussian is counted once per voxel it
covers), hence the rejection fraction is exact and grows with the tile.

That motivates

    T(tile) = U*l + L*s_full + (S - L)*s_rej

Fitting it on tiles 2/4/8 is an exact fit (three unknowns, three points) and
therefore *not* evidence.  This script fits on 2/4/8 and predicts tiles 3 and 6,
which were measured separately and took no part in the fit -- a real holdout.

    python3.10 benchmarks/p4/tile_holdout.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

FIT_TILES = (2, 4, 8)
HOLDOUT_TILES = (3, 6)


def load_rows(path):
    """``{tile_edge: row}`` for rows that carry a measured p50."""
    target = ROOT / path
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding='utf-8'))
    rows = payload.get('bounds') or payload.get('rows') or []
    out = {}
    for row in rows:
        if row.get('p50_ms'):
            out[int(row['tile_edge'])] = row
    return out


def design(row):
    """``[U, L, S-L]`` for one tile: union entries, accepted scans, rejected."""
    scans = row['block_scan_iterations']
    accepted = row['L_voxel_loads']
    return [row['U_tile_loads'], accepted, scans - accepted]


def fit(rows):
    matrix = np.array([design(rows[tile]) for tile in FIT_TILES], dtype=float)
    times = np.array([rows[tile]['p50_ms'] for tile in FIT_TILES], dtype=float)
    per_load, per_full, per_reject = np.linalg.solve(matrix, times)
    return per_load, per_full, per_reject


def fit_two_term(rows):
    """Two-parameter fit with the rejection term forced to zero."""
    matrix = np.array([[rows[tile]['U_tile_loads'], rows[tile]['L_voxel_loads']]
                       for tile in FIT_TILES], dtype=float)
    times = np.array([rows[tile]['p50_ms'] for tile in FIT_TILES], dtype=float)
    return np.linalg.lstsq(matrix, times, rcond=None)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', default='benchmarks/results/p4_tile_sweep.json')
    parser.add_argument('--holdout',
                        default='benchmarks/results/p4_tile_sweep_holdout.json')
    parser.add_argument('--json',
                        default='benchmarks/results/p4_tile_holdout.json')
    args = parser.parse_args(argv)

    fit_rows = load_rows(args.fit)
    holdout_rows = load_rows(args.holdout)
    missing = [tile for tile in FIT_TILES if tile not in fit_rows]
    if missing:
        raise SystemExit(f'fit tiles missing from {args.fit}: {missing}')
    missing = [tile for tile in HOLDOUT_TILES if tile not in holdout_rows]
    if missing:
        raise SystemExit(f'holdout tiles missing from {args.holdout}: {missing}')

    per_load, per_full, per_reject = fit(fit_rows)
    print(f'three-term fit on tiles {FIT_TILES}:')
    print(f'  l = {per_load:.4e} ms per union entry')
    print(f'  s_full = {per_full:.4e} ms per ACCEPTED scan')
    print(f'  s_rej  = {per_reject:.4e} ms per REJECTED scan')
    print(f'  ratio s_rej/s_full = {per_reject / per_full:.3f} '
          f'({"plausible" if 0 < per_reject < per_full else "IMPLAUSIBLE"})')
    # Each acceptance fraction is exact: accepted == L for every tile.
    for tile in FIT_TILES + HOLDOUT_TILES:
        row = (fit_rows | holdout_rows)[tile]
        scans = row['block_scan_iterations']
        print(f'  tile {tile}: U={row["U_tile_loads"]:,} '
              f'rejected={scans - row["L_voxel_loads"]:,} '
              f'({(scans - row["L_voxel_loads"]) / scans:.1%})')

    print(f'\nholdout predictions (tiles {HOLDOUT_TILES} took no part in the fit):')
    print('| tile | three-term | two-term (s_rej=0) | measured | 3-term error |'
          ' 2-term error |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: |')
    two_term = fit_two_term(fit_rows)
    records = {}
    worst = 0.0
    for tile in HOLDOUT_TILES:
        row = holdout_rows[tile]
        measured = row['p50_ms']
        three = float(np.dot(design(row), [per_load, per_full, per_reject]))
        pair = [row['U_tile_loads'], row['L_voxel_loads']]
        two = float(np.dot(pair, two_term))
        three_error = (three - measured) / measured
        two_error = (two - measured) / measured
        worst = max(worst, abs(three_error))
        records[str(tile)] = {
            'measured': measured, 'three_term': three, 'two_term': two,
            'three_term_error': three_error, 'two_term_error': two_error,
        }
        print(f'| {tile} | {three:.3f} ms | {two:.3f} ms | {measured:.3f} ms | '
              f'{three_error:+.1%} | {two_error:+.1%} |')
    print(f'\nworst three-term holdout error: {worst:.1%}')
    if worst <= 0.10:
        print('VERDICT: the three-term model predicts unseen tiles within 10% -- '
              'the cheap-rejection branch is the missing term.')
    else:
        print('VERDICT: the three-term model still misses the holdout by more '
              'than 10%; at least a fourth term is missing (occupancy?), and this '
              'analytic model should not be used for absolute predictions.')

    target = ROOT / args.json
    target.write_text(json.dumps({
        'script': 'benchmarks/p4/tile_holdout.py',
        'fit_tiles': list(FIT_TILES), 'holdout_tiles': list(HOLDOUT_TILES),
        'per_load_ms': per_load, 'per_full_scan_ms': per_full,
        'per_rejected_scan_ms': per_reject,
        'rejection_ratio': per_reject / per_full,
        'rows': records, 'worst_holdout_error': worst,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
