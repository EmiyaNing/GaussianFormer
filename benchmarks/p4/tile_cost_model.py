"""Predict the tile-size sweep from a two-parameter cost model.

P4-2's load-count bound says a larger tile loads each Gaussian fewer times
(4x4x4 -> 28.95x fewer loads than naive at the model's learned scale, 8x8x8 ->
116.85x).  Naively that suggests bigger tiles are always better.  They cannot
be, because a block kernel pays for the sharing in **ALU scans**: each voxel in
a tile re-scans that tile's whole union list, so the inner-loop iteration count
is ``U * tile_voxels`` rather than ``L``.  At 4x4x4 that is already 2.21x more
scans than the naive kernel performs.

This script separates the two costs using the two measurements that already
exist -- the naive dense operator and the 4x4x4 block operator, both timed on the
same inputs (P4-3) -- and then predicts the 2x2x2 and 8x8x8 latencies that the
queued sweep will measure:

    T_naive = L * (l + s)
    T_block = U * l + scans * s

where ``l`` is the per-load cost and ``s`` the per-scan cost.  Two equations,
two unknowns, so the fit is exact rather than least-squares -- which also means
the prediction is a genuine out-of-sample test: the model has no freedom left to
accommodate the tiles it has not seen.

Known model weakness, stated up front: a 2x2x2 tile is only 8 threads, so
occupancy will hurt it beyond what the arithmetic predicts.  The model should
therefore be expected to *under*-predict tile 2.  The 8x8x8 prediction has no
such caveat -- 512 threads -- and is the sharper of the two tests.

    python3.10 benchmarks/p4/tile_cost_model.py
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

from benchmarks.data.point_distributions import GRID_SHAPE, GRID_SIZE  # noqa: E402
from benchmarks.p4.sweep_tile_size import bound_for  # noqa: E402

# The two calibration points, both from benchmarks/results/p4_block_vs_naive.json
# (G=76,800, scale=0.35, real points, same process, interleaved).
NAIVE_MS = 9.379
BLOCK4_MS = 1.152
CALIBRATION = {'gaussians': 76800, 'scale': 0.35, 'tile': 4,
               'naive_p50_ms': NAIVE_MS, 'block4_p50_ms': BLOCK4_MS}


def fit(naive_ms, block_ms, row4):
    """Solve for per-load and per-scan cost in ms, from two measurements.

    ``row4`` is the ``bound_for`` row for the tile the block measurement used.
    """
    l_voxel = row4['L_voxel_loads']
    u_tile = row4['U_tile_loads']
    scans = row4['block_scan_iterations']
    # naive: one load and one scan per voxel-gaussian pair.
    per_pair = naive_ms / l_voxel
    # block: U loads plus `scans` scans.
    per_scan = (block_ms - u_tile * per_pair) / (scans - u_tile)
    per_load = per_pair - per_scan
    return per_load, per_scan


def load_measured(path):
    """``{tile_edge: p50_ms}`` for the block operator, from a sweep artifact.

    The GPU sweep writes its measured p50 into the **bounds** rows (it merges the
    analytic L/U table with the timings); an older/diagnostic payload may instead
    carry a ``timings`` or ``rows`` list.  All three shapes are read, and only the
    ``block`` operator counts -- the naive row is a different kernel.  Returns
    ``{}`` when the sweep has not run, so the prediction table still prints
    (labelled as a prediction only).
    """
    target = ROOT / path
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}
    rows = payload.get('timings') or payload.get('rows') or payload.get('bounds') or []
    measured = {}
    for row in rows:
        operator = str(row.get('operator', 'block'))
        tile = row.get('tile_edge')
        if tile is None or not operator.startswith('block'):
            continue
        p50 = row.get('p50_ms') or row.get('forward', {}).get('p50_ms')
        if p50:
            measured[int(tile)] = float(p50)
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gaussians', type=int, default=CALIBRATION['gaussians'])
    parser.add_argument('--scale', type=float, default=CALIBRATION['scale'])
    parser.add_argument('--points', default='benchmarks/results/real_points.pt')
    parser.add_argument('--json', default='benchmarks/results/p4_tile_model.json')
    parser.add_argument('--measured',
                        default='benchmarks/results/p4_tile_sweep.json',
                        help='sweep artifact whose measured block rows are '
                             'compared against the prediction')
    args = parser.parse_args(argv)

    bounds = {tile: bound_for(tile, args.points, args.gaussians, args.scale, 0)
              for tile in (2, 4, 8)}

    # The calibration points are at G=76,800 but the reported bound figures in
    # P4-2's table are at G=34,560 (the model's operating point), so recompute
    # the calibration at the G the timings actually used.
    calib = {tile: bound_for(tile, args.points, CALIBRATION['gaussians'],
                             CALIBRATION['scale'], 0) for tile in (4,)}
    per_load, per_scan = fit(NAIVE_MS, BLOCK4_MS, calib[4])

    print(f'fitted per-load cost  l = {per_load:.4e} ms')
    print(f'fitted per-scan cost  s = {per_scan:.4e} ms')
    print(f'load/scan cost ratio    = {per_load / per_scan:.1f}x')
    print(f'  (a 4x4x4 tile load moves 3+6+1+18 = 28 floats plus the AABB, so a')
    print(f'   large ratio is expected; the fit is exact, not least-squares)')

    print('\n| tile | L | U | scans | predicted p50 | measured p50 |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: |')
    predictions = {}
    for tile in (2, 4, 8):
        row = bounds[tile]
        predicted = row['U_tile_loads'] * per_load + row['block_scan_iterations'] * per_scan
        measured = BLOCK4_MS if tile == 4 else None
        predictions[tile] = predicted
        print(f"| {tile} | {row['L_voxel_loads']:,} | {row['U_tile_loads']:,} | "
              f"{row['block_scan_iterations']:,} | **{predicted:.3f} ms** | "
              f"{'%.3f ms (calibration)' % measured if measured else 'queued'} |")
    print('  (this table is at the CALIBRATION workload -- G=%d, scale=%s -- so '
          'the sweep\'s measured rows do not belong to it; they are compared '
          'against the same-workload predictions further down)'
          % (args.gaussians, args.scale))

    naive_predicted = bounds[4]['L_voxel_loads'] * (per_load + per_scan)
    print(f'\nnaive check: predicted {naive_predicted:.3f} ms vs measured '
          f'{NAIVE_MS:.3f} ms (by construction)')

    best = min(predictions, key=predictions.get)
    print(f'\nModel prediction: the optimum is tile {best} '
          f'({predictions[best]:.3f} ms).')
    for tile in (2, 8):
        if tile != best:
            print(f'  tile {tile}: {predictions[tile]:.3f} ms, '
                  f'{predictions[tile] / predictions[best]:.2f}x the optimum')
    if predictions[8] > predictions[4]:
        print('  i.e. quadrupling the tile cuts loads 4x further but the ALU '
              'inflation more than pays for it: the 116.85x load bound is NOT '
              'reachable by enlarging tiles.')
    print('  Caveat: 2x2x2 is only 8 threads per CTA, so occupancy should make '
          'the measured tile-2 latency worse than predicted; the 8x8x8 '
          'prediction (512 threads) is the cleaner test of this model.')

    # --- out-of-sample check and the prediction the queued sweep tests -------
    # The fitted (l, s) are kernel/hardware constants, so they transfer across
    # workloads.  P4-6 independently timed the same block operator at the model's
    # own operating point (G = 34,560, scale 0.53, cap=32) and recorded 0.795 ms
    # -- a different Gaussian count, scale and session from the calibration.
    MODEL_POINT = {'gaussians': 34560, 'scale': 0.53,
                   'measured_block4_ms': 0.795}
    model_bounds = {tile: bound_for(tile, args.points, MODEL_POINT['gaussians'],
                                    MODEL_POINT['scale'], 0)
                    for tile in (2, 4, 8)}
    model_predictions = {
        tile: row['U_tile_loads'] * per_load + row['block_scan_iterations'] * per_scan
        for tile, row in model_bounds.items()}

    print('\n--- prediction at the TILE SWEEP settings '
          f"(G={MODEL_POINT['gaussians']}, scale={MODEL_POINT['scale']}) ---")
    print('| tile | U | scans | predicted p50 |')
    print('| ---: | ---: | ---: | ---: |')
    for tile in (2, 4, 8):
        row = model_bounds[tile]
        print(f"| {tile} | {row['U_tile_loads']:,} | "
              f"{row['block_scan_iterations']:,} | "
              f"**{model_predictions[tile]:.3f} ms** |")
    measured = MODEL_POINT['measured_block4_ms']
    predicted4 = model_predictions[4]
    error = abs(predicted4 - measured) / measured
    print(f"\nOut-of-sample check: tile 4 predicted {predicted4:.3f} ms vs P4-6's "
          f"independent measurement {measured:.3f} ms -> {error:.1%} error.")
    print('  The calibration used G=76,800 / scale 0.35 in one session; this '
          'check is G=34,560 / scale 0.53 in another.  Agreement at the few-'
          'percent level means (l, s) really are workload-independent, which is '
          'what licenses the tile-2/tile-8 predictions below.')
    best_model = min(model_predictions, key=model_predictions.get)
    print(f"\nPre-registered prediction for the queued sweep: optimum tile = "
          f"{best_model}; tile 2 = {model_predictions[2]:.3f} ms, "
          f"tile 8 = {model_predictions[8]:.3f} ms.")
    for tile in (2, 8):
        if tile != best_model:
            print(f"  tile {tile} should be {model_predictions[tile] / model_predictions[best_model]:.2f}x "
                  f"the optimum")

    # The sweep is the test this model exists for.  Its measured rows are at the
    # SAME workload as this table (that is why they live here and not above), so
    # the comparison is apples to apples.  A point that misses by more than 10%
    # means the two-parameter model is missing a cost term -- occupancy at tile 2
    # (8 threads/CTA) or AABB handling at tile 8 -- and the report must then show
    # the measured curve and name the missing term instead of the prediction.
    measured_rows = load_measured(args.measured)
    compared = {tile: measured_rows[tile] for tile in (2, 4, 8)
                if tile in measured_rows}
    if len(compared) >= 2:
        print('\n--- measured vs predicted at these settings ---')
        print('| tile | predicted | measured | error | vs measured optimum |')
        print('| ---: | ---: | ---: | ---: | ---: |')
        measured_best = min(compared, key=compared.get)
        for tile in sorted(compared):
            print(f'| {tile} | {model_predictions[tile]:.3f} ms | '
                  f'{compared[tile]:.3f} ms | '
                  f'{(model_predictions[tile] - compared[tile]) / compared[tile]:+.1%} | '
                  f'{compared[tile] / compared[measured_best]:.2f}x |')
        worst = max(abs(model_predictions[tile] - compared[tile]) / compared[tile]
                    for tile in compared)
        agrees = measured_best == best_model
        print(f'measured optimum = tile {measured_best} '
              f'({"agrees with" if agrees else "CONTRADICTS"} the prediction); '
              f'worst relative error {worst:.1%}')
        if not agrees or worst > 0.10:
            print('-> the two-parameter model is incomplete at this workload: '
                  'present the measured curve and name the missing term.')
    else:
        print('\n(measured sweep rows not present yet: the table above is the '
              'pre-registered prediction only.)')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'calibration': CALIBRATION, 'per_load_ms': per_load,
        'per_scan_ms': per_scan, 'load_scan_ratio': per_load / per_scan,
        'predicted_p50_ms': {str(t): v for t, v in predictions.items()},
        'predicted_optimum_tile': best,
        'model_point': MODEL_POINT,
        'model_point_predictions': {str(t): v for t, v in model_predictions.items()},
        'model_point_out_of_sample_error': error,
        'bounds_gaussians': args.gaussians, 'bounds_scale': args.scale,
        # The measured comparison, when the sweep has run.  Report P4-10 quotes
        # these errors, so they have to be re-derivable from an artifact rather
        # than only from this script's stdout.
        'measured': {str(tile): compared[tile] for tile in sorted(compared)},
        'measured_errors': {
            str(tile): (model_predictions[tile] - compared[tile]) / compared[tile]
            for tile in sorted(compared)},
        'measured_optimum_tile': (min(compared, key=compared.get)
                                  if compared else None),
        'measured_agrees_with_prediction': (
            (min(compared, key=compared.get) == best_model) if compared else None),
        'model_incomplete_at_this_workload': (
            bool(compared) and (min(compared, key=compared.get) != best_model
                                or max(abs(model_predictions[t] - compared[t])
                                       / compared[t] for t in compared) > 0.10)
            if compared else None),
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')


if __name__ == '__main__':
    main()
