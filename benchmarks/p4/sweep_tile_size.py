"""P4.3-adjacent design-space probe: how does the block G2V tile edge trade off?

``P4-2`` derived an analytic bound: cooperative tile sharing can cut Gaussian
*load counts* from ``L = sum_g |voxels(g)|`` to ``U = sum_g |tiles(g)|``, a
factor of ~22x for the real workload.  ``P4-3`` measured the default 4x4x4 tile
at 8.14x faster than the naive dense operator, i.e. ~37% of that bound.  The
obvious next question is whether the tile edge is the knob that closes the gap.

Two effects compete, and they are separable:

* **Sharing** — a larger tile means a Gaussian's voxel AABB is covered by fewer
  tiles, so ``U`` falls and the bound ``L/U`` rises.  This is pure geometry and
  computable on the CPU, which is what ``--bound-only`` does.
* **Occupancy / wasted work** — a larger tile means a bigger CTA and a longer
  per-tile union list scanned by every one of its voxel threads.  The union list
  is re-scanned once per voxel per chunk (``LOAD_CHUNK``), so the redundant work
  grows with the tile's union size times its voxel count.  That part needs the
  GPU.

Because the tile edge is a compile-time constant, each measurement needs a
rebuild.  The driver rebuilds, re-runs the numerical test for that build (a
larger tile changes the sharing logic, which is exactly where the original
coordinate-space bug lived), then times the block operator alone and restores
the default build even on failure.

    python3.10 benchmarks/p4/sweep_tile_size.py --bound-only
    python3.10 benchmarks/p4/sweep_tile_size.py --tiles 2 4 8 --reps 15
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _display(path):
    """Path for messages: repo-relative when possible, absolute otherwise."""
    try:
        return Path(path).relative_to(ROOT)
    except ValueError:
        return Path(path)
sys.path.insert(0, str(ROOT))

from benchmarks.data.point_distributions import GRID_SHAPE, GRID_SIZE  # noqa: E402

CU = ROOT / 'model/ops/block_g2v/block_g2v.cu'
SO = ROOT / 'model/ops/block_g2v/block_g2v.cpython-310-x86_64-linux-gnu.so'
DEFAULT_TILE = 4


def build_tile_rect(means_int, radii, grid, tile_edge):
    """Replicate the kernel's ``voxelRect`` + ``tileRect`` in torch.

    Kept in lockstep with ``block_g2v.cu`` on purpose: the two coordinate spaces
    (voxel indices vs tile indices) must not be mixed, and this function is the
    CPU authority the sweep's analytic curve is computed from.
    """
    grid_t = torch.tensor(grid, dtype=torch.int64)
    lo = torch.maximum(means_int - radii.unsqueeze(1), torch.zeros(3, dtype=torch.int64))
    hi = torch.minimum(means_int + radii.unsqueeze(1) + 1, grid_t)
    lo = torch.minimum(lo, grid_t)
    tile_lo = lo // tile_edge
    tile_hi = (hi + tile_edge - 1) // tile_edge
    return lo, hi, tile_lo, tile_hi


def bound_for(tile_edge, points_path, gaussians, scale, seed):
    """Analytic ``L`` and ``U`` for one tile edge, on the real point workload.

    ``means_int`` and ``radii`` are derived exactly as
    ``benchmarks/p3/bench_dense_g2v.py:build`` derives them, so this curve
    describes the same workload the timings use.  Note the ``* 3.0`` in the
    radius: the operator's reach is three scales, not one.
    """
    from benchmarks.data.point_distributions import PC_RANGE  # noqa: PLC0415

    payload = torch.load(ROOT / points_path, map_location='cpu')
    pool = payload['points'].float()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    order = torch.randperm(pool.shape[0], generator=generator)[:gaussians]
    means = pool[order]

    lower = means.new_tensor(PC_RANGE[:3])
    means_int = ((means - lower) / GRID_SIZE).to(torch.int32)
    radii = torch.ceil(torch.full((gaussians,), scale) * 3.0 / GRID_SIZE) \
        .to(torch.int32).clamp_min(1)

    lo, hi, tile_lo, tile_hi = build_tile_rect(means_int, radii, GRID_SHAPE, tile_edge)
    voxel_counts = (hi - lo).prod(dim=1).clamp(min=0)
    tile_counts = (tile_hi - tile_lo).prod(dim=1).clamp(min=0)
    total_voxels = int(voxel_counts.sum())
    total_tiles = int(tile_counts.sum())

    # The load-count bound L/U is NOT the whole story.  A voxel in a block kernel
    # re-scans its tile's *union* list, not its own list, so the inner-loop
    # iteration count is U * TILE_VOXELS rather than L.  Blocks therefore trade
    # unique memory loads (down by L/U) for redundant ALU scans (up by the
    # inflation below).  Which one dominates decides the best tile edge, and
    # neither number alone predicts it.
    tile_voxels = tile_edge ** 3
    scan_iterations = total_tiles * tile_voxels
    return {
        'tile_edge': tile_edge,
        'L_voxel_loads': total_voxels,
        'U_tile_loads': total_tiles,
        'bound_ratio': total_voxels / total_tiles if total_tiles else float('nan'),
        'naive_scan_iterations': total_voxels,
        'block_scan_iterations': scan_iterations,
        'alu_inflation': scan_iterations / total_voxels if total_voxels else float('nan'),
        'tile_voxels': tile_voxels,
        'tile_grid': [GRID_SHAPE[i] // tile_edge + (1 if GRID_SHAPE[i] % tile_edge else 0)
                      for i in range(3)],
    }


def rebuild(tile_edge):
    """Rebuild the extension in place for one tile edge."""
    # The render kernel puts one thread on each tile voxel, so a tile must fit in
    # one CTA; the kernel has a matching static_assert, but failing here saves a
    # confusing template dump.
    if tile_edge ** 3 > 1024:
        raise SystemExit(f'tile edge {tile_edge} needs {tile_edge ** 3} threads '
                         '(> 1024); the render kernel cannot use it')
    env = dict(os.environ, BLOCK_G2V_TILE_EDGE=str(tile_edge))
    command = [sys.executable, 'setup.py', 'build_ext', '--inplace']
    completed = subprocess.run(command, cwd=str(ROOT / 'model/ops/block_g2v'),
                               env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).strip().splitlines()[-5:]
        raise SystemExit(f'build failed for tile {tile_edge}: ' + ' | '.join(tail))

    # Verify the build instead of trusting the label.  ``tile_edge()`` returns
    # the compile-time constant baked into the freshly built extension; if the
    # -D plumbing ever broke, every row of the sweep would silently re-measure
    # the default tile and the report would present three identical numbers as a
    # design-space study.
    probe = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, ".");'
         ' from model.ops.block_g2v import tile_edge; print(tile_edge())'],
        cwd=str(ROOT), capture_output=True, text=True)
    reported = probe.stdout.strip()
    if probe.returncode != 0 or reported != str(tile_edge):
        raise SystemExit(
            f'build for tile {tile_edge} reports tile_edge()={reported!r} '
            f'(stderr: {probe.stderr.strip()[-200:]})')
    print(f'    verified: extension reports tile_edge()={reported}', flush=True)
    return completed


def check_numerics():
    """Re-verify the freshly built kernel before trusting any timing from it."""
    completed = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_block_g2v.py', '-q'],
        cwd=str(ROOT), capture_output=True, text=True)
    ok = completed.returncode == 0
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()][-1:]
    print(f'    numerics: {"PASS" if ok else "FAIL"} {tail[0] if tail else ""}',
          flush=True)
    return ok


DEFAULT_JSON = 'benchmarks/results/p4_tile_sweep.json'


def bound_only_target(json_arg):
    """Where ``--bound-only`` writes: never the sweep artifact unless asked.

    ``--bound-only`` is a CPU diagnostic, but it used to write the same default
    path as the GPU sweep -- so running it after the sweep would silently replace
    the measured rows with a bounds-only payload, destroying exactly the data the
    P4-10 report needs.  An explicit ``--json`` still goes where it is asked.
    """
    return ('benchmarks/results/p4_tile_bounds.json'
            if json_arg == DEFAULT_JSON else json_arg)


def time_tile(tile_edge, args):
    command = [sys.executable, 'benchmarks/p3/bench_dense_g2v.py',
               '--only', 'block', '--tile', str(tile_edge),
               '--gaussians', str(args.gaussians), '--scale', str(args.scale),
               '--reps', str(args.reps), '--warmup', str(args.warmup),
               '--points', args.points,
               '--json', f'benchmarks/results/p4_tile_{tile_edge}.json']
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    print(completed.stdout.strip()[-600:], flush=True)
    if completed.returncode != 0:
        print(completed.stderr.strip()[-400:], flush=True)
        return None
    payload = json.loads((ROOT / f'benchmarks/results/p4_tile_{tile_edge}.json')
                        .read_text(encoding='utf-8'))
    return next(r for r in payload['rows'] if r['operator'].startswith('block'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', type=int, nargs='*', default=[2, 4, 8])
    parser.add_argument('--gaussians', type=int, default=34560)
    parser.add_argument('--scale', type=float, default=0.53,
                        help='the model-learned scale (see P4-1/P4-6)')
    parser.add_argument('--points', default='benchmarks/results/real_points.pt')
    parser.add_argument('--reps', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bound-only', action='store_true',
                        help='compute the analytic L/U curve on the CPU only')
    parser.add_argument('--json', default='benchmarks/results/p4_tile_sweep.json')
    args = parser.parse_args(argv)

    bounds = [bound_for(tile, args.points, args.gaussians, args.scale, args.seed)
              for tile in args.tiles]
    print(f'analytic sharing bound (G={args.gaussians}, scale={args.scale}):')
    for row in bounds:
        print(f"  tile {row['tile_edge']}x{row['tile_edge']}x{row['tile_edge']}: "
              f"L={row['L_voxel_loads']:,} U={row['U_tile_loads']:,} "
              f"load_reduction={row['bound_ratio']:.2f}x "
              f"scans={row['block_scan_iterations']:,} "
              f"(ALU inflation {row['alu_inflation']:.2f}x) "
              f"tile_grid={row['tile_grid']}")

    if args.bound_only:
        payload = {'gaussians': args.gaussians, 'scale': args.scale,
                   'bounds': bounds, 'timings': None}
        target = ROOT / bound_only_target(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=1), encoding='utf-8')
        print(f'wrote {_display(target)} (no GPU timing)')
        return

    timings = {}
    try:
        for tile in args.tiles:
            print(f'--- tile {tile}: rebuild', flush=True)
            rebuild(tile)
            if not check_numerics():
                raise SystemExit(f'tile {tile} build failed its numerical test; '
                                 'refusing to report timings for it')
            timings[tile] = time_tile(tile, args)
    finally:
        # Never leave a non-default kernel in the tree: every other P3/P4 result
        # was produced with tile 4, and the in-place .so is shared.
        print(f'--- restoring default tile {DEFAULT_TILE}', flush=True)
        rebuild(DEFAULT_TILE)
        # ``rebuild`` already verified tile_edge() == DEFAULT_TILE.  The
        # numerical check must not be merely *called*: the restored kernel is the
        # one every downstream stage will use, so a failure here has to fail the
        # run rather than print and pass.  Raising inside ``finally`` would mask
        # an in-flight exception, so record it and raise afterwards.
        restore_ok = check_numerics()
        if not restore_ok:
            print('FATAL: the restored default-tile kernel failed its numerical '
                  'check; downstream results would be built on it',
                  file=sys.stderr, flush=True)

    if not restore_ok:
        raise SystemExit(
            f'the default tile ({DEFAULT_TILE}) is not numerically verified '
            f'after the sweep; refusing to leave the tree in that state')

    for row, tile in zip(bounds, args.tiles):
        timing = timings.get(tile)
        row['p50_ms'] = timing['p50_ms'] if timing else None
        # Every row here is backed by a build whose tile_edge() was verified to
        # equal this number before it was timed.
        row['verified_tile_edge'] = tile if timing else None
        row['speedup_vs_tile4'] = None
    base = bounds[args.tiles.index(DEFAULT_TILE)]['p50_ms'] \
        if DEFAULT_TILE in args.tiles and timings.get(DEFAULT_TILE) else None
    if base:
        for row in bounds:
            if row['p50_ms']:
                row['speedup_vs_tile4'] = round(base / row['p50_ms'], 3)
    payload = {'gaussians': args.gaussians, 'scale': args.scale,
               'reps': args.reps, 'bounds': bounds}
    target = ROOT / args.json
    target.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(f'wrote {_display(target)}')


if __name__ == '__main__':
    main()
