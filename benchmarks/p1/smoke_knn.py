"""Smoke-test a KNN adapter config on the GPU before spending a training run.

``A2'`` uses ``num_neighbors=215``, three orders of magnitude above the k=8 the
adapter has been trained with.  Two things could fail only on the real card and
only at the real k:

* the CUDA k-NN implementation (``mmcv.ops.knn``) may not support that k, and
* the gather's index tensor scales linearly with k -- 4 batch items x 34,560
  points x 215 neighbours x 8 bytes is ~238 MB per horizon, ~1 GB across the four
  training horizons, on top of whatever the op allocates internally.

Both are cheap to check in seconds and expensive to discover 50 minutes into a
queued run.  This script therefore builds the adapter **from the real config** on
the real device, runs a forward and a backward on a small but structurally
identical input, and asserts that the shapes survive and the gradients are
finite.  The queue calls it immediately before launching training.

    python3.10 benchmarks/p1/smoke_knn.py --config config/opus/ablations/p1_a2_knn_k215.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mmengine import Config  # noqa: E402

from benchmarks.common_paths import display  # noqa: E402


def build(config_path):
    from model.head.ablations.knn_residual import KNNResidualAggregator

    cfg = Config.fromfile(str(config_path))
    adapter = cfg.model.semantic_gaussian
    if not isinstance(adapter, dict):
        adapter = dict(adapter)
    kind = adapter.pop('type')
    assert kind == 'KNNResidualAggregator', (
        f'this smoke test only covers KNNResidualAggregator, got {kind}')
    # Build from the config's own parameters: a reduced stand-in would not
    # exercise the same num_children/k combination the run will use.
    head = KNNResidualAggregator(**adapter)
    return head, adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--queries', type=int, default=96,
                        help='queries in the smoke input (x children points)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')

    head, adapter = build(ROOT / args.config)
    nearest = int(adapter.get('num_neighbors', 8))
    head = head.to(args.device).float()
    queries, children = args.queries, head.num_children
    generator = torch.Generator(device=args.device).manual_seed(0)
    points = torch.rand(1, queries, children, 3, generator=generator,
                        device=args.device) * 10 - 5
    token = torch.randn(1, queries, head.embed_dims, generator=generator,
                        device=args.device)
    valid = torch.ones(1, queries, dtype=torch.bool, device=args.device)

    internal_k = min(nearest, queries * children)
    index_bytes = 1 * queries * children * internal_k * 8
    print(f'config        : {display(args.config)}')
    print(f'num_neighbors : {nearest} (k-NN over {queries * children} points)')
    print(f'index tensor  : {index_bytes / 2 ** 20:.1f} MiB per horizon for this '
          f'smoke input; at batch 4 x 34,560 points x 4 horizons it would be '
          f'{4 * 34560 * nearest * 8 * 4 / 2 ** 20:.0f} MiB')

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    out = head(token, points, valid)
    loss = out['residual'].float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - started

    residual = out['residual']
    assert residual.shape == (1, queries * children, head.num_classes), residual.shape
    assert torch.isfinite(residual).all(), 'non-finite residual'
    gradients = [p.grad for p in head.parameters() if p.grad is not None]
    assert gradients, 'no gradients reached the adapter'
    assert all(bool(torch.isfinite(g).all()) for g in gradients), \
        'non-finite gradient'
    print(f'forward+backward OK in {elapsed:.2f} s, residual finite, '
          f'{len(gradients)} parameter tensors received finite gradients')
    # ``max_memory_allocated`` is CUDA-only; reporting 0.0 MiB on a CPU dry run
    # would look like a measurement rather than an absence.
    peak = (f'{torch.cuda.max_memory_allocated() / 2 ** 20:.1f} MiB'
            if args.device == 'cuda' else 'n/a (cpu dry run)')
    print(f'peak allocated: {peak}')
    print('SMOKE OK')


if __name__ == '__main__':
    main()
