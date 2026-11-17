"""P3 enabler: whole-model inference profile for a config + checkpoint.

Plan reference: ``plans/sparse_render_plans/P3_fair_end_to_end_experiments.md``
and the efficiency story recommended by
``analysis/sparse_render/results/P1-4_full_control_matrix.md``.

The P0 cost matrix measures the *operator*.  A systems claim needs the whole
model: end-to-end latency and peak memory per frame, including the frozen
backbone, the transformer and the rasteriser.  This script produces exactly
that, and can sweep the Gaussian adapter's density cap in place because the cap
changes inference cost without changing the weights.

    python3.10 benchmarks/p3/profile_model.py \
      --config config/opus/ablations/p1_a5_gaussian.py \
      --checkpoint out/p1_a5_gaussian/epoch_40.pth \
      --caps 0 32 128 --iters 30

Notes
-----
* Latency is reported as p50/p95 from CUDA events after a warm-up, matching the
  P0 methodology so the two tables are comparable.
* ``--torch-threads`` is left alone; the dataloader is created with
  ``val_only=True`` and ``num_workers`` from the config.
* Cross-hardware speedups are never computed here: one device, one run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def build_model(config_path, checkpoint, strict=True):
    from mmengine.config import Config
    from mmseg.models import build_segmentor
    import model  # noqa: F401  (registers modules)

    config = Config.fromfile(str(ROOT / config_path))
    segmentor = build_segmentor(config.model)
    segmentor.init_weights()
    if checkpoint:
        state = torch.load(str(ROOT / checkpoint), map_location='cpu')
        state = state.get('state_dict', state)
        report = segmentor.load_state_dict(state, strict=strict)
        if not strict:
            # Never let a lenient load hide a real mismatch: print both sides.
            print(f'lenient load: missing={list(report.missing_keys)[:8]} '
                  f'unexpected={list(report.unexpected_keys)[:8]}')
    segmentor = segmentor.cuda()
    segmentor.eval()
    return config, segmentor


def get_val_loader(config):
    from dataset import get_dataloader
    _, val_loader = get_dataloader(
        config.train_dataset_config, config.val_dataset_config,
        config.train_loader, config.val_loader, dist=False, val_only=True,
        seed=config.get('dataloader_seed', None))
    return val_loader


def describe(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = 0
    adapter_module = getattr(model, 'semantic_gaussian', None)
    # OPUS keeps the adapter under the prediction head; SparseWorld's legacy
    # path keeps it directly on the segmentor.  Supporting both lets this
    # profiler compare historical pilot results without misidentifying the
    # current OPUS-v1 experiment.
    if adapter_module is None:
        adapter_module = getattr(getattr(model, 'head', None), 'gaussian', None)
    if adapter_module is not None:
        adapter = sum(p.numel() for p in adapter_module.parameters())
        cap = getattr(adapter_module, 'max_neighbours', None)
    else:
        cap = None
    return {'total_params': total, 'trainable_params': trainable,
            'adapter_params': adapter, 'max_neighbours': cap}


def collect_batches(loader, count):
    batches = []
    for data in loader:
        if len(batches) >= count:
            break
        batches.append(data)
    if not batches:
        raise RuntimeError('validation loader produced no batches')
    return batches


def run_once(model, data):
    moved = {key: (value.cuda() if isinstance(value, torch.Tensor) else value)
             for key, value in data.items()}
    images = moved.pop('img')
    with torch.no_grad():
        return model(imgs=images, metas=moved)


def set_cap(model, cap):
    if cap is None:
        return
    adapter = getattr(model, 'semantic_gaussian', None)
    if adapter is None:
        adapter = getattr(getattr(model, 'head', None), 'gaussian', None)
    if adapter is None or not hasattr(adapter, 'max_neighbours'):
        raise AttributeError('configured model does not expose a CUDA Gaussian density cap')
    adapter.max_neighbours = int(cap)


def profile_paired(model, batches, caps, warmup):
    """Time every cap on every batch, cycling caps within each batch.

    A sequential per-cap sweep is confounded by any drift in GPU load, which
    matters here because the card is shared with other jobs.  Cycling the caps
    inside each iteration makes the comparison paired: every cap sees exactly
    the same inputs under the same conditions, so only the *difference* between
    caps is interpreted, never the absolute level.
    """
    for data in batches[:warmup]:
        for cap in caps:
            set_cap(model, cap)
            run_once(model, data)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = {cap: [] for cap in caps}
    peaks = {cap: 0.0 for cap in caps}
    for data in batches[warmup:]:
        for cap in caps:
            set_cap(model, cap)
            torch.cuda.reset_peak_memory_stats()
            start.record()
            run_once(model, data)
            end.record()
            torch.cuda.synchronize()
            samples[cap].append(start.elapsed_time(end))
            peaks[cap] = max(peaks[cap],
                             torch.cuda.max_memory_allocated() / 2 ** 20)

    rows = []
    for cap in caps:
        times = samples[cap]
        rows.append({
            'max_neighbours': cap,
            'frames': len(times),
            'p50_ms': round(percentile(times, 0.50), 3),
            'p95_ms': round(percentile(times, 0.95), 3),
            'min_ms': round(min(times), 3),
            'max_ms': round(max(times), 3),
            'fps_p50': round(1000.0 / percentile(times, 0.50), 3),
            'peak_alloc_mib': round(peaks[cap], 2),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--caps', nargs='*', type=int, default=None,
                        help='density-cap values to sweep (default: as configured)')
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--json', default=None)
    parser.add_argument('--no-strict', action='store_true',
                        help='load the checkpoint leniently and print the '
                             'missing/unexpected keys (used when comparing a '
                             'dense-G2V control against sparse weights)')
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')

    config, model = build_model(args.config, args.checkpoint,
                                strict=not args.no_strict)
    loader = get_val_loader(config)
    info = describe(model)
    print(json.dumps({'config': args.config, 'checkpoint': args.checkpoint,
                      'device': torch.cuda.get_device_name(0), **info}, indent=1))

    caps = args.caps if args.caps is not None else [info['max_neighbours']]
    torch.cuda.empty_cache()
    batches = collect_batches(loader, args.warmup + args.iters)
    rows = profile_paired(model, batches, caps, args.warmup)
    for row in rows:
        print(f"cap={str(row['max_neighbours']):>6}  "
              f"p50={row['p50_ms']:8.3f} ms  p95={row['p95_ms']:8.3f} ms  "
              f"fps={row['fps_p50']:7.3f}  "
              f"peak={row['peak_alloc_mib']:8.1f} MiB", flush=True)

    if args.json:
        target = ROOT / args.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(
            {'config': args.config, 'checkpoint': args.checkpoint, **info,
             'profile': rows}, indent=1), encoding='utf-8')
        print(f'wrote {target.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
