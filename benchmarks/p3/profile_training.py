"""P3 enabler: training-step cost probe (peak memory and iteration time).

Plan reference: ``plans/sparse_render_plans/P3_fair_end_to_end_experiments.md``
("整模型 FPS、训练迭代时间、训练/推理峰值显存").

Why this exists
---------------
The parent analysis' central claim is that *training* is still dominated by
materialising a dense ``V x C`` field and by support-domain overlap.  The P0
matrix measured that at the **operator** level and P3-1 at **inference** level;
neither answers the training question.  This probe measures a full
forward+backward step (the frozen backbone contributes no graph, so this is
essentially the adapter plus the head) and reports peak memory per density cap.

Design notes
------------
* Deliberately a **separate script**: ``tools/run_p1_a3_and_seeds.sh``
  re-launches ``train.py`` for each remaining seed, so editing ``train.py``
  would risk the running chain.
* **Both axes are paired within one process.**  Caps were already cycled within
  each batch because the shared GPU's drift is larger than the effects being
  measured (P4-6 later measured the same A0 configuration at 433 ms and 99.8 ms
  in two groups).  ``--epochs`` extends the same discipline to the horizon
  curriculum: previously the two horizon points came from two separate
  invocations ~2 minutes apart, which is exactly the comparison this environment
  cannot make reliably -- and the resulting 5.4x was indeed inflated.  The paired
  run (``--epochs 0 8``) measured 2.15x, and it also showed the horizon count at
  epoch 8 is 4, not 6.  Interleave them instead of invoking twice.

* Caps are cycled **within each batch** (paired design), because the GPU is
  shared with the seed training and any drift would otherwise be attributed to
  the cap.
* The optimizer step is skipped: with only ~180k trainable adapter parameters
  its cost is negligible, and skipping it keeps the weights identical across
  caps so the comparison is exact.

    python3.10 benchmarks/p3/profile_training.py \
      --config config/opus/ablations/p1_a5_gaussian.py \
      --checkpoint out/p1_a5_gaussian/epoch_40.pth \
      --caps 0 32 --batch-size 1 --iters 20
"""
from __future__ import annotations

import argparse
import json
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


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def build(config_path, checkpoint, batch_size, num_workers, strict=True):
    from mmengine.config import Config
    from mmengine.optim import build_optim_wrapper
    from mmseg.models import build_segmentor
    import model  # noqa: F401
    from loss import OPENOCC_LOSS
    from dataset import get_dataloader

    config = Config.fromfile(str(ROOT / config_path))
    segmentor = build_segmentor(config.model)
    segmentor.init_weights()
    if checkpoint:
        state = torch.load(str(ROOT / checkpoint), map_location='cpu')
        state = state.get('state_dict', state)
        report = segmentor.load_state_dict(state, strict=strict)
        if not strict:
            print(f'lenient load: missing={list(report.missing_keys)[:8]} '
                  f'unexpected={list(report.unexpected_keys)[:8]}')
    segmentor = segmentor.cuda()
    segmentor.train()

    optimizer = build_optim_wrapper(segmentor, config.optimizer)
    loss_func = OPENOCC_LOSS.build(config.loss).cuda()

    config.train_loader = dict(config.train_loader, batch_size=batch_size,
                               num_workers=num_workers, shuffle=True)
    train_loader, _ = get_dataloader(
        config.train_dataset_config, config.val_dataset_config,
        config.train_loader, config.val_loader, dist=False,
        seed=config.get('dataloader_seed', None))
    return config, segmentor, optimizer, loss_func, train_loader


def set_cap(model, cap):
    if cap is None:
        return
    adapter = getattr(model, 'semantic_gaussian', None)
    if adapter is None:
        adapter = getattr(getattr(model, 'head', None), 'gaussian', None)
    if adapter is None or not hasattr(adapter, 'max_neighbours'):
        raise AttributeError('configured model does not expose a CUDA Gaussian density cap')
    adapter.max_neighbours = int(cap)


def training_step(model, loss_func, optimizer, config, data, global_iter):
    moved = {}
    for key, value in data.items():
        moved[key] = value.cuda() if isinstance(value, torch.Tensor) else value
    images = moved.pop('img')
    optimizer.zero_grad()
    result = model(imgs=images, metas=moved, global_iter=global_iter)
    loss_input = {'metas': moved, 'global_iter': global_iter}
    for key, value in config.loss_input_convertion.items():
        loss_input[key] = result[value]
    loss, _ = loss_func(loss_input)
    loss.backward()
    return loss


def gradient_is_alive(model):
    """True when at least one trainable tensor received a non-zero gradient.

    A silent early-out in the loss would leave every gradient at zero, and the
    subsequent ``backward`` would be near-free -- the peak memory would then
    describe a forward pass, not a training step.  Every probe must confirm
    this before its numbers are reported.
    """
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if float(parameter.grad.abs().sum()) > 0:
                return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--caps', nargs='*', type=int, default=[0])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--set-epoch', type=int, default=None,
                        help='drive the forecast-horizon curriculum. Training '
                             'uses max(1, min(epoch - finetune_epoch + 1, '
                             'num_fu_frames)) horizons, so epoch 0 measures 1 and '
                             'epoch 8 measures 4 (not 6); pass a large value for '
                             'steady state.')
    parser.add_argument('--epochs', nargs='*', type=int, default=None,
                        help='interleave several set_epoch values in ONE process '
                             '(paired design) instead of one invocation per '
                             'horizon. Use e.g. --epochs 0 8.')
    parser.add_argument('--json', default=None)
    parser.add_argument('--no-strict', action='store_true')
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available')

    config, model, optimizer, loss_func, loader = build(
        args.config, args.checkpoint, args.batch_size, args.num_workers,
        strict=not args.no_strict)
    if args.set_epoch is not None and hasattr(model, 'set_epoch'):
        model.set_epoch(args.set_epoch)
        print(f'set_epoch({args.set_epoch}) -> pretrain={getattr(model, "pretrain", None)}')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({'config': args.config, 'checkpoint': args.checkpoint,
                      'batch_size': args.batch_size,
                      'trainable_params': trainable,
                      'device': torch.cuda.get_device_name(0)}, indent=1))

    batches = []
    for data in loader:
        if len(batches) >= args.warmup + args.iters:
            break
        batches.append(data)
    if not batches:
        raise SystemExit('no training batches')

    # The paired design: every (epoch, cap) combination is measured inside one
    # process, cycling combinations within each batch, so allocator state and
    # GPU contention hit all settings alike.
    epochs = args.epochs if args.epochs else [args.set_epoch]
    settings = [(epoch, cap) for epoch in epochs for cap in args.caps]

    def apply(epoch, cap):
        if epoch is not None and hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        set_cap(model, cap)

    for index, data in enumerate(batches[:args.warmup]):
        for epoch, cap in settings:
            apply(epoch, cap)
            training_step(model, loss_func, optimizer, config, data, index)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = {setting: [] for setting in settings}
    peaks = {setting: 0.0 for setting in settings}
    for index, data in enumerate(batches[args.warmup:], start=args.warmup):
        for epoch, cap in settings:
            apply(epoch, cap)
            torch.cuda.reset_peak_memory_stats()
            start.record()
            training_step(model, loss_func, optimizer, config, data, index)
            end.record()
            torch.cuda.synchronize()
            times[(epoch, cap)].append(start.elapsed_time(end))
            peaks[(epoch, cap)] = max(
                peaks[(epoch, cap)], torch.cuda.max_memory_allocated() / 2 ** 20)

    alive = gradient_is_alive(model)
    print(f'gradients non-zero after a probe step: {alive}')
    if not alive:
        raise SystemExit('probe step produced no gradient; refusing to report '
                         'memory numbers for a step that did not train')

    rows = []
    for epoch, cap in settings:
        rows.append({
            'set_epoch': epoch,
            'max_neighbours': cap,
            'steps': len(times[(epoch, cap)]),
            'p50_ms': round(percentile(times[(epoch, cap)], 0.50), 3),
            'p95_ms': round(percentile(times[(epoch, cap)], 0.95), 3),
            'peak_alloc_mib': round(peaks[(epoch, cap)], 2),
        })
        print(f"set_epoch={str(epoch):>4} cap={str(cap):>6}  "
              f"p50={rows[-1]['p50_ms']:9.3f} ms  "
              f"p95={rows[-1]['p95_ms']:9.3f} ms  "
              f"peak={rows[-1]['peak_alloc_mib']:9.2f} MiB", flush=True)

    # When several epochs are interleaved, report the ratio directly: that is
    # the quantity the report quotes, and computing it inside one process is the
    # entire point of --epochs.
    if len(epochs) > 1:
        for cap in args.caps:
            base = next((r for r in rows
                         if r['set_epoch'] == epochs[0] and r['max_neighbours'] == cap),
                        None)
            if base is None or base['p50_ms'] <= 0:
                continue
            for row in rows:
                if row['max_neighbours'] != cap or row['set_epoch'] == epochs[0]:
                    continue
                print(f"  cap={cap}: epoch {epochs[0]} -> {row['set_epoch']} "
                      f"= {row['p50_ms'] / base['p50_ms']:.2f}x "
                      f"(paired, single process)")

    if args.json:
        target = ROOT / args.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(
            {'config': args.config, 'checkpoint': args.checkpoint,
             'batch_size': args.batch_size, 'trainable_params': trainable,
             'epochs': epochs, 'interleaved': bool(args.epochs),
             'rows': rows}, indent=1), encoding='utf-8')
        print(f'wrote {_display(target)}')


if __name__ == '__main__':
    main()
