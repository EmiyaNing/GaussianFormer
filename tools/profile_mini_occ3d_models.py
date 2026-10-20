"""Replay mini Occ3D model steps to measure memory and batch-one latency.

Run one model and one phase per process so optimizer state and CUDA allocator
state cannot leak from training into validation or inference measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mmengine import Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

import model  # noqa: F401  register models
from dataset import get_dataloader
from loss import OPENOCC_LOSS


MODELS = (
    'opusv1', 'opusv2', 'opusv1_gaussian_sparse_render', 'gaussianformer',
    'opusv1_gaussian_margin', 'opusv1_gaussian_softplus',
    'opusv1_gaussian_margin_softplus',
    'opusv1_gaussian_softplus_camera_mask',
    'opusv1_gaussian_softplus_learned_bias',
    'opusv1_gaussian_softplus_class_margin',
)
BEST_EPOCHS = (5,) * len(MODELS)
MIB = 1024 ** 2


def prepare_batch(batch):
    metas = {key: value.cuda(non_blocking=False) if isinstance(value, torch.Tensor)
             else value for key, value in batch.items() if key != 'img'}
    imgs = batch['img'].cuda(non_blocking=False)
    return imgs, metas


def make_loss_input(cfg, result, metas, global_iter):
    inputs = {'metas': metas, 'global_iter': global_iter}
    for name, result_key in cfg.loss_input_convertion.items():
        inputs[name] = result[result_key]
    return inputs


def memory_snapshot():
    torch.cuda.synchronize()
    return {
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / MIB,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / MIB,
        'current_allocated_mib': torch.cuda.memory_allocated() / MIB,
        'current_reserved_mib': torch.cuda.memory_reserved() / MIB,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Run sampled profiles for all models')
    parser.add_argument('--full-split', action='store_true',
                        help='Replay all 4096 train and 1024 validation frames for every model')
    parser.add_argument('--full-infer', action='store_true',
                        help='Measure inference on all 1024 validation frames for every model')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--phase', choices=('train', 'val', 'infer'))
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--no-cudnn-benchmark', action='store_true',
                        help='Disable cuDNN autotuning for a diagnostic control')
    parser.add_argument('--out')
    parser.add_argument('--py-config', help='Override config for full-data diagnostics')
    parser.add_argument('--checkpoint', help='Override checkpoint for diagnostics')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Use configured pretraining and a fresh optimizer for a new-budget smoke check')
    parser.add_argument('--trace-memory', action='store_true',
                        help='Record cumulative peaks at model/loss/backward boundaries')
    parser.add_argument('--retain-step-outputs', action='store_true',
                        help='Keep previous output/loss references as train.py does')
    args = parser.parse_args()
    if args.all or args.full_split or args.full_infer:
        root = ROOT / 'work_dirs/mini_occ3d_4096_1024/hardware_profile'
        root.mkdir(parents=True, exist_ok=True)
        status_name = ('full_status.txt' if args.full_split else
                       'full_infer_status.txt' if args.full_infer else 'status.txt')
        status = root / status_name
        phases = ({'train': (0, 4096), 'val': (0, 1024)} if args.full_split else
                  {'infer': (0, 1024)} if args.full_infer else
                  {'train': (8, 128), 'val': (8, 128), 'infer': (16, 64)})
        for name in MODELS:
            for phase, (warmup, samples) in phases.items():
                basename = f'{"full_" if args.full_split or args.full_infer else ""}{name}_{phase}'
                output_path = root / f'{basename}.json'
                if output_path.is_file():
                    existing = json.loads(output_path.read_text())
                    if not args.full_infer or 'peak_sample_index' in existing:
                        print(f'skip complete {name} {phase}', flush=True)
                        continue
                status.write_text(f'running {name} {phase}\n')
                command = [sys.executable, str(Path(__file__).resolve()),
                           '--model', name, '--phase', phase,
                           '--warmup', str(warmup), '--samples', str(samples),
                           '--out', str(output_path)]
                with (root / f'{basename}.log').open('w') as log:
                    completed = subprocess.run(command, cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    status.write_text(f'failed {name} {phase}: exit {completed.returncode}\n')
                    raise RuntimeError(f'{name} {phase} failed; inspect its log')
                print(f'complete {name} {phase}', flush=True)
        status.write_text('complete\n')
        return
    if args.model is None or args.phase is None or args.out is None:
        parser.error('--model, --phase and --out are required unless --all is set')
    if args.samples < 1 or args.warmup < 0:
        parser.error('--samples must be positive and --warmup nonnegative')
    if args.phase == 'train' and (args.samples % 4 or args.warmup % 4):
        parser.error('training sample and warmup counts must be divisible by four')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.cuda.set_device(0)
    config_path = ROOT / 'config/mini_occ3d_4096_1024' / f'{args.model}.py'
    if args.py_config:
        config_path = Path(args.py_config)
    cfg = Config.fromfile(str(config_path))
    set_random_seed(928)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
    os.environ['amp'] = 'true' if cfg.get('amp', False) else 'false'
    os.environ['eval'] = 'false' if args.phase == 'train' else 'true'

    # The original runs used batch one and accumulation four. Workers affect
    # loading time, which is deliberately outside the measured GPU step.
    train_loader_cfg = dict(cfg.train_loader)
    val_loader_cfg = dict(cfg.val_loader)
    train_loader_cfg['num_workers'] = 0
    val_loader_cfg['num_workers'] = 0
    train_loader, val_loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        train_loader_cfg, val_loader_cfg,
        val_only=args.phase != 'train',
        seed=cfg.get('dataloader_seed'))
    loader = train_loader if args.phase == 'train' else val_loader
    model_instance = build_segmentor(cfg.model)
    model_instance.init_weights()
    model_instance.cuda()
    checkpoint_path = ROOT / 'work_dirs/mini_occ3d_4096_1024' / args.model / (
        f'epoch_{BEST_EPOCHS[MODELS.index(args.model)]}.pth')
    if args.fresh_start:
        checkpoint_path = Path(cfg.get('train_load_from', cfg.load_from)).resolve()
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if args.fresh_start:
        state = checkpoint.get('state_dict', checkpoint)
        try:
            model_instance.load_state_dict(state, strict=False)
        except RuntimeError:
            from misc.checkpoint_util import refine_load_from_sd
            model_instance.load_state_dict(refine_load_from_sd(state), strict=False)
        global_iter = 0
    else:
        model_instance.load_state_dict(checkpoint['state_dict'], strict=True)
        global_iter = int(checkpoint.get('global_iter', 0))
    optimizer = None
    if args.phase == 'train':
        optimizer = build_optim_wrapper(model_instance, cfg.optimizer)
        if not args.fresh_start:
            optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint
    loss_module = (OPENOCC_LOSS.build(cfg.loss).cuda()
                   if args.phase in ('train', 'val') else None)
    amp = cfg.get('amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    model_instance.train(args.phase == 'train')
    if cfg.get('enable_model_set_epoch', False) and args.phase == 'train':
        model_instance.set_epoch(BEST_EPOCHS[MODELS.index(args.model)] - 1)

    total = args.warmup + args.samples
    trace = []
    active_index = -1
    def record_stage(stage, module=None, inputs=None, output=None):
        entry = {'sample': active_index, 'stage': stage, **memory_snapshot()}
        if stage == 'aggregator' and isinstance(output, torch.Tensor):
            node = output.grad_fn
            if node is not None and hasattr(node, 'num_rendered'):
                entry['gaussian_voxel_pairs'] = node.num_rendered
                entry['saved_tensor_mib'] = [
                    t.numel() * t.element_size() / MIB for t in node.saved_tensors]
        trace.append(entry)
    if args.trace_memory:
        for name in ('img_backbone', 'img_neck', 'encoder', 'head'):
            module = getattr(model_instance, name, None)
            if module is not None:
                module.register_forward_hook(
                    lambda m, i, o, name=name: record_stage(name, m, i, o))
        aggregator = getattr(model_instance.head, 'aggregator', None)
        if aggregator is not None:
            aggregator.register_forward_hook(
                lambda m, i, o: record_stage('aggregator', m, i, o))
    timings_ms = []
    baseline = None
    highest_allocated = 0
    peak_sample_index = None
    # Model initialization consumes a model-dependent number of RNG draws.
    # Reset before sampling so the benchmark training window is identical.
    set_random_seed(928)
    for index, batch in enumerate(loader):
        if index >= total:
            break
        imgs, metas = prepare_batch(batch)
        active_index = index
        if index == args.warmup:
            torch.cuda.synchronize()
            baseline = {
                'allocated_mib': torch.cuda.memory_allocated() / MIB,
                'reserved_mib': torch.cuda.memory_reserved() / MIB,
            }
            torch.cuda.reset_peak_memory_stats()
        if args.phase == 'train':
            with torch.cuda.amp.autocast(enabled=amp):
                result = model_instance(imgs=imgs, metas=metas,
                                        global_iter=global_iter)
                loss_inputs = make_loss_input(cfg, result, metas, global_iter)
                loss, _ = loss_module(loss_inputs)
                loss = loss / 4
            if args.trace_memory:
                record_stage('loss')
            if amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if args.trace_memory:
                record_stage('backward')
            if (index + 1) % 4 == 0:
                if amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(),
                                               cfg.grad_max_norm)
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            if args.trace_memory:
                record_stage('step_end')
            global_iter += 1
        elif args.phase == 'val':
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                result = model_instance(imgs=imgs, metas=metas)
                loss, _ = loss_module(make_loss_input(cfg, result, metas, global_iter))
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                result = model_instance(imgs=imgs, metas=metas)
            torch.cuda.synchronize()
            if index >= args.warmup:
                timings_ms.append((time.perf_counter() - start) * 1000)
        if index >= args.warmup:
            allocated_peak = torch.cuda.max_memory_allocated()
            if allocated_peak > highest_allocated:
                highest_allocated = allocated_peak
                peak_sample_index = index - args.warmup
        # Drop references so the next sample tests the normal steady state.
        del imgs, metas, batch
        if not args.retain_step_outputs:
            del result
            if args.phase != 'infer':
                del loss
            if args.phase == 'train':
                del loss_inputs
        if (index + 1) % 8 == 0:
            print(f'{args.model} {args.phase} {index + 1}/{total}', flush=True)

    if index + 1 < total:
        raise RuntimeError(f'Loader supplied only {index + 1}/{total} samples')
    measured = memory_snapshot()
    output = {
        'model': args.model, 'phase': args.phase,
        'config': str(config_path),
        'checkpoint': str(checkpoint_path.relative_to(ROOT)),
        'device': torch.cuda.get_device_name(0),
        'warmup_samples': args.warmup,
        'measured_samples': args.samples,
        'batch_size': 1,
        'gradient_accumulation': 4 if args.phase == 'train' else None,
        'amp': bool(amp),
        'cudnn_benchmark': not args.no_cudnn_benchmark,
        'measurement': measured,
        'baseline_after_warmup': baseline,
        'peak_sample_index': peak_sample_index,
        'retain_step_outputs': args.retain_step_outputs,
    }
    if args.trace_memory:
        output['memory_trace'] = trace
    if timings_ms:
        ordered = sorted(timings_ms)
        output['latency_ms'] = {
            'median': statistics.median(ordered),
            'mean': statistics.mean(ordered),
            'p95': ordered[min(len(ordered)-1, int(0.95*len(ordered)))],
            'min': ordered[0], 'max': ordered[-1],
            'raw': timings_ms,
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ('latency_ms', 'memory_trace')}), flush=True)


if __name__ == '__main__':
    main()
