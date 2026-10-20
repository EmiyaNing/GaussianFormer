"""Bounded memory diagnostic using the real train.py loop and initialization.

The diagnostic writes only its own work directory. A per-process allocator
limit prevents this probe from consuming all memory beside an ongoing job.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train


class ProbeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--memory-fraction', type=float, default=0.18)
    args = parser.parse_args()
    if args.steps < 4 or args.steps % 4:
        parser.error('--steps must be a positive multiple of four')
    run = Path(args.work_dir)
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'memory.jsonl').exists() or (run / 'latest.pth').exists():
        raise RuntimeError('Use a fresh diagnostic work directory')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    state = {'iteration': -1, 'optimizer_steps': 0}
    log = (run / 'memory.jsonl').open('w', buffering=1)

    def record(stage, **extra):
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        row = dict(state, stage=stage,
                   benchmark=torch.backends.cudnn.benchmark,
                   allocated_mib=torch.cuda.memory_allocated() / 2**20,
                   reserved_mib=torch.cuda.memory_reserved() / 2**20,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                   inactive_split_mib=stats.get('inactive_split_bytes.all.current', 0) / 2**20,
                   num_alloc_retries=stats.get('num_alloc_retries', 0), **extra)
        log.write(json.dumps(row) + '\n')

    original_build = train.build_segmentor
    def build(*a, **kw):
        network = original_build(*a, **kw)
        def before(*unused):
            state['iteration'] += 1
            if state['iteration'] % 32 == 0:
                record('before_forward')
        network.register_forward_pre_hook(before)
        def after(*unused):
            if state['iteration'] % 32 == 0:
                record('after_forward')
        network.register_forward_hook(after)
        def aggregator_after(module, inputs, output):
            if state['iteration'] % 32 == 0:
                node = output.grad_fn
                record('aggregator', pairs=node.num_rendered,
                       binning_mib=node.saved_tensors[1].numel() / 2**20)
        network.head.aggregator.register_forward_hook(aggregator_after)
        return network
    train.build_segmentor = build

    original_optim = train.build_optim_wrapper
    def build_optim(*a, **kw):
        optimizer = original_optim(*a, **kw)
        original_zero = optimizer.zero_grad
        def zero_grad(*a, **kw):
            result = original_zero(*a, **kw)
            state['optimizer_steps'] += 1
            record('after_optimizer')
            if state['optimizer_steps'] * 4 >= args.steps:
                raise ProbeComplete()
            return result
        optimizer.zero_grad = zero_grad
        return optimizer
    train.build_optim_wrapper = build_optim
    status = 'failed'
    try:
        train.main(0, SimpleNamespace(
            py_config=args.py_config, work_dir=str(run), seed=928,
            gpus=1, gradient_accumulation=4, resume_from='',
            iter_resume=False, dataset='nuscenes'))
    except ProbeComplete:
        record('probe_complete')
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        record('after_empty_cache', released_mib=(before - torch.cuda.memory_reserved()) / 2**20)
        status = 'complete'
    except torch.cuda.OutOfMemoryError:
        record('probe_memory_limit')
        status = 'memory_limit_reached'
        raise
    finally:
        (run / 'status.json').write_text(json.dumps(dict(state, status=status,
            requested_steps=args.steps, memory_fraction=args.memory_fraction), indent=2) + '\n')
        log.close()


if __name__ == '__main__':
    main()
