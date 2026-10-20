"""Unchanged train.py execution with per-process time and CUDA peak accounting."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools.center_support_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--seed', type=int, default=928)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for training')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    previous_argv = sys.argv
    sys.argv = [str(ROOT / 'train.py'), '--py-config', args.py_config,
                '--work-dir', args.work_dir, '--seed', str(args.seed),
                '--gradient-accumulation', str(args.gradient_accumulation)]
    status, error = 'failed', None
    try:
        runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
        status = 'complete'
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        sys.argv = previous_argv
        write_json(Path(args.work_dir) / 'training_runtime.json', dict(status=status,
            protocol=vars(args), training_and_validation_seconds=time.monotonic() - started,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            device=torch.cuda.get_device_name(), error=error,
            accounting='This training process, including its validation; no renderer/optimizer changes'))


if __name__ == '__main__':
    main()
