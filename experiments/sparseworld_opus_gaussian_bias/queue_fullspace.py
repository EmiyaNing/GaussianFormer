"""Durable full-space restart, per-epoch diagnostics, best recheck and report."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from .train import atomic_json

ROOT = Path('out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e')
ANALYSIS = Path('analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e')
CONFIG = 'config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e.py'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / 'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another full-space queue is running')
    lock.write(str(os.getpid()))
    lock.flush()
    strict_source = Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e/sparseworld_strict').resolve()
    if not (strict_source / 'completed.json').exists() or not (strict_source / 'best_recheck').exists():
        raise ValueError('completed Strict baseline and independent best recheck are required')
    baseline = ROOT / 'sparseworld_strict'
    if not baseline.exists():
        baseline.symlink_to(strict_source, target_is_directory=True)
    elif baseline.resolve() != strict_source:
        raise ValueError('full-space baseline must refer to the original completed Strict run')
    work = ROOT / 'sparseworld_opus_gaussian_bias'
    work.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()),
               MPLCONFIGDIR='/tmp/sparseworld_fullspace_mpl',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for phase in ('train', 'best_recheck', 'final_report'):
        if phase == 'train' and (work / 'completed.json').exists():
            continue
        if phase == 'final_report':
            command = [sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.report_fullspace']
            log_path = ROOT / 'report.log'
        else:
            command = [sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.train_fullspace',
                       '--config', CONFIG, '--work-dir', str(work)]
            if phase == 'best_recheck':
                command.append('--evaluate-best')
            log_path = work / f'{phase}.log'
        atomic_json(ANALYSIS / 'status.json', dict(
            status='running', model='sparseworld_opus_gaussian_bias_fullspace',
            phase=phase, command=command, strict_baseline_reused=str(strict_source)))
        print(f'Starting full-space {phase}', flush=True)
        with log_path.open('a', buffering=1) as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if process.returncode:
            atomic_json(ANALYSIS / 'status.json', dict(
                status='failed', phase=phase, returncode=process.returncode, log=str(log_path)))
            raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
