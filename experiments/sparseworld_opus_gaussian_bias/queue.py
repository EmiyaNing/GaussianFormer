"""Durable sequential execution, independent best rechecks, then final report."""
import json
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from .train import atomic_json


def main():
    root=Path('out/sparseworld_opus_gaussian_bias_4096_1024_15e')
    analysis=Path('analysis/sparseworld_opus_gaussian_bias_4096_1024_15e')
    root.mkdir(parents=True,exist_ok=True)
    analysis.mkdir(parents=True,exist_ok=True)
    lock = open(root / 'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another experiment queue is already running')
    lock.write(str(os.getpid()))
    lock.flush()
    env=dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_bias_mpl',
             OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    variants=('sparseworld_strict','sparseworld_opus_gaussian_bias')
    for variant in variants:
        work=root/variant
        work.mkdir(exist_ok=True)
        config=f'config/sparseworld_opus_gaussian_bias/{variant}_4096_1024_15e.py'
        for phase in ('train','best_recheck'):
            if phase=='train' and (work/'completed.json').exists():
                continue
            command=[sys.executable,'-m','experiments.sparseworld_opus_gaussian_bias.train',
                     '--config',config,'--work-dir',str(work)]
            if phase=='best_recheck':
                command.append('--evaluate-best')
            atomic_json(analysis/'status.json',dict(status='running',model=variant,phase=phase,command=command))
            print(f'Starting {variant} {phase}',flush=True)
            with open(work/f'{phase}.log','a',buffering=1) as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            if process.returncode:
                atomic_json(analysis/'status.json',dict(status='failed',model=variant,
                    phase=phase,returncode=process.returncode,log=str(work/f'{phase}.log')))
                raise SystemExit(process.returncode)
    command=[sys.executable,'-m','experiments.sparseworld_opus_gaussian_bias.report']
    atomic_json(analysis/'status.json',dict(status='running',phase='final_report',command=command))
    with open(root/'report.log','a',buffering=1) as log:
        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    if process.returncode:
        atomic_json(analysis/'status.json',dict(status='failed',phase='final_report',
            returncode=process.returncode,log=str(root/'report.log')))
        raise SystemExit(process.returncode)


if __name__=='__main__':
    main()
