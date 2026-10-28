"""Run A epoch one, B epoch one, diagnostic comparison, optional continuation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from ..train import atomic_json
from .protocol import ANALYSIS, ARMS, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue-after-diagnosis', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / 'queue.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another A/B queue is running')
    lock.write(str(os.getpid()))
    lock.flush()
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_ab_mpl',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

    def run(command, log_path, **status):
        atomic_json(ANALYSIS / 'status.json', dict(status='running', command=command, **status))
        print(f'Starting {status}', flush=True)
        with log_path.open('a', buffering=1) as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        if process.returncode:
            atomic_json(ANALYSIS / 'status.json', dict(status='failed', returncode=process.returncode,
                                                     log=str(log_path), **status))
            raise SystemExit(process.returncode)

    for arm, config in ARMS.items():
        work = ROOT / arm
        work.mkdir(exist_ok=True)
        if (work / 'first_epoch_completed.json').exists():
            continue
        run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.train',
             '--config', str(config), '--work-dir', str(work), '--stop-after-epoch', '1'],
            work / 'train.log', phase='first_epoch_train_and_diagnose', arm=arm,
            planned_epochs=15, diagnostic_epoch=1)
    run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.report'],
        ROOT / 'report.log', phase='first_epoch_comparison', diagnostic_epoch=1)
    if not args.continue_after_diagnosis:
        atomic_json(ANALYSIS / 'status.json', dict(
            status='paused_after_first_epoch_diagnosis', planned_epochs_per_arm=15,
            completed_training_epochs_per_arm=1, diagnostic_epoch=1,
            report=str(ANALYSIS / 'report.md'), arms=list(ARMS)))
        print('Both arms paused after complete epoch-one diagnosis.', flush=True)
        return
    for arm, config in ARMS.items():
        work = ROOT / arm
        if not (work / 'completed.json').exists():
            run([sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.train',
                 '--config', str(config), '--work-dir', str(work)], work / 'continuation.log',
                phase='continue_training', arm=arm, planned_epochs=15, diagnostic_epoch=1)
    atomic_json(ANALYSIS / 'status.json', dict(status='training_completed', epochs_per_arm=15,
                diagnostic_epoch=1, report=str(ANALYSIS / 'report.md')))


if __name__ == '__main__':
    main()
