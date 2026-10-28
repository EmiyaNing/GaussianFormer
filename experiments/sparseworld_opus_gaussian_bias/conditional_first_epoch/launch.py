"""Launch the verified scratch C experiment as a durable background job."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch

from ..ab_first_epoch.report import read
from ..train import atomic_json, digest
from .protocol import ANALYSIS, BASE_ANALYSIS, BASE_CONFIG, BASE_WORK, CONFIG, ROOT, WORK


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if WORK.exists() or (ANALYSIS / 'launch.json').exists():
        raise ValueError('formal C launch requires a fresh directory and launch record')
    audit = read(ANALYSIS / 'configuration_audit.json')
    smoke = read(ROOT / 'smoke/all_passed.json')
    tests = read(ANALYSIS / 'unit_checks.json')
    if (tests['status'] != 'passed' or smoke['status'] != 'passed' or
            [r['epoch_index'] for r in smoke['checks']] != [0, 14] or
            smoke['config_sha256'] != audit['config_sha256'] or
            digest(CONFIG) != audit['config_sha256'] or digest(BASE_CONFIG) != audit['source_sha256'] or
            smoke['initial_model_sha256'] != read(BASE_WORK / 'initialization.json')['initial_model_sha256']):
        raise ValueError('configuration, scratch initialization and CPU/GPU checks must pass before launch')
    if read(BASE_ANALYSIS / 'status.json')['status'] != 'paused_after_first_epoch_diagnosis':
        raise ValueError('A/B queue is not at its preserved pause boundary')
    if not torch.cuda.is_available():
        raise ValueError('GPU unavailable in the launcher environment')
    compute_pids = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if compute_pids:
        raise ValueError('GPU compute jobs are already active; preserve them and do not start a competing experiment')
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_conditional_mpl',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch.queue']
    record = dict(status='prepared', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  command=command, config=str(CONFIG), config_sha256=audit['config_sha256'],
                  source_sha256={str(path): digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
                  planned_epochs=15, total_updates=15360, runtime_stop_after_epoch=1,
                  auxiliary_weight=1., free_relative_weight=.1, train_samples=4096, validation_samples=1024,
                  initialization='scratch world model, same image initialization and complete initial model as B',
                  validated_initial_model_sha256=smoke['initial_model_sha256'],
                  report=str(ANALYSIS / 'report.md'))
    # Queue reads this audit immediately; publish it before starting the child.
    atomic_json(ANALYSIS / 'launch.json', record)
    with (ROOT / 'queue.log').open('a', buffering=1) as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=Path.cwd(), start_new_session=True)
    (ROOT / 'queue.pid').write_text(str(process.pid) + '\n')
    record.update(status='launched', queue_pid=process.pid)
    atomic_json(ANALYSIS / 'launch.json', record)
    print(f'Conditional semantic CE experiment launched: queue PID {process.pid}', flush=True)


if __name__ == '__main__':
    main()
