"""Launch the verified A/B queue into a fresh durable background session."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from ..train import atomic_json, digest
from .protocol import ANALYSIS, ARMS, ROOT
from .report import read


def main():
    smoke = read(ROOT / 'smoke' / 'all_passed.json')
    if smoke['status'] != 'passed' or len(set(smoke['initial_model_sha256'].values())) != 1:
        raise ValueError('A/B GPU checks and identical initialization must pass before launch')
    audit = read(ANALYSIS / 'configuration_audit.json')
    for arm, config in ARMS.items():
        if (ROOT / arm).exists():
            raise ValueError('first launch must use fresh formal training directories')
        if digest(config) != audit['arms'][arm]['config_sha256']:
            raise ValueError('config changed after validation')
    paused = read(Path('analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/paused_for_ab_first_epoch.json'))
    for target in paused['signaled']:
        proc = Path('/proc') / str(target['pid'])
        if proc.exists() and proc.joinpath('cmdline').read_bytes():
            raise ValueError('old full-space processes have not fully exited')
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_ab_mpl',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.queue']
    with (ROOT / 'queue.log').open('a', buffering=1) as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=str(Path.cwd()), start_new_session=True)
    (ROOT / 'queue.pid').write_text(str(process.pid) + '\n')
    record = dict(status='launched', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  queue_pid=process.pid, command=command, model_initialization='from_scratch',
                  planned_epochs_per_arm=15, total_updates_per_arm=15360,
                  diagnostic_epoch=1, runtime_stop_after_epoch=1,
                  train_samples=4096, validation_samples=1024,
                  sequential_arms=list(ARMS), configs=audit['arms'],
                  validated_initial_model_sha256=next(iter(smoke['initial_model_sha256'].values())))
    atomic_json(ANALYSIS / 'launch.json', record)
    print(f'A/B queue launched: PID {process.pid}', flush=True)


if __name__ == '__main__':
    main()
