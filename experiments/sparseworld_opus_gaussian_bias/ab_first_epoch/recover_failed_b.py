"""Retry the previously authorized B startup without advancing completed A."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


def main():
    root = Path('out/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e')
    analysis = Path('analysis/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e')
    read = lambda path: json.loads(path.read_text())
    failure = read(analysis / 'status.json')
    if failure.get('status') != 'failed' or failure.get('arm') != 'B_ce_free_0_1':
        raise ValueError('recovery requires a recorded B failure')
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a_boundary = read(root / 'A_softmax_focal' / 'first_epoch_completed.json')
    if a_boundary['trained_epochs'] != 1 or a_boundary['planned_epochs'] != 15:
        raise ValueError('completed A must remain paused at epoch one')
    b_work = root / 'B_ce_free_0_1'
    if set(p.name for p in b_work.iterdir()) != {'train.log'}:
        raise ValueError('this recovery is only for failure before B initialization')
    audit = read(analysis / 'configuration_audit.json')
    for arm in audit['arms'].values():
        if hashlib.sha256(Path(arm['config']).read_bytes()).hexdigest() != arm['config_sha256']:
            raise ValueError('validated standalone configuration changed')
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    stamp = now.strftime('%Y%m%d_%H%M%S')
    saved_failure = analysis / f'B_startup_failure_{stamp}.json'
    saved_failure.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(b_work / 'train.log', analysis / f'B_startup_failure_{stamp}.log')
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()),
               MPLCONFIGDIR='/tmp/sparseworld_ab_mpl', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.queue']
    fcntl.flock(lock, fcntl.LOCK_UN)
    with (root / 'queue.log').open('a', buffering=1) as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=Path.cwd(), start_new_session=True)
    (root / 'queue.pid').write_text(str(process.pid) + '\n')
    record = dict(status='recovery_launched', started_at=now.isoformat(), queue_pid=process.pid,
                  command=command, preserved_failure=str(saved_failure),
                  failure_returncode=failure['returncode'],
                  cause='undetermined; SIGSEGV before B initialization and before logger output',
                  changes='Enable Python faulthandler and unbuffered logs; no training changes.',
                  A_action='skip completed first epoch; remain paused',
                  B_action='retry from identical validated initialization; stop after epoch one',
                  planned_epochs=15, total_updates=15360, diagnostic_epoch=1)
    (analysis / f'B_recovery_{stamp}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
