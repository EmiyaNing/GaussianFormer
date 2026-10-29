"""Wait for an already-running A1, then finish the registered experiment queue."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gaussian_responsibility_experiment import RUN, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-pid', type=int)
    args = parser.parse_args()
    while args.wait_for_pid and not (RUN / 'train/A1/complete.json').exists():
        try:
            os.kill(args.wait_for_pid, 0)
        except ProcessLookupError:
            raise RuntimeError('A1 process exited without a completion audit')
        write(RUN / 'queue_status.json', dict(status='waiting_for_running_A1',
            pid=args.wait_for_pid, at=datetime.now().astimezone().isoformat()))
        time.sleep(15)
    write(RUN / 'queue_status.json', dict(status='running_remaining_queue',
        at=datetime.now().astimezone().isoformat()))
    code = subprocess.call([sys.executable, 'tools/gaussian_responsibility_experiment.py', 'queue'], cwd=ROOT)
    write(RUN / 'queue_status.json', dict(status='complete' if code == 0 else 'failed',
        returncode=code, at=datetime.now().astimezone().isoformat()))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
