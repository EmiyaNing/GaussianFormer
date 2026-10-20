"""Keep the local progress document current independently of the chat turn."""
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/empty_mechanism_history'


def main():
    with (RUN / 'progress_monitor.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        previous = None
        while True:
            result = subprocess.run([sys.executable, 'tools/empty_mechanism_progress.py'],
                cwd=ROOT, capture_output=True, text=True, check=True)
            state = json.loads(result.stdout)
            status = 'cancelled_by_user' if state.get('cancelled') else 'failed' if state['failed'] else 'complete' if state['complete'] else 'running'
            (RUN / 'progress_monitor_status.json').write_text(json.dumps(dict(
                status=status, at=datetime.now().astimezone().isoformat(),
                message=state['message']), ensure_ascii=False, indent=2) + '\n')
            if state['message'] != previous:
                print(state['message'], flush=True)
                previous = state['message']
            if status != 'running':
                break
            time.sleep(45)


if __name__ == '__main__':
    main()
