"""Update the frozen-routing report as complete splits become available."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/small_query_budget_camera_full_1ep/routing_diagnosis'


def main():
    pid = json.loads((RUN / 'launch.json').read_text())['pid']
    last_update = 0.
    previous_val_complete = False
    while True:
        complete = all((RUN / (s + '_complete.json')).exists() for s in ('val', 'train'))
        val_complete = (RUN / 'val_complete.json').exists()
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        now = time.monotonic()
        if complete or not alive or val_complete != previous_val_complete or now - last_update >= 300:
            cmd = [sys.executable, 'tools/report_query_budget_routing.py']
            if not complete:
                cmd.append('--partial')
            result = subprocess.run(cmd, cwd=ROOT)
            state = dict(time=datetime.now().astimezone().isoformat(), report_exit_code=result.returncode,
                data_complete=complete, inference_process_alive=alive, validation_complete=val_complete)
            temp = RUN / 'report_watch_status.json.tmp'
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(RUN / 'report_watch_status.json')
            print(json.dumps(state), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
            last_update = now
        if complete or not alive:
            return
        previous_val_complete = val_complete
        time.sleep(30)


if __name__ == '__main__':
    main()
