"""Wait for the authorized uncapped probe, then start fresh full Occ3D training."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
root = ROOT / 'work_dirs/full_occ3d_10e_no_benchmark'
probe_pid = int(sys.argv[1])
(root / 'restart_queue.pid').write_text(str(os.getpid()) + '\n')
while True:
    stat = Path(f'/proc/{probe_pid}/stat')
    if not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z '):
        break
    time.sleep(2)
status_path = root / 'memory_probe_native_uncapped_1024/status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {}
if status.get('status') != 'complete':
    (root / 'status.txt').write_text('uncapped diagnosis failed; inspect probe logs before restarting\n')
    raise RuntimeError(f'Uncapped diagnostic did not complete: {status}')
os.execvp('bash', ['bash', 'tools/run_full_occ3d_learned_bias_10e_expandable.sh'])
