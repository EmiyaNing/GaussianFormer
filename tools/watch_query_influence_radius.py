"""Refresh the radius report after completed phases and audit the final census."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/query_influence_radius_full'


def main():
    previous = None
    while True:
        if (RUN / 'failure.json').exists():
            raise RuntimeError((RUN / 'failure.json').read_text())
        finished = (RUN / 'complete.json').exists()
        phases = tuple(sorted(str(path.relative_to(RUN)) for path in RUN.glob('*/*_complete.json')))
        key = (finished, phases)
        if phases and key != previous:
            args = [sys.executable, str(ROOT / 'tools/report_query_influence_radius.py')]
            if not finished:
                args.append('--partial')
            print(datetime.now().astimezone().isoformat(), args, flush=True)
            subprocess.run(args, cwd=ROOT, check=True)
            subprocess.run(['node', '.chk.mjs', 'analysis/query_influence_radius_design.md',
                            'analysis/query_influence_radius_results.md'], cwd=ROOT, check=True)
            previous = key
            if finished:
                report = ROOT / 'analysis/query_influence_radius_results.md'
                (RUN / 'report_complete.json').write_text(json.dumps(dict(status='complete',
                    time=datetime.now().astimezone().isoformat(),
                    report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                    phases=list(phases)), indent=2) + '\n')
                return
        time.sleep(10)


if __name__ == '__main__':
    main()
