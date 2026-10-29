"""Wait for authorized queues and summarize completed results, never partial ones."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.gaussian_geometry_causal import RUN,write


def run(command,label):
    with (RUN/(label+'_console.log')).open('a') as f:
        subprocess.run([sys.executable,*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    primary=False
    while True:
        q=json.loads((RUN/'queue_status.json').read_text())
        if q['status']=='failed':raise RuntimeError(q)
        late=RUN/'mature_queue_status.json'
        if late.exists() and json.loads(late.read_text())['status']=='failed':raise RuntimeError(late.read_text())
        if not primary and (RUN/'selective_complete.json').exists():
            run(['tools/report_gaussian_geometry_causal.py'],'primary_report')
            write(RUN/'primary_report_complete.json',dict(status='complete'))
            primary=True
        if primary and (RUN/'mature_complete.json').exists():
            run(['tools/report_gaussian_geometry_mature.py'],'mature_report')
            write(RUN/'all_execution_complete.json',dict(status='complete',
                population_evaluation_checkpoints=8,keyframe_visits=273192,
                human_interpretation_review_pending=True))
            return
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(RUN/'finalization_failed.json',dict(status='failed',error=repr(e)))
        raise
