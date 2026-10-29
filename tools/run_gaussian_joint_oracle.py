"""Persistent runner for all three joint/ordered GT-assisted diagnostics."""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_gaussian_semantic_oracle as lifecycle

RUN=ROOT/'work_dirs/gaussian_joint_oracle_full_val_v1'
lifecycle.RUN=RUN


def audit():
    data=json.loads((RUN/'delivery_audit.json').read_text());assert data['status']=='passed'
    for p,digest in data['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    for name in ('gaussian_joint_oracle_full_val_design.md','gaussian_joint_oracle_full_val_results.md'):
        p=ROOT/'analysis'/name;content=p.read_text()
        assert content.count('```')%2==0 and content.count('$$')%2==0
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
            assert (p.parent/target).resolve().exists(),target
    for name,count in [('metrics.csv',21),('per_class.csv',357),('per_scene.csv',3150),
        ('comparisons.csv',21),('attribute_changes.csv',21),('loss_trajectory.csv',123),
        ('second_stage_gains.csv',3),('final_per_class.csv',119)]:
        with (RUN/'aggregate'/name).open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==count,(name,len(rows),count)
    return len(data['artifact_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    signal.signal(signal.SIGINT,lifecycle.stop);signal.signal(signal.SIGTERM,lifecycle.stop)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (RUN/'pilot.json').exists():
                lifecycle.execute('eval_gaussian_joint_oracle.py',['--action','pilot'],RUN/'pilot_console.log','pilot')
            lifecycle.execute('eval_gaussian_joint_oracle.py',['--action','run','--workers',str(args.workers)],
                ROOT/'work_dirs/gaussian_joint_oracle_full_val_console.log','evaluation')
            lifecycle.execute('report_gaussian_joint_oracle.py',[],RUN/'report_console.log','report')
            count=audit()
            lifecycle.save(dict(status='complete',pid=os.getpid(),frames=6019,scenes=150,arms=3,
                steps_per_arm=40,seconds=time.monotonic()-started,verified_artifacts=count,
                report='analysis/gaussian_joint_oracle_full_val_results.md'))
            print('All three schedules, full report and artifact audit passed.',flush=True)
        except Exception as exc:
            lifecycle.save(dict(status='failed',pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
