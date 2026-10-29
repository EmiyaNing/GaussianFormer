"""Persistent semantic-only oracle queue: pilot, full validation, report, audit."""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/gaussian_semantic_oracle_full_val_v1'
child=None


def save(value):
    path=RUN/'queue_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def stop(sig,frame):
    if child is not None and child.poll() is None:
        child.terminate()
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    save(dict(status='stopped',signal=sig,pid=os.getpid()))
    raise SystemExit(128+sig)


def execute(script,args,log,phase):
    global child
    with log.open('a') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'tools'/script),*args],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        last=0
        while child.poll() is None:
            if time.monotonic()-last>=30:
                p=RUN/'progress.json'
                save(dict(status='running',phase=phase,pid=os.getpid(),child_pid=child.pid,
                    progress=json.loads(p.read_text()) if p.exists() else None))
                last=time.monotonic()
            time.sleep(2)
        code=child.returncode;child=None
        if code:raise RuntimeError(f'{phase} failed with exit code {code}; see {log}')


def audit():
    data=json.loads((RUN/'delivery_audit.json').read_text());assert data['status']=='passed'
    for path,digest in data['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for name in ('gaussian_semantic_oracle_full_val_design.md','gaussian_semantic_oracle_full_val_results.md'):
        p=ROOT/'analysis'/name;content=p.read_text();assert content.count('```')%2==0
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
            assert (p.parent/target).resolve().exists(),target
    for name,count in [('metrics.csv',4),('per_class.csv',68),('per_scene.csv',600),
        ('comparisons.csv',15),('semantic_changes.csv',4),('loss_trajectory.csv',21)]:
        with (RUN/'aggregate'/name).open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==count,(name,len(rows),count)
    return len(data['artifact_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);start=time.monotonic()
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (RUN/'pilot.json').exists():
                execute('eval_gaussian_semantic_oracle.py',['--action','pilot'],RUN/'pilot_console.log','pilot')
            execute('eval_gaussian_semantic_oracle.py',['--action','run','--workers',str(args.workers)],
                ROOT/'work_dirs/gaussian_semantic_oracle_full_val_console.log','evaluation')
            execute('report_gaussian_semantic_oracle.py',[],RUN/'report_console.log','report')
            count=audit()
            save(dict(status='complete',pid=os.getpid(),frames=6019,scenes=150,
                seconds=time.monotonic()-start,verified_artifacts=count,
                report='analysis/gaussian_semantic_oracle_full_val_results.md'))
            print('Semantic-only diagnostic, report and artifact checks passed.',flush=True)
        except Exception as exc:
            save(dict(status='failed',pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
