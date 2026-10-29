"""Persistent queue: run all Tiny attribute diagnostics, report, then audit.

May be launched detached. Does not restart a failed or interrupted child.
"""
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
RUN=ROOT/'work_dirs/gaussian_attribute_oracle_full_val_v1'
child=None


def save(value):
    path=RUN/'queue_status.json'
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    tmp.replace(path)


def stop(sig,frame):
    if child is not None and child.poll() is None:
        child.terminate()
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill();child.wait()
    save(dict(status='stopped',signal=sig,pid=os.getpid()))
    raise SystemExit(128+sig)


def execute(script,args,log,phase):
    global child
    with log.open('a') as stream:
        child=subprocess.Popen([sys.executable,str(ROOT/'tools'/script),*args],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        last=0
        while child.poll() is None:
            if time.monotonic()-last>=30:
                progress=RUN/'progress.json'
                save(dict(status='running',phase=phase,pid=os.getpid(),child_pid=child.pid,
                    progress=json.loads(progress.read_text()) if progress.exists() else None))
                last=time.monotonic()
            time.sleep(2)
        code=child.returncode
        child=None
        if code: raise RuntimeError(f'{phase} failed with exit code {code}; see {log}')


def audit():
    value=json.loads((RUN/'delivery_audit.json').read_text())
    assert value['status']=='passed'
    for name,digest in value['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name in ('gaussian_attribute_oracle_full_val_design.md','gaussian_attribute_oracle_full_val_results.md'):
        p=ROOT/'analysis'/name;text=p.read_text()
        assert text.count('```')%2==0
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            assert (p.parent/target).resolve().exists(),target
    for name,count in [('metrics.csv',61),('per_class.csv',1020),('per_class_differences.csv',765),
        ('per_scene.csv',9000),('attribute_changes.csv',60),('loss_trajectory.csv',315),
        ('conditional_gains.csv',32),('interactions.csv',24),('error_overlaps.csv',420),('shapley.csv',4)]:
        with (RUN/'aggregate'/name).open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==count,(name,len(rows),count)
    return len(value['artifact_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    start=time.monotonic()
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not (RUN/'pilot.json').exists():
                execute('eval_gaussian_attribute_oracle.py',['--action','pilot'],
                    RUN/'pilot_console.log','pilot')
            execute('eval_gaussian_attribute_oracle.py',['--action','run','--workers',str(args.workers)],
                ROOT/'work_dirs/gaussian_attribute_oracle_full_val_console.log','evaluation')
            execute('report_gaussian_attribute_oracle.py',[],RUN/'report_console.log','report')
            artifacts=audit()
            save(dict(status='complete',pid=os.getpid(),frames=6019,scenes=150,arms=15,
                seconds=time.monotonic()-start,verified_artifacts=artifacts,
                report='analysis/gaussian_attribute_oracle_full_val_results.md'))
            print('Experiment, report, links, CSV sizes and file hashes verified.',flush=True)
        except Exception as exc:
            save(dict(status='failed',pid=os.getpid(),error=repr(exc)))
            raise


if __name__=='__main__': main()
