"""Durable endpoint evaluation/report queue; safe to resume after interruptions."""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/empty_mechanism_stage3'


def status(**kwargs):
    kwargs['at']=datetime.now().astimezone().isoformat()
    path=RUN/'completion_queue_status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(kwargs,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def command(args,log):
    print('RUN',' '.join(args),flush=True)
    with (RUN/log).open('a') as f:
        result=subprocess.run([sys.executable]+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        status(status='failed',command=args,log=log,exit_code=result.returncode)
        raise SystemExit(result.returncode)


def main():
    RUN.mkdir(exist_ok=True)
    with (RUN/'completion_queue.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('An existing completion queue owns this run.',flush=True);return
        pending=['B0','S','E','SE','original5','no_final_set5']
        while pending:
            ready=[]
            for name in pending:
                if name not in ['B0','S','E','SE'] or (RUN/'train'/name/'complete.json').exists():ready.append(name)
            for name in ready:
                status(status='evaluating',model=name,pending=pending)
                command(['tools/eval_empty_mechanism_stage3.py','extract','--model',name],f'eval_{name}.log')
                pending.remove(name)
            if pending:
                p=RUN/'queue_status.json'
                training=json.loads(p.read_text()) if p.exists() else {}
                if training.get('status')=='failed':
                    status(status='training_failed',training=training,pending=pending);raise SystemExit(1)
                status(status='waiting_for_training',pending=pending,training=training)
                time.sleep(15)
        for name in ['B0','S','E','SE','original5','no_final_set5']:
            status(status='statistics',model=name)
            command(['tools/eval_empty_mechanism_stage3.py','statistics','--model',name],f'statistics_{name}.log')
        status(status='endpoint_factor_swaps')
        command(['tools/mediate_empty_mechanism_stage3.py'],'mediation.log')
        status(status='reporting')
        command(['tools/report_empty_mechanism_stage3.py'],'report.log')
        status(status='auditing')
        command(['tools/audit_empty_mechanism_stage3.py'],'audit.log')
        status(status='complete',completed=['four complete final-epoch training runs','six full validation/calibration comparisons','paired-scene primary rescue analysis','exploratory output factor swaps','report and integrity audit'])


if __name__=='__main__':main()
