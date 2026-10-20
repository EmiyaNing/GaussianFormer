"""Run the full-history factorial after late-stage training, then finish reports."""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/empty_mechanism_history'


def status(**values):
    values['at']=datetime.now().astimezone().isoformat()
    p=RUN/'queue_status.json';temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def execute(args,log):
    status(status='running',command=args)
    with (RUN/log).open('a') as f:
        command=args if args[0]=='node' else [sys.executable]+args
        code=subprocess.call(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if code:
        status(status='failed',command=args,exit_code=code,log=log);raise SystemExit(code)


def main():
    RUN.mkdir(exist_ok=True)
    if (RUN/'cancellation.json').exists():
        print('Remaining full-history experiments were cancelled by the user; preserving completed and partial artifacts.')
        return
    with (RUN/'queue.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        while not all((ROOT/'work_dirs/empty_mechanism_stage3/train'/n/'complete.json').exists() for n in ['B0','S','E','SE']):
            status(status='waiting_for_final_epoch_training')
            time.sleep(20)
        if not (RUN/'native_pilot/pilot_success.json').exists():
            status(status='failed',reason='Native-runner pilot not verified');raise SystemExit(1)
        for name in ['B0','S','E','SE']:
            execute(['tools/empty_mechanism_history.py','train','--condition',name],f'queue_train_{name}.log')
            execute(['tools/empty_mechanism_history.py','extract','--condition',name],f'eval_{name}.log')
        for name in ['B0','S','E','SE']:
            execute(['tools/empty_mechanism_history.py','statistics','--condition',name],f'statistics_{name}.log')
        execute(['tools/report_empty_mechanism_history.py'],'report.log')
        for name in ['design','results']:
            execute(['node','work_dirs/empty_mechanism_stage2/document_review/validate.js',f'analysis/empty_mechanism_history_{name}.md'],f'document_{name}.log')
        status(status='complete',training_epoch_runs=20,training_frame_visits=81920,optimizer_updates=20480)
        closeout=dict(status='complete',at=datetime.now().astimezone().isoformat(),
            required_work_remaining=[],audit=str((RUN/'completion_audit.json').relative_to(ROOT)),
            documents=['analysis/empty_mechanism_history_design.md','analysis/empty_mechanism_history_results.md'])
        (RUN/'closeout.json').write_text(json.dumps(closeout,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
