"""Durable GPU handoff, experiment execution, and report closeout."""
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.gf2_probability_experiment import RUN,read,write


def status(**values):
    values['time']=time.time();write(RUN/'queue_status.json',values)


def resume_history():
    if (RUN/'previous_experiment_cancelled.json').exists():return
    path=RUN/'history_queue_yield.json'
    if not path.exists():return
    record=read(path)
    if not record.get('active'):return
    stat=Path(f"/proc/{record['pid']}/stat")
    if stat.exists() and stat.read_text().split()[21]==record['start_ticks']:
        os.kill(record['pid'],signal.SIGCONT)
        record.update(active=False,resumed_at=time.time());write(path,record)
    else:raise RuntimeError('Original history supervisor identity changed; inspect before signaling')


def execute(args,log):
    status(status='running',command=args,log=log)
    command=args if args[0]=='node' else [sys.executable]+args
    with (RUN/log).open('a') as stream:
        result=subprocess.call(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result:raise RuntimeError(f'{args} failed with exit code {result}; see {log}')


def main():
    RUN.mkdir(exist_ok=True)
    with (RUN/'queue.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        def terminate(*_):raise SystemExit('Experiment supervisor terminated')
        signal.signal(signal.SIGTERM,terminate)
        try:
            previous=ROOT/'work_dirs/empty_mechanism_history/closeout.json'
            while not previous.exists():
                cancellation=RUN/'previous_experiment_cancelled.json'
                if cancellation.exists():
                    assert read(cancellation)['status']=='cancelled_by_user'
                    baseline=ROOT/'work_dirs/empty_mechanism_history/train/B0/complete.json'
                    assert read(baseline)['status']=='complete'
                    break
                old_status=previous.parent/'queue_status.json'
                if old_status.exists() and read(old_status)['status']=='failed':
                    raise RuntimeError('Earlier experiment queue failed; preserve its priority and repair before proceeding')
                status(status='waiting_for_all_previous_experiments_to_finish',previous=str(previous.relative_to(ROOT)))
                time.sleep(20)
            if previous.exists():assert read(previous)['status']=='complete'
            while not (RUN/'implementation_ready.json').exists():
                status(status='waiting_for_implementation_review');time.sleep(20)
            if not (RUN/'native_pilot/pilot_success.json').exists():
                execute(['tools/check_gf2_probability_native.py'],'native_pilot.log')
            execute(['tools/gf2_probability_experiment.py','train'],'queue_train.log')
            for model in ['baseline','probability']:
                execute(['tools/eval_gf2_probability_decomposition.py','extract','--model',model],f'extract_{model}.log')
                if not (RUN/'evaluation'/model/'timing.json').exists():
                    execute(['tools/eval_gf2_probability_decomposition.py','profile','--model',model],f'profile_{model}.log')
            resume_history()
            for model in ['baseline','probability']:
                execute(['tools/eval_gf2_probability_decomposition.py','statistics','--model',model],f'statistics_{model}.log')
            execute(['tools/report_gf2_probability_decomposition.py'],'report.log')
            for name in ['design','results']:
                execute(['node','work_dirs/empty_mechanism_stage2/document_review/validate.js',f'analysis/gf2_probability_decomposition_{name}.md'],f'document_{name}.log')
            status(status='complete')
            write(RUN/'closeout.json',dict(status='complete',required_work_remaining=[],
                documents=['analysis/gf2_probability_decomposition_design.md','analysis/gf2_probability_decomposition_results.md']))
        except BaseException as error:
            status(status='failed',error=repr(error));raise
        finally:resume_history()


if __name__=='__main__':main()
