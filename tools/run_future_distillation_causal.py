#!/usr/bin/env python3
"""Sequential, resumable stage runner; does not interrupt existing training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmengine import Config
from tools.summarize_future_distillation_causal import update_reports,DOCUMENTS

REPO=Path(__file__).resolve().parents[1]


def save(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));temporary.replace(path)


def completed(path):
    return path.exists() and json.loads(path.read_text()).get('status')=='complete'


def alive(pid):
    try:
        os.kill(pid,0)
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ',1)[1][0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False


def main(args):
    cfg=Config.fromfile(args.config)
    root=Path(args.output_dir or (cfg.causal_root+'_smoke2' if args.smoke else cfg.causal_root));root.mkdir(parents=True,exist_ok=True)
    def refresh():update_reports(root,root/'reports' if args.smoke else DOCUMENTS)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=[sys.executable]
    suffix=['--config',args.config,'--output-dir',str(root)]
    if args.smoke:suffix.append('--smoke')
    state=dict(status='running',phase='waiting_mechanisms',pid=os.getpid(),stages={})
    if args.wait_mechanisms_pid:
        while alive(args.wait_mechanisms_pid):
            save(root/'pipeline_state.json',state);refresh()
            time.sleep(10)
    stages=[('S0_S1','tools/experiment_future_distillation_mechanisms.py',root/'s1_gradients.json'),
            ('S2','tools/train_future_distillation_causal_screen.py',root/'s2_screens.json'),
            ('S4_U1','tools/experiment_future_distillation_conditions.py',root/'s4_conditions.json')]
    for name,script,result in stages:
        if completed(result):
            state['stages'][name]='complete';continue
        state['phase']=name;save(root/'pipeline_state.json',state)
        command=base+[script]+suffix
        print('[pipeline start]',name,command,flush=True)
        process=subprocess.Popen(command,cwd=REPO)
        while process.poll() is None:
            refresh();time.sleep(10)
        refresh()
        if process.returncode or not completed(result):
            state.update(status='failed',error=f'{name} exited {process.returncode}; inspect pipeline.log')
            save(root/'pipeline_state.json',state)
            raise RuntimeError(state['error'])
        state['stages'][name]='complete';save(root/'pipeline_state.json',state)
    state.update(status='completed_available_stages',phase='handoff',
                 not_run={'S3':'Independent full-parameter confirmation requires resolving S0 degeneration and an untouched confirmation split.',
                          'S5':'Combination confirmation requires independently replicated beneficial interventions.',
                          'U2':'No verified identical-history multi-future branch environment/data.'})
    save(root/'pipeline_state.json',state);refresh()
    print('[available stages complete]',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_causal.py')
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--wait-mechanisms-pid',type=int)
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args()
    if args.background:
        cfg=Config.fromfile(args.config);root=Path(args.output_dir or (cfg.causal_root+'_smoke2' if args.smoke else cfg.causal_root))
        root.mkdir(parents=True,exist_ok=True)
        with (root/'pipeline.lock').open('a') as lock:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[sys.executable,str(Path(__file__).resolve()),'--config',args.config,'--output-dir',str(root)]
        if args.smoke:command.append('--smoke')
        if args.wait_mechanisms_pid:command.extend(['--wait-mechanisms-pid',str(args.wait_mechanisms_pid)])
        environment=dict(os.environ,MPLCONFIGDIR='/tmp/matplotlib-cache',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (root/'pipeline.log').open('a',buffering=1) as log:
            process=subprocess.Popen(command,cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'job.json',dict(pid=process.pid,command=command,started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
        print('[background causal pipeline]',process.pid,root/'pipeline.log')
    else:
        try:main(args)
        except BlockingIOError:raise
        except Exception as error:
            cfg=Config.fromfile(args.config);root=Path(args.output_dir or (cfg.causal_root+'_smoke2' if args.smoke else cfg.causal_root))
            if root.exists():
                path=root/'pipeline_state.json';state=json.loads(path.read_text()) if path.exists() else {}
                state.update(status='failed',error=f'{type(error).__name__}: {error}');save(path,state)
                update_reports(root,root/'reports' if args.smoke else DOCUMENTS)
            raise
