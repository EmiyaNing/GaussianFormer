"""Freeze and execute both final-stage loss variants, followed by a unified report."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.bidirectional_coverage_experiment import *

def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():return assert_protocol()
    checks={}
    for name in ('numerical_checks.json','pilot.json','smoke_eval/union/smoke_passed.json'):
        assert json.loads((RUN/name).read_text())['status']=='passed',name
        checks[name]=sha(RUN/name)
    base=config('baseline').to_dict();pilot=json.loads((RUN/'pilot.json').read_text())
    for arm in ARMS:
        c=config(arm).to_dict();extra=c['loss']['loss_cfgs'].pop()
        assert extra['mode']==arm and extra['weight']==pilot['weights'][arm]
        assert extra['reverse_weight']==(1. if arm=='union' else pilot['decision_reverse_weight'])
        assert c['loss']['loss_cfgs'][0]['final_geometry_weight']==0.
        c['loss']['loss_cfgs'][0]['final_geometry_weight']=1.
        c['custom_imports']['imports'].remove('loss.bidirectional_union_coverage_loss')
        assert c==base,'Unexpected change outside the final geometry replacement'
    controls={str(INITIAL.relative_to(ROOT)):sha(INITIAL)}
    for name,folder in CONTROL_FOLDERS.items():
        checkpoint=CONTROL_CHECKPOINTS[name];train=json.loads((checkpoint.parent/'complete.json').read_text())
        meta=json.loads((folder/'complete.json').read_text())
        assert train['frames']==4096 and train['optimizer_updates']==2048
        assert train['data_order_sha256']=='5dec912988b2ef3fd691f67580b57c5b79edb0c449c13b1b061ff925b085f8c0'
        assert meta['frames']==6019 and meta['scenes']==150
        assert sha(checkpoint)==train['checkpoint_sha256']==meta['checkpoint_sha256']
        paths=[checkpoint,checkpoint.parent/'complete.json',folder/'complete.json',
               *sorted((folder/'scenes').glob('*.json')),*sorted((folder/'scenes').glob('*.npz'))]
        assert len(paths)==303
        controls.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    refs=sorted((REFERENCE_RUN/'references').glob('*/*.npz'));assert len(refs)==12038
    controls.update({str(p.relative_to(ROOT)):sha(p) for p in refs})
    sources=protocol_files();protocol=dict(version=1,created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},controls=controls,checks=checks,
        initial=str(INITIAL.relative_to(ROOT)),train_frames=4096,train_scenes=698,epochs=1,
        updates=2048,val_frames=6019,val_scenes=150,seed=928,
        replacement='Only final-stage original center geometry disabled; early geometry, all original classification, Occupancy unchanged',
        reverse_sampling='4096 uniform valid voxels without replacement, inverse inclusion probability; identical independent frame seeds',
        exact_core_search='Bounding lower costs; no fixed K, no exponential-tail truncation',
        calibration=pilot['weights'],decision_reverse_weight=pilot['decision_reverse_weight'],
        historical_controls=list(CONTROL_FOLDERS),no_validation_tuning=True)
    for p in sources:
        target=RUN/'sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    write(RUN/'protocol.json',protocol);return protocol

def run():
    freeze();stop=threading.Event();report=ROOT/'analysis/bidirectional_coverage_mini_results.md'
    def refresh():
        while not stop.is_set():
            lines=['# 两种双向覆盖损失：实验结果','','**实验进行中，尚不判断方案是否有效。**','',
                '定义、系数与对照口径见[实验设计](bidirectional_coverage_mini_design.md)。','',
                '| 新方案 | 训练 | 完整验证 |','| --- | --- | --- |']
            for arm,label in [('union','U：max联合支持'),('decision','R：真实渲染错误')]:
                tp=RUN/'train'/arm/'progress.json';ep=RUN/'eval'/arm/'progress.json'
                t=json.loads(tp.read_text()) if tp.exists() else {};e=json.loads(ep.read_text()) if ep.exists() else {}
                ts='4096/4096' if t.get('status')=='complete' else f"{t.get('microbatches',0)}/4096"
                es='6019/6019' if e.get('status')=='complete' else f"{e.get('processed',0)}/6019，{e.get('scenes',0)}/150场景"
                lines.append(f'| {label} | {ts} | {es} |')
            report.write_text('\n'.join(lines)+'\n');stop.wait(55)
    watcher=threading.Thread(target=refresh,daemon=True);watcher.start()
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    jobs=[]
    for arm in ARMS:
        jobs.extend([(f'train_{arm}',['tools/train_bidirectional_coverage_mini.py',arm]),
                     (f'eval_{arm}',['tools/eval_bidirectional_coverage_mini.py',arm])])
    jobs.append(('report',['tools/report_bidirectional_coverage_mini.py']))
    try:
        for phase,args in jobs:
            if phase=='report':stop.set();watcher.join(timeout=3)
            write(RUN/'queue_progress.json',dict(status='running',phase=phase,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print('Starting '+phase,flush=True)
            with (RUN/(phase+'.log')).open('a',buffering=1) as log:
                result=subprocess.run([sys.executable,*args],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{phase} returned {result.returncode}; see {phase}.log')
        write(RUN/'queue_progress.json',dict(status='complete',pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    finally:stop.set();watcher.join(timeout=3)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:freeze() if a.freeze_only else run()
        except Exception as error:
            write(RUN/'queue_progress.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
            raise
