"""Original CD retained; train and fully evaluate two extra center objectives."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.semantic_neighborhood_experiment import *

def freeze():
    from tools.semantic_field_experiment_common import assert_protocol as original_protocol
    original_protocol();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():return assert_protocol()
    for file in ('numerical_checks.json','pilot.json','smoke_eval/baseline/smoke_passed.json'):
        assert json.loads((RUN/file).read_text())['status']=='passed',file
    c=config(ARMS[0]);b=config('baseline')
    assert c.model==b.model and c.optimizer==b.optimizer
    assert c.loss.loss_cfgs[:2]==b.loss.loss_cfgs and len(c.loss.loss_cfgs)==3
    assert c.train_dataset_config==b.train_dataset_config and c.val_dataset_config==b.val_dataset_config
    assert c.loss.loss_cfgs[0].final_geometry_weight==1
    previous=json.loads((REFERENCE_RUN/'train/baseline/complete.json').read_text())
    assert previous['frames']==4096 and previous['optimizer_updates']==2048 and sha(BASELINE)==previous['checkpoint_sha256']
    paths=protocol_files()
    protocol=dict(version=1,created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in paths},initial=str(INITIAL.relative_to(ROOT)),
        initial_sha256=sha(INITIAL),baseline=str(BASELINE.relative_to(ROOT)),baseline_sha256=sha(BASELINE),
        original_protocol_sha256=sha(REFERENCE_RUN/'protocol.json'),seed=928,
        annotations={s:sha(ROOT/c[s+'_dataset_config'].imageset) for s in ('train','val')},
        train_frames=4096,train_scenes=698,epochs=1,updates=2048,accumulation=2,val_frames=6019,val_scenes=150,
        fresh_optimizer=True,learning_rate=2e-5,warmup_updates=50,minimum_lr_ratio=.1,
        unchanged='All six stages of both original CD directions; all Set classification and camera CE+Lovasz; exact original architecture and renderer',
        removed='Every extra loss of the previous field-supervision experiment',
        added='Final GS center mean Euclidean distance to ALL camera-valid GT centers of its predicted class inside strict 3sigma; plus nearest-center attraction for GT with zero ANY-class 1sigma coverage',
        mean_weights=dict(mean3=.5,uncovered1=.5),units='meters, no extra grid-size division',
        empty_neighborhood='Zero added mean loss, fixed N denominator; original CD remains',
        class_policy='Detached current semantic argmax, classes0..16',
        masks='Detached exact ellipsoid memberships; GT centers, no jitter; no opacity/amplitude factors',
        denominator='All final GS N and all camera-valid foreground GT J, not only active items',
        gradients='Direct added-loss gradients only for centers; no direct scale/rotation/semantic/opacity path',
        baseline_reuse='Same initialization/data order/augmentation/AdamW/LR/budget as verified previous baseline; full validation repeated for new diagnostics',
        user_GT_rule='Confirmed first option: only GT uncovered by every GS core has extra attraction; not priority for idle GS',
        checks={f:sha(RUN/f) for f in ('numerical_checks.json','pilot.json','smoke_eval/baseline/smoke_passed.json')})
    for file in paths:
        target=RUN/'sources'/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    write(RUN/'protocol.json',protocol);return protocol

def run():
    freeze()
    from tools.report_semantic_neighborhood_mean import progress
    stop=threading.Event()
    def refresh():
        while not stop.is_set():
            if not all((RUN/'eval'/n/'complete.json').exists() for n in ('baseline',*ARMS)):progress()
            stop.wait(55)
    watcher=threading.Thread(target=refresh,daemon=True);watcher.start()
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    jobs=[('train_'+ARMS[0],['tools/train_semantic_neighborhood_mean.py',ARMS[0]]),
          ('eval_baseline',['tools/eval_semantic_neighborhood_mean.py','baseline']),
          ('eval_'+ARMS[0],['tools/eval_semantic_neighborhood_mean.py',ARMS[0]]),
          ('report',['tools/report_semantic_neighborhood_mean.py'])]
    try:
        for phase,args in jobs:
            write(RUN/'queue_progress.json',dict(status='running',phase=phase,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print('Starting '+phase,flush=True)
            with (RUN/(phase+'.log')).open('a',buffering=1) as log:
                result=subprocess.run([sys.executable,*args],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{phase} returned{result.returncode}; see {phase}.log')
        write(RUN/'queue_progress.json',dict(status='complete',pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    finally:stop.set();watcher.join(timeout=3)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.freeze_only:freeze()
            else:run()
        except Exception as e:
            write(RUN/'queue_progress.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()))
            raise
