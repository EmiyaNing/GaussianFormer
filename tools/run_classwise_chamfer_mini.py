"""Freeze, fine-tune, fully validate and report the classwise CD replacement."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.classwise_chamfer_experiment import *


def freeze():
    from tools.semantic_neighborhood_experiment import assert_protocol as previous_protocol
    previous_protocol();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():return assert_protocol()
    checks=('numerical_checks.json','pilot.json','smoke_eval/classwise/smoke_passed.json')
    for name in checks:assert json.loads((RUN/name).read_text())['status']=='passed',name
    c=config('classwise');b=config('baseline')
    assert c.model==b.model and c.optimizer==b.optimizer
    assert c.train_dataset_config==b.train_dataset_config and c.val_dataset_config==b.val_dataset_config
    assert len(c.loss.loss_cfgs)==2 and c.loss.loss_cfgs[1]==b.loss.loss_cfgs[1]
    a=dict(c.loss.loss_cfgs[0]);a.pop('geometry_class_weights');a['type']=b.loss.loss_cfgs[0].type
    assert a==b.loss.loss_cfgs[0]
    prior=json.loads((REFERENCE_RUN/'train/baseline/complete.json').read_text())
    e=json.loads((BASELINE_EVAL/'complete.json').read_text())
    assert prior['frames']==4096 and prior['optimizer_updates']==2048
    assert sha(BASELINE)==prior['checkpoint_sha256']==e['checkpoint_sha256']
    assert (e['frames'],e['scenes'])==(6019,150)
    paths=protocol_files()
    base_files=[BASELINE_EVAL/'complete.json',*sorted((BASELINE_EVAL/'scenes').glob('*.json')),*sorted((BASELINE_EVAL/'scenes').glob('*.npz'))]
    assert len(base_files)==301
    refs=sorted((REFERENCE_RUN/'references').glob('*/*.npz'));assert len(refs)==12038
    protocol=dict(version=1,created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in paths},initial=str(INITIAL.relative_to(ROOT)),initial_sha256=sha(INITIAL),
        baseline=str(BASELINE.relative_to(ROOT)),baseline_sha256=sha(BASELINE),
        baseline_eval=str(BASELINE_EVAL.relative_to(ROOT)),baseline_artifacts={str(f.relative_to(ROOT)):sha(f) for f in base_files},
        reference_manifest_sha256=hashlib.sha256(json.dumps({str(f.relative_to(ROOT)):sha(f) for f in refs},sort_keys=True).encode()).hexdigest(),
        original_protocol_sha256=sha(REFERENCE_RUN/'protocol.json'),
        annotations={s:sha(ROOT/c[s+'_dataset_config'].imageset) for s in ('train','val')},seed=928,
        train_frames=4096,train_scenes=698,epochs=1,updates=2048,accumulation=2,val_frames=6019,val_scenes=150,
        fresh_optimizer=True,learning_rate=2e-5,warmup_updates=50,minimum_lr_ratio=.1,
        replacement='Both geometry directions in all six stages; exact nearest same-class sets, original Smooth-L1 beta=.2 and original within-class GT-distance weights',
        partition='Detached current per-stage semantic argmax, foreground0..16; camera GT voxel centers',
        geometry_class_weights=list(c.loss.loss_cfgs[0].geometry_class_weights),
        reduction='Per-class directional mean, then sum(w_c*CD_c)/sum(active w_c), not sample-cardinality-weighted class reduction',
        empty_class='Skip both directions for classes with either side empty; excluded denominator; no cross-class fallback; counts logged',
        unchanged='Original all-class Set focal assignment/weights, lambda_cls=2, lambda_pts=.5, stage_weights=1, camera CE+Lovasz, architecture, renderer, optimizer and data',
        removed='All previous added geometry/field/core/SDF losses and candidate importance architecture; only Set and Occupancy remain',
        direct_gradient='Centers only, hard semantic partitions and NN indices detached',
        baseline_reuse='Verified previous matched 4096-frame original-objective training and complete6019-frame evaluation; read-only reuse',
        no_validation_tuning=True,checks={n:sha(RUN/n) for n in checks})
    for f in paths:
        target=RUN/'sources'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    write(RUN/'protocol.json',protocol);return protocol


def run():
    freeze()
    from tools.report_classwise_chamfer_mini import progress
    stop=threading.Event()
    def refresh():
        while not stop.is_set():
            if not (RUN/'eval/classwise/complete.json').exists():progress()
            stop.wait(55)
    watcher=threading.Thread(target=refresh,daemon=True);watcher.start()
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    jobs=[('train_classwise',['tools/train_classwise_chamfer_mini.py','classwise']),
        ('eval_classwise',['tools/eval_classwise_chamfer_mini.py','classwise']),
        ('report',['tools/report_classwise_chamfer_mini.py'])]
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
        except Exception as error:
            write(RUN/'queue_progress.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
            raise
