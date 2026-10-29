"""Freeze checked strict-classwise protocol, train, fully validate, report."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.strict_classwise_set_experiment import *


def freeze():
    from tools.classwise_chamfer_experiment import assert_protocol as previous_protocol
    previous_protocol();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():return assert_protocol()
    checks=('numerical_checks.json','pilot.json','smoke_eval/strict_classwise/smoke_passed.json')
    for name in checks:assert json.loads((RUN/name).read_text())['status']=='passed',name
    c=config(ARMS[0]);previous=config('classwise');baseline=config('baseline')
    for other in (previous,baseline):
        assert c.model==other.model and c.optimizer==other.optimizer
        assert c.train_dataset_config==other.train_dataset_config and c.val_dataset_config==other.val_dataset_config
        assert len(c.loss.loss_cfgs)==2 and c.loss.loss_cfgs[1]==other.loss.loss_cfgs[1]
    first=dict(c.loss.loss_cfgs[0]);first['type']=previous.loss.loss_cfgs[0].type
    assert first==previous.loss.loss_cfgs[0]
    controls={str(p.relative_to(ROOT)):sha(p) for p in (INITIAL,BASELINE,PREVIOUS)}
    for folder in (BASELINE_EVAL,PREVIOUS_RUN/'eval/classwise'):
        metadata=json.loads((folder/'complete.json').read_text())
        assert metadata['frames']==6019 and metadata['scenes']==150
        checkpoint=BASELINE if folder==BASELINE_EVAL else PREVIOUS
        assert metadata['checkpoint_sha256']==sha(checkpoint)
        paths=[folder/'complete.json',*sorted((folder/'scenes').glob('*.json')),*sorted((folder/'scenes').glob('*.npz'))]
        assert len(paths)==301
        controls.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    paths=protocol_files()
    protocol=dict(version=1,created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in paths},controls=controls,
        initial=str(INITIAL.relative_to(ROOT)),seed=928,train_frames=4096,train_scenes=698,
        epochs=1,updates=2048,accumulation=2,val_frames=6019,val_scenes=150,
        class_weights=list(c.loss.loss_cfgs[0].geometry_class_weights),
        geometry='Exactly the previous classwise geometry implementation, all six stages, both SmoothL1 directions',
        classification='Nearest same predicted-class GT; full17-output sigmoid focal; class mean then one outer class weight divided by active weight sum; original GT spatial weight only',
        classification_target='Equals detached current predicted argmax; user explicitly accepted self reinforcement',
        absent_class='Either side empty: skip class for all three terms, exclude outer denominator',
        preserved='camera GT filter, Occupancy, model, renderer, optimizer, lambda_cls=2, lambda_pts=.5, six stage weights1',
        annotations={s:sha(ROOT/c[s+'_dataset_config'].imageset) for s in ('train','val')},
        user_confirmation='2026-10-27: 不错，和我预想的一样，请帮我进行这组实验，并统计实验结果。',
        no_validation_tuning=True,checks={name:sha(RUN/name) for name in checks})
    for p in paths:
        target=RUN/'sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    write(RUN/'protocol.json',protocol);return protocol


def run():
    freeze()
    from tools.report_strict_classwise_set_mini import progress
    stop=threading.Event()
    def refresh():
        while not stop.is_set():
            if not (RUN/'eval/strict_classwise/complete.json').exists():progress()
            stop.wait(55)
    watcher=threading.Thread(target=refresh,daemon=True);watcher.start()
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    jobs=[('train_strict_classwise',['tools/train_strict_classwise_set_mini.py','strict_classwise']),
          ('eval_strict_classwise',['tools/eval_strict_classwise_set_mini.py','strict_classwise']),
          ('report',['tools/report_strict_classwise_set_mini.py'])]
    try:
        for phase,args in jobs:
            write(RUN/'queue_progress.json',dict(status='running',phase=phase,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print('Starting '+phase,flush=True)
            with (RUN/(phase+'.log')).open('a',buffering=1) as log:
                result=subprocess.run([sys.executable,*args],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{phase} returned {result.returncode}; see {phase}.log')
        write(RUN/'queue_progress.json',dict(status='complete',pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    finally:stop.set();watcher.join(timeout=3)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.freeze_only:freeze()
            else:run()
        except Exception as error:
            write(RUN/'queue_progress.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
            raise
