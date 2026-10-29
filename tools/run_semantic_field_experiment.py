"""Freeze and run initial evaluation + three matched mini-set fine-tuning arms."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.semantic_field_experiment_common import *


def freeze():
    import scipy,torch
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():return assert_protocol()
    for file in ('numerical_checks.json','pilot.json','smoke_eval/initial/smoke_passed.json'):
        assert json.loads((RUN/file).read_text())['status']=='passed',file
    c=config('baseline');paths=protocol_files()
    p=dict(version=1,created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        sources={str(f.relative_to(ROOT)):sha(f) for f in paths},initial=str(INITIAL.relative_to(ROOT)),
        initial_sha256=sha(INITIAL),train_frames=4096,train_scenes=698,val_frames=6019,val_scenes=150,
        annotations={s:sha(ROOT/c[s+'_dataset_config'].imageset) for s in ('train','val')},
        seed=928,fresh_optimizer=True,epochs=1,accumulation=2,updates=2048,learning_rate=2e-5,
        original_losses='All six-stage camera-filtered CD and Set classification; camera CE+Lovasz retained',
        original_renderer='Additive softplus semantic GS with learned empty bias; architecture unchanged',
        field_weights=dict(geometry=.5,semantic=.5),gt_samples='One uniform interior point per camera foreground cube',
        GS_samples='Eight standard-normal points conditioned norm<=3, independent CUDA RNG per frame',
        nearest='Exact Euclidean solid ellipsoid, global branch-and-bound with maxscale0.4m; not fixed-K',
        semantic_target='Nearest camera-FG semantic 6-connected component, voxel centers within 2m of current GS mean',
        correspondence='GT-led one-hot component/class; confidence detached, semantic share blend0->.25 in512frames',
        valid='Camera mask and GT0..17, tested at actual field sample position; fixed NK denominator',
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,scipy=scipy.__version__),
        inference='No change; auxiliary field loss only used during training',
        tests={f:sha(RUN/f) for f in ('numerical_checks.json','pilot.json','smoke_eval/initial/smoke_passed.json')})
    for file in paths:
        target=RUN/'sources'/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
    write(RUN/'protocol.json',p);return p


def run():
    freeze();env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    jobs=[('eval_initial',['tools/eval_semantic_field_mini.py','initial'])]
    for arm in ARMS:jobs += [('train_'+arm,['tools/train_semantic_field_mini.py',arm]),('eval_'+arm,['tools/eval_semantic_field_mini.py',arm])]
    jobs += [('report',['tools/report_semantic_field_mini.py'])]
    for phase,args in jobs:
        write(RUN/'queue_progress.json',dict(status='running',phase=phase,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
        print('Starting '+phase,flush=True)
        with (RUN/(phase+'.log')).open('a',buffering=1) as log:
            completed=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError(f'{phase} exited {completed.returncode}; see {phase}.log')
        subprocess.run([sys.executable,'tools/report_semantic_field_mini.py'],cwd=ROOT,env=env,check=True,
            stdout=subprocess.DEVNULL)
    write(RUN/'queue_progress.json',dict(status='complete',pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))

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
