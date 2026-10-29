"""Run the corrected geometry-only uniform-class ablation through full validation."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.unweighted_classwise_geometry_experiment import *

def freeze():
    from tools.classwise_chamfer_experiment import assert_protocol as previous
    previous();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():return assert_protocol()
    check=json.loads((RUN/'numerical_checks.json').read_text());assert check['status']=='passed'
    old,new=config('classwise').to_dict(),config().to_dict()
    old['loss']['loss_cfgs'][0].pop('geometry_class_weights')
    assert new['loss']['loss_cfgs'][0].pop('geometry_class_weights')==[1.]*17
    assert old==new
    controls={str(p.relative_to(ROOT)):sha(p) for p in (INITIAL,BASELINE,CLASSWISE_RUN/'train/classwise/epoch_1.pth')}
    for folder,checkpoint in [(BASELINE_EVAL,BASELINE),
            (CLASSWISE_RUN/'eval/classwise',CLASSWISE_RUN/'train/classwise/epoch_1.pth')]:
        metadata=json.loads((folder/'complete.json').read_text())
        assert metadata['frames']==6019 and metadata['scenes']==150
        assert metadata['checkpoint_sha256']==sha(checkpoint)
        paths=[folder/'complete.json',*sorted((folder/'scenes').glob('*.json')),*sorted((folder/'scenes').glob('*.npz'))]
        assert len(paths)==301
        controls.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    paths=protocol_files();p=dict(version=1,created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        sources={str(f.relative_to(ROOT)):sha(f) for f in paths},controls=controls,
        seed=928,train_frames=4096,train_scenes=698,epochs=1,updates=2048,
        val_frames=6019,val_scenes=150,initial=str(INITIAL.relative_to(ROOT)),
        only_changed_config='geometry_class_weights = [1]*17',
        classification='Original all-class nearest GT focal and original class/spatial weights; no same-class classification',
        geometry='Uniform mean of per-class directional means; geometry skip rules retained',
        first_frame_guard='All six classification terms and Occupancy must equal C exactly before first update',
        no_validation_tuning=True,user_correction='但是，我希望你实现的是：逐类几何+原分类；类间等权呢',
        numerical_check_sha256=sha(RUN/'numerical_checks.json'))
    for f in paths:
        target=RUN/'sources'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    write(RUN/'protocol.json',p);return p

def run():
    freeze();stop=threading.Event()
    report=ROOT/'analysis/unweighted_classwise_geometry_mini_results.md'
    def progress():
        while not stop.is_set():
            tpath=RUN/'train/unweighted_geometry/progress.json';epath=RUN/'eval/unweighted_geometry/progress.json'
            t=json.loads(tpath.read_text()) if tpath.exists() else {}
            e=json.loads(epath.read_text()) if epath.exists() else {}
            if not (RUN/'eval/unweighted_geometry/complete.json').exists():
                tr='4096/4096' if t.get('status')=='complete' else f"{t.get('microbatches',0)}/4096"
                ev='6019/6019' if e.get('status')=='complete' else f"{e.get('processed',0)}/6019，{e.get('scenes',0)}/150场景"
                report.write_text('\n'.join(['# 逐类几何等权＋原分类：实验结果','','**实验进行中，尚未得出性能结论。**','',
                    '本轮只改变几何项类别权重，原分类保持原值。见[实验设计](unweighted_classwise_geometry_mini_design.md)。','',
                    '| 阶段 | 进度 |','| --- | --- |',f'| 训练 | {tr}帧 |',f'| 验证 | {ev} |','']))
            stop.wait(55)
    watcher=threading.Thread(target=progress,daemon=True);watcher.start()
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    jobs=[('train',['tools/train_unweighted_classwise_geometry_mini.py','unweighted_geometry']),
          ('eval',['tools/eval_unweighted_classwise_geometry_mini.py','unweighted_geometry']),
          ('report',['tools/report_unweighted_classwise_geometry_mini.py'])]
    try:
        for phase,args in jobs:
            if phase=='report':stop.set();watcher.join(timeout=3)
            write(RUN/'queue_progress.json',dict(status='running',phase=phase,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print('Starting '+phase,flush=True)
            with (RUN/(phase+'.log')).open('a',buffering=1) as log:
                result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{phase} returned {result.returncode}; see {phase}.log')
        write(RUN/'queue_progress.json',dict(status='complete',pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    finally:stop.set();watcher.join(timeout=3)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:freeze() if args.freeze_only else run()
        except Exception as error:
            write(RUN/'queue_progress.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
            raise
