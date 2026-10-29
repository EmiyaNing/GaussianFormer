"""Matched low-LR continuation of the actual trained 8-frame checkpoint."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import tools.gaussian_geometry_causal as trial
from tools.gaussian_geometry_causal import RUN,write,sha
from tools.gaussian_geometry_selective_control import Recorder,SelectiveLoss

CHECKPOINT=ROOT/'ckpts/opus_gaussian_8f_tiny.pth'
NAMES=('mature_standard','mature_no_empty_center_boost')


def train(name):
    native_build=trial.build;native_routes=trial.Routes
    def build(*args,**kwargs):
        c,net,opt,loss,state=native_build(*args,**kwargs)
        # The generic branch runner multiplies initial_lr by 0.5. Set that
        # metadata so the actual LR is 10x the checkpoint's tiny terminal LR
        # (approximately 2e-6), still 1/100 of the nominal base training LR.
        for group in opt.param_groups:group['initial_lr']=20*group['lr']
        if name==NAMES[1]:loss=SelectiveLoss(loss,net.head)
        return c,net,opt,loss,state
    trial.build=build
    if name==NAMES[1]:trial.Routes=Recorder
    try:return trial.training(name,1024,932,CHECKPOINT,8,False)
    finally:trial.build=native_build;trial.Routes=native_routes


def run():
    protocol=dict(registered_at=datetime.now().astimezone().isoformat(),
        purpose='Reversibility check on the actual checkpoint that produced the full-scene phenomenon; do not use it to reconstruct earlier training causes',
        reason='The new common checkpoint already approaches attribute bounds; a mature matched continuation complements the early-trajectory experiment.',
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),
        source_sha256=sha(__file__),selective_source_sha256=sha(ROOT/'tools/gaussian_geometry_selective_control.py'),
        names=NAMES,seed=932,microbatches=1024,updates=128,accumulation=8,
        budget_revision='Before any mature training: the complete common-checkpoint validation split has mIoU 4.37%, so it is too underfit to be the main final-generalization test. Extend the matched mature pair to 128 updates at the previously registered low LR; no treatment outcome used for this decision.',
        learning_rate='10x the saved terminal LR per group (about 2e-6); preserve optimizer moments and freeze the same empty parameters in both arms',
        lr_revision='Before any mature training: strict optimizer audit found all saved rates near 2e-7, so use a declared 10x diagnostic step size instead of treating an almost-zero update as a null causal result; previous registration retained.',
        evaluation='Every train/val scene and keyframe, same as primary experiment')
    path=RUN/'mature_protocol.json'
    if not path.exists():write(path,protocol)
    else:assert json.loads(path.read_text())['source_sha256']==sha(__file__)
    while not (RUN/'selective_complete.json').exists():
        status=json.loads((RUN/'queue_status.json').read_text())
        if status['status']=='failed':raise RuntimeError('Primary experiment failed')
        time.sleep(5)
    for name in NAMES:
        write(RUN/'mature_queue_status.json',dict(status='training',name=name))
        train(name)
    import numpy as np
    traces=[];snapshots=[]
    for name in NAMES:
        folder=RUN/'train'/f'{name}_s932'
        traces.append([json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()])
        snapshots.append(np.load(folder/'initial_forward.npz'))
    assert [(r['index'],r['fingerprint']) for r in traces[0]]==[(r['index'],r['fingerprint']) for r in traces[1]]
    errors={k:float(np.abs(snapshots[0][k]-snapshots[1][k]).max()) for k in snapshots[0].files}
    for k in errors:assert np.allclose(snapshots[0][k],snapshots[1][k],rtol=1e-5,atol=2e-5)
    write(RUN/'mature_pairing_audit.json',dict(status='passed',microbatches=1024,
        all_data_fingerprints_equal=True,initial_max_abs_errors=errors))
    for name in NAMES:
        checkpoint=RUN/'train'/f'{name}_s932/final.pth'
        write(RUN/'mature_queue_status.json',dict(status='evaluating',name=name))
        with (RUN/(name+'_eval_console.log')).open('a') as f:
            subprocess.run([sys.executable,'tools/eval_gaussian_geometry_causal.py',
                '--name',name,'--checkpoint',str(checkpoint)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(RUN/'mature_complete.json',dict(status='complete',names=NAMES))
    write(RUN/'mature_queue_status.json',dict(status='complete'))


if __name__=='__main__':
    try:run()
    except BaseException as e:
        write(RUN/'mature_queue_status.json',dict(status='failed',error=repr(e)))
        raise
