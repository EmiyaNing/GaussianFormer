"""Resume the authorized paired experiment queue; fail closed on audit errors."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.gaussian_geometry_causal import RUN,CONFIG,ARMS,cfg,write,sha


def execute(label,command):
    path=RUN/(label+'_console.log')
    start=time.perf_counter()
    with path.open('a') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        write(RUN/'queue_status.json',dict(status='running',stage=label,pid=child.pid,
            command=command,updated_at=datetime.now().astimezone().isoformat()))
        code=child.wait()
    record=dict(stage=label,command=command,returncode=code,seconds=time.perf_counter()-start)
    with (RUN/'queue_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if code:raise RuntimeError(f'{label} exited {code}; see {path}')


def run(common_pid=None):
    RUN.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'tools/gaussian_geometry_causal.py',ROOT/'tools/eval_gaussian_geometry_causal.py',
        ROOT/'tools/audit_gaussian_geometry_causal.py',Path(__file__),
        ROOT/'analysis/gaussian_geometry_causal_design.md']
    inherited=json.loads((ROOT/'work_dirs/boundary_gaussian_full/protocol.json').read_text())
    for name,h in inherited['source_sha256'].items():assert sha(ROOT/name)==h,name
    protocol=dict(version='paired_8f_geometry_amplitude_routes_v1',
        initialized_at=datetime.now().astimezone().isoformat(),config=str(CONFIG.relative_to(ROOT)),
        seed=928,common_microbatches=2048,branch_microbatches=1024,gradient_accumulation=8,
        arms=list(ARMS[:4]),expected=inherited['expected'],
        annotation_sha256=inherited['annotation_sha256'],
        common_launch_source_sha256=sha(RUN/'sources/common/gaussian_geometry_causal.py'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        inherited_source_sha256=inherited['source_sha256'],
        local_audits_only='Two-update pilot branches are implementation checks, not statistical evidence',
        training_scope='Finite updates drawn from a permutation of the full training pool; full-scene evaluation, no sampled evaluation subset')
    if not (RUN/'protocol.json').exists():
        write(RUN/'protocol.json',protocol)
        cfg().dump(RUN/'resolved_config.py')
        for p in files:
            target=RUN/'sources/paired'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    else:
        old=json.loads((RUN/'protocol.json').read_text())
        for p,h in old['source_sha256'].items():assert sha(ROOT/p)==h,p
    # The common trunk was already running when the independent queue was
    # prepared. Its directory is relocated only after its completion marker.
    initial=RUN/'common/standard_s928/final.pth'
    legacy=RUN/'train/standard_s928'
    while not initial.exists():
        if (legacy/'complete.json').exists():
            initial.parent.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(legacy),str(initial.parent))
            meta=json.loads((initial.parent/'complete.json').read_text())
            # The old runner hashed its source at finish, after an optimizer
            # freeze-order fix for future branches. Preserve that observation,
            # and attach the exact source snapshot used by the common process.
            write(initial.parent/'complete_original.json',meta)
            meta['source_sha256_observed_at_finish']=meta['source_sha256']
            meta['source_sha256']=protocol['common_launch_source_sha256']
            meta['source_provenance_note']='Common process used the frozen launch copy; later changes fix branch optimizer construction and directory separation only.'
            write(initial.parent/'complete.json',meta)
            break
        if common_pid:
            try:os.kill(common_pid,0)
            except ProcessLookupError:raise RuntimeError('Common training exited before completion')
        write(RUN/'queue_status.json',dict(status='running',stage='waiting_common',pid=common_pid,
            updated_at=datetime.now().astimezone().isoformat()))
        time.sleep(5)
    for audit in ('numerical_audit','real_route_audit','evaluation_audit'):
        assert json.loads((RUN/(audit+'.json')).read_text())['status']=='passed',audit
    python=sys.executable
    for arm in ARMS[:4]:
        execute('pilot_'+arm,[python,'tools/gaussian_geometry_causal.py','train','--arm',arm,
            '--initial',str(initial),'--seed','929','--steps','16','--pilot'])
    execute('pilot_pairing',[python,'tools/audit_gaussian_geometry_causal.py','pairing',
        '--folder',str(RUN/'pilot'),'--seed','929'])
    for arm in ARMS[:4]:
        execute('train_'+arm,[python,'tools/gaussian_geometry_causal.py','train','--arm',arm,
            '--initial',str(initial),'--seed','928','--steps','1024'])
    execute('training_pairing',[python,'tools/audit_gaussian_geometry_causal.py','pairing',
        '--folder',str(RUN/'train'),'--seed','928'])
    targets=[('common',initial)]+[(arm,RUN/'train'/f'{arm}_s928/final.pth') for arm in ARMS[:4]]
    for name,checkpoint in targets:
        execute('eval_'+name,[python,'tools/eval_gaussian_geometry_causal.py',
            '--name',name,'--checkpoint',str(checkpoint)])
    write(RUN/'queue_status.json',dict(status='execution_complete',
        updated_at=datetime.now().astimezone().isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--common-pid',type=int)
    args=p.parse_args()
    try:run(args.common_pid)
    except BaseException as e:
        write(RUN/'queue_status.json',dict(status='failed',error=repr(e),
            updated_at=datetime.now().astimezone().isoformat()))
        raise
