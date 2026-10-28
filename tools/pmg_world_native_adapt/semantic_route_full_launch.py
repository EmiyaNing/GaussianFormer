"""Launch only the GPU-validated, resumable full 15-epoch S2 run."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.sparse_opus_gausisan_world import runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--work-dir',required=True)
    parser.add_argument('--smoke-dir',required=True)
    args=parser.parse_args()
    smoke=json.loads((Path(args.smoke_dir)/'smoke.json').read_text())
    assert smoke['passed'] and smoke['seven_voxel_horizons']==list(range(7))
    assert smoke['actual_optimizer_updates']>=1
    assert smoke['config_sha256']==runtime.digest(args.config)
    for name,digest in smoke['identity']['source_sha256'].items():
        if runtime.digest(name)!=digest:raise ValueError('Implementation changed since smoke: '+name)
    directory=Path(args.work_dir);directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'launch.json'
    if marker.exists():
        previous=json.loads(marker.read_text());status=Path('/proc')/str(previous['pid'])/'stat'
        if status.exists() and status.read_text().split()[2]!='Z':
            command=(status.parent/'cmdline').read_bytes().decode(errors='replace')
            if args.work_dir in command:raise RuntimeError('Training process already alive')
    command=[sys.executable,'-u','-m','tools.pmg_world_native_adapt.semantic_route_full_train',
             '--config',args.config,'--work-dir',args.work_dir]
    env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/pmg_mpl_cache')
    with (directory/'train.log').open('a',buffering=1) as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True,env=env)
    value=dict(pid=process.pid,command=command,log=str(directory/'train.log'),
        config_sha256=smoke['config_sha256'],planned_epochs=15,wall_time_limit_hours=0,
        launched_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))
    runtime.atomic_json(marker,value)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
