"""Launch only validated D continuation, independently of the first-epoch runs."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch

from ..train import atomic_json, digest
from .prepare import read
from .protocol import ANALYSIS, CONFIG, ROOT


def main():
    lock = (ROOT/'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ANALYSIS/'launch.json').exists():
        raise ValueError('D continuation already launched')
    for name in ('preparation.json', 'cpu_validation.json', 'gpu_validation.json'):
        if read(ANALYSIS/name)['status']!='passed':
            raise ValueError(f'validation not passed: {name}')
    if not torch.cuda.is_available():
        raise ValueError('GPU unavailable')
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if pids:
        raise ValueError(f'GPU has active jobs: {pids}')
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_d15_mpl',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [sys.executable,'-m','experiments.sparseworld_opus_gaussian_bias.balanced_D_full15e.queue']
    record = dict(status='prepared', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command,
        source_sha256={str(path):digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
        config_sha256=digest(CONFIG), planned_epochs=15, resume_epoch=2, previous_updates=1024,
        planned_total_updates=15360, per_epoch_diagnostics=True, report=str(ANALYSIS/'report.md'))
    atomic_json(ANALYSIS/'launch.json',record)
    with (ROOT/'queue.log').open('a',buffering=1) as stream:
        child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=Path.cwd(),start_new_session=True)
    (ROOT/'queue.pid').write_text(str(child.pid)+'\n')
    record.update(status='launched',queue_pid=child.pid)
    atomic_json(ANALYSIS/'launch.json',record)
    (ANALYSIS/'execution.md').write_text('# D完整15轮续训\n\n'
        f"启动时间：{record['started_at']}；queue PID={child.pid}。\n\n"
        '从D已核验第一轮边界继续第2～15轮，沿用4096/1024子集、原15轮学习率进度及模型/优化器/RNG。每轮完整评估并诊断，结束后独立复评最佳checkpoint，自动与Strict完整15轮结果对比。\n\n'
        '- [运行状态](status.json)\n- [逐轮进度报告](progress_report.md)\n- [续训来源核验](preparation.json)\n- [最终报告（完成后生成）](report.md)\n')
    print(f'D continuation launched: PID={child.pid}',flush=True)


if __name__ == '__main__':
    main()
