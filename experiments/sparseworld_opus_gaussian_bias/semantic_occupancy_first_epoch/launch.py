"""Start verified D/E experiments without resuming or overwriting old runs."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch

from ..train import atomic_json, digest
from ..ab_first_epoch.report import read
from .protocol import ANALYSIS, ARMS, C_ANALYSIS, C_CONFIG, C_WORK, ROOT


def main():
    lock = (ROOT / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ANALYSIS / 'launch.json').exists() or any((ROOT / name).exists() for name in ARMS):
        raise ValueError('formal launch requires fresh D/E work directories')
    audit = read(ANALYSIS / 'configuration_audit.json')
    smoke = read(ROOT / 'smoke/all_passed.json')
    tests = read(ANALYSIS / 'unit_checks.json')
    if tests['status'] != 'passed' or smoke['status'] != 'passed':
        raise ValueError('CPU/GPU validation must pass before formal launch')
    expected = read(C_WORK / 'initialization.json')['initial_model_sha256']
    if smoke['configuration_audit_sha256'] != digest(ANALYSIS / 'configuration_audit.json'):
        raise ValueError('configuration audit changed after GPU verification')
    for name, path in ARMS.items():
        checks = [row for row in smoke['checks'] if row['arm'] == name]
        if ([row['epoch_index'] for row in checks] != [0, 14] or
                any(row['config_sha256'] != digest(path) or row['shared_initial_model_sha256'] != expected for row in checks) or
                digest(path) != audit['arms'][name]['config_sha256'] or
                digest(C_CONFIG) != audit['arms'][name]['source_sha256']):
            raise ValueError(f'validated configuration or initial model differs: {name}')
    if read(C_ANALYSIS / 'status.json')['status'] != 'paused_after_first_epoch_diagnosis':
        raise ValueError('preserved C run is no longer paused')
    if not torch.cuda.is_available():
        raise ValueError('GPU unavailable')
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if pids:
        raise ValueError('GPU compute jobs already active; preserve them')
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MPLCONFIGDIR='/tmp/sparseworld_de_mpl',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [sys.executable, '-m', 'experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.queue']
    record = dict(status='prepared', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command,
        source_sha256={str(path): digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
        configuration_audit_sha256=digest(ANALYSIS / 'configuration_audit.json'),
        planned_epochs_per_arm=15, total_updates_per_arm=15360, runtime_stop_after_epoch=1,
        train_samples=4096, validation_samples=1024, geometry_route_deferred=True,
        shared_initial_model_sha256=expected, report=str(ANALYSIS / 'report.md'),
        significance_protocol=dict(primary='current mIoU', draws=2000, resampling_unit='scene',
            two_route_familywise_confidence=.95, minimum_relative_gain_percent=5., training_seeds=1),
        preserved_C_status_sha256=digest(C_ANALYSIS / 'status.json'),
        preserved_C_checkpoint_sha256=read(C_WORK / 'first_epoch_completed.json')['checkpoint_sha256'])
    atomic_json(ANALYSIS / 'launch.json', record)
    with (ROOT / 'queue.log').open('a', buffering=1) as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                   cwd=Path.cwd(), start_new_session=True)
    (ROOT / 'queue.pid').write_text(str(process.pid)+'\n')
    record.update(status='launched', queue_pid=process.pid)
    atomic_json(ANALYSIS / 'launch.json', record)
    (ANALYSIS / 'execution.md').write_text(
        '# D/E 实验执行记录\n\n'
        f"启动时间：{record['started_at']}；queue PID={process.pid}。\n\n"
        '4096训练/1024评估，scratch初始化，15轮学习率计划，每组完成第一轮训练及完整诊断后暂停。'
        '先只读重评C收集场景计数，再依次执行D和E；每组结束自动更新对比报告。\n\n'
        '- D：仅平衡17类条件辅助CE。\n'
        '- E：独立占用证据分支和固定二元决策，保留原几何/时序与损失权重。\n'
        '- 本次不启用几何研究路线；不恢复旧训练，不覆盖旧结果。\n\n'
        '[运行状态](status.json) · [独立配置审计](configuration_audit.json) · [结果报告](report.md)\n')
    print(f'D/E durable queue launched: PID={process.pid}; report={ANALYSIS / "report.md"}', flush=True)


if __name__ == '__main__':
    main()

