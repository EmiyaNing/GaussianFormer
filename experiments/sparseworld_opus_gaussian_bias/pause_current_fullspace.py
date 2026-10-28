"""Pause only the explicitly requested full-space queue and trainer."""
from datetime import datetime
import json
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

import torch


def main():
    root = Path('out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e')
    output = Path('analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e')
    record_path = output / 'paused_for_ab_first_epoch.json'
    if record_path.exists():
        print(record_path.read_text(), flush=True)
        return
    work = root / 'sparseworld_opus_gaussian_bias'
    targets = [('queue', int((root / 'queue.pid').read_text()),
                'experiments.sparseworld_opus_gaussian_bias.queue_fullspace', signal.SIGTERM),
               ('trainer', int((work / 'run.lock').read_text()),
                'experiments.sparseworld_opus_gaussian_bias.train_fullspace', signal.SIGINT)]
    signaled = []
    for role, pid, module, sig in targets:
        proc = Path('/proc') / str(pid)
        if not proc.exists():
            continue
        cmdline = proc.joinpath('cmdline').read_bytes().split(b'\x00')
        if module.encode() not in cmdline:
            raise ValueError(f'refusing to signal unrelated {role} PID {pid}')
        if role == 'trainer' and str(Path('config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e.py')).encode() not in cmdline:
            raise ValueError('trainer configuration does not match requested run')
        os_signal = int(sig)
        import os
        os.kill(pid, os_signal)
        signaled.append(dict(role=role, pid=pid, signal=os_signal))
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        alive = []
        for target in signaled:
            proc = Path('/proc') / str(target['pid'])
            if proc.exists() and proc.joinpath('cmdline').read_bytes():
                alive.append(target['pid'])
        if not alive:
            break
        time.sleep(.25)
    if alive:
        raise RuntimeError(f'requested processes still shutting down: {alive}')
    saved = torch.load(work / 'latest.pth', map_location='cpu', weights_only=False)
    record = dict(status='paused', reason='user requested independent A/B first-epoch verification',
                  paused_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  signaled=signaled, last_logged_progress=json.loads((work / 'progress.json').read_text()),
                  checkpoint=str(work / 'latest.pth'), checkpoint_epoch_index=saved['epoch'],
                  checkpoint_next_sample=saved['next_iter'], checkpoint_updates=saved['updates'],
                  checkpoint_phase=saved['phase'], config_sha256=saved['config_sha256'],
                  resume_command='python3.10 -m experiments.sparseworld_opus_gaussian_bias.queue_fullspace')
    for path in (record_path, output / 'status.json'):
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
