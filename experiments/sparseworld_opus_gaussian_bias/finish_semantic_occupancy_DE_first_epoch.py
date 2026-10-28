"""Finish durable result analysis after both queued first epochs complete."""
import csv
from datetime import datetime
import time
from zoneinfo import ZoneInfo

from . import analyze_semantic_occupancy_DE_first_epoch as analysis
from .ab_first_epoch.report import read
from .semantic_occupancy_first_epoch.protocol import ANALYSIS, ARMS, C_ANALYSIS, C_WORK, ROOT
from .train import atomic_json, digest


def main():
    deadline = time.monotonic() + 12 * 3600
    while time.monotonic() < deadline:
        status = read(ANALYSIS / 'status.json')
        if status['status'] == 'failed':
            raise RuntimeError(f'Formal D/E queue failed: {status}')
        if status['status'] == 'paused_after_first_epoch_diagnosis':
            break
        time.sleep(30)
    else:
        raise TimeoutError('D/E first-epoch queue has not completed within twelve hours')
    analysis.main()
    launch = read(ANALYSIS / 'launch.json')
    checks = dict(validated_sources_unchanged=all(digest(path)==expected for path,expected in launch['source_sha256'].items()),
        preserved_C_status_unchanged=digest(C_ANALYSIS/'status.json')==launch['preserved_C_status_sha256'],
        preserved_C_checkpoint_unchanged=digest(C_WORK/'epoch_01.pth')==launch['preserved_C_checkpoint_sha256'])
    for name in ARMS:
        boundary = read(ROOT/name/'first_epoch_completed.json')
        budget = read(ROOT/name/'optimization_budget.json')
        checks[name+'_saved_epoch_one'] = (boundary['trained_epochs']==1 and boundary['updates']==1024 and boundary['next_iter']==0 and boundary['planned_epochs']==15)
        checks[name+'_15_epoch_schedule'] = budget['total_updates']==15360 and budget['planned_epochs']==15
        checks[name+'_checkpoint_hash'] = digest(ROOT/name/'epoch_01.pth')==boundary['checkpoint_sha256']
    with (ANALYSIS/'per_horizon_per_class_iou.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    checks['complete_126_class_horizon_rows'] = len(rows)==126 and len({(r['seconds'],r['class_index']) for r in rows})==126
    payload = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
        completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        report=str(ANALYSIS/'report.md'), mechanism_analysis=str(ANALYSIS/'results_analysis.md'),
        figure=str(ANALYSIS/'first_epoch_comparison.png'), judgements=read(ANALYSIS/'comparison.json')['judgements'])
    atomic_json(ANALYSIS/'final_verification.json',payload)
    if not all(checks.values()):
        raise ValueError('D/E final preservation or completion check failed')
    print(f'DE_FINAL_VERIFIED report={ANALYSIS / "report.md"}',flush=True)


if __name__ == '__main__':
    main()
