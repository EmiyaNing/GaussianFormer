"""Independently check completeness and provenance of the uniform ablation."""
from pathlib import Path
import json, math, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.unweighted_strict_classwise_set_experiment import RUN, STRICT_RUN, assert_protocol, sha, write


def main():
    protocol = assert_protocol()
    for name, digest in protocol['sources'].items():
        assert sha(RUN/'sources'/name) == digest, name
    assert json.loads((RUN/'queue_progress.json').read_text())['status'] == 'complete'
    train = json.loads((RUN/'train/unweighted_classwise/complete.json').read_text())
    evaluation = json.loads((RUN/'eval/unweighted_classwise/complete.json').read_text())
    summary = json.loads((RUN/'summary.json').read_text())
    previous = json.loads((STRICT_RUN/'train/strict_classwise/complete.json').read_text())
    assert train['frames'] == train['unique_frames'] == 4096 and train['scenes'] == 698
    assert train['epochs'] == 1 and train['optimizer_updates'] == 2048
    assert train['data_order_sha256'] == previous['data_order_sha256']
    assert evaluation['frames'] == 6019 and evaluation['scenes'] == 150
    assert sha(Path(train['checkpoint'])) == train['checkpoint_sha256'] == evaluation['checkpoint_sha256']
    for name in ('numerical_checks.json', 'smoke_eval/unweighted_classwise/smoke_passed.json'):
        assert json.loads((RUN/name).read_text())['status'] == 'passed'

    def finite(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for v in value.values(): finite(v)
        elif isinstance(value, list):
            for v in value: finite(v)
    finite(summary)
    rows = updates = 0
    indices = []
    with (RUN/'train/unweighted_classwise/trace.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); finite(row)
            rows += 1; indices.append(row['index'])
            assert row['microbatch'] == rows
            if 'optimizer_update' in row:
                updates += 1; assert row['optimizer_update'] == updates
    assert rows == len(set(indices)) == 4096 and updates == 2048
    assert summary['optimization_audit']['data_order_exact']
    assert summary['optimization_audit']['first_occupancy_max_difference'] == 0
    frames, tokens = [], []
    paths = sorted((RUN/'eval/unweighted_classwise/scenes').glob('*.json'))
    assert len(paths) == 150
    for path in paths:
        meta = json.loads(path.read_text())
        assert meta['checkpoint_sha256'] == evaluation['checkpoint_sha256']
        frames.extend(meta['indices']); tokens.extend(meta['tokens'])
        assert path.with_suffix('.npz').exists()
    assert sorted(frames) == list(range(6019)) and len(set(tokens)) == 6019
    with np.load(RUN/'aggregate/unweighted_classwise.npz') as data:
        cm = data['confusion']; cc = data['class_coverage']
        assert int(cc[:, 0].sum()) == 121325057
        assert data['load'].shape == data['query_core'].shape == (6019, 1200)
        tp = np.diag(cm)[:17]
        union = cm[:17].sum(1) + cm[:, :17].sum(0) - tp
        miou = 100*np.mean(tp/union)
        occupied_tp = cm[:17, :17].sum()
        binary_iou = 100*occupied_tp/(occupied_tp+cm[17, :17].sum()+cm[:17, 17].sum())
        reported = summary['metrics']['unweighted_classwise']
        assert abs(miou-reported['mIoU']) < 1e-10
        assert abs(binary_iou-reported['IoU']) < 1e-10
        for key in ('GS_class_count', 'initial_GS_class_count', 'baseline_GS_class_count'):
            assert data[key].sum() == 6019*76800
        for name in ('initial', 'baseline'):
            transition = data[name+'_to_current_classes']
            np.testing.assert_array_equal(transition.sum(0), data['GS_class_count'])
            np.testing.assert_array_equal(transition.sum(1), data[name+'_GS_class_count'])
    for name in summary['metrics']:
        groups = [row for row in summary['query_quintiles'] if row['arm'] == name]
        assert len(groups) == 5 and abs(sum(row['GT_share'] for row in groups)-100) < 1e-9
    design = ROOT/'analysis/unweighted_strict_classwise_set_mini_design.md'
    report = ROOT/'analysis/unweighted_strict_classwise_set_mini_results.md'
    check = subprocess.run(['node', '.chk.mjs', str(design), str(report)], cwd=ROOT,
                           capture_output=True, text=True, check=True)
    artifacts = [design, report, RUN/'summary.json', RUN/'optimization_audit.json',
        RUN/'interpretation_audit.json', RUN/'numerical_checks.json', Path(__file__).resolve(),
        ROOT/'tools/finalize_unweighted_strict_classwise_set_report.py',
        ROOT/'analysis/figures/unweighted_strict_classwise_set_mini_comparison.png',
        ROOT/'analysis/figures/unweighted_strict_classwise_set_mini_comparison.pdf',
        *sorted((RUN/'aggregate').glob('*.csv'))]
    audit = dict(status='passed', checked=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        frozen_sources_verified=len(protocol['sources']), controls_verified=len(protocol['controls']),
        frozen_copies_verified=True, train_frames=4096, optimizer_updates=2048,
        val_frames=6019, val_scenes=150, GT_foreground=121325057,
        checkpoint_sha256=train['checkpoint_sha256'], matched_training_order=True,
        all_losses_and_gradients_finite=True, independent_confusion_metrics_exact=True,
        histogram_marginals_exact=True, quintile_denominators_exact=True,
        documentation_check=check.stdout.strip(),
        artifacts={str(path.relative_to(ROOT)):sha(path) for path in artifacts})
    write(RUN/'final_integrity_audit.json', audit)
    print(json.dumps({key:value for key,value in audit.items() if key != 'artifacts'}))


if __name__ == '__main__':
    main()
