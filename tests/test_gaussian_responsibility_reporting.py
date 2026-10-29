"""Isolated synthetic pipeline test; never touches actual experiment output."""
import json
import numpy as np


def test_equal_models_have_zero_paired_difference_and_report(tmp_path, monkeypatch):
    import tools.report_gaussian_responsibility as report
    root, run = tmp_path, tmp_path / 'run'
    (root / 'analysis').mkdir()
    monkeypatch.setattr(report, 'ROOT', root)
    monkeypatch.setattr(report, 'RUN', run)
    cm = np.zeros((18, 18), np.int64)
    cm[0, 0], cm[0, 17], cm[17, 0], cm[17, 17] = 5, 2, 3, 90
    point = dict(counts=cm.tolist(), foreground=7, rank_correct=6, conditional_NLL_sum=4.)
    intervention = dict(point, correct_to_empty=0, missed_to_correct=0,
                        empty_to_foreground=0, foreground_to_empty=0)
    probes = [dict(split=split, baseline=point,
        render_only=dict(supported_gaussians=10, mismatched_gaussians=2,
            mismatch_mass=1., responsibility_mass=7., nearest_share_mass_sum=6.,
            guard_voxels=1, gt_voxels=7), sparse_renderer_max_error=1e-7,
        exposure=[dict(voxels=7, rank_correct=6, final_correct=5, missed=2,
                       low=i / 4, high=(i + 1) / 4) for i in range(4)],
        interventions=[dict(method=m, logit_rms_step=s, outcome=intervention)
                       for m in report.MODELS for s in (.01, .03, .1)])
              for split in ('train', 'val') for _ in range(32)]
    monkeypatch.setattr(report, 'audit', lambda: ({'synthetic_fixture': True}, probes))
    runtime = dict(training_and_validation_seconds=60., peak_cuda_allocated_mib=10.)
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    write(root / 'work_dirs/gf2_probability_decomposition/train/probability/training_runtime.json', runtime)
    counts = np.tile(cm[None, None], (2, 4, 1, 1))
    for m in report.MODELS:
        dest = run / 'evaluation' / m
        dest.mkdir(parents=True)
        np.savez(dest / 'scene_counts.npz', counts=counts, scenes=['a', 'b'])
        write(dest / 'statistics.json', dict(rank_accuracy=100 * 6 / 7, conditional_NLL=4 / 7))
        write(dest / 'selection.json', dict(global_threshold=0., class_thresholds=[0.] * 17))
        if m != 'A0':
            write(run / 'train' / m / 'training_runtime.json', runtime)
    report.main()
    result = json.loads((run / 'summary.json').read_text())
    for comparison in result['comparisons'].values():
        for rule in comparison.values():
            assert rule['delta_pp'] == 0 and rule['ci'] == [0, 0]
    np.testing.assert_allclose(result['models']['A0']['rules']['native']['mIoU']['value'], 100 * .5 / 17)
    assert result['probes']['val']['interventions']['A2']['0.03']['delta_mIoU_pp'] == 0
    assert (root / 'analysis/gaussian_responsibility/assets/overview.png').exists()
    document = (root / 'analysis/gaussian_responsibility/gaussian_responsibility_results.md').read_text()
    assert '四组五轮结果' in document
    assert '没有得到足以确认性能增益的证据' in document
    assert '尚未确认稳定收益' in document
    assert '在该固定设置下带来收益' not in document
