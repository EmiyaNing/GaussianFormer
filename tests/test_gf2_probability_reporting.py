"""CPU-only synthetic fixture; never writes into real experiment results."""
import csv
import json
import numpy as np


def test_report_pipeline_with_zero_paired_difference(tmp_path,monkeypatch):
    import tools.report_gf2_probability_decomposition as report
    from tools.calibrate_empty_mechanism_stage2 import EDGES,NB
    run=tmp_path/'run';run.mkdir();(tmp_path/'analysis').mkdir()
    (tmp_path/'analysis/gf2_probability_decomposition_design.md').write_text('# Synthetic fixture\n')
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'RUN',run)
    monkeypatch.setattr(report,'audit',lambda:dict(synthetic_fixture=True))
    matrix=np.zeros((18,18),np.int64);matrix[0,0]=5;matrix[0,17]=2;matrix[17,0]=3;matrix[17,17]=90
    confusion=matrix.tolist()
    rule=dict(counts=confusion,rescued=0,newly_empty=0,newly_wrong=0)
    temp=dict(counts=confusion,correct_to_empty=0,missed_to_correct=0,empty_to_foreground=0,
        foreground_to_empty=0,rank_changes=0,empty_probability_changes=0,hierarchical_changes=0,numerical_argmax_ties=0)
    records=[dict(index=i,foreground=7,empty=93,native_missed=2,cohort=2,rank_correct=7,
        saturated_zero=0,saturated_one=0,gap=dict(gap=0,gap_true_foreground=0,gap_true_empty=0,
        gap_correct_winner=0,gap_wrong_winner=0),rules={r:rule for r in report.RULES},
        temperatures={t:temp for t in ['0.5','0.75','1.0','1.5','2.0']}) for i in range(1024)]
    meta=[dict(index=i,scene='scene'+str(i//512),scene_frame=i%512) for i in range(1024)]
    def hist(events):
        result=np.zeros((2,NB),np.int64)
        for score,count in events:result[:,np.searchsorted(EDGES,score,side='right')]=512*count
        return result
    objects={}
    for name in report.MODELS:
        dest=run/'evaluation'/name;dest.mkdir(parents=True)
        objects[dest/'statistics_complete.json']=dict(status='complete')
        objects[dest/'statistics.json']=dict(records=records,scenes=['scene0','scene1'])
        objects[dest/'validation_frames.json']=meta
        objects[dest/'selection.json']=dict(risk=dict(ratio=1))
        objects[dest/'timing.json']=dict(mean_seconds=.1,fps=10,additional_risk_gate_seconds=None)
        for r in ['joint','hierarchical','joint_class','hierarchical_class','risk_class']:
            np.savez(dest/f'curves_{r}.npz',empty_hist=hist([(1,3),(-2,90)]),
                rescue_hist=hist([(-10,2)]),baseline_correct_hist=hist([(1,5)]),
                baseline_fp=np.array([1536,1536]),cohort_size=np.array([1024,1024]),
                scenes=np.array(['scene0','scene1']))
    original=report.read
    monkeypatch.setattr(report,'read',lambda p:objects[p] if p in objects else original(p))
    report.main()
    summary=json.loads((run/'summary.json').read_text())
    for difference in summary['cross_model'].values():
        assert difference['delta_pp']==0 and difference['ci']==[0,0]
    np.testing.assert_allclose(summary['baseline']['rules']['joint']['mIoU']['value'],100*.5/17)
    np.testing.assert_allclose(summary['baseline']['rules']['joint']['foreground_final_accuracy']['value'],100*5/7)
    with (run/'performance.csv').open() as stream:assert len(list(csv.DictReader(stream)))==12
    assert (run/'figures/overview.png').exists()
    assert '三项诊断均已完成' in (tmp_path/'analysis/gf2_probability_decomposition_results.md').read_text()
