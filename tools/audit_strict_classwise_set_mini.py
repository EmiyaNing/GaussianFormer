"""Final integrity audit of completed strict-classwise experiment artifacts."""
from pathlib import Path
import json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.strict_classwise_set_experiment import RUN,PREVIOUS_RUN,assert_protocol,sha,write


def main():
    p=assert_protocol()
    queue=json.loads((RUN/'queue_progress.json').read_text())
    assert queue['status']=='complete'
    train=json.loads((RUN/'train/strict_classwise/complete.json').read_text())
    evaluation=json.loads((RUN/'eval/strict_classwise/complete.json').read_text())
    summary=json.loads((RUN/'summary.json').read_text())
    previous=json.loads((PREVIOUS_RUN/'train/classwise/complete.json').read_text())
    assert train['frames']==train['unique_frames']==4096 and train['scenes']==698
    assert train['optimizer_updates']==2048 and train['epochs']==1
    assert train['data_order_sha256']==previous['data_order_sha256']
    assert evaluation['frames']==6019 and evaluation['scenes']==150
    assert sha(Path(train['checkpoint']))==train['checkpoint_sha256']==evaluation['checkpoint_sha256']
    def finite(value):
        if isinstance(value,float):assert math.isfinite(value)
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    finite(summary)
    for name in ('numerical_checks.json','pilot.json','smoke_eval/strict_classwise/smoke_passed.json'):
        assert json.loads((RUN/name).read_text())['status']=='passed'
        assert sha(RUN/name)==p['checks'][name]
    assert summary['optimization_audit']['data_order_exact']
    assert summary['optimization_audit']['first_geometry_and_occupancy_max_difference']==0
    frames=[];tokens=[]
    for path in sorted((RUN/'eval/strict_classwise/scenes').glob('*.json')):
        meta=json.loads(path.read_text());frames+=meta['indices'];tokens+=meta['tokens']
        assert meta['checkpoint_sha256']==evaluation['checkpoint_sha256']
    assert sorted(frames)==list(range(6019)) and len(set(tokens))==6019
    with np.load(RUN/'aggregate/strict_classwise.npz') as z:
        assert int(z['class_coverage'][:,0].sum())==121325057
        assert z['load'].shape==z['query_core'].shape==(6019,1200)
        for key in ('GS_class_count','initial_GS_class_count','baseline_GS_class_count'):
            assert z[key].sum()==6019*76800
        for name in ('initial','baseline'):
            matrix=z[name+'_to_current_classes']
            np.testing.assert_array_equal(matrix.sum(0),z['GS_class_count'])
            np.testing.assert_array_equal(matrix.sum(1),z[name+'_GS_class_count'])
    for name in ('baseline','classwise','strict_classwise'):
        rows=[r for r in summary['query_quintiles'] if r['arm']==name]
        assert len(rows)==5 and abs(sum(r['GT_share'] for r in rows)-100)<1e-9
    paths=[ROOT/'analysis/strict_classwise_set_mini_design.md',
           ROOT/'analysis/strict_classwise_set_mini_results.md',RUN/'summary.json',
           RUN/'optimization_audit.json',Path(__file__).resolve(),
           RUN/'interpretation_audit.json',ROOT/'tools/finalize_strict_classwise_set_report.py',
           ROOT/'analysis/figures/strict_classwise_set_mini_comparison.png',
           ROOT/'analysis/figures/strict_classwise_set_mini_comparison.pdf']
    paths+=sorted((RUN/'aggregate').glob('*.csv'))
    audit=dict(status='passed',checked=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        frozen_sources_verified=len(p['sources']),controls_verified=len(p['controls']),
        train_frames=4096,optimizer_updates=2048,val_frames=6019,val_scenes=150,
        GT_foreground=121325057,checkpoint_sha256=train['checkpoint_sha256'],
        matched_data_order=True,initial_geometry_and_occupancy_exact=True,
        all_summary_values_finite=True,quintile_denominators_exact=True,
        class_transition_marginals_exact=True,
        artifacts={str(path.relative_to(ROOT)):sha(path) for path in paths})
    write(RUN/'final_integrity_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='artifacts'}))

if __name__=='__main__':main()
