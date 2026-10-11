import json
import numpy as np
import pytest

from tools.future_distillation_confirmation_results import (
    load_counts, metrics_from_counts, mean_metrics, final_rows,
    paired_scene_bootstrap, write_report)
from tools.experiment_future_distillation_followup import canonical_manifest
from tools.validate_future_branch_dataset import validate


def counts(scale=1):
    hist = np.broadcast_to(np.eye(18)*10+np.ones((18,18)), (4,3,18,18)).copy()*scale
    return dict(hist=hist, intersection=np.full((4,3),10*scale), union=np.full((4,3),20*scale))


def test_manifest_resume_compares_json_semantics_without_ignoring_changes():
    raw=dict(aug=dict(scale=(.3,.5)),horizons={2:'1s'},epochs=5)
    saved=json.loads(json.dumps(raw))
    assert canonical_manifest(raw)==saved
    assert canonical_manifest(dict(raw,epochs=4))!=saved


def test_missing_branch_source_does_not_count_as_u2_experiment():
    result=validate(None)
    assert result['status']=='blocked_no_source'
    assert result['branches']==0 and not result['actual_u2_experiment']


def branch_contract_fixture(tmp_path):
    # Mechanical validator fixture only: not images, physical simulation,
    # teacher inference, or research evidence.
    for name in ['history','action','fork','future']:
        (tmp_path/name).write_bytes(name.encode())
    anchors=np.array([[0.,0.,0.],[1.,0.,0.]])
    np.save(tmp_path/'anchors.npy',anchors)
    branches=[]
    for i in range(2):
        np.savez(tmp_path/f'branch{i}.npz',world_anchors=anchors,
                 gt_labels=np.full((3,2),i,dtype=np.int64),valid=np.ones((3,2),dtype=bool),
                 teacher_features=np.zeros((3,2,256)),teacher_repeats=np.zeros((5,3,2,256)),
                 future_ego_poses=np.broadcast_to(np.eye(4),(3,4,4)))
        branches.append(dict(id=str(i),student_history_files=['history'],ego_action_file='action',
                             fork_state_file='fork',teacher_checkpoint_sha256='fixture-only',
                             future_image_files={str(h):['future'] for h in (1,2,3)},
                             anchor_targets_file=f'branch{i}.npz'))
    root=dict(id='root',scene_family='scene-a',split='train',student_history_files=['history'],
              ego_action_file='action',fork_state_file='fork',world_anchors_file='anchors.npy',branches=branches)
    record=dict(schema=1,horizons=[1,2,3],environment_provenance='unit-test fixture, NOT scene evidence',
                sensor_configuration='fixture',teacher_checkpoint_sha256='fixture-only',roots=[root])
    return record


def test_valid_branch_contract_is_not_automatically_causal_evidence(tmp_path):
    record=branch_contract_fixture(tmp_path)
    source=tmp_path/'manifest.json';source.write_text(json.dumps(record))
    result=validate(source)
    assert result['status']=='mechanical_contract_passed'
    assert result['roots'][0]['gt_disagreement_anchors']==6
    assert not result['actual_u2_experiment']


def test_branch_contract_rejects_different_history(tmp_path):
    record=branch_contract_fixture(tmp_path)
    (tmp_path/'other-history').write_bytes(b'different')
    record['roots'][0]['branches'][1]['student_history_files']=['other-history']
    source=tmp_path/'manifest.json';source.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Different student history'):
        validate(source)


def test_branch_contract_rejects_root_family_split_leakage(tmp_path):
    record=branch_contract_fixture(tmp_path)
    second=dict(record['roots'][0],id='second',split='confirmation')
    record['roots'].append(second)
    source=tmp_path/'manifest.json';source.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='crosses data splits'):
        validate(source)


def test_native_scores_match_counts_and_not_feature_readout():
    row = metrics_from_counts(counts())
    assert row['metrics']['1s']['mIoU'] == pytest.approx(100*11/45)
    assert row['metrics']['3s']['OccIoU'] == 50
    assert set(row['groups']['1s']) == {'dynamic','static'}


def test_invalid_occupancy_counts_are_rejected(tmp_path):
    row = counts();row['intersection'][0,0]=1000
    np.savez(tmp_path/'counts.npz',**row)
    with pytest.raises(ValueError, match='exceeds union'):
        load_counts(tmp_path/'counts.npz',4)


def test_seed_mean_is_mean_of_metrics_not_confusion_ensemble():
    a,b=metrics_from_counts(counts()),metrics_from_counts(counts(100))
    b['metrics']['1s']['mIoU']=90
    result=mean_metrics([a,b])
    assert result['metrics']['1s']['mIoU'] == (a['metrics']['1s']['mIoU']+90)/2


def test_identical_paired_seeds_and_scenes_produce_zero_interval():
    values={1:counts(),2:counts(2),3:counts(3)}
    result=paired_scene_bootstrap(values,values,['a','a','b','c'],repetitions=20)
    assert result['mean_delta_pp']==0
    assert result['nested_seed_scene_ci95']['mean_1_2_3s']==[0,0]
    assert result['scenes']==3 and result['seeds']==[1,2,3]
    with pytest.raises(ValueError,match='seed identities'):
        paired_scene_bootstrap(values,{1:counts()},['a','a','b','c'],20)


def test_report_does_not_turn_one_finished_seed_into_three(tmp_path):
    manifest=dict(smoke=True,seeds=[1,2,3],supervisions=['task','teacher','shuffled'],
                  epochs=5,updates=20,train=dict(indices=[1],scenes=['train']),
                  val=dict(indices=[2,3,4,5],scenes=['a','a','b','c']))
    baseline=metrics_from_counts(counts());baseline['rows']='not_used'
    report=dict(status='running',baseline=baseline,arms={'teacher_seed1':
                dict(status='complete',epochs={'5':baseline},progress=20)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'results.json').write_text(json.dumps(report))
    assert len(final_rows(report,manifest)['teacher'])==1
    write_report(tmp_path)
    value=(tmp_path/'reports/results.md').read_text()
    assert '全部种子指标均值' not in value
    assert '没有新探针' in value
    assert '尚无完成最终 epoch 的种子' in value
