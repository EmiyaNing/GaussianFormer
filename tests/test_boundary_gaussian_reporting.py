"""Report aggregation keeps voxel/Gaussian denominators and scene completeness."""
import csv
import json
from copy import deepcopy
import numpy as np
import pytest

from tools.boundary_gaussian_stats.report import generate


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def sample_scene(split,token,frames,boundary,base_counts):
    # Count-based fixture; partitions are intentionally different-sized scenes.
    predicted=76800*frames
    gaussians=[dict(predicted=predicted,supported=predicted,boundary_any=boundary,
                    boundary_foreground=boundary,boundary_fg_empty=0,
                    class_count_histogram=[0,predicted-boundary,boundary]+[0]*16) for _ in range(4)]
    c=np.zeros((4,9,5),np.int64)
    c[:,0]=base_counts;c[:,1]=base_counts;c[:,2]=base_counts
    c[:,4]=base_counts;c[:,5]=base_counts
    rows=[]
    for i in range(frames):
        # Make the first frame carry all counts; others are valid empty records.
        rows.append(dict(error_counts=c.tolist() if i==0 else np.zeros_like(c).tolist()))
    return dict(status='complete',split=split,scene_token=token,scene_name='scene-'+token,
                indices=list(range(frames)),frames=frames,gaussian_counts=gaussians,
                error_counts=c.tolist(),records=rows,checkpoint_sha256='checkpoint-fixture')


def setup_fixture(tmp_path):
    folder=tmp_path/'experiment'
    dump(folder/'protocol.json',dict(expected=dict(train=dict(frames=1,scenes=1),val=dict(frames=2,scenes=1)),
        checkpoint='fixture.pth',checkpoint_sha256='checkpoint-fixture',config='fixture.py'))
    dump(folder/'availability.json',dict(train=dict(scene_frame_counts={'train':1}),val=dict(scene_frame_counts={'val':2})))
    dump(folder/'pilot.json',dict(empty_bias=-.6))
    train=sample_scene('train','train',1,76800,[8,2,0,0,10])
    val=sample_scene('val','val',2,0,[90,0,3,4,100])
    dump(folder/'scenes/train/train.json',train)
    dump(folder/'scenes/val/val.json',val)
    return folder


def test_report_global_is_count_weighted_and_scene_csv_is_not_duplicated(tmp_path):
    folder=setup_fixture(tmp_path)
    result=generate(folder,tmp_path/'report.md')
    assert result['status']=='complete'
    total=result['statistics']['total']
    assert total['gaussian_counts'][0]['boundary_any_fraction']==1/3
    assert total['scene_macro_boundary_fraction']['native_all']==.5
    e=total['errors']['native_all']['union_any']
    assert e['wrong_class']==2 and e['miss']==3 and e['false_positive']==4
    assert e['voxels']==217 and e['wrong_class_fraction']==2/217
    with (folder/'scene_statistics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==2 and {r['split'] for r in rows}=={'train','val'}
    assert float(rows[0]['all_boundary_any_fraction'])==1
    assert (folder/'complete.json').exists()


def test_report_rejects_scene_with_incomplete_frame_records(tmp_path):
    folder=setup_fixture(tmp_path)
    path=folder/'scenes/val/val.json';row=json.loads(path.read_text());row['records'].pop();dump(path,row)
    with pytest.raises(AssertionError):generate(folder,tmp_path/'report.md')
