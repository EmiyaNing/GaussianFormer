import numpy as np
import torch
from tools.sparseworld_region_ablation_common import windows,window,patch_mask,shifted_mask


def test_windows_fixed_budget_and_temporal_contract():
    for h in (2,4,6):
        w=windows(h)
        assert all(len(v)==5 and len(set(v))==5 for v in w.values())
        assert max(w['H'])==0 and max(w['P'])==h
        assert set(w['M'])<=set(w['H']+w['P'])
        assert w['M'][0]==h and w['M'][-1]==-4


def test_window_projection_respects_source_coordinate_frames():
    transforms=torch.eye(4).repeat(1,6,1,1)
    transforms[0,:,0,3]=torch.arange(1,7.)
    # A common world camera at each time: future-native projection includes that time's ego translation.
    projection=torch.eye(4).repeat(1,78,1,1)
    for h in range(1,7):projection[:,6*(6+h):6*(7+h),0,3]=h
    data=dict(projection_mat=projection,image_wh=torch.ones(1,78,2)*32,future_ego_to_current=transforms)
    images=torch.arange(78.).reshape(1,78,1,1,1)
    for relative in windows(6).values():
        _,metas=window(data,images,relative,6)
        expected=torch.eye(4);expected[0,3]=6
        torch.testing.assert_close(metas['projection_mat'],expected.repeat(1,30,1,1))
    assert projection[0,0,0,3]==0  # metadata is not mutated


def test_target_mask_and_random_controls_have_identical_per_view_area():
    points=np.array([[8.,8.,1.],[-2.,4.,1.],[1.,1.,-1.]])
    projection=np.eye(4)[None];wh=np.array([[32,32]])
    mask=patch_mask(points,projection,wh,32,32,radius=2)
    assert mask.sum()==25
    random=shifted_mask(mask,20261008)
    np.testing.assert_array_equal(mask.sum((1,2)),random.sum((1,2)))
    assert not np.array_equal(mask,random)
