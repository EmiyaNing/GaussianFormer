import numpy as np
import torch
from tools.sparseworld_full_regions_common import future_window
from tools.sparseworld_motion_geometry import rotation,overlap,rigid_probe,extent_probe,decomposition,corrected_labels_at_truth


def cloud():
    return np.stack(np.meshgrid(np.arange(10)*.4-1.8,np.arange(4)*.4-.6,np.arange(3)*.4+.2,indexing='ij'),-1).reshape(-1,3)


def test_four_history_future_windows_use_real_frame_slots():
    imgs=torch.arange(66).reshape(1,66,1,1,1)
    data=dict(projection_mat=torch.eye(4).reshape(1,1,4,4).repeat(1,66,1,1),image_wh=torch.ones(1,66,2),
              future_ego_to_current=torch.eye(4).reshape(1,1,4,4).repeat(1,6,1,1))
    for h,expected in [(2,[6,5,0,1,2]),(4,[8,7,6,5,0]),(6,[10,9,8,7,6])]:
        selected,meta,frames=future_window(data,imgs,h)
        assert frames==expected
        assert selected.flatten().tolist()==[6*f+c for f in expected for c in range(6)]
        torch.testing.assert_close(meta['projection_mat'],data['projection_mat'][:,:30],rtol=0,atol=0)


def test_rigid_oracle_recovers_known_motion_and_stores_same_scored_cloud():
    truth=cloud();pred=truth@rotation(np.deg2rad(30)).T+np.array([4.,-2.,0.])
    out=rigid_probe(pred,truth,np.eye(4))
    assert out['base']['iou']<.1
    assert out['rigid']['iou']>.8
    assert out['rigid']['iou']>=out['translation']['iou'] and out['rigid']['iou']>=out['yaw']['iou']
    assert overlap(out['_points']['rigid'],truth,np.eye(4))['iou']==out['rigid']['iou']


def test_extent_and_rigid_components_conserve_error_without_calling_residual_shape():
    truth=cloud();pred=truth*np.array([.6,1.,1.])+np.array([3.2,0.,0.])
    r=rigid_probe(pred,truth,np.eye(4));s=extent_probe(pred,truth,np.eye(4),np.eye(3),r);d=decomposition(r,s)
    assert s['rigid_size']['iou']>=max(s['size']['iou'],r['rigid']['iou'])
    assert d['pose']>=-1e-10 and d['extent']>=-1e-10
    assert abs(d['pose']+d['extent']+d['residual']-d['raw_error'])<1e-10
    assert abs(d['position']+d['orientation']-d['pose_only_gain'])<1e-10


def test_identity_voxel_relocation_preserves_original_semantics_and_empty_is_missing():
    truth=cloud();labels=np.full(len(truth),4,np.uint8)
    result,conflicts=corrected_labels_at_truth(truth,truth,labels,truth,np.eye(4),labels)
    assert np.array_equal(result,labels) and conflicts==0
    empty=np.empty((0,3));out=rigid_probe(empty,truth,np.eye(4))
    assert out['rigid']['iou']==0 and out['base']['iou']==0


def test_out_of_grid_false_positives_are_not_clipped_to_raise_iou():
    truth=cloud();pred=np.concatenate([truth,truth+np.array([80.,0.,0.])])
    assert overlap(pred,truth,np.eye(4))['iou']==.5


def test_query_shape_is_permutation_invariant_under_rigid_motion():
    from tools.sparseworld_query_geometry import fit_rigid
    a=np.concatenate([cloud(),cloud()+np.array([0.,0.,1.6])])
    b=a@rotation(.47).T+np.array([7.,-3.,.8])
    b=b[np.random.default_rng(13).permutation(len(b))]
    r=fit_rigid(a,b,limit=64)
    assert r['nonrigid_residual_squared']<1e-8
    assert r['pair_distance_change_m']<1e-8
    assert abs(r['raw_squared']-r['pose_removable_squared']-r['nonrigid_residual_squared'])<1e-9


def test_query_invariant_detects_size_change_and_native_drop_reasons_conserve_points():
    from tools.sparseworld_query_geometry import fit_rigid,native_point_validity
    a=cloud();b=a*np.array([2.,1.,1.]);r=fit_rigid(a,b)
    assert r['pair_distance_change_m']>.2 and r['nonrigid_residual_squared']>.01
    p=np.asarray([[[0,0,0],[20,0,0],[40,0,0]],[[80,0,0],[80.2,0,0],[79.8,0,0]]],np.float32)
    v=native_point_validity(p,np.full((2,3),4,np.uint8),np.full((2,3),-8,np.float32))
    bins=[~v['geometry'],v['geometry']&~v['inside'],v['geometry']&v['inside']&~v['confident'],v['kept']]
    assert sum(int(x.sum()) for x in bins)==p.shape[0]*p.shape[1]
