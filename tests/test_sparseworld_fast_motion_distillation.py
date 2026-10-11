"""Behavior contracts for fast-only occupancy distillation."""
from types import SimpleNamespace
import numpy as np
import torch
from tools.sparseworld_fast_motion_common import MotionIndex,occupancy_kd,voxel_scores
from tools.build_sparseworld_region_attributes import speed


def test_motion_mask_respects_class_overlap_and_speed():
    index=MotionIndex.__new__(MotionIndex)
    index.centers=np.array([[0.,0.,0.],[.3,0,0],[5,0,0],[10,0,0]],np.float32)
    def annotation(token,label,x,v):
        return dict(token=token,sample_token='b',instance_token=token,label=label,translation=[x,0,0],size=[1,1,1],rotation=[1,0,0,0],prev=token+'p',next=token+'n')
    a=annotation('a',4,0,3);b=annotation('z',7,5,0)
    index.by_sample={'b':[a,b]};index.annotations={}
    for record,v in ((a,3),(b,0)):
        index.annotations[record['token']]=record
        for key,t,dx in [('prev','p',-v),('next','n',v)]:
            index.annotations[record[key]]=dict(record,sample_token=t,translation=[record['translation'][0]+dx,0,0])
    index.times={'p':0,'b':1000000,'n':2000000}
    # Use the same implementation on a minimal geometry by inspecting reshape.
    labels=np.full(640000,17,np.uint8);labels[:4]=[4,7,7,4]
    centers=np.zeros((640000,3),np.float32);centers[:4]=index.centers;index.centers=centers
    bins=index.bins(labels,'b',np.eye(4)).reshape(-1)
    assert bins[:4].tolist()==[2,3,0,3]


def test_score_path_differentiable_without_kd_geometry_gradient():
    torch.set_num_threads(2)
    raster=SimpleNamespace(pc_range=(-40,-40,-1,40,40,5.4),grid_size=.4,grid_shape=(200,200,16),center_distance_threshold=3.,padding=False,threshold=torch.full((17,),.35))
    instance=SimpleNamespace(head=SimpleNamespace(rasterizer=raster))
    points=torch.tensor([[[.1,.1,.1]]*48],requires_grad=True)
    logits=torch.zeros((1,48,17),requires_grad=True)
    volume=voxel_scores(instance,points,logits)
    index=torch.floor((points.detach()[0,0]-torch.tensor([-40,-40,-1]))/.4).long()
    value=volume[index[0]*3200+index[1]*16+index[2],4]
    assert torch.allclose(value,torch.tensor(.5))
    value.backward()
    assert logits.grad.abs().sum()>0
    assert points.grad is None


def test_kd_masks_only_fast_voxels_and_keeps_teacher_free(monkeypatch):
    import tools.sparseworld_fast_motion_common as common
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda':
        import pytest
        pytest.skip('KD runner uses CUDA tensors')
    scores=torch.full((4,17),.3,device=device,requires_grad=True)
    monkeypatch.setattr(common,'voxel_scores',lambda *args:scores)
    out=dict(points=[None]*6,logits=[scores]*6)
    targets=dict(teacher=np.array([[4,17,2,2]]*3,np.uint8),fast_0=np.array([0,1]),fast_1=np.array([],np.int32),fast_2=np.array([],np.int32))
    value,detail=common.occupancy_kd(None,out,targets)
    expected=torch.nn.functional.binary_cross_entropy(scores[:2],torch.tensor([[0,0,0,0,1]+[0]*12,[0]*17],device=device,dtype=torch.float32),reduction='none').sum(-1).mean()/3
    torch.testing.assert_close(value,expected)
    value.backward()
    assert scores.grad[2:].abs().sum()==0
    assert scores.grad[0,4]<0 and scores.grad[1].min()>0
    assert detail[0]['teacher_free']==1 and detail[1]['voxels']==0
