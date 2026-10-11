"""Meaningful contracts: raw values retained and invalid API semantics exposed."""
from types import SimpleNamespace
import numpy as np
import torch
from tools.sparseworld_raw_logit_kd import raw_voxel_logits,elementwise_loss,raw_occupancy_kd

def test_raw_readout_preserves_negative_values_and_has_no_confidence_cutoff():
    r=SimpleNamespace(pc_range=(-1,-1,-1,1,1,1),grid_size=.4,grid_shape=(5,5,5),center_distance_threshold=3.)
    model=SimpleNamespace(head=SimpleNamespace(rasterizer=r))
    p=torch.zeros(1,48,3,requires_grad=True);z=torch.full((1,48,17),-20.);z[0,0,4]=-7.;z.requires_grad_()
    out=raw_voxel_logits(model,p,z);valid=torch.isfinite(out).all(-1)
    assert valid.sum()==1 and out[valid][0,4]==-7 and out[valid][0,0]==-20
    assert torch.isneginf(out[~valid]).all()
    out[valid].sum().backward();assert z.grad.abs().sum()>0 and p.grad is None

def test_literal_default_rejects_negative_teacher_by_producing_nan():
    s=torch.tensor([[-2.,1.]],requires_grad=True);t=torch.tensor([[-1.,.5]])
    assert not torch.isfinite(elementwise_loss(s,t,'raw_api_default')).all()

def test_literal_log_target_has_no_matching_stationary_point_and_no_lower_bound():
    t=torch.tensor([[-2.,1.]])
    s=t.clone().requires_grad_();loss=elementwise_loss(s,t,'raw_api_log_target').sum()
    assert loss==0;loss.backward();torch.testing.assert_close(s.grad,-t.exp())
    assert elementwise_loss(t+10,t,'raw_api_log_target').sum()<0
    torch.testing.assert_close(elementwise_loss(t+10,t,'raw_api_log_target').sum(),-10*t.exp().sum())

def test_gaussian_kl_is_half_squared_error_with_stationary_matching_target():
    s=torch.tensor([[-4.,2.]],requires_grad=True);t=torch.tensor([[-2.,-1.]],requires_grad=True)
    loss=elementwise_loss(s,t,'gaussian_raw',sigma=2.)
    torch.testing.assert_close(loss,.5*(s-t.detach()).square()/4)
    loss.sum().backward();assert t.grad is None
    torch.testing.assert_close(s.grad,(s.detach()-t.detach())/4)
    same=t.detach().clone().requires_grad_();equal=elementwise_loss(same,t,'gaussian_raw').sum();equal.backward()
    assert equal==0 and same.grad.abs().sum()==0

def test_mask_support_and_original_denominator(monkeypatch):
    import tools.sparseworld_raw_logit_kd as common
    z=torch.ones(4,17);z[1]=-torch.inf;z.requires_grad_()
    monkeypatch.setattr(common,'raw_voxel_logits',lambda *a:z)
    target=dict(teacher_raw_0=np.zeros((3,17),np.float32),teacher_raw_valid_0=np.array([True,True,False]),fast_0=np.array([0,1,2]))
    for j in (1,2):target.update({f'teacher_raw_{j}':np.zeros((0,17),np.float32),f'teacher_raw_valid_{j}':np.zeros(0,bool),f'fast_{j}':np.zeros(0,int)})
    out=dict(points=[None]*6,logits=[z[:1]]*6)
    value,detail=raw_occupancy_kd(None,out,target,'gaussian_raw')
    torch.testing.assert_close(value,torch.tensor(.5/9));value.backward()
    assert z.grad[0].abs().sum()>0 and z.grad[1:].abs().sum()==0
    assert detail[0]['retained']==1

def test_raw_path_does_not_call_sigmoid_softmax_or_log_softmax(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Probability activation entered raw-logit KD')
    for obj,name in [(torch,'sigmoid'),(torch,'softmax'),(torch.Tensor,'sigmoid'),(torch.Tensor,'softmax'),
                     (torch.nn.functional,'softmax'),(torch.nn.functional,'log_softmax')]:
        monkeypatch.setattr(obj,name,forbidden)
    r=SimpleNamespace(pc_range=(-1,-1,-1,1,1,1),grid_size=.4,grid_shape=(5,5,5),center_distance_threshold=3.)
    model=SimpleNamespace(head=SimpleNamespace(rasterizer=r))
    z=torch.full((1,48,17),-2.);p=torch.zeros(1,48,3)
    voxels=raw_voxel_logits(model,p,z);s=voxels[torch.isfinite(voxels).all(-1)]
    for policy in ('raw_api_default','raw_api_log_target','gaussian_raw'):elementwise_loss(s,s,policy)
