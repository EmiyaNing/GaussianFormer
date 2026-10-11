"""Regression checks for native gradient/renderer and safe occupancy targets."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from model.head.sparseworld_strict_head import _Rasterizer
from model.response_distillation_forecast import forecast_with_position_gradient
from tools.sparseworld_fast_motion_common import native_voxel_logits,occupancy_kd


def test_native_raster_parity_with_closing_filtering_and_compactness():
    torch.set_num_threads(2);torch.manual_seed(7)
    raster=_Rasterizer((-4,-4,-2,4,4,2),.4,(20,20,10),[.35]*15+[.25,.3],3.,True)
    instance=SimpleNamespace(head=SimpleNamespace(rasterizer=raster))
    centers=torch.tensor([[-2.,-2.,0.],[0.,0.,0.],[2.,2.,0.],[-3.,2.,0.]])
    points=(centers[:,None]+torch.randn(4,48,3)*1.1).reshape(1,-1,3).requires_grad_()
    z=torch.randn(1,192,17)*2-3;z[0,48:96,15]=2.;z[0,96:144,16]=2.;z[0,144:]=-10
    z.requires_grad_()
    stable=native_voxel_logits(instance,points,z);scores=stable.sigmoid()
    threshold=raster.threshold
    labels=torch.full((4000,),17,dtype=torch.long);occupied=(scores>threshold).any(-1)
    labels[occupied]=scores[occupied].argmax(-1)
    assert torch.equal(labels,raster(points.detach(),z.detach())[0])
    # Every retained row has evidence in all classes; missing rows are excluded.
    supported=torch.isfinite(stable).all(-1)
    assert supported.any() and (~supported).any()
    torch.nn.functional.binary_cross_entropy_with_logits(stable[supported],torch.zeros_like(stable[supported])).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert z.grad[0,144:].abs().sum()==0  # Native hard score cutoff.
    assert points.grad is None


def test_safe_target_selection_and_denominator(monkeypatch):
    import tools.sparseworld_fast_motion_common as common
    z=torch.full((4,17),-2.);z[2]=-torch.inf;z.requires_grad_()
    monkeypatch.setattr(common,'native_voxel_logits',lambda *args:z)
    # Raw network logits are finite; -inf only belongs to the raster output.
    out=dict(points=[None]*6,logits=[z[:1]]*6)
    # [positive supported, teacher miss, unsupported positive, outside mask]
    targets=dict(teacher=np.array([[4,17,4,4]]*3),fast_0=np.array([0,1,2]),fast_1=np.array([],int),fast_2=np.array([],int))
    loss,detail=occupancy_kd(None,out,targets,policy='positive_supported_native')
    target=torch.nn.functional.one_hot(torch.tensor([4]),17).float()
    expected=torch.nn.functional.binary_cross_entropy_with_logits(z[:1],target,reduction='sum')/9
    torch.testing.assert_close(loss,expected);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad[0,4]<0 and z.grad[0,0]>0
    assert z.grad[1:].abs().sum()==0
    assert detail[0]['retained']==1 and detail[0]['teacher_free']==1 and detail[0]['positive_unsupported']==1


def test_extreme_supported_logits_have_finite_nonzero_gradients():
    raster=_Rasterizer((-1,-1,-1,1,1,1),.4,(5,5,5),[.35]*17,3.,False)
    instance=SimpleNamespace(head=SimpleNamespace(rasterizer=raster))
    p=torch.zeros(1,48,3);z=torch.full((1,48,17),-100.);z[:,:,1]=100.;z.requires_grad_()
    scores=native_voxel_logits(instance,p,z)
    row=scores[torch.isfinite(scores).all(-1)]
    target=torch.nn.functional.one_hot(torch.tensor([4]),17).float()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(row,target,reduction='sum');loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad[:,:,1].sum()>.9 and z.grad[:,:,4].sum()<-.9


def test_all_ignored_horizons_still_allow_backward(monkeypatch):
    import tools.sparseworld_fast_motion_common as common
    z=torch.ones(1,17,requires_grad=True)
    monkeypatch.setattr(common,'native_voxel_logits',lambda *args:z)
    out=dict(points=[None]*6,logits=[z]*6)
    target=dict(teacher=np.full((3,1),17),**{f'fast_{j}':np.array([0]) for j in range(3)})
    loss,_=occupancy_kd(None,out,target,policy='positive_supported_native');loss.backward()
    assert loss==0 and torch.equal(z.grad,torch.zeros_like(z))


def test_rollout_detach_policy_preserves_forward_and_native_gradients():
    class Attention(nn.Module):
        def forward(self,ego,features,positions):return ego
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.future_steps=1;self.num_refines=48;self.ego_attention=Attention()
            self.position=nn.Linear(48*4,4);self.reg=nn.Linear(4,48*3)
            self.velocity=nn.Linear(4,48*2);self.semantic=nn.Linear(4,48*17);self.traj=nn.Linear(4,2)
        def _refine(self,points,offset):return points+offset
        def _world(self,points):return points*80-40
    torch.manual_seed(5);m=Tiny()
    encoded=dict(query_features=torch.randn(1,2,4),query_points=torch.rand(1,2,48,3))
    stamps=torch.tensor([0,1]);ego=torch.randn(1,1,4)
    a=forecast_with_position_gradient(m,encoded,stamps,ego,position_gradient=True)
    b=forecast_with_position_gradient(m,encoded,stamps,ego,position_gradient=False)
    for key in ('points','logits'):torch.testing.assert_close(a[key][0],b[key][0],rtol=0,atol=0)
    assert a['points'][0].requires_grad and not b['points'][0].requires_grad
    (b['points'][0].square().mean()+b['logits'][0].square().mean()+b['trajectories'].square().mean()).backward()
    assert m.reg.weight.grad is None and m.velocity.weight.grad is None
    assert m.semantic.weight.grad.abs().sum()>0
