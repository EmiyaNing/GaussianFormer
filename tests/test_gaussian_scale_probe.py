"""Nontrivial algebra/gradient checks for the scale-supervision probe."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def test_scale_jacobian_and_mass_controls_are_distinct():
    raw=torch.tensor([0.,4.,8.],dtype=torch.float64,requires_grad=True)
    sigma=.08+.32*raw.sigmoid()
    grad=torch.autograd.grad(sigma.sum(),raw)[0]
    torch.testing.assert_close(grad,(sigma-.08)*(.4-sigma)/.32)
    assert grad[-1]<grad[0]/100
    # Scale changes can redistribute evidence even after matching its sum.
    old=torch.tensor([.2,1.,.2]);new=torch.tensor([.05,1.,.05])
    ratio=new.sum()/old.sum()
    assert torch.isclose((old*ratio).sum(),new.sum())
    assert not torch.allclose(old*ratio,new)


def test_fp32_saturation_can_block_a_nonzero_empty_ce_scale_gradient():
    raw=torch.tensor(30.,dtype=torch.float32,requires_grad=True)
    sigma=.08+.32*raw.sigmoid()
    foreground=.95*4.*torch.exp(-.5*(.2/sigma).square())
    logits=torch.stack((foreground,foreground.new_tensor(1.)))
    loss=F.cross_entropy(logits[None],torch.tensor([1]))
    physical,parameter=torch.autograd.grad(loss,(sigma,raw))
    assert foreground>logits[1]  # This is an actual false positive.
    assert physical>0           # Empty CE requests a smaller physical scale.
    assert parameter==0         # Exact FP32 sigmoid, not a small-gradient cutoff.


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_center_subgroups_preserve_the_exclusive_census():
    from tools.probe_gaussian_scale_supervision import (
        accumulator,add_grouped,center_support_flags,FIELDS,FLAGS,SUBGROUP_FLAGS)
    device='cuda'
    center=torch.tensor([[0,0,0],[1,0,0],[2,0,0]],device=device,dtype=torch.int32)
    labels=torch.tensor([17,1,17],device=device)
    camera=torch.tensor([True,False,False],device=device)
    bits=torch.tensor([[1<<17,(1<<17)|2,1<<17],[1<<17]*3],device=device)
    values=center_support_flags(center,(3,1,1),labels,camera,bits)
    assert values[3].tolist()==[True,False,False]
    assert values[4].tolist()==[False,True,False]
    assert values[5].tolist()==[False,False,True]
    group=torch.full((3,),4,device=device,dtype=torch.long)
    flags=torch.zeros((3,len(FLAGS)),device=device,dtype=torch.bool)
    for name,value in zip(('center_camera','center_foreground','center_empty',*SUBGROUP_FLAGS,
                           'all_support_has_foreground'),values):
        flags[:,FLAGS.index(name)]=value
    a=accumulator();m=torch.ones((3,len(FIELDS)),device=device);s=torch.full((3,3),.4,device=device)
    add_grouped(a,group,m,flags,s)
    assert a['counts'].tolist()==[0,0,0,0,3,0,3,1,1,1,2]
    assert a['counts'][:6].sum()==a['counts'][6]
    # Extra subsets overlap; they must never be included again in the total.
    for k in ('sums','abs_sums','positive','negative','hist','coverage_hist','axis_hist','saturation_hist','flags'):
        a[k][6]=a[k][:6].sum(0)
    torch.testing.assert_close(a['hist'].sum(2),a['counts'][:,None].expand(-1,len(FIELDS)))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_probe_matches_independent_dense_autograd():
    from tools.gaussian_scale_probe import extension
    device='cuda';shape=(6,5,4);step=.4
    xyz=(torch.stack(torch.meshgrid(*(torch.arange(n,device=device) for n in shape),indexing='ij'),-1).reshape(-1,3)+.5)*step
    means=xyz[[1,36,80]].clone().requires_grad_(True)
    scales=torch.tensor([[.31,.23,.17],[.26,.37,.19],[.34,.22,.3]],device=device,requires_grad=True)
    alpha=torch.tensor([.7,.9,.8],device=device,requires_grad=True)
    raw=torch.linspace(-2,1,51,device=device).reshape(3,17).clone().requires_grad_(True)
    semantic=F.softplus(raw);rotation=torch.eye(3,device=device).expand(3,-1,-1).contiguous()
    inverse=torch.zeros((3,6),device=device);inverse[:,:3]=scales.detach().square().reciprocal()
    centers=(means.detach()/step).int().contiguous();radii=torch.ceil(scales.detach().max(1).values*3/step).int()
    labels=torch.arange(len(xyz),device=device).remainder(3).int();labels[::2]=17
    camera=torch.arange(len(xyz),device=device).remainder(4)!=0
    prediction=torch.zeros(len(xyz),device=device,dtype=torch.int32);winner=prediction.clone()
    assigned=torch.tensor([1,2,0],device=device,dtype=torch.int32)
    rng=torch.Generator(device=device).manual_seed(928)
    ce=torch.rand((len(xyz),17),generator=rng,device=device)-.5
    lov=torch.rand((len(xyz),17),generator=rng,device=device)-.5
    ce[~camera]=0;lov[~camera]=0;lov[labels==17]=0
    sigmoid=raw.detach().sigmoid().gather(1,assigned.long()[:,None]).flatten().contiguous()
    actual=extension().probe(means.detach().contiguous(),scales.detach().contiguous(),inverse.contiguous(),rotation,
        alpha.detach(),semantic.detach().contiguous(),centers,radii,xyz.contiguous(),labels,camera,
        prediction,winner,assigned,sigmoid,ce,lov,*shape)
    cell=(xyz/step).int()
    inside=((cell[None]-centers[:,None]).abs()<=radii[:,None,None]).all(2)
    kernel=torch.exp(-.5*((means[:,None]-xyz[None])/scales[:,None]).square().sum(2))*inside
    scores=(alpha[:,None,None]*kernel[:,:,None]*semantic[:,None]).sum(0)
    for s in range(3):
        weight=ce.clone() if s<2 else lov
        if s<2:weight=weight*(labels==17 if s==0 else labels<17)[:,None]
        truth=torch.autograd.grad((scores*weight).sum(),(means,scales,alpha,raw),retain_graph=True)
        a=actual[1][:,10:].reshape(3,3,9)[:,s]
        torch.testing.assert_close(a[:,0],truth[2],atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(a[:,1:4],truth[1],atol=2e-5,rtol=1e-4)
        torch.testing.assert_close(a[:,4:7],truth[0],atol=2e-5,rtol=1e-4)
        torch.testing.assert_close(a[:,7],truth[3].gather(1,assigned.long()[:,None]).flatten(),atol=1e-5,rtol=1e-4)
