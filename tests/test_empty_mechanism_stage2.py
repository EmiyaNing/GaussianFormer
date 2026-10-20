"""Checks that protect causal interpretation rather than mirror report formatting."""
import torch
import torch.nn.functional as F
from tools.empty_mechanism_stage2 import ce_parts, project_set


def test_ce_sources_sum_to_original_value_and_gradient_with_unequal_weights():
    torch.manual_seed(928)
    scores=torch.randn(23,18,dtype=torch.double,requires_grad=True)
    labels=torch.tensor([17]*15+[0,1,3,3,8,9,14,16])
    weights=torch.linspace(.5,2,18,dtype=torch.double)
    a,b=ce_parts(scores,labels,weights,10.)
    original=10*F.cross_entropy(scores,labels,weight=weights)
    torch.testing.assert_close(a+b,original)
    g=torch.autograd.grad(a+b,scores,retain_graph=True)[0]
    ref=torch.autograd.grad(original,scores,retain_graph=True)[0]
    torch.testing.assert_close(g,ref)
    empty_grad=torch.autograd.grad(b,scores)[0]
    assert (empty_grad[labels==17,:17]>0).all()
    assert empty_grad[labels<17].count_nonzero()==0


def test_projection_eliminates_amplitude_derivative_but_keeps_descent_direction():
    torch.manual_seed(928)
    u=torch.randn(1,11,17,dtype=torch.double)
    g=torch.randn_like(u)
    projected,error=project_set(u,g)
    assert error<1e-14
    assert torch.all((projected*g).sum(-1)>=-1e-14)
    eps=1e-5
    finite=(F.softplus(u-eps*projected).sum(-1)-F.softplus(u+eps*projected).sum(-1))/(2*eps)
    torch.testing.assert_close(finite,torch.zeros_like(finite),atol=1e-8,rtol=0)


def test_native_adjoint_recovers_gaussian_spatial_empty_fraction():
    from local_aggregate import LocalAggregator
    if not torch.cuda.is_available():
        return
    agg=LocalAggregator(scale_multiplier=3,H=16,W=16,D=16,pc_min=[0,0,0],grid_size=.25).cuda()
    means=torch.tensor([[[1.2,1.2,1.2],[1.4,1.1,1.3]]],device="cuda")
    cells=torch.tensor([[4,4,4],[5,4,4],[4,5,4],[4,4,5]],device="cuda")
    xyz=((cells.float()+.5)*.25)[None].contiguous()
    alpha=torch.tensor([[.3,.8]],device="cuda")
    scales=torch.full_like(means,.5)
    inv=torch.eye(3,device="cuda")[None,None].repeat(1,2,1,1)*4
    sem=torch.zeros(1,2,18,device="cuda",requires_grad=True)
    response=agg(xyz,means,alpha,sem,scales,inv)[:,0]
    total=torch.autograd.grad(response.sum(),sem,retain_graph=True)[0][0,:,0]
    empty=torch.autograd.grad(response[:2].sum(),sem)[0][0,:,0]
    w=alpha[0][None]*torch.exp(-2*(xyz[0,:,None]-means[0,None]).square().sum(-1))
    torch.testing.assert_close(total,w.sum(0),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(empty/total,w[:2].sum(0)/w.sum(0),atol=2e-6,rtol=2e-6)
