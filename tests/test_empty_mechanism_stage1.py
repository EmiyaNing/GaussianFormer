"""Invariant and production-CUDA checks for mechanism interventions."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tools.empty_mechanism_stage1 import decompose,factorial_semantics,inverse_softplus
from tools.report_empty_mechanism_stage1 import arrays,statistics


def test_factorial_paths_preserve_the_factor_that_is_held_fixed():
    raw=torch.linspace(-18,18,3*17).reshape(1,3,17)
    before=F.softplus(raw);after=F.softplus(raw+torch.sin(raw)*.7)
    paths=factorial_semantics(before,after)
    a,p=decompose(before);b,q=decompose(after)
    amp,composition=decompose(paths['amplitude']);amp2,composition2=decompose(paths['composition'])
    torch.testing.assert_close(amp,b,atol=2e-6,rtol=2e-7)
    torch.testing.assert_close(composition,p,atol=1e-8,rtol=2e-7)
    torch.testing.assert_close(amp2,a,atol=2e-6,rtol=2e-7)
    torch.testing.assert_close(composition2,q,atol=1e-8,rtol=2e-7)
    torch.testing.assert_close(paths['sham'],before,atol=0,rtol=0)
    assert paths['full'] is after


def test_inverse_softplus_is_stable_over_experiment_dynamic_range():
    raw=torch.tensor([-60.,-30.,-12.,-1.,0.,1.,20.,60.])
    recovered=inverse_softplus(F.softplus(raw))
    torch.testing.assert_close(recovered,raw,atol=1e-6,rtol=1e-6)


def test_amplitude_path_can_change_aggregated_foreground_winner():
    before=torch.tensor([[[9.,1.],[1.,9.]]]);after=before*torch.tensor([[[.3],[1.]]])
    weights=torch.tensor([[[.6],[.4]]]);paths=factorial_semantics(before,after)
    assert (weights*before).sum(1).argmax(-1).item()==0
    assert (weights*paths['amplitude']).sum(1).argmax(-1).item()==1
    torch.testing.assert_close(paths['composition'],before)


def test_confusion_rates_do_not_count_semantic_errors_as_empty_errors():
    c=np.zeros((18,18));c[2,2]=3;c[2,4]=2;c[2,17]=5;c[17,3]=4;c[17,17]=6
    o=dict(counts=c,ce=1.,lovasz=.2,set_cls=.3,rescued=0,correct_to_empty=0,
        foreground_winner_changed=0,foreground_correct=5,
        cohorts=dict(recoverable=dict(n=5,supported=5,amplitude_sum=15,share_sum=2.5,margin_sum=-5),correct=dict(n=3)))
    s=statistics(arrays([o]),np.ones((1,1)))
    assert s['fnr'][0]==50 and s['fpr'][0]==40
    assert s['foreground_correct_percent'][0]==50
    assert s['recoverable_amplitude'][0]==3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native CUDA required')
def test_production_renderer_semantic_gradient_matches_analytic_chain_rule_and_fd():
    from local_aggregate import LocalAggregator
    device='cuda';torch.manual_seed(928)
    aggregator=LocalAggregator(scale_multiplier=3,H=16,W=16,D=16,pc_min=[0,0,0],grid_size=.25).to(device)
    means=(1.05+.3*torch.rand(1,4,3,device=device)).contiguous()
    # Native backward stores one query per grid cell, as in dense occupancy.
    # Multiple random queries in one cell would overwrite its voxel2pts entry.
    cell=torch.tensor([[4,4,4],[5,4,4],[4,5,4],[4,4,5],[5,5,4],[5,4,5]],device=device)
    xyz=((cell.float()+.5)*.25)[None].contiguous()
    scales=torch.full_like(means,.5);inverse=torch.eye(3,device=device)[None,None].repeat(1,4,1,1)*4
    alpha=torch.tensor([[.2,.4,.7,.9]],device=device)
    raw=torch.randn(1,4,17,device=device,requires_grad=True)
    upstream=torch.randn(6,18,device=device)
    def objective(u):
        sem=F.pad(F.softplus(u),(0,1)).contiguous()
        return (aggregator(xyz,means,alpha,sem,scales,inverse)*upstream).sum()
    actual=torch.autograd.grad(objective(raw),raw)[0]
    weight=alpha[0][None]*torch.exp(-2*(xyz[0,:,None]-means[0,None]).square().sum(-1))
    expected=(weight.T@upstream[:,:17])[None]*raw.sigmoid()
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    direction=torch.randn_like(raw);direction/=direction.norm();eps=.002
    with torch.no_grad():finite=(objective(raw+eps*direction)-objective(raw-eps*direction))/(2*eps)
    derivative=(actual*direction).sum()
    torch.testing.assert_close(finite,derivative,atol=5e-4,rtol=5e-3)
