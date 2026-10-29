"""Independent 3D assignments/gradients and the detached true-render branch."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from loss.bidirectional_union_coverage_loss import BidirectionalUnionCoverageLoss,negative_log_shares,geometry_render,rho
from loss.semantic_gaussian_field.geometry import solid_ellipsoid_distance
from model.utils.utils import get_rotation_matrix
from tools.gaussian_geometry_causal import write

RUN=ROOT/'work_dirs/bidirectional_coverage_mini_v1'

def reference(means,scales,rotation,neglogp,points,classes,h):
    n,g=len(points),len(means)
    distance=solid_ellipsoid_distance(points[:,None].expand(n,g,3),means[None].expand(n,g,3),
        scales[None].expand(n,g,3),rotation[None].expand(n,g,3,3))
    values=rho(distance/h)+neglogp[:,classes].T
    return values.min(1)

def main():
    torch.set_num_threads(4);torch.manual_seed(928)
    device='cuda';g=34
    means=(torch.randn(g,3,device=device)*.35+torch.tensor([0.,0.,1.],device=device)).requires_grad_()
    scales=(.1+.3*torch.rand(g,3,device=device)).requires_grad_()
    quat=torch.randn(g,4,device=device,requires_grad=True);rotation=get_rotation_matrix(quat)
    raw=(torch.randn(g,17,device=device)*.4+F.one_hot(torch.arange(g,device=device)%17,17)*3.).requires_grad_()
    p=negative_log_shares(raw)
    xyz=torch.randn(17,3,device=device)*.7+torch.tensor([0.,0.,1.],device=device)
    labels=torch.arange(17,device=device);rev=xyz[:3]+.13
    loss=BidirectionalUnionCoverageLoss(grid_shape=(20,20,20),pc_range=(-4.,-4.,-4.,4.,4.,4.))
    values,info=loss.costs(means,scales,rotation,p,xyz,labels,rev)
    expected,ids=reference(means,scales,rotation,p,info['points'],info['classes'],.4)
    torch.testing.assert_close(values,expected,atol=2e-5,rtol=2e-5)
    assert torch.equal(info['ids'],ids)
    variables=(means,scales,quat)
    actual_grads=torch.autograd.grad(values.sum(),variables,retain_graph=True)
    ref_grads=torch.autograd.grad(expected.sum(),variables,retain_graph=True)
    errors=[]
    for actual,ref in zip(actual_grads,ref_grads):
        torch.testing.assert_close(actual,ref,atol=3e-4,rtol=3e-4);errors.append(float((actual-ref).abs().max()))
    assert torch.autograd.grad(values.sum(),raw,allow_unused=True,retain_graph=True)[0] is None
    # Duplicating every GS preserves all support costs and chooses lowest IDs.
    duplicate,di=loss.costs(torch.cat((means,means)),torch.cat((scales,scales)),
        torch.cat((rotation,rotation)),torch.cat((p,p)),xyz,labels,rev)
    torch.testing.assert_close(values,duplicate,atol=1e-6,rtol=1e-6)
    assert torch.equal(info['ids'],di['ids'])
    amplitude=F.softplus(raw.detach())*100.
    inverse=amplitude+torch.log(-torch.expm1(-amplitude))
    torch.testing.assert_close(p,negative_log_shares(inverse),atol=2e-6,rtol=2e-6)
    # One solid core: saturated interior and nonlocal exterior attraction.
    cm=torch.tensor([[0.,0.,1.]],device=device,requires_grad=True)
    ss=torch.tensor([[.2,.3,.4]],device=device,requires_grad=True)
    rr=torch.eye(3,device=device)[None].requires_grad_()
    vv,_=loss.costs(cm,ss,rr,p[:1],cm.detach()+.01,labels[:1])
    ig=torch.autograd.grad(vv.sum(),(cm,ss,rr),retain_graph=True)
    assert all(torch.count_nonzero(a)==0 for a in ig)
    far=torch.tensor([[2.5,0.,1.]],device=device)
    vv,_=loss.costs(cm,ss,rr,p[:1],far,labels[:1])
    fg=torch.autograd.grad(vv.sum(),(cm,ss));assert fg[0][0,0]<0 and fg[1][0,0]<0
    # Actual additive renderer, including trainable empty scalar and bias.
    from mmengine import Config
    from mmengine.registry import MODELS
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    import model
    c=Config.fromfile(ROOT/'config/mini_semantic_field/small_baseline_1ep.py')
    head=MODELS.build(c.model.head).cuda()
    with torch.no_grad():head.empty_scalar.fill_(.2);head.empty_bias.zero_()
    opacity=torch.full((1,g,1),.7,device=device,requires_grad=True)
    semantic=F.softplus(raw)[None]
    gaussian=GaussianPrediction(means[None],scales[None],quat[None],opacity,semantic)
    actual=geometry_render(head,gaussian,xyz)
    args=head.prepare_gaussian_args(gaussian)
    original=head._render_batched(xyz[None],*args)[0].T
    original=torch.cat((original[:,:17],original[:,17:]-head.empty_bias),-1)
    torch.testing.assert_close(actual,original,atol=0,rtol=0)
    target=torch.full((len(xyz),),17,device=device)
    wrong=actual[:,:17].max(-1).values;decision=(wrong-actual[:,17]).relu().mean()
    grads=torch.autograd.grad(decision,(means,scales,quat,raw,opacity,head.empty_scalar,head.empty_bias),allow_unused=True)
    assert all(torch.isfinite(a).all() for a in grads if a is not None)
    assert all(a is not None and a.norm()>0 for a in grads[:3])
    assert all(a is None for a in grads[3:])
    result=dict(status='passed',all17_semantic_core_assignments_exact=True,
        distance_value_max_error=float((values-expected).abs().max()),geometry_gradient_max_errors=errors,
        duplicate_cost_and_lowest_index_tie_invariant=True,amplitude_invariant_log_shares=True,
        covered_core_zero_geometry_gradient=True,far_hole_keeps_geometry_gradient=True,
        true_renderer_values_exact=True,decision_geometry_gradients_finite_nonzero=True,
        no_direct_semantic_opacity_or_empty_parameter_gradient=True)
    write(RUN/'numerical_checks.json',result);print(json.dumps(result))

if __name__=='__main__':main()
