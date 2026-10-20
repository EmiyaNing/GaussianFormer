import torch
import torch.nn.functional as F
from loss.amplitude_projected_set_loss import (
    AmplitudeProjectedSetLoss, _ClassificationAmplitudeProjection,
)
from loss.center_support_diagnostic_loss import CenterSupportDiagnosticSetLoss


def test_occupancy_gradient_bypasses_the_classification_projection():
    torch.manual_seed(928)
    u=torch.randn(1,4,17,dtype=torch.double,requires_grad=True)
    target=torch.zeros_like(u);target[:,:,3]=1
    native_cls=F.binary_cross_entropy_with_logits(u,target)
    field=F.softplus(u).square().mean()
    gc=torch.autograd.grad(native_cls,u,retain_graph=True)[0]
    gf=torch.autograd.grad(field,u,retain_graph=True)[0]
    d=u.detach().sigmoid()
    projected=gc-d*(d*gc).sum(-1,keepdim=True)/d.square().sum(-1,keepdim=True)
    actual=F.binary_cross_entropy_with_logits(_ClassificationAmplitudeProjection.apply(u),target)+field
    torch.testing.assert_close(actual,native_cls+field)
    g=torch.autograd.grad(actual,u)[0]
    torch.testing.assert_close(g,gf+projected)
    torch.testing.assert_close((d*(g-gf)).sum(-1),torch.zeros_like(d[:,:,0]),atol=1e-15,rtol=0)


def test_original_assignment_geometry_earlier_classification_and_value_are_preserved():
    torch.manual_seed(928)
    kwargs=dict(stage_weights=[1.,1.],class_weights=[1.]*17,gt_domain="all",final_gt_domain="all")
    original=CenterSupportDiagnosticSetLoss(**kwargs)
    modified=AmplitudeProjectedSetLoss(**kwargs)
    logits=[torch.randn(1,3,17,requires_grad=True) for _ in range(2)]
    points=[torch.tensor([[[0.,0.,0.],[1.,.2,0.],[2.,0.,0.]]],requires_grad=True) for _ in range(2)]
    data=dict(opus_pred_points=points,opus_pred_logits=logits,
        opus_point_valid_masks=[torch.ones(1,3,dtype=torch.bool)]*2,
        metas=dict(occ_xyz=torch.tensor([[[.1,0.,0.],[1.1,0.,0.],[2.1,0.,0.]]]),
                   occ_label=torch.tensor([[1,5,8]]),occ_cam_mask=torch.ones(1,3,dtype=torch.bool)))
    a,_=original.components(data);b,_=modified.components(data)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    for stage in [0,1]:
        ga=torch.autograd.grad(a[f"stage_{stage}/classification"],logits[stage],retain_graph=True)[0]
        gb=torch.autograd.grad(b[f"stage_{stage}/classification"],logits[stage],retain_graph=True)[0]
        if stage==0:torch.testing.assert_close(ga,gb,atol=0,rtol=0)
        else:
            d=logits[stage].detach().sigmoid()
            rounding_bound=8*torch.finfo(gb.dtype).eps*(d*gb).abs().sum(-1)
            assert torch.all((d*gb).sum(-1).abs()<=rounding_bound)
    for stage in [0,1]:
        geometry=f"stage_{stage}/pred_to_gt"
        ga=torch.autograd.grad(a[geometry],points[stage],retain_graph=True)[0]
        gb=torch.autograd.grad(b[geometry],points[stage],retain_graph=True)[0]
        torch.testing.assert_close(ga,gb,atol=0,rtol=0)
