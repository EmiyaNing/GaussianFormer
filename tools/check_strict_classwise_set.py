"""Independent classwise focal reference, finite differences and unchanged CD."""
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from loss.strict_classwise_set_loss import StrictClasswiseSetLoss
from loss.classwise_chamfer_set_loss import ClasswiseChamferSetLoss
from tools.strict_classwise_set_experiment import RUN,write


def reference(p,z,g,y,module):
    classes=z.detach().argmax(-1)
    numerator=z.sum()*0.;denominator=z.new_zeros(())
    lower=p.new_tensor(module.pc_range[:3]);upper=p.new_tensor(module.pc_range[3:])
    for c in range(17):
        pm=classes==c;gm=y==c
        if not pm.any() or not gm.any():continue
        dist=(p[pm,None]-g[gm][None]).square().sum(-1)
        target=g[gm][dist.detach().argmin(1)]
        spatial=(target[:,:2]-(lower+upper)[:2]/2).norm(dim=-1)/(upper-lower)[:2].norm()+1
        labels=F.one_hot(torch.full((int(pm.sum()),),c,device=z.device),17).to(z)
        prob=z[pm].sigmoid()
        pt=prob*labels+(1-prob)*(1-labels)
        alpha=module.focal_alpha*labels+(1-module.focal_alpha)*(1-labels)
        term=(alpha*(1-pt).pow(module.focal_gamma)*
              F.binary_cross_entropy_with_logits(z[pm],labels,reduction='none')).sum(-1)
        weight=module.geometry_class_weights[c].to(z)
        numerator=numerator+weight*(term*spatial).mean();denominator=denominator+weight
    return numerator/denominator.clamp_min(torch.finfo(z.dtype).tiny)


def main():
    torch.set_num_threads(4);torch.manual_seed(928)
    weights=[10,5,10,5,5,10,10,5,10,5,5,1,5,1,1,2,1]
    module=StrictClasswiseSetLoss(stage_weights=[1.],class_weights=weights,geometry_class_weights=weights)
    old=ClasswiseChamferSetLoss(stage_weights=[1.],class_weights=weights,geometry_class_weights=weights)
    cls=torch.cat([torch.full((1+c%3,),c) for c in range(17)])
    y=torch.cat([torch.full((2+c%4,),c) for c in range(17)])
    p=torch.randn(len(cls),3,dtype=torch.double,requires_grad=True)
    z=(F.one_hot(cls,17)*4.+torch.randn(len(cls),17)*.1).double().requires_grad_()
    g=torch.randn(len(y),3,dtype=torch.double);cam=torch.ones(len(y),dtype=torch.bool)
    global_labels=y[torch.cdist(p.detach(),g).argmin(1)]
    loss,rows=module.classification(p,z,g,y,global_labels)
    expected=reference(p,z,g,y,module)
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(torch.autograd.grad(loss,z,retain_graph=True)[0],
                               torch.autograd.grad(expected,z,retain_graph=True)[0])
    assert torch.autograd.grad(loss,p,allow_unused=True,retain_graph=True)[0] is None
    gradcheck=torch.autograd.gradcheck(lambda zz:module.classification(p,zz,g,y,global_labels)[0],
                                      (z,),eps=1e-5,atol=2e-5,rtol=2e-4)
    result=module._single(p,z,g,y,cam);previous=old._single(p,z,g,y,cam)
    for i in (1,2,3,4):torch.testing.assert_close(result[i],previous[i],atol=0,rtol=0)
    ng=torch.autograd.grad(result[1]+result[2],p,retain_graph=True)[0]
    og=torch.autograd.grad(previous[1]+previous[2],p,retain_graph=True)[0]
    torch.testing.assert_close(ng,og,atol=0,rtol=0)
    # Missing GT class: its predictions receive zero focal gradient.
    zz=z.detach().clone().requires_grad_();keep=y!=16
    sparse,_=module.classification(p,zz,g[keep],y[keep],global_labels)
    sparse.backward();assert torch.count_nonzero(zz.grad[cls==16])==0
    # Completely disjoint classes: differentiable zero, not a cross-class fallback.
    absent_z=F.one_hot(torch.full((3,),16),17).double().requires_grad_()
    absent,_=module.classification(p[:3],absent_z,g[y==0],y[y==0],global_labels[:3])
    assert absent.item()==0;absent.backward();assert torch.count_nonzero(absent_z.grad)==0
    # One positive target + all 16 negative classes, even when only one class exists.
    onez=torch.zeros((2,17),dtype=torch.double);onez[:,3]=2;onez.requires_grad_()
    onep=p[:2];oneg=g[:3];oney=torch.full((3,),3)
    one,_=module.classification(onep,onez,oneg,oney,torch.full((2,),3))
    onegrad=torch.autograd.grad(one,onez)[0]
    assert (onegrad[:,3]<0).all() and (onegrad[:,0]>0).all()
    # The outer class weight cancels for a single active class (no double weight).
    unit=StrictClasswiseSetLoss(geometry_class_weights=[.01]*17)
    actual,_=unit.classification(onep,onez,oneg,oney,torch.full((2,),3))
    torch.testing.assert_close(one,actual)
    cuda=StrictClasswiseSetLoss(geometry_class_weights=weights).cuda()
    cp=p.detach().float().cuda();cz=z.detach().float().cuda().requires_grad_()
    cg=g.float().cuda();cy=y.cuda();gl=global_labels.cuda()
    cl,_=cuda.classification(cp,cz,cg,cy,gl);cr=reference(cp,cz,cg,cy,cuda)
    torch.testing.assert_close(cl,cr)
    torch.testing.assert_close(torch.autograd.grad(cl,cz,retain_graph=True)[0],
                               torch.autograd.grad(cr,cz)[0],rtol=1e-5,atol=1e-6)
    out=dict(status='passed',all17_class_reference=True,logits_gradient_reference=True,
        logits_finite_difference=bool(gradcheck),both_geometry_values_and_gradients_exact=True,
        missing_GT_and_disjoint_classes_zero=True,all17_focal_outputs=True,
        no_double_class_weight=True,no_direct_classification_center_gradient=True,
        CUDA_reference_and_gradient=True)
    write(RUN/'numerical_checks.json',out);print(json.dumps(out))

if __name__=='__main__':main()
