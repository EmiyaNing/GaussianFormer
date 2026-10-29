"""Independent finite-difference and all-pair checks of classwise OPUS CD."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from loss.classwise_chamfer_set_loss import ClasswiseChamferSetLoss
from loss.opus_set_loss import OPUSSetLoss
from tools.classwise_chamfer_experiment import RUN,write


def reference(p,z,g,y,cam,module):
    classes=z.detach().argmax(-1);ps=[];gs=[];ws=[]
    for c in range(17):
        a=p[classes==c];b=g[y==c]
        if not len(a) or not len(b):continue
        d=(a[:,None]-b[None]).square().sum(-1)
        pi=d.detach().argmin(1);gi=d.detach().argmin(0)
        weight=torch.ones(len(b),dtype=p.dtype,device=p.device)
        weight[(d.detach()[gi,torch.arange(len(b),device=p.device)].sqrt()>.2)&cam[y==c]]=5
        ps.append(F.smooth_l1_loss(a,b[pi],beta=.2,reduction='none').sum(-1).mean())
        gs.append((F.smooth_l1_loss(b,a[gi],beta=.2,reduction='none').sum(-1)*weight).mean())
        ws.append(module.geometry_class_weights[c].to(p))
    w=torch.stack(ws)
    return (torch.stack(ps)*w).sum()/w.sum(),(torch.stack(gs)*w).sum()/w.sum()


def main():
    torch.manual_seed(928)
    weights=[10,5,10,5,5,10,10,5,10,5,5,1,5,1,1,2,1]
    module=ClasswiseChamferSetLoss(stage_weights=[1.],geometry_class_weights=weights,class_weights=weights)
    old=OPUSSetLoss(stage_weights=[1.],class_weights=weights)
    # All 17 classes present with deliberately unequal cardinalities.
    pc=torch.cat([torch.full((1+c%4,),c) for c in range(17)])
    y=torch.cat([torch.full((2+c%3,),c) for c in range(17)])
    p=torch.randn(len(pc),3,dtype=torch.double,requires_grad=True)
    g=torch.randn(len(y),3,dtype=torch.double)
    z=(F.one_hot(pc,17)*4.+torch.randn(len(pc),17)*.1).double().requires_grad_()
    cam=torch.rand(len(y))>.2
    a,b,stats=module.geometry(p,z,g,y,cam);ra,rb=reference(p,z,g,y,cam,module)
    torch.testing.assert_close(a,ra);torch.testing.assert_close(b,rb)
    ga=torch.autograd.grad(a+b,p,retain_graph=True)[0];gr=torch.autograd.grad(ra+rb,p,retain_graph=True)[0]
    torch.testing.assert_close(ga,gr)
    assert torch.autograd.grad(a+b,z,allow_unused=True)[0] is None
    gradcheck=torch.autograd.gradcheck(lambda v:sum(module.geometry(v,z,g,y,cam)[:2]),(p,),eps=1e-5,atol=2e-5,rtol=2e-4)
    # Original focal stays bit-for-bit identical even when geometric matching differs.
    cls,_,_,pd,gd=module._single(p,z,g,y,cam);original=old._single(p,z,g,y,cam)
    torch.testing.assert_close(cls,original[0],atol=0,rtol=0)
    torch.testing.assert_close(pd,original[3],atol=0,rtol=0)
    # A single represented class is exactly the original geometry.
    y0=torch.zeros_like(y);z0=torch.full_like(z,-4.);z0[:,0]=4.
    one=module._single(p,z0,g,y0,cam);baseline=old._single(p,z0,g,y0,cam)
    torch.testing.assert_close(one[1],baseline[1]);torch.testing.assert_close(one[2],baseline[2])
    # Active-class weighted mean, not voxel-weighted reduction, with both absent cases.
    pp=torch.tensor([[0.,0.,0.],[.7,0.,0.],[4.,0.,0.]],dtype=torch.double,requires_grad=True)
    zz=F.one_hot(torch.tensor([0,0,2]),17).double()
    gg=torch.tensor([[1.,0.,0.],[2.,0.,0.],[5.,0.,0.]],dtype=torch.double)
    yy=torch.tensor([0,0,1]);cc=torch.ones(3,dtype=torch.bool)
    va,vb,st=module.geometry(pp,zz,gg,yy,cc);ea,eb=reference(pp,zz,gg,yy,cc,module)
    torch.testing.assert_close(va,ea);torch.testing.assert_close(vb,eb)
    assert st[0,2]==1 and st[1,2]==0 and st[2,2]==0
    only=ClasswiseChamferSetLoss(geometry_class_weights=[.01]*17)
    tiny=only.geometry(pp,zz,gg,yy,cc)
    torch.testing.assert_close(tiny[0],va);torch.testing.assert_close(tiny[1],vb)
    zzabs=F.one_hot(torch.full((3,),16),17).double()
    empty=module.geometry(pp,zzabs,gg,yy,cc)
    assert float((empty[0]+empty[1]).detach())==0
    torch.testing.assert_close(torch.autograd.grad(empty[0]+empty[1],pp)[0],torch.zeros_like(pp))
    # CUDA KNN independent all-pair reference and gradients.
    cuda=ClasswiseChamferSetLoss(stage_weights=[1.],geometry_class_weights=weights,class_weights=weights).cuda()
    cp=p.detach().float().cuda().requires_grad_();cz=z.detach().float().cuda();cg=g.float().cuda();cy=y.cuda();cm=cam.cuda()
    ca,cb,_=cuda.geometry(cp,cz,cg,cy,cm);cr,cs=reference(cp,cz,cg,cy,cm,cuda)
    torch.testing.assert_close(ca,cr);torch.testing.assert_close(cb,cs)
    torch.testing.assert_close(torch.autograd.grad(ca+cb,cp,retain_graph=True)[0],torch.autograd.grad(cr+cs,cp)[0],rtol=1e-5,atol=1e-6)
    result=dict(status='passed',all_17_classes_reference=True,analytical_gradients=True,finite_difference_gradcheck=bool(gradcheck),
        classification_exact=True,original_global_distance_diagnostics_exact=True,single_class_matches_original=True,
        weighted_class_mean=True,missing_GT_or_GS_skipped=True,no_class_match_zero=True,
        no_direct_geometric_logits_gradient=True,CUDA_KNN_and_gradients=True)
    write(RUN/'numerical_checks.json',result);print(json.dumps(result))
if __name__=='__main__':main()
