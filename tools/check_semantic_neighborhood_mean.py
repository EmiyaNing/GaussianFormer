"""Numerical checks for the literal same-class 3sigma mean and any-GS core gate."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.semantic_neighborhood_experiment import RUN,write
from loss.semantic_neighborhood_mean import local_mean
from model.utils.utils import get_rotation_matrix

def grid(shape,h=.4):
    return torch.stack(torch.meshgrid(*[(torch.arange(n)+.5)*h for n in shape],indexing='ij'),-1).reshape(-1,3)

def call(m,s,R,c,x,y,fg,shape):
    ids=torch.where(fg)[0];lookup=torch.full((len(x),),-1,device='cuda',dtype=torch.int32)
    lookup[ids]=torch.arange(len(ids),device='cuda',dtype=torch.int32)
    return local_mean(m,s,R,c,x,y,lookup,len(ids),shape,(0.,0.,0.),.4)

def main():
    torch.manual_seed(928);torch.cuda.manual_seed_all(928)
    shape=(12,11,8);x=grid(shape).cuda().contiguous();y=torch.randint(0,3,(len(x),),device='cuda')
    fg=torch.rand(len(x),device='cuda')>.25
    m=(.5+torch.rand(29,3,device='cuda')*2.2).requires_grad_()
    s=(.12+torch.rand(29,3,device='cuda')*.35).requires_grad_()
    R=get_rotation_matrix(torch.randn(29,4,device='cuda')).detach().requires_grad_()
    c=torch.randint(0,3,(29,),device='cuda')
    v,n,core,same,pergs=call(m,s,R,c,x,y,fg,shape)
    # Independent FP64 all-pair reference, with frozen discrete membership.
    md=m.detach().cpu().double().requires_grad_();sd=s.detach().cpu().double();rd=R.detach().cpu().double()
    xd=x[fg].cpu().double();yd=y[fg].cpu();cd=c.cpu()
    diff=md[:,None]-xd[None];power=(torch.einsum('gij,gtj->gti',rd,diff)/sd[:,None]).square().sum(-1)
    support3=power.detach()<=9;support1=power.detach()<=1;sel=support3&(yd[None]==cd[:,None])
    distances=diff.norm(dim=-1);reference=(distances*sel).sum(-1)/sel.sum(-1).clamp_min(1)
    torch.testing.assert_close(v.cpu().double(),reference,rtol=3e-6,atol=3e-6)
    np.testing.assert_array_equal(n.cpu(),sel.sum(-1))
    np.testing.assert_array_equal(core.cpu(),support1.sum(0));np.testing.assert_array_equal(same.cpu(),(support1&(yd[None]==cd[:,None])).sum(0))
    np.testing.assert_array_equal(pergs.cpu(),support1.sum(-1))
    expected=torch.autograd.grad(reference.sum(),md)[0]
    gm,gs,gr=torch.autograd.grad(v.sum(),[m,s,R],allow_unused=True)
    torch.testing.assert_close(gm.cpu().double(),expected,rtol=2e-5,atol=3e-6)
    assert gs is None and gr is None
    # Actual CUDA finite differences, only where the entire 3sigma membership stays unchanged.
    checks=0;max_error=0.
    for g in range(29):
        for axis in range(3):
            eps=2e-4;plus=m.detach().clone();minus=m.detach().clone();plus[g,axis]+=eps;minus[g,axis]-=eps
            stable=True
            for a in (plus,minus):
                d=a[g].cpu().double()-xd
                p=((d@rd[g].T)/sd[g]).square().sum(-1)
                stable &= bool(torch.equal(p<=9,support3[g]))
            if not stable:continue
            vp=call(plus,s,R,c,x,y,fg,shape)[0][g];vm=call(minus,s,R,c,x,y,fg,shape)[0][g]
            finite=float((vp-vm)/(plus[g,axis]-minus[g,axis]));error=abs(finite-float(gm[g,axis]));max_error=max(max_error,error)
            assert error<.002,(g,axis,error);checks+=1
            if checks==12:break
        if checks==12:break
    assert checks==12
    noncentroid=[]
    for g in range(29):
        if int(sel[g].sum())>1:
            center_distance=(md[g]-xd[sel[g]].mean(0)).norm()
            noncentroid.append(float(reference[g].detach()-center_distance.detach()))
    assert max(noncentroid)>.01
    # Closest center misses, a farther large GS covers: the GT must stop attraction.
    sh=(6,6,6);xx=grid(sh).cuda().contiguous();yy=torch.full((len(xx),),17,device='cuda',dtype=torch.long)
    target=(xx-torch.tensor([1.,1.,1.],device='cuda')).norm(dim=-1).argmin();yy[target]=0;ff=yy<17
    mm=torch.tensor([[.8,1.,1.],[1.4,1.,1.]],device='cuda',requires_grad=True)
    ss=torch.tensor([[.1]*3,[.6]*3],device='cuda');rr=torch.eye(3,device='cuda').repeat(2,1,1)
    vv,nn,cc,_,_=call(mm,ss,rr,torch.zeros(2,device='cuda',dtype=torch.long),xx,yy,ff,sh)
    assert int(cc[0])==1 and int(nn[0])==1 and int(nn[1])==1
    gt=xx[ff];nearest=(gt[:,None]-mm[None]).square().sum(-1).argmin(-1)
    assert int(nearest[0])==0
    uncovered=(cc==0).float();zero=((gt-mm[nearest]).norm(dim=-1)*uncovered).mean()
    torch.testing.assert_close(torch.autograd.grad(zero,mm)[0],torch.zeros_like(mm))
    # Missing predicted class has zero additional mean and zero center gradient.
    absent=call(mm,ss,rr,torch.full((2,),9,device='cuda',dtype=torch.long),xx,yy,ff,sh)[0]
    assert bool((absent==0).all())
    torch.testing.assert_close(torch.autograd.grad(absent.sum(),mm)[0],torch.zeros_like(mm))
    result=dict(status='passed',all_pair_mean_and_counts=True,any_GS_core_coverage=True,
        gradient_matches_FP64=True,no_direct_shape_gradient=True,CUDA_finite_difference_checks=checks,
        maximum_finite_difference_error=max_error,not_distance_to_centroid=True,
        closest_miss_farther_cover_stops=True,empty_same_class_neighborhood_zero=True)
    RUN.mkdir(parents=True,exist_ok=True);write(RUN/'numerical_checks.json',result);print(json.dumps(result))
if __name__=='__main__':main()
