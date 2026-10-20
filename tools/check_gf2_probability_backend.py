"""Tiny exact-support forward/gradient check of the unchanged official kernel."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from model.head.gf2_probability_backend import probability_backend
from tools.gf2_probability_experiment import RUN,write


def main():
    torch.manual_seed(928)
    aggregator=probability_backend().LocalAggregator(scale_multiplier=3,H=16,W=16,D=16,
        pc_min=[0.,0.,0.],grid_size=.5).cuda()
    # Official backward has one lookup entry per voxel; use distinct cells,
    # as in the actual occupancy-grid ABI (not several points in one cell).
    xyz=torch.tensor([[[2.21,2.13,2.17],[2.70,2.25,2.10],[2.1,2.90,2.2]]],device='cuda')
    means=torch.tensor([[[2.,2.,2.],[2.5,2.2,2.3]]],device='cuda',requires_grad=True)
    opa=torch.tensor([[.3,.8]],device='cuda',requires_grad=True)
    raw=torch.randn(1,2,17,device='cuda',requires_grad=True)
    inv=(torch.eye(3,device='cuda').reshape(1,1,3,3).repeat(1,2,1,1)*4).requires_grad_()
    scales=torch.full((1,2,3),.5,device='cuda')
    semantic=torch.cat([raw.softmax(-1),torch.zeros(1,2,1,device='cuda')],-1)
    actual=aggregator(xyz,means,opa,semantic,scales,inv)
    d=xyz[0,:,None]-means[0,None]
    k=torch.exp(-.5*torch.einsum('ngd,gde,nge->ng',d,inv[0],d))
    weight=k*opa[0]*torch.linalg.det(inv[0]).sqrt()/(2*torch.pi)**1.5
    q=(weight@semantic[0])/weight.sum(-1,keepdim=True)
    expected=(q,1-(1-k).prod(-1),k.sum(-1))
    errors=[]
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6);errors.append(float((a-b).abs().max()))
    payload=torch.linspace(-.5,.7,q.numel(),device='cuda').reshape_as(q)
    loss=lambda values:(values[0]*payload).sum()+.3*values[1].sum()+.01*values[2].sum()
    ga=torch.autograd.grad(loss(actual),(means,opa,raw,inv),retain_graph=True)
    ge=torch.autograd.grad(loss(expected),(means,opa,raw,inv))
    # Official covariance input stores only the six independent symmetric
    # entries. Compare its symmetric gradient rather than arbitrary skew terms.
    gradient_errors=[]
    for index,(a,b) in enumerate(zip(ga,ge)):
        if index==3:a=(a+a.transpose(-1,-2))/2;b=(b+b.transpose(-1,-2))/2
        torch.testing.assert_close(a,b,rtol=3e-3,atol=3e-4)
        gradient_errors.append(float((a-b).abs().max()))
    write(RUN/'verification/backend_check.json',dict(status='passed',forward_max_errors=errors,
        gradient_max_errors=gradient_errors,compared=['means','opacity','raw_semantics','symmetric_inverse_covariance'],
        support='All tiny-example pairs inside identical official local support'))
    print('Official backend matches differentiable reference, forward and source gradients.')


if __name__=='__main__':main()
