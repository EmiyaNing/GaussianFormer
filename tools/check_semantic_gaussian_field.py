"""Numerical and semantic invariants for the section-4.1 field experiment."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from loss.semantic_gaussian_field.geometry import project_standard, solid_ellipsoid_distance
from loss.semantic_gaussian_field import extension
from loss.semantic_gaussian_field.data import construct
from model.utils.utils import get_rotation_matrix


def main():
    torch.set_num_threads(4)
    torch.manual_seed(928)
    # Envelope gradients in all three attributes, checked against re-solved
    # FP64 closest points rather than a detached world projection.
    p=torch.tensor([[.8,.6,.4],[.4,.3,.8]],dtype=torch.float64)
    mu=torch.tensor([[.1,.2,.1],[.2,.1,.1]],dtype=torch.float64,requires_grad=True)
    scale=torch.tensor([[.12,.25,.37],[.22,.13,.34]],dtype=torch.float64,requires_grad=True)
    quat=torch.tensor([[.9,.2,.1,.15],[.8,-.1,.2,.3]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda m,s,q:solid_ellipsoid_distance(p,m,s,get_rotation_matrix(q)),
        (mu,scale,quat),eps=1e-5,atol=3e-5,rtol=1e-3)
    inside=solid_ellipsoid_distance(mu.detach(),mu,scale,get_rotation_matrix(quat))
    assert torch.equal(inside,torch.zeros_like(inside))
    sphere=torch.ones_like(scale)*.2
    actual=solid_ellipsoid_distance(p,mu,sphere,get_rotation_matrix(quat))
    torch.testing.assert_close(actual,((p-mu).norm(dim=-1)-.2).clamp_min(0),atol=1e-12,rtol=1e-12)
    # Half-grid construction must equal exact cube distances at every grid center.
    shape=(8,9,5);h=.4;lower=[0.,0.,0.]
    y=np.full(shape,17,np.int64);camera=np.ones(shape,bool)
    y[2,2,2]=0;y[5,6,2]=0;y[3,3,1]=1
    distance,component=construct(y,camera,h)
    xyz=(np.stack(np.meshgrid(*[np.arange(n) for n in shape],indexing='ij'),-1)+.5)*h
    gt=xyz[y<17]
    expected=np.linalg.norm(np.maximum(np.abs(xyz[...,None,:]-gt)-h/2,0),axis=-1).min(-1)
    np.testing.assert_allclose(distance,expected,atol=2e-7,rtol=2e-7)
    assert component[2,2,2]!=component[5,6,2]
    # Global accelerated nearest ellipsoid agrees with exhaustive FP64 solution.
    device='cuda';n=73;nt=45
    m=torch.rand(n,3)*torch.tensor(shape)*h
    s=torch.rand(n,3)*.3+.08
    q=torch.randn(n,4);R=get_rotation_matrix(q)
    points=torch.rand(nt,3)*torch.tensor(shape)*h
    m,s,R,points=[a.to(device).contiguous() for a in (m,s,R,points)]
    seed=torch.cdist(points,m).argmin(1).contiguous()
    ext=extension();heads,links=ext.build_hash(m,shape,lower,h)
    assigned=torch.ones(n,2,device=device,dtype=torch.int32)
    assigned[n//2:,1]=2
    pc=torch.ones(nt,device=device,dtype=torch.int32)
    eligible=torch.ones(n,device=device,dtype=torch.bool)
    ids,u,d,fallback=ext.cover(points,m,s,R,heads,links,seed,pc,assigned,eligible,
        shape,lower,h,.4,2.,True)
    pp=points.cpu().double()[:,None].expand(-1,n,-1)
    mm=m.cpu().double()[None].expand(nt,-1,-1)
    ss=s.cpu().double()[None].expand(nt,-1,-1)
    rr=R.cpu().double()[None].expand(nt,-1,-1,-1)
    brute=solid_ellipsoid_distance(pp,mm,ss,rr)
    geometric=brute.min(1).values
    torch.testing.assert_close(d[:,0].cpu().double(),geometric,atol=3e-6,rtol=3e-5)
    allowed=(assigned[:,1].cpu()[None]==pc.cpu()[:,None])&((pp-mm).norm(dim=-1)<=2.)
    same=brute.masked_fill(~allowed,float('inf')).min(1).values
    torch.testing.assert_close(d[:,1].cpu().double(),same,atol=3e-6,rtol=3e-5)
    # A wrong-component core may geometrically cover a point, but is fallback.
    pc.fill_(999)
    _,_,d2,fb=ext.cover(points,m,s,R,heads,links,seed,pc,assigned,eligible,
        shape,lower,h,.4,2.,True)
    assert fb.all()
    torch.testing.assert_close(d2[:,0],d2[:,1],atol=0,rtol=0)
    # Same class, disconnected component: a point in the other component is
    # still outside this GS's local target. Unknown samples are skipped.
    ym=torch.tensor([[1.,1.,1.]],device=device)
    samples=torch.tensor([[2.2,2.6,1.],[.9,1.,1.]],device=device)
    ac=torch.tensor([[0,int(component[2,2,2])]],device=device,dtype=torch.int32)
    active=torch.tensor([True,False],device=device)
    proj,found=ext.fit(samples,ym,ac,torch.tensor(component,device=device,dtype=torch.int32).flatten(),
        active,2,shape,lower,h,2.)
    assert found[0] and not found[1]
    truth=np.clip(samples[0].cpu().numpy(),xyz[2,2,2]-.2,xyz[2,2,2]+.2)
    np.testing.assert_allclose(proj[0].cpu().numpy(),truth,atol=2e-7)
    assert torch.equal(proj[1],samples[1])
    out=dict(status='passed',fp64_center_scale_rotation_gradcheck=True,
        interior_exact_zero=True,isotropic_closed_form=True,
        half_lattice_voxel_solid_exact=True,disconnected_components_separated=True,
        cuda_branch_bound_exhaustive_fp64_max_error=float((d[:,0].cpu()-geometric).abs().max()),
        same_component_cuda_matches_fp64=True,missing_same_candidate_is_explicit_fallback=True,
        local_component_cube_projection_exact=True,unknown_samples_skipped=True)
    path=ROOT/'work_dirs/semantic_field_mini_v1/numerical_checks.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
