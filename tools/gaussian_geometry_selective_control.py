"""One-sided amplitude control: keep empty correction, suppress selected boosts.

The mask uses training GT only for causal intervention. This is a mechanism
experiment, not a proposed inference-time method or a GT-dependent evaluator.
"""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import tools.gaussian_geometry_causal as trial
from tools.gaussian_geometry_causal import RUN,write,sha

ARM='no_empty_center_boost'


class Recorder:
    def __init__(self,net,arm):
        self.head=net.head
        self.original=self.head.prepare_gaussian_args
        def prepare(head,gaussian):
            head._selective_gaussian=gaussian
            args=self.original(gaussian)
            head._selective_inverse=args[-1][:,:-1]
            return args
        self.head.prepare_gaussian_args=types.MethodType(prepare,self.head)
    def close(self):self.head.prepare_gaussian_args=self.original


@torch.no_grad()
def foreground_in_core(head,g,inputs):
    """Exact 1-sigma GT test in the bounded-scale model (27 candidate cells).

    Every principal sigma is <= one voxel (0.4m), so any core voxel center
    must be in the containing cell or one of its 26 immediate neighbors.
    """
    assert g.means.shape[0]==1 and g.scales.max()<=head.aggregator.grid_size+1e-6
    cells=((g.means[0]-head.aggregator.pc_min)/head.aggregator.grid_size).to(torch.int32).long()
    offsets=torch.cartesian_prod(*[torch.arange(-1,2,device=cells.device)]*3)
    candidates=cells[:,None]+offsets[None]
    dims=torch.tensor([head.aggregator.H,head.aggregator.W,head.aggregator.D],device=cells.device)
    inside=((candidates>=0)&(candidates<dims)).all(-1)
    flat=(candidates[...,0]*head.aggregator.W+candidates[...,1])*head.aggregator.D+candidates[...,2]
    safe=flat.clamp(0,int(dims.prod())-1)
    labels=inputs['sampled_label'][0].long()[safe]
    camera=inputs['occ_mask'][0].flatten().bool()[safe]
    delta=inputs['sampled_xyz'][0][safe]-g.means[0,:,None]
    d2=torch.einsum('nvi,nij,nvj->nv',delta,head._selective_inverse[0].detach(),delta)
    return (inside&camera&(labels>=0)&(labels<17)&(d2<=1)).any(-1)[None]


class SelectiveLoss(torch.nn.Module):
    def __init__(self,base,head):
        super().__init__();self.base=base
        object.__setattr__(self,'head',head)
    def forward(self,inputs):
        total,details=self.base(inputs)
        occupancy=self.base.losses[1](inputs)
        g=self.head._selective_gaussian
        a,alpha=g.semantics,g.opacities
        ga,galpha=torch.autograd.grad(occupancy,(a,alpha),retain_graph=True)
        with torch.no_grad():
            A=a.sum(-1,keepdim=True);p=a/A
            dA=(p*ga).sum(-1,keepdim=True)
            cells=((g.means-self.head.aggregator.pc_min)/self.head.aggregator.grid_size).to(torch.int32).long()
            flat=(cells[...,0]*self.head.aggregator.W+cells[...,1])*self.head.aggregator.D+cells[...,2]
            labels=inputs['sampled_label'].long()
            camera=inputs['occ_mask'].flatten(1).bool()
            center_empty=camera.gather(1,flat)&(labels.gather(1,flat)==17)
            core_fg=foreground_in_core(self.head,g,inputs)
            eligible=center_empty&~core_fg
            mask=eligible.unsqueeze(-1)&(dA<0)
        # Subtract only this direct renderer-amplitude VJP. All Set gradients,
        # class-direction and geometry gradients, and every lowering direction
        # remain available. The forward scalar remains exactly the original.
        correction=-(a*(dA*mask).detach()).sum()-(alpha*(galpha*mask).detach()).sum()
        modified=total+(correction-correction.detach())
        assert torch.equal(modified.detach(),total.detach())
        details.update(selective_camera_empty_fraction=float(center_empty.float().mean()),
            selective_eligible_fraction=float(eligible.float().mean()),
            selective_blocked_fraction=float(mask.float().mean()),
            selective_removed_dlogA=float((ga*a*mask).sum().detach()),
            selective_forward_error=float((modified-total).detach().abs()))
        return modified,details


def train():
    original_build=trial.build
    def build(*args,**kwargs):
        c,net,opt,loss,state=original_build(*args,**kwargs)
        return c,net,opt,SelectiveLoss(loss,net.head),state
    trial.build=build;trial.Routes=Recorder
    return trial.training(ARM,1024,928,RUN/'common/standard_s928/final.pth',8,False)


def audit():
    """Independent VJP algebra, including a gate that must keep lowering."""
    torch.manual_seed(928)
    a=torch.rand(32,17,dtype=torch.float64,requires_grad=True)+.1
    alpha=torch.rand(32,1,dtype=torch.float64,requires_grad=True)+.1
    k=torch.randn(32,17,dtype=torch.float64)
    loss=(alpha*a*k).sum()
    ga,galpha=torch.autograd.grad(loss,(a,alpha),retain_graph=True)
    A=a.sum(-1,keepdim=True);p=a/A
    dA=(p*ga).sum(-1,keepdim=True)
    eligible=(torch.arange(32)%2==0).unsqueeze(-1)
    mask=eligible&(dA.detach()<0)
    correction=-(a*(dA*mask).detach()).sum()-(alpha*(galpha*mask).detach()).sum()
    new=loss+(correction-correction.detach())
    actual,al=torch.autograd.grad(new,(a,alpha))
    expected=ga-dA*mask
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(al,galpha*(~mask),atol=1e-14,rtol=1e-14)
    assert torch.equal(loss.detach(),new.detach())
    assert torch.equal(actual[~mask.flatten()],ga[~mask.flatten()])
    result=dict(status='passed',selected=int(mask.sum()),preserved_nonboost_or_ineligible=True,
        identity_forward_exact=True,projected_radial_max=float((actual*a).sum(-1)[mask.flatten()].abs().max()))
    write(RUN/'selective_gradient_audit.json',result)


def run():
    protocol=dict(registered_at=datetime.now().astimezone().isoformat(),arm=ARM,
        source_sha256=sha(__file__),base_source_sha256=sha(ROOT/'tools/gaussian_geometry_causal.py'),
        source_selection='Registered during common warmup, before all four formal branches',
        rule='Camera-visible empty center AND no camera-valid foreground voxel within 1-sigma core AND negative Occupancy dL/dA; subtract direct A and alpha boosting VJP only; preserve lowering, class direction, geometry, and all Set loss',
        revision='Before any selective training: exclude empty-center Gaussians with foreground in their core, because those can be useful cross-boundary representations. The prior preregistration is retained separately.',
        formal_steps=1024,accumulation=8,seed=928,
        scope='Same full training pool and same all-850-scene evaluation as the original four arms',
        rationale='Global amplitude blocking also removes legitimate empty-CE correction, so it cannot alone isolate harmful compensation')
    if not (RUN/'selective_protocol.json').exists():write(RUN/'selective_protocol.json',protocol)
    else:assert json.loads((RUN/'selective_protocol.json').read_text())['source_sha256']==sha(__file__)
    audit()
    # Serialize with the primary queue to avoid training/evaluation contention.
    while True:
        path=RUN/'queue_status.json'
        if path.exists():
            status=json.loads(path.read_text())
            if status['status']=='failed':raise RuntimeError('Primary queue failed')
            if status['status']=='execution_complete':break
        time.sleep(5)
    train()
    checkpoint=RUN/'train'/f'{ARM}_s928/final.pth'
    with (RUN/'selective_eval_console.log').open('a') as f:
        subprocess.run([sys.executable,'tools/eval_gaussian_geometry_causal.py','--name',ARM,
            '--checkpoint',str(checkpoint)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(RUN/'selective_complete.json',dict(status='complete',checkpoint_sha256=sha(checkpoint)))


if __name__=='__main__':run()
