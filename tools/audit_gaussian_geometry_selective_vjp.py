"""Independent real-renderer VJP audit, including repeated backward safety."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.gaussian_geometry_causal import RUN,build,forward,to_cuda,write,seed_all
from tools.gaussian_geometry_selective_control import Recorder,SelectiveLoss,foreground_in_core
from tools.boundary_gaussian_stats.data import EvaluationFrames,FeatureCache
from dataset.utils import custom_collate_fn_temporal


def main():
    seed_all(928);torch.set_num_threads(4)
    c,net,opt,loss,state=build(ROOT/'ckpts/opus_gaussian_8f_tiny.pth',freeze_empty=True)
    net.eval();dataset=EvaluationFrames(c,'val');cache=FeatureCache(net,dataset)
    batch=to_cuda(custom_collate_fn_temporal([dataset[0]]))
    with torch.no_grad():out=forward(net,batch)
    y=batch['occ_label'].flatten().long();cam=batch['occ_cam_mask'].flatten().bool()
    pred=out['final_occ'].flatten()
    cm=torch.bincount(y[cam]*18+pred[cam],minlength=324).reshape(18,18).cpu().numpy()
    scene=dataset.keyframes[0][0]
    old=json.loads((ROOT/'work_dirs/gaussian_scale_supervision_full/scenes/val'/f'{scene}.json').read_text())
    reference=next(x for x in old['records'] if x['index']==0)
    np.testing.assert_array_equal(cm,np.asarray(reference['confusion'][0]))
    names=('semantics','opacities','means','scales','rotations')
    leaves=[getattr(out['gaussian'],k).detach().clone().requires_grad_() for k in names]
    g=out['gaussian']._replace(**dict(zip(names,leaves)))
    recorder=Recorder(net,'audit')
    z=net.head._render_batched(out['sampled_xyz'],*net.head.prepare_gaussian_args(g))
    z=torch.cat((z[:,:17],z[:,17:]-net.head.empty_bias.detach()),1)
    inputs=dict(pred_occ=[z],sampled_xyz=out['sampled_xyz'],sampled_label=out['sampled_label'],
                occ_mask=out['occ_cam_mask'])
    occ=loss.losses[1]
    base=occ(inputs)
    gradients=torch.autograd.grad(base,leaves,retain_graph=True)
    a,alpha=leaves[:2]
    with torch.no_grad():
        p=a/a.sum(-1,keepdim=True);dA=(p*gradients[0]).sum(-1,keepdim=True)
        cells=((g.means-net.head.aggregator.pc_min)/net.head.aggregator.grid_size).to(torch.int32).long()
        flat=(cells[...,0]*200+cells[...,1])*16+cells[...,2]
        eligible=cam[flat]&(y[flat]==17)&~foreground_in_core(net.head,g,inputs)
        mask=eligible[...,None]&(dA<0)
    expected=[gradients[0]-dA*mask,gradients[1]*(~mask),*gradients[2:]]
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.losses=torch.nn.ModuleList([torch.nn.Identity(),occ])
        def forward(self,inputs):return occ(inputs),{}
    modified,details=SelectiveLoss(Adapter(),net.head)(inputs)
    actual=torch.autograd.grad(modified,leaves)
    errors={}
    for name,want,got in zip(names,expected,actual):
        err=float((got-want).double().norm()/want.double().norm().clamp_min(1e-30))
        errors[name]=err
        torch.testing.assert_close(got,want,rtol=2e-4,atol=2e-8)
        assert err<2e-4,(name,err)
    assert torch.equal(base.detach(),modified.detach())
    write(RUN/'selective_real_vjp_audit.json',dict(status='passed',
        mature_reference_native_confusion_exact=True,selected_gaussians=int(mask.sum()),
        independent_vjp_relative_errors=errors,forward_loss_exact=True,
        repeated_native_backward_safe=True))
    cache.close();recorder.close()
    print('Independent real VJP and original mature checkpoint parity passed')


if __name__=='__main__':main()
