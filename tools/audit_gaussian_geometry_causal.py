"""Real-frame VJP checks and exact paired-data checks for causal trials."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.gaussian_geometry_causal import RUN, ARMS, Routes, build, forward, seed_all, to_cuda, write


def real(checkpoint):
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    seed_all(928);torch.set_num_threads(4)
    c,net,opt,loss,state=build(checkpoint,freeze_empty=True)
    net.eval()
    dataset=OPENOCC_DATASET.build(c.val_dataset_config)
    batch=to_cuda(custom_collate_fn_temporal([dataset[0]]))
    with torch.no_grad():out=forward(net,batch)
    original=out['gaussian']
    names=('means','scales','rotations','opacities','semantics')
    leaves=[getattr(original,k).detach().clone().requires_grad_() for k in names]
    g=original._replace(**dict(zip(names,leaves)))
    controller=Routes(net)
    baseline_args=None;baseline_logits=None;gradients={};results=[]
    for arm in ARMS:
        controller.arm=arm
        args=net.head.prepare_gaussian_args(g)
        if baseline_args is None:baseline_args=[a.detach().clone() for a in args]
        for a,b in zip(args,baseline_args):assert torch.equal(a.detach(),b),arm
        z=net.head._render_batched(out['sampled_xyz'],*args)
        z=torch.cat((z[:,:17],z[:,17:]-net.head.empty_bias.detach()),1)
        if baseline_logits is None:baseline_logits=z.detach().clone()
        torch.testing.assert_close(z.detach(),baseline_logits,rtol=1e-6,atol=1e-6)
        inp=dict(pred_occ=[z],sampled_xyz=out['sampled_xyz'],sampled_label=out['sampled_label'],
                 occ_mask=out['occ_cam_mask'])
        value=loss.losses[1](inp)
        if isinstance(value,tuple):value=value[0]
        grads=torch.autograd.grad(value,leaves,allow_unused=True)
        grads=[torch.zeros_like(p) if v is None else v for p,v in zip(leaves,grads)]
        gradients[arm]=grads
        norms={k:float(v.double().norm()) for k,v in zip(names,grads)}
        radial=float((grads[-1]*leaves[-1]).sum(-1).abs().max())
        if arm in ('no_amplitude','no_both'):
            assert norms['opacities']==0
        if arm in ('no_geometry','no_both'):
            assert all(norms[k]==0 for k in names[:3])
        assert norms['semantics']>0
        results.append(dict(arm=arm,loss=float(value.detach()),gradient_norms=norms,
            semantic_radial_dot_max=radial,renderer_inputs_bitwise_identical=True,
            logits_max_abs_error=float((z.detach()-baseline_logits).abs().max())))
    p=leaves[-1].detach()/leaves[-1].detach().sum(-1,keepdim=True)
    expected=gradients['standard'][-1]-(gradients['standard'][-1]*p).sum(-1,keepdim=True)
    torch.testing.assert_close(gradients['no_amplitude'][-1],expected,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(gradients['no_geometry'][-1],gradients['standard'][-1],rtol=1e-5,atol=1e-6)
    # The controls neither mutate the Gaussian outputs nor replace the raw
    # logits/points passed to the original OPUS Set loss.
    for key,value in zip(names,leaves):assert torch.equal(getattr(g,key),value)
    controller.close()
    result=dict(status='passed',checkpoint=str(checkpoint),real_frame='val index 0',
        set_inputs_untouched=True,results=results)
    write(RUN/'real_route_audit.json',result)
    return result


def pairing(folder,seed=928):
    paths=[folder/(arm+f'_s{seed}') for arm in ARMS[:4]]
    traces=[[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()] for p in paths]
    base=traces[0]
    snapshots=[np.load(p/'initial_forward.npz') for p in paths]
    errors={}
    for arm,rows,snapshot in zip(ARMS[:4],traces,snapshots):
        assert [(x['index'],x['fingerprint']) for x in rows]==[(x['index'],x['fingerprint']) for x in base],arm
        errors[arm]={}
        for key in snapshots[0].files:
            diff=float(np.abs(snapshot[key]-snapshots[0][key]).max())
            errors[arm][key]=diff
            assert np.allclose(snapshot[key],snapshots[0][key],rtol=1e-5,atol=2e-5),(arm,key,diff)
    result=dict(status='passed',microbatches=len(base),all_data_fingerprints_equal=True,
                initial_max_abs_errors=errors)
    write(folder/'pairing_audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['real','pairing'])
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--folder',type=Path,default=RUN/'train')
    p.add_argument('--seed',type=int,default=928)
    args=p.parse_args()
    print(real(args.checkpoint) if args.action=='real' else pairing(args.folder,args.seed))
