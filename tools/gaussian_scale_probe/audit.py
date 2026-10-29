"""Independent native-backward and loss checks before the complete census."""
import numpy as np
import torch
import torch.nn.functional as F


def audit_frame(network,out,batch,source,result,values,shape,occ):
    from tools.probe_gaussian_scale_supervision import render,layout
    head=network.head;g=out['gaussian']
    means=g.means.detach().clone().requires_grad_(True)
    scales=g.scales.detach().clone().requires_grad_(True)
    opacity=g.opacities.detach().clone().requires_grad_(True)
    raw=out['opus_pred_logits'][-1].detach().clone().requires_grad_(True)
    leaf=g._replace(means=means,scales=scales,opacities=opacity,semantics=F.softplus(raw))
    mask=source['mask'];xyz=out['sampled_xyz'][0,mask]
    with torch.enable_grad():z=render(head,leaf,xyz)
    torch.testing.assert_close(z,source['z'],atol=2e-4,rtol=2e-4)
    original=occ.loss_voxel(out['pred_occ'],out['sampled_xyz'],out['sampled_label'],batch['occ_cam_mask'])
    error=abs(float(original)-source['losses']['ce']-source['losses']['lovasz'])
    assert error<5e-5,error
    target=source['assigned'].long().clamp_min(0)
    analytic=result[1][:,10:].reshape(-1,3,9);records=[]
    for s in range(3):
        gradient=source['ce' if s<2 else 'lov'][mask].T
        if s<2:gradient=gradient*(source['target']==17 if s==0 else source['target']<17)
        objective=(z[:17]*gradient).sum()
        actual=torch.autograd.grad(objective,(means,scales,opacity,raw),retain_graph=s<2)
        true_mean,true_scale,true_alpha,true_raw=[t[0].detach() for t in actual]
        true_target=true_raw.gather(1,target[:,None]).flatten()
        true_amp=(true_raw*(F.softplus(raw[0].detach().double())/raw[0].detach().double().sigmoid().clamp_min(1e-300))).sum(1)
        pairs=[('opacity',analytic[:,s,0],true_alpha[:,0]),('scales',analytic[:,s,1:4],true_scale),
               ('means',analytic[:,s,4:7],true_mean),('assigned_raw',analytic[:,s,7],true_target),
               ('log_amplitude',analytic[:,s,8],true_amp)]
        row={}
        for name,a,b in pairs:
            a=a.double();b=b.double()
            relative=float((a-b).norm()/b.norm().clamp_min(1e-30))
            absolute=float((a-b).abs().max())
            assert relative<.003 or absolute<1e-7,(s,name,relative,absolute)
            row[name]=dict(relative_l2=relative,max_absolute=absolute)
        records.append(row)
    # The mass helper must return exactly the same quantities on original args.
    from tools.gaussian_scale_probe import extension
    m=extension().mass(values[0],values[2],values[3],values[5],values[6],
        out['sampled_xyz'][0].contiguous(),batch['occ_cam_mask'].flatten().bool().contiguous(),*shape)
    torch.testing.assert_close(m,result[1][:,6],atol=2e-5,rtol=2e-5)
    # Integer candidate boxes do not change for the registered 0.30 m cap.
    saturated=g.scales.max(-1).values>=.39
    proposed=g._replace(scales=torch.where(saturated[...,None],g.scales.clamp(max=.30),g.scales))
    _,new=layout(head,proposed)
    assert torch.equal(values[6],new[6])
    scale_raw=network._scale_probe_raw[...,:3].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        actual_scales=.08+.32*scale_raw.sigmoid()
        chain=torch.autograd.grad(actual_scales.sum(),scale_raw)[0]
    torch.testing.assert_close(chain[0],out['_scale_probe_jacobian'],atol=1e-8,rtol=1e-6)
    return dict(loss_api_max_error=error,native_backward=records,native_box_unchanged_at_cap30=True,
                exact_raw_scale_chain_verified=True,all_gaussians_checked=len(values[0]))
