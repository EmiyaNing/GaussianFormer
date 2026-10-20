"""Validate the intervention on a real model graph, including sibling views."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage3 import (RUN,build_origin,training_dataset,load_item,
    network_output,set_negative_gradients,suppressive_part,ce_parts,seed_all,write_json)


def main():
    seed_all(31928);net,opt,sched,cfg,checkpoint,module,mods=build_origin();net.eval()
    batch=load_item(training_dataset(),0,3192800000)
    out=network_output(net,batch)
    inp={k:out[k] for k in ['opus_pred_points','opus_pred_logits','opus_point_valid_masks']};inp['metas']=batch
    components,_=module.components(inp)
    closed=set_negative_gradients(out,batch,module)
    from loss.opus_set_loss import _nearest_indices
    yy=batch['occ_label'][0].long().flatten();xyz=batch['occ_xyz'][0].reshape(-1,3)
    cam=batch['occ_cam_mask'][0].bool().flatten();set_checks=[]
    for stage,raw in enumerate(out['opus_pred_logits']):
        pm=out['opus_point_valid_masks'][stage][0].bool()
        domain=module.final_gt_domain if stage==5 else module.gt_domain
        tm=(yy>=0)&(yy<17)
        if domain=='camera':tm&=cam
        assigned=_nearest_indices(out['opus_pred_points'][stage][0,pm],xyz[tm],module.chunk_size)
        target=yy[tm][assigned];valid_raw=raw[0,pm]
        q=valid_raw.sigmoid();onehot=F.one_hot(target,17).to(raw)
        weights=module.class_weights.to(raw)[target]*module._distance_weight(xyz[tm][assigned])
        multiplier=module.stage_weights[stage]*module.lambda_cls*(module.final_classification_weight if stage==5 else 1)
        positive=(module.focal_alpha*(1-q).pow(module.focal_gamma)*F.softplus(-valid_raw)*onehot*weights[:,None]).sum()*multiplier/pm.sum()
        actual=torch.autograd.grad(components[f'stage_{stage}/classification']-positive,raw,retain_graph=True)[0]
        error=float((actual-closed[stage]).abs().max())
        relative=float((actual-closed[stage]).double().norm()/actual.double().norm().clamp_min(1e-30))
        assert relative<2e-6,(stage,error,relative)
        set_checks.append(dict(stage=stage,max_abs_error=error,relative_error=relative))
    y=out['sampled_label'][0].long();mask=out['occ_cam_mask'][0].flatten().bool()&(y>=0)&(y<18)
    empty=sum(ce_parts(s[:,:,mask],y[mask],mods[1].class_weights,mods[1].loss_voxel_ce_weight)[1]
              for s in out['pred_occ'])/len(out['pred_occ'])
    ce_checks=[]
    for stage in out['supervised_stage_indices']:
        raw=out['opus_pred_logits'][stage];sem=out['gaussians'][stage].semantics
        gs=torch.autograd.grad(empty,sem,retain_graph=True)[0]
        expected=gs*raw.detach().sigmoid();parent=raw._base
        assert parent is not None and parent.numel()==raw.numel()
        actual=torch.autograd.grad(empty,parent,retain_graph=True)[0].reshape_as(raw)
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-8)
        removed,_=suppressive_part(raw,expected)
        correction=-(raw*removed).sum();modified=empty+(correction-correction.detach())
        modified_g=torch.autograd.grad(modified,parent,retain_graph=True)[0].reshape_as(raw)
        torch.testing.assert_close(modified_g,actual-removed,rtol=3e-5,atol=1e-8)
        geo=out['gaussians'][stage]
        targets=[geo.means,geo.opacities,geo.scales,geo.rotations,net.head.empty_bias,net.head.empty_scalar]
        native=torch.autograd.grad(empty,targets,retain_graph=True)
        revised=torch.autograd.grad(modified,targets,retain_graph=True)
        errors=[]
        for a,b in zip(native,revised):
            torch.testing.assert_close(a,b,rtol=2e-6,atol=1e-8)
            errors.append(float((a-b).abs().max()))
        ce_checks.append(dict(stage=stage,raw_gradient_abs_error=float((actual-expected).abs().max()),
            blocked_raw_gradient_abs_error=float((modified_g-(actual-removed)).abs().max()),
            geometry_and_empty_direct_gradient_errors=errors))
    from loss.empty_history_path_loss import EmptyHistoryPathLoss
    history_inputs=dict(inp,global_iter=0,pred_occ=out['pred_occ'],sampled_label=out['sampled_label'],
        sampled_xyz=out['sampled_xyz'],occ_mask=out['occ_cam_mask'],gaussians=out['gaussians'],
        supervised_stage_indices=out['supervised_stage_indices'])
    values=[]
    reference_early=None
    reference_last=None
    last_stage=out['supervised_stage_indices'][-1]
    last_raw=out['opus_pred_logits'][last_stage]
    removed_set,_=suppressive_part(last_raw,closed[last_stage])
    sem=out['gaussians'][last_stage].semantics
    ce_source=torch.autograd.grad(empty,sem,retain_graph=True)[0]*last_raw.detach().sigmoid()
    removed_ce,_=suppressive_part(last_raw,ce_source)
    for condition in ['B0','S','E','SE']:
        wrapper=EmptyHistoryPathLoss(condition=condition).cuda()
        loss,_=wrapper(history_inputs);values.append(float(loss.detach()))
        early=torch.autograd.grad(loss,out['opus_pred_logits'][0],retain_graph=True)[0]
        if reference_early is None:reference_early=early
        torch.testing.assert_close(early,reference_early,atol=0,rtol=0)
        last_gradient=torch.autograd.grad(loss,last_raw._base,retain_graph=True)[0].reshape_as(last_raw)
        if reference_last is None:reference_last=last_gradient
        expected=reference_last.clone()
        if condition in ['S','SE']:expected-=removed_set
        if condition in ['E','SE']:expected-=removed_ce
        torch.testing.assert_close(last_gradient,expected,atol=1e-8,rtol=3e-5)
        with torch.no_grad():native,_=wrapper(history_inputs)
        torch.testing.assert_close(native,loss,atol=0,rtol=0)
        del wrapper,loss,native
    assert len(set(values))==1
    write_json(RUN/'live_graph_audit.json',dict(status='passed',set_checks=set_checks,ce_checks=ce_checks,
        history_native_runner_wrapper_values=values,
        source='real epoch-4 network, one training frame, eval graph',
        verified='Raw-logit sibling views share their actual parent; injected CE correction reaches and cancels the intended derivative, preserving direct spatial/empty derivatives.'))
    print('Real-model source gradients, CE view cancellation and preserved direct gradients verified.')


if __name__=='__main__':main()
