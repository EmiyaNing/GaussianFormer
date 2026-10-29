"""Fixed training-only gradient calibration for the two documented objectives."""
from pathlib import Path
import gc,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.bidirectional_coverage_experiment import *
from loss.bidirectional_union_coverage_loss import BidirectionalUnionCoverageLoss,geometry_render,sample_valid

def norm(gradient):return float(gradient.norm()) if gradient is not None else None

def pilot():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from loss import OPENOCC_LOSS
    c,net,opt,objective,state=build('union',training=True);del opt,state
    original=OPENOCC_LOSS.build(config('baseline').loss.loss_cfgs[0]).cuda()
    ds=OPENOCC_DATASET.build(c.train_dataset_config);wrapped=TrainingFrames(ds,c.seed)
    assert len(ds)==4096
    chosen=np.random.default_rng(c.seed).permutation(len(ds))[:4].tolist()
    modes={a:BidirectionalUnionCoverageLoss(mode=a).cuda() for a in ARMS}
    reg=net.encoder.layers[-1].reg_branch[-1].weight
    shared=net.encoder.layers[-1].ffn[0].weight
    parameter_names=[next(name for name,p in net.named_parameters() if p is target) for target in (reg,shared)]
    records=[];vectors=[]
    for index in chosen:
        seed_all(c.seed*1000000+index)
        batch=to_cuda(custom_collate_fn_temporal([wrapped[index]]))
        out=forward(net,batch);inp=loss_input(c,out,batch);inp['renderer_head']=net.head
        parts,_=original.components(inp)
        old=parts['stage_5/pred_to_gt']+parts['stage_5/gt_to_pred']
        base=torch.autograd.grad(old,(reg,shared),retain_graph=True)
        row=dict(index=index,original_final_geometry=float(old.detach()),
            original_reg_gradient=norm(base[0]),original_shared_gradient=norm(base[1]),arms={})
        vec={'base':[x.detach().cpu() for x in base]}
        common_cover=None
        for arm,module in modes.items():
            torch.cuda.synchronize();start=time.monotonic()
            cover,reverse,metrics=module.components(inp)
            torch.cuda.synchronize();seconds=time.monotonic()-start
            if common_cover is None:common_cover=cover.detach()
            else:torch.testing.assert_close(cover,common_cover,rtol=0,atol=0)
            g=out['gaussian']
            variables=(reg,shared,g.means,g.scales,g.rotations,g.semantics,g.opacities,
                       out['opus_pred_logits'][-1],net.head.empty_scalar,net.head.empty_bias)
            grads=[]
            for value in (cover,reverse):
                grad=torch.autograd.grad(value,variables,retain_graph=True,allow_unused=True)
                assert all(torch.isfinite(x).all() for x in grad if x is not None)
                assert all(x is None for x in grad[5:]),'Auxiliary loss has a direct semantic/opacity/empty gradient'
                assert all(x is not None for x in grad[:5])
                grads.append(grad)
            if arm=='decision':
                labels=batch['occ_label'].flatten().long();cam=batch['occ_cam_mask'].flatten().bool()
                ids=sample_valid(cam&(labels>=0)&(labels<18),4096,int(batch['union_seed'].flatten()[0]))
                z=geometry_render(net.head,g,batch['occ_xyz'].reshape(-1,3)[ids].float())
                reference=out['pred_occ'][-1][0].T[ids]
                torch.testing.assert_close(z,reference,rtol=1e-6,atol=1e-6)
                render_error=float((z-reference).abs().max())
            else:render_error=None
            names=('reg','shared','center','scale','rotation','semantic','opacity','raw_logits','empty_scalar','empty_bias')
            row['arms'][arm]=dict(cover=float(cover.detach()),reverse=float(reverse.detach()),seconds=seconds,
                cover_gradients={n:norm(x) for n,x in zip(names,grads[0])},
                reverse_gradients={n:norm(x) for n,x in zip(names,grads[1])},
                geometry_renderer_max_error=render_error,metrics={k:float(v) for k,v in metrics.items()})
            vec[arm]=[[x.detach().cpu() for x in grad[:2]] for grad in grads]
            del cover,reverse,grads
        records.append(row);vectors.append(vec);print(json.dumps(row),flush=True)
        del batch,out,inp,parts,old,base,g,vec
        gc.collect();torch.cuda.empty_cache()
    beta=float(np.median([v['decision'][0][0].norm().item()/max(v['decision'][1][0].norm().item(),1e-20) for v in vectors]))
    weights={};ratios={}
    for arm in ARMS:
        bw=1. if arm=='union' else beta
        candidates=[];rows=[]
        for v in vectors:
            pair=[v[arm][0][k]+bw*v[arm][1][k] for k in range(2)]
            ratio=[v['base'][k].norm().item()/max(pair[k].norm().item(),1e-20) for k in range(2)]
            candidates.append(min(ratio));rows.append(ratio)
        weights[arm]=float(np.median(candidates));ratios[arm]=rows
        assert np.isfinite(weights[arm]) and weights[arm]>0 and np.isfinite(bw) and bw>0
        path=ROOT/f'config/mini_bidirectional_coverage/small_{arm}_1ep.py'
        cfg=Config.fromfile(path);cfg.loss.loss_cfgs[-1].weight=weights[arm];cfg.loss.loss_cfgs[-1].reverse_weight=bw
        cfg.dump(path)
    # Full objective gradient on the first fixed train frame, after calibration.
    checks={};index=chosen[0];seed_all(c.seed*1000000+index)
    batch=to_cuda(custom_collate_fn_temporal([wrapped[index]]));out=forward(net,batch)
    inp=loss_input(c,out,batch);inp['renderer_head']=net.head
    base_result=objective.losses[0](inp);occ_result=objective.losses[1](inp)
    base_loss=base_result[0] if isinstance(base_result,tuple) else base_result
    occ=occ_result[0] if isinstance(occ_result,tuple) else occ_result
    for arm in ARMS:
        module=modes[arm];module.weight=weights[arm];module.reverse_weight=1. if arm=='union' else beta
        value,metrics=module(inp);total=base_loss+occ+value
        trainable=[p for p in net.parameters() if p.requires_grad]
        grad=torch.autograd.grad(total,trainable,retain_graph=True,allow_unused=True)
        assert all(torch.isfinite(x).all() for x in grad if x is not None)
        checks[arm]=dict(total=float(total.detach()),added=float(value.detach()),
            total_gradient_norm=float(torch.stack([x.norm() for x in grad if x is not None]).norm()))
        module.last_raw=None;del grad,total,value
    result=dict(status='passed',training_indices=chosen,records=records,
        weights=weights,decision_reverse_weight=beta,union_reverse_weight=1.,
        calibration='beta_decision = median cover/decision gradient norm on final reg weight; lambda_arm = median min(original/new reg gradient norm, original/new shared-FFN gradient norm)',
        parameter_groups=parameter_names,
        original_to_new_raw_gradient_ratios=ratios,joint_backward=checks,
        no_validation_tuning=True,common_cover_exact=True,no_direct_semantic_opacity_empty_gradient=True)
    write(RUN/'pilot.json',result);net.flush_memory_phase()
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','original_to_new_raw_gradient_ratios')}),flush=True)

if __name__=='__main__':pilot()
