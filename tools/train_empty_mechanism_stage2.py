"""Paired AdamW-state branches for stage-two diagnostics and 64-step continuations."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage2 import (
    RUN, CONFIG, initialize, read, write_json, file_sha256, seed_all,
    build_loader, to_cuda, network_output, ce_parts, project_set, measure, Frame,
)
from tools.empty_mechanism_stage1 import modules as probe_modules

VARIANTS=["B0","B1","B2","B3","half_set","project_set","norm_set","zero_current","no_all_set"]


def build(epoch, mode="full"):
    from mmengine import Config
    from mmengine.optim import build_optim_wrapper
    from mmseg.models import build_segmentor
    from loss import OPENOCC_LOSS
    from timm.scheduler import CosineLRScheduler
    import model
    cfg=Config.fromfile(CONFIG)
    checkpoint=torch.load(ROOT/initialize()["checkpoint"][str(epoch)]["path"],
                          map_location="cpu",weights_only=False)
    net=build_segmentor(cfg.model)
    net.init_weights()
    net=net.cuda()
    optimizer=build_optim_wrapper(net,cfg.optimizer)
    scheduler=CosineLRScheduler(optimizer,t_initial=4096*cfg.max_epochs,
        lr_min=cfg.optimizer.optimizer.lr*cfg.get("min_lr_ratio",.1),
        warmup_t=cfg.get("warmup_iters",500),warmup_lr_init=cfg.get("warmup_lr_init",1e-6),
        t_in_epochs=False)
    net.load_state_dict(checkpoint["state_dict"],strict=True)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler"]))
    if mode=="classifier":
        prefix=f"encoder.layers.{len(net.encoder.layers)-1}.cls_branch."
        for name,v in net.named_parameters():
            v.requires_grad_(name.startswith(prefix))
        net.eval()
    else:
        net.train()
    set_cfg=copy.deepcopy(cfg.loss.loss_cfgs[0])
    set_loss=OPENOCC_LOSS.build(set_cfg).cuda()
    mods=probe_modules(cfg)
    return net,optimizer,scheduler,cfg,checkpoint,set_loss,mods


def loss_terms(net,output,batch,set_loss,mods):
    inp=dict(metas=batch)
    for key in ["opus_pred_points","opus_pred_logits","opus_point_valid_masks"]:
        inp[key]=output[key]
    components,_=set_loss.components(inp)
    last=components[f"stage_{len(output['opus_pred_logits'])-1}/classification"]
    all_cls=sum(v for k,v in components.items() if k.endswith("/classification"))
    voxel=dict(pred_occ=output["pred_occ"],sampled_xyz=output["sampled_xyz"],
               sampled_label=output["sampled_label"],occ_mask=output["occ_cam_mask"])
    ce=mods[1](voxel)
    lov=mods[2](voxel)
    labels=output["sampled_label"][0].long()
    mask=output["occ_cam_mask"][0].flatten().bool()&(labels>=0)&(labels<18)
    ce_empty=0
    for scores in output["pred_occ"]:
        _,part=ce_parts(scores[:,:,mask],labels[mask],mods[1].class_weights,
                        mods[1].loss_voxel_ce_weight)
        ce_empty=ce_empty+part/len(output["pred_occ"])
    raw=output["opus_pred_logits"][-1]
    gu=torch.autograd.grad(last,raw,retain_graph=True)[0]
    projected,error=project_set(raw.detach(),gu.detach())
    ratio=float(projected.double().norm()/gu.detach().double().norm().clamp_min(1e-30))
    surrogate=(raw*projected).sum()
    projected_loss=last.detach()+surrogate-surrogate.detach()
    return dict(base=sum(components.values())+ce+lov,last=last,empty=ce_empty,
                project=projected_loss,norm=last*ratio,all_cls=all_cls), dict(
        original_loss=float((sum(components.values())+ce+lov).detach()),
        last_cls=float(last.detach()),empty_ce=float(ce_empty.detach()),
        projection_error=error,norm_ratio=ratio,supervised_stages=output["supervised_stage_indices"])


def modified_loss(terms,variant):
    base,last,empty=terms["base"],terms["last"],terms["empty"]
    if variant=="B0": return base
    if variant=="B1": return base-last
    if variant=="B2": return base-empty
    if variant=="B3": return base-last-empty
    if variant=="half_set": return base-.5*last
    if variant=="project_set": return base-last+terms["project"]
    if variant=="norm_set": return base-last+terms["norm"]
    if variant=="no_all_set": return base-terms["all_cls"]
    raise KeyError(variant)


def training_dataset():
    from mmengine import Config
    from dataset import OPENOCC_DATASET
    cfg=Config.fromfile(CONFIG)
    return OPENOCC_DATASET.build(copy.deepcopy(cfg.train_dataset_config))


def load_item(dataset,index,seed):
    from dataset.utils import custom_collate_fn_temporal
    seed_all(seed)
    return to_cuda(custom_collate_fn_temporal([dataset[index]]))


def restore(net,optimizer,scheduler,checkpoint,mode):
    net.load_state_dict(checkpoint["state_dict"],strict=True)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler"]))
    optimizer.zero_grad()
    net.eval() if mode=="classifier" else net.train()


@torch.no_grad()
def evaluation(net,mods,selected,reference_dir=None,capture=False):
    was_training=net.training
    net.eval()
    result=[]
    _,loader,_=build_loader("val",[m["index"] for m in selected])
    for k,(meta,batch) in enumerate(zip(selected,loader)):
        batch=to_cuda(batch)
        out=network_output(net,batch)
        frame=Frame(net,out,batch,mods)
        if reference_dir is not None:
            path=reference_dir/f'{meta["index"]:04d}.npz'
            if capture:
                reference_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,pred=frame.pred0.cpu().numpy().astype(np.int8),
                                    fg=frame.fg0.cpu().numpy().astype(np.int8))
            else:
                old=np.load(path);y=frame.y
                old_pred=torch.tensor(old["pred"],device=y.device)
                old_fg=torch.tensor(old["fg"],device=y.device)
                frame.correct=(y<17)&(old_pred==y)
                frame.recoverable=(y<17)&(old_fg==y)&(old_pred==17)
                frame.cohorts=dict(correct=frame.correct,recoverable=frame.recoverable,true_empty=y==17)
        value=measure(frame,frame.base,frame.sem)
        result.append(dict(frame=meta,outcome=value,empty_bias=float(net.head.empty_bias),
                           empty_scalar=float(net.head.empty_scalar)))
        del frame,out,batch
    net.train(was_training)
    return result


def single_evaluation(net,mods,batch,original):
    with torch.no_grad():
        net.eval()
        out=network_output(net,batch)
        current=Frame(net,out,batch,mods)
        current_support=current.support
        value=measure(original,current.base,current.sem,support=current_support)
        args=current.args
        errors={k:float((args[j]-original.args[j]).abs().max())
                for k,j in [("means",0),("opacity",1),("scales",3),("inverse_covariance",4)]}
        value["empty_bias"]=float(net.head.empty_bias)
        value["empty_scalar"]=float(net.head.empty_scalar)
        return value,errors


def add_gradients(cache,key,gradients,parameters):
    if key not in cache:
        cache[key]=[torch.zeros_like(p,device="cpu") if g is not None else None
                    for p,g in zip(parameters,gradients)]
    for i,g in enumerate(gradients):
        if g is not None:
            if cache[key][i] is None:
                cache[key][i]=torch.zeros_like(g,device="cpu")
            cache[key][i].add_(g.detach().cpu()/4)


def install_gradients(net,cache,variant,parameters):
    coefficients={
        "B0":{"base":1},"B1":{"base":1,"last":-1},
        "B2":{"base":1,"empty":-1},"B3":{"base":1,"last":-1,"empty":-1},
        "half_set":{"base":1,"last":-.5},
        "project_set":{"base":1,"last":-1,"project":1},
        "norm_set":{"base":1,"last":-1,"norm":1},
        "no_all_set":{"base":1,"all_cls":-1},"zero_current":{"base":0},
    }[variant]
    for i,p in enumerate(parameters):
        if cache["base"][i] is None:
            p.grad=None
            continue
        value=torch.zeros_like(p,device="cpu")
        for key,mult in coefficients.items():
            if cache[key][i] is not None:
                value.add_(cache[key][i],alpha=mult)
        p.grad=value.to(p.device)


def one_step(epoch,mode,limit=0,pilot=False):
    p=initialize()
    net,opt,sched,cfg,checkpoint,set_loss,mods=build(epoch,mode)
    parameters=[v for v in net.parameters() if v.requires_grad]
    train=training_dataset()
    val,_,_=build_loader("val")
    dest=RUN/("adamw_pilot" if pilot else "adamw")/f"epoch{epoch}"/mode
    dest.mkdir(parents=True,exist_ok=True)
    pairs=p["pairs"][:limit] if limit else p["pairs"]
    start=time.monotonic()
    for index,pair in enumerate(pairs):
        path=dest/f"pair_{index:02d}.json"
        if path.exists():continue
        restore(net,opt,sched,checkpoint,mode)
        net.eval()
        source=load_item(train,pair["source"]["index"],1928+index*4)
        heldout=load_item(val,pair["heldout"]["index"],2928+index)
        with torch.no_grad():
            source_out=network_output(net,source)
            source_frame=Frame(net,source_out,source,mods)
            heldout_out=network_output(net,heldout)
            heldout_frame=Frame(net,heldout_out,heldout,mods)
        bases=dict(source=measure(source_frame,source_frame.base,source_frame.sem),
                   heldout=measure(heldout_frame,heldout_frame.base,heldout_frame.sem))
        cache={};micro_audits=[]
        net.eval() if mode=="classifier" else net.train()
        for micro in range(4):
            batch=load_item(train,pair["source"]["index"],1928+index*4+micro)
            out=network_output(net,batch)
            terms,audit=loss_terms(net,out,batch,set_loss,mods)
            audit.update(raw_sum=float(out["opus_pred_logits"][-1].detach().double().sum()),
                         image_sum=float(batch["img"].double().sum()))
            micro_audits.append(audit)
            for name in ["base","last","empty","project","norm","all_cls"]:
                # Preserve the original reentrant activation-checkpoint path.
                opt.zero_grad()
                terms[name].backward(retain_graph=True)
                grads=[v.grad for v in parameters]
                add_gradients(cache,name,grads,parameters)
                del grads
            del terms,out,batch
        trained_buffers={name:value.detach().cpu().clone() for name,value in net.named_buffers()}
        parity={}
        if pilot:
            # Compare against the actual unmodified MultiLoss + backward runner.
            from loss import OPENOCC_LOSS
            original_loss=OPENOCC_LOSS.build(cfg.loss).cuda()
            restore(net,opt,sched,checkpoint,mode)
            direct_audits=[]
            for micro in range(4):
                batch=load_item(train,pair["source"]["index"],1928+index*4+micro)
                out=network_output(net,batch)
                inputs=dict(metas=batch,global_iter=checkpoint["global_iter"]+micro)
                for key,value in cfg.loss_input_convertion.items():inputs[key]=out[value]
                loss,_=original_loss(inputs)
                direct_audits.append(dict(loss=float(loss.detach()),
                    raw_sum=float(out["opus_pred_logits"][-1].detach().double().sum()),
                    image_sum=float(batch["img"].double().sum())))
                (loss/4).backward()
                del out,batch,loss,inputs
            max_error=0.
            debug=[]
            names={id(v):n for n,v in net.named_parameters()}
            for v,expected in zip(parameters,cache["base"]):
                if expected is None:
                    assert v.grad is None
                else:
                    max_error=max(max_error,float((v.grad.cpu()-expected).abs().max()))
                    debug.append(dict(name=names[id(v)],max_error=float((v.grad.cpu()-expected).abs().max()),
                        norm=float(expected.double().norm()),error_norm=float((v.grad.cpu()-expected).double().norm())))
            write_json(dest/"parity_debug.json",dict(cached=micro_audits,direct=direct_audits,gradients=debug))
            direct_grad=[v.grad.detach().cpu().clone() if v.grad is not None else None for v in parameters]
            restore(net,opt,sched,checkpoint,mode)
            for micro in range(4):
                batch=load_item(train,pair["source"]["index"],1928+index*4+micro)
                out=network_output(net,batch)
                inputs=dict(metas=batch,global_iter=checkpoint["global_iter"]+micro)
                for key,value in cfg.loss_input_convertion.items():inputs[key]=out[value]
                loss,_=original_loss(inputs)
                (loss/4).backward()
                del out,batch,loss,inputs
            noise2=0.;difference2=0.;reference2=0.
            for v,expected,direct in zip(parameters,cache["base"],direct_grad):
                if expected is not None:
                    noise2+=float((v.grad.cpu()-direct).double().square().sum())
                    difference2+=float((direct-expected).double().square().sum())
                    reference2+=float(direct.double().square().sum())
            native_relative=(noise2/max(reference2,1e-30))**.5
            decomposition_relative=(difference2/max(reference2,1e-30))**.5
            assert decomposition_relative<max(5*native_relative,5e-5), (decomposition_relative,native_relative)
            parity=dict(original_multiloss_accumulated_gradient_max_error=max_error,
                        original_replay_relative_gradient_noise=native_relative,
                        cached_vs_original_relative_gradient_error=decomposition_relative,
                        activation_checkpointing="original reentrant implementation",
                        accumulation=4)
            del original_loss
        rows=[]
        for variant in VARIANTS:
            restore(net,opt,sched,checkpoint,mode)
            for name,value in net.named_buffers():value.copy_(trained_buffers[name])
            for micro in range(3):sched.step_update(checkpoint["global_iter"]+micro)
            install_gradients(net,cache,variant,parameters)
            grad_norm=torch.nn.utils.clip_grad_norm_(parameters,cfg.grad_max_norm,error_if_nonfinite=True)
            opt.step();opt.zero_grad()
            sched.step_update(checkpoint["global_iter"]+3)
            row=dict(condition=variant,gradient_norm_before_clip=float(grad_norm),outcomes={},frozen_errors={})
            for role,batch,original in [("source",source,source_frame),("heldout",heldout,heldout_frame)]:
                value,errors=single_evaluation(net,mods,batch,original)
                row["outcomes"][role]=value;row["frozen_errors"][role]=errors
                if mode=="classifier":
                    assert max(errors.values())==0
                    for name,cohort in value["cohorts"].items():
                        assert cohort["empty_sum"]==bases[role]["cohorts"][name]["empty_sum"]
            rows.append(row)
        write_json(path,dict(status="complete",epoch=epoch,mode=mode,pair=pair,
            baselines=bases,interventions=rows,microbatch_audits=micro_audits,
            parity=parity,
            optimizer_state_restored=True,amp=cfg.get("amp",False),accumulation=4,
            randomness="paired reconstructed augmentations; original RNG absent from checkpoint",
            protocol_sha256=file_sha256(RUN/"protocol.json"),source_sha256=file_sha256(__file__)))
        del cache,source,heldout,source_frame,heldout_frame,source_out,heldout_out
        print(f"adamw epoch={epoch} mode={mode} pair={index+1}/{len(pairs)} "
              f"elapsed={time.monotonic()-start:.1f}s",flush=True)


def short(epoch,variant,pilot=False):
    p=initialize()
    dest=RUN/("short_pilot" if pilot else "short")/f"epoch{epoch}"/variant
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"complete.json").exists():return
    seed_all(928+epoch)
    net,opt,sched,cfg,checkpoint,set_loss,mods=build(epoch,"full")
    original_loss=None
    if variant=="B0_reference":
        from loss import OPENOCC_LOSS
        original_loss=OPENOCC_LOSS.build(cfg.loss).cuda()
    dataset=training_dataset()
    rng=np.random.default_rng(928+epoch)
    sequence=rng.permutation(len(dataset))[:256].tolist()
    selected=p["scene_samples"]["val"][:2] if pilot else p["scene_samples"]["val"]
    baselines=evaluation(net,mods,selected,reference_dir=dest/"cohorts",capture=True)
    write_json(dest/"baseline.json",dict(records=baselines))
    endpoints=[1] if pilot else [1,16,64]
    start=time.monotonic();global_iter=checkpoint["global_iter"]
    trace=[];net.train();os.environ["amp"]="false"
    for step in range(1,max(endpoints)+1):
        opt.zero_grad()
        micro_audits=[]
        for micro in range(4):
            k=(step-1)*4+micro
            batch=load_item(dataset,sequence[k],928+epoch*100000+k)
            out=network_output(net,batch)
            if original_loss is None:
                terms,audit=loss_terms(net,out,batch,set_loss,mods)
                loss=modified_loss(terms,variant)/4
            else:
                inputs=dict(metas=batch,global_iter=global_iter)
                for key,value in cfg.loss_input_convertion.items():inputs[key]=out[value]
                total,_=original_loss(inputs)
                audit=dict(original_loss=float(total.detach()),projection_error=0.,norm_ratio=1.)
                loss=total/4
                terms=inputs
            assert torch.isfinite(loss)
            loss.backward()
            if micro<3:
                sched.step_update(global_iter)
            global_iter+=1
            micro_audits.append(audit)
            del terms,out,batch,loss
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),cfg.grad_max_norm,error_if_nonfinite=True)
        opt.step();opt.zero_grad()
        sched.step_update(global_iter-1)
        # Original runner steps scheduler after every microbatch, including update microbatch.
        # Parameter LR for this update is the one after the preceding microbatch.
        trace.append(dict(step=step,grad_norm=float(norm),
                          loss=float(np.mean([a["original_loss"] for a in micro_audits])),
                          projection_error=max(a["projection_error"] for a in micro_audits),
                          set_norm_ratio=float(np.mean([a["norm_ratio"] for a in micro_audits]))))
        if step in endpoints:
            records=evaluation(net,mods,selected,reference_dir=dest/"cohorts")
            write_json(dest/f"step_{step:02d}.json",dict(epoch=epoch,condition=variant,step=step,
                records=records,trace=trace,source_sha256=file_sha256(__file__),
                protocol_sha256=file_sha256(RUN/"protocol.json")))
            if step==max(endpoints):
                torch.save(dict(state_dict=net.state_dict(),epoch_origin=epoch,optimizer_steps=step,
                                condition=variant),dest/"final.pth")
        if step==1 or step%8==0:
            print(f"short epoch={epoch} condition={variant} step={step}/{max(endpoints)} "
                  f"elapsed={time.monotonic()-start:.1f}s",flush=True)
    write_json(dest/"complete.json",dict(status="complete",epoch=epoch,condition=variant,
        endpoints=endpoints,sequence=sequence,elapsed_seconds=time.monotonic()-start,
        source_sha256=file_sha256(__file__),protocol_sha256=file_sha256(RUN/"protocol.json"),
        reconstructed_randomness=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["adamw","short"])
    parser.add_argument("--epoch",type=int,default=5)
    parser.add_argument("--mode",choices=["classifier","full"],default="full")
    parser.add_argument("--condition",choices=VARIANTS+["B0_reference"],default="B0")
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--pilot",action="store_true")
    a=parser.parse_args()
    if a.stage=="adamw":one_step(a.epoch,a.mode,a.limit,a.pilot)
    else:short(a.epoch,a.condition,a.pilot)


if __name__=="__main__":
    main()
