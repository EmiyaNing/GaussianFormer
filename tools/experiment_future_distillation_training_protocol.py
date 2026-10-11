#!/usr/bin/env python3
"""S0 continuation: paired BN, loss-scope, and geometry-gradient controls."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from loss import OPENOCC_LOSS
from tools.future_distillation_causal_common import deterministic, selected_images, future_loss
from tools.future_distillation_training_protocol_common import (
    TrainingProtocolStudent, set_training_mode, schedule_factor, bn_snapshot, bn_unchanged)
from tools.train_future_predictable_full import image_loader, optimizer_for, atomic_torch_save, checkpoint, evaluate, alignment_loss
from tools.diagnose_teacher_alignment import seed_all, tree_to
from tools.experiment_future_predictable_information import save_json
from tools.evaluate_future_predictable_native import relative_to_baseline
from tools.summarize_future_distillation_training_protocol import write_report

ARMS=('full_bn_train','full_bn_fixed','future_bn_train','future_bn_fixed',
      'full_bn_fixed_detached_geom','future_bn_fixed_detached_geom')


def build(cfg, strict):
    spec=copy.deepcopy(strict.model).to_dict();spec.pop('type')
    spec['img_backbone']['frozen_stages']=-1;spec['img_backbone']['norm_eval']=False
    instance=TrainingProtocolStudent(**spec).cuda()
    original=torch.load(cfg.student_checkpoint,map_location='cpu',weights_only=False,mmap=True)['state_dict']
    mismatch=instance.load_state_dict(original,strict=False)
    missing={name for name,_ in instance.named_parameters() if name.startswith('future_adapter.')}
    if set(mismatch.missing_keys)!=missing or mismatch.unexpected_keys:raise RuntimeError(str(mismatch))
    instance.set_epoch(100);instance.requires_grad_(True);instance.eval()
    return instance


def save(root, report):
    save_json(root/'results.json',report)
    write_report(root)


@torch.no_grad()
def initialization_check(instance,cfg,dataset,baseline_rows):
    from tools.experiment_future_response_distillation import build_student
    reference=build_student(cfg,Config.fromfile(cfg.student_model_config))
    data=tree_to(next(iter(image_loader(dataset,0))),'cuda');imgs=data.pop('img')
    expected=reference(imgs=imgs,metas=data);actual=instance(imgs=imgs,metas=data)
    for step in range(6):
        for field in ('future_pred_points','future_pred_logits'):
            torch.testing.assert_close(actual[field][step],expected[field][step],atol=1e-5,rtol=0)
        count=(720+sum(instance.future_queries[:step+1]))*48
        if actual['future_pred_points'][step].shape[1]!=count:raise RuntimeError('Future-born query count changed')
    if baseline_rows.exists():
        with np.load(baseline_rows) as new,np.load(Path(cfg.causal_root)/'s1_baseline.npz') as old:
            for name in ('hist','intersection','union'):
                if not np.array_equal(new[name],old[name][:len(new[name])]):raise RuntimeError('Native baseline parity failed')
    del reference,expected,actual,data,imgs;torch.cuda.empty_cache()
    return dict(six_step_points_logits=True,future_born_preserved=True,original_native_metrics=True)


def run_arm(cfg,strict,root,report,manifest,train_images,val_images,arm,seed,supervision='task'):
    label=f'{arm}_{supervision}_seed{seed}';destination=root/label;destination.mkdir(exist_ok=True)
    if report.get('arms',{}).get(label,{}).get('status')=='complete':return
    seed_all(seed);instance=build(cfg,strict)
    instance.detach_future_geometry=arm.endswith('detached_geom')
    fixed_bn='bn_fixed' in arm;future_only=arm.startswith('future_')
    original_bn=bn_snapshot(instance)
    optimizer=optimizer_for(instance,cfg);objective=OPENOCC_LOSS.build(strict.loss).cuda()
    generator=torch.Generator(device='cuda').manual_seed(seed+917)
    row=report['arms'].setdefault(label,dict(status='running',steps={},arm=arm,seed=seed,
                                           supervision=supervision,fixed_bn=fixed_bn,future_only=future_only,
                                           detached_geometry=instance.detach_future_geometry))
    saved_file=destination/'latest.pth';start=0
    if saved_file.exists():
        saved=torch.load(saved_file,map_location='cpu',weights_only=False)
        if saved['manifest']!=manifest:raise RuntimeError('Training protocol resume manifest changed')
        instance.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['steps'];random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng']);generator.set_state(saved['shuffle_rng'])
        # Recover validation that was interrupted after checkpoint commit.
        if start in cfg.protocol_eval_steps and str(start) not in row['steps']:
            row['steps'][str(start)]=relative_to_baseline(evaluate(instance,val_images.dataset,cfg,destination/f'step_{start}.npz'),report['baseline'])
    count=len(train_images.dataset);audit=[];torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    for epoch in range(start//count,(cfg.protocol_updates+count-1)//count):
        order=np.random.default_rng(seed+epoch*1009).permutation(count).tolist()
        offset=start%count if epoch==start//count else 0
        for sample_position,data in zip(order[offset:],image_loader(train_images.dataset,0,order[offset:])):
            if start>=cfg.protocol_updates:break
            set_training_mode(instance,fixed_bn)
            for group in optimizer.param_groups:
                group['lr']=group['initial_lr']*schedule_factor(start,cfg.full_tune_warmup_steps,cfg.protocol_schedule_steps)
            data=tree_to(data,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):output=instance(imgs=data.pop('img'),metas=data)
            if future_only:task=future_loss(objective,output);details={}
            else:task,details=objective(output)
            align=task.new_zeros(())
            if supervision!='task':
                index=manifest['train']['indices'][sample_position];scene=manifest['train']['scenes'][sample_position]
                path=Path(cfg.causal_reference_root)/'teacher_features/train'/f'{index:06d}.pth'
                teacher=torch.load(path,map_location='cpu',weights_only=False)
                if (teacher['index'],teacher['scene'],teacher['checkpoint_sha256'])!=(index,scene,cfg.student_checkpoint_sha256):raise RuntimeError('Teacher identity changed')
                align=alignment_loss(instance,output,teacher,'teacher_shuffled' if supervision=='shuffled' else 'teacher_all',generator)
            value=task+cfg.full_tune_feature_weight*align
            if not torch.isfinite(value):raise FloatingPointError(label)
            optimizer.zero_grad(set_to_none=True);value.backward()
            if start==0:
                row['gradient_audit']=dict(trainable_parameters=sum(p.numel() for p in instance.parameters() if p.requires_grad),
                                          frozen_names=[n for n,p in instance.named_parameters() if not p.requires_grad],
                                          no_gradient_names=[n for n,p in instance.named_parameters() if p.grad is None])
            norm=torch.nn.utils.clip_grad_norm_(instance.parameters(),5.,error_if_nonfinite=True);optimizer.step();start+=1
            if start%32==0 or start in cfg.protocol_eval_steps:
                record=dict(step=start,task=float(task),align=float(align),unclipped_grad_norm=float(norm),
                            components={k:float(v) for k,v in details.items()})
                row.setdefault('loss_log',[]).append(record)
                print(f'[protocol {label}] {start}/{cfg.protocol_updates} task={float(task):.4f} grad={float(norm):.3f} elapsed={time.monotonic()-began:.0f}s',flush=True)
            if start%32==0 or start in cfg.protocol_eval_steps:
                atomic_torch_save(checkpoint(instance,optimizer,start//count,start%count,start,generator,manifest),saved_file)
            if start in cfg.protocol_eval_steps:
                row['steps'][str(start)]=relative_to_baseline(evaluate(instance,val_images.dataset,cfg,destination/f'step_{start}.npz'),report['baseline'])
                print('[protocol eval]',label,start,[round(row['steps'][str(start)]['metrics'][h]['mIoU'],3) for h in ('1s','2s','3s')],flush=True)
            row['progress']=start
            if start%32==0:save(root,report)
            del output,data,value,task,align
    row['bn_unchanged']=bn_unchanged(instance,original_bn)
    if fixed_bn and not row['bn_unchanged']:raise RuntimeError('Fixed BN buffers changed')
    row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    row['progress']=start
    row['status']='complete';save(root,report)
    del instance,optimizer,objective;torch.cuda.empty_cache()


@torch.no_grad()
def bn_only(cfg,strict,root,report,manifest,train_images,val_images):
    if report.get('bn_only',{}).get('status')=='complete':return
    seed_all(cfg.protocol_seeds[0]);instance=build(cfg,strict);instance.requires_grad_(False)
    parameters={n:p.detach().cpu().clone() for n,p in instance.named_parameters()}
    row=report.setdefault('bn_only',dict(status='running',steps={}))
    count=len(train_images.dataset);completed=0
    # No optimizer, no task/teacher gradients: image-backbone BN buffers only.
    for epoch in range((cfg.protocol_updates+count-1)//count):
        order=np.random.default_rng(cfg.protocol_seeds[0]+epoch*1009).permutation(count).tolist()
        for data in image_loader(train_images.dataset,0,order):
            if completed>=cfg.protocol_updates:break
            instance.train();data=tree_to(data,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):instance.extract_img_feat(imgs=data['img'],metas=data)
            completed+=1
            if completed in cfg.protocol_eval_steps:
                row['steps'][str(completed)]=relative_to_baseline(evaluate(instance,val_images.dataset,cfg,root/f'bn_only_step{completed}.npz'),report['baseline'])
                save(root,report)
                print('[BN buffers only]',completed,[round(row['steps'][str(completed)]['metrics'][h]['mIoU'],3) for h in ('1s','2s','3s')],flush=True)
    if not all(torch.equal(p.detach().cpu(),parameters[n]) for n,p in instance.named_parameters()):raise RuntimeError('BN-only altered a parameter')
    row['parameters_unchanged']=True;row['status']='complete';save(root,report)
    del instance,parameters;torch.cuda.empty_cache()


def main(args):
    cfg=Config.fromfile(args.config);deterministic()
    torch.cuda.set_per_process_memory_fraction(.75)
    if args.smoke:
        cfg.protocol_train_samples=cfg.protocol_val_samples=2;cfg.protocol_updates=2;cfg.protocol_eval_steps=[2];cfg.protocol_seeds=[20260930]
    root=Path(args.output_dir or (cfg.protocol_root+'_smoke2' if args.smoke else cfg.protocol_root));root.mkdir(parents=True,exist_ok=True)
    lock=(root/'train.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    strict=Config.fromfile(cfg.student_model_config)
    train_images,train_manifest=selected_images(cfg,'train',cfg.protocol_train_samples)
    val_images,val_manifest=selected_images(cfg,'val',cfg.protocol_val_samples)
    if set(train_manifest['scenes'])&set(val_manifest['scenes']):raise RuntimeError('Scene overlap')
    manifest=dict(train=train_manifest,val=val_manifest,updates=cfg.protocol_updates,seeds=cfg.protocol_seeds,
                  checkpoint_sha256=cfg.student_checkpoint_sha256,config=args.config,
                  schedule=dict(warmup=cfg.full_tune_warmup_steps,total=cfg.protocol_schedule_steps),
                  evaluation_steps=cfg.protocol_eval_steps,arms=cfg.protocol_arms)
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=manifest:raise RuntimeError('Protocol manifest differs')
    save_json(root/'manifest.json',manifest)
    report=json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else dict(status='running',arms={},manifest=str(root/'manifest.json'))
    report['status']='running'
    if 'baseline' not in report:
        instance=build(cfg,strict);report['baseline']=evaluate(instance,val_images.dataset,cfg,root/'baseline.npz')
        report['initialization_parity']=initialization_check(instance,cfg,val_images.dataset,root/'baseline.npz')
        del instance;torch.cuda.empty_cache();save(root,report)
    supervisions=args.supervisions or [args.supervision]
    if 'task' in supervisions:bn_only(cfg,strict,root,report,manifest,train_images,val_images)
    for arm in args.arms or cfg.protocol_arms:
        for supervision in supervisions:
            for seed in cfg.protocol_seeds:run_arm(cfg,strict,root,report,manifest,train_images,val_images,arm,seed,supervision)
    if any(s!='task' for s in supervisions):
        from tools.audit_future_distillation_training_protocol import audit
        audit(cfg,strict,root,manifest,val_images.dataset,build)
    report['status']='complete';report['completed_request']=dict(arms=args.arms or cfg.protocol_arms,supervisions=supervisions)
    save(root,report);print('[training protocol complete]',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_training_protocol.py')
    parser.add_argument('--output-dir');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--arms',nargs='+',choices=ARMS)
    parser.add_argument('--supervision',choices=['task','teacher','shuffled'],default='task')
    parser.add_argument('--supervisions',nargs='+',choices=['task','teacher','shuffled'])
    main(parser.parse_args())
