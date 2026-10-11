#!/usr/bin/env python3
"""S2 native-forecast screens: strength, direction, head adaptation, response."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from loss import OPENOCC_LOSS
from tools.future_distillation_causal_common import *
from tools.experiment_future2current import _future_window
from tools.train_future_predictable_full import atomic_torch_save
from tools.diagnose_teacher_alignment import loader_for

ARMS=('task_heads_train','feature_heads_train','damped_heads_train','project_heads_train',
      'task_heads_frozen','feature_heads_frozen','response_heads_train')


@torch.no_grad()
def prepare_native_teacher_grids(cfg,root,manifest):
    instance,_=model_for(cfg); instance.requires_grad_(False)
    for split in ('train','val'):
        count=len(manifest[split]['indices'])
        images,selected=loader_for(cfg,split,count,cfg.causal_workers,cfg.identity_sample_seed)
        if selected != manifest[split]: raise RuntimeError('Teacher image selection differs')
        paths=[root/'teacher_grids'/split/f'{index:06d}.pth' for index in selected['indices']]
        missing=[i for i,p in enumerate(paths) if not p.exists()]
        if not missing: continue
        for i,data in zip(missing,image_loader(images.dataset,0,missing)):
            data=tree_to(data,'cuda'); imgs=data.pop('img')
            ego=instance.plan_head(data['temporal_ego_states'].float().reshape(1,-1)).unsqueeze(1)
            grids={}
            for h in HORIZONS:
                selected_imgs,metas=_future_window(data,imgs,h)
                state=dict(imgs=selected_imgs,metas=metas,points=None)
                state.update(instance.extract_img_feat(**state)); state.update(instance.lifter(**state))
                state['query_points']=state['query_points']*((torch.tanh(instance.points_scale(ego))+1.)*.35+.8)
                state.update(instance.encoder(**state)); state.update(instance.head(**state))
                grids[h]=state['final_occ_grid'][0].cpu().to(torch.uint8)
            paths[i].parent.mkdir(parents=True,exist_ok=True)
            atomic_torch_save(dict(index=selected['indices'][i],scene=selected['scenes'][i],grids=grids,
                              checkpoint_sha256=cfg.student_checkpoint_sha256),paths[i])
            if (i+1)%32==0 or i==missing[-1]: print(f'[teacher native grids] {split} {i+1}/{count}',flush=True)
    del instance
    torch.cuda.empty_cache()


def native_response_loss(instance,output,record):
    losses=[]
    for h in HORIZONS:
        points=output['future_pred_points'][h-1][0].detach()
        logits=output['future_pred_logits'][h-1][0]
        index=torch.floor((points-points.new_tensor(instance.pc_range[:3]))/.4).long()
        grid=record['grids'][h].cuda()
        inside=((index>=0)&(index<index.new_tensor(grid.shape))).all(-1)
        if not inside.any(): continue
        selected=index[inside]
        labels=grid[selected[:,0],selected[:,1],selected[:,2]].long()
        target=F.one_hot(labels,num_classes=18)[:,:17].to(logits)
        # Teacher empty voxels map to all-zero sigmoid targets, not class 17.
        losses.append(F.binary_cross_entropy_with_logits(logits[inside],target))
    return torch.stack(losses).mean() if losses else output['pred_traj'].sum()*0.


def controlled_gradients(task_grads,aux_grads,weight,project=False,max_rescale=2.):
    shared=[(t,a) for t,a in zip(task_grads,aux_grads) if t is not None and a is not None]
    device=next(g for g in task_grads if g is not None).device
    zero=torch.zeros((),device=device)
    norm_task=sum((t.square().sum() for t,a in shared),zero)
    norm_aux=sum((a.square().sum() for t,a in shared),zero)
    dot=sum(((t*a).sum() for t,a in shared),zero)
    cosine=dot/(norm_task*norm_aux).sqrt().clamp_min(1e-30)
    coefficient=(dot/norm_task.clamp_min(1e-30)).clamp_max(0.) if project else zero
    projected=[(a-coefficient*t if a is not None and t is not None else a) for t,a in zip(task_grads,aux_grads)]
    remaining=sum((a.square().sum() for t,a in zip(task_grads,projected) if t is not None and a is not None),zero)
    scale=(norm_aux/remaining.clamp_min(1e-30)).sqrt().clamp_max(max_rescale) if project else zero+1.
    if float(remaining)<1e-20: scale=zero
    merged=[]
    for t,a in zip(task_grads,projected):
        merged.append(None if t is None and a is None else
                      (0. if t is None else t)+(0. if a is None else weight*scale*a))
    return merged,dict(cosine=float(cosine),rescale=float(scale),
                       raw_aux_norm=float(norm_aux.sqrt()),task_norm=float(norm_task.sqrt()),
                       effective_aux_norm=float(weight*scale*remaining.sqrt()))


@torch.no_grad()
def feature_validation(instance,cfg,root,manifest):
    values=[]
    for source in source_records(root,manifest,'val'):
        output=cached_rollout(instance,source)
        target=teacher_record(cfg,'val',source['index'],source['scene'])
        values.append(float(feature_loss(output,matched_targets(instance,output,target))))
    return float(np.mean(values))


def train_arm(cfg,root,manifest,report,arm,seed):
    seed_all(seed)
    instance,strict=model_for(cfg)
    heads_trainable=not arm.endswith('heads_frozen')
    configure_forecast(instance,heads_trainable)
    named=[(n,p) for n,p in instance.named_parameters() if p.requires_grad]
    parameters=[p for _,p in named]
    objective=OPENOCC_LOSS.build(strict.loss).cuda()
    optimizer=torch.optim.AdamW(parameters,lr=cfg.causal_lr,weight_decay=.01)
    sources=list(source_records(root,manifest['train'],'train'))
    order=np.random.default_rng(seed).permutation(len(sources)).tolist()
    label=f'{arm}_seed{seed}'; destination=root/'screens'/label; destination.mkdir(parents=True,exist_ok=True)
    gradients=[]; start_step=0
    latest=destination/'latest.pth'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['arm']!=arm or saved['seed']!=seed or saved['updates']!=cfg.causal_updates:
            raise RuntimeError('Screen resume differs')
        instance.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        start_step=saved['step']; gradients=saved['gradients']
    for step in range(start_step,cfg.causal_updates):
        source=sources[order[step%len(order)]]
        output=cached_rollout(instance,source)
        task=future_loss(objective,output)
        auxiliary=task.new_zeros(())
        is_feature=arm.startswith(('feature','damped','project'))
        if is_feature:
            teacher=teacher_record(cfg,'train',source['index'],source['scene'])
            auxiliary=feature_loss(output,matched_targets(instance,output,teacher))
        elif arm.startswith('response'):
            teacher=torch.load(root/'teacher_grids/train'/f'{source["index"]:06d}.pth',map_location='cpu',weights_only=False)
            if teacher['scene']!=source['scene']: raise RuntimeError('Response teacher identity differs')
            auxiliary=native_response_loss(instance,output,teacher)
        optimizer.zero_grad(set_to_none=True)
        weight=cfg.causal_feature_weight*(.1 if arm.startswith(('damped','project')) else 1.)
        if is_feature:
            gt=torch.autograd.grad(task,parameters,retain_graph=True,allow_unused=True)
            ga=torch.autograd.grad(auxiliary,parameters,allow_unused=True)
            merged,stats=controlled_gradients(gt,ga,weight,project=arm.startswith('project'))
            for parameter,gradient in zip(parameters,merged): parameter.grad=gradient
            gradients.append(stats)
            del gt,ga,merged
        else: (task+weight*auxiliary).backward()
        grad=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%64==0 or step+1==cfg.causal_updates:
            print(f'[S2 {label}] {step+1}/{cfg.causal_updates} task={float(task):.4f} aux={float(auxiliary):.4f} grad={float(grad):.3f}',flush=True)
            record=dict(state_dict=instance.state_dict(),optimizer=optimizer.state_dict(),arm=arm,seed=seed,
                        updates=cfg.causal_updates,step=step+1,gradients=gradients,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            atomic_torch_save(record,latest)
            report['arms'].setdefault(label,{})['progress']=step+1
            save_json(root/'s2_screens.json',report)
        del output,task,auxiliary
    metrics=evaluate_cached(instance,cfg,root,manifest['val'],destination/'native.npz')
    metrics=relative_to_baseline(metrics,report['baseline'])
    metrics.update(status='complete',checkpoint=str(latest),updates=cfg.causal_updates,seed=seed,
                   trainable_names=[n for n,_ in named],trainable_parameters=sum(p.numel() for p in parameters),
                   feature_validation_loss=feature_validation(instance,cfg,root,manifest['val']),
                   gradients=gradients,heads_trainable=heads_trainable)
    report['arms'][label]=metrics
    save_json(root/'s2_screens.json',report)
    del instance,optimizer,objective,sources
    torch.cuda.empty_cache()


def main(args):
    cfg=Config.fromfile(args.config); deterministic(); torch.cuda.set_per_process_memory_fraction(.3)
    root=Path(args.output_dir or (cfg.causal_root+'_smoke2' if args.smoke else cfg.causal_root)); root.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        cfg.causal_train_samples=cfg.causal_val_samples=2; cfg.causal_updates=2; cfg.causal_seeds=[20260930]
    manifest=json.loads((root/'manifest.json').read_text())
    if len(manifest['train']['indices'])!=cfg.causal_train_samples: raise RuntimeError('Screen training manifest differs')
    if 'response_heads_train' in args.arms: prepare_native_teacher_grids(cfg,root,manifest)
    report_path=root/'s2_screens.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(status='running',arms={},
                     protocol='future-only native task loss; frozen history encoder; all original future-born queries; paired seed/order; 512 updates')
    if 'baseline' not in report:
        initial,_=model_for(cfg); initial.requires_grad_(False)
        report['baseline']=evaluate_cached(initial,cfg,root,manifest['val'],root/'s2_baseline.npz')
        del initial; torch.cuda.empty_cache()
    save_json(report_path,report)
    for arm in args.arms:
        for seed in cfg.causal_seeds:
            if report['arms'].get(f'{arm}_seed{seed}',{}).get('status')=='complete': continue
            train_arm(cfg,root,manifest,report,arm,seed)
    report['status']='complete'; report['requested_arms']=args.arms
    save_json(report_path,report)
    print('[S2 complete]',report_path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_causal.py')
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    main(parser.parse_args())
