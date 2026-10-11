#!/usr/bin/env python3
"""S0/S1: native-head training pathology, feature interventions, gradient audit."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from loss import OPENOCC_LOSS
from tools.future_distillation_causal_common import *
from tools.train_future_predictable_full import alignment_loss
from tools.experiment_future2current import baseline


@torch.no_grad()
def audit_training(cfg, root, count):
    images, manifest = selected_images(cfg, 'val', count)
    checkpoints = Path(cfg.causal_reference_root)
    trained = checkpoints / 'task_only_seed20260930/epoch_5.pth'
    if not trained.exists():
        return dict(status='waiting', reason='task-only epoch5 checkpoint absent')
    original = torch.load(cfg.student_checkpoint, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    full_weights = torch.load(trained, map_location='cpu', weights_only=False, mmap=True)['state_dict']
    cases = ('baseline', 'task_epoch5', 'reset_bn', 'disable_adapter', 'reset_heads', 'reset_perception', 'reset_bn_disable_adapter')
    result = dict(status='running', validation=manifest, checkpoint=str(trained), cases={})
    for case in cases:
        instance, _ = model_for(cfg)
        if case != 'baseline':
            weights = dict(full_weights)
            if case in ('reset_bn', 'reset_bn_disable_adapter'):
                for key in weights:
                    if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                        weights[key] = original[key]
            elif case == 'reset_heads':
                for key in weights:
                    if key.split('.')[0] in ('semantic', 'reg', 'velocity'):
                        weights[key] = original[key]
            elif case == 'reset_perception':
                for key in weights:
                    if key.split('.')[0] in ('img_backbone', 'img_neck', 'lifter', 'encoder', 'plan_head', 'points_scale'):
                        weights[key] = original[key]
            instance.load_state_dict(weights, strict=True)
            if case in ('disable_adapter', 'reset_bn_disable_adapter'):
                instance.future_adapter = torch.nn.Identity()
        instance.eval().requires_grad_(False)
        rows = []
        for data in image_loader(images.dataset, 0):
            data = tree_to(data, 'cuda')
            output = instance(imgs=data.pop('img'), metas=data)
            rows.append(prediction_rows(instance, output, cfg))
        destination = root / f's0_{case}.npz'
        save_rows(destination, rows)
        metrics = summarized_rows(rows, cfg)
        metrics['rows'] = str(destination)
        result['cases'][case] = (metrics if case == 'baseline' else relative_to_baseline(metrics, result['cases']['baseline']))
        save_json(root / 's0_training_audit.json', result)
        print('[S0]', case, [round(metrics['metrics'][h]['mIoU'], 3) for h in ('1s','2s','3s')], flush=True)
        del instance, output, rows
        torch.cuda.empty_cache()
    result['status'] = 'complete'
    save_json(root / 's0_training_audit.json', result)
    return result


@torch.no_grad()
def feature_intervention(cfg, root, manifest, count):
    instance, _ = model_for(cfg)
    instance.requires_grad_(False)
    rows, errors, matches = {}, {}, []
    for geometry in ('fixed', 'native_joint'):
        for alpha in cfg.causal_mix:
            key = f'{geometry}_mix{alpha:g}'
            rows[key], errors[key] = [], []
    reference_rows = []
    parity_checked = False
    for i, source in enumerate(source_records(root, manifest, 'val', count)):
        output = cached_rollout(instance, source)
        if not parity_checked:
            official = baseline(instance, source)
            for h in range(6):
                torch.testing.assert_close(output['future_pred_points'][h], official['points'][h], atol=1e-5, rtol=0)
                torch.testing.assert_close(output['future_pred_logits'][h], official['logits'][h], atol=1e-5, rtol=0)
            parity_checked = True
        reference_rows.append(prediction_rows(instance, output, cfg))
        teacher = teacher_record(cfg, 'val', source['index'], source['scene'])
        targets = matched_targets(instance, output, teacher)
        for h, (_, valid) in targets.items():
            state = output['future_query_states'][h-1]
            centers = instance._world(state['positions']).mean(2)[0]
            distances = torch.cdist(centers.float(), teacher['states'][h]['centers'][0].cuda().float()).min(-1).values
            matches.append(dict(index=source['index'], horizon=h, queries=len(valid), valid=int(valid.sum()),
                                mean_nearest_m=float(distances.mean()), median_nearest_m=float(distances.median())))
        for geometry in ('fixed', 'native_joint'):
            for alpha in cfg.causal_mix:
                changed = dict(output, future_pred_points=list(output['future_pred_points']),
                               future_pred_logits=list(output['future_pred_logits']))
                distances = []
                for h, (target, valid) in targets.items():
                    state = output['future_query_states'][h-1]
                    mixed = state['features'].clone()
                    mixed[:, valid] = (1-alpha)*mixed[:, valid] + alpha*target[:, valid].to(mixed)
                    positions, semantic = instance.decode(mixed, state['interaction_positions'])
                    changed['future_pred_logits'][h-1] = semantic.flatten(1,2).float()
                    if geometry == 'native_joint':
                        changed['future_pred_points'][h-1] = instance._world(positions).flatten(1,2).float()
                    if valid.any():
                        distances.append(float((1-F.cosine_similarity(mixed[:, valid].float(),target[:,valid].float(),dim=-1)).mean()))
                key = f'{geometry}_mix{alpha:g}'
                rows[key].append(prediction_rows(instance, changed, cfg))
                errors[key].append(float(np.mean(distances)) if distances else None)
        if (i+1)%16 == 0:
            print(f'[S1 R2] {i+1}/{count}', flush=True)
    save_rows(root/'s1_baseline.npz', reference_rows)
    base = summarized_rows(reference_rows,cfg)
    result = dict(status='complete', baseline=base, identity_parity=parity_checked, cases={}, matching=matches,
                  validation={k:v[:count] if isinstance(v,list) else v for k,v in manifest.items()},
                  intervention='pooled teacher feature mix, native semantic head; invalid queries unchanged; no recurrent feedback')
    for key, values in rows.items():
        destination=root/f's1_{key}.npz'; save_rows(destination,values)
        record=relative_to_baseline(summarized_rows(values,cfg),base)
        record.update(rows=str(destination),feature_cosine_error=float(np.mean([e for e in errors[key] if e is not None])))
        result['cases'][key]=record
    save_json(root/'s1_feature_intervention.json',result)
    del instance
    torch.cuda.empty_cache()
    return result


def gradient_stats(names, task_grads, teacher_grads, weight):
    sums = {}
    for name, task, teacher in zip(names,task_grads,teacher_grads):
        key=name.split('.')[0]
        item=sums.setdefault(key,dict(task2=0.,teacher2=0.,dot=0.,shared_parameters=0,task_only_parameters=0,teacher_only_parameters=0))
        if task is not None and teacher is not None:
            item['task2']+=float(task.float().square().sum()); item['teacher2']+=float(teacher.float().square().sum())
            item['dot']+=float((task.float()*teacher.float()).sum()); item['shared_parameters']+=task.numel()
        elif task is not None: item['task_only_parameters']+=task.numel()
        elif teacher is not None: item['teacher_only_parameters']+=teacher.numel()
    combined={key:sum(v[key] for v in sums.values()) for key in next(iter(sums.values()))}
    sums['shared_global']=combined
    for item in sums.values():
        tn,dn=item['task2']**.5,item['teacher2']**.5
        item['cosine']=item['dot']/max(tn*dn,1e-30) if tn and dn else None
        item['weighted_teacher_to_task_norm']=weight*dn/tn if tn else None
    return sums


def gradient_audit(cfg,root,count):
    seed_all(20260930)
    instance,strict=model_for(cfg)
    objective=OPENOCC_LOSS.build(strict.loss).cuda()
    images,manifest=selected_images(cfg,'train',count)
    named=list(instance.named_parameters()); names=[n for n,_ in named]; parameters=[p for _,p in named]
    samples=[]
    # Eval mode is intentional: no BN/dropout changes confound paired gradients.
    for i,data in enumerate(image_loader(images.dataset,0)):
        data=tree_to(data,'cuda'); imgs=data.pop('img')
        teacher=teacher_record(cfg,'train',manifest['indices'][i],manifest['scenes'][i])
        # Three independent forwards keep memory bounded on a shared 24GB GPU.
        # Eval BN/dropout plus deterministic full-set losses make these paired
        # evaluations of the same model/sample. Reentrant checkpointing requires
        # ordinary backward, not autograd.grad.
        def gradients(kind):
            instance.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): out=instance(imgs=imgs,metas=data)
            if kind=='align':value=feature_loss(out,matched_targets(instance,out,teacher))
            elif kind=='future':value=future_loss(objective,out)
            else:value,_=objective(out)
            score=float(value.detach());value.backward()
            tensors=tuple(None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters)
            del out,value
            instance.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            return tensors,score
        aux,align=gradients('align')
        future_grad,future=gradients('future')
        row=dict(index=manifest['indices'][i],scene=manifest['scenes'][i],future=future,align=align,
                 future_groups=gradient_stats(names,future_grad,aux,cfg.causal_feature_weight))
        del future_grad
        task_grad,task=gradients('task');row['task']=task
        row['total_groups']=gradient_stats(names,task_grad,aux,cfg.causal_feature_weight)
        samples.append(row)
        save_json(root/'s1_gradients.json',dict(status='running',samples=samples,mode='full student native loss; eval BN/dropout; BF16 graph'))
        print(f'[S1 R3] {i+1}/{count} cos={row["total_groups"]["shared_global"]["cosine"]:.4f}',flush=True)
        del task_grad,aux,data,imgs
        torch.cuda.empty_cache()
    save_json(root/'s1_gradients.json',dict(status='complete',samples=samples,mode='full student native loss; eval BN/dropout; BF16 graph',
                                         limitation='gradient observation only; AdamW finite-step intervention in S2 required'))
    del instance,objective
    torch.cuda.empty_cache()


def main(args):
    cfg=Config.fromfile(args.config); deterministic(); seed_all(20260930)
    torch.cuda.set_per_process_memory_fraction(.42)
    root=Path(args.output_dir or (cfg.causal_root+'_smoke2' if args.smoke else cfg.causal_root)); root.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        cfg.causal_train_samples=cfg.causal_val_samples=cfg.causal_audit_samples=cfg.causal_gradient_samples=2
    if 's0' in args.stages: audit_training(cfg,root,cfg.causal_audit_samples)
    if 'prepare' in args.stages: prepare_sources(cfg,root)
    if 'r2' in args.stages:
        manifest=json.loads((root/'manifest.json').read_text())['val']
        feature_intervention(cfg,root,manifest,cfg.causal_audit_samples)
    if 'r3' in args.stages: gradient_audit(cfg,root,cfg.causal_gradient_samples)
    print('[mechanisms complete]',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_causal.py')
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stages',nargs='+',choices=['s0','prepare','r2','r3'],default=['s0','prepare','r2','r3'])
    main(parser.parse_args())
