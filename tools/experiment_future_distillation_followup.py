#!/usr/bin/env python3
"""Resumable approved S3 development run, followed by explicit S5/U2 gates."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest(value):
    # Persisted JSON converts tuples to lists and integer keys to strings.
    # Compare the same wire representation, while preserving all values.
    return json.loads(json.dumps(value))


def launch(args):
    from mmengine import Config
    cfg = Config.fromfile(args.config)
    root = Path(args.output_dir or cfg.confirmation_root).resolve()
    if args.smoke:
        root = Path(str(root)+'_smoke2')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'launch.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous = read(root/'job.json')
    if previous:
        pid = previous.get('pid')
        # Avoid duplicate launch, including PID reuse; do not kill anything.
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.exists() and str(Path(__file__).resolve()).encode() in cmdline.read_bytes():
            raise RuntimeError(f'Follow-up worker already alive: {pid}')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', args.config,
               '--output-dir', str(root), '--worker']
    if args.smoke:
        command.append('--smoke')
    environment = os.environ.copy()
    environment.setdefault('OMP_NUM_THREADS', '4')
    environment.setdefault('MKL_NUM_THREADS', '4')
    environment.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-cache')
    with (root/'run.log').open('a') as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(root/'job.json', dict(pid=process.pid, command=command, launched_at=time.time(),
                                   status='launched_not_completed', log=str(root/'run.log')))
    print(json.dumps(dict(pid=process.pid, root=str(root), log=str(root/'run.log')), indent=2))


def stage_gates(root, state):
    from tools.audit_future_distillation_confirmation import combination_gate
    entry = read(REPO/'out/future_distillation_causal/20261007_confirmation_entry/entry_audit.json')
    # A successful S3 development replication is still NOT independent evidence
    # for two distinct useful repairs. It must never auto-populate this ledger.
    state['S5'] = dict(status='not_entered', gate=combination_gate([]),
                       reason='No two independently confirmed useful interventions; BN repair is the shared protocol, not a teacher benefit.')
    state['U2'] = dict(status='blocked_no_verified_branch_source',
                       verified_branch_datasets=[],
                       environment_audit=entry.get('U2') if entry else None,
                       reason='User approval cannot create identical-history/action multi-future data; waiting for a concrete source.')
    save_json(root/'followup_state.json', state)


def worker(args):
    import numpy as np
    import torch
    from mmengine import Config
    from loss import OPENOCC_LOSS
    from tools.future_distillation_causal_common import deterministic, selected_images
    from tools.experiment_future_distillation_training_protocol import build
    from tools.future_distillation_training_protocol_common import set_training_mode, schedule_factor, bn_snapshot, bn_unchanged
    from tools.train_future_predictable_full import image_loader, optimizer_for, atomic_torch_save, checkpoint, evaluate, alignment_loss
    from tools.diagnose_teacher_alignment import seed_all, tree_to
    from tools.evaluate_future_predictable_native import relative_to_baseline
    from tools.future_distillation_confirmation_results import load_counts, metrics_from_counts, write_report, analyze

    cfg = Config.fromfile(args.config)
    if not cfg.confirmation_development_only:
        raise ValueError('This approved runner must be labelled development-only')
    root = Path(args.output_dir or cfg.confirmation_root).resolve()
    if args.smoke and not args.worker:
        root = Path(str(root)+'_smoke2')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'train.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    # All follow-up workers serialize GPU usage. Existing trainers are not
    # modified; current GPU occupancy is checked before the initial launch.
    global_lock = (REPO/'out/future_distillation_causal/followup_gpu.lock').open('a')
    fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    deterministic()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: rerun with host GPU permission')
    torch.cuda.set_per_process_memory_fraction(.75)
    strict = Config.fromfile(cfg.student_model_config)
    if args.smoke:
        cfg.confirmation_train_samples=cfg.confirmation_val_samples=2
        cfg.confirmation_epochs=1;cfg.confirmation_seeds=[20260930]
        cfg.confirmation_evaluation_epochs=[1];cfg.confirmation_workers=0
        cfg.confirmation_alignment_samples=2
    cfg.full_tune_workers = cfg.confirmation_workers
    train, train_manifest = selected_images(cfg, 'train', cfg.confirmation_train_samples)
    val, val_manifest = selected_images(cfg, 'val', cfg.confirmation_val_samples)
    reference = Path(cfg.causal_reference_root)
    source_manifest = read(reference/'manifest.json')
    if set(train_manifest['scenes']) & set(val_manifest['scenes']):
        raise ValueError('Training and validation scene overlap')
    if sha256(cfg.student_checkpoint) != cfg.student_checkpoint_sha256:
        raise ValueError('Actual pretrained checkpoint SHA differs')
    for split, selected in [('train', train_manifest), ('val', val_manifest)]:
        for field in ('indices', 'scenes'):
            if selected[field] != source_manifest[split][field][:len(selected[field])]:
                raise ValueError(f'{split}/{field}: differs from approved source sample manifest')
    updates = len(train_manifest['indices'])*cfg.confirmation_epochs
    manifest = dict(schema=1, smoke=args.smoke, development_only=True,
                    train=train_manifest, val=val_manifest,
                    seeds=list(cfg.confirmation_seeds), supervisions=list(cfg.confirmation_supervisions),
                    epochs=cfg.confirmation_epochs, updates=updates,
                    evaluation_epochs=list(cfg.confirmation_evaluation_epochs),
                    checkpoint_sha256=cfg.student_checkpoint_sha256,
                    source_manifest_sha256=sha256(reference/'manifest.json'),
                    config=cfg.to_dict(), config_path=args.config,
                    fixed_bn=True, all_student_parameters_allowed=True,
                    source_hashes={str(p): sha256(p) for p in [
                        Path(__file__), REPO/'tools/future_distillation_confirmation_results.py',
                        REPO/'tools/future_distillation_training_protocol_common.py',
                        REPO/'model/segmentor/sparseworld_predictability_finetune_segmentor.py',
                        REPO/'tools/train_future_predictable_full.py']})
    manifest = canonical_manifest(manifest)
    prior = read(root/'manifest.json')
    if prior is not None and prior != manifest:
        raise ValueError('Immutable run manifest changed; refuse incompatible resume')
    save_json(root/'manifest.json', manifest)
    report = read(root/'results.json') or dict(status='running', arms={}, phase='preflight')
    if report['status'] == 'complete_development_only':
        write_report(root);print('[followup already complete]', root, flush=True);return
    state = read(root/'followup_state.json') or dict(S3=dict(status='running_development_only'), smoke=args.smoke)
    stage_gates(root, state)

    def commit():
        report['last_update_unix'] = time.time()
        save_json(root/'results.json', report)
        write_report(root)

    try:
        report['status']='running';report.pop('error', None);report['phase']='teacher_cache_audit';commit()
        # Audit every selected cache identity and tensor shape, not just existence.
        cache_audit = {}
        for split, selected in [('train', train_manifest), ('val', val_manifest)]:
            for index, scene in zip(selected['indices'], selected['scenes']):
                item = torch.load(reference/'teacher_features'/split/f'{index:06d}.pth',
                                  map_location='cpu', weights_only=False)
                if (item['index'],item['scene'],item['checkpoint_sha256']) != (index,scene,cfg.student_checkpoint_sha256):
                    raise ValueError(f'Teacher cache identity differs: {split}/{index}')
                for horizon in (2,4,6):
                    target = item['states'][horizon]
                    if target['features'].shape != (1,720,256) or target['centers'].shape != (1,720,3):
                        raise ValueError(f'Teacher cache shape differs: {split}/{index}/{horizon}')
                    if not torch.isfinite(target['features']).all() or not torch.isfinite(target['centers']).all():
                        raise ValueError('Nonfinite teacher cache')
            cache_audit[split]=len(selected['indices'])
        report['teacher_cache_audit']=cache_audit;commit()
        if 'baseline' not in report:
            report['phase']='native_baseline_validation';commit()
            seed_all(cfg.confirmation_seeds[0]);instance=build(cfg,strict)
            from tools.experiment_future_response_distillation import build_student
            original=build_student(cfg,strict)
            data=tree_to(next(iter(image_loader(val.dataset,0))), 'cuda');imgs=data.pop('img')
            with torch.no_grad():
                expected=original(imgs=imgs,metas=data);actual=instance(imgs=imgs,metas=data)
            for step in range(6):
                for field in ('future_pred_points','future_pred_logits'):
                    torch.testing.assert_close(actual[field][step],expected[field][step],atol=1e-5,rtol=0)
                if actual['future_pred_points'][step].shape[1] != (720+sum(instance.future_queries[:step+1]))*48:
                    raise ValueError('Future-born query count changed')
            del original,data,imgs,expected,actual;torch.cuda.empty_cache()
            result=evaluate(instance,val.dataset,cfg,root/'baseline.npz')
            observed=load_counts(root/'baseline.npz',len(val_manifest['indices']))
            with np.load(reference/'baseline.npz') as saved:
                for field, values in observed.items():
                    if not np.array_equal(values, saved[field][:len(values)]):
                        raise ValueError(f'Native baseline differs from source: {field}')
            report['baseline']=result
            report['initialization_parity']=dict(all_six_step_points_logits=True,future_born=True,
                                               native_baseline_counts=True)
            # Reuse teacher raw counts ONLY after prefix identity and actual
            # pretrained identity checks; record provenance and no new perception.
            with np.load(reference/'teacher_native.npz') as saved:
                teacher_counts={k:saved[k][:len(val_manifest['indices'])].copy()
                                for k in ('hist','intersection','union')}
            np.savez_compressed(root/'teacher_native.npz', **teacher_counts)
            report['teacher']=metrics_from_counts(load_counts(root/'teacher_native.npz',len(val_manifest['indices'])))
            report['teacher']['provenance']=dict(source=str(reference/'teacher_native.npz'),
                                                sha256=sha256(reference/'teacher_native.npz'),
                                                reuse='Previously evaluated native future perception, verified identities')
            commit();del instance;torch.cuda.empty_cache()
        for seed in cfg.confirmation_seeds:
            for supervision in cfg.confirmation_supervisions:
                label=f'{supervision}_seed{seed}';destination=root/label;destination.mkdir(exist_ok=True)
                row=report['arms'].setdefault(label,dict(status='pending',epochs={},progress=0,
                                                         seed=seed,supervision=supervision))
                if row['status']=='complete':continue
                seed_all(seed);instance=build(cfg,strict)
                optimizer=optimizer_for(instance,cfg);objective=OPENOCC_LOSS.build(strict.loss).cuda()
                generator=torch.Generator(device='cuda').manual_seed(seed+917)
                bn=bn_snapshot(instance);latest=destination/'latest.pth';start=0
                if latest.exists():
                    saved=torch.load(latest,map_location='cpu',weights_only=False)
                    if saved['manifest']!=manifest:raise ValueError('Checkpoint manifest differs')
                    instance.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
                    start=saved['steps'];random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
                    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng']);generator.set_state(saved['shuffle_rng'])
                    del saved
                    # Recover interrupted epoch validation without changing the
                    # RNG that the uninterrupted continuation would have used.
                    if start%len(train.dataset)==0:
                        number=start//len(train.dataset)
                        if number in cfg.confirmation_evaluation_epochs and str(number) not in row['epochs']:
                            row['epochs'][str(number)]=relative_to_baseline(evaluate(instance,val.dataset,cfg,destination/f'epoch_{number}.npz'),report['baseline'])
                row['status']='running';report['phase']=label;commit()
                torch.cuda.reset_peak_memory_stats();began=time.monotonic();segment_start=start
                count=len(train.dataset)
                for epoch in range(start//count,cfg.confirmation_epochs):
                    order=np.random.default_rng(seed+epoch*1009).permutation(count).tolist()
                    offset=start%count if epoch==start//count else 0
                    for position,data in zip(order[offset:],image_loader(train.dataset,cfg.confirmation_workers,order[offset:])):
                        set_training_mode(instance,True)
                        for group in optimizer.param_groups:
                            group['lr']=group['initial_lr']*schedule_factor(start,cfg.full_tune_warmup_steps,updates)
                        data=tree_to(data,'cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):output=instance(imgs=data.pop('img'),metas=data)
                        task,details=objective(output);align=task.new_zeros(())
                        if supervision!='task':
                            index=train_manifest['indices'][position];scene=train_manifest['scenes'][position]
                            teacher=torch.load(reference/'teacher_features/train'/f'{index:06d}.pth',map_location='cpu',weights_only=False)
                            if (teacher['index'],teacher['scene'],teacher['checkpoint_sha256'])!=(index,scene,cfg.student_checkpoint_sha256):
                                raise ValueError('Teacher cache changed during training')
                            align=alignment_loss(instance,output,teacher,'teacher_shuffled' if supervision=='shuffled' else 'teacher_all',generator)
                        value=task+cfg.full_tune_feature_weight*align
                        if not torch.isfinite(value):raise FloatingPointError(label)
                        optimizer.zero_grad(set_to_none=True);value.backward()
                        if start==0:
                            row['gradient_audit']=dict(trainable_parameters=sum(p.numel() for p in instance.parameters() if p.requires_grad),
                                                      frozen_names=[n for n,p in instance.named_parameters() if not p.requires_grad],
                                                      no_gradient_names=[n for n,p in instance.named_parameters() if p.grad is None])
                            if row['gradient_audit']['frozen_names']:raise ValueError('Student unexpectedly frozen')
                        norm=torch.nn.utils.clip_grad_norm_(instance.parameters(),5.,error_if_nonfinite=True)
                        optimizer.step();start+=1
                        row['progress']=start;row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                        if start%cfg.confirmation_log_every==0 or start==updates:
                            record=dict(step=start,epoch=epoch+1,task=float(task),align=float(align),unclipped_grad_norm=float(norm),
                                        components={k:float(v) for k,v in details.items()},
                                        elapsed_seconds=time.monotonic()-began,
                                        segment_steps=start-segment_start)
                            row.setdefault('loss_log',[]).append(record)
                            print(f'[S3 development {label}] {start}/{updates} task={float(task):.4f} align={float(align):.4f} '
                                  f'peak={row["peak_allocated_gib"]:.2f}GiB elapsed={time.monotonic()-began:.0f}s',flush=True)
                            commit()
                        if start%cfg.confirmation_checkpoint_every==0 or start%count==0:
                            saved=checkpoint(instance,optimizer,start//count,start%count,start,generator,manifest)
                            atomic_torch_save(saved,latest)
                            if start%count==0:
                                atomic_torch_save(saved,destination/f'epoch_{start//count}.pth')
                            del saved
                        del output,data,value,task,align
                    number=epoch+1
                    if number in cfg.confirmation_evaluation_epochs:
                        row['epochs'][str(number)]=relative_to_baseline(evaluate(instance,val.dataset,cfg,destination/f'epoch_{number}.npz'),report['baseline'])
                        print('[S3 native epoch]',label,number,[row['epochs'][str(number)]['metrics'][h]['mIoU'] for h in ('1s','2s','3s')],flush=True)
                        commit()
                row['bn_unchanged']=bn_unchanged(instance,bn)
                if not row['bn_unchanged']:raise ValueError('Fixed BN buffers changed')
                row['status']='complete';commit()
                analyze(root,cfg.confirmation_bootstrap_repetitions);write_report(root)
                del instance,optimizer,objective;torch.cuda.empty_cache()
        report['phase']='fixed_feature_audit';commit()
        alignment_audit(cfg,strict,root,manifest,val.dataset)
        report['phase']='native_result_verification';commit()
        verify(root)
        report['status']='complete_development_only';report['phase']='handoff';commit()
        state['S3']=dict(status='complete_development_only',independent_confirmation=False,
                         result=str(root/'results.json'),paired_analysis=str(root/'paired_analysis.json'))
        stage_gates(root,state)
        print('[followup development completed; S5/U2 gates remain explicit]',root,flush=True)
    except BaseException as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}'
        save_json(root/'traceback.json',dict(error=report['error'],traceback=traceback.format_exc()))
        commit();state['S3']=dict(status='failed',error=report['error']);stage_gates(root,state)
        raise


def alignment_audit(cfg,strict,root,manifest,dataset):
    import torch
    from torch.utils.data import Subset
    from tools.experiment_future_distillation_training_protocol import build
    from tools.future_distillation_causal_common import matched_targets, feature_loss, teacher_record
    from tools.diagnose_teacher_alignment import tree_to
    from tools.train_future_predictable_full import image_loader, atomic_torch_save
    result=read(root/'alignment_audit.json') or dict(status='running',cases={},
        protocol='Fixed targets and masks at baseline query slots, plus moving matches. Trained geometry may move; not identical physical sampling.')
    count=min(cfg.confirmation_alignment_samples,len(dataset))
    audit_dataset=Subset(dataset,list(range(count)))
    for label in ['baseline']+[f'{s}_seed{seed}' for seed in manifest['seeds'] for s in manifest['supervisions']]:
        if label in result['cases']:continue
        instance=build(cfg,strict)
        if label!='baseline':
            saved=torch.load(root/label/'latest.pth',map_location='cpu',weights_only=False,mmap=True)
            instance.load_state_dict(saved['state_dict']);del saved
        instance.eval().requires_grad_(False);rows=[]
        with torch.no_grad():
            for position,data in enumerate(image_loader(audit_dataset,cfg.confirmation_workers)):
                data=tree_to(data,'cuda');output=instance(imgs=data.pop('img'),metas=data)
                index=manifest['val']['indices'][position];scene=manifest['val']['scenes'][position]
                teacher=teacher_record(cfg,'val',index,scene);moving=matched_targets(instance,output,teacher)
                path=root/'fixed_alignment_targets'/f'{index:06d}.pth'
                if label=='baseline':
                    path.parent.mkdir(parents=True,exist_ok=True)
                    atomic_torch_save(dict(index=index,scene=scene,targets={h:(f.cpu(),v.cpu()) for h,(f,v) in moving.items()}),path)
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if (saved['index'],saved['scene'])!=(index,scene):raise ValueError('Fixed target identity differs')
                fixed={h:(f.cuda(),v.cuda()) for h,(f,v) in saved['targets'].items()}
                distances={}
                for h,(_,valid) in moving.items():
                    centers=instance._world(output['future_query_states'][h-1]['positions'].float()).mean(2)[0]
                    teacher_centers=teacher['states'][h]['centers'][0].cuda().float()
                    nearest=torch.cdist(centers,teacher_centers).min(-1).values
                    distances[str(h)]=dict(queries=len(valid),valid=int(valid.sum()),
                                           nearest_mean_m=float(nearest.mean()),
                                           nearest_median_m=float(nearest.median()),
                                           nearest_p90_m=float(torch.quantile(nearest,.9)))
                rows.append(dict(index=index,scene=scene,fixed=float(feature_loss(output,fixed)),
                                 moving=float(feature_loss(output,moving)),
                                 valid_fixed=sum(int(v.sum()) for _,v in fixed.values()),
                                 valid_moving=sum(int(v.sum()) for _,v in moving.values()),
                                 nearest_teacher_distance=distances))
        result['cases'][label]=dict(fixed_feature_loss=sum(r['fixed'] for r in rows)/count,
                                   moving_feature_loss=sum(r['moving'] for r in rows)/count,rows=rows)
        save_json(root/'alignment_audit.json',result);print('[S3 feature audit]',label,flush=True)
        del instance;torch.cuda.empty_cache()
    result['status']='complete';save_json(root/'alignment_audit.json',result)


def verify(root):
    import numpy as np
    from tools.future_distillation_confirmation_results import load_counts, metrics_from_counts
    report,manifest=read(root/'results.json'),read(root/'manifest.json')
    count=len(manifest['val']['indices']);checked=[]
    candidates=[('baseline',report['baseline'],root/'baseline.npz'),('teacher',report['teacher'],root/'teacher_native.npz')]
    for label,row in report['arms'].items():
        if row['status']!='complete' or row['progress']!=manifest['updates'] or not row['bn_unchanged']:
            raise ValueError(f'Incomplete or invalid arm: {label}')
        if row['gradient_audit']['frozen_names']:raise ValueError('Unexpected frozen student parameters')
        candidates.extend((f'{label}/{epoch}',value,Path(value['rows'])) for epoch,value in row['epochs'].items())
    for label,value,path in candidates:
        actual=metrics_from_counts(load_counts(path,count))
        for section in ('metrics','groups'):
            def compare(a,b):
                if isinstance(a,dict):
                    for key in a:compare(a[key],b[key])
                elif not np.isclose(a,b,atol=1e-8,rtol=0,equal_nan=True):raise ValueError(f'{label}: reported native score differs')
            compare(actual[section],value[section])
        checked.append(label)
    save_json(root/'verification.json',dict(status='passed',checked_native_results=len(checked),cases=checked,
                                           independent_confirmation=False,scope='Raw native counts, budget, declared all-parameter and fixed-BN checks'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_development_confirmation.py')
    parser.add_argument('--output-dir');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--launch',action='store_true');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    launch(args) if args.launch else worker(args)
