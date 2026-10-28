"""Resumable 15-epoch full-split S2 training, with complete per-epoch diagnostics."""

import argparse
import fcntl
import gc
import json
import math
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from model.segmentor.pmg_semantic_gaussian_route_world import FORECAST_MODULES
from tools.pmg_world.regions import HORIZONS, REGIONS, histogram, load_masks, region_result, token_at
from tools.pmg_world_full.train import accumulation_divisor, PauseRequested
from tools.pmg_world_native_adapt.semantic_route_full_report import diagnostic_report, refresh
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics


TRAINABLE=(*FORECAST_MODULES,'query_update_gate')
STATIC=(0,1,11,12,13,14,15,16)


def is_trainable(name):
    return name.split('.',1)[0] in TRAINABLE


def initialize(cfg):
    model=build_segmentor(cfg.model)
    source=torch.load(cfg.adaptation_checkpoint,map_location='cpu',mmap=True,weights_only=False)
    missing=model.load_state_dict(source['state_dict'],strict=False)
    expected={'reference_forecast_modules.'+name for name in model.reference_forecast_modules.state_dict()}
    expected|={'query_update_gate.linear.weight','query_update_gate.linear.bias'}
    assert set(missing.missing_keys)==expected and not missing.unexpected_keys
    for name in FORECAST_MODULES:
        model.reference_forecast_modules[name].load_state_dict({
            key[len(name)+1:]:value for key,value in source['state_dict'].items() if key.startswith(name+'.')},strict=True)
    for name,param in model.named_parameters():
        param.requires_grad_(is_trainable(name))
    model.cuda()
    model.pts_bbox_head.ind_stamps_all=source['query_stamps'].cuda()
    model.set_epoch(0)
    assert torch.equal(source['query_stamps'],model.pts_bbox_head.ind_stamps_all.cpu())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1715980
    return model,source


def frozen_audit(model,source):
    state=model.state_dict()
    changed=[name for name,value in source['state_dict'].items()
             if not is_trainable(name) and not torch.equal(value,state[name].detach().cpu())]
    changed_ref=[name for name,value in source['state_dict'].items()
                 if name.split('.',1)[0] in FORECAST_MODULES and
                 not torch.equal(value,state['reference_forecast_modules.'+name].detach().cpu())]
    query_exact=torch.equal(source['query_stamps'],model.pts_bbox_head.ind_stamps_all.cpu())
    result=dict(frozen_shared_changed_keys=changed,frozen_reference_changed_keys=changed_ref,
        frozen_state_exact=not changed and not changed_ref,query_assignment_exact=query_exact)
    assert result['frozen_state_exact'] and query_exact,result
    return result


def add_static(metric):
    for horizon in metric['horizons']:
        horizon['static_class_miou_fullspace']=float(np.asarray(horizon['class_iou'])[list(STATIC)].mean())
    return metric


def write_counts(path,metric,scenes,samples,regions=None,scene_regions=None):
    names=sorted(scenes)
    data=dict(confusion=metric.confusion,scenes=np.asarray(names),
        scene_confusion=np.stack([scenes[name] for name in names]),
        sample_counts=np.asarray([samples[name] for name in names]))
    if regions is not None:
        data.update(region_confusion=regions,scene_region_confusion=np.stack([scene_regions[name] for name in names]))
    np.savez_compressed(path,**data)


@torch.no_grad()
def evaluate(model,dataset,cfg,report,epoch,updates,should_stop,progress,limit=None):
    full,subset,baseline=HorizonMetrics(),HorizonMetrics(),HorizonMetrics()
    full_scenes,full_samples,sub_scenes,sub_samples,base_scenes={},{},{},{},{}
    region_counts=np.zeros((3,len(REGIONS),18,18),dtype=np.int64)
    sub_regions={}
    manifest=json.loads(Path(cfg.subset_diagnostics.selection_manifest).read_text())
    subset_tokens={item['token'] for item in manifest['samples']}
    region_root=Path(cfg.subset_diagnostics.reference_directory)/'regions'
    seen=set()
    rng=runtime.capture_rng()
    started=time.monotonic()
    model.eval()
    model.memory.clear();model.memory_queue.clear()
    model.reset_route_statistics(enabled=True)
    model.reset_gate_statistics(enabled=True)
    capture_baseline=not (report/'reference_full/epoch_00_metrics.json').exists()
    model.capture_reference_grids=capture_baseline
    try:
        for index,data in enumerate(runtime.loader(dataset,cfg.val_loader)):
            if should_stop():raise PauseRequested()
            images,metas=runtime.move_batch(data,'cuda')
            result=model(imgs=images,metas=metas)
            before=full.confusion.copy();full.add(result,metas);delta=full.confusion-before
            scene=str(dataset.keyframes[index][0])
            full_scenes.setdefault(scene,np.zeros_like(delta));full_scenes[scene]+=delta
            full_samples[scene]=full_samples.get(scene,0)+1
            if capture_baseline:
                original=dict(result,final_occ_grid=model.last_reference_grids[0],
                    future_occ_grids=model.last_reference_grids[1:],pred_traj=model.last_reference_pred_traj)
                before=baseline.confusion.copy();baseline.add(original,metas)
                base_scenes.setdefault(scene,np.zeros_like(delta));base_scenes[scene]+=baseline.confusion-before
                del original
            token=token_at(dataset,index,0)
            if token in subset_tokens:
                if token in seen:raise ValueError('Duplicate diagnostic token')
                seen.add(token)
                before=subset.confusion.copy();subset.add(result,metas)
                sub_scenes.setdefault(scene,np.zeros_like(delta));sub_scenes[scene]+=subset.confusion-before
                sub_samples[scene]=sub_samples.get(scene,0)+1
                sub_regions.setdefault(scene,np.zeros_like(region_counts))
                for slot,horizon in enumerate(HORIZONS):
                    pred=result['future_occ_grids'][horizon-1].cpu().numpy().reshape(-1)
                    target=metas['future_occ_labels'][:,horizon-1].cpu().numpy().reshape(-1)
                    masks=load_masks(region_root/(token+'.npz'),horizon)
                    for r,name in enumerate(REGIONS):
                        count=histogram(pred,target,masks[name])
                        region_counts[slot,r]+=count;sub_regions[scene][slot,r]+=count
            del result,images,metas,data
            if (index+1)%100==0:
                progress('evaluating',phase='evaluate',validation_processed=index+1,
                    diagnostic_processed=len(seen),evaluation_seconds=time.monotonic()-started)
            if limit is not None and index+1>=limit:break
    finally:
        model.capture_reference_grids=False
        model.collect_route_statistics=False;model.collect_gate_statistics=False
        model.memory.clear();model.memory_queue.clear()
        runtime.restore_rng(rng)
    assert full.samples==(min(limit,len(dataset)) if limit is not None else len(dataset))
    assert np.array_equal(sum(full_scenes.values()),full.confusion)
    metric=add_static(dict(full.results(),epoch=epoch,is_smoke=limit is not None,
        evaluation_seconds=time.monotonic()-started,planned_epochs=15,
        dataset_scope='all_eligible_official_validation_samples',training_updates=updates))
    runtime.atomic_json(report/f'epoch_{epoch:02d}_metrics.json',metric)
    write_counts(report/f'epoch_{epoch:02d}_counts.npz',full,full_scenes,full_samples)
    runtime.atomic_json(report/f'epoch_{epoch:02d}_route_statistics.json',model.route_statistics())
    runtime.atomic_json(report/f'epoch_{epoch:02d}_gate_statistics.json',model.gate_statistics())
    if capture_baseline:
        assert np.array_equal(full.confusion[0],baseline.confusion[0])
        folder=report/'reference_full';folder.mkdir(parents=True,exist_ok=True)
        ref=add_static(dict(baseline.results(),epoch=0,is_smoke=limit is not None,
            source_checkpoint=cfg.adaptation_checkpoint,dataset_scope=metric['dataset_scope'],
            note='Reference occupancy evaluated alongside candidate with shared perception'))
        runtime.atomic_json(folder/'epoch_00_metrics.json',ref)
        write_counts(folder/'epoch_00_counts.npz',baseline,base_scenes,full_samples)
    else:
        with np.load(report/'reference_full/epoch_00_counts.npz') as old:
            assert np.array_equal(old['confusion'][0],full.confusion[0])
            assert np.array_equal(old['scene_confusion'].sum(-1),np.stack([full_scenes[s] for s in sorted(full_scenes)]).sum(-1))
    if limit is None:
        assert seen==subset_tokens and subset.samples==1024 and len(sub_scenes)==150
        with np.load(cfg.subset_diagnostics.baseline_counts) as old:
            assert np.array_equal(old['scenes'],np.asarray(sorted(sub_scenes)))
            assert np.array_equal(old['sample_counts'],np.asarray([sub_samples[s] for s in sorted(sub_scenes)]))
            assert np.array_equal(old['scene_confusion'][:,0],np.stack([sub_scenes[s][0] for s in sorted(sub_scenes)]))
            assert np.array_equal(old['scene_region_confusion'].sum(-1),np.stack([sub_regions[s] for s in sorted(sub_regions)]).sum(-1))
        folder=report/'diagnostics'/f'epoch_{epoch:02d}';folder.mkdir(parents=True,exist_ok=True)
        sub=add_static(dict(subset.results(),epoch=epoch,partial=False,planned_epochs=15,
            future_123_mean_miou=float(np.mean([subset.results()['horizons'][h]['miou'] for h in HORIZONS])),
            evaluation_seconds=time.monotonic()-started,training_updates=updates,
            regions=[dict(seconds=h*.5,metrics={name:region_result(region_counts[slot,r])
                     for r,name in enumerate(REGIONS)}) for slot,h in enumerate(HORIZONS)],
            region_reference=cfg.subset_diagnostics.reference_directory,
            scope='Fixed 1024 sample monitor within full validation, not a full-split regional result'))
        runtime.atomic_json(folder/f'epoch_{epoch:02d}_metrics.json',sub)
        write_counts(folder/f'epoch_{epoch:02d}_counts.npz',subset,sub_scenes,sub_samples,region_counts,sub_regions)
        diagnostic_report(folder,epoch,updates)
    return metric


def engineering_check(model,objective,train,val,cfg,output):
    rng=runtime.capture_rng()
    expected=json.loads(Path('analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.json').read_text())
    index=next(i for i in range(len(val)) if token_at(val,i,0)==expected['token'])
    data=next(iter(runtime.loader(val,cfg.val_loader,[index],cfg.seed)))
    images,metas=runtime.move_batch(data,'cuda')
    model.eval();model.capture_reference_grids=True
    with torch.no_grad():
        result=model(imgs=images,metas=metas)
        assert all(torch.equal(a,b) for a,b in zip([result['final_occ_grid'],*result['future_occ_grids']],model.last_reference_grids))
        targets=[metas['occ_label'],*metas['future_occ_labels'].unbind(1)]
        counts=np.stack([histogram(p.cpu().numpy(),t.cpu().numpy()) for p,t in zip(model.last_reference_grids,targets)])
        with np.load('analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.npz') as old:
            assert np.array_equal(counts,old['global_confusion'][0])
    model.capture_reference_grids=False
    del result,images,metas,data,targets
    model.train();model.audit_route_fields=True;model.capture_candidate_gradients=True
    data=next(iter(runtime.loader(train,cfg.train_loader,[0],cfg.seed)))
    images,metas=runtime.move_batch(data,'cuda')
    result=model(imgs=images,metas=metas)
    assert len(result['forecast_points_list'])==6 and set(result['gaussian_logits_by_horizon'])==set(range(7))
    assert not any(model.last_reference_requires_grad)
    assert all(all(row.values()) for row in model.last_route_audit)
    loss,details=objective(dict(result,metas=metas,update_stamp_statistics=False))
    assert torch.isfinite(loss)
    loss.backward()
    gradient_norms={}
    for name in TRAINABLE:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(name+'.') and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),name
        gradient_norms[name]=math.sqrt(sum(float(g.double().square().sum()) for g in grads))
        assert gradient_norms[name]>0,name
    assert all(p.grad is None for name,p in model.named_parameters() if not is_trainable(name))
    field_gradients=[]
    for step,(pack,mask) in enumerate(zip(model.candidate_gaussian_gradient_packs,model._route_masks),1):
        record={}
        for field,value in zip(pack._fields,pack):
            if value is None:continue
            assert value.grad is not None and torch.isfinite(value.grad).all(),field
            unselected=float(value.grad[~mask].abs().max()) if (~mask).any() else 0.
            selected=float(value.grad[mask].norm()) if mask.any() else 0.
            assert unselected==0.,(field,unselected)
            record[field]=dict(selected_gradient_norm=selected,unselected_gradient_max=unselected)
        assert record['means']['selected_gradient_norm']>0 and record['semantics']['selected_gradient_norm']>0
        field_gradients.append(dict(seconds=step*.5,fields=record))
    audit=dict(status='passed',optimizer_updates=0,initial_seven_grids_match_native_exact=True,
        initial_counts_match_old_T_A0_exact=True,validation_token=expected['token'],
        all_seven_voxel_losses_active=True,static_all_fields_exact=True,
        candidate_modules_gradient_norms=gradient_norms,decoded_candidate_field_gradients=field_gradients,
        unselected_decoded_Gaussian_voxel_gradients_exact_zero=True,
        frozen_parameter_gradients_absent=True,reference_gaussian_graph_detached=True,
        total_loss=float(loss.detach()),query_assignment_fixed=True)
    runtime.atomic_json(output/'engineering_check.json',audit)
    print('S2_PREFLIGHT '+json.dumps({k:v for k,v in audit.items() if k!='decoded_candidate_field_gradients'}),flush=True)
    model.zero_grad(set_to_none=True)
    model.capture_candidate_gradients=False;model.candidate_gaussian_gradient_packs=[];model.audit_route_fields=False
    del result,loss,details,images,metas,data,pack,value
    gc.collect();torch.cuda.empty_cache();runtime.restore_rng(rng)


def run(args):
    cfg=Config.fromfile(args.config)
    assert '_base_' not in Path(args.config).read_text()
    assert cfg.max_epochs==cfg.schedule_epochs==15 and cfg.model.finetune_epoch==0
    assert list(cfg.model.gaussian_loss_horizons)==list(range(7))
    runtime.configure_runtime(cfg)
    output=Path(args.work_dir);output.mkdir(parents=True,exist_ok=True)
    report=output/'smoke_report' if args.smoke else Path(cfg.report_directory)
    report.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.smoke and (output/'latest.pth').exists():raise FileExistsError('Smoke directory contains training state')
    train=OPENOCC_DATASET.build(cfg.train_dataset_config);val=OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert (len(train),len(val))==(19730,4219)
    manifest=json.loads(Path(cfg.subset_diagnostics.selection_manifest).read_text())
    full_tokens={token_at(val,i,0) for i in range(len(val))}
    assert {r['token'] for r in manifest['samples']}<=full_tokens
    assert all((Path(cfg.subset_diagnostics.reference_directory)/'regions'/(r['token']+'.npz')).exists() for r in manifest['samples'])
    from tools.pmg_world_stage2.train import implementation_identity
    files=[args.config,__file__,'model/segmentor/pmg_query_update_gate_world.py',
        'model/segmentor/pmg_semantic_gaussian_route_world.py','model/segmentor/pmg_semantic_route_train_world.py',
        'tools/pmg_world_native_adapt/semantic_route_full_report.py',
        'tools/pmg_world_native_adapt/semantic_route_full_launch.py',
        'tools/pmg_world_native_adapt/gate_optimization_report.py','tools/pmg_world_native_adapt/gate_report.py',
        'tools/pmg_world_full/train.py','loss/pmg_repair_distance_weight.py']
    sources=implementation_identity();sources.update({str(p):runtime.digest(p) for p in files})
    identity=dict(config_sha256=runtime.digest(args.config),source_sha256=sources,
        source_checkpoint_sha256=runtime.digest(cfg.adaptation_checkpoint),manifests=runtime.audit_manifests(cfg),
        diagnostic_manifest_sha256=runtime.digest(cfg.subset_diagnostics.selection_manifest))
    identity_path=output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Run identity changed')
    runtime.atomic_json(identity_path,identity)
    cfg.dump(str(output/'resolved_config.py'))
    model,source=initialize(cfg)
    objective=OPENOCC_LOSS.build(cfg.loss).cuda()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg.base_lr,weight_decay=cfg.weight_decay)
    for group in optimizer.param_groups:group['base_lr']=cfg.base_lr
    accumulation=cfg.gradient_accumulation;updates_per_epoch=math.ceil(len(train)/accumulation)
    planned=updates_per_epoch*15
    assert cfg.train_loader.batch_size==1 and planned==73995
    protocol=dict(arm=cfg.pmg_experiment.arm,train_samples=len(train),validation_samples=len(val),
        planned_epochs=15,updates_per_epoch=updates_per_epoch,planned_updates=planned,
        initialization_checkpoint=cfg.adaptation_checkpoint,initialization_scope='T-A0 subset epoch10 world checkpoint',
        trainable_modules=list(TRAINABLE),trainable_parameters=sum(p.numel() for p in parameters),
        frozen_perception=True,frozen_reference=True,frozen_attribute_decoder=True,
        query_assignment_fixed=True,dropout_and_BN_disabled=True,gradient_accumulation=accumulation,
        microbatch_size=1,learning_rate=cfg.base_lr,warmup_updates=cfg.warmup_iters,min_lr_ratio=cfg.min_lr_ratio,
        route_class_ids=list(model.route_class_ids),route_source='Frozen reference predicted semantics, no GT routing',
        native_set_and_trajectory_loss='Candidate all children; preserves original definition',
        voxel_loss='Selected complete Gaussians from reference/candidate, seven horizons',
        future_focal_distance_weight_detached=True,distillation=False,first_five_epochs_are_formal=True,
        full_validation_every_epoch=True,fixed_subset_diagnostics=1024,wall_time_limit_hours=0,
        resume_note='Sample order, optimizer, RNG and allocation saved; outstanding DataLoader augmentation work is not bitwise replayed')
    runtime.atomic_json(output/'protocol.json',protocol)
    print('S2_FULL_PROTOCOL '+json.dumps(protocol),flush=True)
    latest=output/'latest.pth'
    epoch,next_sample,updates,best,phase=0,0,0,-1.,'train'
    resume_rng=None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',mmap=True,weights_only=False)
        assert saved['identity']==identity
        model.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        epoch,next_sample,updates,best,phase=(saved[k] for k in ('epoch','next_sample','updates','best','phase'))
        model.pts_bbox_head.ind_stamps_all=saved['query_stamps'].cuda();model.pts_bbox_head.reset_mask()
        resume_rng=saved['rng'];del saved
    else:
        engineering_check(model,objective,train,val,cfg,output)
        frozen_audit(model,source)
    requested=False
    def request_stop(signum,frame):
        nonlocal requested
        requested=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def progress(status,**values):
        value=dict(status=status,epoch=min(epoch+1,15),completed_epochs=(epoch+1 if status=='epoch_completed' else epoch),updates=updates,
            planned_epochs=15,planned_updates=planned,train_samples=len(train),validation_samples=len(val),**values)
        runtime.atomic_json(output/'progress.json',value);print('PROGRESS '+json.dumps(value),flush=True)
    def save(save_epoch,sample,save_phase):
        runtime.atomic_save(latest,dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=save_epoch,next_sample=sample,updates=updates,best=best,phase=save_phase,
            query_stamps=model.pts_bbox_head.ind_stamps_all,rng=runtime.capture_rng(),identity=identity))
    if not args.smoke and not latest.exists():save(0,0,'train')
    while epoch<15:
        model.train();model.set_epoch(epoch);train.set_epoch(epoch)
        if resume_rng is not None:runtime.restore_rng(resume_rng);resume_rng=None
        model.memory.clear();model.memory_queue.clear();optimizer.zero_grad(set_to_none=True)
        order=np.random.default_rng(cfg.seed+epoch).permutation(len(train)).tolist()
        start=time.monotonic();losses=[];torch.cuda.reset_peak_memory_stats()
        if phase=='train':
            for local_index,data in enumerate(runtime.loader(train,cfg.train_loader,order[next_sample:],cfg.seed+epoch)):
                sample=next_sample+local_index;processed=sample+1
                images,metas=runtime.move_batch(data,'cuda')
                result=model(imgs=images,metas=metas)
                if set(result['gaussian_logits_by_horizon'])!=set(range(7)):raise ValueError('Missing voxel horizon')
                loss,details=objective(dict(result,metas=metas,update_stamp_statistics=False,global_iter=epoch*len(train)+sample))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss/accumulation_divisor(sample,len(train),accumulation)).backward()
                losses.append(float(loss.detach()))
                boundary=processed%accumulation==0 or processed==len(train)
                norm=None
                if boundary:
                    norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg.grad_max_norm,error_if_nonfinite=True))
                    runtime.set_lr(optimizer,updates,planned,cfg.warmup_iters,cfg.min_lr_ratio)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                details={key:float(value) for key,value in details.items()} if processed%100==0 or args.smoke else {}
                del result,loss,images,metas,data
                if boundary and (updates%cfg.print_freq==0 or args.smoke):
                    seconds=(time.monotonic()-start)/(local_index+1)
                    progress('smoke_training' if args.smoke else 'training',phase='train',samples=processed,
                        mean_loss=float(np.mean(losses[-100:])),gradient_norm=norm,current_lr=optimizer.param_groups[0]['lr'],
                        seconds_per_microbatch=seconds,estimated_remaining_train_hours=(len(train)-processed)*seconds/3600,
                        peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,losses=details)
                if args.smoke and boundary:
                    audit=frozen_audit(model,source)
                    metric=evaluate(model,val,cfg,report,1,updates,lambda:False,progress,limit=1)
                    runtime.atomic_json(output/'smoke.json',dict(passed=True,config_sha256=identity['config_sha256'],
                        identity=identity,actual_optimizer_updates=updates,seven_voxel_horizons=list(range(7)),
                        finite_gradient_norm=norm,frozen_state_audit=audit,validation_samples=metric['num_samples']))
                    progress('smoke_completed',samples=processed,phase='train')
                    return
                if boundary and (requested or updates%cfg.checkpoint_interval_steps==0):
                    save(epoch,processed,'train')
                    if requested:
                        progress('paused_at_optimizer_boundary',samples=processed,phase='train');return
            save(epoch,len(train),'evaluate');phase='evaluate'
        audit=frozen_audit(model,source)
        runtime.atomic_json(report/f'epoch_{epoch+1:02d}_state_audit.json',audit)
        progress('evaluating',phase='evaluate',validation_processed=0)
        try:
            metric=evaluate(model,val,cfg,report,epoch+1,updates,lambda:requested,progress)
        except PauseRequested:
            progress('paused_before_validation_completed',phase='evaluate');return
        score=metric['selection_mean_miou']
        if score>best:
            best=score
            runtime.atomic_save(output/'best.pth',dict(state_dict=model.state_dict(),epoch_index=epoch,
                updates=updates,query_stamps=model.pts_bbox_head.ind_stamps_all,metrics=metric,identity=identity))
        save(epoch+1,0,'train');refresh(report)
        progress('epoch_completed',selection_mean_miou=score,best=best,phase='train',samples=len(train))
        epoch,next_sample,phase=epoch+1,0,'train'
        if requested:
            progress('paused_after_epoch',phase='train');return
    assert updates==planned
    progress('completed',phase='complete',best=best)
    runtime.atomic_json(output/'completed.json',dict(status='completed',epochs=15,updates=updates,best=best))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--work-dir',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    output=Path(args.work_dir);output.mkdir(parents=True,exist_ok=True)
    try:run(args)
    except Exception as error:
        runtime.atomic_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),time_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())))
        progress=output/'progress.json'
        previous=json.loads(progress.read_text()) if progress.exists() else {}
        runtime.atomic_json(progress,dict(previous,status='failed',error_type=type(error).__name__,error=str(error)))
        traceback.print_exc();raise


if __name__=='__main__':main()
