#!/usr/bin/env python3
"""Run fixed-perception and full-native Student occupancy KD factorial."""
import argparse,copy,fcntl,json,os,random,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader,Subset,Dataset
from dataset.utils import custom_collate_fn_temporal
from dataset.sparseworld_trajectory import _pose
from loss import OPENOCC_LOSS
from tools.diagnose_teacher_alignment import loader_for,tree_to,seed_all
from tools.experiment_future_response_distillation import paths_for
from tools.run_sparseworld_region_ablations import perceive,render
from tools.export_sparseworld_region_pairs import metadata
from tools.train_future_predictable_full import historical_config
from tools.future_distillation_training_protocol_common import schedule_factor
from tools.sparseworld_region_utils import ROOT,sha256,write_json
from tools.sparseworld_fast_motion_common import *


def atomic(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def image_loader(dataset,workers,order=None):
    if order is not None:dataset=Subset(dataset,order)
    opts=dict(batch_size=1,num_workers=workers,pin_memory=True,collate_fn=custom_collate_fn_temporal,
              generator=torch.Generator().manual_seed(20261008))
    if workers:opts.update(prefetch_factor=1)
    return DataLoader(dataset,**opts)

def loaders(cfg,split):
    count=cfg.fast_samples_train if split=='train' else cfg.fast_samples_val
    return loader_for(cfg,split,count,cfg.fast_workers,cfg.identity_sample_seed)

def cache_path(root,split,index):return root/'targets'/split/f'{index:06d}.npz'

def reuse_targets(cfg,manifest,root):
    """Reuse audited frozen-teacher labels without writing to the old run."""
    source=Path(cfg.fast_target_cache).resolve()
    previous=json.loads((source/'protocol.json').read_text())
    assert previous['checkpoint_sha256']==cfg.student_checkpoint_sha256
    for split in ('train','val'):
        expected=manifest[split]
        # A smoke run is a prefix of the same frozen target cache.
        assert previous[split]['indices'][:len(expected['indices'])]==expected['indices']
        assert previous[split]['scenes'][:len(expected['scenes'])]==expected['scenes']
        for ix,scene in zip(expected['indices'],expected['scenes']):
            with np.load(cache_path(source,split,ix)) as data:
                assert int(data['index'])==ix and str(data['scene'])==scene
    link=root/'targets'
    if link.exists():assert link.resolve()==source/'targets'
    else:link.symlink_to(source/'targets',target_is_directory=True)
    records=json.loads((source/'samples.json').read_text())
    write_json(root/'samples.json',{split:records[split][:len(manifest[split]['indices'])] for split in ('train','val')})

def prepare(cfg,strict,manifest,root):
    reference=build(cfg,strict).eval().requires_grad_(False)
    objective=OPENOCC_LOSS.build(strict.loss).cuda()
    motion=MotionIndex();sample_records={}
    for split in ('val','train'):
        images,selected=loaders(cfg,split);assert selected==manifest[split]
        ds=images.dataset.dataset
        recs=[metadata(ds,i,ix,scene) for i,(ix,scene) in enumerate(zip(selected['indices'],selected['scenes']))]
        sample_records[split]=recs
        sources=paths_for(Path(manifest['source_cache']),split,selected)
        missing=[i for i,ix in enumerate(selected['indices']) if not cache_path(root,split,ix).exists()]
        started=time.monotonic()
        for i,data in zip(missing,image_loader(images.dataset,cfg.fast_workers,missing)):
            rec=recs[i];source=torch.load(sources[i],map_location='cpu')
            assert (source['index'],source['scene'])==(rec['index'],rec['scene'])
            labels=source['metas']['future_occ_labels'][0].numpy()
            bins=np.stack([motion.bins(labels[h-1],rec['targets'][j]['token'],np.asarray(rec['targets'][j]['pose'],np.float32)) for j,h in enumerate(HORIZONS)])
            data=tree_to(data,'cuda');imgs=data.pop('img');source_ego=source['ego'].cuda()
            with torch.no_grad():
                if split=='val':
                    with np.load(ROOT/'pairs'/(rec['token']+'.npz')) as pair:
                        teacher=pair['P'].copy();gt=pair['GT'].copy();native=pair['F'].copy()
                        assert np.array_equal(gt,labels[[1,3,5]])
                    with np.load(ROOT/'attributes'/(rec['token']+'.npz')) as attrs:
                        assert np.array_equal(bins,attrs['speed']),'Speed definition changed'
                    # First and last sample replay authentic future RGB and native P.
                    if i in (0,len(recs)-1):
                        for j,h in enumerate(HORIZONS):
                            selected_img,me=window_for(data,imgs,h)
                            p,l,_,_=perceive(reference,selected_img,me,source_ego)
                            assert np.array_equal(render(reference,p,l),teacher[j]),'Teacher occupancy replay changed'
                    constant=0.
                else:
                    teacher=[]
                    for h in HORIZONS:
                        selected_img,me=window_for(data,imgs,h)
                        p,l,_,_=perceive(reference,selected_img,me,source_ego)
                        teacher.append(render(reference,p,l))
                    teacher=np.stack(teacher)
                    histmeta=dict(data,projection_mat=data['projection_mat'][:,:30],image_wh=data['image_wh'][:,:30])
                    state,ego=encode(reference,imgs[:,:30],histmeta)
                    if i in (0,len(recs)-1):
                        torch.testing.assert_close(state['representation'][-1]['query_features'].cpu(),source['encoded']['query_features'],atol=1e-5,rtol=0)
                        torch.testing.assert_close(ego.cpu(),source['ego'],atol=1e-5,rtol=0)
                    out=pack_output(reference,state,ego,histmeta)
                    current,_=objective.current_loss(out)
                    temporal,_=objective._temporal_loss(out,list(zip(out['temporal_pred_points'],out['temporal_pred_logits'])))
                    constant=float(current+temporal)
            dest=cache_path(root,split,rec['index']);dest.parent.mkdir(parents=True,exist_ok=True)
            values=dict(teacher=teacher,speed=bins,index=rec['index'],scene=rec['scene'],token=rec['token'],constant_task=constant)
            values.update({f'fast_{j}':np.flatnonzero(bins[j].reshape(-1)==2).astype(np.int32) for j in range(3)})
            if split=='val':values.update(native=native,GT=gt)
            tmp=dest.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**values)
            tmp.replace(dest)
            if (i+1)%32==0 or i==missing[-1]:print(f'[prepare {split}] {i+1}/{len(recs)} {time.monotonic()-started:.1f}s',flush=True)
        del images
    write_json(root/'samples.json',sample_records)
    del reference,objective,motion;torch.cuda.empty_cache()


def window_for(data,imgs,h):
    from tools.experiment_future2current import _future_window
    return _future_window(data,imgs,h)

def attach_raw_target(root,split,index,scene,target):
    with np.load(root/'teacher_logits'/split/f'{index:06d}.npz') as data:
        assert int(data['index'])==index and str(data['scene'])==scene
        for j in range(3):
            assert np.array_equal(data[f'fast_{j}'],target[f'fast_{j}'])
            target[f'teacher_raw_{j}']=data[f'teacher_raw_{j}']
            target[f'teacher_raw_valid_{j}']=data[f'teacher_raw_valid_{j}']
    return target

class FrozenData(Dataset):
    def __init__(self,sources,targets,raw_root=None):self.sources=sources;self.targets=targets;self.raw_root=raw_root
    def __len__(self):return len(self.sources)
    def __getitem__(self,i):
        source=torch.load(self.sources[i],map_location='cpu')
        with np.load(self.targets[i]) as d:target={k:d[k] for k in ('index','scene','token','constant_task','teacher','fast_0','fast_1','fast_2')}
        assert int(target['index'])==source['index'] and str(target['scene'])==source['scene']
        if self.raw_root is not None:attach_raw_target(self.raw_root,'train',source['index'],source['scene'],target)
        return source,target

def evaluate(instance,cfg,manifest,root,destination,full):
    instance.eval();rows=[];counts=[];sources=paths_for(Path(manifest['source_cache']),'val',manifest['val'])
    if full:
        images,selected=loaders(historical_config(cfg),'val');assert selected==manifest['val']
        stream=image_loader(images.dataset,cfg.fast_workers)
    else:stream=range(len(sources))
    with torch.no_grad():
        for i,data in enumerate(stream):
            with np.load(cache_path(root,'val',manifest['val']['indices'][i])) as d:target=dict(d)
            if full:
                data=tree_to(data,'cuda');state,ego=encode(instance,data.pop('img'),data)
                out=pack_output(instance,state,ego,data,position_gradient=cfg.get('fast_position_gradient',True));p,l=out['future_pred_points'],out['future_pred_logits']
            else:
                source=torch.load(sources[i],map_location='cpu')
                out=forecast_with_position_gradient(instance,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda(),position_gradient=cfg.get('fast_position_gradient',True));p,l=out['points'],out['logits']
            sample=[];regional=[]
            for j,h in enumerate(HORIZONS):
                pred=render(instance,p[h-1].float(),l[h-1].float())
                hist,c=region_counts(target['GT'][j],pred,target['speed'][j]);sample.append(hist);regional.append(c)
            rows.append(sample);counts.append(regional)
            if (i+1)%128==0:print(f'[eval {destination.stem}] {i+1}/{len(sources)}',flush=True)
    np.savez_compressed(destination,hist=np.asarray(rows),counts=np.asarray(counts),scenes=np.asarray(manifest['val']['scenes']),indices=np.asarray(manifest['val']['indices']))


def train_arm(cfg,strict,manifest,root,arm):
    full=arm.startswith('full');kd=arm.endswith('kd');seed_all(cfg.fast_seed)
    instance=build(cfg,strict,full=full);objective=OPENOCC_LOSS.build(strict.loss).cuda()
    groups={}
    for name,p in instance.named_parameters():
        if p.requires_grad:groups.setdefault(cfg.fast_backbone_lr if name.startswith('img_backbone.') else cfg.fast_lr,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=lr,initial_lr=lr) for lr,ps in groups.items()],weight_decay=.01)
    directory=root/arm;directory.mkdir(exist_ok=True);logpath=directory/'train.jsonl';latest=directory/'latest.pth'
    step=0;logs=[];trainable=[n for n,p in instance.named_parameters() if p.requires_grad]
    frozen_original={n:v.cpu().clone() for n,v in instance.state_dict().items() if n.split('.')[0] not in FORECAST} if not full else None
    geometry_original={n:v.cpu().clone() for n,v in instance.state_dict().items() if n.split('.')[0] in ('reg','velocity')} if not cfg.get('fast_position_gradient',True) else None
    bn_original={n:v.cpu().clone() for n,v in instance.state_dict().items() if 'running_mean' in n or 'running_var' in n or n.endswith('num_batches_tracked')}
    protocol=json.loads((root/'protocol.json').read_text())
    if latest.exists():
        saved=torch.load(latest,map_location='cpu');assert saved['protocol']==protocol
        instance.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer']);step=saved['step']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    if full:
        images,selected=loaders(historical_config(cfg),'train');assert selected==manifest['train'];dataset=images.dataset
    else:
        sources=paths_for(Path(manifest['source_cache']),'train',manifest['train'])
        targets=[cache_path(root,'train',ix) for ix in manifest['train']['indices']];dataset=FrozenData(sources,targets,root if cfg.get('fast_raw_logits',False) else None)
    if full and step==0:
        # Verify the experiment-local differentiable forward against original
        # frozen encoding, with no training and no future RGB in Student input.
        instance.eval()
        sample=tree_to(next(iter(image_loader(dataset,0,[0]))),'cuda')
        with torch.no_grad():
            state,ego=encode(instance,sample.pop('img'),sample)
            source=torch.load(paths_for(Path(manifest['source_cache']),'train',manifest['train'])[0],map_location='cpu')
            torch.testing.assert_close(state['representation'][-1]['query_features'].cpu(),source['encoded']['query_features'],atol=1e-5,rtol=0)
            torch.testing.assert_close(state['representation'][-1]['query_points'].cpu(),source['encoded']['query_points'],atol=1e-5,rtol=0)
            torch.testing.assert_close(ego.cpu(),source['ego'],atol=1e-5,rtol=0)
        write_json(directory/'initial_parity.json',dict(status='passed',sample=manifest['train']['indices'][0],history_images=30,precision='FP32'))
        del sample,state,ego,source
    total=cfg.fast_epochs*len(dataset);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    audit=None
    def save():
        atomic(dict(state_dict=instance.state_dict(),optimizer=optimizer.state_dict(),step=step,protocol=protocol,
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),latest)
    for ep in range(step//len(dataset),cfg.fast_epochs):
        order=np.random.default_rng(cfg.fast_seed+1009*ep).permutation(len(dataset)).tolist();offset0=step%len(dataset)
        if full:stream=image_loader(dataset,cfg.fast_workers,order[offset0:])
        else:stream=DataLoader(Subset(dataset,order[offset0:]),batch_size=1,num_workers=cfg.fast_workers,collate_fn=lambda r:r[0])
        mode(instance,full,True)
        for offset,item in enumerate(stream,offset0):
            position=order[offset]
            factor=schedule_factor(step,cfg.fast_warmup,total)
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
            if full:
                with np.load(cache_path(root,'train',manifest['train']['indices'][position])) as d:
                    target={k:d[k] for k in ('teacher','fast_0','fast_1','fast_2')}
                if cfg.get('fast_raw_logits',False):attach_raw_target(root,'train',manifest['train']['indices'][position],manifest['train']['scenes'][position],target)
                data=tree_to(item,'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    state,ego=encode(instance,data.pop('img'),data);packed=pack_output(instance,state,ego,data,position_gradient=cfg.get('fast_position_gradient',True))
                task,_=objective(packed)
                out=dict(points=packed['future_pred_points'],logits=packed['future_pred_logits'])
            else:
                source,target=item
                out=forecast_with_position_gradient(instance,tree_to(source['encoded'],'cuda'),source['stamps'].cuda(),source['ego'].cuda(),position_gradient=cfg.get('fast_position_gradient',True))
                task,_=cached_task(objective,out,tree_to(source['metas'],'cuda'),target['constant_task'])
            if kd and cfg.get('fast_raw_logits',False):
                from tools.sparseworld_raw_logit_kd import raw_occupancy_kd
                distill,detail=raw_occupancy_kd(instance,out,target,cfg.fast_arm_policies[arm],cfg.fast_gaussian_sigma)
            else:distill,detail=occupancy_kd(instance,out,target,policy=cfg.get('fast_kd_policy','legacy')) if kd else (task*0,[])
            value=task+cfg.fast_kd_weight*distill
            assert torch.isfinite(value),f'Nonfinite {arm} step {step}'
            optimizer.zero_grad(set_to_none=True);value.backward()
            grad=torch.nn.utils.clip_grad_norm_(instance.parameters(),5.,error_if_nonfinite=True)
            if audit is None:
                audit=dict(trainable_parameters=sum(p.numel() for p in instance.parameters() if p.requires_grad),trainable_names=trainable,
                           no_gradient_names=[n for n,p in instance.named_parameters() if p.requires_grad and p.grad is None])
                write_json(directory/'gradient_audit.json',audit)
            optimizer.step();step+=1
            row=dict(step=step,epoch=ep+1,position=position,index=manifest['train']['indices'][position],task=float(task.detach()),kd=float(distill.detach()),total=float(value.detach()),grad_norm=float(grad),lr=optimizer.param_groups[0]['lr'],fast=detail,elapsed=time.monotonic()-start)
            with logpath.open('a') as f:f.write(json.dumps(row)+'\n')
            if step%32==0:print(f'[{arm}] {step}/{total} task={row["task"]:.4f} kd={row["kd"]:.4f} elapsed={row["elapsed"]:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB',flush=True)
            if step%128==0:save()
            del out,task,distill,value
            if full:del packed,state,data
            else:del source
        save()
        if frozen_original:
            assert all(torch.equal(instance.state_dict()[n].cpu(),v) for n,v in frozen_original.items()),'Frozen perception changed'
        if geometry_original:
            assert all(torch.equal(instance.state_dict()[n].cpu(),v) for n,v in geometry_original.items()),'Native detached geometry heads changed'
        assert all(torch.equal(instance.state_dict()[n].cpu(),v) for n,v in bn_original.items()),'Fixed BN statistics changed'
        destination=directory/f'epoch_{ep+1}.npz'
        if not destination.exists():evaluate(instance,cfg,manifest,root,destination,full)
    # A crash may occur after saving the final training state but before the
    # validation write. Recover all missing epoch evaluations on resume.
    for completed_epoch in range(1,cfg.fast_epochs+1):
        destination=directory/f'epoch_{completed_epoch}.npz'
        if not destination.exists():
            if completed_epoch!=cfg.fast_epochs:raise RuntimeError('Intermediate epoch checkpoint unavailable')
            evaluate(instance,cfg,manifest,root,destination,full)
    write_json(directory/'completion.json',dict(status='complete',updates=step,epochs=cfg.fast_epochs,train_samples=len(dataset),val_samples=cfg.fast_samples_val,
                                               frozen_perception_unchanged=not full,native_geometry_heads_unchanged=bool(geometry_original),bn_statistics_unchanged=True,
                                               peak_gib=torch.cuda.max_memory_allocated()/2**30,checkpoint_sha256=sha256(latest)))
    del instance,objective,optimizer;torch.cuda.empty_cache()


def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(args.config)
    if args.epochs:cfg.fast_epochs=args.epochs
    if args.workers is not None:cfg.fast_workers=args.workers
    if args.smoke:cfg.fast_samples_train=2;cfg.fast_samples_val=2;cfg.fast_workers=0
    root=Path(cfg.fast_root+('_smoke' if args.smoke else ''));root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    strict=Config.fromfile(cfg.student_model_config)
    manifest=json.loads(Path('out/future2current/feature_identity_2048x512/manifest.json').read_text())
    if args.smoke:
        for split,n in [('train',cfg.fast_samples_train),('val',cfg.fast_samples_val)]:
            manifest[split]['indices']=manifest[split]['indices'][:n];manifest[split]['scenes']=manifest[split]['scenes'][:n]
    assert len(set(manifest['train']['scenes'])&set(manifest['val']['scenes']))==0
    assert sha256(cfg.student_checkpoint)==cfg.student_checkpoint_sha256==manifest['checkpoint_sha256']
    protocol=dict(train=manifest['train'],val=manifest['val'],checkpoint=cfg.student_checkpoint,checkpoint_sha256=cfg.student_checkpoint_sha256,
                  epochs=cfg.fast_epochs,seed=cfg.fast_seed,kd_weight=cfg.fast_kd_weight,lr=cfg.fast_lr,backbone_lr=cfg.fast_backbone_lr,
                  warmup=cfg.fast_warmup,task='native current + temporal + all six future + trajectory',kd='hard native teacher occupancy; 17-class sigmoid BCE on speed>=2 GT; horizons 2/4/6',
                  student_score='max per class and native closing; omit hard score cutoff during KD; detach voxel assignments',
                  teacher='fixed identical SparseWorld perception, future five-frame windows',precision='FP32 teacher/evaluation, BF16 full training, FP32 frozen training',bn='fixed statistics; affine trainable in full',arms=cfg.fast_arms)
    if 'fast_kd_policy' in cfg:
        protocol.update(kd_policy=cfg.fast_kd_policy,position_gradient=cfg.fast_position_gradient,
                        target_cache=str(Path(cfg.fast_target_cache).resolve()),
                        kd='teacher nonempty hard occupancy on GT speed>=2; unsupported native raster cells ignored; original fast-cell denominator; horizons 2/4/6',
                        student_score='native hard point cutoff + max per class + closing; stable logit-space BCE; geometry/bin/keep decisions detached',
                        task_gradient='strict native detach before future point outputs; all task terms retained')
    if cfg.get('fast_raw_logits',False):
        raw_protocol=json.loads((root/'raw_cache_protocol.json').read_text())
        assert raw_protocol['train']==manifest['train'] and raw_protocol['val']==manifest['val']
        assert raw_protocol['checkpoint_sha256']==cfg.student_checkpoint_sha256
        protocol.update(kd='GT fast mask; raw teacher/student voxel logits; joint finite support; sum/(original mask size * 17), three horizons',
                        student_score=raw_protocol['raw_readout'],arm_policies=dict(cfg.fast_arm_policies),gaussian_sigma=cfg.fast_gaussian_sigma,
                        matched_task_root=cfg.fast_matched_task_root,raw_cache_protocol=raw_protocol,kd_policy='raw_logits_suite')
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol,'Protocol differs'
    write_json(root/'protocol.json',protocol)
    write_json(root/'status.json',dict(status='running',stage=args.stage,pid=os.getpid(),started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))
    if 'fast_target_cache' in cfg:reuse_targets(cfg,manifest,root)
    elif args.stage in ('all','prepare'):prepare(cfg,strict,manifest,root)
    if args.stage in ('all','train'):
        for arm in args.arms or cfg.fast_arms:train_arm(cfg,strict,manifest,root,arm)
    write_json(root/'status.json',dict(status='complete',stage=args.stage,finished_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/opus/sparseworld_fast_motion_occupancy_kd.py');p.add_argument('--stage',choices=['all','prepare','train'],default='all');p.add_argument('--arms',nargs='+');p.add_argument('--epochs',type=int);p.add_argument('--workers',type=int);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:main(args)
    except Exception as error:
        cfg=Config.fromfile(args.config)
        root=Path(cfg.fast_root+('_smoke' if args.smoke else ''))
        write_json(root/'status.json',dict(status='failed',error=f'{type(error).__name__}: {error}',pid=os.getpid()))
        traceback.print_exc();raise
