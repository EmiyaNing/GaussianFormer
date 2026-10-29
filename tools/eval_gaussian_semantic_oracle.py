"""Fit only nonnegative GS semantics to each validation frame's own GT.

Output-space diagnostic, not network fine-tuning or ordinary validation.
"""
from pathlib import Path
import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader, Subset
from tools.stat_boundary_gaussians import read, write, sha, forward, to_cuda
from tools.boundary_gaussian_stats.data import EvaluationFrames, FeatureCache
from tools.eval_gaussian_sizes import paths, build, CLASS_NAMES, CLASS_COLUMNS
from tools.eval_gaussian_center_oracle import objective, loss_terms, TRANSITIONS
from tools.eval_gaussian_attribute_oracle import Frame
from tools.semantic_field_support import extension

RUN = ROOT / 'work_dirs/gaussian_semantic_oracle_full_val_v1'
HISTORY = ROOT / 'work_dirs/gaussian_sizes_full_val'
ATTRIBUTE_HISTORY = ROOT / 'work_dirs/gaussian_attribute_oracle_full_val_v1'
STEPS = (0, 5, 10, 20)
LR = .05
SEMANTIC_STATS = ('GS', 'entries', 'sum_semantic_mass', 'initial_sum_semantic_mass',
    'sum_abs_change', 'max_abs_change', 'changed_GS', 'changed_argmax_GS',
    'zero_entries', 'zero_mass_GS', 'entropy_sum', 'initial_entropy_sum',
    'mass_increased_GS', 'mass_decreased_GS')


def initialize():
    config, checkpoint = paths('tiny')
    cfg = Config.fromfile(config)
    previous = read(ATTRIBUTE_HISTORY/'protocol.json')
    assert read(ATTRIBUTE_HISTORY/'delivery_audit.json')['status'] == 'passed'
    assert sha(checkpoint) == previous['checkpoint_sha256']
    for name, digest in previous['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    files = list(dict.fromkeys([Path(__file__), *[ROOT/p for p in previous['source_sha256']]]))
    protocol = dict(version=1, model='tiny', frames=6019, scenes=150,
        GS=76800, queries=600, children=128, seed=928, steps=list(STEPS),
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
        config=str(config.relative_to(ROOT)),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256=sha(cfg.val_dataset_config.imageset),
        active='All 17 post-softplus nonnegative semantic values per final-stage predicted GS',
        parameterization='Direct output values initialized exactly at original semantics; project >=0; no normalization or upper bound',
        fixed='Network, centers, scales, rotations, opacity, empty Gaussian/scalar/bias, GS count, parent queries',
        loss='Identical configured final-stage Occupancy CE+Lovasz as attribute experiment; no Set or coverage loss',
        domain='All camera-valid GT labels 0..17; CE includes empty; Lovasz ignores 17',
        optimizer=dict(type='Adam', lr=LR, betas=[.9,.999], eps=1e-8,
            weight_decay=0, reset='Independent zero state per frame'),
        interpretation='Same-frame validation-GT-assisted output fitting, not normal inference or a new checkpoint',
        comparison='20 updates; 17 semantic freedoms per GS vs C3/S3/R3/O1 with different physical constraints; not a causal error ranking',
        coverage='Geometry and opacity fixed; geometric/native support exactly fixed. Same-class support uses updated semantic argmax; zero-total-semantic GS have no class.',
        semantic_statistics=list(SEMANTIC_STATS), transitions=list(TRANSITIONS),
        class_columns=CLASS_COLUMNS,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    if (RUN/'protocol.json').exists():
        assert read(RUN/'protocol.json') == protocol, 'Protocol changed; choose a new output directory'
    else:
        write(RUN/'protocol.json', protocol)
        cfg.dump(RUN/'resolved_config.py')
        for p in files:
            dest = RUN/'sources'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    return cfg, protocol


class SemanticFrame(Frame):
    def render_semantic(self, semantic):
        semantic18 = torch.cat((semantic, torch.zeros_like(semantic[..., :1])), -1)
        z = self.agg(self.xyz[None], self.base.means, self.base.opacities.reshape(1,-1),
            semantic18, self.base.scales, self.args[-1][:,:-1]).T
        return torch.cat((z[:17], self.empty_logits[None]), 0)

    @torch.no_grad()
    def semantic_coverage(self, semantic):
        means = self.base.means[0].contiguous()
        alpha = self.base.opacities.flatten().contiguous()
        inverse = self.args[-1][0,:-1].flatten(1)[:,[0,4,8,1,5,2]].contiguous()
        centers = ((means-self.agg.pc_min)/self.agg.grid_size).int().contiguous()
        radii = torch.ceil(self.base.scales[0].max(-1).values*
            self.agg.scale_multiplier/self.agg.grid_size).int().contiguous()
        classes = semantic[0].argmax(-1).int().contiguous()
        classes[semantic[0].sum(-1)==0] = -1
        _,counts,_ = extension().support(means,inverse,alpha,centers,radii,self.xyz_full,
            self.lookup,self.labels,classes,classes,len(self.foreground_ids),
            self.protocol['queries'],self.protocol['children'],self.agg.H,self.agg.W,self.agg.D)
        self.last_support_counts = counts
        hit = counts[:,[0,1,2,3,4,5,7]] > 0
        cc = torch.zeros((17,8), device=means.device, dtype=torch.int64)
        y = self.labels[self.foreground_ids].long()
        cc[:,0] = torch.bincount(y,minlength=17)
        for j in range(7): cc[:,j+1] = torch.bincount(y[hit[:,j]],minlength=17)
        return cc

    @torch.no_grad()
    def semantic_statistics(self, semantic):
        initial = self.base.semantics.double()
        current = semantic.double()
        delta = (current-initial).abs()
        mass, oldmass = current.sum(-1), initial.sum(-1)
        def entropy(x, total):
            p = x/total.clamp_min(1e-30)[...,None]
            return -(p*p.clamp_min(1e-30).log()).sum(-1)
        values = [current.new_tensor(current.shape[1]), current.new_tensor(current.numel()),
            mass.sum(), oldmass.sum(), delta.sum(), delta.max(),
            (delta.max(-1).values>1e-6).sum(),
            (current.argmax(-1)!=initial.argmax(-1)).sum(), (current==0).sum(),
            (mass==0).sum(), entropy(current,mass).sum(), entropy(initial,oldmass).sum(),
            (mass>oldmass+1e-6).sum(), (mass<oldmass-1e-6).sum()]
        return torch.stack(values)

    @torch.no_grad()
    def observe_semantic(self, semantic, logits):
        pred = logits.argmax(0)
        cm = torch.bincount(self.target*18+pred,minlength=324).reshape(18,18)
        fg, oldfg, newfg = self.target<17, self.base_pred<17, pred<17
        oldok, newok = self.base_pred==self.target, pred==self.target
        masks = [fg&~oldfg&newfg, fg&oldfg&~newfg, ~fg&oldfg&~newfg, ~fg&~oldfg&newfg,
            ~oldok&newok, oldok&~newok, fg&oldfg&newfg&~oldok&newok,
            fg&oldfg&newfg&oldok&~newok]
        return dict(confusion=cm, class_coverage=self.semantic_coverage(semantic),
            transitions=torch.stack([x.sum() for x in masks]),
            semantic_statistics=self.semantic_statistics(semantic))


def refine(net, out, batch, occ, protocol, audit=False):
    frame = SemanticFrame(net,out,batch,protocol)
    semantic = torch.nn.Parameter(frame.base.semantics.clone())
    optimizer = torch.optim.Adam([semantic],lr=LR,betas=(.9,.999),eps=1e-8,weight_decay=0)
    original = {k:getattr(frame.base,k).clone() for k in ('means','scales','rotations','opacities','semantics')}
    rows, trajectory, gradients = {}, [], []
    reference = None
    if audit:
        full_semantic = torch.cat((torch.cat((semantic,torch.zeros_like(semantic[...,:1])),-1),
                                   frame.args[2][:,-1:]),1)
        m,a,_,s,inv = frame.args
        full = frame.agg(frame.xyz[None],m,a.reshape(1,-1),full_semantic,s,inv).T
        full = torch.cat((full[:17],full[17:]-frame.head.empty_bias.detach()),0)
        torch.testing.assert_close(full,frame.base_logits,rtol=0,atol=0)
        reference, = torch.autograd.grad(loss_terms(full,frame.target,occ)[0],semantic)
        old = occ.loss_voxel([frame.base_logits[None]],frame.xyz[None],frame.target[None],None)*occ.weight
        torch.testing.assert_close(old,loss_terms(frame.base_logits,frame.target,occ)[0],rtol=1e-6,atol=1e-6)
    for step in range(21):
        optimizer.zero_grad(set_to_none=True)
        logits = frame.render_semantic(semantic)
        if step == 0: torch.testing.assert_close(logits,frame.base_logits,rtol=0,atol=0)
        values = loss_terms(logits,frame.target,occ)
        assert bool(torch.isfinite(values).all()), ('nonfinite objective',step)
        trajectory.append(values.detach())
        if step in STEPS:
            observed = frame.observe_semantic(semantic,logits.detach())
            for key,value in observed.items(): rows.setdefault(key,[]).append(value)
        if step == 20: break
        values[0].backward()
        assert semantic.grad is not None and bool(torch.isfinite(semantic.grad).all())
        if step == 0 and reference is not None:
            torch.testing.assert_close(semantic.grad,reference,rtol=0,atol=0)
        gradients.append(semantic.grad.abs().max().detach())
        optimizer.step()
        with torch.no_grad(): semantic.clamp_(min=0)
    for key,value in original.items():
        assert torch.equal(getattr(frame.base,key),value),key
    assert all(p.grad is None for p in net.parameters())
    assert bool(torch.isfinite(semantic).all()) and float(semantic.detach().min())>=0
    result = {key:torch.stack(value).cpu().numpy() for key,value in rows.items()}
    for key in ('class_coverage',):
        for j in range(4):
            np.testing.assert_array_equal(result[key][j][:,[0,1,2,3,7]],result[key][0][:,[0,1,2,3,7]])
    result['loss_trajectory'] = torch.stack(trajectory).cpu().numpy()
    result['max_gradient'] = torch.stack(gradients).max().cpu().numpy()
    if audit:
        # Independently enumerate all GS for fixed foreground samples at the final step.
        cls = semantic.detach()[0].argmax(-1)
        cls[semantic.detach()[0].sum(-1)==0] = -1
        means = frame.base.means[0]
        inverse = frame.args[-1][0,:-1].flatten(1)[:,[0,4,8,1,5,2]]
        centers = ((means-frame.agg.pc_min)/frame.agg.grid_size).int()
        radii = torch.ceil(frame.base.scales[0].max(-1).values*frame.agg.scale_multiplier/frame.agg.grid_size).int()
        for t in np.random.default_rng(928).choice(len(frame.foreground_ids),min(16,len(frame.foreground_ids)),replace=False):
            index=frame.foreground_ids[t];x=frame.xyz_full[index]
            grid=((x-frame.agg.pc_min[0])/frame.agg.grid_size).int()
            box=((grid>=centers-radii[:,None])&(grid<=centers+radii[:,None])).all(1)
            delta=means-x
            power=-.5*(inverse[:,:3]*delta.square()).sum(1)-(
                inverse[:,3:]*delta[:,[0,1,0]]*delta[:,[1,2,2]]).sum(1)
            distance=(-2*power).clamp_min(0).sqrt()
            positive=box&(frame.base.opacities.flatten()*power.exp()>0)
            same=cls==frame.labels[index]
            expected=[int((positive&(distance<=r)).sum()) for r in (1,2,3)]
            expected += [int((positive&(distance<=r)&same).sum()) for r in (1,2,3)]
            expected += [expected[3],int(positive.sum())]
            np.testing.assert_array_equal(frame.last_support_counts[t].cpu().numpy(),expected)
        result['_pilot'] = dict(native_zero_logits_exact=True,full_renderer_semantic_gradient_exact=True,
            original_occupancy_loss_exact=True,geometry_and_opacity_frozen=True,
            no_network_gradients=True,geometric_support_bitwise_constant=True,
            same_class_coverage_uses_updated_labels=True,
            dynamic_coverage_independently_enumerated=True)
    return result


def numerical_audit(net,protocol):
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    device = next(net.parameters()).device
    means = torch.tensor([[[-.13,-.17,.31],[.43,.29,.71],[-.53,.41,.51]]],device=device)
    scales = means.new_tensor([[[.23,.29,.31],[.28,.25,.33],[.27,.32,.24]]])
    rotations = means.new_tensor([[[1.,0,0,0]]]).repeat(1,3,1)
    opacity = means.new_tensor([[[.8],[.9],[.7]]])
    semantic = means.new_full((1,3,17),.2)
    semantic[0,0,4]=3.;semantic[0,1,11]=3.;semantic[0,2,16]=3.
    g = GaussianPrediction(means,scales,rotations,opacity,semantic)
    x=torch.arange(-.6,.61,.4,device=device);z=torch.arange(-.4,1.21,.4,device=device)
    xyz=torch.stack(torch.meshgrid(x,x,z,indexing='ij'),-1).reshape(-1,3)
    with torch.no_grad():
        m,a,s,sc,inv=net.head.prepare_gaussian_args(g)
        full=net.head.aggregator(xyz[None],m,a.reshape(1,-1),s,sc,inv).T
        full=torch.cat((full[:17],full[17:]-net.head.empty_bias),0)
    batch=dict(occ_label=torch.full((len(xyz),),17,device=device),
               occ_cam_mask=torch.ones(len(xyz),device=device,dtype=torch.bool))
    batch['occ_label'][:len(xyz)//2]=4
    frame=SemanticFrame(net,dict(sampled_xyz=xyz[None],pred_occ=[full[None]],gaussian=g),batch,protocol)
    p=semantic.detach().clone().requires_grad_(True)
    loss=torch.nn.functional.cross_entropy(frame.render_semantic(p)[None].double(),frame.target[None])
    grad,=torch.autograd.grad(loss,p)
    errors=[]
    for idx in grad.abs().flatten().argsort(descending=True)[:5].tolist():
        plus=semantic.clone();minus=semantic.clone()
        plus.flatten()[idx]+=.002;minus.flatten()[idx]-=.002
        fd=(torch.nn.functional.cross_entropy(frame.render_semantic(plus)[None].double(),frame.target[None])-
            torch.nn.functional.cross_entropy(frame.render_semantic(minus)[None].double(),frame.target[None]))/.004
        error=float((fd-grad.flatten()[idx]).abs()/grad.flatten()[idx].abs().clamp_min(1e-3))
        assert error<.04,(idx,float(fd),float(grad.flatten()[idx]),error)
        errors.append(error)
    return dict(status='passed',coordinates=5,max_relative_error=max(errors))


def pilot():
    from dataset.utils import custom_collate_fn_temporal
    cfg,protocol=initialize();net=build(cfg,'tiny');net.requires_grad_(False);occ=objective(cfg)
    numeric=numerical_audit(net,protocol)
    dataset=EvaluationFrames(cfg,'val');cache=FeatureCache(net,dataset)
    try:
        batch=to_cuda(custom_collate_fn_temporal([dataset[0]]))
        with torch.no_grad(): out=forward(net,batch)
        start=time.monotonic();result=refine(net,out,batch,occ,protocol,True)
        scene=dataset.keyframes[0][0]
        with np.load(HISTORY/'tiny/scenes'/f'{scene}.npz') as old:
            for key in ('confusion','class_coverage'):
                np.testing.assert_array_equal(result[key][0],old[key][0])
        checks=result.pop('_pilot')
        np.savez_compressed(RUN/'pilot_arrays.npz',**result)
        info=dict(status='passed',numeric=numeric,**checks,historical_baseline_exact=True,
            seconds_per_frame=time.monotonic()-start)
        write(RUN/'pilot.json',info);print(json.dumps(info),flush=True)
    finally:
        cache.close();net.flush_memory_phase();del net,cache,dataset
        gc.collect();torch.cuda.empty_cache()


def evaluate(workers):
    from dataset.utils import custom_collate_fn_temporal
    cfg,protocol=initialize();assert read(RUN/'pilot.json')['status']=='passed'
    if (RUN/'complete.json').exists(): return
    net=build(cfg,'tiny');net.requires_grad_(False);occ=objective(cfg)
    dataset=EvaluationFrames(cfg,'val');assert len(dataset)==6019
    scenes={}
    for i,(scene,_) in enumerate(dataset.keyframes): scenes.setdefault(scene,[]).append(i)
    assert len(scenes)==150
    pending=[]
    for scene,ids in scenes.items():
        path=RUN/'scenes'/f'{scene}.json'
        if path.exists():
            old=read(path);assert old['indices']==ids and old['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert path.with_suffix('.npz').exists()
        else: pending+=ids
    loader=DataLoader(Subset(dataset,pending),batch_size=1,num_workers=workers,shuffle=False,
        collate_fn=custom_collate_fn_temporal,pin_memory=True,persistent_workers=workers>0,
        **(dict(prefetch_factor=2) if workers else {}))
    cache=FeatureCache(net,dataset);start=time.monotonic();done=0;active=None
    try:
        for index,batch in zip(pending,loader):
            scene,f=dataset.keyframes[index]
            if active is None:
                active=scene;arrays={};indices=[];tokens=[]
                original=np.load(HISTORY/'tiny/scenes'/f'{scene}.npz')
            assert active==scene
            batch=to_cuda(batch)
            with torch.no_grad(): out=forward(net,batch)
            result=refine(net,out,batch,occ,protocol)
            for key in ('confusion','class_coverage'):
                np.testing.assert_array_equal(result[key][0],original[key][len(indices)])
            for key,value in result.items(): arrays.setdefault(key,[]).append(value)
            indices.append(index);tokens.append(dataset.base.scene_infos[scene][f]['token']);done+=1
            if index==scenes[scene][-1]:
                original.close();path=RUN/'scenes'/f'{scene}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,**{k:np.stack(v) for k,v in arrays.items()})
                path.with_suffix('.tmp').replace(path)
                write(path.with_suffix('.json'),dict(status='complete',scene=scene,indices=indices,tokens=tokens,
                    frames=len(indices),checkpoint_sha256=protocol['checkpoint_sha256']))
                active=None
            if done%10==0 or active is None:
                status=dict(status='running',frames=6019-len(pending)+done,expected_frames=6019,
                    scenes=len(list((RUN/'scenes').glob('*.json'))),expected_scenes=150,
                    steps=20,elapsed_seconds=time.monotonic()-start,
                    diagnostic_frames_per_second=done/(time.monotonic()-start))
                write(RUN/'progress.json',status);print(json.dumps(status),flush=True)
            del out,batch,result
        all_paths=list((RUN/'scenes').glob('*.json'))
        ids=[i for p in all_paths for i in read(p)['indices']]
        assert len(all_paths)==150 and sorted(ids)==list(range(6019))
        for name,digest in protocol['source_sha256'].items(): assert sha(ROOT/name)==digest,name
        assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
        info=dict(status='complete',frames=6019,scenes=150,steps=20,
            seconds=time.monotonic()-start,checkpoint_sha256=protocol['checkpoint_sha256'])
        write(RUN/'complete.json',info);write(RUN/'progress.json',info)
    finally:
        cache.close();net.flush_memory_phase();del net,cache,dataset,loader
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('pilot','run'),default='run')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(RUN/'status.json',dict(status='running',action=args.action,pid=os.getpid()))
        try:
            if args.action=='pilot': pilot()
            else: evaluate(args.workers)
            write(RUN/'status.json',dict(status=args.action+'_complete'))
        except BaseException as exc:
            write(RUN/'status.json',dict(status='failed',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__': main()
