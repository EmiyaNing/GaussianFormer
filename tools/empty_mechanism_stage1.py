"""Camera-default path interventions; checkpoints are never modified on disk."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_decision_optimization import (CONFIG,read,write_json,file_sha256,build_loader,to_cuda,seed_all)

RUN=ROOT/'work_dirs/empty_mechanism_stage1'
EPOCHS=[1,3,5]
BRANCHES=['set_cls','voxel_ce','voxel_lovasz','voxel_total','joint']
RADII=[1e-4,1e-3,1e-2]
PATHS=['amplitude','composition','full']
PARAM_BRANCHES=['set_cls','voxel_total','joint']
PARAM_RADII=[1e-5,1e-4]


def initialize():
    RUN.mkdir(exist_ok=True);path=RUN/'protocol.json'
    if path.exists():return read(path)
    from mmengine import Config
    cfg=Config.fromfile(CONFIG)
    record=dict(at=datetime.now().astimezone().isoformat(),config=CONFIG,
        scope='all occupancy statistics and voxel losses use original camera mask; no spatial split comparison',
        checkpoint={},scene_samples={},output_branches=BRANCHES,output_radii=RADII,output_primary_radius=.001,
        parameter_branches=PARAM_BRANCHES,parameter_radii=PARAM_RADII,parameter_primary_radius=.0001,
        paths=PATHS,output_update='negative raw-logit gradient, same relative output L2 radius',
        parameter_update='normalized SGD on final cls_branch only; restore before every intervention',
        geometry_opacity_and_empty_frozen=True,set_loss='unchanged final-stage training definition and actual multipliers',
        losses=[dict(x) for x in cfg.loss.loss_cfgs],
        census=dict(epoch=5,validation_frames=1024),
        matches=dict(class_same=True,distance_bin_m=5,height_bin_m=1,within_frame=True,max_pairs=512),
        profile_samples_per_gt_group=1024,bootstrap_repetitions=500,seed=928,
        controls=dict(renderer_atol=.0002,renderer_rtol=.0002,split_gaussian_frames_per_checkpoint=2),
        limitations='one backbone seed; local probes are not original AdamW trajectory or full retraining; exploratory validation',
        source_sha256=file_sha256(__file__))
    for epoch in EPOCHS:
        checkpoint=f'work_dirs/center_support_contribution1/train/a0_original_seed928/epoch_{epoch}.pth'
        record['checkpoint'][str(epoch)]=dict(path=checkpoint,sha256=file_sha256(ROOT/checkpoint))
    for split,count in [('train',16),('val',32)]:
        dataset,_,_=build_loader(split);by_scene={}
        for i,(scene,frame) in enumerate(dataset.keyframes):by_scene.setdefault(scene,dict(index=i,scene=scene,scene_frame=int(frame)))
        scenes=sorted(by_scene)
        # Train scenes must belong to the fitting cohort used in preceding experiments.
        if split=='train':
            fit=set(read(ROOT/'work_dirs/empty_decision_optimization/protocol.json')['fit_scenes'])
            scenes=[s for s in scenes if s in fit]
        chosen=np.linspace(0,len(scenes)-1,count,dtype=int)
        record['scene_samples'][split]=[by_scene[scenes[k]] for k in chosen]
    selected=record['scene_samples']['val'][:16]
    record['parameter_pairs']=[dict(source=selected[i],heldout=selected[i+1]) for i in range(0,16,2)]
    write_json(path,record);return record


def decompose(semantic):
    amplitude=semantic.double().sum(-1,keepdim=True)
    composition=semantic.double()/amplitude.clamp_min(1e-30)
    return amplitude,composition


def factorial_semantics(before,after):
    a,p=decompose(before);b,q=decompose(after)
    return {'amplitude':(b*p).to(before),'composition':(a*q).to(before),'full':after,
            'sham':(a*p).to(before)}


def inverse_softplus(value):
    x=value.double().clamp_min(1e-30)
    return (x+torch.log(-torch.expm1(-x))).to(value)


def modules(config):
    from loss import OPENOCC_LOSS
    settings=next(copy.deepcopy(x) for x in config.loss.loss_cfgs if 'SetLoss' in x['type'])
    settings['stage_weights']=settings['stage_weights'][-1:]
    set_loss=OPENOCC_LOSS.build(settings).cuda()
    voxel=next(copy.deepcopy(x) for x in config.loss.loss_cfgs if x['type']=='OccupancyLoss')
    ce=copy.deepcopy(voxel);ce['use_lovasz_loss']=False
    lovasz=copy.deepcopy(voxel);lovasz['multi_loss_weights']['loss_voxel_ce_weight']=0.
    return set_loss,OPENOCC_LOSS.build(ce).cuda(),OPENOCC_LOSS.build(lovasz).cuda()


class Frame:
    def __init__(self,network,output,batch,loss_modules):
        self.head=network.head;self.modules=loss_modules
        self.args=tuple(t.detach() for t in self.head.prepare_gaussian_args(output['gaussian']))
        self.sem=self.args[2][:,:-1,:17].detach();self.u=output['opus_pred_logits'][-1].detach()
        assert torch.allclose(self.sem,F.softplus(self.u),atol=1e-7,rtol=1e-7)
        labels=output['sampled_label'][0].long()
        mask=batch['occ_cam_mask'][0].bool().flatten()&(labels>=0)&(labels<18)
        self.mask=mask
        self.xyz=output['sampled_xyz'][:,mask].detach().float().contiguous();self.y=labels[mask]
        self.bias=self.head.empty_bias.detach().clone()
        self.native=output['pred_occ'][-1][:,:,mask].detach()
        self.base=self.render(self.sem).detach()
        self.parity_error=float((self.native-self.base).abs().max())
        assert torch.allclose(self.native,self.base,atol=2e-4,rtol=2e-4),self.parity_error
        payload=torch.zeros_like(self.args[2]);payload[:,:-1,0]=1.
        self.support=self.aggregate(payload)[0].detach()
        self.fg0=self.base[0,:17].argmax(0);self.pred0=self.base[0].argmax(0)
        self.correct=(self.y<17)&(self.pred0==self.y)
        self.recoverable=(self.y<17)&(self.fg0==self.y)&(self.pred0==17)
        self.cohorts={'recoverable':self.recoverable,'correct':self.correct,'true_empty':self.y==17}
        self.make_set_targets(output,batch)

    def aggregate(self,semantic,args=None):
        means,opacity,_,scales,inv=self.args if args is None else args
        return self.head.aggregator(self.xyz,means.contiguous(),opacity.reshape(1,-1).contiguous(),
            semantic.contiguous(),scales.contiguous(),inv.contiguous()).T

    def render(self,semantic):
        padded=torch.cat([semantic,torch.zeros_like(semantic[:,:,:1])],-1)
        all_sem=torch.cat([padded,self.args[2][:,-1:]],1)
        scores=self.aggregate(all_sem)[None]
        return torch.cat([scores[:,:17],scores[:,17:]-self.bias],1)

    def make_set_targets(self,output,batch):
        from loss.opus_set_loss import _nearest_indices
        module=self.modules[0]
        valid=output['opus_point_valid_masks'][-1][0].bool()
        points=output['opus_pred_points'][-1][0].detach()[valid]
        labels=batch['occ_label'][0].long().flatten();xyz=batch['occ_xyz'][0].reshape(-1,3)
        target=(labels>=0)&(labels<17)
        if module.final_gt_domain=='camera':target &= batch['occ_cam_mask'][0].bool().flatten()
        assert target.any()
        nearest=_nearest_indices(points,xyz[target],module.chunk_size)
        self.assigned=labels[target][nearest];self.point_valid=valid
        self.set_weights=module.class_weights.to(points)[self.assigned]*module._distance_weight(xyz[target][nearest])
        self.set_multiplier=module.stage_weights[-1]*module.lambda_cls*module.final_classification_weight
        self.target_audit=dict(predictions=int(valid.sum()),targets=int(target.sum()),domain=module.final_gt_domain,
                              set_multiplier=float(self.set_multiplier))

    def set_objective(self,raw):
        return self.modules[0]._sigmoid_focal(raw[0,self.point_valid],self.assigned,self.set_weights)*self.set_multiplier

    def audit_losses(self,output,batch):
        """Compare cached assignment and masked rendering to original loss APIs."""
        raw=self.u.clone().requires_grad_(True)
        points=output['opus_pred_points'][-1].detach().clone().requires_grad_(True)
        original,_=self.modules[0].components(dict(opus_pred_points=[points],opus_pred_logits=[raw],
            opus_point_valid_masks=[output['opus_point_valid_masks'][-1]],metas=batch))
        reference=original['stage_0/classification'];cached=self.set_objective(raw)
        g1=torch.autograd.grad(reference,raw,retain_graph=True)[0]
        g2=torch.autograd.grad(cached,raw,retain_graph=True)[0]
        geometry=original['stage_0/pred_to_gt']+original['stage_0/gt_to_pred']
        direct=torch.autograd.grad(geometry,raw,allow_unused=True)[0]
        geometry_norm=0. if direct is None else float(direct.double().norm())
        assert torch.allclose(reference,cached,atol=1e-6,rtol=1e-6)
        assert torch.allclose(g1,g2,atol=1e-7,rtol=1e-6)
        assert geometry_norm==0.
        full_inputs=dict(pred_occ=[self.native],sampled_xyz=self.xyz,sampled_label=self.y[None],
            occ_mask=torch.ones_like(self.y[None],dtype=torch.bool))
        ce,lovasz=self.field_objectives(self.base)
        native_ce=self.modules[1](full_inputs);native_lovasz=self.modules[2](full_inputs)
        assert torch.allclose(ce,native_ce,atol=1e-6,rtol=1e-6)
        assert torch.allclose(lovasz,native_lovasz,atol=1e-6,rtol=1e-6)
        return dict(classification_value_error=float((reference-cached).abs().detach()),
            classification_gradient_max_error=float((g1-g2).abs().max()),
            geometry_gradient_to_raw_logits_norm=geometry_norm,
            native_ce_error=float((ce-native_ce).abs()),native_lovasz_error=float((lovasz-native_lovasz).abs()))

    def field_objectives(self,scores):
        inputs=dict(pred_occ=[scores],sampled_xyz=self.xyz,sampled_label=self.y[None],
                    occ_mask=torch.ones_like(self.y[None],dtype=torch.bool))
        return self.modules[1](inputs),self.modules[2](inputs)

    def objectives(self,raw):
        scores=self.render(F.softplus(raw));ce,lovasz=self.field_objectives(scores);cls=self.set_objective(raw)
        return dict(set_cls=cls,voxel_ce=ce,voxel_lovasz=lovasz,voxel_total=ce+lovasz,joint=cls+ce+lovasz)

    @torch.no_grad()
    def measure(self,scores,semantic):
        front=scores[0,:17];top,fg=front.max(0);pred=scores[0].argmax(0);empty=scores[0,17]
        mass=front.sum(0);share=top/mass.clamp_min(1e-30);amp=mass/self.support.clamp_min(1e-30);margin=top-empty
        code=self.y*18+pred;counts=torch.bincount(code,minlength=324).reshape(18,18)
        ce,lovasz=self.field_objectives(scores)
        cls=self.set_objective(inverse_softplus(semantic))
        quantities=[ce,lovasz,cls]
        transition=[((self.y<17)&(pred==17)).sum(),((self.y==17)&(pred<17)).sum(),
            (self.correct&(pred==17)).sum(),(self.recoverable&(pred==self.y)).sum(),
            (fg!=self.fg0).sum(),((self.y<17)&(fg==self.y)).sum()]
        values=torch.stack(quantities+[t.float() for t in transition]).double().cpu().tolist()
        cohort={}
        for name,mask in self.cohorts.items():
            n=int(mask.sum());positive=mask&(self.support>1e-12)
            fields=[amp[positive].sum(),share[mask].sum(),top[mask].sum(),empty[mask].sum(),margin[mask].sum()]
            v=torch.stack(fields).double().cpu().tolist()
            cohort[name]=dict(n=n,supported=int(positive.sum()),amplitude_sum=v[0],share_sum=v[1],top_sum=v[2],empty_sum=v[3],margin_sum=v[4])
        return dict(counts=counts.cpu().tolist(),ce=values[0],lovasz=values[1],set_cls=values[2],
            empty_fn=int(values[3]),empty_fp=int(values[4]),correct_to_empty=int(values[5]),rescued=int(values[6]),
            foreground_winner_changed=int(values[7]),foreground_correct=int(values[8]),cohorts=cohort)

    @torch.no_grad()
    def controls(self):
        sham=self.render(factorial_semantics(self.sem,self.sem)['sham'])
        means,alpha,sem,scales,inv=self.args
        def duplicate(x):return torch.cat([x[:,:-1],x[:,:-1],x[:,-1:]],1)
        copied=duplicate(sem);opacity=torch.cat([alpha[:,:-1]/2,alpha[:,:-1]/2,alpha[:,-1:]],1)
        split=self.aggregate(copied,(duplicate(means),opacity,copied,duplicate(scales),duplicate(inv)))[None]
        split=torch.cat([split[:,:17],split[:,17:]-self.bias],1)
        result={}
        for name,scores in [('sham',sham),('split',split)]:
            assert torch.allclose(scores,self.base,atol=2e-4,rtol=2e-4),name
            result[name]=dict(max_score_error=float((scores-self.base).abs().max()),
                              prediction_changes=int((scores.argmax(1)!=self.base.argmax(1)).sum()))
        return result


@torch.no_grad()
def profile(frame,seed):
    front=frame.base[0,:17];top,fg=front.max(0);mass=front.sum(0);s=frame.support
    a=mass/s.clamp_min(1e-30);r=top/mass.clamp_min(1e-30);e=frame.base[0,17]
    fields=torch.stack([s,a,r,e,top-e,frame.xyz[0,:,:2].norm(dim=-1),frame.xyz[0,:,2]],-1).cpu().numpy()
    y=frame.y.cpu().numpy();pred=frame.pred0.cpu().numpy();fg=fg.cpu().numpy()
    recover=(y<17)&(fg==y)&(pred==17);correct=(y<17)&(pred==y)
    rng=np.random.default_rng(seed);sample=[];weight=[]
    for mask in [y<17,y==17]:
        ids=np.where(mask)[0];chosen=rng.choice(ids,min(1024,len(ids)),replace=False)
        sample.extend(chosen);weight.extend([len(ids)/max(len(chosen),1)]*len(chosen))
    sample=np.asarray(sample,dtype=np.int64)
    codes=y.astype(np.int64)*1000+np.minimum((fields[:,5]/5).astype(int),99)*10+np.clip(((fields[:,6]+1)//1).astype(int),0,9)
    fail=np.where(recover&(fields[:,0]>1e-12))[0];good=np.where(correct&(fields[:,0]>1e-12))[0]
    order=np.argsort(codes[good]);good=good[order];sorted_codes=codes[good]
    left=np.searchsorted(sorted_codes,codes[fail],side='left');right=np.searchsorted(sorted_codes,codes[fail],side='right')
    usable=right>left;eligible=fail[usable];left=left[usable];right=right[usable]
    selected=rng.choice(len(eligible),min(512,len(eligible)),replace=False)
    cases=eligible[selected]
    controls=good[left[selected]+(rng.random(len(selected))*(right[selected]-left[selected])).astype(int)]
    delta=np.empty((len(cases),4),np.float64)
    if len(cases):
        delta[:,:3]=np.log(fields[cases,:3].clip(1e-30))-np.log(fields[controls,:3].clip(1e-30))
        delta[:,3]=fields[cases,3]-fields[controls,3]
    result=dict(valid=len(y),occupied=int((y<17).sum()),recoverable=int(recover.sum()),correct=int(correct.sum()),
                recoverable_zero_support=int((recover&(fields[:,0]<=1e-12)).sum()),
                matched_population=len(eligible),pairs=len(cases),renderer_max_error=frame.parity_error)
    arrays=dict(X=fields[sample],gt=y[sample],fg=fg[sample],pred=pred[sample],weight=np.asarray(weight),
                paired_delta=delta,paired_weight=np.full(len(cases),len(eligible)/max(len(cases),1)))
    return result,arrays


def output_interventions(frame):
    raw=frame.u.clone().requires_grad_(True);objectives=frame.objectives(raw)
    gradients={}
    for i,(name,loss) in enumerate(objectives.items()):
        gradients[name]=torch.autograd.grad(loss,raw,retain_graph=i<len(objectives)-1)[0].detach()
    scale=float(raw.detach()[0,frame.point_valid].double().norm());rows=[]
    del objectives,raw
    with torch.no_grad():
        for branch in BRANCHES:
            grad=gradients[branch];norm=float(grad.double().norm())
            for radius in RADII:
                lr=radius*scale/max(norm,1e-30);updated=frame.u-lr*grad
                split=factorial_semantics(frame.sem,F.softplus(updated))
                for pathway in PATHS:
                    sem=split[pathway];scores=frame.render(sem)
                    rows.append(dict(branch=branch,radius=radius,pathway=pathway,
                        gradient_norm=norm,effective_sgd_lr=lr,
                        raw_step_relative=float((updated-frame.u).double().norm())/max(scale,1e-30),
                        outcome=frame.measure(scores,sem)))
    return rows


def build_network(epoch):
    from mmengine import Config
    from mmseg.models import build_segmentor
    import model
    protocol=initialize();cfg=Config.fromfile(CONFIG)
    network=build_segmentor(cfg.model).cuda().eval().requires_grad_(False)
    checkpoint=ROOT/protocol['checkpoint'][str(epoch)]['path']
    assert file_sha256(checkpoint)==protocol['checkpoint'][str(epoch)]['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'];network.load_state_dict(state,strict=True)
    return network,modules(cfg)


def network_output(network,batch):
    return network(imgs=batch['img'],metas={k:v for k,v in batch.items() if k!='img'})


def local(epoch,limit=0):
    p=initialize();seed_all(928);network,mods=build_network(epoch);dest=RUN/f'epoch{epoch}';dest.mkdir(exist_ok=True)
    start=time.monotonic();completed=[]
    for split in ['train','val']:
        selected=p['scene_samples'][split][:limit] if limit else p['scene_samples'][split]
        dataset,loader,_=build_loader(split,[x['index'] for x in selected])
        for n,(meta,batch) in enumerate(zip(selected,loader)):
            stem=f'{split}_{meta["index"]:04d}';path=dest/(stem+'.json')
            if path.exists():completed.append(str(path.relative_to(ROOT)));continue
            batch=to_cuda(batch)
            with torch.no_grad():out=network_output(network,batch)
            frame=Frame(network,out,batch,mods)
            with torch.no_grad():baseline=frame.measure(frame.base,frame.sem)
            controls=frame.controls() if split=='val' and n<2 else {}
            if controls:controls['loss_parity']=frame.audit_losses(out,batch)
            summary,arrays=profile(frame,928+meta['index'])
            np.savez_compressed(dest/(stem+'_profile.npz'),**arrays)
            rows=output_interventions(frame)
            record=dict(status='complete',epoch=epoch,split=split,frame=meta,baseline=baseline,profile=summary,
                interventions=rows,controls=controls,set_targets=frame.target_audit,
                empty_bias=float(frame.bias),empty_scalar=float(network.head.empty_scalar.detach()),
                geometry_raw_logits_direct_path=False,protocol_sha256=file_sha256(RUN/'protocol.json'),
                source_sha256=file_sha256(__file__))
            write_json(path,record);completed.append(str(path.relative_to(ROOT)))
            del frame,out,batch,rows
            print('output paths',epoch,split,n+1,'/',len(selected),'seconds',round(time.monotonic()-start,1),flush=True)
    if not limit:write_json(dest/'local_manifest.json',dict(status='complete',files=completed,source_sha256=file_sha256(__file__)))


def census():
    p=initialize();seed_all(928);network,mods=build_network(5);dest=RUN/'census';dest.mkdir(exist_ok=True)
    dataset,loader,_=build_loader('val');start=time.monotonic();records=[]
    local_dest=RUN/'epoch5'
    for index,batch in enumerate(loader):
        path=dest/f'frame_{index:04d}.json'
        if path.exists():records.append(read(path));continue
        local_path=local_dest/f'val_{index:04d}.json'
        if local_path.exists():
            prior=read(local_path);record=dict(frame=prior['frame'],baseline=prior['baseline'],profile=prior['profile'])
            shutil.copy2(local_dest/f'val_{index:04d}_profile.npz',dest/f'frame_{index:04d}.npz')
        else:
            batch=to_cuda(batch)
            with torch.no_grad():
                out=network_output(network,batch);frame=Frame(network,out,batch,mods)
                baseline=frame.measure(frame.base,frame.sem);summary,arrays=profile(frame,928+index)
            scene,sf=dataset.keyframes[index];record=dict(frame=dict(index=index,scene=scene,scene_frame=int(sf)),baseline=baseline,profile=summary)
            np.savez_compressed(dest/f'frame_{index:04d}.npz',**arrays)
            del out,frame
        record.update(status='complete',source_sha256=file_sha256(__file__),protocol_sha256=file_sha256(RUN/'protocol.json'))
        write_json(path,record);records.append(record)
        if index==0 or (index+1)%128==0:print('full default-mask census',index+1,'/1024','seconds',round(time.monotonic()-start,1),flush=True)
    write_json(dest/'manifest.json',dict(status='complete',frames=[r['frame'] for r in records],
        source_sha256=file_sha256(__file__),protocol_sha256=file_sha256(RUN/'protocol.json')))


def parameter_probes(epoch,limit=0):
    from dataset.utils import custom_collate_fn_temporal
    p=initialize();seed_all(928);network,mods=build_network(epoch);dataset,_,_=build_loader('val')
    prefix=f'encoder.layers.{len(network.encoder.layers)-1}.cls_branch.'
    parameters=[v for name,v in network.named_parameters() if name.startswith(prefix)]
    assert parameters
    for v in parameters:v.requires_grad_(True)
    originals=[v.detach().clone() for v in parameters];scale=sum(float(v.double().square().sum()) for v in originals)**.5
    dest=RUN/f'epoch{epoch}'/'parameters';dest.mkdir(exist_ok=True);pairs=p['parameter_pairs'][:limit] if limit else p['parameter_pairs'];files=[]
    def load(meta):return to_cuda(custom_collate_fn_temporal([dataset[meta['index']]]))
    for index,pair in enumerate(pairs):
        path=dest/f'pair_{index:02d}.json'
        if path.exists():files.append(str(path.relative_to(ROOT)));continue
        source=load(pair['source']);heldout=load(pair['heldout'])
        output=network_output(network,source);frame=Frame(network,output,source,mods)
        # Link objectives to the actual final classification branch, rather than the detached probe leaf.
        raw=output['opus_pred_logits'][-1];objectives=frame.objectives(raw);gradients={}
        for name in PARAM_BRANCHES:
            gradients[name]=[g.detach() for g in torch.autograd.grad(objectives[name],parameters,retain_graph=True)]
        with torch.no_grad():other_output=network_output(network,heldout);other=Frame(network,other_output,heldout,mods)
        bases=[frame,other];baselines=[f.measure(f.base,f.sem) for f in bases]
        del raw,objectives,output
        rows=[];frozen_error=0.
        with torch.no_grad():
            for branch in PARAM_BRANCHES:
                grad=gradients[branch];norm=sum(float(g.double().square().sum()) for g in grad)**.5
                for radius in PARAM_RADII:
                    lr=radius*scale/max(norm,1e-30)
                    try:
                        for v,b,g in zip(parameters,originals,grad):v.copy_(b-lr*g)
                        for role,data,original in [('source',source,frame),('heldout',heldout,other)]:
                            new=network_output(network,data)
                            new_args=tuple(t.detach() for t in network.head.prepare_gaussian_args(new['gaussian']))
                            error=max(float((new_args[k]-original.args[k]).abs().max()) for k in [0,1,3,4])
                            frozen_error=max(frozen_error,error);assert error<1e-6,error
                            new_sem=new_args[2][:,:-1,:17]
                            actual=new['pred_occ'][-1][:,:,original.mask]
                            for pathway,sem in factorial_semantics(original.sem,new_sem).items():
                                if pathway=='sham':continue
                                scores=original.render(sem)
                                if pathway=='full':assert torch.allclose(scores,actual,atol=2e-4,rtol=2e-4)
                                rows.append(dict(branch=branch,radius=radius,pathway=pathway,role=role,
                                    gradient_norm=norm,effective_sgd_lr=lr,outcome=original.measure(scores,sem)))
                            del new,new_args
                    finally:
                        for v,b in zip(parameters,originals):v.copy_(b)
        result=dict(status='complete',epoch=epoch,pair=pair,baselines=dict(zip(['source','heldout'],baselines)),
            interventions=rows,frozen_geometry_max_error=frozen_error,parameter_count=sum(v.numel() for v in parameters),
            protocol_sha256=file_sha256(RUN/'protocol.json'),source_sha256=file_sha256(__file__))
        write_json(path,result);files.append(str(path.relative_to(ROOT)))
        del frame,other,bases,source,heldout,other_output,gradients
        print('actual parameter paths',epoch,index+1,'/',len(pairs),flush=True)
    if not limit:write_json(dest/'manifest.json',dict(status='complete',files=files,source_sha256=file_sha256(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','local','census','parameters']);parser.add_argument('--epoch',type=int,default=5);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.stage=='initialize':initialize()
    elif args.stage=='local':local(args.epoch,args.limit)
    elif args.stage=='parameters':parameter_probes(args.epoch,args.limit)
    else:census()
