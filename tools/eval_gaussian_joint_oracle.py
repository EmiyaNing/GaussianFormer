"""Joint and ordered geometry/semantic output fitting to each frame's GT.

G follows historical CSRO: center, scale, rotation, opacity. E is 17
nonnegative semantic values. No network or empty parameter is updated.
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

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader,Subset
from tools.stat_boundary_gaussians import read,write,sha,forward,to_cuda
from tools.boundary_gaussian_stats.data import EvaluationFrames,FeatureCache
from tools.eval_gaussian_sizes import paths,build,CLASS_NAMES,CLASS_COLUMNS
from tools.eval_gaussian_center_oracle import objective,loss_terms,TRANSITIONS
from tools.eval_gaussian_attribute_oracle import (
    Frame,ARMS as ATTRIBUTE_ARMS,ATTR_STATS,LR as GEOMETRY_LR,
    numerical_audit as geometry_numerical_audit)
from tools.eval_gaussian_semantic_oracle import (
    SemanticFrame,SEMANTIC_STATS,LR as SEMANTIC_LR,
    numerical_audit as semantic_numerical_audit)
from tools.semantic_field_support import extension

RUN=ROOT/'work_dirs/gaussian_joint_oracle_full_val_v1'
HISTORY=ROOT/'work_dirs/gaussian_sizes_full_val'
G_HISTORY=ROOT/'work_dirs/gaussian_attribute_oracle_full_val_v1'
E_HISTORY=ROOT/'work_dirs/gaussian_semantic_oracle_full_val_v1'
ARMS=('J','E_then_G','G_then_E')
STEPS=(0,5,10,20,25,30,40)
LR={**GEOMETRY_LR,'E':SEMANTIC_LR}
COUPLING_STATS=('GS','zero_opacity','zero_semantic_mass','both_zero',
    'zero_opacity_positive_semantics','positive_opacity_zero_semantics')


def initialize():
    config,checkpoint=paths('tiny');cfg=Config.fromfile(config)
    prior=read(E_HISTORY/'protocol.json')
    for history in (G_HISTORY,E_HISTORY):assert read(history/'delivery_audit.json')['status']=='passed'
    assert sha(checkpoint)==prior['checkpoint_sha256']
    for p,digest in prior['source_sha256'].items():assert sha(ROOT/p)==digest,p
    files=list(dict.fromkeys([Path(__file__),*[ROOT/p for p in prior['source_sha256']]]))
    protocol=dict(version=1,model='tiny',frames=6019,scenes=150,queries=600,children=128,GS=76800,
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
        config=str(config.relative_to(ROOT)),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256=sha(cfg.val_dataset_config.imageset),seed=928,
        arms=list(ARMS),steps=list(STEPS),
        schedules=dict(J='CSROE simultaneously for 40 updates; report both step20 and step40',
            E_then_G='E for 20 updates, freeze E, CSRO for 20 updates',
            G_then_E='CSRO for 20 updates, freeze CSRO, E for 20 updates'),
        G_definition='Historical CSRO; opacity is evidence strength and is included for direct historical comparability, not called pure geometry',
        optimizer=dict(type='Adam',learning_rates=LR,betas=[.9,.999],eps=1e-8,weight_decay=0,
            reset='Independent state per frame and arm; new optimizer for second block; J retains state through40'),
        constraints=read(G_HISTORY/'protocol.json')['constraints'],
        semantic_constraint='Direct nonnegative output values, both class proportions and total mass free; no upper bound',
        geometry_reference='Displacement <=0.4m and rotation <=45deg relative to original prediction, never rebased at phase transition',
        fixed='Network, empty Gaussian/scalar/bias, GS count, query ownership; inactive block fixed bitwise within each phase',
        loss='Identical final-stage Occupancy CE+Lovasz, CE includes empty, Lovasz ignores17; no Set/coverage objectives',
        domain='All camera-valid GT labels0..17 in all150 validation scenes',
        budget='J20 and ordered40 each update each block20 times; J40 and ordered40 use40 total backward steps but J40 updates each block40 times',
        initialization='Both first-stage prefixes recomputed from original outputs and checked bitwise against historical E20/G20 statistics and losses per frame',
        interpretation='Per-frame GT-assisted output fitting, not network training or ordinary validation; no deployable checkpoint',
        coverage='Native support positive opacity*kernel; same-class uses current semantics, zero semantic mass has no class',
        class_columns=CLASS_COLUMNS,transitions=list(TRANSITIONS),
        attribute_statistics=list(ATTR_STATS),semantic_statistics=list(SEMANTIC_STATS),
        coupling_statistics=list(COUPLING_STATS),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    if (RUN/'protocol.json').exists():assert read(RUN/'protocol.json')==protocol,'Protocol changed; use a new run directory'
    else:
        write(RUN/'protocol.json',protocol);cfg.dump(RUN/'resolved_config.py')
        for p in files:
            dest=RUN/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
    return cfg,protocol


def transition_counts(target,old,new):
    fg,oldfg,newfg=target<17,old<17,new<17
    oldok,newok=old==target,new==target
    masks=(fg&~oldfg&newfg,fg&oldfg&~newfg,~fg&oldfg&~newfg,~fg&~oldfg&newfg,
        ~oldok&newok,oldok&~newok,fg&oldfg&newfg&~oldok&newok,fg&oldfg&newfg&oldok&~newok)
    return torch.stack([x.sum() for x in masks])


class CoupledFrame(SemanticFrame):
    def all_parameters(self):
        p=Frame.parameters(self,'CSRO');p['E']=torch.nn.Parameter(self.base.semantics.clone())
        return p

    def gaussian_all(self,params):
        return Frame.gaussian(self,params)._replace(semantics=params['E'])

    def render_all(self,params):
        g=self.gaussian_all(params)
        inverse=self.head.prepare_gaussian_args(g)[-1][:,:-1]
        semantic=torch.cat((g.semantics,torch.zeros_like(g.semantics[...,:1])),-1)
        z=self.agg(self.xyz[None],g.means,g.opacities.reshape(1,-1),semantic,g.scales,inverse).T
        return torch.cat((z[:17],self.empty_logits[None]),0),g,inverse

    @torch.no_grad()
    def project_all(self,params,active):
        Frame.project(self,{key:params[key] for key in active if key in 'CSRO'})
        if 'E' in active:params['E'].clamp_(min=0)

    @torch.no_grad()
    def coverage_all(self,g,inverse):
        self.classes=g.semantics[0].argmax(-1).int().contiguous()
        self.classes[g.semantics[0].sum(-1)==0]=-1
        return Frame.coverage(self,g,inverse)

    @torch.no_grad()
    def observe_all(self,params,z,g,inverse,pred20=None):
        pred=z.argmax(0)
        cm=torch.bincount(self.target*18+pred,minlength=324).reshape(18,18)
        a=g.opacities.flatten()==0;e=g.semantics.sum(-1).flatten()==0
        cs=torch.stack((a.new_tensor(a.numel(),dtype=torch.int64),a.sum(),e.sum(),(a&e).sum(),
            (a&~e).sum(),(~a&e).sum()))
        return dict(confusion=cm,class_coverage=self.coverage_all(g,inverse),
            transitions=transition_counts(self.target,self.base_pred,pred),
            post20_transitions=(transition_counts(self.target,pred20,pred) if pred20 is not None
                else torch.zeros(8,device=z.device,dtype=torch.int64)),
            attribute_stats=Frame.statistics(self,params,g,inverse),
            semantic_statistics=SemanticFrame.semantic_statistics(self,g.semantics),
            coupling_statistics=cs)


def optimize(frame,arm,occ):
    params=frame.all_parameters();rows={};trajectory=[];max_gradient=[];pred20=None
    phases=(('CSROE',40),) if arm=='J' else ((('E',20),('CSRO',20)) if arm=='E_then_G' else (('CSRO',20),('E',20)))
    total_steps=0;last_logits=None
    for phase_index,(active,updates) in enumerate(phases):
        for key,value in params.items():value.requires_grad_(key in active);value.grad=None
        optimizer=torch.optim.Adam([dict(params=[params[key]],lr=LR[key]) for key in active],
            betas=(.9,.999),eps=1e-8,weight_decay=0)
        frozen={key:value.detach().clone() for key,value in params.items() if key not in active}
        for local_step in range(updates+1):
            step=total_steps+local_step
            optimizer.zero_grad(set_to_none=True)
            z,g,inverse=frame.render_all(params)
            if step==0:torch.testing.assert_close(z,frame.base_logits,rtol=0,atol=0)
            if phase_index>0 and local_step==0:
                torch.testing.assert_close(z,last_logits,rtol=0,atol=0)
            values=loss_terms(z,frame.target,occ)
            assert bool(torch.isfinite(values).all()),(arm,step,'loss')
            if not (phase_index>0 and local_step==0):
                trajectory.append(values.detach())
                if step==20:pred20=z.detach().argmax(0).clone()
                if step in STEPS:
                    observed=frame.observe_all(params,z.detach(),g,inverse,pred20)
                    for key,value in observed.items():rows.setdefault(key,[]).append(value.detach())
            if local_step==updates:
                last_logits=z.detach().clone();break
            values[0].backward()
            grad=torch.stack([params[key].grad.abs().max() if params[key].grad is not None
                else z.new_tensor(float('nan')) for key in active])
            assert bool(torch.isfinite(grad).all()),(arm,step,'gradient')
            max_gradient.append(grad.max().detach());optimizer.step();frame.project_all(params,active)
        for key,value in frozen.items():assert torch.equal(params[key],value),(arm,phase_index,key)
        total_steps+=updates
    assert total_steps==40 and len(trajectory)==41
    assert all(p.grad is None for p in frame.net.parameters())
    with torch.no_grad():
        assert float(params['C'].norm(dim=-1).max())<=.40001
        assert float(params['R'].norm(dim=-1).max())<=np.pi/4+1e-5
        assert float(params['S'].min())>=.08-1e-6 and float(params['S'].max())<=.4+1e-6
        assert float(params['O'].min())>=0 and float(params['O'].max())<=1
        assert bool(torch.isfinite(params['E']).all()) and float(params['E'].min())>=0
        assert bool((params['O'][frame.zero_opacity]==0).all())
    result={key:torch.stack(value).cpu().numpy() for key,value in rows.items()}
    result['loss_trajectory']=torch.stack(trajectory).cpu().numpy()
    result['max_gradient']=torch.stack(max_gradient).max().cpu().numpy()
    if arm=='G_then_E':
        # Geometry and opacity are frozen throughout the second phase.
        for j in (4,5,6):
            np.testing.assert_array_equal(result['class_coverage'][j][:,[0,1,2,3,7]],
                                           result['class_coverage'][3][:,[0,1,2,3,7]])
            np.testing.assert_array_equal(result['attribute_stats'][j],result['attribute_stats'][3])
    if arm=='E_then_G':
        for j in (4,5,6):
            np.testing.assert_array_equal(result['semantic_statistics'][j],result['semantic_statistics'][3])
    return result


def process_frame(net,out,batch,occ,protocol):
    frame=CoupledFrame(net,out,batch,protocol);rows={}
    for arm in ARMS:
        result=optimize(frame,arm,occ)
        for key,value in result.items():rows.setdefault(key,[]).append(value)
    return {key:np.stack(values) for key,values in rows.items()}


def check_history(result,old_e,old_g,index):
    for arm in range(3):
        for key in ('confusion','class_coverage'):
            np.testing.assert_array_equal(result[key][arm,0],old_e[key][index,0])
    gi=ATTRIBUTE_ARMS.index('CSRO')
    for key in ('confusion','class_coverage','transitions'):
        np.testing.assert_array_equal(result[key][1,:4],old_e[key][index],err_msg='E prefix '+key)
        np.testing.assert_array_equal(result[key][2,:4],old_g[key][index,gi],err_msg='G prefix '+key)
    np.testing.assert_array_equal(result['loss_trajectory'][1,:21],old_e['loss_trajectory'][index])
    np.testing.assert_array_equal(result['loss_trajectory'][2,:21],old_g['loss_trajectory'][index,gi])
    np.testing.assert_array_equal(result['semantic_statistics'][1,:4],old_e['semantic_statistics'][index])
    np.testing.assert_array_equal(result['attribute_stats'][2,:4],old_g['attribute_stats'][index,gi])


def pilot():
    from dataset.utils import custom_collate_fn_temporal
    cfg,protocol=initialize();net=build(cfg,'tiny');net.requires_grad_(False);occ=objective(cfg)
    numerical=dict(G=geometry_numerical_audit(net,protocol),E=semantic_numerical_audit(net,protocol))
    dataset=EvaluationFrames(cfg,'val');cache=FeatureCache(net,dataset)
    try:
        batch=to_cuda(custom_collate_fn_temporal([dataset[0]]))
        with torch.no_grad():out=forward(net,batch)
        coupled=CoupledFrame(net,out,batch,protocol);p=coupled.all_parameters()
        z,_,_=coupled.render_all(p)
        combined=torch.autograd.grad(loss_terms(z,coupled.target,occ)[0],tuple(p.values()))
        gf=Frame(net,out,batch,protocol);gp=gf.parameters('CSRO')
        gz,_,_=gf.render(gp)
        ggrads=torch.autograd.grad(loss_terms(gz,gf.target,occ)[0],tuple(gp.values()))
        ef=SemanticFrame(net,out,batch,protocol)
        ep=ef.base.semantics.clone().requires_grad_(True)
        egrad,=torch.autograd.grad(loss_terms(ef.render_semantic(ep),ef.target,occ)[0],ep)
        for actual,expected in zip(combined,(*ggrads,egrad)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        del coupled,p,z,combined,gf,gp,gz,ggrads,ef,ep,egrad
        start=time.monotonic();result=process_frame(net,out,batch,occ,protocol)
        scene=dataset.keyframes[0][0]
        with np.load(E_HISTORY/'scenes'/f'{scene}.npz') as e,np.load(G_HISTORY/'scenes'/f'{scene}.npz') as g:
            check_history(result,e,g,0)
        np.savez_compressed(RUN/'pilot_arrays.npz',**result)
        info=dict(status='passed',numerical=numerical,all_three_zero_logits_exact=True,
            joint_initial_five_gradients_exactly_match_separate_paths=True,
            E_prefix_bitwise_exact=True,G_prefix_bitwise_exact=True,
            phase_boundary_logits_exact=True,inactive_blocks_bitwise_frozen=True,
            constraints_relative_to_original=True,no_network_gradients=True,
            seconds_all_three_arms=time.monotonic()-start)
        write(RUN/'pilot.json',info);print(json.dumps(info),flush=True)
    finally:
        cache.close();net.flush_memory_phase();del net,cache,dataset
        gc.collect();torch.cuda.empty_cache()


def evaluate(workers):
    from dataset.utils import custom_collate_fn_temporal
    cfg,protocol=initialize();assert read(RUN/'pilot.json')['status']=='passed'
    if (RUN/'complete.json').exists():return
    net=build(cfg,'tiny');net.requires_grad_(False);occ=objective(cfg)
    dataset=EvaluationFrames(cfg,'val');assert len(dataset)==6019
    scenes={}
    for i,(scene,_) in enumerate(dataset.keyframes):scenes.setdefault(scene,[]).append(i)
    assert len(scenes)==150;pending=[]
    for scene,ids in scenes.items():
        p=RUN/'scenes'/f'{scene}.json'
        if p.exists():
            old=read(p);assert old['indices']==ids and old['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert p.with_suffix('.npz').exists()
        else:pending+=ids
    loader=DataLoader(Subset(dataset,pending),batch_size=1,num_workers=workers,shuffle=False,
        collate_fn=custom_collate_fn_temporal,pin_memory=True,persistent_workers=workers>0,
        **(dict(prefetch_factor=2) if workers else {}))
    cache=FeatureCache(net,dataset);start=time.monotonic();done=0;active=None
    try:
        for index,batch in zip(pending,loader):
            scene,f=dataset.keyframes[index]
            if active is None:
                active=scene;arrays={};indices=[];tokens=[]
                old_e=np.load(E_HISTORY/'scenes'/f'{scene}.npz')
                old_g=np.load(G_HISTORY/'scenes'/f'{scene}.npz')
            assert active==scene;batch=to_cuda(batch)
            with torch.no_grad():out=forward(net,batch)
            result=process_frame(net,out,batch,occ,protocol)
            check_history(result,old_e,old_g,len(indices))
            for key,value in result.items():arrays.setdefault(key,[]).append(value)
            indices.append(index);tokens.append(dataset.base.scene_infos[scene][f]['token']);done+=1
            if index==scenes[scene][-1]:
                old_e.close();old_g.close();p=RUN/'scenes'/f'{scene}.npz';p.parent.mkdir(parents=True,exist_ok=True)
                with p.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,**{key:np.stack(value) for key,value in arrays.items()})
                p.with_suffix('.tmp').replace(p)
                write(p.with_suffix('.json'),dict(status='complete',scene=scene,indices=indices,tokens=tokens,
                    frames=len(indices),checkpoint_sha256=protocol['checkpoint_sha256']))
                active=None
            if done%10==0 or active is None:
                status=dict(status='running',frames=6019-len(pending)+done,expected_frames=6019,
                    scenes=len(list((RUN/'scenes').glob('*.json'))),expected_scenes=150,arms=3,
                    steps_per_arm=40,elapsed_seconds=time.monotonic()-start,
                    diagnostic_frames_per_second=done/(time.monotonic()-start))
                write(RUN/'progress.json',status);print(json.dumps(status),flush=True)
            del out,batch,result
        files=list((RUN/'scenes').glob('*.json'));ids=[i for p in files for i in read(p)['indices']]
        assert len(files)==150 and sorted(ids)==list(range(6019))
        for p,digest in protocol['source_sha256'].items():assert sha(ROOT/p)==digest,p
        assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
        info=dict(status='complete',frames=6019,scenes=150,arms=3,steps_per_arm=40,
            seconds=time.monotonic()-start,checkpoint_sha256=protocol['checkpoint_sha256'])
        write(RUN/'complete.json',info);write(RUN/'progress.json',info)
    finally:
        cache.close();net.flush_memory_phase();del net,cache,dataset,loader
        gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('pilot','run'),default='run')
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(RUN/'status.json',dict(status='running',action=args.action,pid=os.getpid()))
        try:
            if args.action=='pilot':pilot()
            else:evaluate(args.workers)
            write(RUN/'status.json',dict(status=args.action+'_complete'))
        except BaseException as exc:
            write(RUN/'status.json',dict(status='failed',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__':main()
