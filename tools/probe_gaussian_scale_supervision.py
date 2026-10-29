"""Full train/val census of Gaussian supervision, saturation and scale controls.

No training, scene selection or frame subsampling. All 34,149 keyframes.
python3.10 tools/probe_gaussian_scale_supervision.py --action pilot
python3.10 tools/probe_gaussian_scale_supervision.py --action run
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from tools.stat_boundary_gaussians import CONFIG,CHECKPOINT,build,forward,to_cuda,read,write,sha
from tools.boundary_gaussian_stats.data import EvaluationFrames,FeatureCache
from tools.stat_gaussian_support_regions import gaussian_types
from tools.gaussian_scale_probe import extension,SOURCE_NAMES,GRAD_NAMES
from model.utils.utils import get_rotation_matrix

RUN=ROOT/'work_dirs/gaussian_scale_supervision_full'
GROUPS=('foreground_mixed','foreground_empty_interface','pure_foreground',
        'pure_empty_no_fp_owner','pure_empty_fp_owner','unsupported','all',
        'pure_empty_fp_owner_visible_empty_center',
        'pure_empty_fp_owner_outside_camera_fg_center',
        'pure_empty_fp_owner_outside_camera_empty_center',
        'pure_empty_fp_owner_all_support_empty')
SUBGROUP_FLAGS=('center_visible_empty','center_outside_camera_foreground',
                'center_outside_camera_empty','all_support_empty')
CONDITIONS=('baseline','scale_cap30','opacity_mass_matched','scale_cap30_mass_restored')
FIELDS=('opacity','semantic_max','semantic_sum','scale_min','scale_max','aspect_ratio',
        'camera_mass_fraction','camera_kernel_mass','camera_fp_kernel_mass','fp_owner_voxels',
        'nearest_gt_distance','scale_jacobian_max_axis_relative',
        'ce_empty_opacity','ce_empty_raw_opacity','ce_empty_scale_x','ce_empty_scale_y','ce_empty_scale_z',
        'ce_empty_raw_scale_x','ce_empty_raw_scale_y','ce_empty_raw_scale_z',
        'ce_empty_target','ce_foreground_target','lovasz_target','set_target','joint_target',
        'ce_empty_log_amplitude','ce_foreground_log_amplitude','lovasz_log_amplitude','set_log_amplitude','joint_log_amplitude',
        'ce_foreground_opacity','lovasz_opacity',
        'ce_foreground_scale_x','ce_foreground_scale_y','ce_foreground_scale_z',
        'lovasz_scale_x','lovasz_scale_y','lovasz_scale_z',
        'occupancy_scale_x','occupancy_scale_y','occupancy_scale_z',
        'raw_occupancy_scale_x','raw_occupancy_scale_y','raw_occupancy_scale_z')
FLAGS=('camera_fraction_lt_01','camera_fraction_ge_05','ce_empty_target_positive',
       'ce_empty_amplitude_positive','ce_empty_scale_any_positive','ce_empty_scale_any_negative',
       'lovasz_amplitude_nonzero','set_target_negative','ce_set_target_opposed','joint_target_negative',
       'scale_max_ge_039','all_axes_ge_039','scale_jacobian_lt_01_peak','eligible_intervention',
       'scale_max_sigmoid_gradient_zero','all_scale_sigmoid_gradients_zero',
       'ce_scale_wants_shrink_but_raw_zero','joint_target_negative_strong_camera','strong_camera_and_ce_positive',
       'center_camera','center_foreground','center_empty',*SUBGROUP_FLAGS,'all_support_has_foreground')
LOG_EDGES=np.arange(-30.,10.01,.5)
COVERAGE_EDGES=np.array([0,.001,.01,.05,.1,.25,.5,.75,.9,.99,1.000001])
SCALE_EDGES=np.array([.08,.10,.15,.20,.25,.30,.35,.38,.39,.395,.399,.3999,.400001])
MARGIN_EDGES=np.linspace(-5.,5.,201)


def modules(cfg):
    from loss import OPENOCC_LOSS
    occ=next(deepcopy(x) for x in cfg.loss.loss_cfgs if x.type=='OccupancyLoss')
    setting=next(deepcopy(x) for x in cfg.loss.loss_cfgs if x.type=='OPUSSetLoss')
    setting['stage_weights']=[setting['stage_weights'][-1]]
    return OPENOCC_LOSS.build(occ).cuda(),OPENOCC_LOSS.build(setting).cuda()


def capture_attributes(network):
    """Read exact FP32 pre-sigmoid values; no model code/output is modified."""
    branch=network.encoder.attribute_branches['5']
    def hook(module,inputs,output):
        network._scale_probe_raw=output.detach().reshape(output.shape[0],-1,8)
    return branch.net[-1].register_forward_hook(hook)


def layout(head,gaussian):
    args=head.prepare_gaussian_args(gaussian)
    means,alpha,semantic,scales,inv=[v[0,:-1].contiguous() for v in args]
    agg=head.aggregator
    center=((means-agg.pc_min)/agg.grid_size).to(torch.int32).contiguous()
    radii=torch.ceil(scales.max(-1).values*agg.scale_multiplier/agg.grid_size).to(torch.int32).contiguous()
    inverse=inv.flatten(1)[:,[0,4,8,1,5,2]].contiguous()
    return args,(means,scales,inverse,alpha.flatten().contiguous(),semantic[:,:17].contiguous(),center,radii)


def render(head,gaussian,xyz):
    means,alpha,sem,scales,inv=head.prepare_gaussian_args(gaussian)
    z=head.aggregator(xyz.unsqueeze(0).contiguous(),means,alpha.reshape(1,-1),sem,scales,inv).T
    return torch.cat((z[:17],z[17:]-head.empty_bias.detach()),0)


def losses_and_sources(out,batch,occ,setmod):
    from loss.utils.lovasz_softmax import lovasz_softmax
    from loss.opus_set_loss import _nearest_indices
    y=batch['occ_label'].flatten().long();camera=batch['occ_cam_mask'].flatten().bool()
    mask=camera&(y>=0)&(y<18);target=y[mask]
    z=out['pred_occ'][-1][0,:,mask].detach().requires_grad_(True)
    weights=occ.class_weights.to(z)
    with torch.enable_grad():
        nll=F.cross_entropy(z.unsqueeze(0),target.unsqueeze(0),weight=weights,reduction='none')[0]
        ce=nll.sum()/weights[target].sum()*occ.loss_voxel_ce_weight
        lov=lovasz_softmax(z.softmax(0).T,target,ignore=occ.lovasz_ignore)*occ.loss_voxel_lovasz_weight
        gce=torch.autograd.grad(ce,z,retain_graph=True)[0].detach()
        glov=torch.autograd.grad(lov,z)[0].detach()
    full_ce=torch.zeros((len(y),17),device=y.device);full_lov=torch.zeros_like(full_ce)
    full_ce[mask]=gce[:17].T;full_lov[mask]=glov[:17].T
    points=out['opus_pred_points'][-1][0].detach();valid=out['opus_point_valid_masks'][-1][0].bool()
    foreground=(y>=0)&(y<17);gt=batch['occ_xyz'].reshape(-1,3)[foreground]
    assigned=torch.full((len(points),),-1,device=y.device,dtype=torch.int32)
    distance=torch.zeros(len(points),device=y.device)
    nearest=_nearest_indices(points[valid],gt,setmod.chunk_size)
    labels=y[foreground][nearest];assigned[valid]=labels.int()
    distance[valid]=(points[valid]-gt[nearest]).norm(dim=1)
    raw=out['opus_pred_logits'][-1].detach()[0].clone().requires_grad_(True)
    with torch.enable_grad():
        sample_weights=setmod.class_weights.to(raw)[labels]*setmod._distance_weight(gt[nearest])
        set_cls=setmod._sigmoid_focal(raw[valid],labels,sample_weights)*setmod.lambda_cls*setmod.stage_weights[-1]
        gset=torch.autograd.grad(set_cls,raw)[0].detach()
    index=assigned.long().clamp_min(0)
    target_sigmoid=raw.detach().sigmoid().gather(1,index[:,None]).flatten().contiguous()
    set_target=gset.gather(1,index[:,None]).flatten();set_target[~valid]=0
    amplitude_ratio=(F.softplus(raw.detach().double())/raw.detach().double().sigmoid().clamp_min(1e-300)).float()
    set_amplitude=(gset*amplitude_ratio).sum(1)
    return dict(mask=mask,y=y,target=target,z=z.detach(),ce=full_ce.contiguous(),lov=full_lov.contiguous(),
        assigned=assigned.contiguous(),target_sigmoid=target_sigmoid,set_target=set_target,
        set_amplitude=set_amplitude,distance=distance,
        losses=dict(ce=float(ce.detach()),lovasz=float(lov.detach()),set_cls=float(set_cls.detach())))


def probe(head,out,batch,source):
    _,v=layout(head,out['gaussian']);means,scales,inverse,alpha,sem,center,radii=v
    rotation=get_rotation_matrix(out['gaussian'].rotations)[0].contiguous()
    xyz=out['sampled_xyz'][0].contiguous();camera=batch['occ_cam_mask'].flatten().bool().contiguous()
    y=source['y'].int().contiguous();pred=out['final_occ'].flatten().int().contiguous()
    winner=out['pred_occ'][-1][0,:17].argmax(0).int().contiguous()
    shape=(head.aggregator.H,head.aggregator.W,head.aggregator.D)
    result=extension().probe(means,scales,inverse,rotation,alpha,sem,center,radii,xyz,y,camera,
        pred,winner,source['assigned'],source['target_sigmoid'],source['ce'],source['lov'],*shape)
    return result,v,shape


def counterfactuals(head,out,batch,result,v,shape):
    g=out['gaussian'];scales=g.scales[0];saturated=scales.max(1).values>=.39
    proposed=torch.where(saturated[:,None],scales.clamp(max=.30),scales)
    candidate=g._replace(scales=proposed.unsqueeze(0))
    _,new=layout(head,candidate)
    xyz=out['sampled_xyz'][0].contiguous();camera=batch['occ_cam_mask'].flatten().bool().contiguous()
    newmass=extension().mass(new[0],new[2],new[3],new[5],new[6],xyz,camera,*shape)
    oldmass=result[1][:,6]
    eligible=saturated&(oldmass>1e-30)&(newmass>1e-30)
    ratio=torch.ones_like(oldmass);ratio[eligible]=(newmass[eligible]/oldmass[eligible]).clamp(max=1)
    sigma=torch.where(eligible[:,None],proposed,scales)
    scaled=g._replace(scales=sigma.unsqueeze(0))
    attenuated=g._replace(opacities=g.opacities*ratio[None,:,None])
    restored=g._replace(scales=sigma.unsqueeze(0),semantics=g.semantics/ratio[None,:,None])
    query=xyz[camera]
    scores=[out['pred_occ'][-1][0,:,camera].detach()]
    for state in (scaled,attenuated,restored):
        value=render(head,state,query)
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite counterfactual')
        scores.append(value.detach())
    mass=torch.tensor([float(z[:17].double().sum()) for z in scores],device=xyz.device,dtype=torch.float64)
    match=float(abs(mass[1]-mass[2])/mass[1].clamp_min(1e-30))
    restore=float(abs(mass[0]-mass[3])/mass[0].clamp_min(1e-30))
    assert match<5e-4 and restore<5e-4,(match,restore)
    return scores,eligible,dict(mass_match_relative_error=match,mass_restore_relative_error=restore,
        eligible=int(eligible.sum()),saturated=int(saturated.sum()),
        unsupported_restore=int((saturated&~eligible).sum()),maximum_restore_factor=float(ratio.reciprocal().max()))


def center_support_flags(center,shape,labels,camera,bits):
    """Native center cell, not nearest voxel; all-space support is separately tested."""
    flat=(center[:,0].long()*shape[1]+center[:,1])*shape[2]+center[:,2]
    visible=camera[flat];foreground=labels[flat]<17;empty=labels[flat]==17
    return (visible,foreground,empty,visible&empty,~visible&foreground,~visible&empty,
            bits[0]==(1<<17),(bits[0]&((1<<17)-1))!=0)


def features(out,source,result,eligible,batch,v,shape):
    bits,stats,packed=result;g=out['gaussian'];sigma=g.scales[0];alpha=g.opacities[0,:,0]
    sem=g.semantics[0];gradient=stats[:,10:].reshape(-1,3,9)
    owner=packed[packed>0];ids=(0xffffffff-(owner&0xffffffff)).long()
    fp_owner=torch.bincount(ids,minlength=len(sigma))
    kinds=gaussian_types(bits[1]);group=kinds.clone();group[kinds==4]=5
    group[(kinds==3)&(fp_owner>0)]=4
    maximum,axis=sigma.max(1);minimum=sigma.min(1).values
    jac=out['_scale_probe_jacobian']
    max_jac=jac.gather(1,axis[:,None]).flatten()/.08
    ce=gradient[:,0];fg=gradient[:,1];lov=gradient[:,2]
    target=ce[:,7]+fg[:,7]+lov[:,7]+source['set_target']
    amplitude=ce[:,8]+fg[:,8]+lov[:,8]+source['set_amplitude']
    coverage=stats[:,6]/stats[:,5].clamp_min(1e-30)
    metrics=torch.stack((alpha,sem.max(1).values,sem.sum(1),minimum,maximum,maximum/minimum,
        coverage,stats[:,6],stats[:,9],fp_owner.float(),source['distance'],max_jac,
        ce[:,0],ce[:,0]*alpha*(1-alpha),ce[:,1],ce[:,2],ce[:,3],
        ce[:,1]*jac[:,0],ce[:,2]*jac[:,1],ce[:,3]*jac[:,2],
        ce[:,7],fg[:,7],lov[:,7],source['set_target'],target,
        ce[:,8],fg[:,8],lov[:,8],source['set_amplitude'],amplitude,
        fg[:,0],lov[:,0],fg[:,1],fg[:,2],fg[:,3],lov[:,1],lov[:,2],lov[:,3],
        ce[:,1]+fg[:,1]+lov[:,1],ce[:,2]+fg[:,2]+lov[:,2],ce[:,3]+fg[:,3]+lov[:,3],
        (ce[:,1]+fg[:,1]+lov[:,1])*jac[:,0],(ce[:,2]+fg[:,2]+lov[:,2])*jac[:,1],
        (ce[:,3]+fg[:,3]+lov[:,3])*jac[:,2]),1)
    flags=torch.stack((coverage<.1,coverage>=.5,ce[:,7]>0,ce[:,8]>0,(ce[:,1:4]>0).any(1),
        (ce[:,1:4]<0).any(1),lov[:,8]!=0,source['set_target']<0,
        (ce[:,7]>0)&(source['set_target']<0),target<0,maximum>=.39,(sigma>=.39).all(1),max_jac<.1,eligible,
        jac.gather(1,axis[:,None]).flatten()==0,(jac==0).all(1),
        (ce[:,1:4]>0).any(1)&((ce[:,1:4]*jac)==0).all(1),
        (target<0)&(coverage>=.5),(coverage>=.5)&(ce[:,7]>0),
        *center_support_flags(v[5],shape,source['y'],batch['occ_cam_mask'].flatten().bool(),bits)),1)
    # A visible foreground center must occur in the native candidate box.
    assert not ((group==4)&flags[:,FLAGS.index('center_camera')]&flags[:,FLAGS.index('center_foreground')]).any()
    assert metrics.shape[1]==len(FIELDS) and flags.shape[1]==len(FLAGS)
    return group,metrics,flags,sigma


def accumulator():
    options=dict(device='cuda');nf=len(FIELDS);ng=len(GROUPS)
    return dict(counts=torch.zeros(ng,dtype=torch.int64,**options),
        sums=torch.zeros((ng,nf),dtype=torch.float64,**options),
        abs_sums=torch.zeros((ng,nf),dtype=torch.float64,**options),
        positive=torch.zeros((ng,nf),dtype=torch.int64,**options),negative=torch.zeros((ng,nf),dtype=torch.int64,**options),
        hist=torch.zeros((ng,nf,len(LOG_EDGES)+2),dtype=torch.int64,**options),
        coverage_hist=torch.zeros((ng,len(COVERAGE_EDGES)+1),dtype=torch.int64,**options),
        axis_hist=torch.zeros((ng,3,len(SCALE_EDGES)+1),dtype=torch.int64,**options),
        saturation_hist=torch.zeros((ng,4),dtype=torch.int64,**options),
        flags=torch.zeros((ng,len(FLAGS)),dtype=torch.int64,**options),
        confusion=torch.zeros((4,18,18),dtype=torch.int64,**options),
        margin_hist=torch.zeros((4,18,17,len(MARGIN_EDGES)+1),dtype=torch.int64,**options))


def add_grouped(acc,group,metrics,flags,sigma):
    nf=len(FIELDS);ng=len(GROUPS);device=metrics.device
    acc['counts'][6]+=len(group)
    # Six exclusive groups plus four overlapping diagnostic subsets of group 4.
    extra=[torch.where((group==4)&flags[:,FLAGS.index(name)])[0] for name in SUBGROUP_FLAGS]
    indices=torch.cat((torch.arange(len(group),device=device),*extra))
    group=torch.cat((group,*[torch.full_like(ids,7+j) for j,ids in enumerate(extra)]))
    metrics=metrics[indices];flags=flags[indices];sigma=sigma[indices]
    acc['counts']+=torch.bincount(group,minlength=ng)
    acc['sums'].index_add_(0,group,metrics.double());acc['abs_sums'].index_add_(0,group,metrics.abs().double())
    acc['positive'].index_add_(0,group,(metrics>0).long());acc['negative'].index_add_(0,group,(metrics<0).long())
    acc['flags'].index_add_(0,group,flags.long())
    log_edges=torch.tensor(LOG_EDGES,device=device)
    bins=torch.bucketize(metrics.abs().clamp_min(1e-38).log10().contiguous(),log_edges)+1
    bins[metrics==0]=0;nb=len(LOG_EDGES)+2
    key=((group[:,None]*nf+torch.arange(nf,device=device))*nb+bins).flatten()
    acc['hist'].view(-1).add_(torch.bincount(key,minlength=ng*nf*nb))
    cov=torch.bucketize(metrics[:,6].contiguous(),torch.tensor(COVERAGE_EDGES,device=device),right=True)
    nb=len(COVERAGE_EDGES)+1
    acc['coverage_hist'].view(-1).add_(torch.bincount(group*nb+cov,minlength=ng*nb))
    ordered=sigma.sort(1).values
    sb=torch.bucketize(ordered.contiguous(),torch.tensor(SCALE_EDGES,device=device),right=True);nb=len(SCALE_EDGES)+1
    key=((group[:,None]*3+torch.arange(3,device=device))*nb+sb).flatten()
    acc['axis_hist'].view(-1).add_(torch.bincount(key,minlength=ng*3*nb))
    sat=(sigma>=.39).sum(1)
    acc['saturation_hist'].view(-1).add_(torch.bincount(group*4+sat,minlength=ng*4))


def add_scores(acc,scores,labels):
    audits=[];edges=torch.tensor(MARGIN_EDGES,device=labels.device)
    for i,z in enumerate(scores):
        pred=z.argmax(0);conf=torch.bincount(labels*18+pred,minlength=324).reshape(18,18)
        acc['confusion'][i]+=conf
        value,cls=z[:17].max(0);margin=value-z[17]
        bins=torch.bucketize(margin.contiguous(),edges,right=True);nb=len(MARGIN_EDGES)+1
        flat=(labels*17+cls)*nb+bins
        acc['margin_hist'][i].view(-1).add_(torch.bincount(flat,minlength=18*17*nb).reshape(18,17,nb).view(-1))
        audits.append(conf.cpu().tolist())
    return audits


def initialize():
    cfg=Config.fromfile(CONFIG);old=read(ROOT/'work_dirs/boundary_gaussian_full/protocol.json')
    for p,digest in old['source_sha256'].items():assert sha(ROOT/p)==digest,p
    paths=[Path(__file__),*sorted((ROOT/'tools/gaussian_scale_probe').glob('*'))]
    paths=[p for p in paths if p.is_file()]
    protocol=dict(status='registered',version='exact_raw_attribute_center_support_v3',created_date='2026-10-20',config=str(CONFIG.relative_to(ROOT)),
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=old['checkpoint_sha256'],
        expected=old['expected'],sampling='None: every train and validation keyframe',
        seed=928,groups=GROUPS,fields=FIELDS,flags=FLAGS,conditions=CONDITIONS,
        log_edges=LOG_EDGES.tolist(),coverage_edges=COVERAGE_EDGES.tolist(),scale_edges=SCALE_EDGES.tolist(),
        margin_edges=MARGIN_EDGES.tolist(),scale_cap=.30,saturation_threshold=.39,
        support='Exact native integer boxes; centers/radii calculated by production PyTorch expressions',
        grouping='Groups 0..5 partition all Gaussians, group 6 is their sum; groups 7..10 overlap and are subsets of group 4. Center uses native grid cell. All-support-empty uses full-grid GT without camera masking.',
        losses='Actual weighted CE10, foreground Lovasz1; exact final-stage Set classification2. No full parameter update.',
        raw_attribute_derivatives='Exact sigmoid output captured before affine scale transform; never reconstructed from rounded physical scales',
        intervention='Geometry-only saturation selection, camera-integrated per-Gaussian mass controls; no GT used to select intervention Gaussians',
        limitations='Frozen-output gradients and interventions, not original optimizer dynamics or attribution across training history',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    p=RUN/'protocol.json'
    if p.exists():assert read(p)==json.loads(json.dumps(protocol)),'Use a fresh run after source changes.'
    else:
        write(p,protocol)
        import shutil
        for source in paths:
            target=RUN/'sources'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    return cfg,protocol


def process(network,batch,occ,setmod,audit=False):
    with torch.no_grad():out=forward(network,batch)
    raw=network._scale_probe_raw
    sigmoid=raw[...,:3].sigmoid()
    expected=.08+.32*sigmoid
    torch.testing.assert_close(expected,out['gaussian'].scales,atol=0,rtol=0)
    out['_scale_probe_jacobian']=(.32*sigmoid*(1-sigmoid))[0]
    source=losses_and_sources(out,batch,occ,setmod)
    with torch.no_grad():
        result,v,shape=probe(network.head,out,batch,source)
        scores,eligible,control=counterfactuals(network.head,out,batch,result,v,shape)
        group,metrics,flags,sigma=features(out,source,result,eligible,batch,v,shape)
    if audit:
        from tools.gaussian_scale_probe.audit import audit_frame
        audit_record=audit_frame(network,out,batch,source,result,v,shape,occ)
    else:audit_record=None
    return out,source,result,group,metrics,flags,sigma,scores,control,audit_record


def pilot():
    cfg=Config.fromfile(CONFIG);network=build(cfg);capture_attributes(network)
    for p in network.parameters():p.requires_grad_(False)
    occ,setmod=modules(cfg);records=[]
    from dataset.utils import custom_collate_fn_temporal
    for split in ('val','train'):
        dataset=EvaluationFrames(cfg,split);cache=FeatureCache(network,dataset)
        for index in (0,7):
            start=time.perf_counter();batch=to_cuda(custom_collate_fn_temporal([dataset[index]]))
            values=process(network,batch,occ,setmod,True)
            out,source,result,group,metrics,flags,sigma,scores,control,audit=values
            a=accumulator();add_grouped(a,group,metrics,flags,sigma)
            add_scores(a,scores,source['y'][source['mask']]);torch.cuda.synchronize()
            records.append(dict(split=split,index=index,seconds=time.perf_counter()-start,control=control,audit=audit))
            print(json.dumps(records[-1]),flush=True)
            del values,out,source,result,group,metrics,flags,sigma,scores,a,batch
        cache.close()
    _,protocol=initialize()
    write(RUN/'pilot.json',dict(status='passed',records=records,checkpoint_sha256=protocol['checkpoint_sha256']))


def run(workers):
    cfg,protocol=initialize();assert read(RUN/'pilot.json')['status']=='passed'
    network=build(cfg);capture_attributes(network)
    for p in network.parameters():p.requires_grad_(False)
    occ,setmod=modules(cfg)
    from torch.utils.data import DataLoader,Subset
    from dataset.utils import custom_collate_fn_temporal
    processed=0;completed=0;start=time.perf_counter()
    for split in ('val','train'):
        dataset=EvaluationFrames(cfg,split);by_scene={}
        for index,(scene,frame) in enumerate(dataset.keyframes):by_scene.setdefault(scene,[]).append(index)
        pending=[]
        for scene,ids in by_scene.items():
            path=RUN/'scenes'/split/(scene+'.json')
            if path.exists():
                record=read(path);assert record['indices']==ids and path.with_suffix('.npz').exists();completed+=1
            else:pending.extend(ids)
        loader=DataLoader(Subset(dataset,pending),batch_size=1,shuffle=False,num_workers=workers,
            collate_fn=custom_collate_fn_temporal,pin_memory=True,
            **(dict(prefetch_factor=2,persistent_workers=True) if workers else {}))
        cache=FeatureCache(network,dataset);active=None
        for index,batch in zip(pending,loader):
            scene,frame=dataset.keyframes[index]
            if active is None:
                active=scene;acc=accumulator();records=[]
                old=read(ROOT/'work_dirs/boundary_gaussian_full/scenes'/split/(scene+'.json'))
                previous={r['index']:r for r in old['records']}
            assert scene==active
            batch=to_cuda(batch)
            values=process(network,batch,occ,setmod)
            out,source,result,group,metrics,flags,sigma,scores,control,_=values
            with torch.no_grad():
                add_grouped(acc,group,metrics,flags,sigma)
                confusion=add_scores(acc,scores,source['y'][source['mask']])
                c=np.array(confusion[0]);fg=c[:17];orig=np.array(previous[index]['error_counts'])[1,0]
                actual=[np.trace(fg[:,:17]),fg[:,:17].sum()-np.trace(fg[:,:17]),fg[:,17].sum(),c[17,:17].sum(),c[17,17]]
                np.testing.assert_array_equal(actual,orig)
            records.append(dict(index=index,scene_frame=int(frame),sample_token=previous[index]['sample_token'],
                gaussian_counts=torch.bincount(group,minlength=6).cpu().tolist(),losses=source['losses'],
                control=control,confusion=confusion))
            processed+=1
            if len(records)==len(by_scene[scene]):
                for key in ('sums','abs_sums','positive','negative','hist','coverage_hist','axis_hist','saturation_hist','flags'):
                    acc[key][6]=acc[key][:6].sum(0)
                path=RUN/'scenes'/split/(scene+'.json');path.parent.mkdir(parents=True,exist_ok=True)
                arrays={k:v.cpu().numpy() for k,v in acc.items()}
                with open(path.with_suffix('.npz.tmp'),'wb') as f:np.savez_compressed(f,**arrays)
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                write(path,dict(status='complete',split=split,scene=scene,scene_name=old['scene_name'],
                    frames=len(records),indices=by_scene[scene],records=records,checkpoint_sha256=protocol['checkpoint_sha256']))
                completed+=1;elapsed=time.perf_counter()-start
                write(RUN/'progress.json',dict(status='running',completed_scenes=completed,total_scenes=850,
                    split=split,last_scene=old['scene_name'],frames_this_process=processed,seconds=elapsed,
                    frames_per_second=processed/elapsed,updated_at=datetime.now().astimezone().isoformat()))
                if completed%10==0:print(f'{completed}/850 scenes; {processed} frames; {processed/elapsed:.2f} frames/s',flush=True)
                active=None;del acc,arrays
            del values,out,source,result,group,metrics,flags,sigma,scores,batch
        assert active is None;cache.close();del cache,dataset,loader
    write(RUN/'execution_complete.json',dict(status='complete',scenes=completed,frames_this_process=processed,
        seconds=time.perf_counter()-start,checkpoint_sha256=protocol['checkpoint_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['pilot','run'],required=True)
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    pilot() if a.action=='pilot' else run(a.workers)
