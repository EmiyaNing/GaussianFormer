"""Fast-region native occupancy KD; experiment helpers without production changes."""
import copy,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion
from mmseg.models import build_segmentor
from model.response_distillation_forecast import forecast_with_position_gradient
from tools.build_sparseworld_region_attributes import category_label,speed
from tools.sparseworld_region_utils import DYNAMIC,STATIC,voxel_centers

OUT=Path('out/sparseworld_fast_motion_distillation/20261008_2048x512')
REPORT=Path('analysis/hypotheses/prediction_perception_regions/fast_motion_distillation')
HORIZONS=(2,4,6)
FORECAST=('ego_attention','position','reg','velocity','semantic','traj')
GROUPS=('occupied','fast','slow','moderate','motion_unknown','static')

class MotionIndex:
    def __init__(self):
        base=Path('data/nuscenes/v1.0-trainval')
        categories={a['token']:a['name'] for a in json.loads((base/'category.json').read_text())}
        instances={a['token']:category_label(categories[a['category_token']]) for a in json.loads((base/'instance.json').read_text())}
        self.times={a['token']:a['timestamp'] for a in json.loads((base/'sample.json').read_text())}
        self.annotations={a['token']:a for a in json.loads((base/'sample_annotation.json').read_text())}
        self.by_sample={}
        for a in self.annotations.values():
            a['label']=instances[a['instance_token']]
            if a['label'] is not None:self.by_sample.setdefault(a['sample_token'],[]).append(a)
        self.centers=voxel_centers().reshape(-1,3)
    def bins(self,labels,token,pose):
        flat=np.asarray(labels).reshape(-1)
        ids=np.flatnonzero(np.isin(flat,DYNAMIC))
        world=self.centers[ids]@pose[:3,:3].T+pose[:3,3]
        assignment=np.zeros(len(ids),np.int32);vs=[]
        for i,a in enumerate(self.by_sample.get(token,[]),1):
            local=(world-np.asarray(a['translation'],np.float32))@Quaternion(a['rotation']).rotation_matrix.astype(np.float32)
            size=np.asarray(a['size'],np.float32)[[1,0,2]]
            inside=(np.abs(local)<=size/2+1e-5).all(-1)&(flat[ids]==a['label'])
            assignment[inside&(assignment!=0)]=-1
            assignment[inside&(assignment==0)]=i
            vs.append(speed(a,self.annotations,self.times))
        result=np.full(len(flat),3,np.uint8)
        for i,v in enumerate(vs,1):
            if np.isfinite(v):result[ids[assignment==i]]=0 if v<.5 else (1 if v<2 else 2)
        return result.reshape(200,200,16)

def build(cfg,strict,full=False):
    spec=copy.deepcopy(strict.model)
    if full:
        spec.img_backbone.frozen_stages=-1
        spec.img_backbone.norm_eval=True
        spec.freeze_img_backbone=False
        spec.freeze_img_neck=False
    instance=build_segmentor(spec).cuda()
    weights=torch.load(cfg.student_checkpoint,map_location='cpu')['state_dict']
    instance.load_state_dict(weights,strict=True);instance.set_epoch(100)
    instance.eval().requires_grad_(full)
    if not full:
        for name in FORECAST:getattr(instance,name).requires_grad_(True)
    return instance

def mode(instance,full,training=True):
    if full:
        instance.train(training)
        # Batch one: affine weights train, running-stat buffers fixed.
        for module in instance.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    else:
        instance.eval()
        for name in FORECAST:getattr(instance,name).train(training)

def encode(instance,imgs,metas):
    state=dict(imgs=imgs,metas=metas,points=None)
    state.update(instance.extract_img_feat(**state));state.update(instance.lifter(**state))
    ego=instance.plan_head(metas['temporal_ego_states'].float().reshape(1,-1)).unsqueeze(1)
    state['query_points']=state['query_points']*((torch.tanh(instance.points_scale(ego))+1)*.35+.8)
    state.update(instance.encoder(**state))
    return state,ego

def pack_output(instance,state,ego,metas,position_gradient=True):
    stages=state['representation'];current=state['query_stamps']==0
    forecast=forecast_with_position_gradient(instance,stages[-1],state['query_stamps'],ego,position_gradient=position_gradient)
    return dict(opus_pred_points=[instance._world(s['query_points'][:,current]).flatten(1,2).float() for s in stages],
                opus_pred_logits=[s['opus_logits'][:,current].flatten(1,2).float() for s in stages],
                temporal_pred_points=[instance._world(s['query_points']).flatten(1,2).float() for s in stages],
                temporal_pred_logits=[s['opus_logits'].flatten(1,2).float() for s in stages],
                future_pred_points=[p.float() for p in forecast['points']],future_pred_logits=[l.float() for l in forecast['logits']],
                pred_traj=forecast['trajectories'],sparseworld_pretrain=False,metas=metas)

def cached_task(objective,out,metas,constant):
    value=out['points'][0].new_tensor(float(constant));details={}
    # Retain all six native future task horizons, no GT-point truncation.
    for step in range(6):
        cls,geo=objective._set_loss(out['points'][step][0].float(),out['logits'][step][0].float(),metas['future_occ_labels'][0,step].long())
        value=value+objective.lambda_future_cls*cls+objective.lambda_future_pts*geo
        details[f'future_{step+1}_cls']=float(cls.detach());details[f'future_{step+1}_geo']=float(geo.detach())
    traj=F.mse_loss(out['trajectories'],metas['temporal_trajs'][:,:6].float())
    return value+objective.lambda_traj*traj,details

def voxel_scores(instance,points,logits):
    """Native max-per-class / closing, relaxing hard score threshold for KD.

    Geometry/bin decisions and compactness weights are detached. KD updates
    semantic evidence; native task geometry retains its differentiable path.
    """
    raster=instance.head.rasterizer
    p=points[0].detach().float();scores=logits[0].float().sigmoid()
    lower=p.new_tensor(raster.pc_range[:3]);extent=p.new_tensor(raster.pc_range[3:])-lower
    with torch.no_grad():
        grouped=p.reshape(-1,48,3);norm=(grouped-lower)/extent
        distance=(norm-norm.mean(1,keepdim=True)).norm(dim=-1).flatten()
        labels=scores.detach().argmax(-1);factor=torch.ones_like(distance)
        compact=(labels==15)|(labels==16)
        factor[compact]=(.1/distance[compact].clamp_min(1e-8)).clamp(max=1.)
        keep=(p-grouped.mean(1).repeat_interleave(48,0)).norm(dim=-1)<raster.center_distance_threshold
        indices=((p-lower)/raster.grid_size).floor().long()
        keep&=((indices>=0)&(indices<indices.new_tensor(raster.grid_shape))).all(-1)
        ix=indices[keep];flat=ix[:,0]*3200+ix[:,1]*16+ix[:,2]
    scores=scores*factor[:,None]
    dense=scores.new_zeros(640000,17).scatter_reduce(0,flat[:,None].expand(-1,17),scores[keep],reduce='amax',include_self=True)
    volume=dense.reshape(200,200,16,17).permute(3,0,1,2)[None]
    if raster.padding:
        closed=-F.max_pool3d(-F.max_pool3d(volume,3,1,1),3,1,1)
        threshold=raster.threshold.to(volume)
        original=(volume.max(1).values>threshold[volume.argmax(1)]).unsqueeze(1)
        volume=torch.where(original,volume,closed)
    return volume[0].permute(1,2,3,0).reshape(640000,17)

def native_voxel_logits(instance,points,logits):
    """Native filtered/max/closing occupancy scores in stable logit space.

    Sigmoid is monotone, so max aggregation and max/min closing commute with
    sigmoid. Missing cells are -inf. Discrete geometry and keep decisions are
    detached, exactly as at inference; supported class evidence retains a
    gradient. Unlike probability clamping, BCEWithLogits also trains extreme
    finite logits. The compactness adjustment is evaluated stably in log space.
    """
    raster=instance.head.rasterizer
    p=points[0].detach().float();z=logits[0].float()
    lower=p.new_tensor(raster.pc_range[:3]);extent=p.new_tensor(raster.pc_range[3:])-lower
    threshold=raster.threshold.to(z).expand(z.shape[-1])
    with torch.no_grad():
        scores=z.sigmoid();factor=torch.ones_like(scores[:,0]);keep=torch.ones(len(p),device=p.device,dtype=torch.bool)
        if len(p)%48==0:
            grouped=p.reshape(-1,48,3);normalized=(grouped-lower)/extent
            distance=(normalized-normalized.mean(1,keepdim=True)).norm(dim=-1).flatten()
            compact=(scores.argmax(-1)==15)|(scores.argmax(-1)==16)
            factor[compact]=(.1/distance[compact].clamp_min(1e-8)).clamp(max=1.)
            keep&=(p-grouped.mean(1).repeat_interleave(48,0)).norm(dim=-1)<raster.center_distance_threshold
        adjusted=scores*factor[:,None]
        keep&=adjusted.amax(-1)>threshold[adjusted.argmax(-1)]
        indices=((p-lower)/raster.grid_size).floor().long()
        keep&=((indices>=0)&(indices<indices.new_tensor(raster.grid_shape))).all(-1)
        ix=indices[keep];flat=ix[:,0]*(raster.grid_shape[1]*raster.grid_shape[2])+ix[:,1]*raster.grid_shape[2]+ix[:,2]
    # logit(f * sigmoid(z)); for f=1 this is exactly z, even at +/-100.
    log_f=factor.log()[:,None];log_not_f=torch.log1p(-factor)[:,None]
    adjusted_z=z+log_f-torch.logaddexp(torch.zeros_like(z),z+log_not_f)
    size=int(np.prod(raster.grid_shape))
    dense=z.new_full((size,z.shape[-1]),-torch.inf).scatter_reduce(0,flat[:,None].expand(-1,z.shape[-1]),adjusted_z[keep],reduce='amax',include_self=True)
    volume=dense.reshape(*raster.grid_shape,z.shape[-1]).permute(3,0,1,2)[None]
    if raster.padding:
        closed=-F.max_pool3d(-F.max_pool3d(volume,3,1,1),3,1,1)
        probs=volume.detach().sigmoid()
        original=(probs.max(1).values>threshold[probs.argmax(1)]).unsqueeze(1)
        volume=torch.where(original,volume,closed)
    return volume[0].permute(1,2,3,0).reshape(size,z.shape[-1])

def occupancy_kd(instance,out,targets,policy='legacy'):
    if policy=='positive_supported_native':return positive_supported_occupancy_kd(instance,out,targets)
    if policy!='legacy':raise ValueError(f'Unknown occupancy KD policy: {policy}')
    losses=[];details=[]
    for j,h in enumerate(HORIZONS):
        ids=torch.as_tensor(targets[f'fast_{j}'],device='cuda',dtype=torch.long)
        if not len(ids):
            details.append(dict(voxels=0,score_supported=0,teacher_free=0));continue
        scores=voxel_scores(instance,out['points'][h-1],out['logits'][h-1])[ids]
        labels=torch.as_tensor(targets['teacher'][j].reshape(-1),device='cuda',dtype=torch.long)[ids]
        target=F.one_hot(labels,18)[:,:17].to(scores)
        # Empty teacher label 17 corresponds to all seventeen class targets 0.
        loss=F.binary_cross_entropy(scores.clamp(1e-6,1-1e-6),target,reduction='none').sum(-1).mean()
        losses.append(loss)
        details.append(dict(voxels=len(ids),score_supported=int((scores.detach().amax(-1)>1e-6).sum()),teacher_free=int((labels==17).sum())))
    # Empty horizons contribute zero; fixed divide-by-three keeps temporal weights fixed.
    loss=sum(losses,out['logits'][0].sum()*0)/3
    return loss,details

def positive_supported_occupancy_kd(instance,out,targets):
    """Occupancy-only KD: ignore teacher misses and untrainable raster cells.

    This is a target-policy change, not an oracle teacher-correctness filter:
    no GT class is read here. The pre-existing GT-fast mask selects the region.
    The denominator remains ALL original fast cells, preserving the scale of
    each surviving voxel's gradient; empty horizons still have weight zero.
    """
    losses=[];details=[];device=out['logits'][0].device
    for j,h in enumerate(HORIZONS):
        ids=torch.as_tensor(targets[f'fast_{j}'],device=device,dtype=torch.long)
        detail=dict(voxels=len(ids),score_supported=0,teacher_free=0,retained=0,positive_unsupported=0)
        if not len(ids):details.append(detail);continue
        labels=torch.as_tensor(targets['teacher'][j].reshape(-1),device=device,dtype=torch.long)[ids]
        z=native_voxel_logits(instance,out['points'][h-1],out['logits'][h-1])[ids]
        supported=torch.isfinite(z).all(-1);positive=labels<17;selected=supported&positive
        detail.update(score_supported=int(supported.sum()),teacher_free=int((~positive).sum()),retained=int(selected.sum()),positive_unsupported=int((positive&~supported).sum()))
        if selected.any():
            target=F.one_hot(labels[selected],17).to(z)
            losses.append(F.binary_cross_entropy_with_logits(z[selected],target,reduction='none').sum()/len(ids))
        details.append(detail)
    # Zero connected to semantic parameters: backward remains valid for an
    # entirely empty/unsupported batch and never evaluates BCE at -inf.
    return sum(losses,out['logits'][0].sum()*0)/3,details

def region_counts(gt,pred,bins):
    valid=gt<18;occupied=gt<17
    masks=[occupied,bins==2,bins==0,bins==1,np.isin(gt,DYNAMIC)&(bins==3),np.isin(gt,STATIC)]
    counts=np.array([[m.sum(),((pred==gt)&m).sum(),((pred<17)&m).sum()] for m in masks],np.int64)
    hist=np.bincount(gt[valid].astype(np.int64)*18+pred[valid],minlength=324).reshape(18,18)
    return hist,counts
