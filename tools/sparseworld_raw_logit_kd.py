"""Raw-logit readout and explicit KL API diagnostics, without activations.

The two F.kl_div policies deliberately test invalid probability interpretations.
Gaussian KL treats the raw values as means, and is NOT categorical semantic KL.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal,kl_divergence
from tools.sparseworld_fast_motion_common import HORIZONS

POLICIES=('raw_api_default','raw_api_log_target','gaussian_raw')

def raw_voxel_logits(instance,points,logits):
    """Per-class maximum of original logits in each .4m voxel.

    Keep the native geometric group-distance and range checks. There is no
    sigmoid, softmax, score threshold, compactness rescaling, closing, or
    invented free logit. A missing voxel has -inf in all channels. Every finite
    component is one of the network's original point logits.
    """
    raster=instance.head.rasterizer;p=points[0].detach().float();z=logits[0].float()
    with torch.no_grad():
        lower=p.new_tensor(raster.pc_range[:3]);index=((p-lower)/raster.grid_size).floor().long()
        keep=((index>=0)&(index<index.new_tensor(raster.grid_shape))).all(-1)
        if len(p)%48==0:
            groups=p.reshape(-1,48,3)
            keep&=(p-groups.mean(1).repeat_interleave(48,0)).norm(dim=-1)<raster.center_distance_threshold
        ix=index[keep];flat=ix[:,0]*(raster.grid_shape[1]*raster.grid_shape[2])+ix[:,1]*raster.grid_shape[2]+ix[:,2]
    size=int(np.prod(raster.grid_shape))
    return z.new_full((size,z.shape[-1]),-torch.inf).scatter_reduce(0,flat[:,None].expand(-1,z.shape[-1]),z[keep],reduce='amax',include_self=True)

def elementwise_loss(student,teacher,policy,sigma=1.):
    teacher=teacher.detach()
    if policy=='raw_api_default':return F.kl_div(student,teacher,reduction='none',log_target=False)
    if policy=='raw_api_log_target':return F.kl_div(student,teacher,reduction='none',log_target=True)
    if policy=='gaussian_raw':
        if sigma<=0:raise ValueError('Gaussian sigma must be positive')
        scale=torch.full_like(student,float(sigma))
        return kl_divergence(Normal(teacher,scale,validate_args=False),Normal(student,scale,validate_args=False))
    raise ValueError(f'Unknown raw-logit policy: {policy}')

def paired_raw_logits(instance,out,targets,j):
    device=out['logits'][0].device;h=HORIZONS[j]
    ids=torch.as_tensor(targets[f'fast_{j}'],device=device,dtype=torch.long)
    teacher=torch.as_tensor(targets[f'teacher_raw_{j}'],device=device,dtype=torch.float32)
    teacher_valid=torch.as_tensor(targets[f'teacher_raw_valid_{j}'],device=device,dtype=torch.bool)
    assert teacher.shape==(len(ids),17) and teacher_valid.shape==(len(ids),)
    if not len(ids):return teacher,teacher,teacher_valid,dict(voxels=0,teacher_supported=0,student_supported=0,retained=0)
    student=raw_voxel_logits(instance,out['points'][h-1],out['logits'][h-1])[ids]
    valid=teacher_valid&torch.isfinite(teacher).all(-1)&torch.isfinite(student).all(-1)
    detail=dict(voxels=len(ids),teacher_supported=int(teacher_valid.sum()),student_supported=int(torch.isfinite(student).all(-1).sum()),retained=int(valid.sum()))
    return student[valid],teacher[valid].detach(),valid,detail

def raw_occupancy_kd(instance,out,targets,policy,sigma=1.):
    losses=[];details=[]
    for j in range(3):
        student,teacher,_,detail=paired_raw_logits(instance,out,targets,j)
        if len(student):
            # Per-component average, divided by the ORIGINAL GT-fast mask.
            term=elementwise_loss(student,teacher,policy,sigma)
            losses.append(term.sum()/(detail['voxels']*17))
            detail.update(student_mean=float(student.detach().mean()),teacher_mean=float(teacher.mean()),
                          raw_rmse=float((student.detach()-teacher).square().mean().sqrt()),teacher_negative_elements=int((teacher<0).sum()))
        details.append(detail)
    return sum(losses,out['logits'][0].sum()*0)/3,details
