"""Section 4.1: solid-ellipsoid coverage and local same-component field fitting.

All original Set/Occupancy losses remain separate. This loss only supervises
the final predicted GS geometry; it never scales a term by opacity or amplitude.
"""
import torch
from torch import nn
from torch.nn import functional as F
from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices
from .semantic_gaussian_field import extension
from .semantic_gaussian_field.geometry import distance_with_projection
from model.utils.utils import get_rotation_matrix


def smooth(value):
    return F.smooth_l1_loss(value, torch.zeros_like(value), beta=1., reduction='none')


@OPENOCC_LOSS.register_module()
class SemanticGaussianFieldLoss(nn.Module):
    def __init__(self, mode='semantic', weight=.1, grid_shape=(200,200,16),
                 pc_range=(-40.,-40.,-1.,40.,40.,5.4), grid_size=.4,
                 samples_per_gaussian=8, local_radius=2., max_scale=.4,
                 purity_start=.5, purity_full=.8, reliable_threshold=.5,
                 prediction_blend=.25, blend_warmup_frames=512, seed=928):
        super().__init__()
        if mode not in ('geometry','semantic'):
            raise ValueError(mode)
        self.mode,self.weight=mode,float(weight)
        self.shape,self.lower=tuple(grid_shape),tuple(pc_range[:3])
        self.h,self.samples,self.radius,self.max_scale=grid_size,samples_per_gaussian,local_radius,max_scale
        self.purity_start,self.purity_full=purity_start,purity_full
        self.reliable_threshold=reliable_threshold
        self.prediction_blend,self.blend_warmup=prediction_blend,blend_warmup_frames
        self.seed=seed
        self.last_geometry=None
        self.last_raw=None

    def _membership(self, points, labels, camera):
        with torch.no_grad():
            lower=points.new_tensor(self.lower)
            cells=torch.floor((points-lower)/self.h).long()
            shape=torch.tensor(self.shape,device=points.device)
            inside=((cells>=0)&(cells<shape)).all(-1)
            cells=torch.minimum(cells.clamp_min(0),shape-1)
            ids=(cells[...,0]*self.shape[1]+cells[...,1])*self.shape[2]+cells[...,2]
            known=inside&camera[ids]&(labels[ids]>=0)&(labels[ids]<18)
        return ids,known

    def correspondence(self, means, semantic, xyz, labels, camera, components, step):
        fg=camera&(labels>=0)&(labels<17)
        fg_ids=torch.where(fg)[0].to(torch.int32).contiguous()
        targets=xyz[fg].contiguous()
        seed=_nearest_indices(means.detach(),targets,1024)
        ext=extension()
        with torch.no_grad():
            assigned,purity,count=ext.correspondence(means.detach().contiguous(),seed.contiguous(),
                fg_ids,labels.contiguous(),components.contiguous(),self.shape,self.lower,self.h,self.radius)
            distance=(means.detach()-targets[seed]).norm(dim=-1)
            confidence=((purity-self.purity_start)/(self.purity_full-self.purity_start)).clamp(0,1)
            confidence*=((distance<=self.radius)&(count>=2)).float()
            p=semantic.detach()/semantic.detach().sum(-1,keepdim=True).clamp_min(1e-20)
            agreement=p.gather(1,assigned[:,:1].long()).squeeze(1)
            blend=self.prediction_blend*min(1.,max(0.,step/max(self.blend_warmup,1)))
            confidence*=1-blend+blend*agreement
        return fg,assigned,confidence,purity,count

    def forward(self, inputs):
        g,metas=inputs['gaussian'],inputs['metas']
        if g.means.shape[0]!=1:
            raise ValueError('First verification uses batch size one with accumulation')
        means,scales=g.means[0].float(),g.scales[0].float()
        rotation=get_rotation_matrix(g.rotations.float())[0]
        labels=metas['occ_label'].flatten().long()
        camera=metas['occ_cam_mask'].flatten().bool()
        xyz=metas['occ_xyz'].reshape(-1,3).float()
        components=metas['field_components'].flatten().to(torch.int32).contiguous()
        volume=metas['field_distance'][0].float()
        zero=means.sum()*0
        if not bool((camera&(labels>=0)&(labels<17)).any()):
            self.last_raw=zero
            return zero,dict(empty_foreground=1)
        if float(scales.detach().amax())>self.max_scale+1e-6:
            raise ValueError('Branch-and-bound maximum scale violated')
        step=int(inputs.get('global_iter',0))
        fg,assigned,confidence,purity,count=self.correspondence(means,g.semantics[0].float(),
            xyz,labels,camera,components,step)
        # Independent RNG: adding this objective cannot change model dropout,
        # data augmentation or the paired baseline's future random sequence.
        generator=torch.Generator(device=means.device)
        frame_seed=int(metas.get('field_seed',torch.tensor([self.seed])).flatten()[0])
        generator.manual_seed(frame_seed)
        target=xyz[fg]+(torch.rand((int(fg.sum()),3),device=means.device,generator=generator)-.5)*self.h
        heads,links=extension().build_hash(means.detach().contiguous(),self.shape,self.lower,self.h)
        seed=_nearest_indices(target,means.detach(),1024)
        with torch.no_grad():
            ids,u,distances,fallback=extension().cover(target.contiguous(),means.detach().contiguous(),
                scales.detach().contiguous(),rotation.detach().contiguous(),heads,links,seed.contiguous(),
                components[fg].contiguous(),assigned,(confidence>=self.reliable_threshold).contiguous(),
                self.shape,self.lower,self.h,self.max_scale,self.radius,self.mode=='semantic')
        direction=1 if self.mode=='semantic' else 0
        selected=ids[:,direction]
        distance=distance_with_projection(target,means[selected],scales[selected],rotation[selected],
            u[:,direction],distances[:,direction]==0)
        cover=smooth(distance/self.h).mean()
        n,k=len(means),self.samples
        need=n*k
        eps=torch.randn((int(need*1.08)+128,3),device=means.device,generator=generator)
        eps=eps[eps.square().sum(-1)<=9]
        if len(eps)<need:
            raise RuntimeError('Unexpected truncated-normal rejection shortage')
        eps=eps[:need].reshape(n,k,3)
        positions=means[:,None]+torch.einsum('nkj,nji->nki',eps*scales[:,None],rotation)
        flat=positions.reshape(-1,3)
        sample_ids,known=self._membership(flat,labels,camera)
        # Exact grid-center solid distances, trilinearly interpolated off-grid.
        lower=means.new_tensor(self.lower)
        extent=means.new_tensor(self.shape)*self.h
        grid=(2*(flat-lower)/extent-1).view(1,1,1,-1,3)
        sampled=F.grid_sample(volume.permute(2,1,0)[None,None],grid,
            mode='bilinear',padding_mode='border',align_corners=False).flatten()
        empty=known&(labels[sample_ids]==17)
        spill_cost=smooth(sampled/self.h)*empty.float()
        fit=zero
        if self.mode=='semantic':
            active=(known&(confidence.repeat_interleave(k)>0)).contiguous()
            with torch.no_grad():
                closest,found=extension().fit(flat.detach().contiguous(),means.detach().contiguous(),
                    assigned,components,active,k,self.shape,self.lower,self.h,self.radius)
            if bool((active&~found).any()):
                raise RuntimeError('Reliable correspondence has no local component target')
            fit_cost=smooth((flat-closest).norm(dim=-1)/self.h)
            fit=(fit_cost*known.float()*confidence.repeat_interleave(k)).sum()/need
            spill=(spill_cost*(1-confidence.repeat_interleave(k))).sum()/need
        else:
            spill=spill_cost.sum()/need
        raw=cover+fit+spill
        self.last_raw=raw
        self.last_geometry=(means,scales,g.rotations,rotation)
        metrics=dict(cover=cover.detach(),fit=fit.detach(),spill=spill.detach(),raw=raw.detach(),
            gt_foreground=len(target),gaussians=n,samples=need,
            covered_random_fraction=(distances[:,0]==0).float().mean(),
            reliable_fraction=(confidence>=self.reliable_threshold).float().mean(),
            correspondence_fraction=(confidence>0).float().mean(),mean_confidence=confidence.mean(),
            local_purity_mean=purity.mean(),known_sample_fraction=known.float().mean(),
            known_empty_sample_fraction=empty.float().mean(),fallback_fraction=fallback.float().mean(),
            gt_same_core_fraction=((distances[:,1]==0)&~fallback).float().mean(),
            actual_weight=self.weight)
        return self.weight*raw,metrics
