"""Final-stage solid-core coverage with either max-spill or real-render decision cost."""
import torch
from torch import nn
from torch.nn import functional as F
from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices
from .semantic_gaussian_field import extension as hash_extension
from .semantic_gaussian_field.geometry import distance_with_projection
from .bidirectional_union_coverage import extension
from model.utils.utils import get_rotation_matrix


def rho(value):
    return F.smooth_l1_loss(value,torch.zeros_like(value),beta=1.,reduction='none')


@torch.no_grad()
def negative_log_shares(raw_logits):
    """Stable log softplus, with its FP32-exact asymptote in the far negative tail."""
    z=raw_logits.detach().float()
    logs=torch.where(z < -20,z,torch.log(F.softplus(z).clamp_min(torch.finfo(z.dtype).tiny)))
    return (torch.logsumexp(logs,-1,keepdim=True)-logs).contiguous()


@torch.no_grad()
def sample_valid(valid,limit,seed):
    ids=torch.where(valid)[0]
    if len(ids)<=limit:return ids
    generator=torch.Generator(device=ids.device);generator.manual_seed(int(seed))
    return ids[torch.randperm(len(ids),device=ids.device,generator=generator)[:limit]]


def geometry_render(head,gaussian,xyz):
    """Same numerical renderer and empty score; only predicted geometry has gradients."""
    fixed=gaussian._replace(opacities=gaussian.opacities.detach(),semantics=gaussian.semantics.detach())
    means,opacity,semantic,scales,inverse=head.prepare_gaussian_args(fixed)
    logits=head._render_batched(xyz[None],means,opacity.detach(),semantic.detach(),scales,inverse)[0].T
    if hasattr(head,'empty_bias'):
        logits=torch.cat((logits[:,:head.empty_label],
            logits[:,head.empty_label:]-head.empty_bias.detach()),-1)
    if getattr(head,'empty_logit_offset',0.):
        raise ValueError('The experiment requires the trainable-bias renderer without evaluation-only offsets')
    return logits


@OPENOCC_LOSS.register_module()
class BidirectionalUnionCoverageLoss(nn.Module):
    def __init__(self,mode='union',weight=1.,reverse_weight=1.,reverse_samples=4096,
                 grid_shape=(200,200,16),pc_range=(-40.,-40.,-1.,40.,40.,5.4),
                 grid_size=.4,max_scale=.4,seed=928):
        super().__init__()
        if mode not in ('union','decision'):raise ValueError(mode)
        self.mode,self.weight,self.reverse_weight=mode,float(weight),float(reverse_weight)
        self.samples,self.shape,self.lower=int(reverse_samples),tuple(grid_shape),tuple(pc_range[:3])
        self.h,self.max_scale,self.seed=float(grid_size),float(max_scale),int(seed)
        if self.samples<1 or min(self.weight,self.reverse_weight)<0:raise ValueError('Invalid sampling/weights')
        self.last_raw=None;self.last_geometry=None

    def costs(self,means,scales,rotation,neglogp,foreground_xyz,foreground_labels,reverse_xyz=None):
        """Exhaustive semantic minimum using feasible seeds and lower-bound pruning."""
        m=len(foreground_xyz);s=0 if reverse_xyz is None else len(reverse_xyz)
        unique=foreground_xyz if s==0 else torch.cat((foreground_xyz,reverse_xyz),0)
        if not len(unique):
            return means.new_empty(0),dict(ids=torch.empty(0,device=means.device,dtype=torch.long),
                distances=means.new_empty(0),visited=means.new_empty(0),radii=means.new_empty(0))
        with torch.no_grad():
            heads,links=hash_extension().build_hash(means.detach().contiguous(),self.shape,self.lower,self.h)
            global_seed=_nearest_indices(unique,means.detach(),1024)
            fg_seed=global_seed[:m].clone()
            reverse_seed=global_seed[m:,None].expand(-1,17).clone()
            predicted=neglogp.argmin(-1)
            # Predicted-class nearest centers only improve the feasible seed;
            # they do not restrict the candidates in the exhaustive search.
            for c in range(17):
                ids=torch.where(predicted==c)[0]
                mask=foreground_labels==c
                if not len(ids) or not (bool(mask.any()) or s):continue
                positions=foreground_xyz[mask] if s==0 else torch.cat((foreground_xyz[mask],reverse_xyz),0)
                nearest=ids[_nearest_indices(positions,means.detach()[ids],1024)]
                nf=int(mask.sum());fg_seed[mask]=nearest[:nf]
                if s:reverse_seed[:,c]=nearest[nf:]
            points=foreground_xyz if s==0 else torch.cat((foreground_xyz,reverse_xyz[:,None].expand(-1,17,-1).reshape(-1,3)),0)
            classes=foreground_labels if s==0 else torch.cat((foreground_labels,torch.arange(17,device=means.device).repeat(s)),0)
            first=global_seed if s==0 else torch.cat((global_seed[:m],global_seed[m:].repeat_interleave(17)))
            second=fg_seed if s==0 else torch.cat((fg_seed,reverse_seed.flatten()))
            selected,u,d,cost,visited,radii=extension().query(points.contiguous(),classes.int().contiguous(),
                means.detach().contiguous(),scales.detach().contiguous(),rotation.detach().contiguous(),
                neglogp.contiguous(),neglogp.amin(0).contiguous(),heads,links,
                torch.stack((first,second),-1).contiguous(),self.shape,self.lower,self.h,self.max_scale)
        distance=distance_with_projection(points,means[selected],scales[selected],rotation[selected],u,d==0)
        values=rho(distance/self.h)+neglogp[selected,classes]
        return values,dict(ids=selected,distances=d,visited=visited,radii=radii,
                           assignment_value=cost,projection=u,classes=classes,points=points)

    def components(self,inputs):
        g=inputs['gaussian'];metas=inputs['metas']
        if g.means.shape[0]!=1:raise ValueError('This verification uses batch size one with accumulation')
        means,scales=g.means[0].float(),g.scales[0].float()
        rotation=get_rotation_matrix(g.rotations.float())[0]
        labels=metas['occ_label'].reshape(-1).long()
        camera=metas['occ_cam_mask'].reshape(-1).bool()
        xyz=metas['occ_xyz'].reshape(-1,3).float()
        valid=camera&(labels>=0)&(labels<18);foreground=valid&(labels<17)
        m,v=int(foreground.sum()),int(valid.sum());zero=means.sum()*0
        if not v:return zero,zero,dict(gt_foreground=0,valid_voxels=0,reverse_samples=0)
        masks=inputs.get('opus_point_valid_masks')
        if masks is not None and not bool(masks[-1].all()):raise ValueError('Matched baseline requires all queries valid')
        if not bool(torch.isfinite(means).all()&torch.isfinite(scales).all()) or float(scales.detach().max())>self.max_scale+1e-6:
            raise ValueError('Nonfinite geometry or violated branch-and-bound scale limit')
        seed=int(metas.get('union_seed',torch.tensor([self.seed])).flatten()[0])
        sample=sample_valid(valid,self.samples,seed)
        query_xyz,query_y=xyz[foreground],labels[foreground]
        neglogp=negative_log_shares(inputs['opus_pred_logits'][-1][0])
        costs,info=self.costs(means,scales,rotation,neglogp,query_xyz,query_y,
                              xyz[sample] if self.mode=='union' else None)
        cover=costs[:m].mean() if m else zero
        denominator=m if m else v
        if self.mode=='union':
            supports=(-costs[m:].reshape(len(sample),17)).exp()
            wrong=torch.ones_like(supports,dtype=torch.bool)
            fg_sample=labels[sample]<17
            wrong[torch.where(fg_sample)[0],labels[sample][fg_sample]]=False
            reverse=(supports*wrong).sum()/len(sample)*(v/denominator)
        else:
            if 'renderer_head' not in inputs:raise ValueError('Decision mode needs the actual renderer_head')
            logits=geometry_render(inputs['renderer_head'],g,xyz[sample])
            true=logits.gather(1,labels[sample,None]).squeeze(1)
            wrong=logits.scatter(1,labels[sample,None],float('-inf')).max(1).values
            reverse=(wrong-true).relu().mean()*(v/denominator)
        d=info['distances'][:m]
        metrics=dict(gt_foreground=m,valid_voxels=v,reverse_samples=len(sample),
            reverse_sample_inverse_probability=v/len(sample),core_satisfied_fraction=(d==0).float().mean() if m else zero.detach(),
            cover_geometric=rho(d/self.h).mean() if m else zero.detach(),
            mean_projected_candidates=info['visited'].float().mean() if len(costs) else zero.detach(),
            mean_search_radius_m=info['radii'].mean() if len(costs) else zero.detach(),
            actual_weight=self.weight,reverse_weight=self.reverse_weight)
        return cover,reverse,metrics

    def forward(self,inputs):
        cover,reverse,metrics=self.components(inputs)
        raw=cover+self.reverse_weight*reverse;self.last_raw=raw
        metrics.update(cover=cover.detach(),reverse=reverse.detach(),raw=raw.detach(),
            weighted_cover=(self.weight*cover).detach(),weighted_reverse=(self.weight*self.reverse_weight*reverse).detach())
        return self.weight*raw,metrics
