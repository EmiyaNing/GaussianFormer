"""Additional center objectives: same-class 3sigma mean + uncovered-GT attraction.

Both original CD directions, Set classification and Occupancy losses are kept
outside this module. Discrete semantic/membership/coverage selections are fixed
in a backward step. These added objectives have no direct scale/rotation path.
"""
import torch
from torch import nn
from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices
from .semantic_neighborhood_mean import local_mean
from model.utils.utils import get_rotation_matrix

@OPENOCC_LOSS.register_module()
class SemanticNeighborhoodMeanLoss(nn.Module):
    def __init__(self,mean_weight=.5,uncovered_weight=.5,grid_shape=(200,200,16),
                 pc_range=(-40.,-40.,-1.,40.,40.,5.4),grid_size=.4):
        super().__init__()
        self.mean_weight,self.uncovered_weight=mean_weight,uncovered_weight
        self.shape,self.lower,self.h=tuple(grid_shape),tuple(pc_range[:3]),grid_size
    def components(self,inputs):
        g=inputs['gaussian'];m=inputs['metas']
        if g.means.shape[0]!=1:raise ValueError('Verification uses batch1 with accumulation')
        means,scales=g.means[0].float(),g.scales[0].float()
        R=get_rotation_matrix(g.rotations.float())[0]
        classes=g.semantics[0].detach().argmax(-1).long().contiguous()
        xyz=m['occ_xyz'].reshape(-1,3).float().contiguous()
        labels=m['occ_label'].flatten().long().contiguous()
        fg=m['occ_cam_mask'].flatten().bool()&(labels>=0)&(labels<17)
        ids=torch.where(fg)[0];zero=means.sum()*0
        if len(ids)==0:
            return dict(mean3=zero,uncovered1=zero),dict(empty_foreground=1,gs_total=len(means),
                GT_foreground=0,gs_local_same3=0,gs_local_same3_fraction=0.,local_same3_pairs=0,
                mean_same3_count=0.,mean3_active=0.,uncovered_GT=0,uncovered_GT_fraction=0.,
                same_core_GT_fraction=0.,idle_core_GS_fraction=1.,uncovered_center_distance=0.,
                opacity_zero_fraction=(g.opacities[0].detach().flatten()==0).float().mean())
        lookup=torch.full_like(labels,-1,dtype=torch.int32)
        lookup[ids]=torch.arange(len(ids),device=means.device,dtype=torch.int32)
        values,counts,core,same_core,per_gs=local_mean(means,scales,R,classes,xyz,labels,lookup,
            len(ids),self.shape,self.lower,self.h)
        mean3=values.sum()/len(means)
        with torch.no_grad():
            uncovered=core==0
            nearest=_nearest_indices(xyz[ids],means.detach(),1024)
        distance=(xyz[ids]-means[nearest]).norm(dim=-1)
        uncovered1=(distance*uncovered.float()).sum()/len(ids)
        active=counts>0
        diagnostics=dict(gs_total=len(means),GT_foreground=len(ids),gs_local_same3=int(active.sum()),
            gs_local_same3_fraction=active.float().mean(),local_same3_pairs=counts.sum(),
            mean_same3_count=counts.float().mean(),
            mean3_active=values.detach()[active].mean() if bool(active.any()) else zero.detach(),
            uncovered_GT=int(uncovered.sum()),uncovered_GT_fraction=uncovered.float().mean(),
            same_core_GT_fraction=(same_core>0).float().mean(),
            idle_core_GS_fraction=(per_gs==0).float().mean(),
            uncovered_center_distance=distance.detach()[uncovered].mean() if bool(uncovered.any()) else zero.detach(),
            opacity_zero_fraction=(g.opacities[0].detach().flatten()==0).float().mean())
        return dict(mean3=mean3,uncovered1=uncovered1),diagnostics
    def forward(self,inputs):
        pieces,diag=self.components(inputs)
        value=self.mean_weight*pieces['mean3']+self.uncovered_weight*pieces['uncovered1']
        return value,dict(diag,**{k:v.detach() for k,v in pieces.items()},
            mean_weight=self.mean_weight,uncovered_weight=self.uncovered_weight)
