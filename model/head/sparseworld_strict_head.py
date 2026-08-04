"""Source-style SparseWorld point-to-voxel rendering, isolated from OPUSHead."""
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


class _Rasterizer(nn.Module):
    def __init__(self, pc_range, grid_size, grid_shape, score_threshold, center_distance_threshold, padding, empty_label=17):
        super().__init__(); self.pc_range=tuple(pc_range); self.grid_size=float(grid_size); self.grid_shape=tuple(grid_shape)
        self.threshold=torch.as_tensor(score_threshold, dtype=torch.float32); self.center_distance_threshold=float(center_distance_threshold); self.padding=padding; self.empty_label=empty_label

    def forward(self, points, logits, group_size=48):
        scores=logits.sigmoid(); b, n = points.shape[:2]; threshold=self.threshold.to(scores)
        if threshold.numel()==1: threshold=threshold.expand(scores.shape[-1])
        lower=points.new_tensor(self.pc_range[:3]); out=[]
        for i in range(b):
            p, s=points[i], scores[i]
            if group_size > 1 and p.shape[0] % group_size == 0:
                # ``p`` stores Q consecutive groups of R refined points.
                # Expand each group centre back to Q*R before computing the
                # source renderer's centre-distance validity mask.
                centers = p.view(-1, group_size, 3).mean(1)
                centers = centers.repeat_interleave(group_size, dim=0)
                distances = (p - centers).norm(dim=-1)
                p,s=p[distances < self.center_distance_threshold],s[distances < self.center_distance_threshold]
            keep=s.amax(-1) > threshold[s.argmax(-1)]; p,s=p[keep],s[keep]
            index=((p-lower)/self.grid_size).floor().long(); valid=((index>=0)&(index<index.new_tensor(self.grid_shape))).all(-1); index,s=index[valid],s[valid]
            dense=s.new_zeros(*self.grid_shape,s.shape[-1])
            if index.numel():
                flat=index[:,0]*(self.grid_shape[1]*self.grid_shape[2])+index[:,1]*self.grid_shape[2]+index[:,2]
                max_scores=s.new_zeros(self.grid_shape[0]*self.grid_shape[1]*self.grid_shape[2], s.shape[-1])
                max_scores.scatter_reduce_(0, flat[:,None].expand_as(s), s, reduce='amax', include_self=True)
                dense=max_scores.view(*self.grid_shape,s.shape[-1])
            volume=dense.permute(3,0,1,2).unsqueeze(0)
            if self.padding:
                closed=-F.max_pool3d(-F.max_pool3d(volume,3,1,1),3,1,1)
                original=(volume.max(1).values > threshold[volume.argmax(1)]).expand_as(closed)
                closed[original]=volume[original]; volume=closed
            dense=volume[0].permute(1,2,3,0); occupied=(dense>threshold).any(-1)
            labels=torch.full(self.grid_shape,self.empty_label,device=p.device,dtype=torch.long)
            labels[occupied]=dense[occupied].argmax(-1); out.append(labels.flatten())
        return torch.stack(out)


@MODELS.register_module()
class SparseWorldStrictHead(BaseModule):
    """Select current timestamp queries and preserve original max-score voxelization."""
    def __init__(self, num_classes=17, point_multipliers=(1,4,16,24,32,48), pc_range=(-40.,-40.,-1.,40.,40.,5.4), grid_size=.4, grid_shape=(200,200,16), empty_label=17, score_threshold=(.35,)*15+(.25,.3), center_distance_threshold=3., padding=True, embed_dims=None, decoder_outputs_logits=None, init_cfg=None, **kwargs):
        super().__init__(init_cfg); self.num_classes=num_classes; self.point_multipliers=tuple(point_multipliers); self.pc_range=tuple(pc_range)
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unsupported SparseWorldStrictHead options: {unknown}')
        self.rasterizer=_Rasterizer(pc_range,grid_size,grid_shape,score_threshold,center_distance_threshold,padding,empty_label)

    def _world(self, points):
        return points*(points.new_tensor(self.pc_range[3:])-points.new_tensor(self.pc_range[:3]))+points.new_tensor(self.pc_range[:3])

    def forward(self, representation, query_stamps, metas=None, **kwargs):
        current=query_stamps==0; points=[]; logits=[]
        for state, count in zip(representation,self.point_multipliers):
            points.append(self._world(state['query_points'][:,current].flatten(1,2)))
            logits.append(state['opus_logits'][:,current].flatten(1,2))
        final=self.rasterizer(points[-1].detach(),logits[-1].detach(),self.point_multipliers[-1])
        return dict(opus_pred_points=points, opus_pred_logits=logits, opus_points=points[-1], opus_labels=logits[-1].argmax(-1), opus_scores=logits[-1].sigmoid().amax(-1), final_occ=final, final_occ_grid=final.view(final.shape[0],*self.rasterizer.grid_shape))
