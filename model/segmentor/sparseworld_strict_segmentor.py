"""Strict, self-contained SparseWorld trajectory segmentor."""
import torch
import torch.nn.functional as F
from torch import nn
from mmseg.models import SEGMENTORS
from .opus_segmentor import OPUSSegmentor


class _EgoAttention(nn.Module):
    def __init__(self, dims, heads, dropout, pc_range):
        super().__init__(); self.attention=nn.MultiheadAttention(dims,heads,dropout,batch_first=True); self.tau=nn.Linear(dims,heads); self.pc_range=tuple(pc_range)
        nn.init.zeros_(self.tau.weight); nn.init.uniform_(self.tau.bias,0.,2.)
    def forward(self, ego_feature, query_feature, query_points):
        lower=query_points.new_tensor(self.pc_range[:3]); world=query_points.mean(2)*(query_points.new_tensor(self.pc_range[3:])-lower)+lower
        mask=(-(world.norm(dim=-1)[:,None]*F.softplus(self.tau(query_feature)).transpose(1,2)/40.).unsqueeze(2).flatten(0,1))
        return self.attention(ego_feature,query_feature,query_feature,attn_mask=mask,need_weights=False)[0]


@SEGMENTORS.register_module()
class SparseWorldStrictSegmentor(OPUSSegmentor):
    """Faithful SparseWorld topology without changing the legacy segmentor.

    All 1040 queries are encoded by OPUS before autoregressive forecasting.
    The trajectory decoder consumes the encoded timestamp-specific future
    queries rather than zero tokens, and its output is rasterized in the
    future ego grid (the original evaluation convention).
    """
    def __init__(self, future_queries=(60,60,60,60,40,40), finetune_epoch=5, embed_dims=256,
                 num_refines=48, pc_range=(-40.,-40.,-1.,40.,40.,5.4), ego_state_dim=21, dropout=.1, **kwargs):
        super().__init__(**kwargs); self.future_queries=tuple(future_queries); self.future_steps=len(self.future_queries); self.finetune_epoch=finetune_epoch; self.num_refines=num_refines; self.pc_range=tuple(pc_range); self.current_epoch=0; self.pretrain=True
        self.plan_head=nn.Sequential(nn.Linear(ego_state_dim,256),nn.ReLU(True),nn.Linear(256,256),nn.ReLU(True),nn.Linear(256,embed_dims))
        self.points_scale=nn.Sequential(nn.Linear(embed_dims,64),nn.ReLU(True),nn.Linear(64,32),nn.ReLU(True),nn.Linear(32,3))
        self.ego_attention=_EgoAttention(embed_dims,8,dropout,pc_range)
        self.position=nn.Sequential(nn.Linear(4*num_refines,embed_dims),nn.LayerNorm(embed_dims),nn.ReLU(True),nn.Linear(embed_dims,embed_dims),nn.LayerNorm(embed_dims),nn.ReLU(True))
        def branch(out): return nn.Sequential(nn.Linear(embed_dims,embed_dims),nn.ReLU(True),nn.Linear(embed_dims,embed_dims),nn.ReLU(True),nn.Linear(embed_dims,out))
        self.reg,self.velocity,self.semantic=branch(3*num_refines),branch(2*num_refines),branch(17*num_refines)
        self.traj=nn.Sequential(nn.Linear(embed_dims,2*embed_dims),nn.Softplus(),nn.Linear(2*embed_dims,2)); nn.init.constant_(self.semantic[-1].bias,-4.59511985013459)

    @torch.no_grad()
    def set_epoch(self, epoch):
        self.current_epoch=int(epoch); self.pretrain=self.current_epoch<self.finetune_epoch; self.lifter.set_pretrain(self.pretrain)

    def _world(self,p):
        return p*(p.new_tensor(self.pc_range[3:])-p.new_tensor(self.pc_range[:3]))+p.new_tensor(self.pc_range[:3])
    def _norm(self,p):
        return (p-p.new_tensor(self.pc_range[:3]))/(p.new_tensor(self.pc_range[3:])-p.new_tensor(self.pc_range[:3]))
    def _refine(self,p,offset): return self._norm(self._world(p).mean(2,keepdim=True)+offset.view(*offset.shape[:2],self.num_refines,3))

    def forward(self, imgs=None, metas=None, points=None, **kwargs):
        result=dict(imgs=imgs,metas=metas,points=points); result.update(kwargs); result.update(self.extract_img_feat(**result)); result.update(self.lifter(**result))
        state=metas['temporal_ego_states'].to(result['query_points']).reshape(imgs.shape[0],-1); ego=self.plan_head(state).unsqueeze(1)
        # Original range-adaptive perception scales the transformer initial anchors.
        scale=(torch.tanh(self.points_scale(ego))+1.)*.35+.8; result['query_points']=result['query_points']*scale
        result.update(self.encoder(**result)); result.update(self.head(**result))
        # Keep full timestamped decoder predictions for the source temporal
        # pretraining objective. The public head intentionally exposes only
        # stamp-0 points to the ordinary OPUS loss/evaluator.
        temporal_points = [self._world(stage['query_points'].flatten(1, 2))
                           for stage in result['representation']]
        temporal_logits = [stage['opus_logits'].flatten(1, 2)
                           for stage in result['representation']]
        encoded=result['representation'][-1]; all_features=encoded['query_features']; all_points=encoded['query_points']; stamps=result['query_stamps']; current=stamps==0
        features,positions=all_features[:,current],all_points[:,current].detach(); timestamp=positions.new_zeros(*positions.shape[:-1],1)
        horizon=self.future_steps if not self.training else max(1,min(self.current_epoch-self.finetune_epoch+1,self.future_steps)); pred_points=[]; pred_logits=[]; trajectories=[]
        for step in range(horizon):
            fused=self.ego_attention(ego,features.detach(),positions.detach()); trajectories.append(self.traj(fused))
            future=stamps==step+1; features=torch.cat([features,all_features[:,future]],1); positions=torch.cat([positions,all_points[:,future]],1).detach(); timestamp=torch.cat([timestamp,timestamp.new_full((timestamp.shape[0],future.sum(),self.num_refines,1),.5)],1)
            features=features+fused.expand(-1,features.shape[1],-1)+self.position(torch.cat([positions,timestamp],-1).flatten(2))
            semantic=self.semantic(features).view(features.shape[0],features.shape[1],self.num_refines,17); offset=self.reg(features).view(features.shape[0],features.shape[1],self.num_refines,3)*.5; velocity=self.velocity(features).view(features.shape[0],features.shape[1],self.num_refines,2)
            moving=((semantic.argmax(-1)>=2)&(semantic.argmax(-1)<=10)).unsqueeze(-1); offset=torch.cat([offset[...,:2]+velocity*moving,offset[...,2:]],-1); positions=self._refine(positions,offset).detach()
            pred_points.append(self._world(positions).flatten(1,2)); pred_logits.append(semantic.flatten(1,2))
        # During pre-training only temporal all-query OPUS supervision is active;
        # future occupancy follows the original curriculum from finetune_epoch.
        result.update(future_pred_points=[] if self.pretrain else pred_points, future_pred_logits=[] if self.pretrain else pred_logits,
                      future_predictions=pred_points, future_logits=pred_logits, pred_traj=torch.cat(trajectories,1), future_prediction_coordinate='native_future')
        result.update(temporal_pred_points=temporal_points, temporal_pred_logits=temporal_logits,
                      sparseworld_pretrain=self.pretrain)
        return result
