"""Strict, self-contained SparseWorld trajectory segmentor."""
import torch
import torch.nn.functional as F
from torch import nn
from mmseg.models import SEGMENTORS
from .opus_segmentor import OPUSSegmentor


class _EgoAttention(nn.Module):
    def __init__(self, dims, heads, dropout, pc_range):
        super().__init__(); self.attention=nn.MultiheadAttention(dims,heads,dropout,batch_first=True); self.tau=nn.Linear(dims,heads); self.position=nn.Linear(3,dims); self.pc_range=tuple(pc_range)
        nn.init.zeros_(self.tau.weight); nn.init.uniform_(self.tau.bias,0.,2.)
    def forward(self, ego_feature, query_feature, query_points,
                distance_points=None, return_distance_prior=False):
        if distance_points is None:
            distance_points = query_points
        lower=query_points.new_tensor(self.pc_range[:3]); extent=query_points.new_tensor(self.pc_range[3:])-lower
        world=query_points.mean(2)*extent+lower
        distance_world=distance_points.mean(2)*extent+lower
        # Match upstream OPUSCrossAttention exactly: the ego query is located
        # at the normalized ROI centre, while keys use decoded metric centres.
        ego_point=query_points.new_full((query_points.shape[0],1,3),.5)
        ego_world=ego_point*extent+lower
        distance=-torch.cdist(ego_world,distance_world)
        tau=F.softplus(self.tau(query_feature)).transpose(1,2)
        mask=(torch.exp(distance/40.)[:,None]*tau[:,:,None]).flatten(0,1)
        query=ego_feature+self.position(torch.zeros_like(ego_point))
        key=query_feature+self.position(world)
        attended=self.attention(query,key,query_feature,attn_mask=mask,need_weights=False)[0]
        # MMCV MultiheadAttention used upstream adds the identity internally.
        output = ego_feature+attended
        if return_distance_prior:
            # This is the normalized geometry prior supplied as MHA's additive
            # mask, not the full content-dependent attention probability.
            return output, mask.view(query_points.shape[0], -1, query_points.shape[1]).softmax(-1)
        return output


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

    @staticmethod
    def _clone_tensor(value):
        return value.detach().clone() if isinstance(value, torch.Tensor) else value

    def _snapshot_forecast_state(self, state, completed_steps):
        """Clone all state needed to resume before the next interaction."""
        tensor_keys = ('features', 'positions', 'timestamp', 'ego',
                       'all_features', 'all_points', 'stamps', 'slot_ids')
        tensor_keys += tuple(key for key in ('semantic',) if key in state)
        snapshot = {key: self._clone_tensor(state[key]) for key in tensor_keys}
        snapshot['completed_steps'] = int(completed_steps)
        snapshot['active_mask'] = (snapshot['stamps'] <= completed_steps)
        snapshot['positions_m'] = self._world(snapshot['positions'])
        snapshot['tensor_metadata'] = {
            key: dict(shape=list(state[key].shape), dtype=str(state[key].dtype),
                      device=str(state[key].device),
                      requires_grad=bool(state[key].requires_grad))
            for key in tensor_keys
        }
        snapshot['cpu_rng_state'] = torch.get_rng_state().clone()
        if state['positions'].is_cuda:
            device = state['positions'].device
            snapshot['cuda_device'] = device.index
            snapshot['cuda_rng_state'] = torch.cuda.get_rng_state(device).clone()
        return snapshot

    def _forecast_rollout(self, state, horizon, snapshot_after_step=None,
                          distance_positions=None, encoding_positions=None,
                          fused_ego_features=None,
                          collect_trace=False):
        features, positions = state['features'], state['positions']
        timestamp, ego = state['timestamp'], state['ego']
        all_features, all_points = state['all_features'], state['all_points']
        stamps, slot_ids = state['stamps'], state['slot_ids']
        start_step = int(state.get('completed_steps', 0))
        pred_points, pred_logits, trajectories = [], [], []
        requested = (() if snapshot_after_step is None else
                     ((snapshot_after_step,) if isinstance(snapshot_after_step, int)
                      else tuple(snapshot_after_step)))
        snapshots = {}
        trace = []
        for step in range(start_step, horizon):
            clean_distance = (None if distance_positions is None else
                              distance_positions[step - start_step])
            if collect_trace:
                raw_fused, distance_prior = self.ego_attention(
                    ego, features.detach(), positions.detach(),
                    distance_points=clean_distance, return_distance_prior=True)
                fused = raw_fused
                step_trace = dict(
                    input_positions=positions.detach().clone(),
                    input_features=features.detach().clone(),
                    distance_prior=distance_prior.detach().clone(),
                    raw_fused_ego=raw_fused.detach().clone())
            else:
                fused=self.ego_attention(
                    ego,features.detach(),positions.detach(),
                    distance_points=clean_distance)
            if fused_ego_features is not None:
                fused = fused_ego_features[step - start_step]
                if fused.shape != ego.shape:
                    raise ValueError('clean fused ego state has incompatible shape')
            if collect_trace:
                step_trace['used_fused_ego'] = fused.detach().clone()
            trajectories.append(self.traj(fused))
            future=stamps==step+1
            features=torch.cat([features,all_features[:,future]],1)
            positions=torch.cat([positions,all_points[:,future]],1).detach()
            timestamp=torch.cat([timestamp,timestamp.new_full((timestamp.shape[0],future.sum(),self.num_refines,1),.5)],1)
            slot_ids=torch.cat([slot_ids,torch.nonzero(future,as_tuple=False).flatten()])
            if collect_trace:
                step_trace['interaction_positions'] = positions.detach().clone()
            position_input = positions
            if encoding_positions is not None:
                position_input = encoding_positions[step - start_step]
                if position_input.shape[1] != positions.shape[1]:
                    raise ValueError('clean position-encoding state has incompatible query count')
            features=features+fused.expand(-1,features.shape[1],-1)+self.position(torch.cat([position_input,timestamp],-1).flatten(2))
            semantic=self.semantic(features).view(features.shape[0],features.shape[1],self.num_refines,17)
            offset=self.reg(features).view(features.shape[0],features.shape[1],self.num_refines,3)*.5
            velocity=self.velocity(features).view(features.shape[0],features.shape[1],self.num_refines,2)
            moving=((semantic.argmax(-1)>=2)&(semantic.argmax(-1)<=10)).unsqueeze(-1)
            offset=torch.cat([offset[...,:2]+velocity*moving,offset[...,2:]],-1)
            positions=self._refine(positions,offset).detach()
            pred_points.append(self._world(positions).flatten(1,2))
            pred_logits.append(semantic.flatten(1,2))
            if collect_trace:
                step_trace.update(
                    output_positions=positions.detach().clone(),
                    output_features=features.detach().clone())
                trace.append(step_trace)
            if step + 1 in requested:
                snapshots[step + 1] = self._snapshot_forecast_state(dict(
                    features=features, positions=positions,
                    timestamp=timestamp, ego=ego,
                    all_features=all_features, all_points=all_points,
                    stamps=stamps, slot_ids=slot_ids,
                    semantic=semantic), step + 1)
        return dict(points=pred_points, logits=pred_logits,
                    trajectories=trajectories,
                    snapshot=(snapshots.get(requested[0]) if len(requested) == 1 else None),
                    snapshots=snapshots, trace=trace)

    def _replay_forecast(self, snapshot, horizon, **kwargs):
        devices = []
        if snapshot['positions'].is_cuda:
            devices = [snapshot['cuda_device']]
        # fork_rng restores the caller's RNG state after replay. This makes the
        # diagnostic path observational even if stochastic modules are added.
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.set_rng_state(snapshot['cpu_rng_state'])
            if devices:
                torch.cuda.set_rng_state(
                    snapshot['cuda_rng_state'], device=devices[0])
            return self._forecast_rollout(snapshot, horizon, **kwargs)

    def forward(self, imgs=None, metas=None, points=None, **kwargs):
        snapshot_after_step = kwargs.pop('forecast_snapshot_step', None)
        zero_replay = bool(kwargs.pop('forecast_zero_replay', False))
        if zero_replay and snapshot_after_step is None:
            raise ValueError('forecast_zero_replay requires forecast_snapshot_step')
        if zero_replay and self.training:
            raise RuntimeError('zero-perturbation replay is an evaluation diagnostic')
        result=dict(imgs=imgs,metas=metas,points=points); result.update(kwargs); result.update(self.extract_img_feat(**result)); result.update(self.lifter(**result))
        state=metas['temporal_ego_states'].to(result['query_points']).reshape(imgs.shape[0],-1); ego=self.plan_head(state).unsqueeze(1)
        # Original range-adaptive perception scales the transformer initial anchors.
        scale=(torch.tanh(self.points_scale(ego))+1.)*.35+.8; result['query_points']=result['query_points']*scale
        result.update(self.encoder(**result)); result.update(self.head(**result))
        # Preserve the common Occ3D evaluator contract.  The strict head owns
        # prediction rendering only; labels and visibility masks remain
        # immutable metadata and are forwarded here without reshaping away the
        # batch dimension.
        result.update(
            sampled_label=metas['occ_label'].flatten(1),
            sampled_xyz=(None if metas.get('occ_xyz') is None else
                         metas['occ_xyz'].flatten(1, -2)),
            occ_cam_mask=metas.get('occ_cam_mask'),
            occ_lidar_mask=metas.get('occ_lidar_mask'),
            occ_loss_mask=metas.get('occ_loss_mask'))
        # Keep full timestamped decoder predictions for the source temporal
        # pretraining objective. The public head intentionally exposes only
        # stamp-0 points to the ordinary OPUS loss/evaluator.
        temporal_points = [self._world(stage['query_points'].flatten(1, 2))
                           for stage in result['representation']]
        temporal_logits = [stage['opus_logits'].flatten(1, 2)
                           for stage in result['representation']]
        encoded=result['representation'][-1]; all_features=encoded['query_features']; all_points=encoded['query_points']; stamps=result['query_stamps']; current=stamps==0
        features,positions=all_features[:,current],all_points[:,current].detach(); timestamp=positions.new_zeros(*positions.shape[:-1],1)
        horizon=self.future_steps if not self.training else max(1,min(self.current_epoch-self.finetune_epoch+1,self.future_steps))
        rollout=self._forecast_rollout(dict(
            features=features, positions=positions, timestamp=timestamp,
            ego=ego, all_features=all_features, all_points=all_points,
            stamps=stamps,
            slot_ids=torch.nonzero(current,as_tuple=False).flatten(),
            completed_steps=0), horizon, snapshot_after_step)
        pred_points, pred_logits, trajectories = \
            rollout['points'], rollout['logits'], rollout['trajectories']
        # During pre-training only temporal all-query OPUS supervision is active;
        # future occupancy follows the original curriculum from finetune_epoch.
        result.update(future_pred_points=[] if self.pretrain else pred_points, future_pred_logits=[] if self.pretrain else pred_logits,
                      future_predictions=pred_points, future_logits=pred_logits, pred_traj=torch.cat(trajectories,1), future_prediction_coordinate='native_future')
        result.update(temporal_pred_points=temporal_points, temporal_pred_logits=temporal_logits,
                      sparseworld_pretrain=self.pretrain)
        if rollout['snapshot'] is not None:
            result['forecast_snapshot'] = rollout['snapshot']
            if zero_replay:
                result['forecast_zero_replay'] = self._replay_forecast(
                    rollout['snapshot'], horizon)
        if rollout['snapshots']:
            result['forecast_snapshots'] = rollout['snapshots']
        return result
