"""Use the native five-epoch runner with source-specific gradient interventions."""
import json
import hashlib
import copy
from pathlib import Path
import torch
import numpy as np
from . import OPENOCC_LOSS
from .center_support_diagnostic_loss import CenterSupportDiagnosticSetLoss


@OPENOCC_LOSS.register_module()
class EmptyHistoryPathLoss(torch.nn.Module):
    def __init__(self,condition='B0',trace_dir=None):
        super().__init__()
        from mmengine import Config
        from tools.empty_mechanism_stage1 import modules,CONFIG
        cfg=Config.fromfile(CONFIG)
        self.set_loss=OPENOCC_LOSS.build(cfg.loss.loss_cfgs[0])
        mods=modules(cfg)
        self.ce=mods[1];self.lovasz=mods[2]
        self.mods=(None,self.ce,self.lovasz)
        self.condition=condition
        self.trace_dir=Path(trace_dir) if trace_dir else None

    def forward(self,inputs):
        from tools.empty_mechanism_stage3 import loss_with_intervention,set_negative_gradients,suppressive_part
        output=dict(inputs);output['occ_cam_mask']=inputs['occ_mask']
        training=torch.is_grad_enabled() and inputs['opus_pred_logits'][-1].requires_grad
        condition=self.condition if training else 'B0'
        # Only classification outputs that actually feed occupancy rendering
        # instantiate the proposed class/occupancy shared-amplitude mechanism.
        base_condition='E' if condition in ['E','SE'] else 'B0'
        loss,audit=loss_with_intervention(output,inputs['metas'],self.set_loss,self.mods,base_condition)
        if condition in ['S','SE']:
            for stage in output['supervised_stage_indices']:
                # Preserve exact original stage/domain/weight while avoiding
                # five unused nearest-neighbour source computations.
                proxy=copy.copy(self.set_loss)
                proxy.stage_weights=(self.set_loss.stage_weights[stage],)
                last=stage==len(self.set_loss.stage_weights)-1
                domain=self.set_loss.final_gt_domain if last else self.set_loss.gt_domain
                proxy.gt_domain=domain;proxy.final_gt_domain=domain
                proxy.final_classification_weight=self.set_loss.final_classification_weight if last else 1.
                selected={key:[output[key][stage]] for key in
                    ['opus_pred_logits','opus_pred_points','opus_point_valid_masks']}
                gradient=set_negative_gradients(selected,inputs['metas'],proxy)[0]
                raw=output['opus_pred_logits'][stage]
                removed,source_audit=suppressive_part(raw,gradient)
                correction=-(raw*removed).sum();loss=loss+(correction-correction.detach())
                audit['audits'].append(dict(source='S',stage=stage,**source_audit))
        if training and self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True,exist_ok=True)
            iteration=int(inputs['global_iter'])
            row=dict(global_iter=iteration,loss=audit['base'],sources={},rendered_stages=output['supervised_stage_indices'])
            # Record actual paired inputs without changing the loader or RNG.
            fingerprint=hashlib.sha256()
            for key in ['projection_mat','occ_label']:
                value=inputs['metas'][key].detach().cpu().contiguous().numpy()
                fingerprint.update(key.encode());fingerprint.update(value.tobytes())
            row['data_fingerprint']=fingerprint.hexdigest()
            for source in ['S','E']:
                a=[v for v in audit['audits'] if v['source']==source]
                if a:row['sources'][source]={key:sum(v[key] for v in a)/len(a) for key in
                       ['source_norm','removed_norm','suppressive_derivative','removed_derivative','removed_fraction']}
            if iteration==0:
                raw=inputs['opus_pred_logits'][-1].detach().cpu().numpy()
                row['initial_raw_sha256']=hashlib.sha256(raw.tobytes()).hexdigest()
                row['initial_raw_shape']=list(raw.shape)
                np.savez_compressed(self.trace_dir/'initial_forward.npz',raw=raw,
                    labels=inputs['metas']['occ_label'].detach().cpu().numpy())
            with (self.trace_dir/'microbatch_trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return loss,{'loss_total':float(loss.detach())}
