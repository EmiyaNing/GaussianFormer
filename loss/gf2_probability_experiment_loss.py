"""Unchanged Set objective, probability NLL/Lovasz, and paired-input tracing."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mmengine import Config
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class GF2ProbabilityExperimentLoss(torch.nn.Module):
    def __init__(self, trace_dir):
        super().__init__()
        cfg = Config.fromfile('config/center_support_contribution1/a0_original.py')
        setting = cfg.loss.to_dict()
        setting['loss_cfgs'][1]['lovasz_use_softmax'] = False
        self.objective = OPENOCC_LOSS.build(setting)
        self.trace_dir = Path(trace_dir)

    def forward(self, inputs):
        loss, details = self.objective(inputs)
        training = torch.is_grad_enabled() and inputs['opus_pred_logits'][-1].requires_grad
        if training:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            index = int(inputs['global_iter'])
            fingerprint = hashlib.sha256()
            for key in ['projection_mat', 'occ_label']:
                fingerprint.update(key.encode())
                fingerprint.update(inputs['metas'][key].detach().cpu().contiguous().numpy().tobytes())
            row = dict(global_iter=index, loss=float(loss.detach()), data_fingerprint=fingerprint.hexdigest())
            if index == 0:
                raw = inputs['opus_pred_logits'][-1].detach().cpu().numpy()
                row['initial_raw_sha256'] = hashlib.sha256(raw.tobytes()).hexdigest()
                np.savez_compressed(self.trace_dir/'initial_forward.npz', raw=raw,
                    labels=inputs['metas']['occ_label'].detach().cpu().numpy())
            with (self.trace_dir/'microbatch_trace.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        return loss, details
