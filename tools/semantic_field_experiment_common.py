"""Shared deterministic input/build helpers for the section-4.1 experiment."""
from pathlib import Path
import hashlib
import json
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from loss.semantic_gaussian_field.data import cached, construct
from tools.gaussian_geometry_causal import write,sha,to_cuda,forward,loss_input,seed_all

RUN=ROOT/'work_dirs/semantic_field_mini_v1'
INITIAL=ROOT/'work_dirs/small_camera_set_full_1ep/epoch_1.pth'
ARMS=('baseline','geometry','semantic')


def config(arm):
    return Config.fromfile(ROOT/f'config/mini_semantic_field/small_{arm}_1ep.py')


def build(arm,checkpoint=None,training=False):
    from mmseg.models import build_segmentor
    from mmengine.optim import build_optim_wrapper
    from loss import OPENOCC_LOSS
    import model
    c=config(arm)
    seed_all(c.seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    net=build_segmentor(c.model);net.init_weights()
    state=torch.load(checkpoint or INITIAL,map_location='cpu',weights_only=False)
    net.load_state_dict(state['state_dict'],strict=True)
    net=net.cuda()
    if not training:
        net.eval();return c,net
    opt=build_optim_wrapper(net,c.optimizer)
    # Fresh optimizer for every arm. Only resume this experiment's own state.
    if 'field_microbatches' in state:opt.load_state_dict(state['optimizer'])
    objective=OPENOCC_LOSS.build(c.loss).cuda()
    net.train()
    return c,net,opt,objective,state


class TrainingFrames:
    def __init__(self,dataset,seed,include_field=True):
        self.dataset,self.seed,self.include_field=dataset,seed,include_field
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        np.random.seed(self.seed*1000000+index);random.seed(self.seed*1000000+index)
        row=self.dataset[index]
        if self.include_field:
            distance,components=cached(np.asarray(row['occ_label']),np.asarray(row['occ_cam_mask']),RUN/'gt_cache')
            row.update(field_distance=distance,field_components=components,
                field_seed=np.array(self.seed*1000000+index,dtype=np.int64))
        return row


def protocol_files():
    paths=[ROOT/'tools/run_semantic_field_experiment.py',ROOT/'tools/semantic_field_experiment_common.py',
        ROOT/'tools/train_semantic_field_mini.py',ROOT/'tools/eval_semantic_field_mini.py',
        ROOT/'tools/report_semantic_field_mini.py',ROOT/'tools/check_semantic_gaussian_field.py',
        ROOT/'tools/pilot_semantic_field_mini.py',ROOT/'tools/train_dcd_mini.py',
        ROOT/'tools/gaussian_geometry_causal.py',ROOT/'tools/probe_gaussian_scale_supervision.py',
        ROOT/'tools/report_gaussian_geometry_causal.py',ROOT/'model/utils/utils.py',
        ROOT/'loss/semantic_gaussian_field_loss.py',ROOT/'loss/opus_set_loss.py',
        ROOT/'loss/center_support_diagnostic_loss.py',ROOT/'model/encoder/opus_encoder.py',
        ROOT/'model/encoder/opus_v1_gaussian_temporal_encoder.py',
        ROOT/'model/head/gaussian_opus_head.py',ROOT/'tools/boundary_gaussian_stats/data.py']
    paths+=list((ROOT/'loss/semantic_gaussian_field').glob('*.py'))
    paths+=list((ROOT/'loss/semantic_gaussian_field').glob('*.cu'))
    paths+=list((ROOT/'config/mini_semantic_field').glob('*.py'))
    paths+=list((ROOT/'tools/semantic_field_support').glob('*.py'))
    paths+=list((ROOT/'tools/semantic_field_support').glob('*.cu'))
    paths+=list((ROOT/'tools/query_final_support').glob('*.py'))
    paths+=list((ROOT/'tools/query_final_support').glob('*.cu'))
    return paths


def assert_protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,f'Frozen source changed: {name}'
    assert sha(INITIAL)==p['initial_sha256']
    return p
