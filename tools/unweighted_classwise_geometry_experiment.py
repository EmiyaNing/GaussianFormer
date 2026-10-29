"""Classwise geometry with uniform class means and the original Set classifier."""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.gaussian_geometry_causal import write,sha,to_cuda,forward,loss_input,seed_all
from tools.classwise_chamfer_experiment import BASELINE_EVAL,REFERENCE_RUN,INITIAL,BASELINE,TrainingFrames

RUN=ROOT/'work_dirs/unweighted_classwise_geometry_mini_v1'
CLASSWISE_RUN=ROOT/'work_dirs/classwise_chamfer_mini_v1'
ARMS=('unweighted_geometry',)

def config(arm='unweighted_geometry'):
    if arm in ('baseline','classwise'):
        from tools.classwise_chamfer_experiment import config as previous
        return previous(arm)
    assert arm=='unweighted_geometry'
    return Config.fromfile(ROOT/'config/mini_unweighted_classwise_geometry/small_unweighted_geometry_1ep.py')

def build(arm='unweighted_geometry',checkpoint=None,training=False):
    from mmseg.models import build_segmentor
    from mmengine.optim import build_optim_wrapper
    from loss import OPENOCC_LOSS
    import model
    c=config(arm);seed_all(c.seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    net=build_segmentor(c.model);net.init_weights()
    state=torch.load(checkpoint or INITIAL,map_location='cpu',weights_only=False)
    net.load_state_dict(state['state_dict'],strict=True);net=net.cuda()
    if not training:net.eval();return c,net
    opt=build_optim_wrapper(net,c.optimizer)
    if 'field_microbatches' in state:opt.load_state_dict(state['optimizer'])
    objective=OPENOCC_LOSS.build(c.loss).cuda();net.train()
    return c,net,opt,objective,state

def protocol_files():
    from tools.classwise_chamfer_experiment import protocol_files as previous
    extra=['analysis/unweighted_classwise_geometry_mini_design.md',
        'config/mini_unweighted_classwise_geometry/small_unweighted_geometry_1ep.py',
        'tools/unweighted_classwise_geometry_experiment.py',
        'tools/check_unweighted_classwise_geometry.py',
        'tools/train_unweighted_classwise_geometry_mini.py',
        'tools/eval_unweighted_classwise_geometry_mini.py',
        'tools/report_unweighted_classwise_geometry_mini.py',
        'tools/run_unweighted_classwise_geometry_mini.py']
    return list(dict.fromkeys(previous()+[ROOT/name for name in extra]))

def assert_protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,f'Frozen source changed: {name}'
    for name,digest in p['controls'].items():assert sha(ROOT/name)==digest,f'Control changed: {name}'
    return p
