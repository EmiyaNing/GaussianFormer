"""Matched strict classwise classification + geometry experiment."""
from pathlib import Path
import hashlib, json, random, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.gaussian_geometry_causal import write, sha, to_cuda, forward, loss_input, seed_all
from tools.classwise_chamfer_experiment import (
    BASELINE_EVAL, REFERENCE_RUN, INITIAL, BASELINE, TrainingFrames)

RUN = ROOT / 'work_dirs/strict_classwise_set_mini_v1'
PREVIOUS_RUN = ROOT / 'work_dirs/classwise_chamfer_mini_v1'
PREVIOUS = PREVIOUS_RUN / 'train/classwise/epoch_1.pth'
ARMS = ('strict_classwise',)


def config(arm):
    paths = {
        'baseline': 'config/mini_semantic_field/small_baseline_1ep.py',
        'classwise': 'config/mini_classwise_chamfer/small_classwise_cd_1ep.py',
        'strict_classwise': 'config/mini_strict_classwise_set/small_strict_classwise_1ep.py',
    }
    return Config.fromfile(ROOT / paths[arm])


def build(arm, checkpoint=None, training=False):
    from mmseg.models import build_segmentor
    from mmengine.optim import build_optim_wrapper
    from loss import OPENOCC_LOSS
    import model
    c = config(arm)
    seed_all(c.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    net = build_segmentor(c.model)
    net.init_weights()
    state = torch.load(checkpoint or INITIAL, map_location='cpu', weights_only=False)
    net.load_state_dict(state['state_dict'], strict=True)
    net = net.cuda()
    if not training:
        net.eval()
        return c, net
    opt = build_optim_wrapper(net, c.optimizer)
    if 'field_microbatches' in state:
        opt.load_state_dict(state['optimizer'])
    objective = OPENOCC_LOSS.build(c.loss).cuda()
    net.train()
    return c, net, opt, objective, state


def protocol_files():
    from tools.classwise_chamfer_experiment import protocol_files as previous
    names = ['loss/strict_classwise_set_loss.py',
             'config/mini_strict_classwise_set/small_strict_classwise_1ep.py',
             'tools/strict_classwise_set_experiment.py',
             'tools/check_strict_classwise_set.py',
             'tools/pilot_strict_classwise_set.py',
             'tools/train_strict_classwise_set_mini.py',
             'tools/eval_strict_classwise_set_mini.py',
             'tools/report_strict_classwise_set_mini.py',
             'tools/run_strict_classwise_set_mini.py']
    return list(dict.fromkeys(previous() + [ROOT/n for n in names]))


def assert_protocol():
    protocol = json.loads((RUN/'protocol.json').read_text())
    for name, digest in protocol['sources'].items():
        assert sha(ROOT/name) == digest, f'Frozen source changed: {name}'
    for name, digest in protocol['controls'].items():
        assert sha(ROOT/name) == digest, f'Control changed: {name}'
    return protocol
