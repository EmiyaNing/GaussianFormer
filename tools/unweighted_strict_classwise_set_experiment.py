"""Unweighted variant of the completed strict-classwise OPUS mini experiment."""
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

RUN = ROOT / 'work_dirs/unweighted_strict_classwise_set_mini_v1'
STRICT_RUN = ROOT / 'work_dirs/strict_classwise_set_mini_v1'
CLASSWISE_RUN = ROOT / 'work_dirs/classwise_chamfer_mini_v1'
ARMS = ('unweighted_classwise',)


def config(arm='unweighted_classwise'):
    if arm in ('baseline', 'classwise', 'strict_classwise'):
        from tools.strict_classwise_set_experiment import config as old_config
        return old_config(arm)
    if arm != 'unweighted_classwise':
        raise ValueError(arm)
    return Config.fromfile(ROOT / 'config/mini_unweighted_strict_classwise_set/'
                           'small_unweighted_strict_classwise_1ep.py')


def build(arm='unweighted_classwise', checkpoint=None, training=False):
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
    from tools.strict_classwise_set_experiment import protocol_files as strict_files
    extra = [
        ROOT / 'analysis/unweighted_strict_classwise_set_mini_design.md',
        ROOT / 'config/mini_unweighted_strict_classwise_set/small_unweighted_strict_classwise_1ep.py',
        ROOT / 'tools/unweighted_strict_classwise_set_experiment.py',
        ROOT / 'tools/check_unweighted_strict_classwise_set.py',
        ROOT / 'tools/train_unweighted_strict_classwise_set_mini.py',
        ROOT / 'tools/eval_unweighted_strict_classwise_set_mini.py',
        ROOT / 'tools/report_unweighted_strict_classwise_set_mini.py',
        ROOT / 'tools/run_unweighted_strict_classwise_set_mini.py',
    ]
    return list(dict.fromkeys(strict_files() + extra))


def assert_protocol():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    for name, digest in protocol['sources'].items():
        assert sha(ROOT / name) == digest, f'Frozen source changed: {name}'
    for name, digest in protocol['controls'].items():
        assert sha(ROOT / name) == digest, f'Control changed: {name}'
    return protocol
