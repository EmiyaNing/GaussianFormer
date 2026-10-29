"""Matched classwise-CD experiment; read-only historical controls."""
from pathlib import Path
import hashlib,json,random,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.gaussian_geometry_causal import write,sha,to_cuda,forward,loss_input,seed_all
RUN=ROOT/'work_dirs/classwise_chamfer_mini_v1'
BASELINE_EVAL=ROOT/'work_dirs/semantic_neighborhood_mean_mini_v1/eval/baseline'
REFERENCE_RUN=ROOT/'work_dirs/semantic_field_mini_v1'
INITIAL=ROOT/'work_dirs/small_camera_set_full_1ep/epoch_1.pth'
BASELINE=REFERENCE_RUN/'train/baseline/epoch_1.pth'
ARMS=('classwise',)

def config(arm):
    path=(ROOT/'config/mini_semantic_field/small_baseline_1ep.py' if arm=='baseline' else
          ROOT/'config/mini_classwise_chamfer/small_classwise_cd_1ep.py')
    return Config.fromfile(path)

def build(arm,checkpoint=None,training=False):
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
    objective=OPENOCC_LOSS.build(c.loss).cuda();net.train();return c,net,opt,objective,state

class TrainingFrames:
    def __init__(self,dataset,seed,include_field=True):self.dataset,self.seed=dataset,seed
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        np.random.seed(self.seed*1000000+index);random.seed(self.seed*1000000+index)
        return self.dataset[index]

def protocol_files():
    from tools.semantic_neighborhood_experiment import protocol_files as previous
    paths=previous()+[ROOT/'loss/classwise_chamfer_set_loss.py',
        ROOT/'tools/classwise_chamfer_experiment.py', ROOT/'tools/train_classwise_chamfer_mini.py',
        ROOT/'tools/eval_classwise_chamfer_mini.py',ROOT/'tools/check_classwise_chamfer.py',
        ROOT/'tools/pilot_classwise_chamfer.py',ROOT/'tools/report_classwise_chamfer_mini.py',
        ROOT/'tools/run_classwise_chamfer_mini.py',
        ROOT/'config/mini_classwise_chamfer/small_classwise_cd_1ep.py']
    return list(dict.fromkeys(paths))

def assert_protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,f'Frozen source changed: {name}'
    assert sha(INITIAL)==p['initial_sha256'] and sha(BASELINE)==p['baseline_sha256']
    for name,h in p.get('baseline_artifacts',{}).items():assert sha(ROOT/name)==h,f'Baseline artifact changed: {name}'
    return p
