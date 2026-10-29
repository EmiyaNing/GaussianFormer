"""Matched final-stage bidirectional-loss experiment with immutable historical controls."""
from pathlib import Path
import json,random,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.gaussian_geometry_causal import write,sha,to_cuda,forward,loss_input,seed_all

RUN=ROOT/'work_dirs/bidirectional_coverage_mini_v1'
INITIAL=ROOT/'work_dirs/small_camera_set_full_1ep/epoch_1.pth'
REFERENCE_RUN=ROOT/'work_dirs/semantic_field_mini_v1'
BASELINE=REFERENCE_RUN/'train/baseline/epoch_1.pth'
ARMS=('union','decision')
CONTROL_FOLDERS={
    'baseline':ROOT/'work_dirs/semantic_neighborhood_mean_mini_v1/eval/baseline',
    'classwise':ROOT/'work_dirs/classwise_chamfer_mini_v1/eval/classwise',
    'strict_classwise':ROOT/'work_dirs/strict_classwise_set_mini_v1/eval/strict_classwise',
    'unweighted_classwise':ROOT/'work_dirs/unweighted_strict_classwise_set_mini_v1/eval/unweighted_classwise',
    'unweighted_geometry':ROOT/'work_dirs/unweighted_classwise_geometry_mini_v1/eval/unweighted_geometry',
    'neighborhood':ROOT/'work_dirs/semantic_neighborhood_mean_mini_v1/eval/mean3_uncovered1',
    'field_geometry':ROOT/'work_dirs/semantic_field_mini_v1/eval/geometry',
    'field_semantic':ROOT/'work_dirs/semantic_field_mini_v1/eval/semantic',
}
CONTROL_CHECKPOINTS={
    'baseline':BASELINE,
    'classwise':ROOT/'work_dirs/classwise_chamfer_mini_v1/train/classwise/epoch_1.pth',
    'strict_classwise':ROOT/'work_dirs/strict_classwise_set_mini_v1/train/strict_classwise/epoch_1.pth',
    'unweighted_classwise':ROOT/'work_dirs/unweighted_strict_classwise_set_mini_v1/train/unweighted_classwise/epoch_1.pth',
    'unweighted_geometry':ROOT/'work_dirs/unweighted_classwise_geometry_mini_v1/train/unweighted_geometry/epoch_1.pth',
    'neighborhood':ROOT/'work_dirs/semantic_neighborhood_mean_mini_v1/train/mean3_uncovered1/epoch_1.pth',
    'field_geometry':ROOT/'work_dirs/semantic_field_mini_v1/train/geometry/epoch_1.pth',
    'field_semantic':ROOT/'work_dirs/semantic_field_mini_v1/train/semantic/epoch_1.pth',
}

def config(arm):
    path=ROOT/'config/mini_semantic_field/small_baseline_1ep.py' if arm=='baseline' else ROOT/f'config/mini_bidirectional_coverage/small_{arm}_1ep.py'
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
        row=self.dataset[index];row['union_seed']=np.array(self.seed*1000000+index,dtype=np.int64)
        return row

def protocol_files():
    from tools.classwise_chamfer_experiment import protocol_files as previous
    extra=['analysis/bidirectional_union_coverage_loss_design.md',
        'analysis/bidirectional_coverage_precision_revision.md',
        'analysis/bidirectional_coverage_mini_design.md',
        'loss/bidirectional_union_coverage_loss.py','tools/bidirectional_coverage_experiment.py',
        'tools/check_bidirectional_union_coverage.py','tools/pilot_bidirectional_coverage_mini.py',
        'tools/train_bidirectional_coverage_mini.py','tools/eval_bidirectional_coverage_mini.py',
        'tools/run_bidirectional_coverage_mini.py',
        'model/encoder/opus_v1_gaussian_encoder.py','model/encoder/opus_v1_gaussian_softplus_encoder.py',
        'model/head/gaussian_head.py',
        'model/head/opus_v1_gaussian_head.py','model/head/opus_v1_gaussian_trainable_bias_head.py',
        'model/head/localagg/local_aggregate/__init__.py','model/head/localagg/src/forward.cu',
        'model/head/localagg/src/backward.cu']
    return list(dict.fromkeys(previous()+[ROOT/p for p in extra]+
        list((ROOT/'loss/bidirectional_union_coverage').glob('*.py'))+
        list((ROOT/'loss/bidirectional_union_coverage').glob('*.cu'))+
        list((ROOT/'model/head/localagg').glob('*.cu'))+
        list((ROOT/'model/head/localagg/src').glob('*.cu'))+
        list((ROOT/'model/head/localagg/src').glob('*.h'))+
        list((ROOT/'config/mini_bidirectional_coverage').glob('*.py'))))

def assert_protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,f'Frozen source changed: {name}'
    for name,digest in p['controls'].items():assert sha(ROOT/name)==digest,f'Control changed: {name}'
    return p
