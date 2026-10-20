"""Five-epoch mini screening; identical backbone/pretraining and early losses."""
_base_ = ['../mini_occ3d_4096_1024/opusv1_gaussian_softplus_learned_bias.py']

cudnn_benchmark = False
seed = 928
loss = _base_.loss
loss.loss_cfgs[0].type = 'CenterSupportDiagnosticSetLoss'
loss.loss_cfgs[0].gt_domain = 'all'
loss.loss_cfgs[0].final_gt_domain = 'camera'
loss.loss_cfgs[0].final_geometry_weight = 1.0
loss.loss_cfgs[0].final_classification_weight = 1.0
