"""A1: final-stage nearest-GT softmax CE, focal-matched logit gradient norm."""
_base_ = ['../gf2_probability_decomposition/probability.py']
custom_imports = dict(imports=['model.head.opus_gf2_probability_head',
    'loss.gaussian_responsibility_experiment_loss'], allow_failed_imports=False)
loss_input_convertion = dict(gaussian='gaussian')
loss = dict(_delete_=True, type='GaussianResponsibilityExperimentLoss', assignment='nearest',
    trace_dir='work_dirs/gaussian_responsibility/train/A1')
