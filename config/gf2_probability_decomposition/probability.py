"""Five-epoch OPUS with official GF2 probability rendering; Set unchanged."""
_base_ = ['../center_support_contribution1/a0_original.py']
custom_imports = dict(imports=['model.head.opus_gf2_probability_head',
    'loss.gf2_probability_experiment_loss'], allow_failed_imports=False)
model = dict(head=dict(type='OPUSGF2ProbabilityHead'))
loss = dict(_delete_=True, type='GF2ProbabilityExperimentLoss',
    trace_dir='work_dirs/gf2_probability_decomposition/train/probability')
