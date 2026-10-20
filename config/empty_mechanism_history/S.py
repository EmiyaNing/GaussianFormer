"""Five-epoch from-start source-path intervention: S."""
_base_ = ['../center_support_contribution1/a0_original.py']
custom_imports = dict(imports=['loss.empty_history_path_loss'], allow_failed_imports=False)
loss = dict(_delete_=True, type='EmptyHistoryPathLoss',condition='S',trace_dir='work_dirs/empty_mechanism_history/train/S')
loss_input_convertion = dict(gaussians='gaussians',supervised_stage_indices='supervised_stage_indices')
