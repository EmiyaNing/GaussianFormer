"""Raw logits: literal API diagnostic plus fixed-variance Gaussian KL control."""
_base_=['./sparseworld_fast_motion_occupancy_kd_repaired.py']
fast_root='out/sparseworld_fast_motion_distillation/20261009_raw_logits_kl_2048x512'
fast_raw_logits=True
fast_gaussian_sigma=1.
fast_arm_policies=dict(frozen_raw_api_kd='raw_api_log_target',full_raw_api_kd='raw_api_log_target',
                       frozen_gaussian_kd='gaussian_raw',full_gaussian_kd='gaussian_raw')
fast_arms=['frozen_raw_api_kd','frozen_gaussian_kd','full_raw_api_kd','full_gaussian_kd']
fast_matched_task_root='out/sparseworld_fast_motion_distillation/20261009_repaired_2048x512'
