"""Controlled future-perception/forecast feature comparison.

This file supplies the data pipeline and experiment hyperparameters.  The
runner *always* builds the unmodified SparseWorldStrictSegmentor from
``student_model_config``; the inherited teacher-aligned ``model`` field is
intentionally not used.
"""

_base_ = ['./sparseworld_teacher_aligned_r50_704x256_6h.py']

student_model_config = 'config/opus/sparseworld_strict_r50_704x256_5f_occ3d.py'
student_checkpoint = 'out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth'
student_checkpoint_sha256 = 'fb9f955f15362cb23244fd551145e499b6a28b1a46e45849b129d9b60a879828'

# Shared fixed-point/decoder probe.  The adapter consumes query features from
# the *same frozen SparseWorld perception encoder* for the teacher target and
# original SparseWorld future-rollout features for the student.
probe_horizons = [2, 4, 6]
probe_readout_hidden = 256
probe_teacher_neighbor_count = 4
probe_match_radius_m = 4.0
probe_feature_weight = 0.1
probe_lr = 2e-5
probe_readout_lr = 1e-4
probe_readout_updates = 512
probe_forecast_updates = 2048
probe_train_samples = 2048
probe_val_samples = 512
probe_sample_seed = 20260926
probe_train_seeds = [20260926, 20260927, 20260928]
probe_workers = 4
probe_oracle_margin_pp = 0.2
probe_success_margin_pp = 0.5
