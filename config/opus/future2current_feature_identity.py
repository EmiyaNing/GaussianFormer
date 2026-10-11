"""Full-size same-feature versus learned-adapter diagnostic."""
_base_ = ['./future2current_shared_readout.py']

identity_train_samples = 2048
identity_val_samples = 512
identity_sample_seed = 20260926
identity_train_seeds = [20260926, 20260927, 20260928]
identity_horizons = [2, 4, 6]
identity_workers = 4
identity_readout_updates = 2048
identity_forecast_updates = 2048
identity_max_prediction_points = 4096
identity_readout_lr = 1e-4
identity_forecast_lr = 2e-5
identity_adapter_lr = 1e-4
identity_feature_weight = .1
identity_semantic_weight = .1
identity_semantic_temperature = 2.0
identity_teacher_neighbor_count = 4
identity_match_radius_m = 4.0
identity_arms = [
    'task_only', 'raw_align', 'raw_shuffled',
    'adapter_task', 'adapter_align', 'adapter_shuffled',
    'semantic_kd_3s', 'semantic_kd_3s_shuffled',
    'raw_shuffled_matched', 'adapter_shuffled_matched',
    'semantic_kd_3s_shuffled_matched',
]
identity_oracle_gate_pp = .2
identity_practical_gain_pp = .5
identity_bootstrap_repeats = 1000
identity_source_cache = 'out/future_distillation_response_based/2048x512/source_cache'
identity_pilot_teacher_cache = 'out/future2current/pilot256x128/teacher_features'
