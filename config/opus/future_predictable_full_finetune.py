"""4096/1024, native heads, all Student parameters trainable, five epochs."""
_base_ = ['./future_predictable_information.py']

full_tune_root = 'out/future2current/predictable_full_finetune_4096x1024'
full_tune_train_samples = 4096
full_tune_val_samples = 1024
full_tune_epochs = 5
# One paired seed for this substantially longer end-to-end run; the three
# supervision arms retain identical data, initialization and training budget.
full_tune_seeds = [20260930]
full_tune_arms = ['task_only', 'teacher_all', 'teacher_shuffled']
full_tune_workers = 4
full_tune_lr = 2e-5
full_tune_backbone_lr = 2e-6
full_tune_adapter_lr = 1e-4
full_tune_feature_weight = .1
full_tune_warmup_steps = 200
full_tune_log_every = 32
