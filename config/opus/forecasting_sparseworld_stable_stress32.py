"""Short full-pipeline stress test, including samples beyond smoke8."""
_base_ = ['./forecasting_sparseworld_stable_r50_704x256_6f.py']

train_dataset_config = dict(max_samples=32)
val_dataset_config = dict(max_samples=1)
max_epochs = 1
eval_every_epochs = 2
warmup_iters = 32
print_freq = 8
epoch_start_sleep = 0
