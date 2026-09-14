"""One-epoch end-to-end smoke test for the optimized stable recipe."""
_base_ = ['./forecasting_sparseworld_stable_r50_704x256_6f.py']

train_dataset_config = dict(max_samples=8)
val_dataset_config = dict(max_samples=2)
max_epochs = 1
eval_every_epochs = 1
warmup_iters = 8
print_freq = 1
epoch_start_sleep = 0
