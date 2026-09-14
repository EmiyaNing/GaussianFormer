"""Six-epoch bounded first validation run for ForecastingSparseWorld."""
_base_ = ['./forecasting_sparseworld_opusv2_tiny_r50_704x256_6f.py']

# This run proves data/model/loss/backward/checkpoint integration. The full
# config above intentionally has no sample cap.
train_dataset_config = dict(max_samples=6)
val_dataset_config = dict(max_samples=1)
print_freq = 1

