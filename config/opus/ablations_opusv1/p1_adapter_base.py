"""Shared P1 protocol: frozen OPUS-v1-1f plus one trainable adapter.

This is intentionally a 1-frame *screening* protocol.  Its role is to decide
whether a Gaussian adapter deserves the much more expensive OPUS-v1-8f P3
training.  All paper-facing end-to-end numbers must use the P3 8-frame
configs, never this file's numbers.
"""
_base_ = ['../opus_semantic_gaussian_residual_localagg_cuda.py']

model = dict(type='FrozenOPUSAdapterSegmentor', freeze_opus_backbone=True,
             head=dict(eval_mode='ensemble', gaussian_max_neighbours=0))

# This is a trained OPUS-v1 checkpoint in this repository, not a SparseWorld
# checkpoint.  The Gaussian adapter keys are the only intentional misses.
train_load_from = 'out/opusv1_1f/epoch_3.pth'
train_load_strict = False
load_from = train_load_from

# Per-GPU setting.  Four 4090 cards therefore give a global batch of four.
train_loader = dict(batch_size=1, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
max_epochs = 20
eval_every_epochs = 1
print_freq = 20
amp = False
find_unused_parameters = False
