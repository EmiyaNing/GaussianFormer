"""P1/A0: trained OPUS-v1-1f, without an adapter.

Evaluate this exact checkpoint under the same Occ3D camera-mask protocol as
the adapter variants.  It is a denominator, not a trainable experiment.
"""
_base_ = ['../opusv1_t_r50_704x256_1f_occ3d.py']

load_from = 'out/opusv1_1f/epoch_3.pth'
train_load_from = load_from
train_load_strict = True
train_loader = dict(batch_size=1, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
max_epochs = 1
eval_every_epochs = 1
print_freq = 20
