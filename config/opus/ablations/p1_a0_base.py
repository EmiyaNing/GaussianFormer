"""P1/A0: frozen SparseWorldStrict with no adapter - the required baseline.

This is the denominator every adapter variant is compared against.  It uses
the released converted epoch-56 checkpoint and the same reduced split as the
adapter runs, so A0 differs from A5 only by the presence of the Gaussian
branch.

    python3.10 eval.py \
      --py-config config/opus/ablations/p1_a0_base.py \
      --resume-from ckpts/sparseworld_strict_epoch_56.pth \
      --work-dir out/p1_a0_base
"""
_base_ = ['../sparseworld_strict.py']

train_load_from = 'ckpts/sparseworld_strict_epoch_56.pth'
load_from = 'ckpts/sparseworld_strict_epoch_56.pth'

train_dataset_config = dict(imageset='data/reduced_splits/train_reduced.pkl')
val_dataset_config = dict(imageset='data/reduced_splits/val_reduced.pkl')

train_loader = dict(batch_size=1, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
max_epochs = 1
eval_every_epochs = 1
print_freq = 10
