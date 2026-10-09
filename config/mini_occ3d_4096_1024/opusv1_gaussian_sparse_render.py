"""Five-epoch OPUS-V1 Gaussian sparse-render experiment on the mini split."""
_base_ = ['../opus/opusv1_gaussian_sparse_render.py']

seed = 928
max_epochs = 5
eval_every_epochs = 1
train_dataset_config = dict(imageset='data/mini_occ3d_4096_1024/train.pkl')
val_dataset_config = dict(imageset='data/mini_occ3d_4096_1024/val.pkl')
train_loader = dict(batch_size=1, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
