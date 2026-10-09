"""Five-epoch GaussianFormer baseline, matched to 704x256 mini input."""
_base_ = ['../gaussianformer_occ3d.py']

seed = 928
max_epochs = 5
eval_every_epochs = 1
input_shape = (704, 256)
data_aug_conf = dict(
    resize_lim=(0.44, 0.44), final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0),
    H=900, W=1600, rand_flip=True)
train_dataset_config = dict(
    imageset='data/mini_occ3d_4096_1024/train.pkl',
    data_aug_conf=data_aug_conf)
val_dataset_config = dict(
    imageset='data/mini_occ3d_4096_1024/val.pkl',
    data_aug_conf=data_aug_conf)
train_loader = dict(batch_size=1, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
