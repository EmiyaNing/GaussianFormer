"""Lower-memory variant: full v5 plus decoder recomputation."""
_base_ = ['./img_voxel_react_occ3d_fast_v5_fused.py']
input_shape = (704, 256)
data_aug_conf = {
    "resize_lim": (0.44, 0.44),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

# The parent expands its dataset configs with the old ``data_aug_conf``, so
# the override has to be pushed into both dataset configs explicitly.
train_dataset_config = dict(data_aug_conf=data_aug_conf)
val_dataset_config = dict(data_aug_conf=data_aug_conf)

# The inherited loaders default to ``num_workers=0`` (train) / ``2`` (val);
# both are overridden to 4 for this recipe.
train_loader = dict(num_workers=4)
val_loader = dict(num_workers=4)

model = dict(encoder=dict(checkpoint_ffn=True, checkpoint_deformable=True))
