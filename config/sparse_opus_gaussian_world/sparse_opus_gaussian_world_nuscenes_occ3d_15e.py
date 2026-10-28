"""Standalone full-split nuScenes Occ3D training configuration: 15 epochs.

Run from the repository root:
    CUDA_VISIBLE_DEVICES=0 python3.10 train.py \
        --py-config config/sparse_opus_gaussian_world/sparse_opus_gaussian_world_nuscenes_occ3d_15e.py \
        --work-dir out/sparse_opus_gaussian_world_nuscenes_occ3d_15e \
        --gradient-accumulation 2

The directory uses the requested gaussian spelling. The registered model ID
remains sparse_opus_gausisan_world for compatibility with the integrated D model.
The generic train.py runner uses all eligible train/val samples. The earlier
subset-only tools.sparse_opus_gausisan_world.train runner is not used here.
"""

# Registration: only library modules, no experiment runtime.

custom_imports = {
    "allow_failed_imports": False,
    "imports": [
        "model.segmentor.sparse_opus_gausisan_world",
        "loss.sparse_opus_gausisan_world_loss",
    ],
}

experiment_name = "sparse_opus_gaussian_world_nuscenes_occ3d_15e"

# Full nuScenes train/val splits, with four history and six future frames.

data_root = "data/nuscenes/"

anno_root = "data/nuscenes_cam/"

occ3d_path = "data/occ3d"

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

voxel_size = [0.4, 0.4, 0.4]

grid_shape = [200, 200, 16]

img_norm_cfg = {
    "mean": [123.675, 116.28, 103.53],
    "std": [58.395, 57.12, 57.375],
    "to_rgb": True,
}

data_aug_conf = {
    "H": 900,
    "W": 1600,
    "bot_pct_lim": (0.0, 0.0),
    "final_dim": (256, 704),
    "rand_flip": True,
    "resize_lim": (0.38, 0.55),
    "rot_lim": (0.0, 0.0),
}

num_frames = 5

future_steps = 6

future_queries = [60, 60, 60, 60, 40, 40]

common_pipeline = [
    {"color_type": "color", "to_float32": False, "type": "LoadMultiViewImageFromFiles"},
    {
        "color_type": "color",
        "num_cams": 6,
        "num_history": 4,
        "pad_history": False,
        "to_float32": False,
        "type": "LoadMultiViewImageHistory",
    },
    {
        "grid_size": 0.4,
        "model_coord": "ego",
        "occ3d_path": "data/occ3d",
        "pc_range": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        "semantic": True,
        "train_mask_type": "none",
        "type": "LoadOccupancyOcc3D",
    },
    {"occ3d_path": "data/occ3d", "type": "LoadSparseWorldFutureOccupancy"},
    {"type": "ResizeCropFlipImage"},
    {"type": "DefaultFormatBundle"},
    {"num_cams": 30, "type": "NuScenesAdaptor", "use_ego": True},
]

return_keys = [
    "img",
    "projection_mat",
    "image_wh",
    "occ_label",
    "occ_xyz",
    "occ_cam_mask",
    "occ_lidar_mask",
    "occ_nonempty_mask",
    "occ_loss_mask",
    "future_occ_labels",
    "future_occ_cam_masks",
    "future_occ_lidar_masks",
    "future_ego_to_current",
    "temporal_trajs",
    "temporal_ego_states",
    "ego2lidar",
    "img_filename",
]

# Keep D perception, world rollout, learned empty bias and supervision horizons.

model = {
    "data_aug": {
        "img_color_aug": True,
        "img_norm_cfg": {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "to_rgb": True,
        },
        "img_pad_cfg": {"size_divisor": 32},
    },
    "feature_mode": "official_auto",
    "finetune_epoch": 5,
    "gaussian_head": {
        "analytic_empty": True,
        "compact_aggregation": True,
        "apply_loss_type": "fixed_5",
        "cuda_kwargs": {
            "D": 16,
            "H": 200,
            "W": 200,
            "grid_size": 0.4,
            "pc_min": [-40.0, -40.0, -1.0],
            "scale_multiplier": 3,
        },
        "dataset_type": "nusc",
        "empty_args": {"mean": [0.0, 0.0, -1.0], "scale": [100.0, 100.0, 8.0]},
        "empty_label": 17,
        "empty_logit_offset": 0.0,
        "initial_empty_bias": 0.0,
        "max_render_scale": 0.4,
        "num_classes": 18,
        "type": "SparseOpusGaussianRenderer",
        "use_localagg_react": False,
        "use_localaggprob": False,
        "with_empty": True,
    },
    "gaussian_loss_horizons": [0, 2, 4, 6],
    "image_amp_dtype": "bfloat16",
    "render_training_grids": False,
    "image_checkpoint": True,
    "image_chunk_size": 6,
    "img_backbone": {
        "depth": 50,
        "frozen_stages": 1,
        "norm_cfg": {"requires_grad": True, "type": "BN2d"},
        "norm_eval": True,
        "num_stages": 4,
        "out_indices": (0, 1, 2, 3),
        "style": "pytorch",
        "type": "ResNet",
        "with_cp": False,
    },
    "img_neck": {
        "in_channels": [256, 512, 1024, 2048],
        "num_outs": 4,
        "out_channels": 256,
        "type": "FPN",
    },
    "is_pretrain": False,
    "out_dim": 256,
    "pts_bbox_head": {
        "in_channels": 256,
        "is_pretrain": False,
        "num_classes": 17,
        "num_fu_frames": 6,
        "num_fu_query": [60, 60, 60, 60, 40, 40],
        "num_query": 720,
        "pc_range": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        "test_cfg": {
            "padding": True,
            "score_thr": [
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.25,
                0.3,
            ],
        },
        "train_cfg": {
            "cls_weights": [3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1]
        },
        "transformer": {
            "encoder": {
                "attribute_stages": [5],
                "decoder_checkpoint": True,
                "dropout": 0.1,
                "embed_dims": 256,
                "feedforward_channels": 512,
                "initial_opacity": 0.1,
                "initial_scale": 0.2,
                "num_classes": 17,
                "num_decoder": 6,
                "num_frames": 5,
                "num_groups": 4,
                "num_heads": 8,
                "num_levels": 4,
                "num_points": 4,
                "num_refines": [1, 4, 16, 32, 64, 128],
                "num_views": 6,
                "pc_range": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                "scale_max": 0.4,
                "scale_min": 0.08,
                "scales": [0.5],
                "type": "SparseOpusGaussianEncoder",
            },
            "type": "SparseOpusGaussianTransformer",
        },
        "type": "SparseOpusGaussianPerceptionHead",
        "voxel_size": [0.4, 0.4, 0.4],
    },
    "type": "sparse_opus_gausisan_world",
}

# D objective; freeze the validated D multipliers, without re-estimating frequencies.

loss = {
    "current_weight": 1.0,
    "future_weight": 0.5,
    "mask_type": "none",
    "occupancy_loss": {
        "balance_cls_weight": True,
        "conditional_class_multipliers": [
            3.0,
            3.0,
            3.0,
            3.0,
            1.2212631762516837,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            1.9918065420227402,
            0.5101252679144156,
            3.0,
            0.953678730140281,
            0.8868057218225314,
            0.5,
            0.5,
        ],
        "conditional_semantic_weight": 1.0,
        "empty_label": 17,
        "lovasz_ignore": 17,
        "manual_class_weight": [
            1.01552756,
            1.06897009,
            1.30013094,
            1.07253735,
            0.94637502,
            1.10087012,
            1.26960524,
            1.06258364,
            1.189019,
            1.06217292,
            1.00595144,
            0.85706115,
            1.03923299,
            0.90867526,
            0.8936431,
            0.85486129,
            0.8527829,
            0.1,
        ],
        "multi_loss_weights": {
            "loss_voxel_ce_weight": 10.0,
            "loss_voxel_lovasz_weight": 1.0,
        },
        "num_classes": 18,
        "type": "SparseOpusGaussianVoxelLoss",
        "use_dice_loss": False,
        "use_focal_loss": False,
        "use_lovasz_loss": True,
        "use_sem_geo_scal_loss": False,
        "weight": 1.0,
    },
    "strict_loss": {
        "chunk_size": 1024,
        "class_weights": [3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1],
        "empty_dist_thr": 0.2,
        "empty_weight": 3.0,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "focal_weight": 2.0,
        "future_queries": [60, 60, 60, 60, 40, 40],
        "num_query": 720,
        "pc_range": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        "point_weight": 0.5,
        "smooth_l1_beta": 0.2,
        "type": "SparseWorldStrictLoss",
        "voxel_size": 0.4,
    },
    "type": "SparseOpusGaussianWorldLoss",
}

loss_input_convertion = {
    "current_pred_logits": "current_pred_logits",
    "current_pred_points": "current_pred_points",
    "future_pred_logits": "future_pred_logits",
    "future_pred_masks": "future_pred_masks",
    "future_pred_points": "future_pred_points",
    "gaussian_logits_by_horizon": "gaussian_logits_by_horizon",
    "initial_pred_points": "initial_pred_points",
    "pred_traj": "pred_traj",
    "query_stamps": "query_stamps",
    "sparseworld_pretrain": "sparseworld_pretrain",
    "stamp_statistics": "stamp_statistics",
    "temporal_pred_logits": "temporal_pred_logits",
    "temporal_pred_points": "temporal_pred_points",
    "update_stamp_statistics": "update_stamp_statistics",
}

# Parent dataset selects every temporally eligible keyframe; no subset manifest.
# Local audited metadata yields 19730 training / 4219 validation keyframes.
train_dataset_config = dict(
    type="NuScenesSparseWorldStrictDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=common_pipeline,
    pc_range=pc_range,
    occ3d=True,
    occ3d_coord="ego",
    phase="train",
    future_steps=future_steps,
    history_frames=4,
    admlp_state_path="admlp/stp3_val/data_nuscene.pkl",
    trajectory_path="occworld/nuscenes_infos_train_temporal_v3_scene.pkl",
    return_keys=return_keys,
)
val_dataset_config = dict(
    type="NuScenesSparseWorldStrictDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=common_pipeline,
    pc_range=pc_range,
    occ3d=True,
    occ3d_coord="ego",
    phase="val",
    future_steps=future_steps,
    history_frames=4,
    admlp_state_path="admlp/stp3_val/data_nuscene.pkl",
    trajectory_path="occworld/nuscenes_infos_val_temporal_v3_scene.pkl",
    return_keys=return_keys,
)

train_loader = {"batch_size": 2, "num_workers": 4, "shuffle": True}

val_loader = {"batch_size": 1, "num_workers": 4}

# AdamW parameter-group multipliers are unchanged from D.

optimizer = {
    "optimizer": {"lr": 0.0002, "type": "AdamW", "weight_decay": 0.01},
    "paramwise_cfg": {
        "custom_keys": {
            "gaussian_head.empty_bias": {"decay_mult": 0.0, "lr_mult": 25.0},
            "img_backbone": {"lr_mult": 0.1},
            "sampling_offset": {"lr_mult": 0.1},
        }
    },
    "type": "OptimWrapper",
}

# train.py reads accumulation from its CLI, not this metadata field.
# Always pass --gradient-accumulation 2 with this configuration.
gradient_accumulation = 2
max_epochs = 15
grad_max_norm = 5
# The generic runner advances LR every microbatch: 1000 microbatches correspond
# to D's 500 optimizer-update warm-up with accumulation=2. Cosine duration is
# computed by train.py from the actual full-split loader length and 15 epochs.
warmup_iters = 1000
warmup_lr_init = 2e-4 / 3
min_lr_ratio = 0.001
scheduler_step_before_optimizer = True

# First 5 epochs: temporal pretraining. The model enables future set supervision
# progressively after that; rendered voxel loss keeps half-second indices 0/2/4/6.
enable_model_set_epoch = True
seed = 17
dataloader_seed = 17
cudnn_benchmark = False
amp = False  # Backbone BF16/checkpointing is configured inside the model.
fail_on_nonfinite_grad = True
print_freq = 25
cuda_memory_log_interval = 100
iter_checkpoint_interval = 500
eval_every_epochs = 1
# General train.py reports current-frame occupancy each epoch. eval.py can
# additionally report current/1s/2s/3s SparseWorld metrics on saved checkpoints.
dataset_name_flag = "occ3d"
occ3d_eval_mask = "none"
eval_mask_flag = False
sparseworld_strict_eval = True
strict_load = True
# Gaussian attributes and future branches are conditionally used by the curriculum.
find_unused_parameters = True
syncBN = False

# Start from image backbone/FPN pretraining only, not any D trained checkpoint.
load_from = None

train_load_from = "/home/emiya/projects/SparseWorld/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth"

train_load_revise_keys = [
    ("backbone.", "img_backbone."),
    ("neck.", "img_neck."),
]
train_load_strict = False
