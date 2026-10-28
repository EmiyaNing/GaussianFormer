"""Standalone PMG architecture experiment A3: geometry=free, content=False.

No inherited configuration and no experiments imports. The 17 multipliers
are frozen from the original 4096-sample current-frame training histogram.
Use tools.pmg_world.train for matched architecture experiments, or
tools.pmg_world.run_experiments for the complete staged experiment queue.
"""

# Registration and public model name
custom_imports = {
    "allow_failed_imports": False,
    "imports": [
        "model.segmentor.pmg_sparse_opus_world",
        "loss.sparse_opus_gausisan_world_loss",
        "dataset.sparse_opus_gausisan_world_dataset",
    ],
}

experiment_name = "pmg_A3_4096_1024_direct15e"

# Data coordinates, image augmentation and pipelines
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

num_frames = 5

future_steps = 6

future_queries = [60, 60, 60, 60, 40, 40]

# Model: horizons are half-second indices; D directly supervises 0/2/4/6
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
    "finetune_epoch": 0,
    "geometry_mode": 'free',
    "shape_penalty": 2.0,
    "shape_radius_rms": 0.10,
    "persistent_content": False,
    "render_training_grids": False,
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
    "type": "PMGSparseOpusWorld",
}

# Objective: original point/world losses + full-space CE/Lovasz + balanced conditional CE
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

# The original 4096/1024 token manifests and deterministic loaders
train_dataset_config = {
    "admlp_state_path": "admlp/stp3_val/data_nuscene.pkl",
    "data_aug_conf": {
        "H": 900,
        "W": 1600,
        "bot_pct_lim": (0.0, 0.0),
        "final_dim": (256, 704),
        "rand_flip": True,
        "resize_lim": (0.38, 0.55),
        "rot_lim": (0.0, 0.0),
    },
    "data_root": "data/nuscenes/",
    "future_steps": 6,
    "history_frames": 4,
    "imageset": "data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl",
    "occ3d": True,
    "occ3d_coord": "ego",
    "pc_range": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
    "phase": "train",
    "pipeline": [
        {
            "color_type": "color",
            "to_float32": False,
            "type": "LoadMultiViewImageFromFiles",
        },
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
    ],
    "return_keys": [
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
    ],
    "selection_manifest": "data/sparseworld_opus_gaussian_bias_4096_1024/train.json",
    "trajectory_path": "occworld/nuscenes_infos_train_temporal_v3_scene.pkl",
    "type": "SparseOpusGaussianWorldDataset",
}

val_dataset_config = {
    "admlp_state_path": "admlp/stp3_val/data_nuscene.pkl",
    "data_aug_conf": {
        "H": 900,
        "W": 1600,
        "bot_pct_lim": (0.0, 0.0),
        "final_dim": (256, 704),
        "rand_flip": True,
        "resize_lim": (0.38, 0.55),
        "rot_lim": (0.0, 0.0),
    },
    "data_root": "data/nuscenes/",
    "future_steps": 6,
    "history_frames": 4,
    "imageset": "data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl",
    "occ3d": True,
    "occ3d_coord": "ego",
    "pc_range": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
    "phase": "val",
    "pipeline": [
        {
            "color_type": "color",
            "to_float32": False,
            "type": "LoadMultiViewImageFromFiles",
        },
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
    ],
    "return_keys": [
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
    ],
    "selection_manifest": "data/sparseworld_opus_gaussian_bias_4096_1024/val.json",
    "trajectory_path": "occworld/nuscenes_infos_val_temporal_v3_scene.pkl",
    "type": "SparseOpusGaussianWorldDataset",
}

train_loader = {"batch_size": 2, "num_workers": 4, "shuffle": True}

val_loader = {"batch_size": 1, "num_workers": 4}

# Optimizer and the unchanged 15-epoch update schedule
optimizer = {
    "optimizer": {"lr": 0.0002, "type": "AdamW", "weight_decay": 0.01},
    "paramwise_cfg": {
        "custom_keys": {
            "gaussian_head.empty_bias": {"decay_mult": 0.0, "lr_mult": 25.0},
            "img_backbone": {"lr_mult": 0.1},
            "sampling_offset": {"lr_mult": 0.1},
        }
    },
}

max_epochs = 15

gradient_accumulation = 2

grad_max_norm = 5

warmup_iters = 500

warmup_lr_init = 6.666666666666667e-05

min_lr_ratio = 0.001

scheduler_step_before_optimizer = True

seed = 17

dataloader_seed = 17

cudnn_benchmark = False

amp = False

checkpoint_interval_steps = 250

print_freq = 25

# Evaluation and reproducibility metadata
epoch_diagnostics = {"attribute_samples": 16, "gradient_samples": 2}

evaluation_horizons = [0, 1, 2, 3, 4, 5, 6]

selection_horizons = [0, 2, 4, 6]

eval_every_epochs = 1

report_directory = "analysis/pmg_world_stage1/A3"

# Optional report-only reference; never used by the model or training loss.
strict_reference_directory = (
    "out/sparseworld_opus_gaussian_bias_4096_1024_15e/sparseworld_strict"
)

train_load_from = "/home/emiya/projects/SparseWorld/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth"

initialization_scope = "image_backbone_and_neck_only"

legacy_checkpoint_config_sha256 = [
    "0e99c73672fdcd5d54428c665917f40a303da9790b4a5948164fe22941273d3a",
    "9f3f5c3be260eeed2f9a69d907b363ae8aaef365ce818c54cea63c51b5cf303b",
]

legacy_training_contract_sha256 = (
    "ad600e389ae70fcf540dd41654ac8d6d66f642f4e46734871528fcc748898c88"
)

# Compatibility options retained from the original repository configuration.
dataset_name_flag = "occ3d"

enable_model_set_epoch = True

eval_mask_flag = False

find_unused_parameters = False

load_from = None

occ3d_eval_mask = "none"

official_checkpoint = "../SparseWorld/ckpts/epoch_56.pth"

sparseworld_strict_eval = True

strict_load = True

syncBN = True

train_load_revise_keys = [("backbone.", "img_backbone.")]

train_load_strict = False

# Evaluation-only resources; never passed to the model or the objective.
pmg_experiment = {'arm': 'A3', 'planned_epochs': 15, 'reference_directory': 'analysis/pmg_world_stage1/reference', 'attribute_root': 'out/sparseworld_region_comparison/20261010_full_nuscenes/attributes', 'region_horizons': [2, 4, 6], 'architecture_only': True}
