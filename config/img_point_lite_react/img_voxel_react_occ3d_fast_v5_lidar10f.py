"""Fast v5 with ten LiDAR frames: the current scan plus nine history scans.

Only the point cloud is temporal. The camera branch keeps the fast_v5 recipe
(single frame, six views, 256 x 704) and every execution optimization (tiled
React renderer, BF16 image path, 4 workers) is inherited unchanged.

``num_lidar_history = 9`` makes ``NuScenesDataset`` concatenate the current
``LIDAR_TOP`` scan with the nine previous ones, each transformed into the
current LiDAR frame, before ``PrepareReactVoxels`` voxelizes the fused cloud
inside the data-loading worker. In the ``*_sweeps_occ.pkl`` annotations only
keyframes carry ``LIDAR_TOP``, so the history steps are keyframe scans
(~0.5 s apart); scene starts simply provide fewer frames.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5.py']

# Ten LiDAR frames in total = current keyframe scan + nine history scans.
num_lidar_history = 9

train_dataset_config = dict(num_lidar_history=num_lidar_history)
val_dataset_config = dict(num_lidar_history=num_lidar_history)
