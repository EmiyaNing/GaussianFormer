import os 
import pdb

import numpy as np
from open3d_vis_utils import draw_scenes

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ]).astype(np.float32) / 255.

def get_file_names(directory):
    file_names = os.listdir(directory)
    file_names = sorted(file_names)
    return file_names

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

occ3d_root_path = './data/occ3d/gts/'
scene_back_path = get_file_names(occ3d_root_path)
scene_path = [occ3d_root_path + occ_scene + '/' for occ_scene in scene_back_path]




scene_01_root_path = './data/occ3d/gts/scene-0001/'
scene_01_back_path = get_file_names(scene_01_root_path)
scene_01_path = [scene_01_root_path + occ_01_scene + '/labels.npz' for occ_01_scene in scene_01_back_path]

for file in scene_01_path:
    occ3d_data = np.load(file)
    semantics  = occ3d_data['semantics']
    mask_lidar = occ3d_data['mask_lidar']
    mask_camera= occ3d_data['mask_camera']
    grid_coors = get_grid_coords([200, 200, 16], [0.4, 0.4, 0.4])
    semantics_r  = semantics.reshape(-1, 1)
    mask_lidar_r = mask_lidar.reshape(-1, 1)
    mask_camera_r= mask_camera.reshape(-1, 1)

    empty_mask = semantics_r.squeeze() < 17
    grid_coors = grid_coors[empty_mask]
    filter_sems= semantics_r[empty_mask].squeeze()
    filter_lidar_mask = mask_lidar_r[empty_mask].squeeze() > 0
    filter_camera_mask= mask_camera_r[empty_mask].squeeze() > 0
    
    cur_colors = colors_map[filter_sems][:, :3]
    print("current display the original occupancy scenes\n")
    draw_scenes(points=grid_coors, point_colors=cur_colors)
    
    print("current display the camera_masked occupancy scenes\n")
    camera_coors = grid_coors[filter_camera_mask]
    camera_colors= cur_colors[filter_camera_mask]
    draw_scenes(points=camera_coors, point_colors=camera_colors)

    print("current display the lidar_masked occupancy scenes\n")
    lidar_coors  = grid_coors[filter_lidar_mask]
    lidar_colors = cur_colors[filter_lidar_mask]
    draw_scenes(points=lidar_coors, point_colors=lidar_colors)
    print("\n\n\n\n")
    #pdb.set_trace()