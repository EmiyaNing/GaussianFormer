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


surround_root   = './data/surroundocc/samples/'
scene_back_list = get_file_names(surround_root)
scene_path_list = [surround_root + scene for scene in scene_back_list]

for scene_path in scene_path_list:
    surround_data = np.load(scene_path)
    coords = surround_data[:, :3] * 0.5 - np.array([50.0, 50.0, 5.0])
    #pdb.set_trace()
    sems   = surround_data[:, 3]
    cur_colors = colors_map[sems][:, :3]
    print("current display the scene ", scene_path)
    print("current x axes range: ", str(coords[:, 0].min()), ' ', str(coords[:, 0].max()))
    print("current y axes range: ", str(coords[:, 1].min()), ' ', str(coords[:, 1].max()))
    print("current z axes range: ", str(coords[:, 2].min()), ' ', str(coords[:, 2].max()))
    draw_scenes(points=coords, point_colors=cur_colors)
    print("\n\n")
    #pdb.set_trace()