import pdb
import os
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from open3d_vis_utils import draw_scenes

pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]

def load_pkl(pkl_path):
    """加载pkl文件，返回infos字典"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['infos']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', default='./data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl',
                        help='相机参数pkl文件路径')
    parser.add_argument('--occ_data_root', default='./data/surroundocc/samples',
                        help='surroundocc数据目录，包含.npy文件')
    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"Camera pkl file not found: {args.pkl_path}")
    
    infos = load_pkl(args.pkl_path)

    for scene_token, frames in tqdm(infos.items()):
        for frame_idx, frame in enumerate(frames):
            if 'LIDAR_TOP' in frame['data'].keys():
                lidar_filename = frame['data']['LIDAR_TOP']['filename']
                lidar_path = './data/nuscenes/' + lidar_filename
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

                mask_x = (points[:, 0] > pc_range[0]) & (points[:, 0] < pc_range[3])
                mask_y = (points[:, 1] > pc_range[1]) & (points[:, 1] < pc_range[4])
                mask_z = (points[:, 2] > pc_range[2]) & (points[:, 2] < pc_range[5])
                mask = mask_x & mask_y & mask_z
                points = points[mask]

                draw_scenes(points=points[:, :3])


if __name__ == '__main__':
    main()