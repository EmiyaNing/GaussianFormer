'''
This script used to move rendered img into right fir dir.
'''

import os
import shutil

import numpy as np
import torch

import pickle
from tqdm import tqdm

def load_pkl(pkl_path):
    """加载pkl文件，返回infos字典"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['infos']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', default='./data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl',
                        help='相机参数pkl文件路径')
    parser.add_argument('--source_img_path', default='./rendered')
    parser.add_argument("--output_img_path", default='./samples/mask')
    args = parser.parse_args()
    
    # 检查pkl文件是否存在
    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"Camera pkl file not found: {args.pkl_path}")
    
    # 加载pkl
    infos = load_pkl(args.pkl_path)
    #print(f"Loaded {len(infos)} scenes")
    
    # 遍历场景
    for scene_token, frames in tqdm(infos.items()):
        for frame_idx, frame in enumerate(frames):
            #output_dir = os.path.join(args.output_root, scene_token, f'frame_{frame_idx:03d}')
            if 'LIDAR_TOP' not in frame['data']:
                continue
            ori_img_path_dir = os.path.join(args.source_img_path, scene_token, f'frame_{frame_idx:03d}')
            ori_front_gs     = frame['data']['CAM_FRONT']['filename'].split('/')[-1].split('.')[0] + '.png'
            ori_back_gs      = frame['data']['CAM_BACK']['filename'].split('/')[-1].split('.')[0] + '.png'
            ori_backl_gs     = frame['data']['CAM_BACK_LEFT']['filename'].split('/')[-1].split('.')[0] + '.png'
            ori_backr_gs     = frame['data']['CAM_BACK_RIGHT']['filename'].split('/')[-1].split('.')[0] + '.png'
            ori_frontl_gs    = frame['data']['CAM_FRONT_LEFT']['filename'].split('/')[-1].split('.')[0] + '.png'
            ori_frontr_gs    = frame['data']['CAM_FRONT_RIGHT']['filename'].split('/')[-1].split('.')[0] + '.png'

            ori_front_gs_path = os.path.join(ori_img_path_dir, 'CAM_FRONT', ori_front_gs)
            ori_back_gs_path  = os.path.join(ori_img_path_dir, 'CAM_BACK', ori_back_gs)
            ori_backl_gs_path = os.path.join(ori_img_path_dir, 'CAM_BACK_LEFT', ori_backl_gs)
            ori_backr_gs_path = os.path.join(ori_img_path_dir, 'CAM_BACK_RIGHT', ori_backr_gs)
            ori_frontl_gs_path= os.path.join(ori_img_path_dir, 'CAM_FRONT_LEFT', ori_frontl_gs)
            ori_frontr_gs_path= os.path.join(ori_img_path_dir, 'CAM_FRONT_RIGHT', ori_frontr_gs)
            

            new_dir_front      = os.path.join(args.output_img_path, 'CAM_FRONT', ori_front_gs)
            new_dir_front_right= os.path.join(args.output_img_path, 'CAM_FRONT_RIGHT', ori_frontr_gs)
            new_dir_front_left = os.path.join(args.output_img_path, 'CAM_FRONT_LEFT', ori_frontl_gs)
            new_dir_back       = os.path.join(args.output_img_path, 'CAM_BACK', ori_back_gs)
            new_dir_back_right = os.path.join(args.output_img_path, 'CAM_BACK_RIGHT', ori_backr_gs)
            new_dir_back_left  = os.path.join(args.output_img_path, 'CAM_BACK_LEFT', ori_backl_gs)
            # copy the figure from rendered to a new dir
            # in this dir all figure retain the 
            shutil.move(ori_front_gs_path, new_dir_front)
            shutil.move(ori_back_gs_path, new_dir_back)
            shutil.move(ori_backl_gs_path, new_dir_back_left)
            shutil.move(ori_backr_gs_path, new_dir_back_right)
            shutil.move(ori_frontl_gs_path, new_dir_front_right)
            shutil.move(ori_frontr_gs_path, new_dir_front_left)
            
            
            

if __name__ == '__main__':
    main()