import os
import pdb
import pickle

import cv2
import torch

import argparse

from tqdm import tqdm
from mmseg.registry import MODELS

import model


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


sensor_types = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT',
]

def load_frame_images(frame, data_root):
    images = {}
    for cam_type in sensor_types:
        cam_info = frame['data'].get(cam_type)
        if cam_info is None:
            continue

        img_path = os.path.join(data_root, cam_info['filename'])
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to load image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images[cam_type] = img_rgb
    return images


def main():
    parser = argparse.ArgumentParser(description="Draw NuScenes dataset images")
    parser.add_argument('--pkl_path', type=str, default='./data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl', help='Path to the NuScenes dataset pickle file')
    parser.add_argument('--data_root', type=str, default='./data/nuscenes', help='Root directory of the NuScenes dataset')

    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        print(f"Pickle file not found: {args.pkl_path}")
        return
    
    infos = load_pkl(args.pkl_path)
    infos = infos['infos']  # Assuming the pickle file has a top-level 'infos' key

    model_dict   = dict(type='ConvNeXt')
    img_backbone = MODELS.build(model_dict)
    img_backbone.eval().cuda()  # Set the model to evaluation mode
    img_backbone.load_state_dict(torch.load('./ckpts/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'), strict=False)


    for scene_token, frames in tqdm(infos.items()):
        for frame_idx, frame in enumerate(frames):
            if 'LIDAR_TOP' not in frame['data'].keys():
                continue

            images = load_frame_images(frame, args.data_root)
            imgs_ndarray = [images[cam_type] for cam_type in sensor_types if cam_type in images]
            imgs_tensor  = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in imgs_ndarray], dim=0).float()  # Shape: (num_cams, 3, H, W)
            imgs_tensor = imgs_tensor.cuda()  # Move to GPU
            results = img_backbone(imgs_tensor)  # Forward pass through the backbone
            pdb.set_trace()


if __name__ == "__main__":
    main()