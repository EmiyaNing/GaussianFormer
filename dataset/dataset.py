import os
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class NuScenesDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        occ3d=False,
        num_samples=0,
        vis_scene_index=-1,
        phase='train',
        return_keys=[
            'img',
            'projection_mat',
            'image_wh',
            'intrinsic',
            'mask_img',
            'lidar2cam',
            'occ_label',
            'occ_xyz',
            'occ_cam_mask',
            'ori_img',
            'cam_positions',
            'focal_positions',
            'lidar_points',
            'lidar_pose',
            'ego_pose'
        ],
    ):
        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

        self.data_aug_conf = data_aug_conf
        self.pc_range  = pc_range
        self.test_mode = (phase != 'train')
        self.occ3d = occ3d
        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.return_keys = return_keys
        if vis_scene_index >= 0:
            frame = self.keyframes[vis_scene_index]
            num_frames = len(self.scene_infos[frame[0]])
            self.keyframes = [(frame[0], i) for i in range(num_frames)]
            print(f'Scene length: {len(self.keyframes)}')
        elif vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):
        scene_token, index = self.keyframes[index]
        info = deepcopy(self.scene_infos[scene_token][index])
        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)
        
        return_dict = {k: input_dict[k] for k in self.return_keys}
        return return_dict
    
    def get_data_info(self, info):
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        ego2image_rts = []
        cam_positions = []
        focal_positions = []
        cam_intrinsic = []
        lidar2cam_list = []

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['data']['LIDAR_TOP']['pose']['translation']).T
        lidar_points = self.load_lidar_points(info['data']['LIDAR_TOP']['filename'])
        if self.occ3d:
            lidar_reflect = lidar_points[:, 2:3]
            lidar_points[:, 3] = 1.0
            lidar_points = lidar2ego[None, ...] @ lidar_points[..., None]
            lidar_points = np.squeeze(lidar_points, axis=-1)
            lidar_points = lidar_points[:, :3]
            mask_x = (lidar_points[:, 0] > self.pc_range[0]) & (lidar_points[:, 0] < self.pc_range[3])
            mask_y = (lidar_points[:, 1] > self.pc_range[1]) & (lidar_points[:, 1] < self.pc_range[4])
            mask_z = (lidar_points[:, 2] > self.pc_range[2]) & (lidar_points[:, 2] < self.pc_range[5])
            mask = mask_x & mask_y & mask_z
            lidar_points = lidar_points[mask]
            lidar_reflect= lidar_reflect[mask]
            lidar_points = np.concatenate([lidar_points, lidar_reflect], axis=-1)

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))

            img2global, cam2global = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            lidar2cam = np.linalg.inv(cam2global) @ lidar2global

            lidar2img_rts.append(lidar2img)
            lidar2cam_list.append(lidar2cam)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

            img2lidar = np.linalg.inv(lidar2global) @ img2global
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            cam_intrinsic.append(viewpad[:3, :3])
            cam_position = img2lidar @ viewpad @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = img2lidar @ viewpad @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
            
        input_dict =dict(
            # sample_idx=info["token"],
            sample_idx=info.get("token", ""),
            # occ_path=info["occ_path"],
            occ_path=info.get("occ_path", ""),
            timestamp=info["timestamp"] / 1e6,
            img_filename=image_paths,
            pts_filename=os.path.join(self.data_path, info['data']['LIDAR_TOP']['filename']),
            intrinsic=np.asarray(cam_intrinsic),
            ego2lidar=ego2lidar,
            lidar2cam=np.asanyarray(lidar2cam_list),
            lidar2img=np.asarray(lidar2img_rts),
            ego2img=np.asarray(ego2image_rts),
            cam_positions=np.asarray(cam_positions),
            focal_positions=np.asarray(focal_positions),
            lidar_points=lidar_points,  # [N, 4] 点云数据 (x, y, z, intensity)
            lidar_pose=lidar2global,    # LiDAR到全局坐标系的变换矩阵
            ego_pose=ego2global         # Ego到全局坐标系的变换矩阵
        )
        return input_dict

    def __len__(self):
        return len(self.keyframes)
    

    def load_lidar_points(self, lidar_filename):
        """加载LiDAR点云数据"""
        lidar_path = os.path.join(self.data_path, lidar_filename)
    
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, 5)  # NuScenes: x, y, z, intensity, ring_index
        points = points[:, :4]
        #points[:, 3] = 1.0
    
        return points