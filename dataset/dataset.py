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
        num_lidar_history=0,
        vis_indices=None,
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        occ3d=False,
        occ3d_coord='ego',
        vis_scene_index=-1,
        phase='train',
        return_keys=[
            'img',
            'projection_mat',
            'image_wh',
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
        self.num_lidar_history = num_lidar_history
        self.test_mode = (phase != 'train')
        self.occ3d = occ3d
        self.occ3d_coord = occ3d_coord
        assert self.occ3d_coord in ('ego', 'lidar')
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
        input_dict = self.get_data_info(info, scene_token=scene_token, frame_index=index)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)
        
        return_dict = {k: input_dict[k] for k in self.return_keys}
        return return_dict
    
    def get_data_info(self, info, scene_token=None, frame_index=None):
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        ego2image_rts = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['data']['LIDAR_TOP']['pose']['translation']).T
        lidar_history = self.collect_lidar_history(info, scene_token=scene_token, frame_index=frame_index)
        lidar_points = self.load_lidar_points_with_history(
            info['data']['LIDAR_TOP']['filename'],
            lidar2global,
            lidar_history,
        )
        if self.occ3d and self.occ3d_coord == 'ego':
            lidar_reflect = lidar_points[:, 3:4].copy()
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
            lidar_points = lidar_points.astype(np.float32, copy=False)

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))

            img2global = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global

            lidar2img_rts.append(lidar2img)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

            if self.occ3d and self.occ3d_coord == 'ego':
                img2model = np.linalg.inv(ego2global) @ img2global
            else:
                img2model = np.linalg.inv(lidar2global) @ img2global
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            cam_position = img2model @ viewpad @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = img2model @ viewpad @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
            
        input_dict =dict(
            # sample_idx=info["token"],
            sample_idx=info.get("token", ""),
            # occ_path=info["occ_path"],
            occ_path=info.get("occ_path", ""),
            scene_token=info.get("scene_token", scene_token),
            frame_index=frame_index,
            timestamp=info["timestamp"] / 1e6,
            img_filename=image_paths,
            pts_filename=os.path.abspath(os.path.join(self.data_path, info['data']['LIDAR_TOP']['filename'])),
            ego2lidar=ego2lidar,
            lidar2img=np.asarray(lidar2img_rts),
            ego2img=np.asarray(ego2image_rts),
            cam_positions=np.asarray(cam_positions),
            focal_positions=np.asarray(focal_positions),
            lidar_points=lidar_points,  # [N, 4] 点云数据 (x, y, z, intensity)
            lidar_pose=lidar2global,    # LiDAR到全局坐标系的变换矩阵
            ego_pose=ego2global,        # Ego到全局坐标系的变换矩阵
        )

        if scene_token is not None and frame_index is not None:
            input_dict['history_context'] = dict(
                scene_infos=self.scene_infos,
                scene_token=scene_token,
                frame_index=frame_index,
                data_path=self.data_path,
                sensor_types=list(self.sensor_types),
            )

        return input_dict

    def __len__(self):
        return len(self.keyframes)
    

    def load_lidar_points(self, lidar_filename):
        """加载LiDAR点云数据"""
        lidar_path = lidar_filename if os.path.isabs(lidar_filename) else os.path.join(self.data_path, lidar_filename)
    
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, 5)  # NuScenes: x, y, z, intensity, ring_index
        points = points[:, :4]
        #points[:, 3] = 1.0
    
        return points

    def collect_lidar_history(self, info, scene_token=None, frame_index=None):
        if self.num_lidar_history <= 0:
            return []

        if scene_token is None:
            scene_token = info['scene_token']
        if frame_index is None:
            raise ValueError('`frame_index` is required when collecting lidar history.')

        history = []
        scene_infos = self.scene_infos[scene_token]
        for prev_idx in range(frame_index - 1, -1, -1):
            prev_info = scene_infos[prev_idx]
            prev_lidar_info = prev_info.get('data', {}).get('LIDAR_TOP')
            if prev_lidar_info is None:
                continue

            history.append(
                dict(
                    pts_filename=os.path.abspath(os.path.join(self.data_path, prev_lidar_info['filename'])),
                    lidar_pose=get_lidar2global(
                        prev_lidar_info['calib'],
                        prev_lidar_info['pose'],
                    ),
                )
            )
            if len(history) >= self.num_lidar_history:
                break

        return history


    def transform_points_to_target(self, points, source_pose, target_pose):
        if points.shape[0] == 0:
            return points

        points_hom = np.concatenate(
            [points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)],
            axis=-1,
        )
        target_from_source = np.linalg.inv(target_pose) @ source_pose
        transformed_xyz = (target_from_source @ points_hom.T).T[:, :3]

        if points.shape[1] > 3:
            transformed_points = np.concatenate([transformed_xyz, points[:, 3:]], axis=-1)
        else:
            transformed_points = transformed_xyz
        return transformed_points.astype(points.dtype, copy=False)

    def load_lidar_points_with_history(self, lidar_filename, lidar_pose, lidar_history):
        current_points = self.load_lidar_points(lidar_filename)
        fused_points = [current_points]

        for sweep in lidar_history:
            sweep_points = self.load_lidar_points(sweep['pts_filename'])
            sweep_points = self.transform_points_to_target(
                sweep_points,
                sweep['lidar_pose'],
                lidar_pose,
            )
            fused_points.append(sweep_points)

        return np.concatenate(fused_points, axis=0) if len(fused_points) > 1 else current_points
