"""Strict DAOcc SurroundOcc dataset using the released DAOcc info files."""

import os
import pickle

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, pil_to_tensor

from . import OPENOCC_DATASET


DAOCC_CLASSES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')


def _resolve_path(path, data_root):
    candidates = [
        path,
        os.path.join(data_root, path),
        os.path.join(data_root, path.lstrip('./')),
    ]
    basename = os.path.basename(path)
    if 'surround' in path or 'nuscenes_occ' in path:
        candidates.append(os.path.join('data/surroundocc/samples', basename))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _image_transform(img, final_dim, is_train):
    """Official ImageAug3D parameters used by the SurroundOcc config."""
    f_h, f_w = final_dim
    width, height = img.size
    resize = float(f_w) / float(width)
    resize_dims = (int(width * resize), int(height * resize))
    new_w, new_h = resize_dims
    crop_h = new_h - f_h
    crop_w = int(np.random.uniform(0, max(0, new_w - f_w))) if is_train else \
        int(max(0, new_w - f_w) / 2)
    crop = (crop_w, crop_h, crop_w + f_w, crop_h + f_h)
    flip = bool(np.random.choice([0, 1])) if is_train else False

    img = img.resize(resize_dims).crop(crop)
    rotation = torch.eye(2) * resize
    translation = -torch.tensor(crop[:2], dtype=torch.float32)
    if flip:
        img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        matrix = torch.tensor([[-1., 0.], [0., 1.]])
        offset = torch.tensor([float(f_w), 0.])
        rotation = matrix @ rotation
        translation = matrix @ translation + offset
    transform = torch.eye(4)
    transform[:2, :2] = rotation
    transform[:2, 3] = translation
    tensor = pil_to_tensor(img).float().div_(255)
    tensor = normalize(
        tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor, transform


@OPENOCC_DATASET.register_module()
class DAOccSurroundOccDataset(Dataset):
    def __init__(self, ann_file, data_root='data/nuscenes',
                 occ_root='data/surroundocc/samples', phase='train',
                 num_sweeps=9, use_valid_flag=True, cbgs=True):
        with open(ann_file, 'rb') as file:
            data = pickle.load(file)
        self.infos = sorted(data['infos'], key=lambda item: item['timestamp'])
        self.data_root = data_root
        self.occ_root = occ_root
        self.training = phase == 'train'
        self.num_sweeps = num_sweeps
        self.use_valid_flag = use_valid_flag
        self.sample_indices = (
            self._cbgs_indices() if self.training and cbgs
            else list(range(len(self.infos))))

    def __len__(self):
        return len(self.sample_indices)

    def _cbgs_indices(self):
        """Match DAOcc's mmdet3d.datasets.CBGSDataset resampling."""
        per_class = {index: [] for index in range(len(DAOCC_CLASSES))}
        for sample_index, info in enumerate(self.infos):
            mask = info['valid_flag'] if self.use_valid_flag else \
                info['num_lidar_pts'] > 0
            names = set(info['gt_names'][mask])
            for class_index, name in enumerate(DAOCC_CLASSES):
                if name in names:
                    per_class[class_index].append(sample_index)
        duplicated = sum(len(indices) for indices in per_class.values())
        target_fraction = 1.0 / len(DAOCC_CLASSES)
        sampled = []
        for indices in per_class.values():
            distribution = len(indices) / duplicated
            ratio = target_fraction / distribution
            sampled.extend(np.random.choice(
                indices, int(len(indices) * ratio)).tolist())
        return sampled

    def _points(self, info):
        key_path = _resolve_path(info['lidar_path'], self.data_root)
        key = np.fromfile(key_path, dtype=np.float32).reshape(-1, 5)
        key[:, 4] = 0
        points = [key]
        sweeps = info['sweeps']
        if not sweeps:
            sweeps = [None] * self.num_sweeps
        else:
            sweeps = sweeps[:self.num_sweeps]
        for sweep in sweeps:
            if sweep is None:
                sweep_points = key.copy()
            else:
                path = _resolve_path(sweep['data_path'], self.data_root)
                sweep_points = np.fromfile(
                    path, dtype=np.float32).reshape(-1, 5).copy()
                close = (np.abs(sweep_points[:, 0]) < 1.0) & (
                    np.abs(sweep_points[:, 1]) < 1.0)
                sweep_points = sweep_points[~close]
                sweep_points[:, :3] = (
                    sweep_points[:, :3] @
                    np.asarray(sweep['sensor2lidar_rotation']).T)
                sweep_points[:, :3] += np.asarray(
                    sweep['sensor2lidar_translation'])
                sweep_points[:, 4] = (
                    info['timestamp'] - sweep['timestamp']) / 1e6
            points.append(sweep_points)
        return np.concatenate(points, axis=0).astype(np.float32)

    def _images_and_geometry(self, info):
        images, camera2lidar, intrinsics, image_aug = [], [], [], []
        for camera in info['cams'].values():
            path = _resolve_path(camera['data_path'], self.data_root)
            image, augmentation = _image_transform(
                Image.open(path).convert('RGB'), (256, 704), self.training)
            images.append(image)
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = camera['sensor2lidar_rotation']
            transform[:3, 3] = camera['sensor2lidar_translation']
            camera2lidar.append(transform)
            intrinsic = np.eye(4, dtype=np.float32)
            intrinsic[:3, :3] = camera['camera_intrinsics']
            intrinsics.append(intrinsic)
            image_aug.append(augmentation.numpy())
        return (
            torch.stack(images),
            np.stack(camera2lidar),
            np.stack(intrinsics),
            np.stack(image_aug))

    def _annotations(self, info):
        mask = info['valid_flag'] if self.use_valid_flag else \
            info['num_lidar_pts'] > 0
        boxes = info['gt_boxes'][mask].copy()
        names = info['gt_names'][mask]
        velocity = info['gt_velocity'][mask].copy()
        velocity[np.isnan(velocity)] = 0
        boxes = np.concatenate((boxes, velocity), axis=-1)
        labels = np.asarray([
            DAOCC_CLASSES.index(name) if name in DAOCC_CLASSES else -1
            for name in names], dtype=np.int64)
        valid = labels >= 0
        from mmdet3d.structures import LiDARInstance3DBoxes
        return (
            LiDARInstance3DBoxes(
                boxes[valid], box_dim=9, origin=(0.5, 0.5, 0.5)),
            torch.from_numpy(labels[valid]))

    def _occupancy(self, info):
        configured = info['surround_occ']['occ_path']
        candidates = [
            _resolve_path(configured, self.data_root),
            os.path.join(self.occ_root, os.path.basename(configured)),
            os.path.join(
                self.occ_root, os.path.basename(info['lidar_path']) + '.npy'),
        ]
        path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        sparse = np.load(path).astype(np.int64)
        dense = np.zeros((200, 200, 16), dtype=np.int64)
        sparse[sparse[:, 3] == 0, 3] = 255
        dense[sparse[:, 0], sparse[:, 1], sparse[:, 2]] = sparse[:, 3]
        return dense

    def __getitem__(self, index):
        info = self.infos[self.sample_indices[index]]
        images, camera2lidar, intrinsics, image_aug = \
            self._images_and_geometry(info)
        points = self._points(info)
        boxes, labels = self._annotations(info)
        occupancy = self._occupancy(info)

        lidar_aug = np.eye(4, dtype=np.float32)
        occ_aug = np.eye(4, dtype=np.float32)
        if self.training:
            horizontal = bool(np.random.choice([0, 1]))
            vertical = bool(np.random.choice([0, 1]))
            if horizontal:
                matrix = np.diag([1., -1., 1.]).astype(np.float32)
                points[:, :3] = points[:, :3] @ matrix.T
                boxes.flip('horizontal')
                occupancy = occupancy[:, ::-1, :].copy()
                lidar_aug[:3, :3] = matrix @ lidar_aug[:3, :3]
                occ_aug[:3, :3] = matrix @ occ_aug[:3, :3]
            if vertical:
                matrix = np.diag([-1., 1., 1.]).astype(np.float32)
                points[:, :3] = points[:, :3] @ matrix.T
                boxes.flip('vertical')
                occupancy = occupancy[::-1, :, :].copy()
                lidar_aug[:3, :3] = matrix @ lidar_aug[:3, :3]
                occ_aug[:3, :3] = matrix @ occ_aug[:3, :3]
        order = np.random.permutation(points.shape[0]) if self.training else \
            np.arange(points.shape[0])
        return dict(
            img=images,
            lidar_points=torch.from_numpy(points[order]),
            camera2lidar=camera2lidar,
            camera_intrinsics=intrinsics,
            img_aug_matrix=image_aug,
            lidar_aug_matrix=lidar_aug,
            occ_aug_matrix=occ_aug,
            occ_label=occupancy,
            occ_cam_mask=occupancy != 255,
            gt_bboxes_3d=boxes,
            gt_labels_3d=labels,
            sample_idx=info['token'])
