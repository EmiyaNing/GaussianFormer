import os
import torch
import numpy as np
from numpy import random
import mmcv
from PIL import Image
import math
from copy import deepcopy

from . import OPENOCC_TRANSFORMS
from .utils import get_img2global
from .utils import get_lidar2global
from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams


@OPENOCC_TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = torch.from_numpy(imgs)
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class NuScenesAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = 'ego2img' if use_ego else 'lidar2img'

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict[self.projection_key])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        return input_dict


@OPENOCC_TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        if aug_configs is None:
            return results
        resize, resize_dims, crop, flip, rotate = aug_configs
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            mat = np.eye(4)
            mat[:3, :3] = ida_mat
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            results["ego2img"][i] = mat @ results["ego2img"][i]

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat


@OPENOCC_TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', crop_size=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.crop_size = crop_size

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.crop_size is not None:
            img = img[:self.crop_size[0], :self.crop_size[1]]
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['ori_img'] = deepcopy(img)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageHistory(object):
    """Load history frame multi-view images and merge into results.

    Reads historical image frames and appends them to results['img'],
    results['lidar2img'], and results['ego2img']. The downstream transforms
    (ResizeCropFlipImage, etc.) then apply identically to all frames.

    Args:
        num_history (int): Number of history frames to load.
        num_cams (int): Number of cameras per frame. Defaults to 6.
        color_type (str): Color type for mmcv.imread. Defaults to 'unchanged'.
        to_float32 (bool): Whether to convert images to float32. Defaults to True.
    """

    def __init__(self, num_history, num_cams=6, color_type='unchanged', to_float32=True,
                 pad_history=False):
        self.num_history = num_history
        self.num_cams = num_cams
        self.color_type = color_type
        self.to_float32 = to_float32
        self.pad_history = pad_history

    def __call__(self, results):
        # Convert numpy arrays to lists so we can append history entries.
        # Downstream transforms (ResizeCropFlipImage, NuScenesAdaptor) use
        # indexing / np.stack which work identically on lists.
        if isinstance(results.get('lidar2img'), np.ndarray):
            results['lidar2img'] = list(results['lidar2img'])
        if isinstance(results.get('ego2img'), np.ndarray):
            results['ego2img'] = list(results['ego2img'])

        ctx = results.get('history_context', None)

        num_history_frame = 0
        if ctx is not None and self.num_history > 0:
            scene_infos = ctx['scene_infos']
            scene_token = ctx['scene_token']
            frame_index = ctx['frame_index']
            data_path = ctx['data_path']
            sensor_types = ctx['sensor_types']
            lidar2global = results['lidar_pose']
            ego2global = results['ego_pose']

            for prev_idx in range(frame_index - 1, -1, -1):
                prev_info = scene_infos[scene_token][prev_idx]

                # Skip frame if any camera is missing
                if not all(cam_type in prev_info.get('data', {}) for cam_type in sensor_types):
                    continue

                for ci, cam_type in enumerate(sensor_types):
                    fname = os.path.join(data_path, prev_info['data'][cam_type]['filename'])
                    img = mmcv.imread(fname, self.color_type)
                    if self.to_float32:
                        img = img.astype(np.float32)
                    results['img'].append(img)

                    img2global = get_img2global(
                        prev_info['data'][cam_type]['calib'],
                        prev_info['data'][cam_type]['pose'],
                    )
                    results['lidar2img'].append(np.linalg.inv(img2global) @ lidar2global)
                    results['ego2img'].append(np.linalg.inv(img2global) @ ego2global)

                num_history_frame += 1
                if num_history_frame >= self.num_history:
                    break

        # The encoder consumes a fixed [T * 6] camera dimension.  Scene
        # starts have fewer real sweeps, so repeat the current frame instead
        # of producing ragged batches or silently dropping those samples.
        if self.pad_history and num_history_frame < self.num_history:
            current_imgs = list(results['img'][:self.num_cams])
            current_lidar2img = list(results['lidar2img'][:self.num_cams])
            current_ego2img = list(results['ego2img'][:self.num_cams])
            for _ in range(self.num_history - num_history_frame):
                results['img'].extend([image.copy() for image in current_imgs])
                results['lidar2img'].extend([matrix.copy() for matrix in current_lidar2img])
                results['ego2img'].extend([matrix.copy() for matrix in current_ego2img])

        results['num_current_img'] = self.num_cams
        results['num_history_frame'] = num_history_frame
        results['img_shape'] = [x.shape[:2] for x in results['img']]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_history={self.num_history}, '
        repr_str += f'num_cams={self.num_cams}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'to_float32={self.to_float32}, pad_history={self.pad_history})'
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFile(object):

    def __init__(self, pc_range, num_pts, use_ego=False, num_lidar_history=9):
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        self.num_lidar_history = num_lidar_history

    def _load_points(self, pts_path):
        scan = np.fromfile(pts_path, dtype=np.float32)
        scan = scan.reshape((-1, 5))[:, :4]
        return scan

    def _transform_points_to_target(self, points, source_pose, target_pose):
        if points.shape[0] == 0:
            return points

        points_hom = np.concatenate(
            [points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)],
            axis=-1,
        )
        target_from_source = np.linalg.inv(target_pose) @ source_pose
        transformed_xyz = (target_from_source @ points_hom.T).T[:, :3]
        return np.concatenate([transformed_xyz, points[:, 3:]], axis=-1).astype(
            points.dtype, copy=False
        )

    def __call__(self, results):
        pts_path = results['pts_filename']
        scan = self._load_points(pts_path)
        lidar_sweeps = results.get('lidar_sweeps', [])
        if self.num_lidar_history > 0 and len(lidar_sweeps) > 0:
            fused_scans = [scan]
            current_pose = results['lidar_pose']
            for sweep in lidar_sweeps[:self.num_lidar_history]:
                sweep_points = self._load_points(sweep['pts_filename'])
                sweep_points = self._transform_points_to_target(
                    sweep_points, sweep['lidar_pose'], current_pose
                )
                fused_scans.append(sweep_points)
            scan = np.concatenate(fused_scans, axis=0)
        scan[:, 3] = 1.0 # n, 4
        if self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.2
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan.astype(np.float32)
        
        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPseudoPointFromFile(object):

    def __init__(self, datapath, pc_range, num_pts, is_ego=True, use_ego=False):
        self.datapath = datapath
        self.is_ego = is_ego
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        pass

    def __call__(self, results):
        pts_path = os.path.join(self.datapath, f"{results['sample_idx']}.npy")
        scan = np.load(pts_path)
        if self.is_ego and (not self.use_ego):
            ego2lidar = results['ego2lidar']
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = ego2lidar[None, ...] @ scan[..., None] # p, 4, 1
            scan = np.squeeze(scan, axis=-1)

        if (not self.is_ego) and self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.3
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan
        
        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False, perturb=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and (not use_ego)
        self.use_sweeps = use_sweeps
        self.perturb = perturb

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1]+'.npy')
        if os.path.exists(label_file):
            label = np.load(label_file)

            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

            mask = new_label != 0

            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        elif self.use_sweeps:
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            mask = new_label != 0
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        else:
            raise NotImplementedError

        xyz = self.xyz.copy()
        if getattr(self, "perturb", False):
            # xyz[..., :3] = xyz[..., :3] + (np.random.rand(*xyz.shape[:-1], 3) - 0.5) * (0.5 - 1e-3)
            norm_distribution = np.clip(np.random.randn(*xyz.shape[:-1], 3) / 6, -0.5, 0.5)
            xyz[..., :3] = xyz[..., :3] + norm_distribution * 0.49

        if not self.use_ego:
            occ_xyz = xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar']) # 4, 4
            occ_xyz = ego2lidar[None, None, None, ...] @ xyz[..., None] # x, y, z, 4, 1
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        results['occ_xyz'] = occ_xyz
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyOcc3D(object):

    def __init__(self, 
                 occ3d_path, 
                 semantic=True, 
                 pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], 
                 grid_size=0.5, 
                 use_ego=False,
                 use_sweeps=False,
                 perturb=False,
                 model_coord=None,
                 train_mask_type='none'):
        self.occ3d_path = occ3d_path
        self.semantic = semantic
        self.use_ego = use_ego
        self.use_sweeps = use_sweeps
        self.perturb = perturb
        self.model_coord = model_coord
        self.train_mask_type = train_mask_type
        if self.model_coord is not None:
            assert self.model_coord in ('ego', 'lidar')
        assert self.train_mask_type in (
            'none', 'camera', 'lidar', 'camera_lidar', 'nonempty')

        xyz = self.get_meshgrid(pc_range, [200, 200, 16], grid_size)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        
        # 预加载映射关系
        self.timestamp_to_sample = self._load_timestamp_to_sample()
        self.scene_token_to_name = self._load_scene_token_to_name()

    def _select_train_mask(self, mask_camera, mask_lidar, mask_nonempty):
        if self.train_mask_type == 'none':
            return np.ones_like(mask_camera, dtype=bool)
        if self.train_mask_type == 'camera':
            return mask_camera
        if self.train_mask_type == 'lidar':
            return mask_lidar
        if self.train_mask_type == 'camera_lidar':
            return mask_camera & mask_lidar
        if self.train_mask_type == 'nonempty':
            return mask_nonempty
        raise NotImplementedError

    def _resolve_label_file(self, results):
        occ_path = results.get('occ_path', '')
        if occ_path:
            candidate_paths = [
                occ_path,
                os.path.join(self.occ3d_path, occ_path),
                os.path.join(os.path.dirname(self.occ3d_path), occ_path),
            ]
            for candidate in candidate_paths:
                if os.path.isdir(candidate):
                    candidate = os.path.join(candidate, 'labels.npz')
                if os.path.exists(candidate):
                    return candidate

        sample_token = results.get('sample_idx', '')
        scene_token = results.get('scene_token', '')
        scene_name = self.scene_token_to_name.get(scene_token)
        if sample_token and scene_name:
            label_file = os.path.join(
                self.occ3d_path, scene_name, sample_token, 'labels.npz')
            if os.path.exists(label_file):
                return label_file

        lidar_filename = results['pts_filename'].split('/')[-1]
        timestamp_str = lidar_filename.split('__')[-1].split('.')[0]
        sample_info = self.timestamp_to_sample.get(timestamp_str)
        if not sample_info:
            raise ValueError(f"时间戳 {timestamp_str} 在sample.json中未找到对应样本")

        frame_token = sample_info['token']
        scene_token = sample_info['scene_token']
        scene_name = self.scene_token_to_name.get(scene_token)
        if not scene_name:
            raise ValueError(f"scene_token {scene_token} 在scene.json中未找到对应场景")

        return os.path.join(self.occ3d_path, scene_name, frame_token, 'labels.npz')

    def _load_timestamp_to_sample(self):
        """从sample.json加载时间戳到样本信息的映射"""
        sample_json_path = "data/nuscenes/v1.0-trainval/sample.json"
        timestamp_to_sample = {}
        if os.path.exists(sample_json_path):
            import json
            with open(sample_json_path, 'r') as f:
                samples = json.load(f)
            for sample in samples:
                timestamp = sample.get('timestamp')
                if timestamp is not None:
                    timestamp_to_sample[str(timestamp)] = sample
        return timestamp_to_sample

    def _load_scene_token_to_name(self):
        """从scene.json加载scene_token到scene_name的映射"""
        scene_json_path = "data/nuscenes/v1.0-trainval/scene.json"
        scene_token_to_name = {}
        if os.path.exists(scene_json_path):
            import json
            with open(scene_json_path, 'r') as f:
                scenes = json.load(f)
            for scene in scenes:
                token = scene.get('token')
                name = scene.get('name')
                if token and name:
                    scene_token_to_name[token] = name
        return scene_token_to_name

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz

    def __call__(self, results):
        label_file = self._resolve_label_file(results)
        
        if os.path.exists(label_file):
            labels = np.load(label_file)
            semantics = labels['semantics'].astype(np.int64)
            mask_camera = labels['mask_camera'].astype(bool)
            mask_lidar = labels['mask_lidar'].astype(bool)
            mask_nonempty = semantics != 17
            mask_loss = self._select_train_mask(
                mask_camera, mask_lidar, mask_nonempty)

            results['occ_label'] = semantics if self.semantic else mask_nonempty
            results['occ_mask'] = mask_nonempty
            results['occ_nonempty_mask'] = mask_nonempty
            results['occ_cam_mask'] = mask_camera
            results['occ_lidar_mask'] = mask_lidar
            results['occ_loss_mask'] = mask_loss
            
        elif self.use_sweeps:
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            mask_nonempty = new_label != 17
            mask_visible = np.zeros_like(mask_nonempty, dtype=bool)
            mask_loss = self._select_train_mask(
                mask_visible, mask_visible, mask_nonempty)
            results['occ_label'] = new_label if self.semantic else mask_nonempty
            results['occ_mask'] = mask_nonempty
            results['occ_nonempty_mask'] = mask_nonempty
            results['occ_cam_mask'] = mask_visible
            results['occ_lidar_mask'] = mask_visible
            results['occ_loss_mask'] = mask_loss
        else:
            raise FileNotFoundError(f"Occ3D标注文件不存在: {label_file}")

        xyz = self.xyz.copy()
        if getattr(self, "perturb", False):
            norm_distribution = np.clip(np.random.randn(*xyz.shape[:-1], 3) / 6, -0.5, 0.5)
            xyz[..., :3] = xyz[..., :3] + norm_distribution * 0.49

        if self.model_coord is None:
            if not self.use_ego:
                occ_xyz = xyz[..., :3]
            else:
                ego2lidar = np.linalg.inv(results['ego2lidar'])
                occ_xyz = ego2lidar[None, None, None, ...] @ xyz[..., None]
                occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        elif self.model_coord == 'ego':
            occ_xyz = xyz[..., :3]
        else:
            ego2lidar = results['ego2lidar']
            occ_xyz = ego2lidar[None, None, None, ...] @ xyz[..., None]
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        
        results['occ_xyz'] = occ_xyz
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyKITTI360(object):

    def __init__(self, occ_path, semantic=False, unknown_to_empty=False, training=False):
        self.occ_path = occ_path
        self.semantic = semantic

        xyz = self.get_meshgrid([0.0, -25.6, -2.0, 51.2, 25.6, 4.4], [256, 256, 32], 0.2)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4
        self.unknown_to_empty = unknown_to_empty
        self.training = training

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):        
        occ_xyz = self.xyz[..., :3].copy()
        results['occ_xyz'] = occ_xyz

        ## read occupancy label
        label_path = os.path.join(
            self.occ_path, results['sequence'], "{}_1_1.npy".format(results['token']))
        label = np.load(label_path).astype(np.int64)
        if getattr(self, "unknown_to_empty", False) and getattr(self, "training", False):
            label[label == 255] = 0

        results['occ_cam_mask'] = (label != 255)
        results['occ_label'] = label
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class EntropyBasedHistoryFrameLoader(object):
    """基于熵增益的自适应历史帧加载器。

    在 Pipeline 中替代 LoadMultiViewImageHistory，根据融合历史帧后
    复合场景熵增益动态选择实际使用的历史帧数量。

    - 图像输出键与 LoadMultiViewImageHistory 完全一致
      (img, lidar2img, ego2img, num_current_img, num_history_frame, img_shape)
    - 点云替换 results['lidar_points']

    Args:
        max_window:             最大历史帧融合窗口
        min_window:             最小历史帧融合窗口
        entropy_gain_threshold: 熵增益阈值，低于此值时停止融合
        data_root:              NuScenes 数据根目录
        pc_range:               点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        num_cams:               相机数量，默认 6
        to_float32:             图像是否转 float32，默认 True
    """

    def __init__(self, max_window, min_window, entropy_gain_threshold,
                 data_root, pc_range, num_cams=6, to_float32=True):
        self.max_window = max_window
        self.min_window = min_window
        self.entropy_gain_threshold = entropy_gain_threshold
        self.data_root = data_root
        self.pc_range = pc_range
        self.num_cams = num_cams
        self.to_float32 = to_float32
        from dataset.entropy_history_loader import EntropyBasedHistoryLoader

        self.engine = EntropyBasedHistoryLoader(
            max_window=self.max_window,
            min_window=self.min_window,
            entropy_gain_threshold=self.entropy_gain_threshold,
            data_root=self.data_root,
            pc_range=self.pc_range,
        )

    def __call__(self, results):
        # ---- Step 0: 前置格式转换（与 LoadMultiViewImageHistory 一致） ----
        if isinstance(results.get('lidar2img'), np.ndarray):
            results['lidar2img'] = list(results['lidar2img'])
        if isinstance(results.get('ego2img'), np.ndarray):
            results['ego2img'] = list(results['ego2img'])

        # ---- Step 1: 前置检查 ----
        ctx = results.get('history_context', None)
        if ctx is None:
            results['num_current_img'] = self.num_cams
            results['num_history_frame'] = 0
            results['img_shape'] = [x.shape[:2] for x in results['img']]
            return results

        # ---- Step 2: 提取场景元数据 ----
        scene_infos = ctx['scene_infos']
        scene_token = ctx['scene_token']
        frame_index = ctx['frame_index']

        # ---- Step 3: 调用熵增益核心引擎 ----
        
        selected_frames, gain_values = self.engine.forward(
            scene_infos, scene_token, frame_index
        )

        # ---- Step 4: 加载选中历史帧图像（与 LoadMultiViewImageHistory 一致） ----
        sensor_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
        ]

        num_history_frame = 0
        for frame in selected_frames:
            for cam_type in sensor_types:
                img_path = frame['img_files'][cam_type]
                img = mmcv.imread(img_path)
                if self.to_float32:
                    img = img.astype(np.float32)

                results['img'].append(img)
                results['lidar2img'].append(frame['lidar2img'][cam_type])
                results['ego2img'].append(frame['ego2img'][cam_type])
            num_history_frame += 1

        # ---- Step 5: 替换 lidar_points 为熵筛选后的融合点云 ----
        current_only = self._load_lidar_points(results['pts_filename'])
        if num_history_frame > 0:
            fused = [current_only]
            for frame in selected_frames:
                fused.append(frame['points'])
            results['lidar_points'] = np.concatenate(fused, axis=0)
        else:
            results['lidar_points'] = current_only

        # ---- Step 6: 记录元信息（与 LoadMultiViewImageHistory 一致的键） ----
        results['num_current_img'] = self.num_cams
        results['num_history_frame'] = num_history_frame
        results['img_shape'] = [x.shape[:2] for x in results['img']]

        return results

    def _load_lidar_points(self, lidar_filename):
        """加载单帧LiDAR点云，返回 (N, 4) — x, y, z, intensity"""
        lidar_path = (
            lidar_filename if os.path.isabs(lidar_filename)
            else os.path.join(self.data_root, lidar_filename)
        )
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        return points[:, :4]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_window={self.max_window}, '
        repr_str += f'min_window={self.min_window}, '
        repr_str += f'entropy_gain_threshold={self.entropy_gain_threshold})'
        return repr_str
