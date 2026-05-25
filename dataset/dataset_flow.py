import numpy as np

from . import OPENOCC_DATASET
from .dataset import NuScenesDataset


@OPENOCC_DATASET.register_module()
class NuScenesFlowDataset(NuScenesDataset):
    """流式NuScenes数据集，确保按场景和时间顺序读取。

    在 NuScenesDataset 的基础上扩展了场景感知能力，支持：
    - 场景边界检测
    - 单帧流式读取（每帧携带场景元数据）
    - 场景级迭代
    - 帧间 LiDAR 坐标系变换矩阵

    Args:
        **kwargs: 其余参数传递给 NuScenesDataset。
    """

    def __init__(
        self,
        **kwargs,
    ):
        # 1. 调用父类 __init__，完成基础初始化
        super().__init__(**kwargs)

        # 2. 将场景元数据键追加到 return_keys，确保场景信息能传递到 DataLoader
        _scene_meta_keys = [
            'scene_token', 'is_first_frame', 'is_last_frame',
            'frame_index_in_scene', 'num_frames_in_scene',
            'lidar2prev', 'lidar2next',
        ]
        for _key in _scene_meta_keys:
            if _key not in self.return_keys:
                self.return_keys.append(_key)

        # 3. 构建场景索引
        self._build_scene_index()

    def _build_scene_index(self):
        """从 scene_infos 和 keyframes 构建场景索引结构。"""
        # 1. 获取所有 scene_token 列表（保持顺序）
        self.scene_tokens = list(self.scene_infos.keys())

        # 2. 为每个 scene 构建帧索引列表
        #    scene_frames[scene_token] = [(flat_idx_in_keyframes, frame_idx_in_scene), ...]
        self.scene_frames = {}
        self.scene_lengths = {}
        self.scene_start_indices = {}
        self.scene_end_indices = {}

        current_scene = None
        scene_frame_list = []

        for flat_idx, (scene_token, frame_idx) in enumerate(self.keyframes):
            if scene_token != current_scene:
                if current_scene is not None:
                    # 保存上一个场景的信息
                    self.scene_frames[current_scene] = scene_frame_list
                    self.scene_lengths[current_scene] = len(scene_frame_list)
                    self.scene_end_indices[current_scene] = flat_idx - 1
                # 开始新场景
                current_scene = scene_token
                scene_frame_list = []
                self.scene_start_indices[scene_token] = flat_idx

            scene_frame_list.append((flat_idx, frame_idx))

        # 保存最后一个场景
        if current_scene is not None:
            self.scene_frames[current_scene] = scene_frame_list
            self.scene_lengths[current_scene] = len(scene_frame_list)
            self.scene_end_indices[current_scene] = len(self.keyframes) - 1

        # 3. 构建 flat_index → scene_info 的快速查找表
        #    flat_to_scene[flat_idx] = (scene_token, frame_index_in_scene)
        self.flat_to_scene = {}
        for scene_token in self.scene_frames:
            for flat_idx, frame_idx_in_scene in self.scene_frames[scene_token]:
                self.flat_to_scene[flat_idx] = (scene_token, frame_idx_in_scene)

    # ──────────────────────────────────────────────
    # 场景边界管理与元数据接口
    # ──────────────────────────────────────────────

    def get_scene_info(self, flat_index):
        """根据 flat index 获取场景信息。

        Args:
            flat_index: 全局扁平索引（对应 keyframes 列表中的位置）。

        Returns:
            dict: {
                'scene_token': str,
                'frame_index_in_scene': int,      # 在当前场景中的帧序号（从0开始）
                'num_frames_in_scene': int,       # 当前场景总帧数
                'is_first_frame': bool,           # 是否为场景首帧
                'is_last_frame': bool,            # 是否为场景尾帧
                'scene_progress': float,          # 场景进度 0.0~1.0
            }
        """
        scene_token, frame_in_scene = self.flat_to_scene[flat_index]
        scene_len = self.scene_lengths[scene_token]

        return {
            'scene_token': scene_token,
            'frame_index_in_scene': frame_in_scene,
            'num_frames_in_scene': scene_len,
            'is_first_frame': (frame_in_scene == 0),
            'is_last_frame': (frame_in_scene == scene_len - 1),
            'scene_progress': (frame_in_scene + 1) / scene_len,
        }

    def is_first_frame_of_scene(self, flat_index):
        """判断给定 flat index 是否为场景首帧。"""
        scene_token, frame_in_scene = self.flat_to_scene[flat_index]
        return frame_in_scene == 0

    def is_last_frame_of_scene(self, flat_index):
        """判断给定 flat index 是否为场景尾帧。"""
        scene_token, frame_in_scene = self.flat_to_scene[flat_index]
        return frame_in_scene == self.scene_lengths[scene_token] - 1

    def get_scene_frame_range(self, scene_token):
        """获取指定场景的帧范围 [start_flat_index, end_flat_index]。"""
        start = self.scene_start_indices[scene_token]
        end = self.scene_end_indices[scene_token]
        return start, end

    def get_scene_length(self, scene_token):
        """获取指定场景的帧数量。"""
        return self.scene_lengths[scene_token]

    def num_scenes(self):
        """返回数据集中的场景总数。"""
        return len(self.scene_tokens)

    def iter_scenes(self):
        """返回场景生成器，按序 yield (scene_token, start_idx, end_idx)。"""
        for scene_token in self.scene_tokens:
            start = self.scene_start_indices[scene_token]
            end = self.scene_end_indices[scene_token]
            yield scene_token, start, end

    def get_scene_frames(self, scene_token):
        """获取指定场景的所有帧索引列表（flat index）。"""
        return [flat_idx for flat_idx, _ in self.scene_frames[scene_token]]

    # ──────────────────────────────────────────────
    # __len__ 与采样器兼容性
    # ──────────────────────────────────────────────

    def __len__(self):
        """返回总帧数，与父类 NuScenesDataset 一致。"""
        return super().__len__()

    # ──────────────────────────────────────────────
    # 流式 __getitem__（单帧模式）
    # ──────────────────────────────────────────────

    def __getitem__(self, index):
        """获取单帧数据样本。

        基于父类 NuScenesDataset.__getitem__ 的逻辑，并额外添加：
        - 场景元数据（scene_token, is_first_frame, is_last_frame 等）
        - 帧间 LiDAR 坐标系变换矩阵（lidar2prev, lidar2next）

        Args:
            index: 扁平索引，指向具体帧。

        Returns:
            dict: 包含 return_keys 中所有键的数据字典。
        """
        return self._get_single_frame(index)

    def _get_single_frame(self, index):
        """获取单帧数据（复用父类 NuScenesDataset.__getitem__ 的逻辑）。"""
        scene_token, frame_index = self.keyframes[index]
        info = self._deepcopy_info(scene_token, frame_index)
        input_dict = self.get_data_info(
            info, scene_token=scene_token, frame_index=frame_index
        )

        # 添加场景元数据
        scene_info = self.get_scene_info(index)
        input_dict['scene_token'] = scene_token
        input_dict['is_first_frame'] = scene_info['is_first_frame']
        input_dict['is_last_frame'] = scene_info['is_last_frame']
        input_dict['frame_index_in_scene'] = scene_info['frame_index_in_scene']
        input_dict['num_frames_in_scene'] = scene_info['num_frames_in_scene']

        # 添加帧间 LiDAR 坐标系变换矩阵
        input_dict = self._add_lidar_flow_transforms(input_dict, index, scene_token)

        # 数据增强
        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)

        return {k: input_dict[k] for k in self.return_keys}

    def _deepcopy_info(self, scene_token, frame_index):
        """深拷贝场景信息（从 copy 导入 deepcopy）。"""
        from copy import deepcopy
        return deepcopy(self.scene_infos[scene_token][frame_index])

    # ──────────────────────────────────────────────
    # 帧间 LiDAR 坐标系变换矩阵
    # ──────────────────────────────────────────────

    def _add_lidar_flow_transforms(self, input_dict, flat_index, scene_token):
        """将当前帧到前一帧/后一帧的 LiDAR 坐标系变换矩阵添加到 input_dict。

        计算两个 4x4 齐次变换矩阵：
          - lidar2prev: 将当前帧 LiDAR 坐标系下的点变换到前一帧 LiDAR 坐标系
          - lidar2next: 将当前帧 LiDAR 坐标系下的点变换到后一帧 LiDAR 坐标系

        场景边界处理：
          - 首帧: lidar2prev 设为单位矩阵
          - 尾帧: lidar2next 设为单位矩阵

        Args:
            input_dict: 当前帧的数据字典（需包含 'lidar_pose'）。
            flat_index: 当前帧的全局扁平索引。
            scene_token: 当前帧所属场景的 token。

        Returns:
            dict: 添加了 'lidar2prev' 和 'lidar2next' 后的数据字典。
        """
        from .utils import get_lidar2global

        curr_lidar2global = input_dict['lidar_pose']  # 4x4 矩阵

        scene_frames = self.scene_frames[scene_token]  # [(flat_idx, frame_idx_in_scene), ...]

        # 查找当前帧在场景帧列表中的位置
        curr_pos = None
        for pos, (f_idx, _) in enumerate(scene_frames):
            if f_idx == flat_index:
                curr_pos = pos
                break

        scene_len = len(scene_frames)

        # --- 前一帧变换: 将当前坐标系下的点变换到前一帧坐标系 ---
        if curr_pos is not None and curr_pos > 0:
            prev_flat_idx = scene_frames[curr_pos - 1][0]
            prev_scene_token, prev_frame_idx = self.keyframes[prev_flat_idx]
            prev_info = self._deepcopy_info(prev_scene_token, prev_frame_idx)
            prev_lidar2global = get_lidar2global(
                prev_info['data']['LIDAR_TOP']['calib'],
                prev_info['data']['LIDAR_TOP']['pose'])
            # curr2prev = inv(prev_lidar2global) @ curr_lidar2global
            input_dict['lidar2prev'] = np.linalg.inv(prev_lidar2global) @ curr_lidar2global
        else:
            input_dict['lidar2prev'] = np.eye(4, dtype=np.float32)

        # --- 后一帧变换: 将当前坐标系下的点变换到后一帧坐标系 ---
        if curr_pos is not None and curr_pos < scene_len - 1:
            next_flat_idx = scene_frames[curr_pos + 1][0]
            next_scene_token, next_frame_idx = self.keyframes[next_flat_idx]
            next_info = self._deepcopy_info(next_scene_token, next_frame_idx)
            next_lidar2global = get_lidar2global(
                next_info['data']['LIDAR_TOP']['calib'],
                next_info['data']['LIDAR_TOP']['pose'])
            # curr2next = inv(next_lidar2global) @ curr_lidar2global
            input_dict['lidar2next'] = np.linalg.inv(next_lidar2global) @ curr_lidar2global
        else:
            input_dict['lidar2next'] = np.eye(4, dtype=np.float32)

        return input_dict


# ──────────────────────────────────────────────────────
# 场景级迭代器
# ──────────────────────────────────────────────────────


class SceneStream:
    """场景流迭代器，按场景顺序逐帧访问数据集。

    用于流式评估，保证同一场景的帧连续访问，并提供场景切换信号。

    Args:
        dataset: NuScenesFlowDataset 实例。
        shuffle_scenes: 是否随机打乱场景顺序（训练时用）。
        shuffle_frames: 是否在场景内随机打乱帧顺序（需谨慎使用）。
        seed: 随机种子。
    """

    def __init__(self, dataset, shuffle_scenes=False, shuffle_frames=False, seed=0):
        self.dataset = dataset
        self.shuffle_scenes = shuffle_scenes
        self.shuffle_frames = shuffle_frames
        self.rng = np.random.RandomState(seed)

        # 构建场景索引
        self.scene_list = list(dataset.scene_tokens)

    def __iter__(self):
        """返回场景流迭代器。

        Yields:
            (data_dict, scene_token, is_first_frame, is_last_frame)
        """
        scenes = list(self.scene_list)
        if self.shuffle_scenes:
            self.rng.shuffle(scenes)

        for scene_token in scenes:
            frames = self.dataset.get_scene_frames(scene_token)
            if self.shuffle_frames:
                self.rng.shuffle(frames)

            for i, flat_idx in enumerate(frames):
                is_first = (i == 0)
                is_last = (i == len(frames) - 1)

                data = self.dataset[flat_idx]
                yield data, scene_token, is_first, is_last

    def __len__(self):
        """返回总帧数。"""
        return len(self.dataset)
