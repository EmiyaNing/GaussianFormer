import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class CustomDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, last_iter: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.first_run = True
        self.last_iter = last_iter

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if not self.first_run:
            assert len(indices) == self.num_samples
        else:
            indices = indices[self.last_iter:]
            self.last_iter = 0
            self.first_run = False

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    
    def set_last_iter(self, last_iter: int):
        self.last_iter = last_iter


class SceneStreamSampler(Sampler):
    """分布式场景流采样器。

    在分布式训练/评估中，将场景分配给不同 GPU，每个 GPU 处理完整场景集合。
    确保同一个场景的所有帧被分配到同一个 rank，避免场景内帧被分割到不同设备。
    """

    def __init__(self, dataset, stream, num_replicas=None, rank=None, shuffle_scenes=False, seed=0):
        """
        Args:
            dataset: NuScenesFlowDataset 实例
            stream: SceneStream 迭代器
            num_replicas: 分布式进程数，默认从 dist 获取
            rank: 当前进程 rank，默认从 dist 获取
            shuffle_scenes: 是否打乱场景分配顺序
            seed: 随机种子
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.stream = stream
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle_scenes = shuffle_scenes
        self.seed = seed

    def __iter__(self):
        """按场景分配帧索引到各 rank。

        收集所有场景的场景 token，将场景均匀分配给各 rank，
        然后 yield 当前 rank 分配到的所有帧的 flat index。
        """
        # 收集所有场景 token
        scene_tokens = list(self.dataset.iter_scenes())
        num_scenes = len(scene_tokens)

        if self.shuffle_scenes:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(num_scenes, generator=g).tolist()
            scene_tokens = [scene_tokens[i] for i in indices]

        # 将场景均匀分配给各 rank
        scenes_per_rank = num_scenes // self.num_replicas
        remaining = num_scenes % self.num_replicas

        # 计算当前 rank 的场景范围
        start = self.rank * scenes_per_rank + min(self.rank, remaining)
        end = start + scenes_per_rank + (1 if self.rank < remaining else 0)

        assigned_scenes = scene_tokens[start:end]

        # 收集分配场景的所有帧索引
        indices = []
        for scene_token in assigned_scenes:
            scene_start, scene_end = self.dataset.get_scene_frame_range(scene_token)
            indices.extend(range(scene_start, scene_end))

        return iter(indices)

    def __len__(self):
        """返回当前 rank 预计处理的帧数（近似值）。"""
        num_scenes = self.dataset.num_scenes()
        scenes_per_rank = num_scenes // self.num_replicas
        remaining = num_scenes % self.num_replicas
        assigned_scenes = scenes_per_rank + (1 if self.rank < remaining else 0)

        # 近似：使用平均场景长度
        avg_scene_len = len(self.dataset) / num_scenes if num_scenes > 0 else 0
        return int(assigned_scenes * avg_scene_len)
