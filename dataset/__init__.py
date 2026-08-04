from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
OPENOCC_TRANSFORMS = Registry('openocc_transforms')

from .dataset import NuScenesDataset
from .daocc_surroundocc import DAOccSurroundOccDataset
from .dataset_flow import NuScenesFlowDataset, SceneStream
from .transform_3d import *
from .sampler import CustomDistributedSampler
from .utils import custom_collate_fn_temporal

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def _limit_worker_threads(_worker_id):
    """避免每个 DataLoader worker 再创建一组 BLAS/OpenMP 线程。"""
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    threadpool_limits(limits=1)


def _worker_options(loader_config):
    """仅在启用 worker 时传递 PyTorch DataLoader 的预取选项。"""
    if loader_config["num_workers"] <= 0:
        return {}
    options = dict(
        persistent_workers=loader_config.get("persistent_workers", False),
        prefetch_factor=loader_config.get("prefetch_factor", 2),
    )
    if loader_config.get("limit_worker_threads", False):
        options["worker_init_fn"] = _limit_worker_threads
    return options


def get_dataloader(
    train_dataset_config, 
    val_dataset_config, 
    train_loader, 
    val_loader, 
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(
        shuffle=True,
        drop_last=True),
    val_sampler_config=dict(
        shuffle=False,
        drop_last=False),
    val_only=False,
):
    if val_only:
        val_wrapper = OPENOCC_DATASET.build(
            val_dataset_config)
                
        val_sampler = None
        if dist:
            val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

        val_dataset_loader = DataLoader(
            dataset=val_wrapper,
            batch_size=val_loader["batch_size"],
            collate_fn=custom_collate_fn_temporal,
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_loader["num_workers"],
            pin_memory=True,
            **_worker_options(val_loader))

        return None, val_dataset_loader

    train_wrapper = OPENOCC_DATASET.build(
        train_dataset_config)
    val_wrapper = OPENOCC_DATASET.build(
        val_dataset_config)
        
    train_sampler = val_sampler = None
    if dist:
        if iter_resume:
            train_sampler = CustomDistributedSampler(train_wrapper, **train_sampler_config)
        else:
            train_sampler = DistributedSampler(train_wrapper, **train_sampler_config)
        val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False if dist else train_loader["shuffle"],
        sampler=train_sampler,
        num_workers=train_loader["num_workers"],
        pin_memory=True,
        **_worker_options(train_loader))
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_loader["num_workers"],
        pin_memory=True,
        **_worker_options(val_loader))

    return train_dataset_loader, val_dataset_loader


def get_stream_dataloader(
    dataset_config,
    loader_config,
    dist=False,
    shuffle_scenes=False,
    sampler_config=None,
):
    """创建流式 DataLoader。

    与 get_dataloader 不同的是，流式 DataLoader 保证：
    1. 同一场景的帧按时间顺序连续返回
    2. 提供场景切换信号
    3. 支持场景级 shuffle（场景顺序打乱，但场景内保持时序）

    Args:
        dataset_config: 数据集配置
        loader_config: DataLoader 配置（batch_size, num_workers 等）
        dist: 是否使用分布式
        shuffle_scenes: 是否在场景级别打乱顺序
        sampler_config: 采样器配置
    """
    dataset = OPENOCC_DATASET.build(dataset_config)
    assert isinstance(dataset, NuScenesFlowDataset), \
        "流式 DataLoader 需要 NuScenesFlowDataset 类型"

    # 创建场景流迭代器
    stream = SceneStream(
        dataset,
        shuffle_scenes=shuffle_scenes,
        shuffle_frames=False,
    )

    # 单帧模式：使用 custom_collate_fn_temporal
    collate_fn = custom_collate_fn_temporal

    # 分布式采样器
    sampler = None
    if dist:
        from .sampler import SceneStreamSampler
        sampler = SceneStreamSampler(
            dataset, stream,
            **(sampler_config or {})
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=loader_config.get("batch_size", 1),
        collate_fn=collate_fn,
        shuffle=False,
        sampler=sampler,
        num_workers=loader_config.get("num_workers", 4),
        pin_memory=True,
    )

    return dataloader
