# 固定 600 query 的 3D voxel OPUS Fusion

本版本直接使用 `img_voxel_lite` 同类的 `MeanVFE + VoxelResBackBone8x` 点云编码器，
对全部有效输出体素中心做 FPS，选出 600 个初始坐标，以对应体素特征初始化 query。
没有空间池化、Top-K 评分筛选或 BEV 高度折叠。所有输出体素保留为 decoder memory。

## 配置

| 配置 | 图像输入 | 点云输入 | 评估 mask |
| --- | --- | --- | --- |
| `opusv1_voxel_fusion_t_r50_704x256_8f_occ3d.py` | 当前 + 7 帧历史、每帧 6 相机 | 当前 + 最多 9 帧历史 | camera |
| `opusv1_voxel_fusion_t_r50_704x256_1f_occ3d.py` | 当前帧 6 相机 | 当前 + 最多 9 帧历史 | camera |
| `opusv1_voxel_fusion_t_r50_704x256_8f_occ3d_nomask.py` | 同主配置 | 同主配置 | 无 |

点云历史复用本仓库 `NuScenesDataset.collect_lidar_history` 的过去 LiDAR 记录，
由全局位姿对齐到当前 ego 坐标系；不跨场景，场景起始处使用实际可用历史。
这里“9 帧历史”的具体时间间隔遵循本地 annotation 中含 `LIDAR_TOP` 的记录，
不宣称与官方 OPUS 的 9 个原始 nuScenes sweeps 具有相同时间间隔。
图像历史沿用相应的现有 OPUS 配置。

三份配置的训练监督一致：OPUS-V1 全集合分类及双向最近邻位置损失，
`train_mask_type='none'`；camera 可见性仍参与既有 OPUS 损失的 GT 距离权重。
`nomask` 仅改变评估区域，不改变训练目标。默认 batch size 为每 GPU 1。
图像 backbone 使用 BF16 分块 checkpoint，FPN、spconv、decoder 和 loss 使用 FP32。
继承的 `load_from` 用于已有图像预训练权重；新 LiDAR/fusion 模块从头初始化，
不加载官方 DAL/BEV 分支的权重。

## 实现与接口

- `model/lifter/fixed_voxel_opus_lifter.py`：原始点以 0.05 m 体素化，8x 输出网格
  pitch 为 0.4 m、256 通道。实际 Z 尺寸依赖 backbone padding，坐标按
  `origin + (index_xyz + 0.5) * pitch` 解码，不按输出 shape 拉伸。
- `model/ops/sparse_voxel_sampling.py`：逐样本按 `(z*H+y)*W+x` 排序后 FPS，
  固定起点为最小 key。CUDA 使用 MMCV FPS，CPU 提供参考路径。
  同一 backend 上重复调用及体素行重排后选择稳定；不承诺不同硬件/CPU-CUDA
  对等距离候选做出完全相同的选择。FPS 用物理空间距离，仅索引选择不求导。
- 当有效体素不足 600 时，全部有效体素成为 seed，其余位置使用确定性的 Halton
  坐标与零初始特征。空点云也能生成固定 600 个有效 query，通过图像进行预测。
  `query_voxel_valid_mask` 区分体素 seed 和这些备用位置。
- memory 保存 `[N,256]` 特征和 `[B,Z,Y,X]` int32 行索引表。索引表每次 forward
  建立一次，供六层复用。以实际 `[15,200,200]` 为例，每样本索引表为 2.4 MB。
- 每层、每 query 生成 `G=4,K=4` 个 3D 采样位置，两模态共用这些位置。
  图像沿用 OPUS 多尺度、多帧投影采样；LiDAR 通过索引表读取每个位置的
  `3x3x3` 网格邻域，按连续高斯距离权重归一化池化，不构造全局距离矩阵。
  缺失邻域输出零及有效性标记；本版不额外增加粗层回退或局部 attention。
- 图像 `[B,Q,G,T*K,D/G]` 与体素 `[B,Q,G,K,D/G]` 沿采样点维拼接，经过
  OPUS 动态 channel/point mixing、self-attention 和 FFN，再输出位置与类别。
  混合的输出采样点数为 `(T+1)*K`，与官方 Fusion 的默认行为一致。
- 采样尺度设 0.4 m 下限，避免初始单点 query 的标准差为零。
  预测点数仍为 `[1,4,16,32,64,128]`，最终 `600*128=76800` 点，query 数不扩展。
  decoder 层启用非重入 checkpoint，邻域 gather 以 1024 个采样位置分块。
- 每层 representation 中的 `lidar_sample_valid` 可用于统计空邻域比例。
  本版本保证实现闭环，不代表已经完成训练或获得精度结果。

## 运行

在项目根目录，单卡训练主配置：

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 train.py \
  --py-config config/opus/opusv1_voxel_fusion_t_r50_704x256_8f_occ3d.py \
  --work-dir out/opusv1_voxel_fusion_t_8f --seed 928
```

多卡训练复用现有入口；例如设置 `CUDA_VISIBLE_DEVICES=0,1,2,3` 后运行同一命令，
`train.py` 会按可见 GPU 数自动启动 DDP。batch size 配置为每 GPU 数量。

真实样本完整前向、损失和反向检查，不加载预训练权重：

```bash
python3.10 tools/smoke_opus_voxel_fusion.py \
  --py-config config/opus/opusv1_voxel_fusion_t_r50_704x256_8f_occ3d.py \
  --out work_dirs/opus_voxel_fusion_smoke/8f.json
```

该检查保留完整 600 query 和六层输出，默认最多使用 GPU 总显存的 50%；
可通过 `--memory-fraction` 调整。输出各分支梯度、FPS 重复性检查、各层有效
体素采样比例和显存峰值。单帧调试只需换成 `1f` 配置。

```bash
python3.10 -m pytest -q tests/test_opus_voxel_fusion.py
```

测试包括 CPU/CUDA 的 FPS 行顺序不变性、边界与空邻域、分组/跨样本采样对照、
坐标/特征梯度、两帧双模态融合反传和配置继承。无 CUDA 时对应 GPU 用例会跳过。

## 本机验证记录

2026-10-29，RTX 4090：新模型测试与现有 OPUS loss/head/encoder 接口回归共
24 项通过（包含 CPU 和 CUDA）。两份配置均对真实验证集第 128 个样本完成
完整前向、OPUS 集合损失及反向传播，未缩小 query 数量或输出点数。

| 项目 | 单帧图像配置 | 8 帧图像配置 |
| --- | --- | --- |
| 相机图像数 | 6 | 48 |
| 合并后的输入点数 | 269,036 | 269,036 |
| 有效 memory 体素数 | 58,998 | 58,998 |
| 体素初始化的 query 数 | 600 | 600 |
| 最终预测点数 | 76,800 | 76,800 |
| PyTorch 峰值 allocated 显存 | 2,575.81 MiB | 5,223.62 MiB |

图像 backbone、LiDAR backbone、memory projection、动态混合、采样偏移和最终
分类/位置分支均得到有限且非零梯度。以上是随机初始化模型的单样本检查，未
分配优化器状态，因此显存数字不代表正式训练峰值；也不包含精度评估结论。
详细记录位于 `work_dirs/opus_voxel_fusion_smoke/1f.json` 和 `8f.json`。
