# React 第五轮：图像、数据流水线和融合 DAF

本轮实现前面讨论的第 1–4 步。推荐速度配置
`img_voxel_react_occ3d_fast_v5_fused.py`，低显存配置
`img_voxel_react_occ3d_fast_v5_mem.py`。原配置和 v4 的默认行为保留。

## 配置选择

| 配置后缀 | 内容 |
| --- | --- |
| `fast_v5_image.py` | 第 1–2 步：图像 backbone 局部 BF16、channels_last、固定 BN |
| `fast_v5.py` | 加入第 3 步：worker 图像转换、4 workers/rank、固定网格缓存 |
| `fast_v5_fused.py` | 完成第 4 步：归一化、采样和累加融合 |
| `fast_v5_mem.py` | 在融合版本上开启 FFN/DAF checkpoint |
| `fast_v5_fp32.py` | 保留图像 FP32，使用固定 BN、数据优化和融合 DAF |

继承原有分辨率、点云范围、体素尺寸、Gaussian 数量规则、采样点数、监督层数、
损失和训练 mask。**默认历史帧数仍是 0**；本次性能测量的 9 帧历史需要显式设置：

```python
train_dataset_config = dict(num_lidar_history=9)
val_dataset_config = dict(num_lidar_history=9)
```

模型参数名和形状兼容原 checkpoint。固定 BN 的派生缓存不写入 checkpoint。
BF16 会改变图像数值，兼容权重不代表训练轨迹相同。

## 四步实现

1. `BEVSegmentor.image_backbone_bf16` 和 `image_channels_last` 控制图像执行。
   仅在 image backbone 内开启 BF16 autocast，参数保持 FP32；进入 FPN 的有效
   输出转换为 FP32。FPN、点云网络、几何、DAF、渲染和损失保持 FP32。配置显式
   设置 `amp=False`，避免训练入口再开启全局 FP16。保留 ResNet 原有 checkpoint。
2. `image_frozen_bn` 将固定图像 BN 改成一次 CUDA 仿射运算：
   `y = x * scale + bias`。内部使用 FP32，输出保持输入 dtype 和布局，反向只
   保存 scale。保留 BN 原始参数和统计量的 state_dict 键，加载状态后更新缓存。
   必须固定 BN 参数并启用 `norm_eval=True`；点云训练 BN 保持更新。训练和评测
   入口的 SyncBN 转换会保留固定图像 BN。没有通过重参数化卷积权重改变优化器语义。
3. `CastReactImage` 在 worker 中、FP32 图像增强和归一化后转换图像，早于
   collation/pin_memory；图像 H2D 字节减半。
   `LoadOccupancyOcc3D(defer_xyz=True)` 输出固定 ego 网格规格，传输函数按规格和
   device 缓存原始 FP32 坐标，省去每帧 CPU 网格复制、拼接和 H2D。首次构造沿用
   原 CPU 运算顺序，保持坐标逐元素一致。perturb 和逐场景坐标变换禁止开启此项。
   缓存最多 8 个规格，首次在 inference_mode 中建立的缓存也可供后续训练使用。
   4 workers/rank、prefetch_factor=1 对本机推理有效；四卡对应 16 个 worker，
   需要按另一台机器的 CPU、内存和存储吞吐调整。训练中 2→4 worker 的额外收益较小。
4. `fuse_normalized_sampling=True` 接收独立 query/camera logits、坐标、可见性
   和 dropout mask，直接输出 `[B,N,C]`，不生成全局广播 logits 和完整归一化
   权重 `[B,N,K,V,L,G]`。CUDA 在共享内存中完成 softmax 并双线性采样、累加；
   反向由较小的 logits 和每行 max/倒数分母重建概率。相机梯度按 64 个 query
   分组归约，减少 atomic 竞争。

第 4 步保持 joint `K*V*L` softmax、全不可见行输出零、归一化前 dropout，
以及 dropout 随机数的形状、顺序和数量。偏置只加一次。要求 camera embedding
和 `fuse_point_sum=True`。当前算子支持最多 16 groups、每组最多 32 channels，
共享概率行不超过 48 KiB。本模型为 4×32 channels，9 点×6 相机×4 层，符合要求。
其他模型可以关闭此开关，继续使用 v4 路径。

## 编译与训练

本机已编译。同步到四卡机器后，需在对应 PyTorch/CUDA 环境重新编译：

```bash
cd model/encoder/gaussian_encoder/ops
MAX_JOBS=2 FORCE_CUDA=1 python3.10 setup.py build_ext --inplace
```

回到项目根目录运行：

```bash
python3.10 train.py --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_fused.py --work-dir work_dirs/react_fast_v5
```

不需要覆盖已安装的 `local_aggregate_react`；新增算子位于项目内 DAF 扩展。

## 性能记录

RTX 4090，六相机 1600×864，batch size 1，9 帧历史，验证场景 128/256 交替，
相同随机初始权重。训练时间包含数据等待、H2D、全部监督层 CE/Lovasz、反向、
梯度裁剪和 AdamW，使用 lr=0 固定权重；排除 worker 启动、DDP、日志、调度器
和 checkpoint I/O。推理完整时间包含数据等待和 H2D。

正序 6 次和反序 12 次合计 18 次测量。时间为均值；显存取记录中
**最高 max_memory_allocated**，不是平均峰值或 reserved。

| 配置 | 训练一步 ms | 训练最高 allocated MiB | 推理模型 ms | 推理完整流水线 ms |
| --- | ---: | ---: | ---: | ---: |
| v4 | 367.06 | 9459.33 | 109.40 | 160.98 |
| v5（截至第 3 步） | 274.81 | 7933.59 | 78.49 | 91.46 |
| v5 fused（第 4 步） | 246.61 | 7724.28 | 77.98 | 93.70 |
| v5 mem | 257.48 | 6583.59 | 77.62 | 89.73 |

v5 fused 相比 v4 训练约快 1.49 倍，峰值下降 18.3%；v5 mem 约快 1.43 倍，
峰值下降 30.4%。第四步相对第三步主要提升训练速度：约 275→247 ms。v5 各配置
推理存在几毫秒波动；checkpoint 在 eval 时关闭，不能将这部分差异归因于它。

首轮分步测量中，仅图像优化让推理模型约 110→75 ms，但 2 workers 时完整流水线
仍约 149 ms；worker BF16 转换和 4 workers 才将完整时间降至约 90 ms。

补充单独测量（各 12 次，同样场景及预算）：

| 配置 | 训练一步 ms | 训练最高 allocated MiB | 推理完整流水线 ms |
| --- | ---: | ---: | ---: |
| 原 v4 mem | 381.04 | 8207.42 | 168.39 |
| v5 FP32 | 331.53 | 8613.54 | 117.37 |

v5 FP32 为不使用图像 BF16 的选择；它没有开启 decoder checkpoint，因此比原
v4 mem 的 allocated 更高。v5 mem 相对原 v4 mem 的峰值仍下降约 19.8%，训练
加速约 1.48 倍。

原始记录位于 `work_dirs/react_execution_optimization_v5/`：
`initial.json`、`reverse.json`、`combined.json`，以及补测 FP32 和原 v4 mem 的
`fp32_and_v4_mem.json`。复现示例：

```bash
python3.10 tools/benchmark_react_execution_v2.py --phase all --paths fast4,fast5,fast5_fused,fast5_mem --lidar-history 9 --warmup 3 --steps 12 --out work_dirs/react_v5_comparison.json
```

benchmark 独立构建各模型，严格恢复同一 state_dict。新 summary 的 `peak_mib`
取记录最大值，另列 `mean_peak_mib`；records 保留每步峰值。前两次 JSON 的旧
summary 使用均值，因此上表从 records 重新计算最高峰。

## 验证及边界

- 相关回归测试 113 项通过，覆盖新旧 decoder、renderer、OPUS、数据、mask/eval。
  新测试检查固定 BN 的 dtype/布局/梯度/状态加载，网格精确值及缓存，SyncBN 保留
  规则，多 batch、全不可见、dropout、所有输入及参数梯度、checkpoint 重算和独立
  CUDA stream，并检查融合 DAF 不保存完整 FP32 概率矩阵。
- 冻结真实 DAF 输入，v4 对比第四步算子：输出最大绝对差 `1.49e-7`，全部输入/
  参数梯度最大相对 L2 差 `5.63e-7`，参数梯度最大相对 L2 差 `4.62e-7`。
  记录在 `real_daf_audit.json`。
- fused 和 mem 均通过真实 `train.py` 的 2 步非零学习率训练、1 次验证和 checkpoint
  保存，启用 `fail_on_nonfinite_grad=True`。对应 `train_smoke_*.console.log`。
- 本机只有单 GPU，尚未验证四卡 DDP 吞吐。
- **尚未验证完整训练 mIoU。** 梯度一致性只证明 FP32 融合 DAF 的数值一致；图像
  BF16 的精度影响需用相同 seed、历史帧和训练预算做完整对照。需要保留图像 FP32
  时使用 `fast_v5_fp32.py`。
