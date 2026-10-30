# React 执行效率优化：第三轮渲染器

后续 decoder 显存优化与配置见 [README_execution_fast_v4.md](README_execution_fast_v4.md)。

入口：`img_voxel_react_occ3d_fast_v3.py`，继承第二轮 `fast_v2.py`。
仅切换渲染器实现，保留 FP32、Gaussian/采样点数量、图像分辨率、历史帧数、
模型参数、全部 CE/Lovasz 监督及 mask。checkpoint 参数键与第二轮兼容。

## 实现

原 React 每个 Gaussian 枚举其 AABB 中所有体素，展开 `(voxel, gaussian)` 记录，
做 CUB radix sort 后再逐点累加；反向保留这些记录，并由一个线程遍历整个
Gaussian 支持域。新实现位于 `model/head/react_render_ops/`，Python 包装器为
`model/head/react_tiled_aggregate.py`，不会覆盖已安装的 `local_aggregate_react`。

1. **空间块 CSR 候选列表**。以默认 4×4×4 个体素为一块，执行计数、前缀和、
   scatter，直接构造 `block → Gaussian IDs`。前向对候选仍执行原始逐体素
   AABB 检查，然后计算同一个 Gaussian 二次型、opacity 与语义加权和。
   空间块仅用于加速查找，不改变查询分辨率或支持域。
2. **去掉全局排序及邻接数量回读**。全部计数和分配后的写入在 GPU 上完成。
   CSR 前缀使用 int64，避免高覆盖量的累加溢出。无需把总邻接数拷回 CPU 来
   决定动态缓冲大小。
3. **容量提示不截断贡献**。`radius_hint=5` 用于估算 workspace，符合当前
   `ceil(0.64×3/0.4)=5` 的模型半径上限。若实际邻接超出容量，相关 block 在
   GPU 上直接遍历全部 Gaussian 并做相同支持域检查；计算仍完整，但会变慢。
   修改模型尺度/网格后应相应调整提示。没有丢弃弱贡献、高斯或远处体素。
4. **不为反向保存展开列表**。反向从 Gaussian 的同一个整数 AABB 重新枚举
   支持体素，通过 query map 找到实际查询点。CSR/排序列表均不进入 autograd
   保存状态。query map 支持稀疏、乱序以及同一体素内的多个查询点。
5. **warp 协作反向与公共梯度提取**。每个 warp 处理一个 Gaussian，各 lane
   处理不同的支持体素，最后做 warp reduction；先计算
   `dot = sum_c(semantic[c] * output_grad[c])`，再一次性计算位置、opacity 和
   协方差梯度，避免在 18 个通道内重复更新几何梯度。
6. **query map 复用**。同一次 head 的多个 render 复用查询拓扑；按 Tensor
   身份和版本失效，state-dict 加载、设备转换时清空。map 不写入 checkpoint。
   无梯度推理不构建 map。所有新 kernel 在 PyTorch 当前 CUDA stream 上运行。

继承的 `cache_render_grid=True` 仍要求固定、未扰动、顺序不变的完整 Occ3D
ego 网格。如果改变查询坐标或子集，需关闭该既有开关；新 renderer 本身支持
非完整查询。当前 CUDA 渲染器与原 React 一样使用 **18 个语义通道**，查询点
不参与求梯度，半径离散化不求导。候选顺序和反向 reduction 顺序变化会产生
FP32 舍入差异，不能据此承诺逐 bit 一致。

## 编译和启动

本机已编译，远端机器需要用其 PyTorch/CUDA/GPU 编译一次。第二轮 DAF 扩展
仍需安装；新 renderer 是独立的项目内扩展，无需重新 pip 安装旧 React 包。

```bash
cd model/head/react_render_ops
MAX_JOBS=2 python3.10 setup.py build_ext --inplace
cd ../../..

python3.10 train.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v3.py \
  --work-dir out/img_voxel_react_occ3d_fast_v3 --seed 928
```

历史帧数仍继承本地原配置的 **0**。延续远端九帧历史实验时，派生配置应设置：

```python
_base_ = ['./img_voxel_react_occ3d_fast_v3.py']
train_dataset_config = dict(num_lidar_history=9)
val_dataset_config = dict(num_lidar_history=9)
```

对照直接用 `fast_v2.py`。若在第三轮派生配置中切回 `react_renderer='legacy'`，
还需移除其 `cuda_kwargs` 中仅属于新 renderer 的 `tile_size`、`radius_hint`、
`cache_query_map` 参数。

## 固定真实渲染输入的测量

RTX 4090，PyTorch 2.12.0+cu126。从验证集索引 128、九帧历史、第二轮模型的
随机权重固定同一组 first/parent/delta/final 输入，640000 个 occupancy 查询点。
每种路径预热 3 次、正式测 10 次，交替执行顺序。计时包含 Python 包装器、
整数中心/半径和 inverse 格式整理；反向测量同时求位置、opacity、semantic、
inverse covariance 梯度。以下是最终层 61558 个 Gaussian 的平均结果。

| 项目 | 第二轮渲染器 | 第三轮渲染器 |
| --- | ---: | ---: |
| 前向 | 3.551 ms | 0.848 ms |
| 前向+反向 | 5.360 ms | 1.600 ms |
| 前向+反向新增 allocated 峰值 | 461.06 MiB | 62.57 MiB |
| 构建的邻接记录 | 17346924 个体素记录 | 769262 个 block 记录 |

前向约 **4.19x**，前向+反向约 **3.35x**；单算子新增显存峰值约减少 **86%**。
全部真实输入均未触发容量回退。2×2×2 块也做了对照：最终层前向约 1.120 ms、
前向+反向约 1.923 ms、峰值约 98.86 MiB，因此此配置保留 4×4×4。

Nsight 对原渲染器第一层前向的采样显示：radix sort 约 1.37 ms、逐体素展开
约 1.00 ms、实际累加约 1.14 ms，展开和排序合计约占 kernel 时间的 64%。
记录位于 `work_dirs/react_execution_optimization_v3/legacy_profile.nsys-rep` 和
`legacy_profile_stats.csv`。此分解只针对该次前向，不是全模型耗时占比。

四组固定真实输入的 logits 最大绝对误差约 **2.87e-6**；位置、opacity、semantic
和 inverse 梯度各自相对 L2 误差均低于 **1.30e-6**。独立 PyTorch 参考测试还
覆盖旋转/尺度通过解析 inverse 的梯度、背景高斯、增量渲染、边界/部分块、
空输入、非默认 stream、查询缓存失效以及强制容量回退。

## 整网测量与回归

保持第二轮基准的定义：六张 1600×864 图像、batch=1、九帧历史、验证集索引
128/256，随机初始权重；完整训练步包含数据等待、H2D、全部 loss、backward、
梯度裁剪和 AdamW（lr=0 以固定模型权重）。每条路径正序 16 次、反序 16 次，
每组另预热 3 次。排除 worker 启动、scheduler、DDP、日志和 checkpoint I/O。

| 项目 | 第二轮 | 第三轮 |
| --- | ---: | ---: |
| 完整训练步 | 411.00 ms | 401.69 ms |
| 训练 head 前向 | 10.41 ms | 4.84 ms |
| 训练 allocated 峰值（所有测量最大值） | 11845.99 MiB | 11017.25 MiB |
| 纯模型推理 forward | 127.27 ms | 126.22 ms |
| 包含数据加载的推理步 | 159.22 ms | 162.98 ms |

完整训练步约 **1.023x**，最大 allocated 峰值减少约 **829 MiB / 7.0%**。
推理流水线本次未见稳定加速：数据等待平均约 25.71→30.62 ms，且执行顺序之间
存在波动；不能用渲染算子的 4.19x 声称整网推理也同幅改善。

最终渲染器和前两轮/共享接口/训练评估回归 **71 项通过**。完整回归命令：

```bash
python3.10 -m pytest -q \
  tests/test_react_renderer.py tests/test_react_execution.py tests/test_react_execution_v2.py \
  tests/test_gaussian_opus.py tests/test_opus_voxel_fusion.py tests/test_opus_official_v1.py \
  tests/test_opus_multimodal_encoder.py tests/test_opus_head_contract.py \
  tests/test_occ3d_eval_mask.py tests/test_eval_all.py
```

真实 `train.py` 入口还用两条九帧历史样本完成了非零学习率的更新、一条样本
验证和 checkpoint 保存，启用全部梯度有限值检查。短跑 loss 约 9.515→8.163，
验证 loss 约 7.609；仅验证训练链路，不作为精度指标。配置和日志位于
`work_dirs/react_execution_optimization_v3/train_smoke.py` 与
`work_dirs/react_execution_optimization_v3/train_smoke.console.log`。

算子大幅改善不代表整网有相同加速。完整训练的 image/encoder backward 等开销
仍然存在。旧稀疏/离散网络路径的重复性限制也仍在；本轮没有完整训练的 mIoU
对照，不能把固定输入算子等价测试当作训练精度实验。

## 复现

```bash
python3.10 -m pytest -q tests/test_react_renderer.py

python3.10 tools/benchmark_react_renderer.py \
  --tile-size 4 --warmup 3 --repeats 10 \
  --out work_dirs/react_execution_optimization_v3/renderer_recheck.json

python3.10 tools/benchmark_react_execution_v2.py \
  --phase all --paths fast2,fast3 --lidar-history 9 --warmup 3 --steps 16 \
  --out work_dirs/react_execution_optimization_v3/whole_recheck.json
```

`benchmark_react_renderer.py` 首次生成并保存同一组真实输入，再供各实现共同使用。
用 `--inputs 新文件路径 --sample 其他索引` 扩大场景覆盖；如需已训练模型输入，
可同时提供 `--checkpoint checkpoint.pth`，加载时使用 strict=True。

原始算子记录：

- `work_dirs/react_execution_optimization_v3/renderer_tile4_final.json`
- `work_dirs/react_execution_optimization_v3/renderer_tile2.json`
- `work_dirs/react_execution_optimization_v3/whole_model_final.json`
- `work_dirs/react_execution_optimization_v3/whole_model_final_reverse.json`
- `work_dirs/react_execution_optimization_v3/whole_model_combined.json`
