# React 执行效率优化：第二轮

第三轮渲染器配置、编译与测量见 [README_execution_fast_v3.md](README_execution_fast_v3.md)。

入口：`img_voxel_react_occ3d_fast_v2.py`，继承第一轮 `fast.py`。
checkpoint 的参数键保持兼容，仍使用原图像分辨率、FP32、backbone checkpoint、
Gaussian 数量、9 个采样点、三层 CE/Lovasz 监督和原训练/评估 mask。
没有开启 BF16，也没有修改采样权重的归一化或 dropout。

## 修改

1. 常规训练/评估不再保留 `ori_img`，并在加载器中跳过原图深拷贝。原先六张
   1600×900 FP32 原图约 98.9 MiB，会被 collate 后上传 GPU，模型和 loss 不使用它。
   原加载器默认仍保留原图，既有配置的行为不变。
2. 将 React 的 **同一个 CPU Point2Voxel 算法**移到数据 worker。点顺序、原始
   0.05 m 体素、每体素最多 5 点、最多 1,600,000 个体素保持不变。仍在 GPU 上
   执行原 MeanVFE 和 spconv；新增 `react_voxels` 批数据和参数匹配检查，避免
   配置不一致时静默使用错误的预处理。原 raw LiDAR 路径继续可用。
3. 每卡 2 个持久 worker，`prefetch_factor=1`。DataLoader 构建器透传这些参数，
   `num_workers=0` 时自动忽略只适用于多进程的选项。既有 pin_memory 配合新增的
   非阻塞 Tensor/voxel 传输，在 train.py 和 eval.py 均生效。raw LiDAR 保持在 CPU。
4. 新增 DAF CUDA 前向采样与 keypoint 求和融合：直接输出 `[B,N,C]`，不再落地
   `[B,N,K,C]`。反向直接读取 query 梯度，避免展开求和梯度。K=9、N=60000、C=128
   时，相比原路径单算子实测峰值减少约 264 MiB。没有减少采样、相机或 FPN 层级。
   新旧 DAF kernel 都在 PyTorch 当前 CUDA stream 上运行，并检查 launch 错误。

worker 的 native voxelizer 惰性构造，持久保存自己的缓冲区。四卡配置共 8 个
worker；CPU 内存和 CPU 核数有限时，可以将每卡 worker 降到 1。设成 0 也可运行，
但没有 worker 与 GPU 之间的预处理重叠。已有第一轮固定 Occ3D 查询网格缓存的
适用条件继续成立，见 `README_execution_fast.md`。

## 编译与使用

**本轮需要重新编译 DAF。远端四卡机器也需要针对其 PyTorch/CUDA/GPU 编译。**
在项目根目录执行：

```bash
cd model/encoder/gaussian_encoder/ops
MAX_JOBS=2 python3.10 setup.py build_ext --inplace
cd ../../../..

python3.10 train.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v2.py \
  --work-dir out/img_voxel_react_occ3d_fast_v2 --seed 928
```

只使用本轮数据流水线、不编译融合算子时，可以将新配置里的
`model.encoder.deformable_model.fuse_point_sum` 设为 False。

**历史帧数仍继承原配置的 0，不会自动变为 9。** 对比远端九帧历史实验时，应在
派生配置中显式设置两个数据集，保持数据预算一致：

```python
_base_ = ['./img_voxel_react_occ3d_fast_v2.py']
train_dataset_config = dict(num_lidar_history=9)
val_dataset_config = dict(num_lidar_history=9)
```

## 验证与本机测量

单元检查覆盖体素坐标/点数/保留点、不同长度样本的 batch 坐标、worker 序列化、
原图开关不改变模型输入像素、配置预算一致性，以及 K=1/3/9、无效视角、空查询、
非连续输入/梯度、非默认 CUDA stream 下的 DAF 输出和所有输入梯度。
DAF 模块检查还比较了相同 dropout 随机数下全部参数梯度。

最终 GPU/共享接口/训练与评估回归 **57 项通过**。真实 `train.py` 入口用两条样本
（九帧历史）完成了非零学习率的优化器更新、一条样本验证及 checkpoint 保存，
所有梯度启用了有限值检查。短跑 loss 约 9.515→8.200，验证 loss 约 7.589；这些
数字仅用于验证训练链路，不作为精度结果。记录位于
`work_dirs/react_execution_optimization_v2/train_smoke.console.log`。

```bash
python3.10 -m pytest -q \
  tests/test_react_execution.py tests/test_react_execution_v2.py \
  tests/test_gaussian_opus.py tests/test_opus_voxel_fusion.py \
  tests/test_opus_official_v1.py tests/test_opus_multimodal_encoder.py \
  tests/test_opus_head_contract.py tests/test_occ3d_eval_mask.py tests/test_eval_all.py
```

worker 数量变化会改变数据增强的随机数序列，即使 seed 相同也不保证逐步复现
第一轮的训练轨迹；完整精度对照应保持训练预算并重新评估。

基准脚本 `tools/benchmark_react_execution_v2.py` 比较：

- `fast1`：第一轮配置，包括原先的原图副本和主进程体素化。
- `pipeline_only`：第二轮数据流水线，关闭 DAF 融合。
- `fast2`：第二轮完整配置。

RTX 4090，PyTorch 2.12.0+cu126，CPU Torch 线程 24，六张 1600×864 图像，batch=1，
当前帧加 9 帧历史点云；验证集索引 128/256 反复交替，使用相同的随机初始权重。
正序各测 6 步、反序各测 10 步，每组额外预热 2 步；下表合并每条路径的 16 次测量。

| 项目 | 第一轮 fast | 只改流水线 | 第二轮完整 |
| --- | ---: | ---: | ---: |
| 推理流水线每步 | 354.82 ms | 158.33 ms | 155.10 ms |
| 推理的模型 forward | 130.65 ms | 129.23 ms | 127.58 ms |
| 完整训练步 | 632.22 ms | 415.39 ms | 410.54 ms |
| 训练中的数据等待 | 213.76 ms | 0.064 ms | 0.066 ms |
| 训练 allocated 峰值（所有测量最大值） | 11931.84 MiB | 11852.32 MiB | 11848.76 MiB |

完整训练步包括 next(DataLoader)、H2D、forward、三层 CE/Lovasz、backward、梯度裁剪
和 AdamW step。为了固定权重，基准中的 AdamW lr=0，但照常计算/更新优化器状态；
排除 worker 启动/预热、DDP、scheduler、日志和 checkpoint I/O。使用验证场景固定
增强，训练模式仍计算全部训练损失。模块事件记录是 CUDA stream 上的时间区间，
可包含 CPU 调度造成的空闲，不能将各 host/module 时间相加当作 kernel 总时间。

相对于第一轮，完整训练步约 1.54x、推理流水线约 2.29x；**纯模型 forward 仅约
1.02x**。主要改善来自移除无用数据和预取；不能把流水线吞吐当作模型单帧延迟。
预取对在线单帧输入、冷磁盘、完整数据集或四卡机器的收益需分别实测。

单独 60000-query DAF，随机视角有效率约 20%，正/反序两次测量：前向约
2.25–2.26→1.17–1.32 ms（1.71–1.93x），前向+反向约 30.82–31.86→29.01–29.20 ms
（1.06–1.09x）。该数字不包含投影、mask/softmax 和 output projection。
原有整网稀疏/离散路径的重复性限制仍在，算子输出/梯度通过不等于完成训练精度验证。
本轮尚未进行完整数据集的 mIoU 对照实验。

原始记录：

- `work_dirs/react_execution_optimization_v2/paired_lidar10.json`
- `work_dirs/react_execution_optimization_v2/paired_lidar10_reverse.json`

复现完整基准：

```bash
python3.10 tools/benchmark_react_execution_v2.py \
  --phase all --lidar-history 9 --warmup 2 --steps 10 \
  --out work_dirs/react_execution_optimization_v2/recheck.json
```

仅测试算子可用 `--phase daf`，仅训练/推理流水线可用 `--phase train` / `--phase eval`。
`--paths fast2,pipeline_only,fast1` 反转执行顺序；可指定 `--samples` 增加场景覆盖。
