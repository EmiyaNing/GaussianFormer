# React 执行效率优化：第一轮

入口配置：`img_voxel_react_occ3d_fast.py`，继承原 `img_voxel_react_occ3d.py`。
模型参数及 checkpoint 键保持兼容；图像分辨率、数据、历史帧数、Gaussian 数量、
采样点数、精度设置、CE/Lovasz 和三层监督均沿用父配置。原配置各开关默认关闭。

## 本轮修改

1. **GPU 解析协方差逆**：用 `R.T @ diag(scale^-2) @ R` 替换
   `Cov.cpu().inverse().cuda()`，包括其反向中的 CPU 路径。
2. **每次 forward 只整理一次图像特征**：两层 deformable aggregation 共用带
   autograd 图的连续布局，移除重复 clone/cat/contiguous。
3. **延后 anchor embedding 刷新**：最后一次 refine 后不再运行没有消费者的
   embedding 网络。末尾 DensifyOnly 跳过不再使用的 anchor 和 feature 输出。
4. **训练增量渲染**：DensifyOnly 明确返回 parent 索引及新增高斯。若 parent 已
   被当前监督策略渲染，最终 logits 复用 parent，只额外渲染新高斯；三层 loss
   仍分别计算。推理只有最终层，直接渲染全部最终高斯。
5. **背景高斯分离**：固定背景几何的空间权重单独计算并缓存，乘可学习的
   `empty_scalar` 后加入 logits。保留原背景的空间分布和 AABB 支持，未替换为
   常数背景。避免对背景高斯展开全网格邻居与排序。
6. **固定查询网格缓存**：缓存 occupancy 查询点的整数网格坐标，关闭热路径中
   GPU 标量范围断言。普通路径仍保留断言。
7. **输入兼容修复**：原 voxel lifter 现在同时接受 ragged NumPy/Tensor 点云。
   体素化算法、点顺序及每体素点数上限没有改变。

Python 包装器位于 `model/head/react_local_aggregate.py`，复用已安装的 React
CUDA/autograd 扩展。**这轮无需重编译或重新安装 CUDA 扩展。**

本轮未修改采样 CUDA、体素化位置或数据加载协议。也未启用混合精度、改变
checkpoint 或减少任何模型计算预算。这些可以在后续单独分析。

## 使用与约束

训练：

```bash
python3.10 train.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast.py \
  --work-dir out/img_voxel_react_occ3d_fast --seed 928
```

原配置默认 `num_lidar_history=0`；如果远端原实验已设置 9 帧历史，需要在新配置
的 `train_dataset_config` 和 `val_dataset_config` 同样设置为 9，再比较性能。
该配置不会自动把既有实验的数据预算改变为 10 帧点云。

已有完整 React checkpoint 可用 `eval.py --resume-from` 或训练恢复入口加载，
严格 state-dict 加载已验证兼容。缓存不写入 checkpoint，加载后会重建。

`cache_render_grid=True` 仅适用于继承配置中的**固定、未扰动、顺序不变的完整
ego-frame Occ3D 网格**。若改变 `occ_xyz` 的坐标、顺序、查询子集或启用 perturb，
将该开关设为 False。可将 `head.cuda_kwargs.validate_inputs=True` 恢复调试断言。

## 验证与实测

单元/数值测试：

```bash
python3.10 -m pytest -q tests/test_react_execution.py
```

覆盖：协方差逆及尺度/旋转梯度、末尾 densify 输出和所有参数梯度、CPU 参考与
真实 CUDA 的背景拆分和增量渲染、FPN 布局复用后的采样/梯度、点云输入兼容
以及配置数据/模型预算一致性。

最终连同 Gaussian/OPUS 共享接口及之前的 voxel-fusion 模型回归，共 36 项测试通过。

RTX 4090，本机 PyTorch CPU 默认线程数/interop 线程数均为 24。原始 `1600x864`
六相机输入，验证集索引 128/256，每样本预热一轮、
每条路径正式测三次，交替运行顺序。两条路径使用同一组**随机初始化权重**。
计时包含 forward 中的 CPU 体素化，训练还包含三层 CE/Lovasz 和 backward；
不包含数据集读取、H2D、优化器 step/state。下表为六次正式测量的平均值。

| 点云历史 | 项目 | 原版 | fast | 加速比 |
| --- | --- | --- | --- | --- |
| 0 | 推理 forward | 103.57 ms | 94.93 ms | 1.09x |
| 0 | 训练 forward + loss + backward | 1490.96 ms | 267.53 ms | 5.57x |
| 9 | 推理 forward | 140.81 ms | 129.47 ms | 1.09x |
| 9 | 训练 forward + loss + backward | 1626.81 ms | 391.93 ms | 4.15x |

9 帧历史时 PyTorch 训练 allocated 峰值由 12,299 MiB 降至 11,700 MiB。推理
峰值均约 1,997 MiB，主要受图像特征提取影响。本结果是两个样本的短时定位
测量，不是已训练模型、四卡吞吐或完整训练显存的最终结论。

单独固定真实场景的同一组 58,998 个主体高斯及 2,560 个新增高斯，比较原版
CPU 求逆/完整渲染与 fast：三层 logits 最大绝对误差不超过 `2.39e-6`；全部
高斯位置、尺度、旋转、opacity、semantic 和 empty scalar 的梯度合计相对 L2
误差为 `8.93e-6`。该检查通过。

**整网逐点等价尚未建立。** 原版在同一组权重、随机种子及排序后的体素输入下
重复运行，后续层 logits 也存在差异；9 历史样本的原版重复最大差约 2.88，
优化版对原版约 2.40，而第一层都在 `4e-5` 内。原版重复的全参数梯度相对
L2 差为约 0.156，优化版约 0.149。这说明整网比较受到原有稀疏/离散路径
重复性的干扰，但尚未定位到具体 kernel，不能宣称这些全网差异已严格解释。
一致性审核中的体素排序仅用于审核，不进入正式计时；独立算子检查使用完全
相同输入。未重新训练或验证完整数据集 mIoU，不以数值测试代替精度对照。

原始记录：

- `work_dirs/react_execution_optimization/paired.json`
- `work_dirs/react_execution_optimization/paired_lidar10.json`

复现（当前帧加 9 帧历史，同时作用于原版和 fast）：

```bash
python3.10 tools/benchmark_react_execution.py \
  --lidar-history 9 --samples 128,256 --warmup 1 --repeats 3 \
  --out work_dirs/react_execution_optimization/recheck.json
```

训练的大幅改善来自本机原路径包含 CPU 求逆及其反向等执行开销；具体收益
依赖 CPU 线程、硬件和高斯分布。真实已训练 checkpoint 和四卡环境仍需复测。
