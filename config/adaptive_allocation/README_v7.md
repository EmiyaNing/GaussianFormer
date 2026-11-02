# AdaptiveAllocationV7：固定预算、共享生成与局部收益监督

入口为 `adaptive_allocationv7_occ3d.py`，使用已有 React v5 FP32 执行路径。
V7 本身没有新增 CUDA 扩展，已有 v5 环境无需额外编译。

## 代码阅读顺序

- `model/encoder/gaussian_encoder/topk_module/adaptive_allocationv7.py`：`_run`
  串联状态编码、固定预算选择、父/child 预测和训练探针。父修正与 child 生成分别
  位于 `_predict_parent`、`_predict_child`；终端来源信息和 anchor 转换分开处理。
- `loss/adaptive_allocation_v7_loss.py`：先读 `gaussian_field` 和
  `local_gain_targets` 理解监督目标，再读 `_score_losses`、`_stability_loss`。
- `tools/adaptive_allocation_v7_common.py`：脚本共用的构建器、缓存传输、Gaussian
  字段操作、渲染和指标，训练/评测使用同一套辅助函数。
- `tools/experiment_adaptive_allocation_v7.py`：`prepare`/`_cache_split` 准备缓存，
  `_train_arm` 训练单组，`_evaluate_allocator` 做固定终点验证。
- `tools/evaluate_adaptive_allocation_v7.py`：`_benchmark_allocator` 计时，
  `_paired_scene_bootstrap` 统计配对区间，两者分开。
- `tools/export_adaptive_allocation_v7.py`：`export_checkpoint` 合并并严格核对完整权重。

本次可读性重构保留计算公式、随机数调用、默认配置、CLI 参数、输出字段和
state_dict 键。重构前源码及验证记录位于
`work_dirs/adaptive_allocation_v7_refactor/`：76 组 CPU/CUDA 核心对照逐元素一致，
126 项相关测试通过；五组短训练最终验证结果一致，导出的 634 个状态张量与元数据
一致。短训练损失最大差约 9.54e-7，权重最大差约 5.83e-7，详见 `verification.json`。
原子集试验的结果与 source audit 保留，重构版本另存审计摘要。

## 模块行为

- 放在最后一次 refine 之后。共享状态编码输入 decoder 特征、归一化位置、尺度、
  opacity 和三个语义统计，不再进行全量 GS 的 FRNN 邻域查询。
- 收益头只决定 TopK 顺序，不缩放任何父 GS 的 opacity。
- 先选父 GS，再运行一个共享生成头，输出父残差及一个 child；不分 clone/split/atten
  专家，也不先生成 4N 个完整 GS。
- `K=floor(N*allocation_ratio)`，训练/推理输出均为 N+K；可选
  `max_new_gaussians` 再限制 K。默认 ratio=.15，与当前本机 v6 配置对齐。
  `adaptive_allocationv7_occ3d_160.py` 显式设为 .6，对应最多 1.6N。
- 父残差零初始化，opacity 不经过 risk gate；child 继承父旋转，初始 opacity 为
  父的 .02，并有 .25 倍最大尺度的非零主轴位移。语义默认沿用 softplus 参数化。
  初始父场近似恒等，child 贡献较小且保持几何梯度。
- 训练早期在固定 K 内留少量随机探索，随后衰减到纯评分选择；eval 不使用随机数，
  不生成 GT 监督探针。`selection_mode='opacity'` 是同结构选点对照。
- `update_parent=False` 可做纯 append 对照；当前没有物理 pruning/compaction。

V7 返回终端替换来源，encoder/head 使用：

```
新场 = 原场 + 更新后的选中父 GS + child - 被替换的旧父 GS
```

完整输出仍是 N+K 个 GS，增量路径不改变输出定义。若禁用增量渲染，会完整渲染该
GS 集合。两条路径的输出和生成器梯度有独立测试。

## 评分训练

`AdaptiveAllocationV7Loss` 每帧随机选 128 个探针父 GS，使用同一生成头在
`no_grad` 下提出候选。每个候选在旧父及 child 中心附近各取 5×5×5 个网格点，
去重、应用训练 mask 和有效标签，再比较操作前后的加权 CE：

```
gain_i = mean_valid_voxels[CE(base, GT) - CE(base + local_delta_i, GT)]
```

局部场使用与 React 相同的协方差方向、整数 AABB 支持和加性语义贡献。收益目标
停止梯度，按帧标准化后监督收益回归和排序；原有 occupancy CE/Lovasz 保留。
开始阶段另有逐渐衰减的场一致性损失。没有三专家 KL、强制降低各向异性或强制
子 GS 语义分化。

这是有限邻域、固定其他 GS 时的局部 CE 收益代理，不是全局 mIoU oracle。
推理只读取特征及 GS 属性，不读取标签。收益监督要求完整、固定、xyz-major 的
Occ3D 网格；动态/扰动坐标需要另行实现对应采样器。

默认配置冻结 backbone、lifter、head 和原 decoder，只训练 allocation 的 40,881
个参数。`frozen_modules_eval=True` 同时固定冻结模块的 BN 统计和 dropout；head
仍保留训练输出/监督逻辑。`adaptive_allocationv7_occ3d_decoder_ft.py` 为第二阶段
解冻 decoder 的入口，图像和 LiDAR 主干仍固定。

## 运行配置

| 配置 | 用途 |
| --- | --- |
| `adaptive_allocationv7_occ3d.py` | 默认 1.15N，共享父修正与 child，收益评分 |
| `adaptive_allocationv7_occ3d_opacity.py` | 相同生成器和损失，按 opacity 选父 GS |
| `adaptive_allocationv7_occ3d_160.py` | 1.6N 预算功能配置 |
| `adaptive_allocationv7_occ3d_decoder_ft.py` | 加载导出的 V7 checkpoint，解冻 decoder |

配置显式使用 9 帧历史、FP32、原分辨率和原训练/评测 mask。默认加载实验生成的
`work_dirs/adaptive_allocation_v7_pilot/warmstart_occ3d.pth`；有匹配当前布局的
Occ3D checkpoint 时，应将 `load_from` 改为该权重。

```bash
python3.10 train.py --py-config config/adaptive_allocation/adaptive_allocationv7_occ3d.py --work-dir work_dirs/allocation_v7_occ3d
```

## 已完成的实验

本机 RTX 4090。官方训练集中 32 个场景、验证集中 16 个场景，每场景取一个中间
帧，场景集合不重叠。冻结主干后缓存确定性增强的完整 FP32 特征，统一 9 帧历史。
每组种子 928/929、192 次更新，即在这 32 帧上重复 6 轮；lr=2e-4，AdamW，
gradient clip=35，监督最终输出的相同 occupancy CE/Lovasz。使用固定最后一步
结果，没有依据验证集选择最佳迭代。

公共主干来自本机 `ckpts/adaptive_allocationv4.pth`。它来自旧 SurroundOcc 实验，
不是另一台机器上的 Occ3D checkpoint。准备脚本移除旧中间 allocator、显式重排
后续层索引，检查全部主干参数匹配，并保留当前 Occ3D 坐标/网格/背景几何 buffer。
因此以下是**移除旧 allocator 后的公共冻结主干上的跨配置模块试验**，不能与远端
img_voxel_lite 的完整训练结果直接比较。

为控制位置因素，v6 对照也放在终端，使用原来的生成器、路由及辅助损失。额外将
大批量 3×3 eigendecomposition 分块，以绕开本机 cuSOLVER 报错；目标公式不变。

指标为固定 17 个前景类、全验证帧聚合的 camera-mask mIoU，17 类在本子集中均有
有效 union；同时保存未加 camera mask 的指标。两个种子的结果为：

| 方法 | seed 928 | seed 929 | 平均 camera mIoU | 平均无 mask mIoU |
| --- | ---: | ---: | ---: | ---: |
| 公共主干，未加 allocator | 20.767 | 20.767 | 20.767 | 9.157 |
| v6，终端对照 | 21.347 | 21.155 | 21.251 | 9.500 |
| v7，opacity 选点 | 20.058 | 19.882 | 19.970 | 9.431 |
| v7，收益选点＋父残差＋child | 21.647 | 21.964 | 21.806 | 10.695 |
| v7，仅追加 child | 20.767 | 20.767 | 20.767 | 9.156 |
| v7，仅修正选中父 GS，无 child（后续诊断） | 21.396 | 21.923 | 21.660 | 10.669 |

验证集平均输入 44,753.75 个 GS；最终 v6 和标准 v7 都平均输出 51,466.25 个，
因此最终结果的实际数量也匹配，不只是相同上限。父修正诊断输出原始 N 个 GS。

当前证据：收益选点优于固定 opacity 策略的均值，V7 的主要增益来自父修正和选点。
新增 child 相对“仅父修正”的额外平均收益约 0.146 个百分点，纯 append 的 child
贡献在训练中趋近零。不能将本轮提升解释成已经充分利用了新增 GS 容量。

按 16 场景配对 bootstrap、固定这两个已训练种子，V7-v6 的 mIoU 差值区间约
[-1.020,+1.377]，V7-仅父修正约 [-.060,+.284]，均包含零。
**子集试验尚不足以证明稳定精度优势，未跑完整 Occ3D 长训练。**

### 执行成本

| 方法 | 缓存特征上的模块训练步 ms | 模块训练最高 allocated MiB | 分配模块＋完整渲染推理 ms |
| --- | ---: | ---: | ---: |
| v6 | 33.51 | 5164.84 | 7.64 |
| v7 opacity | 17.21 | 758.76 | 3.57 |
| v7 收益评分 | 17.49 | 758.76 | 3.64 |
| v7 仅追加 child | 15.98 | 751.74 | 3.54 |
| v7 仅父修正 | 17.16 | 747.12 | 3.39 |

训练时间包含缓存输入的 H2D、module、渲染、主/辅助损失、反向和 AdamW；不含
图像/点云 backbone 和数据加载。显存包含该模块试验的输入及 head 缓存，不是
完整模型训练显存。推理统一完整渲染最终 GS 集合，每帧预热一次、测量 5 次，排除
H2D 和主干提取；没有给 v7 免费复用父场来比较推理速度。约 2.1 倍加速只针对
分配模块＋渲染，不能外推为整网加速。

另已完成：

- 相关测试合计 126 项通过，其中 V7 的 13 项覆盖固定预算、极小/零增长、梯度、
  初始父场、无随机数推理、checkpoint 步数恢复、局部收益/训练 mask、旋转场与
  CUDA renderer 一致性、替换增量梯度和冻结统计量。
- 真实 `train.py` 完成两步非零学习率训练、一次验证及 checkpoint 保存。
  单帧 smoke 的旧训练器 mIoU 不作为上述实验精度指标。
- 1.6N 功能检查完成三步真实缓存特征训练：58,060→92,896 GS，梯度和损失有限。
  这只验证功能，不代表已验证 1.6N 的精度收益。
- 导出 seed 928 的完整模型 `learned_s928_full.pth`，634 个 state tensor 严格加载
  成功。该种子预先按编号选作导出示例，没有挑选验证最高的 seed 929。
- 本机只有单卡，未做四卡 DDP 吞吐或完整数据长训练。

## 复现与产物

下面命令使用默认输出目录；已有本轮产物时，请在新目录复现并为后续命令传入
相同 `--out/--experiment`。准备脚本会拒绝覆盖已完成的 protocol。

```bash
python3.10 tools/experiment_adaptive_allocation_v7.py --phase all --train-frames 32 --val-frames 16 --steps 192 --seeds 928,929 --arms v6,opacity,learned,append --out work_dirs/adaptive_allocation_v7_pilot
python3.10 tools/experiment_adaptive_allocation_v7.py --phase train --steps 192 --seeds 928,929 --arms parent_only --out work_dirs/adaptive_allocation_v7_pilot
python3.10 tools/evaluate_adaptive_allocation_v7.py --repeats 5
python3.10 tools/export_adaptive_allocation_v7.py --seed 928
```

使用自己的终端布局 Occ3D 权重时，第一条增加
`--checkpoint YOUR_CHECKPOINT --checkpoint-layout terminal`；旧中间 allocator
布局使用 `mid`。脚本遇到主干参数缺失/形状不符会失败，不会静默部分加载。

产物在 `work_dirs/adaptive_allocation_v7_pilot/`：

- `protocol.json`：场景索引、权重 SHA256、层映射、预算和训练条件。
- `results.json`、`summary.json`：逐组初始/最终指标、训练轨迹及汇总。
- `inference.json`：逐帧推理测量、场景混淆矩阵和配对区间。
- `source_audit.json`：实现文件摘要。
- `learned_s928.pth`：仅 allocator；`learned_s928_full.pth/.py`：完整可加载模型及配置。
- `train_smoke.console.log`：完整训练入口验证。

若需继续训练已导出的模块，使用 `learned_s928_full.py` 或在第二阶段配置中设置
相应 `load_from`。不要把仅 allocator 的 `.pth` 直接当作完整模型权重加载。
