# OPUS 迁移与训练设计

## 1. 目标与范围

将 **OPUS: Occupancy Prediction Using a Sparse Set（原始 OPUS / V1）** 迁移到本仓库，使其能在现有 NuScenes + Occ3D 数据、训练、评估和可视化流程中稳定训练。OPUS 将占据预测表述为稀疏的三维点集预测：查询直接预测占据点坐标和语义，不建立 Gaussian 表示，也不在网络内部维护稠密体素特征。[论文](https://arxiv.org/abs/2409.09350) [官方实现](https://github.com/jbwang1997/OPUS)

本次迁移的首个可训练目标为：

- 数据集：Occ3D-nuScenes，`pc_range=[-40,-40,-1,40,40,5.4]`，`0.4m` 体素，`200 x 200 x 16`。
- 输入：6 路相机单帧、`704 x 256`，ResNet-50 + FPN。
- 输出：最终稀疏点集及语义；通过确定性的栅格化适配为当前评估和可视化所需的 `final_occ`。
- 保留：`visualize.py`、Open3D 可视化、Occ3D mIoU/RayIoU 路径、断点恢复、DDP、TensorBoard。
- 不修改现有 Gaussian / adaptive allocation 代码的行为，也不复用 `GaussianHead`、`localagg*` CUDA 算子。

先实现 V1，不把 OPUS-V2 的连续到离散投影（CPD）混入首版。V1 的 Chamfer 距离、最近邻语义分配、coarse-to-fine、consistent point sampling 和 adaptive re-weighting 是可验证的最小闭环；V2 作为 V1 收敛后独立的第二阶段。

## 2. 现状与迁移边界

当前仓库已经提供了可复用的外层能力：

| 现有模块 | 迁移中的使用方式 |
| --- | --- |
| `model/segmentor/bev_segmentor.py` | 复用其多相机 backbone/FPN 特征提取约定（`ms_img_feats: [B,N,C,H,W]`），不改变 Gaussian 管线。 |
| `dataset/transform_3d.py:LoadOccupancyOcc3D` | 复用 `occ_xyz`、`occ_label`、`occ_*_mask` 及 ego 坐标约定。 |
| `train.py`、`loss/multi_loss.py` | 复用配置构建、loss 输入映射、日志、DDP 和 checkpoint 流程。 |
| `eval.py`、`misc/occ3d_nus_metrics.py` | 复用指标；移除 `200 x 200 x 16` 等硬编码，改为读取 OPUS 配置的体素规格。 |
| `visualize.py`、`vis_open3d*.py` | 继续消费栅格化后的 `final_occ`；另加原始预测点集显示。 |

不能直接复制官方代码：官方仓库的原始环境是 PyTorch 1.13.1 + MMCV 1.6 + MMDetection 2.28 + MMSegmentation 0.30 + MMDetection3D 1.0.0rc6，而当前仓库使用的是另一套 MMLab 注册与模型构建接口。因此只迁移算法和参数语义，不迁移旧版训练器、数据集、注册器或环境锁定。[官方环境说明](https://github.com/jbwang1997/OPUS)

## 3. 目标架构

新增 `OPUSSegmentor`，保持本仓库的 `img_backbone -> img_neck -> lifter -> encoder -> head` 调用协议：

```text
6 camera images
  -> ResNet-50 + FPN                    (复用)
  -> OPUSQueryLifter                    (Q 个可学习 query / 3D reference points)
  -> OPUSEncoder                        (多层 image cross-attention + query refinement)
  -> OPUSHead                           (每层坐标、语义、置信度)
  -> OPUSSetLoss                         (训练)
  -> OPUSRasterizer -> final_occ         (eval / visualization)
```

### 3.1 新增文件与职责

| 文件 | 职责 |
| --- | --- |
| `model/segmentor/opus_segmentor.py` | 新增注册的端到端分割器；复用当前图像特征提取逻辑，返回 OPUS 标准输出。 |
| `model/lifter/opus_query_lifter.py` | 创建 `Q` 个可学习 query embedding 和归一化 3D reference point；输出 `query_features`、`query_points`。 |
| `model/encoder/opus_encoder.py` | 组织多层 OPUS decoder，保存每一层预测，支持仅最后一层推理。 |
| `model/encoder/opus_decoder.py` | 单层 cross-attention、FFN、坐标残差与 coarse-to-fine query split。 |
| `model/encoder/opus_ops.py` | 相机投影、有效视锥 mask、特征采样；优先使用当前仓库可用的 PyTorch 算子/已有 deformable aggregation 接口，不能兼容时写清晰的纯 PyTorch 基线。 |
| `model/head/opus_head.py` | 坐标归一化/反归一化、语义 logits、训练输出、推理输出。 |
| `model/head/opus_rasterizer.py` | 稀疏点集到 `final_occ` 的 GPU 栅格化；空体素填 `empty_label=17`。 |
| `loss/opus_set_loss.py` | Chamfer、最近邻语义 focal loss、各 decoder stage 辅助损失和 adaptive re-weighting。 |
| `dataset/opus_target.py`（或 `transform_3d.py` 中独立 transform） | 从当前 `occ_xyz/occ_label/mask` 生成可复现、固定预算的非空 GT 点集。 |
| `config/opus/opusv1_t_r50_704x256_1f_occ3d.py` | 首个单帧 Occ3D 训练配置。 |
| `tests/test_opus_*.py` | 单元、契约、短程训练和评估回归测试。 |

同时更新对应的 `__init__.py` 注册导出。不得把 OPUS 专属逻辑塞入 `GaussianOccEncoder`、`GaussianHead` 或 `localagg*`。

### 3.2 Query 与 decoder

首版采用 OPUS-T 规模作为默认起点：`embed_dims=128`、6 个 decoder stage、600 个初始 query。每个 query 具有 feature、归一化位置 `(0,1)^3` 和可见性/有效性状态。每个 stage：

1. 将位置按 `pc_range` 转为 ego 坐标，投影到每一路相机。
2. 仅对有效深度、有效像素和未被图像 augmentation 裁掉的投影采样多尺度 FPN 特征。
3. 使用 cross-attention 聚合相机/尺度特征，再执行 self-attention 和 FFN。
4. 预测坐标残差及语义 logits；坐标更新后限制在 `pc_range` 内。
5. 按 `[1, 4, 16, 32, 64, 128]` 逐层分裂点，最终为约 `600 x 128 = 76,800` 个候选点。分裂 offset 使用固定父子索引，保证不同 stage 的点对应关系可追踪。

原始 OPUS 的核心是可学习 query 同时输出位置和类别，并以 Chamfer 距离处理大规模集合匹配；该实现应保持此语义，而不是退化成稠密 BEV/voxel head。[论文摘要](https://arxiv.org/abs/2409.09350)

### 3.3 特征采样实现策略

第一版必须先使用可调试的 PyTorch 实现：`project -> grid_sample -> attention reduction`，用 `torch.autocast` 和 chunk 控制显存。禁止在未验证投影坐标、相机增广矩阵、梯度和数值一致性之前移植上游 CUDA 扩展。

稳定后再评估两种优化路线：

- 适配当前 `model/encoder/gaussian_encoder/ops` 的 deformable aggregation 算子，前提是其 `points/features/projection` 语义完全一致并通过同输入数值测试。
- 从官方 OPUS 单独移植其最小 CUDA 算子，放在 `model/encoder/opus_ops/`，带 `setup.py`、CUDA/PyTorch 版本检查和纯 PyTorch fallback。

这能避免旧 ABI 或旧 MMCV 扩展污染当前环境。

## 4. 数据、坐标和掩码

### 4.1 单一坐标约定

首版统一用 **ego 坐标**，与现有 Occ3D 配置一致：

- GT 坐标唯一来源为 `LoadOccupancyOcc3D(..., model_coord='ego')` 输出的 `occ_xyz`。
- query 坐标、相机投影、rasterizer、评估传入的点全部使用相同 `pc_range` 和 voxel center 定义。
- 新 transform 必须在首 batch 断言：`occ_xyz.shape[:3] == grid_shape`、标签/mask shape 一致，且点落在范围内。
- `occ_cam_mask` / `occ_lidar_mask` 仅决定监督或评估范围；不得用于改变点坐标或 silently 过滤预测。

### 4.2 GT 点集构造

每个样本从 `occ_label != empty_label` 且满足 `occ_loss_mask` 的体素中心取得 GT 点。为避免不同场景点数差异和 Chamfer 内存爆炸：

- `max_gt_points` 默认 76,800，与最终预测预算对齐；多于预算时使用带固定 seed 的体素分层/均匀采样，少于预算时不重复填充。
- 同一个 sample 的所有 decoder stage 复用同一个 GT 点索引（consistent point sampling），seed 由 `global_seed + epoch + sample_token` 构成，DDP rank 不影响结果。
- 输出 `opus_gt_points [B, M, 3]`、`opus_gt_labels [B, M]`、`opus_gt_valid [B, M]`。空样本要有合法的零点集分支，不能产生 NaN。
- 语义类别使用现有 Occ3D 标签空间；点集分类不预测 empty 类，空体素只由 rasterizer 的默认值表示。

默认先做 Occ3D。SurroundOcc 版本应在 Occ3D 跑通后增加独立配置，并明确 lidar/ego 变换和类别映射，不与首版混测。

## 5. 损失设计

`OPUSSetLoss` 仅消费 OPUS 输出和新增 GT 字段，注册到 `OPENOCC_LOSS`，通过现有 `loss_input_convertion` 输入。每个被监督的 decoder stage 计算：

```text
L_stage = lambda_cd * (CD(pred_xyz, gt_xyz))
        + lambda_cls * focal(pred_logits, NN_label(pred_xyz, gt_xyz))
```

- `CD` 使用双向 Chamfer（pred-to-GT + GT-to-pred），距离在米制坐标下计算并以 voxel size 归一化。
- `NN_label` 取每个预测点最近 GT 点的语义；只对 `pred_to_gt` 最近邻结果有效的预测点计算分类。
- 采用 focal loss 与 OPUS 的自适应重加权：基于该 stage 的匹配几何误差调节分类权重，早期 stage 低权重，最后 stage 权重为 1。
- 辅助 stage 的权重由配置显式列出，例如 `[0.25, 0.35, 0.5, 0.7, 0.85, 1.0]`，不得隐式依赖 decoder 数量。
- 日志至少包含 `loss_cd`, `loss_cls`, `mean_nn_distance`, `valid_gt_points`, `valid_pred_points`，用于定位不收敛是投影、匹配还是类别问题。

严禁对 `76,800 x 76,800` 直接执行全量 `torch.cdist`。优先接入一个已验证、GPU 可用的 KNN 后端（PyTorch3D `knn_points`、FRNN 或从官方 OPUS 提取的最小 KNN 实现），以 chunked `cdist` 仅作为小样本 CI fallback。后端选择和版本须写入 `requirements-opus.txt`，启动时打印实际后端。

## 6. 推理、评估与可视化

### 6.1 Rasterizer 契约

`OPUSRasterizer` 接收最终 `pred_points [B,P,3]`、`pred_logits [B,P,C]` 和 `pc_range/grid_size`，输出：

- `final_occ [B, X, Y, Z]`：当前 `eval.py`、`visualize.py` 的兼容输出，默认 `17`（empty）。
- `opus_points`、`opus_labels`、`opus_scores`：供点云/Open3D 调试与定性可视化。
- `opus_voxel_confidence`：多个点落入同一 voxel 时，以最大类别置信度的点胜出；该规则必须单元测试且训练/验证一致。

越界、NaN/Inf、低于 `score_threshold` 的点应被记录并丢弃。rasterizer 不能使用 Python 三重循环；采用扁平 voxel index、`scatter_reduce_(amax)` 或等价 CUDA 向量化实现。

### 6.2 对现有脚本的最小修改

- `eval.py`：从 cfg/输出读取 `grid_shape`、`empty_label`、坐标系，不再假定 `200,200,16`；Occ3D 仍调用现有 `Metric_mIoU`。
- `visualize.py`：保留现有体素可视化路径；增加 `--vis-opus-points`，对原始 sparse point set 着色并可与 `final_occ` 对照。
- `train.py`：复用现有 loss 构建；在保存可视化样本时同时保存 point set，避免重新前向。
- `fifo_eval.py`：只要消费标准 `final_occ` 即无需 OPUS 分支；新增 smoke test 证明这一点。

首版指标以当前仓库的 Occ3D mIoU 为准。若项目已有可靠的 RayIoU 实现，必须使用同一栅格化结果补充 RayIoU；不要将不同 discretization 的数值与官方 OPUS 论文直接比较。

## 7. 配置与训练策略

新增配置目录：

```text
config/opus/
  opusv1_t_r50_704x256_1f_occ3d.py
  opusv1_t_r50_704x256_1f_occ3d_debug.py
  opusv1_t_r50_704x256_8f_occ3d.py       # 在单帧稳定后启用
```

首个正式配置继承当前 Occ3D 数据配置，不继承 Gaussian encoder/head 配置。关键参数全部可配置：`num_queries`、`split_schedule`、`num_decoder_layers`、`pc_range`、`grid_size`、`max_gt_points`、KNN backend、stage loss weights、score threshold。

训练分三步推进：

1. **Debug 闭环**：1 GPU、10 个样本、`Q=32`、最终不超过 512 点，关闭随机图像增广。验证投影、GT 坐标、loss 有限、反向梯度非零、`final_occ` shape 正确。
2. **单帧基线**：`Q=600`、6 stage、704x256。先冻结 ResNet 1--2 epoch 验证 query/loss，再全量微调；恢复官方使用的 ImageNet/nuImages 预训练权重时要做 key/shape 检查。
3. **8 帧时序**：仅在单帧稳定后迁移。扩展当前 dataset 以提供历史相机图像、时间差和 ego-to-ego 变换；特征必须被变换到当前 ego frame 后再与 query 聚合。不得把历史帧简单拼到 camera 维度。

官方示例使用 `opusv1-t_r50_704x256_8f_nusc-occ3d_100e.py` 和 100 epoch，因此 8 帧设置应作为性能对齐参考，而不是首个集成验收目标。[官方训练命令](https://github.com/jbwang1997/OPUS)

建议最初优化器沿用仓库的 AdamW/调度器模式，单独给 query、reference points 与 backbone 配置 paramwise learning rate；开启 AMP、gradient clipping，并在 config 中声明 `max_tokens_per_gpu` 以便按显存缩放 batch size。必须记录实际有效 batch size 和 `max_gt_points`，否则结果不可复现。

## 8. 实施顺序与验收

| 阶段 | 交付物 | 验收标准 |
| --- | --- | --- |
| P0：契约 | 数据字段、坐标断言、`OPUSRasterizer` | 单样本 GT 点反栅格化与原标签在 valid mask 内一致；空体素为 17。 |
| P1：最小网络 | Query lifter、1-layer decoder、head、纯 PyTorch projection | CPU/GPU 单 batch 前后向无 NaN；所有输出 shape 固定；投影与现有相机矩阵的人工点测试一致。 |
| P2：集合损失 | KNN backend、Chamfer、NN semantic/focal | 人造点集损失满足相同点接近 0、平移后增大、标签替换只改变分类项；DDP 两 rank 无死锁。 |
| P3：训练闭环 | debug config、`train.py` 接入 | 10 sample 过拟合，CD 和分类 loss 显著下降，checkpoint 可 resume。 |
| P4：评估可视化 | rasterizer、`eval.py` metadata 化、OPUS 点可视化 | `eval.py` 和 `visualize.py` 从同一 checkpoint 运行；无硬编码 grid shape；输出可在 Open3D 中检查。 |
| P5：性能 | 6-stage 单帧、算子优化、8-frame | 比较 pure PyTorch 与优化算子的 logits/point 差异、吞吐和显存；仅数值一致后替换默认实现。 |

每一阶段单独提交。不要在 P0--P4 中改变当前 Gaussian 配置或其 checkpoint 兼容性。

## 9. 测试清单

- `test_opus_projection.py`：六相机内外参、resize/crop/flip 后投影与有效 mask。
- `test_opus_target.py`：Occ3D mask、空场景、固定 seed、DDP seed 一致性。
- `test_opus_loss.py`：Chamfer/NN/focal 的数值、梯度、空 GT、chunk/KNN backend 一致性。
- `test_opus_rasterizer.py`：边界点、冲突点、低分点、坐标转换和 `final_occ` shape。
- `test_opus_config.py`：能通过当前 registry 构建；不触发 Gaussian 或 `localagg` 导入。
- `test_opus_smoke_train.py`：2 个 iteration AMP + backward + optimizer step + checkpoint reload。
- `test_opus_eval_vis.py`：mock checkpoint 在 Occ3D eval 和点/体素可视化入口均能运行。

## 10. 风险与决策

| 风险 | 处理决策 |
| --- | --- |
| 上游旧版 MMLab API 与当前仓库不兼容 | 仅端口算法模块，所有组件使用当前 `MODELS/SEGMENTORS/OPENOCC_LOSS` 注册方式重写。 |
| 全量 Chamfer O(PxM) 造成显存/时延不可接受 | P2 前明确 KNN 后端；生产训练禁止全量 `cdist`。 |
| 图像 augmentation 与 3D 投影不一致 | P1 使用固定增强和人工投影测试；后续每项 augmentation 都有回归测试。 |
| 稀疏点到体素的冲突导致指标漂移 | 统一 rasterizer，训练不依赖 rasterizer，评估和可视化共享同一实现并记录阈值。 |
| 单帧实现直接追求官方 8 帧数值 | 将单帧作为功能/回归基线；时序建模在 P5 后单独评估。 |
| CUDA 扩展编译失败或版本不匹配 | 保留纯 PyTorch fallback，并把优化算子设为显式 opt-in。 |

## 11. 完成定义

迁移完成必须同时满足以下条件：

1. `config/opus/opusv1_t_r50_704x256_1f_occ3d.py` 能在当前环境启动训练、保存和恢复 checkpoint。
2. 10 sample debug 过拟合通过，完整训练无 NaN/Inf、无无效投影比例异常，并有可追踪 loss 指标。
3. `eval.py` 可输出 Occ3D 指标，`visualize.py --vis-opus-points` 可显示原始点集且已有体素可视化不回归。
4. Gaussian / allocation 的既有配置可照常构建、训练和评估。
5. 所有新增单元、smoke 和契约测试通过，README/requirements 明确数据、预训练权重、KNN/CUDA 可选依赖和运行命令。
