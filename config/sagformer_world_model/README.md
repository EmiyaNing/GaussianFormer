# SAGFormer World Model

本目录集中保存当前 Gaussian 世界模型的训练与评测配置。

| 配置 | 用途 |
|---|---|
| [img_voxel_react_occ3d_world_v1.py](img_voxel_react_occ3d_world_v1.py) | 冻结九帧历史感知器的基础世界模型配置 |
| [img_voxel_react_occ3d_world_v1_lidar10f.py](img_voxel_react_occ3d_world_v1_lidar10f.py) | 指定云端感知检查点与本地训练设置 |
| [img_voxel_react_occ3d_world_v1_smoke.py](img_voxel_react_occ3d_world_v1_smoke.py) | 两个训练样本、一个验证样本的链路检查 |
| [react_occ3d_world_perception_finetune.py](react_occ3d_world_perception_finetune.py) | 五轮联合训练，感知器学习率为预测器的 0.1 倍 |

感知器配置位于 [../img_point_lite_react/](../img_point_lite_react/)。世界模型基础配置
通过相对路径继承九帧历史感知器；两个目录分别管理感知计算与世界模型训练设置。
迁移仅调整文件位置、继承路径和调用入口，配置展开后的训练参数保持一致。

当前修复缩放后的后台实验使用独立配置快照，整理目录不会影响运行：

- 工作目录：`out/react_occ3d_world_perception_finetune_704x256_resize044_20261101/training`
- 日志：`out/react_occ3d_world_perception_finetune_704x256_resize044_20261101/console.log`
- 初始化：原始云端感知权重，预测器重新初始化；训练五轮。

以下为实现说明和历史实验记录。

以 `img_voxel_react_occ3d_fast_v5_mem_9history.py` 的当前感知结果为起点，同时预测未来占据和自车规划。
本版保留感知计算，不增加历史 GS、历史图像或未来 densify。

云端预训练权重使用 **704×256（宽×高）** 图像。九帧历史感知预设与 world 配置均
已对齐为 `input_shape=(704,256)`、训练/验证 `final_dim=(256,704)`。
按原验证规则，1600×900 原图先缩放为 704×396，再裁剪至 704×256，投影矩阵同步
更新。原始、不带 `_9history` 的 v5 配置保留其历史尺寸，避免改变其他实验。

2026-11-01 训练回归检查进一步发现：最初只修改 `final_dim`，训练端仍继承
`resize_lim=(1,1)`，导致训练直接从原图裁取窄视野，而验证先按 .44 倍缩放。
九帧历史预设现已同时设置 `resize_lim=(.44,.44)`，保证训练和验证的基础缩放与
视野一致；训练保留原有光度增强与随机翻转。仅检查最终张量尺寸不能发现这个问题。

训练与验证均使用 `num_lidar_history=9`：当前点云与同场景之前的 9 帧点云对齐融合后，
作为一个场景输入感知器（历史充足时共 10 帧）。场景开头不足 9 帧时，沿用原数据集
策略，仅使用已有历史，不跨场景取帧。该设置与 `history_frames=0` 不冲突，后者表示
不加载历史图像。世界模型正式配置与 smoke 配置都继承同一九帧历史感知配置。

smoke 的训练索引为 `[136,264]`，验证索引为 `[136]`，用于覆盖历史充足的样本。
审计工具分别记录配置历史帧数与样本实际历史帧数，避免混淆场景起点的回退情况。
验证样本 136 已在真实数据上核对：当前点云 34752 个点，与 9 帧历史融合后为
347328 个点，等于当前及九帧历史点数之和。

## 实现边界

- 感知器的 backbone、neck、lifter、encoder、原 densify、head 配置保持一致。
- encoder 可选导出最终 Gaussian 对应的 128 维特征。子 Gaussian 特征直接读取原
  densify 已计算的 `feature_proj` 输出，父子排列与 Gaussian 一致；默认调用接口不变。
- 感知器参数和 BatchNorm 统计量冻结，新增参数仅位于 `world_decoder.*`。
- 当前感知器仍执行完整原路径并输出当前占据。未来状态只更新中心与语义，Gaussian
  数量、尺度、旋转和 opacity 保持不变。

## 一个预测步

1. ego token 从 Gaussian 特征读取场景信息，再由 GRU 更新。
2. `traj_head` 预测二维位移增量，累积成当前 ego 坐标系下的绝对路点。
3. Gaussian 特征结合位置、时间、ego token、计划路点做残差更新。
4. `reg_branch` 输出受限三维修正；`vel_branch` 输出动态类别的二维位移；
   `cls_branch` 输出语义隐变量残差。
5. 更新后的 Gaussian 用原 React 聚合算子渲染，进入下一个预测步。

六步共享参数，间隔 0.5 秒，覆盖未来 3 秒。v1 新增 436,760 个参数。

`vel` 单位为 **米/步**，不是米/秒；默认每轴上限为 10 米/步。
`reg` 每轴上限为 0.2 米/步，另有小幅回归正则，避免两个位置分支完全互相替代。
动态类别沿用 SparseWorld 的类别 2–10 硬门控，门控本身不求导。

`cls` 不直接替换可靠感知的语义：先对原正值语义做逆 softplus，再递推语义残差，
最后用 softplus 恢复给 renderer。三个残差头初始输出为零，初始场景保持静止。

## 文件职责

| 文件 | 职责 |
|---|---|
| `model/encoder/react_world_decoder.py` | ego/场景交互、reg/cls/vel、轨迹、自回归 |
| `model/segmentor/react_gaussian_world.py` | 保留感知路径、严格加载与冻结、组织输出 |
| `model/head/react_world_render.py` | 复用 React 算子，转换输出坐标 |
| `dataset/react_world.py` | 当前传感器输入，加未来监督标签 |
| `loss/react_world_loss.py` | 六时刻占据、轨迹、reg 正则 |
| `misc/react_world_metric.py` | 当前/未来占据与规划 L2 指标 |
| `tools/audit_react_world.py` | 检查感知参数不变、比较原路径与扩展路径 |

## 坐标与监督

内部 Gaussian 和预测路点全部使用**当前 ego 坐标系**，单位为米。
规划标签由 `future_ego_to_current` 的平移生成，表示未来自车原点在当前 ego 系的位置。
这避免了直接混用已有轨迹文件的 LiDAR 坐标约定。

训练/评测时，未来 Occ3D 标签属于各自未来 ego 坐标系。渲染端使用真实相对位姿把
预测中心、逆协方差和支持域转换过去，仅用于坐标对齐。真实未来位姿、轨迹和占据标签
**不进入 world decoder**，不会改变预测 GS 或 plan。这个位姿已知的读出协议不能解释为
模型自己预测了未来自车位姿。

没有 `future_ego_to_current` 时，未来占据直接渲染在当前 ego 网格，
`future_occ_frame='current_ego'`；不得直接与原生未来 ego 标签比较。
提供对齐位姿时输出为 `'future_ego'`，损失与评测会检查此标记。

当前感知输入保持九帧历史点云版本的原流水线；新数据集仅追加未来标签读取。ego 输入沿用 AD-MLP
当前状态向量的 21 维约定，显式排除其中的 `gt` 字段。不加载历史 GS 或历史相机帧。

未来占据损失沿用底座 CE/Lovasz 和类别权重，六步取加权平均。规划损失为所有
样本/时刻的二维平方距离均值；空标签 255 不参与占据监督。

## 训练与评测

2026-10-31 已接入云端训练的九帧历史感知权重：
`ckpts/img_voxel_react_occ3d_fastv5_lidar10f.pth`。检查点记录为 epoch 13、91416 次迭代，
637 个感知状态张量与本地模型严格匹配。2026-11-01 用户修正了云端评估口径：
无 camera mask 的基线约为 **mIoU 33、IoU 51**，先前的 46.8/70.6 不再作为
无 mask 基线。这里记录的是用户报告的近似值。

此前的独立感知器评估使用了错误的 **1600×864** 输入尺寸：严格加载
同一检查点、当前帧加九帧历史点云、无 camera mask，6019 个验证
样本的 **mIoU 为 26.22**。该结果不能作为尺寸对齐后的感知精度。旧 world model
因要求完整未来监督，验证集为 4219 个样本，连续五轮的当前帧指标稳定在
**mIoU 26.47–26.48、IoU 48.03–48.04**。两者样本集合不同，不能直接要求
指标逐位相等。

已经核对 world model 第一个完整 epoch 后的全部 637 个感知状态张量，与原始
感知检查点逐元素相等。同样本的 64 例对照中，原感知路径与冻结并导出特征的
路径在无 mask 下分别为 25.75/45.40 和 25.83/45.40；完整原路径自身存在
稀疏计算非确定性。这些旧实验使用了不匹配的图像尺寸，不能将其精度差距归因于
world decoder 或冻结后权重退化。

正确尺寸的独立感知器复测使用 `tools/evaluate_react_perception.py`，基于同一份
预测同时统计 camera mask 与无 camera mask 的 mIoU、二分类占据 IoU；输出原始
混淆矩阵、实际网络输入形状、权重 SHA-256 与完整预处理配置。结果写入
`out/react_perception_704x256_20261101/results.json`，日志为同目录 `eval.log`。

2026-11-01 已完成全部 6019 个验证样本，严格加载原云端感知检查点的 637 个
状态张量，实际网络输入为 `[1,6,3,256,704]`，当前点云加九帧历史。最终结果：

| 评估区域 | mIoU | 二分类占据 IoU |
|---|---:|---:|
| 无 camera mask | 33.31 | 50.79 |
| camera mask | 46.82 | 70.46 |

两种统计来自同一份预测，权重没有经过本地训练。结果与云端报告基本一致；
相比此前错误尺寸的 26.22，无 mask mIoU 提高约 7.09 个点。此前的主要差距来自
输入分辨率不匹配。

复现两种 mask 的完整评估：

```bash
python3 tools/evaluate_react_perception.py \
  --config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_9history.py \
  --checkpoint ckpts/img_voxel_react_occ3d_fastv5_lidar10f.pth \
  --output out/react_perception_704x256/results.json
```

完整评估日志、同样本对照结果与 CUDA 数值检查分别保存在本次实验的
`diagnostics/full_perception9_eval.log`、`diagnostics/perception_comparison.json`、
`diagnostics/v5_cuda_checks.log`。18 项 v5 CUDA 数值检查均通过。
启动时的 `resolved_config.py` 和 `code_at_launch/` 保留原始记录，可能仍包含修正前
的云端基线；该字段只用于记录，不参与训练或指标计算。

对应专用配置为 `img_voxel_react_occ3d_world_v1_lidar10f.py`，已显式指定该权重、
九帧历史、冻结感知、无 camera mask、24 轮训练。启动命令为：

```bash
python3 train.py \
  --py-config config/sagformer_world_model/img_voxel_react_occ3d_world_v1_lidar10f.py \
  --work-dir out/react_world_v1_lidar10f
```

本机已启动的独立实验位于 `out/react_world_v1_lidar10f_20261031/`：
`run.json` 记录权重哈希、配置、进程和完成/失败状态；`console.log` 为训练输出，
`training/` 保存每轮检查点和指标；`code_at_launch/` 保留启动时代码。
其 `launch_training.py` 独立于交互终端运行，并使用文件锁防止重复启动同一实验。

如需重新训练感知器，可使用：

```bash
python3 train.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_9history.py \
  --work-dir out/react_perception_9history
```

### 感知器与预测器联合微调

`react_occ3d_world_perception_finetune.py` 是独立的五轮联合训练实验。感知器从
`ckpts/img_voxel_react_occ3d_fastv5_lidar10f.pth` 严格加载，预测器重新初始化，
不沿用旧冻结实验的预测器或优化器状态。仍使用当前点云与九帧历史点云、六个未来
时刻和无 camera mask 的验证协议。

感知器解除整体冻结，ResNet 的 `frozen_stages=-1` 也允许 stem 和第一阶段卷积
参与训练；v5 原有的图像固定 BatchNorm 保持其固定统计量与仿射参数。LiDAR
backbone、neck、Gaussian encoder、densify 和 head 的可学习参数参与反向传播。

预测器基础学习率为 `2e-4`，感知器为 `2e-5`。前 500 次迭代预热，之后按五轮总
迭代数做余弦衰减；预热起点、整个衰减过程与学习率下限都保持 10:1 比例。
`misc/lr_groups.py` 将 MMEngine 的初始分组学习率转换成 timm 的 `lr_scale`，
避免 timm 默认共用预热起点和绝对下限而改变分组比例。训练日志的
`[TRAIN][LearningRates]` 同时列出两组实际学习率。

联合损失为当前占据损失 + 六个未来占据损失的均值 + 规划平方距离 +
`0.01 * reg` 正则；当前与未来占据均沿用感知器的类别权重、CE 和 Lovasz。
当前项通过 `current_occupancy_weight=1` 开启，原冻结配置的默认值仍为零。
为控制全感知反向传播时的显存，关闭 cuDNN benchmark，保留原有激活重计算。

```bash
python3 train.py \
  --py-config config/sagformer_world_model/react_occ3d_world_perception_finetune.py \
  --work-dir out/react_occ3d_world_perception_finetune
```

尺寸对齐后的独立后台实验保存在
`out/react_occ3d_world_perception_finetune_704x256_20261101/`。
每轮保存检查点并评估当前帧、1/2/3 秒未来占据和规划误差，第五轮验证结束后退出。
`console.log` 查看进度，`run.json` 记录运行状态，`training/epoch_5.pth` 为完成五轮
训练后保存的权重。启动前的两步真实训练和单样本验证单独保存在 `preflight/`。
此前使用 1600×864 的联合训练
`out/react_occ3d_world_perception_finetune_20261101/` 已停止，并保留其
`training/epoch_1.pth`。新实验重新加载原始云端感知权重，不接续尺寸不匹配的
联合训练状态。旧冻结实验的五轮检查点也继续保留。

尺寸、投影矩阵、联合监督与学习率相关的 22 项测试通过。旧 1600×864 实验的
两步预检峰值 allocated 为 13765 MiB，该数字不代表正确尺寸下的显存开销。
704×256 的真实训练预检单独记录在新实验目录的 `preflight_result.json`。
两步预检确认感知器五个模块和预测器四个输出分支均有更新，峰值 allocated 为
11023.7 MiB；优化器保存的分组学习率比例为 0.1。新实验的日志在每轮首个训练和
验证 batch 输出实际图像形状，便于核对尺寸。
预检结果只用于确认训练链路，不能代替完整验证集精度。

**该 704×256 联合训练已因上述训练/验证裁剪偏差停止，未完成五轮。** 第一轮的
当前帧 mIoU/IoU 降至 24.53/47.76，1/2/3 秒 mIoU 为 20.58/18.44/16.65。
`training/epoch_1.pth` 已保留，源配置已修正，但不能继续把旧检查点当作正确
预处理下的联合训练结果；重新训练应从原始云端感知权重开始。

现有检查点的 96 个同样本对照中，原始感知器正常验证 mIoU 为 33.42，微调后为
23.82；仅将图像 backbone 换回原始权重可恢复到 30.36，仅恢复 BN 统计量则为
23.25。原始权重使用错误训练裁剪时为 20.42。结果支持图像分支在训练中适应错误
缩放/视野是主要问题，不能将此次下降直接归因为未来预测任务的负迁移。

诊断过程没有创建优化器或修改检查点。完整数据和结论分别为
`out/react_finetune_drop_20261101/paired_diagnosis.json` 与同目录 `diagnosis.md`。
修复后的 23 项回归测试通过，新增测试同时覆盖训练和验证的缩放、裁剪及投影。

### 冻结版本的其他启动与评测方式

训练世界模型时，显式提供九帧历史配置训练的检查点。历史帧数不改变权重形状，无法仅靠
张量形状判定一个检查点的训练输入；应核对它对应的训练配置。

```bash
REACT_PERCEPTION_CHECKPOINT=ckpts/img_voxel_react_occ3d_fastv5_lidar10f.pth \
python3 train.py \
  --py-config config/sagformer_world_model/img_voxel_react_occ3d_world_v1.py \
  --work-dir out/react_world_v1
```

完整 world 检查点已包含感知权重，恢复/评测不要求原感知检查点文件仍然存在：

```bash
python3 eval.py \
  --py-config config/sagformer_world_model/img_voxel_react_occ3d_world_v1.py \
  --work-dir out/react_world_v1 \
  --resume-from out/react_world_v1/epoch_24.pth
```

训练配置假定已有的 `admlp/stp3_val/data_nuscene.pkl`、`occworld` 时序文件、
nuScenes 与 Occ3D 标签可读。只支持 batch size 1，与当前 React 渲染器的约束一致。
没有重新编译 CUDA 扩展的需求。

已有的 `out/img_voxel_react_occ3d_fastv5/iter.pth` 只有 50 次迭代，不能当作成熟九帧历史底座。
可以显式使用它检查程序链路；小样本配置只运行两步训练、一个验证样本：

```bash
REACT_PERCEPTION_CHECKPOINT=ckpts/img_voxel_react_occ3d_fastv5_lidar10f.pth \
python3 train.py \
  --py-config config/sagformer_world_model/img_voxel_react_occ3d_world_v1_smoke.py \
  --work-dir out/react_world_v1_smoke
```

输出包括原 `final_occ`、`final_occ_grid`、六个 `future_occ_grids`、
`future_occ_logits`、`future_gaussians`、`pred_traj [B,6,2]` 和实际秒数列表。
训练验证与独立评测报告 1/2/3 秒占据及轨迹指标；轨迹分别报告该时刻端点 L2 和
截至该时刻平均 L2，单位为米。当前没有实现碰撞损失/碰撞率，也未验证规划闭环性能。

## 验证边界

单元测试覆盖特征导出的几何/梯度一致性、固定数量递推、动态门控、分支梯度、
冻结状态、无未来标签推理、坐标旋转、标签隔离以及配置与指标协议。
GPU 小样本训练只验证程序、梯度、验证与保存链路，不代表预测或规划精度。
没有未来增密/新生 Gaussian 分支，当前未覆盖区域的补全能力尚未验证。

原感知路径独立重复前向本身存在稀疏/离散计算的非确定性，因此审计工具在同一份真实
感知中间状态上比较默认 densify 与特征导出接口，严格检查 GS 属性逐元素不变，并检查
当前渲染结果；不据此宣称独立运行整个感知器必然逐元素相等。

以下 GPU 记录来自切换九帧历史配置之前（`num_lidar_history=0`），不能作为九帧历史版本
的性能记录。2026-10-30 本机验证：44 项相关 CPU 回归通过，20 项需要 CUDA 的既有用例在沙箱中
跳过；另外在 GPU 完成两步真实训练、一次验证、保存及独立评测。检查点里的 637 个
感知状态张量均未变化。同一真实中间状态下 GS 属性差异为零，当前占据类别差异为
0/640000；重复渲染的 logits 最大绝对差为 2.67e-5。

该样本为 23704 个 Gaussian，特征为 `[1,23704,128]`，未来占据共六帧，规划为
`[1,6,2]`。小样本训练峰值 allocated 为 3148.3 MiB；这个数字不代表正式训练吞吐
或其他输入规模。验证日志和审计结果位于 `/tmp/react_world_v1_smoke.console.log`、
`/tmp/react_world_v1_eval.console.log`、`/tmp/react_world_v1_audit.json`。
