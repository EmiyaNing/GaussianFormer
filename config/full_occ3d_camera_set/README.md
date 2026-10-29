# small 微调：Set 和 Occupancy 使用同一 camera mask

本配置从 `ckpts/opus_gaussian_8f_small.pth` 加载模型参数，在完整 Occ3D 训练集上微调 1 个 epoch，随后使用完整验证集进行 camera-mask 评价。

## 修改的目标

六个阶段的 Set loss 都先筛选 GT，再执行最近邻匹配：

```python
target_valid = (gt_label >= 0) & (gt_label < 17) & occ_cam_mask
```

GS → GT、GT → GS 两个几何方向，以及最近 GT 的分类标签分配，全部使用这个筛选后的前景集合。mask 外 GT 不再参与这些计算。预测 GS 仍全部参与 Set loss；没有按 GS 中心位置删除预测点。

实现复用已有的 `CenterSupportDiagnosticSetLoss`，同时设置 `gt_domain='camera'` 和 `final_gt_domain='camera'`，覆盖全部六个阶段。原有分类、几何项的权重及 camera 内缺覆盖 GT 的额外加权保持原设置。Occupancy CE/Lovasz 继续使用 `occ_cam_mask`；它的 GT 包括 mask 内前景和 empty，而 Set GT 只包括其中的前景。

## 微调设置

| 项目 | 设置 |
|---|---|
| 模型 | small，1,200 query，每 query 最终输出 64 个 GS |
| 训练数据 | 完整 700 场景、28,130 帧 |
| 验证数据 | 完整 150 场景、6,019 帧 |
| 轮数 | 1 epoch，结束后验证 |
| 基础学习率 | 2e-5，保留原参数组倍率 |
| 学习率日程 | 前 400 个训练迭代预热，余弦下降，最低比例 0.1 |
| 初始化 | 严格加载模型参数；重新建立优化器和学习率日程 |
| 推荐单卡设置 | batch 1，梯度累积 2，共 14,065 次更新 |

这里的 `gradient_accumulation` 配置字段用于记录推荐设置；通用 `train.py` 实际读取命令行的 `--gradient-accumulation`，因此启动命令需保留该参数。训练帧数可被 2 整除，单卡执行时没有不足一个累积组的尾部。

## 启动命令

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 -u train.py \
  --py-config config/full_occ3d_camera_set/opusv1_gaussian_small_8f_camera_set_1ep.py \
  --work-dir work_dirs/small_camera_set_full_1ep \
  --gradient-accumulation 2
```

使用新的工作目录启动微调。`train.py` 检测到同一工作目录已有 `latest.pth` 时，会恢复该次训练的优化器和进度。

完成后权重保存在 `work_dirs/small_camera_set_full_1ep/epoch_1.pth`，完整验证指标写入该目录的训练日志。需要单独重新验证时执行：

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 -u eval.py \
  --py-config config/full_occ3d_camera_set/opusv1_gaussian_small_8f_camera_set_1ep.py \
  --work-dir work_dirs/small_camera_set_full_1ep_eval \
  --resume-from work_dirs/small_camera_set_full_1ep/epoch_1.pth
```

本次准备配置与启动命令，没有启动训练。

配置检查和 8 项已有 CPU 回归测试已通过。额外数值检查确认：六个阶段都在匹配前过滤 GT；改变 mask 外 GT 的位置和类别不会改变损失或任一输出梯度；结果与原 Set loss 手动传入筛选后 GT 一致。检查记录见 `work_dirs/small_camera_set_config_check/audit.json`。
