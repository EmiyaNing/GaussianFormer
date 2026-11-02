# AdaptiveAllocationV8：先保留 baseline 的生成能力，再学习选择

V7 的四卡端到端训练下降不能用早先冻结主干的 pilot 解释。这里保留 V7 原代码和配置，新增 V8，并用端到端对照分别验证预算、子分支初始化、父残差和选择策略。

## 模型变化

1. **固定预算**：每帧增加 `min(2560, N)` 个子 GS，与原 DensifyOnly 对齐，避免把 `0.15N` 的可变预算混入模块比较。
2. **原生成器与初始化**：V8 继承 DensifyOnly，保留 `feature_proj` 和五个 `*_shift` 的参数名、公式及初始化顺序。子 opacity 仍为 `(parent_opacity + sigmoid(prediction))/2`，没有 V7 的初始 0.02 倍衰减；原父 GS 始终保留。
3. **评分梯度隔离**：独立小 MLP 读取 `features.detach()`。主 CE/Lovasz 更新原生成器及主干；辅助损失只更新评分器。新配置开启 `separate_allocation_grad_clip=True`，分别裁剪任务与评分梯度，避免评分梯度经全局裁剪范数影响主干更新。原配置默认保持整体裁剪。硬 top-k 本身不向评分器传播主任务梯度。
4. **分阶段选择**：先用原 opacity top-k，同时训练评分器；随后逐步增加 learned top-k 名额，并用剩余的 opacity top-k 填满固定预算。索引不重复，训练与推理使用同一比例。checkpoint 保存 `allocation_step`，恢复后延续进度。
5. **效率与可复现性**：只给 128 个随机候选计算评分器的训练图；全量排序分数在 `no_grad` 下计算。候选使用独立 RNG，不改变主干 dropout 的随机流。推理无 probes、标签或辅助损失。

教师沿用 V7 的局部反事实 CE 增益：在相同的最后 refine logits 上，计算增加一个候选子 GS 后局部 CE 的变化。它是选择策略的训练代理，不是全局 mIoU 真值，也没有建模多个子 GS 同时加入时的相互作用。

## 配置

- `adaptive_allocationv8_occ3d.py`：完整 v5_mem 路径，9 帧历史点云，主干正常训练；继承 baseline 的图像预训练，不使用旧 allocator warmstart。
- `adaptive_allocationv8_occ3d_opacity.py`：始终使用原 opacity 选择，但仍训练独立评分器，作为生成器和辅助训练的控制组。

默认 `selection_warmup_steps=1000`、`selection_transition_steps=1000`，单位为**每张卡启用梯度的训练前向次数**，包含梯度累积的 microbatch，不是 epoch 或全局样本数。该默认时长尚未经过完整四卡训练验证；可按实际每卡 loader 长度安排。纯 opacity 控制可先用于复现 baseline 的早期收敛。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.10 train.py \
  --py-config config/adaptive_allocation/adaptive_allocationv8_occ3d.py \
  --work-dir work_dirs/v8_occ3d_joint --seed 928
```

原 baseline checkpoint 的生成器参数名称兼容；新增评分器和 step buffer 应允许缺失。不要把训练 V7 的生成器权重当作 V8 的生成器。相对现有 v5 不需要重新编译 CUDA 算子。

## 逐项实验

`tools/experiment_adaptive_allocation_v8.py` 从完全相同的 baseline 初始化和图像预训练开始，所有组均使用 v5_mem、相同输入顺序、相同 dropout 种子、学习率与损失。它只缓存增强后的输入，每次更新重新计算图像、点云和 decoder 特征；不冻结主干。

| 组 | 相对上一个 V7 组的变化 |
| --- | --- |
| baseline | 原 DensifyOnly，+2560 |
| v7_original | 原 V7，+0.15N，无额外冻结 |
| v7_fixed | 仅固定为 +2560 |
| v7_stronger_child | 仅把子 opacity 初值从父 GS 的 0.02 倍提高到 0.3 倍 |
| v7_append | 再关闭父残差，仅添加子 GS |
| v8_opacity | 使用原生成器和原选择；评分器独立训练 |
| v8_learned | 在 V8 控制组上启用预热及逐渐接管选择 |

```bash
python3.10 tools/experiment_adaptive_allocation_v8.py \
  --out work_dirs/adaptive_allocation_v8_joint \
  --train-frames 32 --val-frames 16 --steps 128 --seeds 928,929
python3.10 tools/summarize_adaptive_allocation_v8.py \
  --out work_dirs/adaptive_allocation_v8_joint
```

恢复未完成的实验使用相同参数加 `--phase train`；已完成的整组会跳过，组内中断会从该组初始状态重跑。`protocol.json` 记录场景、输入来源和预算，`results.json` 保存每步损失及同一模型 allocation 前后的验证混淆矩阵，`summary.json` 与 `RESULTS.md` 汇总结果。每组还保存完整配置和最终 checkpoint。

128 步只用于拆解因素；V7 的 200 步稳定损失尚未结束，不能据此断言长期优劣。正式配置使用 1000/1000 步选择计划，而该短期筛选使用 32/32 步，以实际覆盖 learned selection 阶段。短训数值不能与四卡完整 epoch 的 37/41 mIoU 直接比较。

第二轮使用相同输入，关键四组延长到 256 步，并验证独立裁剪后的 V8：

```bash
python3.10 tools/experiment_adaptive_allocation_v8.py \
  --out work_dirs/adaptive_allocation_v8_joint_256 \
  --input-cache work_dirs/adaptive_allocation_v8_joint \
  --train-frames 32 --val-frames 16 --steps 256 --seeds 928,929 \
  --arms baseline,v7_original,v8_opacity,v8_learned --separate-score-clip
python3.10 tools/summarize_adaptive_allocation_v8.py \
  --out work_dirs/adaptive_allocation_v8_joint_256
```

第一轮 V8 使用整体裁剪，第二轮和发布配置使用独立裁剪，因此不能将两轮差值单独当成训练时长的效果。256 步覆盖 V7 稳定项结束，但仍然是重复小子集的短训。

## 验证范围

测试覆盖 CPU/CUDA 的原生成器初始化、随机流、输出、所有父 GS 字段与模型参数的任务梯度一致性；辅助损失的梯度隔离；固定预算、空输入、小场景、索引去重、恢复进度及推理确定性。模块级相等不代表整个 CUDA 训练轨迹逐 bit 相等。
