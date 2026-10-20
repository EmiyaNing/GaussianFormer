# 完整 Occ3D：OPUS + 概率渲染，四个尺寸

这组配置将 mini 实验已验证的概率渲染推广到完整数据训练，沿用项目已有的四尺寸、8 帧训练设置。它是 OPUS 架构与 GaussianFormer-2 概率渲染的组合，不代表复现完整 GaussianFormer-2 模型。

## 选择配置

这里的“尺寸”指模型 query 和子 Gaussian 的配置，四组图像分辨率均为 **高 256 × 宽 704**。

| 尺寸 | 配置文件 | Query 数 | 每个 query 的采样点数 | 六层子 Gaussian 数 | 最终 Gaussian 数 |
| --- | --- | ---: | ---: | --- | ---: |
| Tiny / T | `opusv1_gaussian_probability_tiny_8f.py` | 600 | 4 | 1, 4, 16, 32, 64, 128 | 76,800 |
| Small / S | `opusv1_gaussian_probability_small_8f.py` | 1,200 | 2 | 1, 4, 8, 16, 32, 64 | 76,800 |
| Middle / M | `opusv1_gaussian_probability_middle_8f.py` | 2,400 | 2 | 1, 2, 4, 8, 16, 32 | 76,800 |
| Large / L | `opusv1_gaussian_probability_large_8f.py` | 4,800 | 2 | 1, 2, 4, 8, 16, 16 | 76,800 |

## 共同训练设置

- 完整 nuScenes Occ3D：训练 28,130 帧，验证 6,019 帧；每个 epoch 完整验证。
- R50/FPN，6 层 decoder，当前帧与 7 个历史帧，每帧 6 路相机。历史帧保留梯度。
- 30 epochs，seed 928，AdamW，基础学习率 `2e-4`，weight decay `0.01`。
- 默认每卡 batch 1；下方单卡入口累积 8 次，对应有效 batch 8。
- 图像 backbone 使用 BF16；FPN、decoder、概率渲染和损失使用 FP32。图像分块与 decoder 激活重计算沿用已有设置。
- 体素网格 `200 × 200 × 16`，间距 0.4 m；Gaussian scale 范围 0.08–0.40 m。
- Occupancy 的训练和评估都使用 camera mask。OPUS Set loss 仍使用全部非空 GT，并保留原有的 camera 距离权重。

## 概率基线究竟改了什么

使用官方 GF2 概率聚合核，分别得到占有概率和“已占有条件下”的前景类别概率。最终 17 个前景类别概率由两者相乘，empty 概率为一减占有概率，最终预测取 18 类概率最大值。

渲染读取原始语义 logits，经 softmax 归一化。Encoder 继承的 softplus 语义字段仍会生成，但不参与概率渲染。模型不再包含全局 empty Gaussian 或 learnable empty bias；也没有额外的逐类别校准。

保留六层原始 `OPUSSetLoss`。最终层的 occupancy 损失使用概率 NLL（权重 10）和直接作用于概率的 Lovasz（权重 1），不再对输出概率做一次 softmax。类别权重和其他损失设置保持不变。

正式训练直接使用 `MultiLoss`，不使用 mini 对照实验的 `GF2ProbabilityExperimentLoss`，因此不会生成逐批数据指纹或写入旧实验目录。

## 启动单卡训练

从项目根目录运行，例如 Tiny：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python3.10 -u tools/train_opus_gaussian_8f.py \
  --py-config config/full_occ3d_probability/opusv1_gaussian_probability_tiny_8f.py \
  --work-dir work_dirs/full_occ3d_probability/tiny \
  --gradient-accumulation 8 --seed 928
```

另外三个尺寸分别将命令中的 `tiny` 替换成 `small`、`middle` 或 `large`，配置和输出目录都要一起替换。入口会自动恢复输出目录已有的 `latest.pth`；新实验使用独立空目录。

预训练权重沿用已有 8 帧配置：`ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`。数据与权重路径均相对项目根目录。

该入口当前明确限制单卡。四张卡可分别设置 `CUDA_VISIBLE_DEVICES=0/1/2/3` 各自运行一个尺寸；这与四卡联合训练一个模型是两种运行方式。若改用其他 DDP 入口，需要另外核对有效 batch、梯度累积和学习率调度。

## 概率 CUDA 核的依赖

概率算子源码位于 `model/head/localagg_prob/`，与项目现有 `model/head/localagg/`、`localagg_react/` 保持相同目录结构。`model/head/gf2_probability_backend.py` 从该目录加载并编译；运行模型不再依赖 `work_dirs` 中的算子源码或旧实验初始化步骤。运行环境同时需要项目原有的 `local_aggregate` 扩展。

首次调用会将编译产物写入 PyTorch 默认扩展缓存（通常在 `~/.cache/torch_extensions/`）。可通过 `TORCH_EXTENSIONS_DIR` 指定缓存位置。迁移机器时复制模型源码，并在目标机器重新编译即可，无需携带旧的构建产物。

默认由 PyTorch 按可见 GPU 选择编译架构，也可以通过 `TORCH_CUDA_ARCH_LIST` 显式指定，例如 RTX 4090 使用 `8.9`，A100 使用 `8.0`。后端默认最多使用两个编译任务，可以通过 `MAX_JOBS` 调整。

## 已完成的检查

四个尺寸均已在 RTX 4090 上通过配置解析、完整数据划分核对、真实 8 帧样本前向与反向、一次 AdamW 更新，以及一个验证样本的检查。输出为归一化的 18 类概率，损失和梯度均为有限值；8 帧均有特征梯度，模型中不存在 empty bias 或 empty scalar 参数。

同时逐项核对了与旧四尺寸配置的 encoder、query、Set loss、预训练权重和学习率调度设置，仅按上述说明改变渲染与 occupancy 损失输入方式。

本次没有启动 30 epoch 正式训练。这些检查不能替代完整梯度累积周期、整轮验证、长时间显存稳定性或 DDP 检查。完整记录见同目录的 `validation.json`；原始检查日志位于 `work_dirs/full_occ3d_probability/preflight/`。

算子迁入模型目录后，另用全新扩展缓存完成了独立编译，前向值和四组输入梯度均通过可微参考实现比对；Tiny 的真实 8 帧训练步与验证样本也重新通过。迁移记录位于 `work_dirs/full_occ3d_probability/preflight/backend_migration_source.json`，检查日志为同目录的 `backend_migration.log` 和 `backend_migration_tiny.log`。原有 `validation.json` 保留为迁移前四尺寸检查的历史记录。
