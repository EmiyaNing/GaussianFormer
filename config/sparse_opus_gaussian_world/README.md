# nuScenes Occ3D，15 epoch

独立配置：`sparse_opus_gaussian_world_nuscenes_occ3d_15e.py`。无配置继承，也不读取 4096/1024 子集 manifest。目录按本次要求使用 `gaussian` 拼写，模型注册名保留为 `sparse_opus_gausisan_world`。

当前本地元数据经原 SparseWorld 时序资格筛选后，使用 **19730 个训练样本、4219 个验证样本**。筛选要求包括 4 帧历史、6 帧未来，以及对应相机、占据和轨迹标注。

在仓库根目录启动：

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 train.py \
  --py-config config/sparse_opus_gaussian_world/sparse_opus_gaussian_world_nuscenes_occ3d_15e.py \
  --work-dir out/sparse_opus_gaussian_world_nuscenes_occ3d_15e \
  --gradient-accumulation 2
```

主要设置：15 epoch，batch size=2，累计梯度=2，AdamW 基础学习率 2e-4，前 5 轮时间预训练，保留 D 的模型、全空间损失及 0/1/2/3 秒 Gaussian 直接监督设置。条件语义 CE 的 17 类乘数沿用已验证的 D 固定值，没有重新估计全量训练集频率。

从图像 backbone/FPN 预训练初始化，世界模型与 Gaussian 部分从头训练。图像预训练权重和数据路径均完整列在配置中，换机器时按实际存储位置修改。

此配置使用仓库通用 `train.py`。该入口按 microbatch 推进学习率，因此配置使用 1000 个 microbatch 的 warmup，对应累计梯度为 2 时的 500 次更新；总 cosine 时长根据完整训练 loader 与 15 epoch 自动计算。通用入口的学习率参数组插值和随机状态管理采用其现有实现，不作为原子集实验逐步复现的入口。

**累计梯度必须通过命令行传入**，通用入口不会读取配置中的同名记录字段。每轮执行通用入口的当前帧验证；`eval.py` 的 SparseWorld 指标入口可对保存权重评估当前、1、2、3 秒。原 `tools.sparse_opus_gausisan_world.train` 以及其完整逐轮诊断仍针对 4096/1024 子集，这份全量配置使用上述通用入口。

配置加载、全量数据集构建、训练/验证首样本读取、模型/损失/优化器构建检查已通过，记录见 `analysis/sparse_opus_gaussian_world_full15e_config/config_check.json`。

## 无 mask 验证接口修复（2026-10-16）

原模型的 `final_occ` 为 `[B, 640000]`，`sampled_label` 却为 `[B, 200, 200, 16]`，通用指标在 `occ3d_eval_mask="none"` 时不会自动展平，因而在第一轮验证触发16与640000维度不匹配。现已在 `model/segmentor/sparse_opus_gausisan_world.py` 将指标标签展平；配置继续使用全空间评估。预测、训练标签、损失及权重格式均保留。

已通过33项CPU回归检查，并对本目录配置、现有第1轮权重和1个真实验证样本完成GPU前向/损失/指标检查。详情见[修复记录](../../analysis/sparse_opus_gaussian_world_full15e_config/metric_shape_fix.md)。这次检查不是4219样本完整验证成绩。

如果是在本文件示例目录中遇到该错误，第1轮权重已在验证前保存，`latest.pth` 指向 `epoch_1.pth`。更新代码后重新执行上面的训练命令即可自动恢复，从第2轮继续，总训练目标仍为15轮。也可明确添加：

```bash
--resume-from out/sparse_opus_gaussian_world_nuscenes_occ3d_15e/epoch_1.pth
```

通用入口从下一个epoch恢复，不会补跑此前失败的第1轮完整验证。
