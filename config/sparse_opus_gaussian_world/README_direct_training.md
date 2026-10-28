# 从第1轮使用正式训练结构：nuScenes Occ3D，15轮

配置：[sparse_opus_gaussian_world_nuscenes_occ3d_direct_15e.py](sparse_opus_gaussian_world_nuscenes_occ3d_direct_15e.py)。完整独立配置，不继承或运行时读取旧配置。

## 训练协议

`model.finetune_epoch=0`，构建时模型和head的`is_pretrain=False`，并保留每轮`set_epoch`。

- 从第1轮使用正式时间因果attention mask，训练与验证采用一致的查询交互结构。
- 从第1轮启用六层当前帧点集监督、原Gaussian全空间CE/Lovasz及平衡条件语义CE。
- 每轮开始分配时间查询，轮内固定；取消前5轮每batch重新分配的特殊阶段。
- 未来点集预测从0.5秒逐步扩展到3秒，第6轮起训练完整六步。学习率warmup仍保留。
- Gaussian直接监督仍用半秒索引`[0,2,4,6]`，即0/1/2/3秒；0.5/1.5/2.5秒的直接体素监督范围保持原方案。

| 轮数 | 训练的未来点集范围 | 直接Gaussian体素监督 |
|---|---|---|
| 1 | 0.5s | 0s |
| 2 | 0.5–1s | 0/1s |
| 3 | 0.5–1.5s | 0/1s |
| 4 | 0.5–2s | 0/1/2s |
| 5 | 0.5–2.5s | 0/1/2s |
| 6–15 | 0.5–3s | 0/1/2/3s |

验证从第1轮就输出全部六个未来时刻；通用`train.py`每轮记录当前帧指标，与原全量配置一致。

相对原全量配置，训练值只修改`finetune_epoch: 5→0`，并使用新的实验名称。模型实现、损失系数、数据集、骨干初始化、15轮总预算、batch=2、累积梯度=2、基础学习率和调度参数均保留。

未来预测课程随`finetune_epoch`一起前移，因此该方案同时提前了未来点集学习。它不是仅切换attention的单因素消融。两组虽然轮数和优化更新数相同，但累计未来点集监督步数不同：原方案各轮合计45步，新方案75步；完整资源比较还应记录实际训练时间。

## 从头训练

使用独立目录，从相同图像backbone/FPN预训练初始化；世界模块及Gaussian模块随机初始化。这是新的训练协议，应从头训练来比较，不将旧第6轮检查点作为这条曲线的起点。

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 train.py \
  --py-config config/sparse_opus_gaussian_world/sparse_opus_gaussian_world_nuscenes_occ3d_direct_15e.py \
  --work-dir out/sparse_opus_gaussian_world_nuscenes_occ3d_direct_15e \
  --gradient-accumulation 2
```

以后恢复该新实验时，使用同一配置和同一目录，通用入口会自动载入该目录自己的`latest.pth`。

## 验证与结果解释

CPU回归检查覆盖配置独立性、变更范围、第1轮因果mask、轮内查询固定，以及实际世界转移在第1/2/3/4/5/6/15轮的预测步数。GPU smoke使用从头初始化的真实训练batch，检查正式损失和世界分支梯度，再检查真实验证样本的指标接口。

证据保存在[direct训练检查目录](../../analysis/sparse_opus_gaussian_world_direct_training/)。单batch功能检查不能证明首轮或最终性能更高，正式结论需新训练曲线。当前已有的4样本mask诊断只支持值得开展该实验，不足以决定该新协议一定优于原15轮方案。
