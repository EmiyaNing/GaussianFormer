# D/E 第一轮筛选

本包独立注册两个实验，不修改既有源文件。配置是完整静态快照，无运行时继承。

```bash
python3.10 -m experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.frequencies
python3.10 -m experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.prepare_config
python3.10 -m pytest -q experiments/sparseworld_opus_gaussian_bias/semantic_occupancy_first_epoch/test_contracts.py
python3.10 -m experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.smoke
python3.10 -m experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.launch
```

launch要求CPU/GPU审计已通过、正式目录新建且GPU没有计算任务。队列只读重评C，串行训练D/E，各自按15轮学习率计划训练完整第一轮并诊断、保存后暂停。不会自动继续第2轮。

- D：仅根据训练频率重加权17类条件辅助CE。
- E：独立child占用证据、归一化联合概率与固定0.5占用门限。
- 结果自动写到 `analysis/sparseworld_opus_gaussian_bias_semantic_occupancy_DE_4096_1024_15e/`。
- 只使用第一轮当前mIoU作主要判断，报告全部时刻类别IoU和配对场景bootstrap区间。
- 几何研究路线暂存，原高斯参数化、时序结构和其余损失权重保持一致。
