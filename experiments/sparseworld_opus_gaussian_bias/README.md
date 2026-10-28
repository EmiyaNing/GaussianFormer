# SparseWorld OPUS Gaussian bias：独立实验入口

所有实现与配置均为新增文件。现有训练/评测入口、注册类型和旧配置保持原样。
运行目录为 GaussianFormer 根目录，Python 为 `python3.10`。

```bash
PYTHONPATH=. python3.10 -m experiments.sparseworld_opus_gaussian_bias.prepare_subset
PYTHONPATH=. python3.10 -m pytest -q experiments/sparseworld_opus_gaussian_bias/test_contracts.py
PYTHONPATH=. python3.10 -m experiments.sparseworld_opus_gaussian_bias.queue
```

GPU 命令需要能够访问 `/dev/nvidia*` 的执行环境。queue 顺序训练两组各 15 epoch，
自动恢复已有 `latest.pth`，每组完成后重新评测最优 checkpoint，再生成最终报告。
不能同时启动两个 queue；查看当前进程和状态后再恢复。

配置（完整配置，没有 `_base_` 或其他配置继承）：

- `config/sparseworld_opus_gaussian_bias/sparseworld_strict_4096_1024_15e.py`
- `config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_4096_1024_15e.py`

子集：`data/sparseworld_opus_gaussian_bias_4096_1024/{train,val,summary}.json`。
包含 4096/1024 个当前样本的 token、历史/未来上下文、文件列表和源文件 SHA256。
数据集保持源场景完整索引，不复制图像，不截断历史。

运行状态：`analysis/sparseworld_opus_gaussian_bias_4096_1024_15e/status.json`。
模型日志、进度和 checkpoint：
`out/sparseworld_opus_gaussian_bias_4096_1024_15e/{sparseworld_strict,sparseworld_opus_gaussian_bias}/`。
每组每个 epoch 完整评测 1024 个样本，输出七个时刻的指标 JSON 和原始混淆矩阵 NPZ。

最终报告：`analysis/sparseworld_opus_gaussian_bias_4096_1024_15e/report.md`。
CSV 包含 7 时刻 × 18 类（17 个非空类别及 free）的两组 IoU、差值和 GT 体素数。
主最优 epoch 由当前与未来 1/2/3 秒平均 mIoU 统一选择。1024 个 held-out 样本承担
checkpoint 选择验证角色，不作为额外独立测试集宣称。

调试命令（仅一个训练 batch 和一个验证样本；不构成正式结果）：

```bash
PYTHONPATH=. python3.10 -m experiments.sparseworld_opus_gaussian_bias.train \
  --config config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_4096_1024_15e.py \
  --work-dir out/sparseworld_opus_gaussian_bias_4096_1024_15e/smoke_gaussian --smoke
```

checkpoint 每 250 次 optimizer 更新及每个 epoch 保存，包含 RNG、样本游标、
时间 query 分配、统计、模型与 optimizer。每样本数据增强按 epoch/index 固定 seed，
恢复时不会因 worker 重启改变图像增强；保存边界始终落在完整梯度累积步。

实施方案：`plans/sparseworld_opus_gaussian_lenared_bias.md`。
