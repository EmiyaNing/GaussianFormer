# 独立A/B第一轮诊断入口

A配置：`config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_focal_A_4096_1024_15e.py`。
B配置：`config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_ce_free_B_4096_1024_15e.py`。
配置均完全展开，保留15轮学习率预算、原未来课程、4096/1024子集及seed 17。
A使用新的softmax focal实现；配置中的`use_focal_loss=False`只禁用仓库默认sigmoid路径，实际分类由`FullspaceSoftmaxFocalOccupancyLoss`执行。
B使用原OccupancyLoss，仅降低free相对权重至0.1。

```bash
python3.10 -m pytest -q experiments/sparseworld_opus_gaussian_bias/ab_first_epoch/test_contracts.py
python3.10 -m experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.smoke
python3.10 -m experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.queue
```

队列依次训练A、B，第一轮完成后保存全部状态并退出训练子进程，再生成第一轮比较报告。
运行时停止边界不修改`max_epochs=15`，不缩短学习率计划；通过`optimization_budget.json`验证总更新数15360。
`first_epoch_completed.json`标明仅完成1轮；不会生成表示15轮完成的`completed.json`。
若希望第一轮报告之后继续训练第2–15轮，使用相同队列的`--continue-after-diagnosis`参数。

训练入口仅在自己的新进程内复用/包装原runner函数，旧训练、损失和诊断源码均保持原样。
每组模型初始化后计算完整state_dict指纹，保证A/B来自相同随机模型和相同图像初始化。
第一轮用同一评估前向完成区域统计及固定小样本诊断；额外的分类统计通过forward hook取得，不额外跑模型。
解析empty bias梯度经过CPU autograd一致性验证；focal gamma=0经过原weighted CE+Lovasz数值及梯度一致性验证。

输出：`out/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e/{A_softmax_focal,B_ce_free_0_1}/`。
报告：`analysis/sparseworld_opus_gaussian_bias_ab_first_epoch_4096_1024_15e/report.md`。
逐时刻逐类别CSV共126行，比较Strict、camera Gaussian、原全空间CE、A、B的相同epoch结果。
每组完整诊断位于报告根目录对应方案的`epoch_diagnostics/`。
早期诊断结果不代表15轮最优性能。
