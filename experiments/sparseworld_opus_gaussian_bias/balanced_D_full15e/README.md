# D完整15轮续训

本目录全部是新增文件。原D/E第一轮目录保持不变；D第一轮模型、optimizer和RNG迁移到新目录，续训第2～15轮，保持原15360次更新的学习率预算。

```bash
python3.10 -m pytest -q experiments/sparseworld_opus_gaussian_bias/balanced_D_full15e/test_contracts.py
python3.10 -m experiments.sparseworld_opus_gaussian_bias.balanced_D_full15e.prepare
python3.10 -m experiments.sparseworld_opus_gaussian_bias.balanced_D_full15e.validate
python3.10 -m experiments.sparseworld_opus_gaussian_bias.balanced_D_full15e.launch
```

prepare/launch均防止重复执行。prepare仅允许输出目录发生配置变化，checkpoint只替换config_sha256并逐值重读验证；完整配置是新的静态文件，没有继承。每轮正式评估1024样本，固定16样本做属性、条件语义及渲染诊断，前2样本检查实际监督输出梯度。

新train.py保留原训练循环，补充完整预算检查，并区分整轮边界与轮内中断：前者让set_epoch初始化新轮query统计，后者保留已累计的统计。这避免将上一轮统计覆盖新轮初始化。

持久queue自动依次执行训练、最佳checkpoint独立复评和最终报告。主选择规则沿用当前/1/2/3秒平均mIoU；报告包含七个时刻的逐类IoU、同轮Strict对照与全部15轮诊断。Strict复用既有完整15轮结果。

工作目录：`out/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024/`。
报告目录：`analysis/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024/`。
进度报告：`progress_report.md`；最终报告：`report.md`。
