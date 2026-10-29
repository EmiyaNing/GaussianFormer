# Query 重要性分支完整微调

配置为 `opusv1_gaussian_small_8f_query_budget_1ep.py`。它从原始 small 权重严格扩展末端输出头，使用完整 Occ3D 训练 1 epoch，并在完整验证集上执行硬选择评价。

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 -u tools/run_gaussian_budget_finetune.py run
```

脚本自动准备初始化权重、执行真实数据检查，然后启动正式训练。当前实验的状态和日志位于 `work_dirs/small_query_budget_camera_full_1ep/`。

训练候选为 1,200×96=115,200，软门控总和为 76,800；推理和最后一级 Set 监督均保留 76,800 个 GS。六个阶段的 Set GT 都先经过 camera mask 过滤。没有新增需要编译的算子，沿用现有 `localagg`。

模型和损失通过配置的 `custom_imports` 注册，无须修改已有模型的导入列表。

详细设置见 [实验设计](../../analysis/query_importance_camera_set_design.md)；当前运行和完成后的指标见 [结果报告](../../analysis/query_importance_camera_set_results.md)。
