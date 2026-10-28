# 全空间监督从头训练与逐轮诊断

独立配置：`config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e.py`。
配置完整展开，运行时不继承其他配置。相对原Gaussian实验，训练语义仅改变`loss.mask_type='none'`。
仍使用4096/1024固定子集、seed 17、15轮、原未来训练课程和相同图像backbone/FPN初始化；不加载原世界模型checkpoint。
CE监督包括整个200×200×16空间的有效标签及free类；Lovasz仍按原设置忽略free。

```bash
python3.10 -m experiments.sparseworld_opus_gaussian_bias.queue_fullspace
```

新输出根目录：`out/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/`。
其中`sparseworld_strict`只读复用已完成15轮的原Strict结果，`sparseworld_opus_gaussian_bias`存放本次全空间实验。
首次启动必须使用空的新训练目录；后续同配置重启由原runner的hash保护和checkpoint支持恢复。

`train_fullspace.py`仅在新入口的进程内替换评估函数，复用原runner其余流程，不修改原源码。
每轮完成训练后评估全部1024个样本，用同一前向统计全空间、camera、noncamera混淆矩阵。
正式成绩为全空间17个非空类别的mIoU，free另列；单checkpoint按照当前/1/2/3秒平均mIoU选取。
1024子集参与checkpoint选择，因此承担验证集角色。

每轮在`analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e/epoch_diagnostics/`生成：

- `epoch_XX_diagnostics.md/json`：所有7个时刻的区域指标、误报/漏报、与Strict同轮对比、Gaussian属性、渲染探针、输出层梯度比较。
- `epoch_XX_class_iou.csv`：7个时刻×18类别的IoU，以及与Strict同轮结果的差值。
- `epoch_XX_visibility_confusion.npz`：完整1024样本的full/camera/noncamera精确计数。

固定16个等距验证样本记录尺度/opacity饱和、feature范数和ROI有效点。
同16样本做point rasterizer、empty bias±1/±2、固定尺度0.2、opacity×0.5渲染对照。
前2个样本比较实际启用监督时刻的末层集合损失与体素损失梯度，使用原未来范围mask；这不是整个模型参数梯度。
探针不更新参数，不调整阈值，不参与checkpoint选择，并恢复诊断前的随机数状态。
正式完整集指标和小样本探针分别标明样本数；smoke诊断放在独立smoke目录。

15轮结束后队列独立复核最优checkpoint，再由`report_fullspace.py`生成最终报告、126行类别对比、曲线及15轮诊断趋势。
报告要求15轮都评估满1024样本、每轮诊断full矩阵等于正式矩阵及camera+noncamera之和，最优复核也必须逐元素一致。
原训练、checkpoint及诊断均保留在旧目录。
