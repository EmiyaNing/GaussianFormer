# OPUS-V1 Gaussian softplus + learned bias：8 帧配置

按最新要求仅准备模型与配置，不进行正式训练。首次启动的队列已停止，四个尺寸均无完整 epoch 验证结果。以下命令仅供以后主动启动使用。

8 帧指当前帧 + 7 个历史帧，每帧 6 路相机。完整 Occ3D：train 28,130 / val 6,019。

| 文件尺寸 | 官方对应 | query 数 | sampling 点数 | 六层子 Gaussian 数 | 最终 Gaussian 数 |
| --- | --- | ---: | ---: | --- | ---: |
| tiny | OPUS-T | 600 | 4 | 1,4,16,32,64,128 | 76,800 |
| small | OPUS-S | 1,200 | 2 | 1,4,8,16,32,64 | 76,800 |
| middle | OPUS-M | 2,400 | 2 | 1,2,4,8,16,32 | 76,800 |
| large | OPUS-L | 4,800 | 2 | 1,2,4,8,16,16 | 76,800 |

尺寸设置来自 [OPUS 论文 Table 7](https://arxiv.org/html/2409.09350v2#A4.T7)。
官方仓库当前 revision `fdb17339f40168ea0ceda9615863e3f5d0c3f5ce` 只有 T/L 的配置文件，S/M 的采样与六层子点设置采用论文原始表格，并非猜测。

统一 R50/FPN 256 维、6 层 decoder、704×256 图像、30 epoch、逐轮完整验证。
继承已验证的 camera-mask occupancy loss、全部非空 GT 的 OPUS set loss、softplus Gaussian semantics 和训练/推理一致的 learned empty bias。

独立组件：`LoadOPUSV1ImageHistory`、`OPUSTemporalSegmentor`、`OPUSV1GaussianTemporalEncoder`。
训练随机 sweep 间隔 4–8；验证取历史关键帧。补帧复用最早已选历史帧，无历史时复用当前帧。
保留全部历史帧梯度。图像 backbone/FPN 用 BF16、按 6 张图像分块并做激活重计算；decoder/GaussianFormer head/loss 用 FP32，decoder 也做激活重计算。

正式入口使用独立 runner，维持参数组 LR 比例及梯度累积口径：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python3.10 -u tools/run_opus_gaussian_8f_queue.py
```

四组按 tiny → small → middle → large 顺序执行；batch 1 / 梯度累积 8，对应 effective batch 8。
每轮保存 checkpoint；`best_mIoU.pth` 和 `best_IoU.pth` 随完整验证更新。
失败后同一入口从各自 `latest.pth` 恢复，不覆盖旧单帧实验。

详细协议及各尺寸设计/结果文档见 `plans/sparse_render_offical_9_28/full_occ3d_gaussian_8f_30e/`。
