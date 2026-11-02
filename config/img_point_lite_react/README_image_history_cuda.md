四帧历史图像的 CUDA 优化配置为
[img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda.py](img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda.py)。
它继承原四帧配置，仍为 **当前＋3 帧图像、当前＋9 帧 LiDAR、六相机 704×256**，默认 `occ3d_eval_mask='none'`。训练数据、GS/query 数量、联合 softmax、学习到的历史 bias 与参数名/形状保持原来的定义；已有四帧权重可严格加载，不需要为了适配算子重新训练。

主要改动：

- 每个图像时刻的 FPN 在一次 CUDA copy 中整理为 `[B,6,sum(HW),C]`，再缓存该结果。两个 decoder 直接使用分帧 packed features，不再拼接 24-view FPN 或重复整理金字塔。packing 的反向只需要形状/stride metadata，不保存一份额外的 FPN 值。
- 将 3D 投影、可见性 mask、dropout、联合 `K*V*L` softmax、双线性采样和归约合并。每个 query 的投影和概率只存在于 CUDA shared memory；反向重算这些量，直接返回三维 keypoint 梯度，不构造全局二维坐标及其梯度。
- 用 warp ballot 将 dropout 存成 bitset。保持 `torch.rand(float32) > attn_drop` 的抽样序列和 generator 推进量，兼容原有 checkpoint RNG 重计算；没有生成全尺寸 float 随机张量。以 60,098 个 query、24 个相机、9 点、4 层、4 组为例，原 float 临时随机张量约 792.3 MiB、bool mask 约 198.1 MiB；bitset 约 24.8 MiB。
- `detach_history=True` 时，反向只为当前 6 个相机分配并写回 FPN 梯度；历史特征仍参与 keypoint、query logits、camera encoder 和历史 bias 的梯度计算。`detach_history=False` 仍支持所有时刻的 FPN 梯度。几何和 FPN 保持 FP32。

这些 buffer 的减少不能直接相加视为整模型峰值下降；完整训练的峰值还取决于点云网络、FFN、渲染和 loss 的阶段。三个真实样本的整模型数据见 [实验结果](../../work_dirs/react_temporal_cuda/RESULTS.md)。

新增一个独立扩展，避免覆盖已加载的原 DAF `.so`。在目标机器自己的 PyTorch/CUDA 环境编译：

```bash
cd model/encoder/gaussian_encoder/temporal_ops
MAX_JOBS=2 python3.10 setup.py build_ext --inplace
```

回到项目根目录。原有 DAF（含固定 image BN）和 renderer 仍是基础依赖，本次不需要重编译它们；也没有修改 spconv/MMCV。不要把本机 `.so` 当作跨机器通用二进制，目标环境应使用上述源码编译。

评估已有四帧权重：

```bash
python3.10 eval.py --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda.py --resume-from out/img_voxel_react_occ3d_lidar10f_img4f.py/epoch_1.pth --work-dir work_dirs/react_img4f_cuda_eval --occ3d-eval-mask none
```

训练入口：

```bash
python3.10 train.py --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda.py --work-dir out/react_img4f_cuda --seed 928
```

优化配置继承原四帧的 load_from/optimizer/schedule。若要延续某次已训练的四帧实验，请相应设置 load_from，或使用其完整 checkpoint 恢复；不要将上面的示例训练指令理解为自动选择最新权重。

推荐配置保留 image、FFN、DAF 重计算，以降低显存。另附
[img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda_speed.py](img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda_speed.py)：只关闭 DAF 重计算，训练进一步加速，显存高于推荐版，部分较小场景也可能高于原版。

接口开关为 `model.pack_temporal_features=True` 和 `model.encoder.deformable_model.fuse_temporal_sampling=True`，需要一起启用。原配置的默认值均为 False。快路径的 segmentor 将 packed 帧和 metadata 直接交给 encoder，`ms_img_feats` 为 None；它不是供外部按传统 FPN 列表解包的接口。缓存的 scene/增广/shape/device 身份检查及清空策略继续沿用。帧指针容量现为 16（较早版本为 8，需要重编译）；每 group 最多 32 channel、最多 16 个 group、shared-memory row 不超过 48 KiB。在当前 9 点/4 层/4 组/每帧6相机的模型中，这个 shared-memory 限制实际允许最多 11 个图像时刻。相机投影和图像尺寸是固定 metadata，不是可训练标定参数。

验收与复现：

```bash
python3.10 -m pytest tests/test_react_temporal_cuda.py tests/test_react_image_history.py tests/test_react_execution_v5.py tests/test_opus_temporal_precision.py tests/test_occ3d_eval_mask.py -q
python3.10 tools/benchmark_react_temporal_cuda.py --samples 139,500,1000 --arms reference,cuda,cuda_no_daf_checkpoint
python3.10 tools/benchmark_react_temporal_cuda.py --phase accuracy --val-frames 256
```

算子输出、所有可训练输入的梯度、dense/channels-last packing、全不可见/空 query、packed RNG 与 checkpoint 均有测试。完整模型含稀疏 voxel 排序和坐标量化，微小浮点差异可能影响后续离散路径，因此不承诺最终预测逐 bit 一致。报告保留了两种 eval mask 的实际结果及原实现自身复测，不能把 256 帧子集直接等同于全量验证。

GPU 特征缓存继续避免重复 image backbone，但不会免除数据端的历史图像读取/解码。模型计算加速不等于数据等待也同比减少；本轮基准明确排除了 DataLoader、H2D、DDP。
