配置入口为 [img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f.py](img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f.py)：**4 个图像时刻 = 当前帧 + 3 帧过去图像**，每时刻 6 相机；LiDAR 保持当前帧 + 9 帧历史，分辨率为 704×256。Occ3D 默认评估无 camera mask。

已有权重可直接使用的 CUDA 加速版见 [四帧 CUDA 配置与编译说明](README_image_history_cuda.md)。

`LoadReactImageHistory` 复用 OPUS 的因果选择：训练从过去 sweeps 按 4–8 的随机间隔取帧，验证取过去 keyframes（约 0.5/1.0/1.5 秒前）。历史相机用自身标定和 pose 投影当前 ego 坐标的三维点，resize/crop/flip 同时更新所有投影。场景起点补齐张量形状，`image_frame_valid` 将补帧从采样 softmax 中排除。补帧直接复制已读取的图像，不重复解码。没有读取未来图像或历史 occupancy 标签。

`ReactTemporalSegmentor` 按图像时刻提取特征，使用 BF16 image backbone、FP32 FPN 和非重入 image checkpoint。默认 `detach_history=True`：仅当前图像分支训练 backbone/FPN；query 坐标、query 特征和时间融合参数仍能从历史采样获得梯度。设为 False 可以对所有时刻反传，显存和计算会增加。

两个 decoder 使用 `TemporalDeformableFeatureAggregation`，将时间和相机折叠为 24 个 view，继续使用既有融合归一化采样 CUDA 算子。在联合可见性 softmax 中，每个历史时刻新增一个可训练 logit bias，初始值 -2，当前时刻 bias 为 0。两个 decoder 总共新增 **6 个参数**；其余参数名和形状兼容单帧权重。query/GS 数量与现有 voxel lifter、DensifyOnly 保持相同规则。`image_time_offsets` 保留用于诊断，当前版本通过投影 embedding 和时间位置 bias 融合，尚不包含动态物体运动预测。

默认从 `ckpts/img_voxel_react_occ3d_fast_v5_mem.pth` **warm-start**，不是恢复旧 optimizer/epoch。主配置使用 AdamW lr=2e-5、图像 backbone lr_mult=0.1、200 step warmup 和 cosine schedule；训练 epoch 数继承父配置，可按预算覆盖。若要从原始 RayDN 预训练做完整的 baseline 对照，应同时调整 load_from、学习率与训练轮数，不要混同为本次短微调结果。

```bash
python3.10 train.py --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f.py --work-dir out/react_lidar10f_img4f --seed 928
```

验证训练得到的权重：

```bash
python3.10 eval.py --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f.py --resume-from out/react_lidar10f_img4f/epoch_1.pth --work-dir work_dirs/react_img4f_eval --occ3d-eval-mask none
python3.10 eval.py --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f.py --resume-from out/react_lidar10f_img4f/epoch_1.pth --work-dir work_dirs/react_img4f_eval_camera --occ3d-eval-mask camera
```

推理 cache 仅在 eval、no_grad、batch=1 且身份 metadata 完整时启用。key 包含 scene、各相机文件、图像增广、shape/dtype/device；每个新 scene、进入训练、加载权重或迁移 device/dtype 时清空。默认最多保留 8 个六相机时刻，FP32 FPN 约 87.66 MiB。连续单 GPU 验证通常可复用前三帧；随机顺序、分布式交错采样和 cache miss 会重新提取相应历史图像。cache 不会消除 DataLoader 的历史图像读取和解码。

可用 `model.clear_image_cache()` 手动清空，`model.image_cache_stats()` 查看命中率。`model.image_cache_size=0` 关闭缓存。调试时可以给两个 Temporal DAF 设置 `history_enabled=False`，屏蔽历史贡献；这不会自动减少图像读取和 backbone 工作量。

本版本没有新增 CUDA 源码。需要已有 v5 的 `normalized_sampling` 和固定图像 BN 仿射算子；若目标机器仅编译过较早的 DAF，应在其 PyTorch/CUDA 环境重编译 `model/encoder/gaussian_encoder/ops`，沿用 v5 的编译方法。已有 tiled renderer、spconv 和 MMCV 不因这个版本增加新的算子。

复现实验：

```bash
python3.10 -m pytest tests/test_react_image_history.py -q
python3.10 tools/experiment_react_image_history.py --phase profile
python3.10 tools/experiment_react_image_history.py --phase pilot --train-steps 64 --val-frames 256
```

`profile` 使用真实历史图像和 pose，比较相同优化设置的单帧对照与 4 帧版本。缓存测试强制当前时刻每次重提，保留三帧真实历史特征；离线测试每次重提全部时刻。时间不包含数据加载/H2D/DDP。

`pilot` 从同一单帧 checkpoint 开始，用同样 64 个完整训练输入、相同顺序和恒定 lr=2e-5 分别更新两个模型，验证采用官方 val 中 256 个固定帧，来自与训练不相交的 scene。每次更新重新提取特征，训练输入缓存不含特征或 occupancy 预测。一次前向同时统计无 mask 与 camera mask 两种口径。短实验不是一个完整 epoch，也不是 6019 帧全验证结论。

实验输出在 [work_dirs/react_image_history_4f](../../work_dirs/react_image_history_4f/)，含原始 profile/pilot JSON、短训练权重及结果报告。
