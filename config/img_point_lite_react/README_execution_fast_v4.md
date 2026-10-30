# React decoder 显存优化

后续图像、数据流水线和融合 DAF 优化见 [第五轮说明](README_execution_fast_v5.md)。

在第三轮渲染器之上提供两个入口：

- `img_voxel_react_occ3d_fast_v4.py`：改写投影、线性变换和权重归一化，保持常规反向。
- `img_voxel_react_occ3d_fast_v4_mem.py`：再开启 DAF、FFN 的激活重计算，优先节省显存。

均保留 FP32、Gaussian 数量、9 个采样点、6 个相机、FPN 层级、图像分辨率、全部
三层监督、loss 和 mask。模型参数和 checkpoint 键与第三轮兼容。各新增开关默认
False，第三轮及更早配置保留原路径。

## 改动

1. **投影矩阵按 batch 计算**：将相机维折入矩阵行，执行
   `[B,V*4,4] @ [B,4,N*K]`。避免为每个点保存 `[B*V*N*K,4,4]` 矩阵副本。
   N=58998、K=9、V=6 时，原单层副本约 194.45 MiB；两层约 388.90 MiB。
2. **拆分相机线性变换**：用 `(Wf+b)+Wc` 计算 `W(f+c)+b`，复用原权重且 bias
   只加一次。避免保存 `[B,N,V,128]` 的展开特征，两层约省 288.08 MiB，线性
   层计算量也随之降低。
3. **紧凑 mask**：Python 路径保留 `[B,N,1,1,1,G]` 的布尔有效标记，通过广播
   `masked_fill` 清零全无效行，不再展开成大浮点乘数。可单独开启、无需新增 CUDA。
4. **融合 masked softmax**：CUDA 按 query/group 对 `K*V*L` 归一化，全无效行
   输出零；反向仅保存最终概率张量，与 DAF 共享同一存储。无需单独保留 softmax
   输出、浮点 mask 乘数或可见性/dropout mask。当前 216 元素的行使用 warp
   寄存器实现，超过 256 元素时使用通用路径。dropout 仍在归一化前执行。
5. **选择性 checkpoint**：只对 DAF 和 FFN 启用 non-reentrant 重计算，保留 RNG
   状态。spconv 索引构建、refine、densify 位于区间外，格式化图像特征继续共享。
   eval/no_grad 时自动跳过重计算包装。

实现位于 `deformable_module.py`、`gaussian_encoder.py` 和
`ops/masked_softmax.py`，CUDA 位于现有 DAF 扩展的 `ops/src/`。

独立开关：

```python
model = dict(encoder=dict(
    checkpoint_ffn=False,
    checkpoint_deformable=False,
    deformable_model=dict(
        batched_projection=True,
        factor_camera_weights=True,
        compact_weight_mask=True,
        fused_weight_softmax=True)))
```

## 编译与使用

**需要重编译现有 DAF 扩展**以提供融合 softmax；第三轮 renderer 无需因本轮再编译。
本机已编译，四卡机器需要使用自己的 PyTorch/CUDA/GPU 编译。

```bash
cd model/encoder/gaussian_encoder/ops
MAX_JOBS=2 python3.10 setup.py build_ext --inplace
cd ../../../..

python3.10 train.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v4_mem.py \
  --work-dir out/img_voxel_react_occ3d_fast_v4_mem --seed 928
```

若先使用纯 Python 改写，将 `fused_weight_softmax=False`；其它优化仍然生效，
无需为本轮重编译。仅开启 FFN checkpoint 也可用上述独立开关配置。

历史帧数继承父配置，本地默认仍为 **0**。延续九帧历史实验时，应在派生配置中
显式对齐训练和验证两个数据集：

```python
_base_ = ['./img_voxel_react_occ3d_fast_v4_mem.py']
train_dataset_config = dict(num_lidar_history=9)
val_dataset_config = dict(num_lidar_history=9)
```

## 整网显存和耗时

RTX 4090，PyTorch 2.12.0+cu126，Torch CPU 线程 24。六张 1600×864 图像，batch=1，
当前帧加 9 帧历史点云，验证集索引 128/256。三条路径使用同一组随机初始权重，
每组预热 2 步、正序和反序各测 12 步，以下合并每条路径 24 次测量。

| 项目 | 第三轮 | 第四轮普通版 | 第四轮重计算版 |
| --- | ---: | ---: | ---: |
| 完整训练步均值 | 404.57 ms | 370.02 ms | 381.39 ms |
| 训练 allocated 最大峰值 | 11020.43 MiB | 9464.59 MiB | 8213.65 MiB |
| 相对第三轮峰值降幅 | — | 14.12% | 25.47% |
| 相对第三轮训练吞吐 | 1.00x | 1.093x | 1.061x |
| 训练 encoder 前向均值 | 38.80 ms | 25.21 ms | 25.44 ms |
| 纯模型推理 forward | 123.28 ms | 109.99 ms | 110.09 ms |
| 含数据加载的推理步 | 155.29 ms | 160.27 ms | 157.74 ms |

训练步包含 DataLoader 等待、H2D、全部 loss、backward、梯度裁剪及 AdamW。
基准 AdamW lr=0 以固定权重，状态照常更新和计入显存；排除 worker 启动、DDP、
scheduler、日志和 checkpoint I/O。allocated 指 PyTorch Tensor 分配，非 reserved
或 nvidia-smi 总显存。表中显存为所有测量的最大值，未使用逐步峰值的平均数。

重计算版比普通版训练约慢 3.1%，最大峰值再少约 1251 MiB。推理 forward 有
改善，但本次包含加载的推理流水线未获得稳定加速；输入准备成为实际吞吐限制。
该结果只覆盖本机两条热缓存样本，四卡、已训练权重或完整数据集需各自复测。

另一次分阶段显存追踪（同一样本 128、九帧历史）显示：

- 第三轮 decoder 前向净增约 4989.67 MiB；重计算版约 2081.85 MiB，下降约 58%。
- 重计算版 forward+loss 结束约 7941.12 MiB；backward 峰值约 8215.31 MiB，
  反向后回落至约 920.01 MiB。重计算的临时分配也包含在峰值中。

纯 Python 三项改写的短消融为约 9952.49 MiB 峰值；再融合 softmax 为约
9465.72 MiB，重计算为约 8218.68 MiB。它们来自独立的 6 步定位测量，正式结论
使用上表的 24 步正反序结果，不混合计算峰值。

## 数值检查与范围

24 项新增检查包括投影及相机矩阵梯度、线性变换与 bias 梯度、dropout mask RNG、
softmax 全无效行/空查询/长行/非连续输入/非默认 stream、单概率缓冲保存，以及
完整两层 encoder 重计算后的所有输入/参数梯度和 RNG 状态。与前三轮及共享接口
合计 **95 项回归通过**。

两个配置还分别通过真实 `train.py` 入口的短训练：两条九帧历史样本、非零学习率、
全部梯度有限值检查、一条样本验证及 checkpoint 保存。普通版短跑 loss 约
9.517→8.173，重计算版约 9.515→8.218；仅用于检查训练链路，不作为精度结果。
配置和日志位于 `work_dirs/react_execution_optimization_v4/train_smoke_fast*` 与
`train_smoke_mem*`。

固定真实场景约 58998 个 Gaussian 的 DAF 输入，保持相同权重和 dropout 随机数：

- 两次独立捕获的输出最大绝对差为约 3.68e-6～6.73e-6。
- 参数梯度最大相对 L2 差约 2.65e-5 以内；可见性 mask 均无变化。
- anchor 输入梯度的相对 L2 差为约 1.07e-4～1.42e-3。逐项隔离将主要差异定位
  到投影改写；投影的浮点执行顺序改变，少量投影点在特征像素的 floor 边界附近
  发生分桶变化。线性拆分和 softmax 的隔离差异更小，详见原始记录。

这些是公式保持一致的 FP32 改写，**不承诺逐 bit 一致或完整训练轨迹一致**。
旧稀疏/离散路径在重复执行时也有差异；真实 DAF 审核在每次捕获内部固定输入，
避免混淆上游重复性。尚未进行完整训练的 mIoU 对照。

```bash
python3.10 -m pytest -q \
  tests/test_react_decoder_memory.py tests/test_react_renderer.py \
  tests/test_react_execution.py tests/test_react_execution_v2.py \
  tests/test_gaussian_opus.py tests/test_opus_voxel_fusion.py tests/test_opus_official_v1.py \
  tests/test_opus_multimodal_encoder.py tests/test_opus_head_contract.py \
  tests/test_occ3d_eval_mask.py tests/test_eval_all.py

python3.10 tools/audit_react_decoder_memory.py --isolate \
  --out work_dirs/react_execution_optimization_v4/recheck_daf.json

python3.10 tools/benchmark_react_execution_v2.py \
  --phase all --paths fast3,fast4,fast4_mem --lidar-history 9 --warmup 2 --steps 12 \
  --out work_dirs/react_execution_optimization_v4/recheck.json
```

用 `--paths fast4_mem,fast4,fast3` 反转顺序。其它消融入口有 `fast4_python`（不使用
融合 softmax）和 `fast4_ffn`（只重计算 FFN）。

原始记录位于 `work_dirs/react_execution_optimization_v4/`：`paired.json`、
`paired_reverse.json`、`combined.json`、`ablation_train.json`、`real_daf_audit.json`、
`real_daf_isolated.json` 和 `memory_audit_mem/report.json`。
