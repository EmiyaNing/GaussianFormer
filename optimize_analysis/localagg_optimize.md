# LocalAgg React 训练显存优化分析

## 1. 结论

`localagg_react` 已经比普通 `localagg` 更适合当前的各向异性 Gaussian：它使用三轴 radius，而不是以 `max(scale)` 构造立方体包围盒。但是，训练峰值的主导项仍然是 **Gaussian-tile 重叠实例数**，记为 `R`。当前实现会：

1. 将每个 Gaussian 覆盖的 tile 显式展开为 `R` 条记录；
2. 对记录排序；
3. 将几何、排序和 image range buffer 全部保存到 autograd context，直到 backward；
4. 在 backward 中再分配梯度和 `voxel2pts` 映射。

因此，应优先减少 `R`，以及减少与 `R` 成正比、但被长期保存的 renderer context。混合精度不是首选，因为最大的 buffer 是整数索引、排序 workspace 和保存的 CUDA context，而非 FP32 feature。

推荐优先级：

1. 保持单层监督，增加 `R`、radius 和 buffer 的运行时观测；
2. 实现 **压缩 renderer context + backward 重建 tile 枚举**；
3. 提供可选的 **renderer checkpoint**，以额外排序时间换峰值显存；
4. 增加带验证的 radius / tile 预算，约束大背景 Gaussian；
5. 仅对真正参与 loss 的 query voxel 渲染；
6. 在上述路径稳定后，再考虑索引压缩和多分辨率 renderer。

## 2. 当前 `localagg_react` 的事实

### 2.1 React 已经消除了一部分无效展开

普通 localagg 使用：

```text
r = ceil(max(sx, sy, sz) * scale_multiplier / grid_size)
V = (2r + 1)^3
```

而 react 使用：

```text
rx, ry, rz = ceil([sx, sy, sz] * scale_multiplier / grid_size)
V = (2rx + 1)(2ry + 1)(2rz + 1)
R = sum_g V_g
```

对应实现见：

- `model/head/localagg_react/local_aggregate_react/__init__.py:176`
- `model/head/localagg_react/src/auxiliary.h:8`
- `model/head/localagg_react/src/forward.cu:23`

这对扁平 Gaussian 很有效。例如 `r=[8, 2, 1]` 时，react 只展开 `17*5*3=255` 个 tile；普通版本会展开 `17^3=4913` 个 tile，理论上少约 19 倍。

### 2.2 当前 bg refine 配置已经避免了多层 renderer 并存

`config/adaptive_allocation/adaptive_allocationv5_bg_refine.py` 使用：

```python
apply_loss_type='random_1'
use_localagg_react=True
```

`random_1` 在训练时只选择最后一层表示进行监督；不会像 `apply_loss_type='all'` 那样同时保存多套 localagg context。这个设定应保留。对比 `GaussianHead.forward()` 的 layer 选择逻辑：`model/head/gaussian_head.py:133-146`。

同一配置中的 `freeze_gaussian_head=True` 只冻结 head 参数；它不会让 localagg 进入 `no_grad`。最后一层 Gaussian 仍依赖可训练的 densify 分支，custom autograd 仍必须保存 renderer context 并向 `means/opacities/semantics/covariance` 回传梯度。因此，冻结 head 不能替代本文的 context 优化。

### 2.3 当前 scale 上界仍可制造极大的 `R`

bg refine 的 `scale_range=[0.08, 1.28]`、`scale_multiplier=3`、`grid_size=0.5`，所以：

```text
r_axis = ceil(scale_axis * 6) in [1, 8]
```

一个三轴都接近上界的 Gaussian 最多覆盖 `17^3=4913` 个 tile。更重要的是，`with_empty=True` 会附加 scale 为 `[100, 100, 8]` 的 empty Gaussian；其 radius 会被 grid 边界截断，但仍覆盖整个 `200*200*16=640,000` tile 网格。它本身就会贡献一个不可忽略的 `R`。

## 3. 显存账本

令：

- `P`：参与 render 的 Gaussian 数，含 empty Gaussian；
- `N`：query voxel 数，当前通常是 `200*200*16=640,000`；
- `C=18`：语义通道数；
- `R=sum_g V_g`：Gaussian-tile overlap 数。

### 3.1 与 `R` 成正比的主项

`BinningState` 为排序前后各保存 key 和 Gaussian id：

```text
point_list                  uint32[R]
point_list_unsorted         uint32[R]
point_list_keys             uint32[R]
point_list_keys_unsorted    uint32[R]
```

原始数组已是 `16R bytes`，见 `model/head/localagg_react/src/aggregator_impl.cu:135-147`。此外 CUB radix sort 还需要 workspace，因此实际 binning buffer 通常约为 `24R` 到 `32R bytes`。该 buffer 被 Python custom autograd 保存到 backward：

`model/head/localagg_react/local_aggregate_react/__init__.py:48-63`。

举例：

| `R` | 四个 uint32 数组 | 含排序 workspace 的合理量级 |
| ---: | ---: | ---: |
| 20 M | 305 MiB | 0.45-0.60 GiB |
| 50 M | 763 MiB | 1.1-1.5 GiB |
| 80 M | 1.19 GiB | 1.8-2.4 GiB |

这些空间在 forward 后不会释放，因为 backward 仍依赖它们。

### 3.2 不随 `R` 增长、但不可忽略的项

| 项 | 估算 | 说明 |
| --- | ---: | --- |
| renderer logits | `N*C*4 = 43.9 MiB` | `out_logits`，见 `local_aggregate.cu:54` |
| tile ranges | `N*uint2 = 4.88 MiB` | 每个 tile 的 `[start,end)` |
| `voxel2pts` | `N*int32 = 2.44 MiB` | backward 时分配 |
| semantic gradient | `P*C*4` | `P=40k` 时约 2.75 MiB |
| covariance gradient | `P*6*4` | 通常较小 |

所以真正决定“是否爆显存”的不是 `P` 本身，而是 `R/P`，即每个 Gaussian 平均覆盖多少 tile。

## 4. 正确性边界

`localagg_react` 直接用 local-axis scale 构造 world-axis AABB。若 Gaussian rotation 显著，正确的 world-axis 包围半径应基于协方差：

```text
sigma_world_axis = sqrt(Cov[axis, axis])
r_axis = ceil(k * sigma_world_axis / grid_size)
```

直接继续缩小现有 `radii` 可能减少显存，但也可能漏掉旋转后仍有贡献的 tile。优化前必须先测量 rotation 分布和 tile 边界漏检率；在 rotation 不接近单位阵时，建议先改为 conservative covariance AABB，再在此基础上做预算裁剪。

## 5. 推荐优化方案

### P0. 增加观测与硬保护

先在 C++ forward 中输出或记录以下量：`R`、`R/P`、每轴 radius histogram、被 grid 边界截断的比例、单个 Gaussian 最大 `V_g`、binning buffer 字节数。应在 Python 日志中按固定频率记录，而不是每步 `cudaMemcpy` 到 CPU。

同时增加可配置保护参数：

```python
cuda_kwargs=dict(
    max_radius_xyz=[4, 4, 2],        # 初始实验值，须验证精度
    max_tiles_per_gaussian=405,
    max_rendered_instances=...,      # R 总预算
)
```

这不是最终算法，只是避免罕见的大 background Gaussian 直接触发 OOM。empty Gaussian 不应按相同规则直接裁掉；它应单独保留为全局 free-space prior，或改为单独的解析分支。

预期：对少量异常大 GS 可立即把 `R` 限制在可预测范围。风险：直接 clipping 会改变监督范围，必须比较 coverage、loss、mIoU 和梯度。

### P1. 压缩保存到 autograd 的 renderer context

这是最有价值、且不改变数值结果的 CUDA 改动。

forward 完成 `identifyTileRanges` 后，以下数据已不再需要用于 forward：

- sorted tile key；
- unsorted Gaussian id；
- CUB sorting workspace。

当前实现仍把整个 `binningBuffer` 保存到 backward。应拆分为：

```text
forward temporary:
  sort input/output + CUB workspace

saved renderer context:
  sorted_gaussian_id[R]  # forward tile ranges 使用
  ranges[N]              # 每 tile 的起止区间
  small per-G metadata
```

随后将 backward 改成直接根据 `means3D_int + radii` 重建每个 Gaussian 的 `(x,y,z)` tile 枚举，而不是读取 `point_list_keys_unsorted[R]` 和 `point_offsets[P]`。现有 backward 的每个 Gaussian 本来就在顺序遍历自己的 tile；将线性 key 数组换成三重循环即可。

最小持久状态可降到约：

```text
sorted_gaussian_id: 4R bytes
ranges:              8N bytes
means3D_int + radii: 24P bytes
```

相对当前约 `24R-32R` 的 binning context，持久 `R` 相关显存理论上可下降约 75% 以上。实现位置：

- `model/head/localagg_react/src/aggregator_impl.cu`
- `model/head/localagg_react/src/backward.cu`
- `model/head/localagg_react/local_aggregate.cu`
- `model/head/localagg_react/local_aggregate_react/__init__.py`

### P2. Renderer checkpoint：以计算换显存

提供 `checkpoint_binning=True`：forward 完成 render 后不在 `ctx.save_for_backward()` 中保存 geometry/binning/image buffer；backward 重新执行 preprocess、scan、duplicate 和 sort，再计算梯度。

优点：forward 与 backward 之间不再持有 `O(R)` context，适合显存受限训练。代价：backward 会多一次 tile 展开与 radix sort，renderer 时间明显增加。建议与 P1 共存：

- 默认模式：保存 P1 压缩 context；
- checkpoint 模式：保存 `means3D_int/radii` 等 `O(P)` 元数据，backward 重建。

### P3. `R` 预算化，而不是只限制 Gaussian 数

仅限制 `P` 不足以控制显存，因为大尺度 GS 的 `V_g` 可以远大于普通 GS。应先计算 `tiles_touched[g]`，再按预算处理：

1. 设定 `R_budget`；
2. 保留 empty prior；
3. 对其余 Gaussian 按 opacity、语义置信度和局部需求计算 priority；
4. 当 `sum V_g > R_budget` 时，优先缩小低 priority 的 radius，或暂时不参与 render；
5. 记录被裁剪比例，并在训练早期只做软预算。

更平滑的选择是按每个 Gaussian 的 `max_tiles_per_gaussian` 做等比例三轴缩放：

```text
alpha = min(1, (tile_cap / V_g)^(1/3))
r' = max(1, floor(alpha * r))
```

它保留 react 的长宽高比例，避免直接把各轴都截成同一个半径。由于 radius 已经 detach，选择本身不需要可微；但其造成的 render support 改变仍需做精度消融。

### P4. 只渲染参与 loss 的 query voxel

当前 `GaussianHead` 在 `_sampling(..., None)` 中先渲染完整 `640k` query voxel，之后 loss 才可能利用 `occ_mask` 过滤。可将 mask 前移到 head：只构造 valid voxel 的 `sampled_xyz/sampled_label`，并让 loss 知道该输出已采样。

这不会降低排序主项 `R`，但会降低：

- logits 和其梯度；
- `pts / points_int`；
- render 与 backward 的 query-side 时间；
- `voxel2pts` 的有效工作量。

必须修改 head 和 loss 的 mask 语义，避免二次 mask 导致 shape 不一致；建议单独开关 `sample_valid_voxels_for_loss`，并做严格的 loss 对齐测试。

### P5. 索引类型压缩

当 `P < 65536` 时，`sorted_gaussian_id` 可以安全使用 `uint16`，从 `4R` 降到 `2R bytes`。当前 history / adaptive 场景通常可能满足这一条件，但必须有 `P>=65536` 时回退到 `uint32` 的实现。

tile id 需要约 20 bit，通常仍保持 `uint32`。不要为了压缩而牺牲 key 的排序正确性。

### P6. 更大规模的模型设计

若 background Gaussian 仍主导 `R`，应避免让它们以大 AABB splat 的方式参与同一 renderer：

- 将 free-space / background 改为低分辨率体素先验，再与局部 Gaussian logits 融合；
- 近场使用 0.5 m grid，远场或大 GS 使用更粗的 tile grid；
- 按 semantic / opacity router 分层，只对高价值 GS 做细网格 splatting；
- 对大型近球 GS 改用解析背景项或共享 basis，而非逐 GS 展开。

这些方案可明显降低 `R`，但会改变模型归纳偏置，应在 P1/P2 后再进入实验。

## 6. 不建议作为第一步的方案

- **只开启 AMP**：当前 extension 用 `data<float>()` 和 FP32 output，且索引 / sort buffer 才是主项。直接 autocast 有兼容性风险，收益有限。
- **只调用 `empty_cache()`**：它只能释放缓存，不改变 live autograd context 的大小，也会降低吞吐。
- **把 `scale_multiplier` 粗暴改到很小**：会同时改变 Gaussian 监督覆盖范围，容易掩盖而非解决 renderer 结构问题。
- **恢复 `apply_loss_type='all'`**：这会按监督层数复制 renderer context；在 localagg 完成 P1/P2 前应避免。

## 7. 建议实施顺序与验收

1. **观测 PR**：记录 `P/R/radius histogram/buffer bytes`，建立可复现实验 batch。
2. **P1 压缩 context PR**：forward 与 backward logits、输入梯度逐元素或相对误差对齐；目标是不改变 loss。
3. **P2 checkpoint PR**：验证峰值显存下降与 renderer 时间增幅，作为显存不足时的开关。
4. **P3 tile budget 消融**：比较 `R`、peak allocated/reserved、loss、mIoU、coverage。
5. **P4 query mask**：验证 sampled label、loss mask 和 metrics 完全对齐。

每一步至少记录：

```text
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
R, R/P, max(V_g)
forward/backward renderer time
loss、梯度范数、validation mIoU
```

目标不是只让 `nvidia-smi` 看起来更低，而是在固定 batch、固定随机种子下，用可解释的 `R` 预算换取可控峰值，并保证 Gaussian 覆盖和训练信号不被意外破坏。
