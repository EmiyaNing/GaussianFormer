# gaussian_statistic.py 统计异常原因分析 v2

## 背景

用户提到的日志路径 `./out/img_voxel_lite/20260610_174794.log` 在当前工作区中不存在。当前目录下最接近、且包含同一组异常现象的日志是：

- `./out/img_voxel_lite/20260610_174749.log`
- `./out/img_voxel_lite/gaussian_statistic_result.json`

该 run 的关键结果：

- `Frames processed = 6019`
- `Total Gaussians = 99,722,290`
- `Coverage Threshold tau = 3.0`
- `Coverage Scope = non_empty_visible`
- `Mean Scale = 0.578907`
- `Mean Coverage = 0.910293`
- distance-wise coverage 全部为 `1.0000`
- distance-wise scale 中 `(10,20)` 和 `(20,30)` 的 `MeanScale = 1.000000`
- category-wise 中 class 15 的 `MeanScale = 1.000000`

其中 `config/nuscenes_gs25600_voxel.py` 和该日志打印的 config 均显示：

```python
scale_range = [0.08, 0.64]
```

因此在这个 run 中，任何 Gaussian 的 `s_hat = mean(scale_x, scale_y, scale_z)` 理论上都不应该超过 `0.64`。如果 category-wise 或 distance-wise 的 mean scale 出现 `1.0`，它不是模型真实尺度，而是统计累加逻辑出了问题。

## 结论概览

最主要原因是 `gaussian_statistic/aggregator.py` 中用于 distance-wise、category-wise、distance-wise coverage 的 GPU 累加器都是默认 `float32`：

```python
self._dist_count = torch.zeros(n_bins, device=device)
self._cat_count = torch.zeros(nc, device=device)
self._dcov_covered = torch.zeros(n_bins, device=device)
self._dcov_total = torch.zeros(n_bins, device=device)
```

`float32` 只能精确表示到 `2^24 = 16,777,216` 以内的逐 1 递增整数。当某个 bin/class 的累计 count 超过这个值后，继续 `+1` 会因为尾数精度不足而不再稳定改变数值。

日志中大量异常数值正好卡在 `16,777,216`：

```text
Class 11 count = 16777216
Class 15 count = 16777216
Class 16 count = 16777216

Distance (0,10)  count = 16777216
Distance (10,20) count = 16777216
Distance (20,30) count = 16777216

distancewise_coverage_debug:
  every bin covered = 16777216
  every bin total   = 16777216
```

这个 `2^24` 指纹非常强，基本可以判定为 float32 计数器精度溢出/饱和导致的统计污染。

## 异常 1：category-wise mean scale 为什么会大于 1.0 或等于 1.0

### 在 `img_voxel_lite/20260610_174749.log` 中

该 run 的 `scale_range = [0.08, 0.64]`，所以 `mean_scale > 0.64` 都不合理。日志中：

```text
class 15 | count 16777216 | mean_scale 1.000000 | NSR 1.0000
class 16 | count 16777216 | mean_scale 0.862517 | NSR 1.0000
```

这与 `scale_range` 明显冲突。

直接原因是：

1. `cat_count` 用 float32 累加，达到 `2^24` 后失真。
2. `cat_sum_scale` 也用 float32 累加大量小数，累计到千万级后精度严重下降。
3. 最终 `mean_scale = cat_sum_scale / cat_count` 使用的是已经失真的分子和分母。

所以 category-wise mean scale 不是对真实 Gaussian scale 的可靠平均。

### 如果是在 adaptive_allocationv4_refine 相关 run 中

需要额外注意：`config/adaptive_allocation/adaptive_allocationv4_refine.py` 里：

```python
scale_range = [0.08, 1.8]
```

因此在 refine 配置下，category-wise mean scale 大于 `1.0` 本身不一定是 bug，可能是合法结果。判断标准应是：

- 如果该 run 的 `scale_range[1] <= 1.0`，则 category-wise mean scale 大于 1.0 明显不合理。
- 如果该 run 的 `scale_range[1] = 1.8`，则 mean scale 大于 1.0 可能合理，但仍要警惕 class count 是否卡在 `16,777,216`。只要 count 出现 `2^24`，该 class 的 mean scale 依然不可信。

## 异常 2：distance-wise scale 中出现两次 1.0

日志中：

```text
(10,20) | count 16777216 | MeanScale 1.000000 | NSR 1.0000
(20,30) | count 16777216 | MeanScale 1.000000 | NSR 1.0000
```

这两个 bin 的 count 都正好等于 `2^24`。在 `scale_range = [0.08, 0.64]` 的 run 中，`MeanScale = 1.0` 不可能来自真实 Gaussian。

对应代码在 `GaussianStatAggregator.add_frame()` 中：

```python
self._dist_count.scatter_add_(0, bc, valid.float())
self._dist_sum_scale.scatter_add_(0, bc, s_hat * valid.float())
self._dist_sum_ar.scatter_add_(0, bc, ar * valid.float())
self._dist_nsr_count.scatter_add_(0, bc, nsr_mask * valid.float())
self._dist_nsr_total.scatter_add_(0, bc, ones_g * valid.float())
```

这里所有 distance-wise 累加器都是 float32。跨 6019 帧累计后，某些 bin 的 Gaussian 数超过 `2^24`，计数器和 sum 都失真，最终 ratio/mean 被污染。

因此 distance-wise scale 的两次 `1.0` 与可视化不一致，不是视觉误差，而是统计计数器类型错误导致的结果。

## 异常 3：distance-wise coverage 全部为 1.0

该 run 的 JSON debug 已经直接暴露了矛盾：

```json
"coverage_debug": {
  "sum_dcov_covered": 83886080.0,
  "sum_dcov_total": 83886080.0,
  "scalar_cov_covered": 217120362.0,
  "scalar_cov_total": 238517124.0,
  "coverage_from_bins": 1.0,
  "coverage_from_scalar": 0.910292554089324,
  "coverage_counter_abs_diff": 0.08970744591067603
}
```

并且每个 distance bin 都是：

```json
"covered": 16777216.0,
"total": 16777216.0,
"coverage": 1.0
```

这说明 distance-wise coverage 的分子和分母都被 float32 累加器卡到了 `2^24`，所以比值被错误地变成了 `1.0`。

全局 `Mean Coverage = 0.910293` 反而更可信，因为全局 coverage 在当前代码中是用每帧 `.sum().item()` 累加到 Python 标量：

```python
self.total_cov_covered += (covered & valid_b).sum().item()
self.total_cov_total += valid_b.sum().item()
```

而 distance-wise coverage 使用的是 GPU float32 scatter 累加器：

```python
self._dcov_total.scatter_add_(0, bc, ones_n * valid_b.float())
self._dcov_covered.scatter_add_(0, bc, covered.float() * valid_b.float())
```

二者来自不同数值精度路径，所以出现了：

- 全局 coverage 约 `0.91`
- distance-wise coverage 全 `1.0`
- counter consistency diff 约 `0.0897`

这不是 coverage 定义问题，而是 distance-wise coverage 计数器已经失真。

## tau=3.0 也会放大 coverage 偏高现象

该日志使用：

```text
Coverage Threshold tau = 3.0
```

`calculator.compute_coverage_and_purity()` 中 coverage 判定是：

```python
search_radius = cov_threshold * scales.max(dim=-1).values
mahal <= cov_threshold ** 2
```

也就是统计的是 `3 sigma` 椭球覆盖。tau=3.0 本来就会显著提高 coverage，尤其当前统计范围又是 `non_empty_visible`，只统计可见非空体素。

不过 tau=3.0 只能解释 coverage 偏高，不能解释每个 bin 都精确等于 `1.0`。精确全 `1.0` 的直接证据仍然是 `covered` 和 `total` 都卡在 `16,777,216`。

## 其他需要注意的问题

### 1. `--exclude-gaussian-classes` 当前没有传进 aggregator

`gaussian_statistic.py` 里 parser 定义了：

```python
parser.add_argument('--exclude-gaussian-classes', ...)
```

但构造 aggregator 时只传了：

```python
exclude_classes=args.exclude_classes,
```

没有传：

```python
exclude_gaussian_classes=args.exclude_gaussian_classes
```

所以如果运行时使用了 `--exclude-gaussian-classes`，它实际上不会生效。这不一定是本次 `1.0` 异常的主因，但会影响 coverage/purity 的统计口径。

### 2. DDP 多卡统计仍可能只输出 rank 0 子集

脚本末尾写着：

```python
# 多卡汇总（简化方案：仅 rank 0 输出）
```

如果多卡运行，没有 all-reduce 聚合各 rank 的 aggregator，那么结果只代表 rank 0 处理的数据子集。这不会制造 `2^24` 饱和现象，但会影响最终统计代表性。

## 建议修复

### P0：修复所有 count / covered / total 累加器的数据类型

不要用 float32 累加千万级计数。建议：

- count / total / covered / nsr_count / nsr_total 使用 `torch.int64`
- sum_scale / sum_ar / sum_purity 使用 `torch.float64`
- finalize 时再转 Python `int` / `float`

重点变量包括：

```python
_dist_count
_dist_nsr_count
_dist_nsr_total
_cat_count
_cat_nsr_count
_cat_nsr_total
_dcov_covered
_dcov_total
```

以及这些 sum 变量建议使用 float64：

```python
_dist_sum_scale
_dist_sum_ar
_cat_sum_scale
_cat_sum_ar
_cat_sum_purity
_cat_purity_frames
```

### P0：保留并强制检查 coverage consistency

当前 JSON 中的 `coverage_debug` 很有价值。建议在 `finalize()` 中加入显式 warning 或 assert：

```python
abs(coverage_from_bins - coverage_from_scalar) < 1e-6
```

如果不满足，就不应该信任 distance-wise coverage。

### P1：增加 scale range sanity check

在 `finalize()` 中记录：

- `observed_min_scale`
- `observed_max_scale`
- `observed_max_s_hat`
- config/model 的 `scale_range`

如果某个 category/bin 的 `mean_scale > observed_max_s_hat + eps`，直接标记异常。这个检查能快速发现本次这类统计污染。

### P1：修复 `exclude_gaussian_classes` 参数传递

构造 aggregator 时应传入：

```python
exclude_gaussian_classes=args.exclude_gaussian_classes,
exclude_voxel_classes=args.exclude_voxel_classes,
```

如果还没有 `--exclude-voxel-classes` parser 参数，也建议补上，避免 Gaussian 过滤和 voxel 过滤口径混在一起。

## 如何验证修复是否成功

修复后重新跑同一配置，至少检查：

1. category/distance 的 count 不再大量精确等于 `16,777,216`。
2. 在 `scale_range = [0.08, 0.64]` 的 run 中，所有 category-wise 和 distance-wise `mean_scale <= 0.64 + eps`。
3. `coverage_debug.coverage_counter_abs_diff` 接近 `0`。
4. distance-wise coverage 的加权平均等于 global `Mean Coverage`。
5. distance-wise coverage 不再全部精确为 `1.0000`，除非 `scalar mean_coverage` 也确实接近 `1.0`。

## 最终判断

这次日志中不可信的主要是：

- category-wise 中 count 达到 `16,777,216` 的类别及其 mean scale / NSR
- distance-wise scale/shape 中 count 达到 `16,777,216` 的 bin
- distance-wise coverage 全部结果

相对可信的是：

- 全局 Mean Scale
- 全局 Mean AR
- 全局 Near-Spherical Ratio
- 全局 LIGR
- 全局 Mean Coverage
- 全局 Mean Purity(valid/penalized)
- Unused Gaussian Ratio

原因是全局指标大多使用 per-frame Python 标量累加或完整 tensor 汇总，没有走这些已经达到 float32 精度边界的 per-bin/per-class scatter count 累加器。
