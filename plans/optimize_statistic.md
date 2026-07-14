# Gaussian Statistic 统计代码优化修改清单

本文档汇总当前 `gaussian_statistic.py`、`aggregator.py`、`calculator.py`、`reporter.py` 四个文件中需要修改和优化的内容，目标是使统计代码与我们之前围绕 **Gaussian-based 3D Occupancy Prediction 中 allocation problem** 讨论的指标体系保持一致，并提高论文实验结果的可信度。

---

## 0. 总体结论

当前统计系统的整体结构是合理的，已经基本覆盖我们希望分析的三类指标：

1. **Primitive Shape Statistics**  
   用于分析 Gaussian primitive 的尺度、形状和各向异性，包括 `Mean Scale`、`Mean AR`、`NSR`、`LIGR`、`Scale Percentiles`、`Scale Volume`。

2. **Allocation Distribution Statistics**  
   用于分析 Gaussian primitive 在空间距离和语义类别上的分配是否合理，包括 `Distance-wise Scale/Shape` 和 `Category-wise Statistics`。

3. **Allocation Quality Statistics**  
   用于分析 Gaussian primitive 是否真正有效覆盖非空语义结构，以及覆盖区域是否语义一致，包括 `Coverage`、`Distance-wise Coverage`、`Purity`。

但是，目前代码中仍有几个关键统计口径与最终论文叙事不完全一致，尤其是：

- Coverage 当前只基于 `occ_cam_mask`，没有显式排除 `empty label`；
- Purity 当前将未覆盖任何体素的 unused Gaussian 记为 `1.0`，会虚高纯度；
- Distance-wise 统计在 `aggregator.py` 和 `calculator.py` 中距离定义不统一；
- 多卡统计只输出 rank 0 的局部统计，没有跨 rank 汇总；
- Reporter 输出中没有突出 `Mean AR`，也尚未支持更严谨的 purity 指标。

因此，建议在正式用于论文实验前，至少完成本文档中“必须修改”的部分。

---


---

## 0.1 新增异常现象诊断：Coverage 统计结果明显异常

根据当前补充观察，统计脚本还出现了两个新的异常现象：

```text
1. distance-wise coverage ratio 在每个距离 bin 中都等于 1.0，明显不合理；
2. mean coverage ratio 只有约 0.4，但这与可视化中观察到的 Gaussian 覆盖效果不一致。
```

这两个现象需要优先排查。尤其是第一个现象非常关键：在当前 `aggregator.py` 的实现中，distance-wise coverage 和 global mean coverage 理论上应该满足一个基本一致性约束：

```text
mean_coverage 应当等于各个 distance bin coverage 按 voxel 数量加权后的平均值。
```

也就是说，如果所有 distance bin 的 coverage 都是 1.0，那么全局 `mean_coverage` 也应该接近 1.0。现在出现 “distance-wise coverage 全为 1.0，但 mean coverage 只有 0.4” 的结果，说明至少存在以下问题之一：

```text
1. distance-wise coverage 的分子/分母累加逻辑存在 bug；
2. global mean coverage 与 distance-wise coverage 使用的 voxel 范围不一致；
3. 输出的 distance-wise coverage 不是当前 run 中 aggregator 的真实统计结果；
4. 某些 bin 的 dcov_total / dcov_covered 被错误累加或被覆盖；
5. coverage 可视化和 coverage 统计使用的 Gaussian 尺度阈值、坐标系或有效体素定义不一致。
```

### 0.1.1 必须加入 coverage 一致性自检

建议在 `aggregator.finalize()` 中加入如下自检，直接验证 distance-wise coverage 和 global coverage 是否来自同一套计数器：

```python
sum_dcov_covered = float(self._dcov_covered.sum().item())
sum_dcov_total = float(self._dcov_total.sum().item())
scalar_cov_covered = float(self.total_cov_covered)
scalar_cov_total = float(self.total_cov_total)

coverage_from_bins = sum_dcov_covered / sum_dcov_total if sum_dcov_total > 0 else 0.0
coverage_from_scalar = scalar_cov_covered / scalar_cov_total if scalar_cov_total > 0 else 0.0

result['coverage_debug'] = {
    'sum_dcov_covered': sum_dcov_covered,
    'sum_dcov_total': sum_dcov_total,
    'scalar_cov_covered': scalar_cov_covered,
    'scalar_cov_total': scalar_cov_total,
    'coverage_from_bins': coverage_from_bins,
    'coverage_from_scalar': coverage_from_scalar,
    'coverage_counter_abs_diff': abs(coverage_from_bins - coverage_from_scalar),
}
```

理论上，在当前代码口径下：

```text
sum_dcov_covered 应该等于 total_cov_covered；
sum_dcov_total 应该等于 total_cov_total；
coverage_from_bins 应该等于 mean_coverage。
```

如果这三个条件不成立，说明 coverage 累加器存在实现错误或统计过程中存在多次 finalize / 多卡 rank / 旧 JSON 结果混用的问题。

### 0.1.2 distance-wise coverage 全为 1.0 的可能原因

#### 原因 A：`cov_threshold` 设置过大

入口脚本中 `--cov-threshold` 默认是 `1.0`，但注释中也明确指出，过大的值如 `3.0` 可能导致 coverage 接近 1.0。若实际运行时使用了：

```bash
--cov-threshold 3.0
```

则 distance-wise coverage 很容易在每个 bin 内接近 1.0。因为 coverage 判定是：

```text
Mahalanobis distance <= tau^2
```

当 `tau=3.0` 时，Gaussian 覆盖区域相当于 3σ 椭球，而不是 1σ 椭球。对于 dense Gaussian prediction，这会显著放大每个 Gaussian 的有效覆盖半径。

**排查建议：** 在日志和 JSON 中显式记录：

```python
result['cov_threshold'] = self.cov_threshold
```

Reporter 中也打印：

```python
logger.info(f"  Coverage Threshold τ: {stats.get('cov_threshold', 'unknown')}")
```

#### 原因 B：distance-wise coverage 的分母可能被错误限制为“已覆盖体素”

从当前 `aggregator.py` 代码看，理论上 `_dcov_total` 使用的是所有 `valid_xyz` 中落入 bin 的体素数：

```python
self._dcov_total.scatter_add_(0, bc, ones_n * valid_b.float())
self._dcov_covered.scatter_add_(0, bc, covered.float() * valid_b.float())
```

这个逻辑本身是对的。但是如果后续改代码时不小心把 `_dcov_total` 改成了只统计 `covered` 的 voxel，就会导致：

```text
_dcov_covered == _dcov_total
```

从而每个 bin 的 coverage 都等于 1.0。

**排查建议：** 输出每个 bin 的 raw count：

```python
result['distancewise_coverage_debug'] = {
    f'({self.distance_bins[i]},{self.distance_bins[i+1]})': {
        'covered': float(dcov_covered[i]),
        'total': float(dcov_total[i]),
        'coverage': _s(dcov_covered[i], dcov_total[i]),
    }
    for i in range(n_bins)
}
```

如果每个 bin 都出现：

```text
covered == total
```

则要继续检查 `covered` 是否真的全为 True，还是 `_dcov_total` 被错误累加。

#### 原因 C：`covered` 在 `compute_coverage_and_purity()` 中被几乎全部置 True

如果 `covered.float().mean()` 在每帧都接近 1.0，那么 distance-wise coverage 全为 1.0 是计算逻辑的自然结果。此时需要排查：

```text
1. cov_threshold 是否过大；
2. scales 是否已经是物理尺度，还是被重复放大；
3. search_radius = tau * max(scale) 是否过大；
4. means / occ_xyz 是否在同一个坐标系；
5. voxel 坐标是否包含 homogeneous 维度或 reshape 错误。
```

建议在 `_accumulate_cov_purity()` 中临时打印或记录每帧 debug 信息：

```python
frame_cov = covered.float().mean().item()
num_valid_voxels = int(valid_xyz.shape[0])
num_covered_voxels = int(covered.sum().item())
mean_scale = float(scales.mean(dim=-1).mean().item())
max_scale = float(scales.max().item())

# 可选写入 logger 或 debug dict
```

#### 原因 D：distance-wise coverage 和 global coverage 来自不同 run 或旧 JSON

如果 reporter 读取的是旧的 `gaussian_statistic_result.json`，或者多次运行复用了同一目录下的结果文件，就可能出现主日志和 JSON 结果不一致。

**排查建议：** 在 JSON 中写入 run metadata：

```python
result['run_debug'] = {
    'num_frames': self.total_frames,
    'distance_bins': self.distance_bins,
    'cov_threshold': self.cov_threshold,
    'coverage_scope': self.coverage_scope,
}
```

并建议每次正式统计使用新的 `work_dir`。

### 0.1.3 mean coverage 只有约 0.4 但可视化中看起来覆盖充分的可能原因

这个现象本身不一定说明代码一定错，因为“可视化覆盖”和“统计覆盖”可能使用了不同定义。常见原因如下。

#### 原因 A：统计使用 1σ 椭球，而可视化可能展示 2σ / 3σ 椭球

当前入口脚本默认：

```python
--cov-threshold 1.0
```

这表示只有位于 1σ 椭球内部的 voxel center 才被视为 covered。可视化时如果展示的是更大的 Gaussian surface，例如 2σ 或 3σ 椭球，那么视觉上会觉得覆盖非常充分，但 1σ hard coverage 可能只有 0.4 左右。

**建议：** 正式统计时同时报告多个阈值：

```text
Coverage@1σ
Coverage@2σ
Coverage@3σ
```

这样可以区分：

```text
核心覆盖能力 vs 扩展覆盖能力
```

推荐新增参数：

```python
parser.add_argument('--cov-thresholds', type=float, nargs='+', default=[1.0, 2.0, 3.0])
```

或至少分别运行三次：

```bash
--cov-threshold 1.0
--cov-threshold 2.0
--cov-threshold 3.0
```

#### 原因 B：统计只检查 voxel center 是否落入椭球，而可视化看的是连续空间重叠

当前 coverage 判定是：

```text
voxel center inside Gaussian ellipsoid
```

如果一个 voxel 的中心点没有落入椭球，但 voxel cube 与椭球有交集，当前代码仍然认为该 voxel 未覆盖。因此在 voxel size = 0.5m 的情况下，统计 coverage 会比视觉上的连续空间覆盖更低。

**可选修正：** 如果希望统计更接近 voxel-level occupancy，可以加入 voxel 半径补偿：

```python
voxel_half_diag = (3 ** 0.5) * voxel_size / 2
search_radius = cov_threshold * scales.max(dim=-1).values + voxel_half_diag
```

或者在 Mahalanobis 判定中采用更宽松的阈值，如 `tau=1.5` 或 `tau=2.0`。

#### 原因 C：统计对象包含大量 empty / free voxels

当前代码如果只使用 `occ_cam_mask`，而 `occ_cam_mask` 不是 non-empty mask，则大量 empty/free voxels 会进入 coverage 统计。可视化时通常关注的是非空结构，例如 road、car、building、vegetation 等，因此视觉上看起来覆盖充分，但全空间 coverage 可能较低。

这也是本文档前面建议加入：

```python
valid_mask = occ_cam_mask & (occ_label != empty_label)
```

的原因。

#### 原因 D：坐标系或 scale 单位不一致

如果可视化中 Gaussian 与场景看起来重合，但统计 coverage 很低，需要重点确认：

```text
1. means 是否与 occ_xyz 处于同一坐标系；
2. use_ego=False / use_ego=True 是否一致；
3. occ_xyz 是否是 voxel center，而不是 grid index；
4. scales 是否已经经过网络中的激活和反归一化；
5. rotations 四元数顺序是否为 (w, x, y, z)；
6. 统计中的 covariance 定义是否与可视化中的 Gaussian 定义一致。
```

特别是 `occ_xyz` 在设计文档中曾标注为 `(200,200,16,4)`，包含 homogeneous 第 4 维；但当前 `aggregator.py` 中直接使用：

```python
occ_flat_xyz = occ_xyz.reshape(-1, 3)
```

如果实际 `occ_xyz` 最后一维是 4，必须先执行：

```python
occ_xyz = occ_xyz[..., :3]
```

否则要么直接报 shape 错误，要么在某些改写版本中发生坐标错位。

#### 原因 E：Gaussian rotation 的方向可能与模型/可视化定义相反

当前 fast Mahalanobis 使用：

```python
d_rot = R @ diff
mahal = sum((d_rot / scales) ** 2)
```

这与当前 `_build_cov_inv()` 内部是自洽的，但未必与模型或可视化中的 Gaussian covariance 方向一致。如果真实定义是：

```text
Cov = R S^2 R^T
```

那么局部坐标变换可能应为：

```python
d_rot = R.transpose(-1, -2) @ diff
```

**建议加入单元测试：** 对同一批 Gaussian 和 voxel，同时计算：

```python
mahal_fast = sum((R @ diff / s) ** 2)
mahal_cov = diff^T CovInv diff
```

并额外与可视化使用的 covariance 构造方式对齐。如果可视化和统计使用的旋转方向不同，会出现视觉重合但统计 coverage 偏低的问题。

### 0.1.4 推荐的 coverage debug 输出

建议在 `finalize()` 中额外输出以下字段，直到确认 coverage 统计稳定后再决定是否保留：

```json
{
  "coverage_debug": {
    "cov_threshold": 1.0,
    "coverage_scope": "non_empty_visible",
    "sum_dcov_covered": 123456,
    "sum_dcov_total": 234567,
    "scalar_cov_covered": 123456,
    "scalar_cov_total": 234567,
    "coverage_from_bins": 0.5263,
    "coverage_from_scalar": 0.5263,
    "coverage_counter_abs_diff": 0.0
  },
  "distancewise_coverage_debug": {
    "(0,10)": {"covered": 1000, "total": 1200, "coverage": 0.8333},
    "(10,20)": {"covered": 2000, "total": 3500, "coverage": 0.5714}
  }
}
```

如果出现以下情况：

```text
coverage_from_bins != coverage_from_scalar
```

则说明 global coverage 和 distance-wise coverage 的累加器不一致，必须先修复统计代码，再讨论可视化差异。

如果出现以下情况：

```text
coverage_from_bins == coverage_from_scalar，但二者都明显低于可视化观察
```

则应重点排查 coverage 定义差异，包括 `cov_threshold`、empty voxel、voxel center 判定、坐标系和 rotation convention。

---

# 1. 必须修改项

## 1.1 Coverage / Purity 应显式排除 empty voxels

### 当前问题

在 `aggregator.py` 和 `calculator.py` 中，目前 coverage / purity 的有效体素主要由：

```python
mask_flat = occ_cam_mask.reshape(-1).bool()
valid_xyz = occ_flat_xyz[mask_flat]
```

决定。

这意味着只要 `occ_cam_mask=True`，该 voxel 就会被纳入 coverage / purity 统计。但是，如果 `occ_cam_mask` 只是表示 camera-visible 区域，而不是 visible non-empty 区域，那么大量 empty/free voxels 也会进入统计。

这会导致 coverage 的含义从：

```text
Gaussian 是否覆盖真实非空语义结构
```

变成：

```text
Gaussian 是否覆盖可见空间区域
```

这与我们论文中希望分析的 **semantic Gaussian allocation quality** 不完全一致。

### 建议修改

在 `GaussianStatAggregator.__init__()` 中增加：

```python
def __init__(..., empty_label=17, ignore_empty=True):
    self.empty_label = empty_label
    self.ignore_empty = ignore_empty
```

在 `_accumulate_cov_purity()` 中修改有效体素 mask：

```python
occ_flat_xyz = occ_xyz.reshape(-1, 3)
mask_flat = occ_cam_mask.reshape(-1).bool()

if occ_label is not None:
    label_flat = occ_label.reshape(-1).long()
    if self.ignore_empty:
        mask_flat = mask_flat & (label_flat != self.empty_label)
else:
    label_flat = None

valid_xyz = occ_flat_xyz[mask_flat]
if valid_xyz.shape[0] == 0:
    return

if label_flat is not None:
    valid_label = label_flat[mask_flat]
else:
    valid_label = torch.zeros(valid_xyz.shape[0], dtype=torch.long, device=device)
```

### 论文口径建议

最终论文中建议将主 coverage 指标命名为：

```text
Non-empty Visible Coverage
```

而不是笼统写作 `Coverage`。

---

## 1.2 Purity 不应将 unused Gaussian 记为 1.0

### 当前问题

当前 `aggregator.py` 中的全局 purity 计算为：

```python
p_g_purity = torch.where(
    purity_total > 0,
    purity_match.float() / purity_total.float().clamp(min=1),
    torch.ones_like(purity_match, dtype=torch.float32)
)
self.total_purity_sum += p_g_purity.sum().item()
```

当前 `calculator.py` 中的 `compute_category_stats()` 也有类似逻辑：

```python
ppg[g] = 1.0 if t == 0 else pm[g].item() / t
```

这意味着没有覆盖任何有效体素的 Gaussian 会被视为 purity=1.0。

这会造成两个问题：

1. 如果某个方法产生大量 unused Gaussian，mean purity 可能被虚高；
2. Purity 无法真实反映 semantic allocation quality。

从 allocation problem 角度看，未覆盖任何有效体素的 Gaussian 应该被视为 **unused / redundant primitive**，不应被奖励为完美纯度。

### 建议修改

在 `GaussianStatAggregator` 中新增累加器：

```python
self.total_valid_purity_sum = 0.0
self.total_valid_purity_count = 0
self.total_penalized_purity_sum = 0.0
self.total_unused_gaussians = 0
self.total_purity_gaussians = 0
```

替换当前 purity 累加逻辑：

```python
valid_g = purity_total > 0
self.total_purity_gaussians += purity_total.numel()
self.total_unused_gaussians += (~valid_g).sum().item()

if valid_g.any():
    valid_purity = purity_match[valid_g].float() / purity_total[valid_g].float().clamp(min=1)
    self.total_valid_purity_sum += valid_purity.sum().item()
    self.total_valid_purity_count += valid_g.sum().item()

penalized_purity = torch.zeros_like(purity_match, dtype=torch.float32)
penalized_purity[valid_g] = purity_match[valid_g].float() / purity_total[valid_g].float().clamp(min=1)
self.total_penalized_purity_sum += penalized_purity.sum().item()
```

在 `finalize()` 中输出：

```python
'mean_purity_valid': _s(self.total_valid_purity_sum, self.total_valid_purity_count),
'mean_purity_penalized': _s(self.total_penalized_purity_sum, self.total_purity_gaussians),
'unused_gaussian_ratio': _s(self.total_unused_gaussians, self.total_purity_gaussians),
```

### 论文口径建议

论文主文建议使用：

```text
Mean Semantic Purity over Covered Gaussians
Unused Gaussian Ratio
```

可以额外报告：

```text
Penalized Mean Purity
```

用于惩罚 unused Gaussians。

---

## 1.3 统一 Distance-wise 统计的距离定义

### 当前问题

`aggregator.py` 中 distance-wise 统计使用 BEV Chebyshev distance：

```python
dist = torch.max(torch.abs(means[..., :2]), dim=-1).values
```

但是 `calculator.py` 中的 `compute_distance_stats()` 使用的是三维欧氏距离：

```python
dist = torch.norm(means, dim=-1)
```

这会导致不同模块中 distance-wise scale/shape 的统计口径不一致。

我们之前讨论的论文口径更适合使用 BEV 方环距离：

```text
max(|x|, |y|)
```

因为 occupancy 预测通常在固定 BEV 范围内分析，如 `[0,10), [10,20), ...`。

### 建议修改

将 `calculator.py` 中的：

```python
dist = torch.norm(means, dim=-1)
```

改为：

```python
dist = torch.max(torch.abs(means[..., :2]), dim=-1).values
```

并在函数注释中明确：

```text
Distance is measured by Chebyshev distance in BEV, i.e., max(|x|, |y|).
```

---

## 1.4 多卡统计需要跨 rank 汇总，或明确只支持单卡完整统计

### 当前问题

`gaussian_statistic.py` 中多卡模式下每个 rank 都会创建独立的 `GaussianStatAggregator`。最终代码只执行：

```python
if distributed:
    dist.barrier()

if local_rank == 0:
    stats = aggregator.finalize()
    report_statistics(stats, logger, args.work_dir)
```

这意味着多卡运行时最终只输出 rank 0 处理到的验证子集结果，而不是完整验证集统计结果。

### 影响

多卡模式下：

```text
num_frames 只等于 rank 0 的帧数；
num_gaussians 只等于 rank 0 的 Gaussian 数量；
mean_scale / mean_ar / coverage / purity 都只基于 rank 0 子集；
distance-wise / category-wise 统计也只基于 rank 0 子集。
```

这会影响论文统计结果的可信度。

### 建议修改方案 A：强制单卡统计

如果统计脚本主要用于论文分析，最简单可靠的方式是明确只使用单卡完整验证集运行：

```text
建议使用 CUDA_VISIBLE_DEVICES=0 python gaussian_statistic.py ...
```

并在 README 或脚本日志中提示：

```text
For faithful full-validation statistics, please run this script on a single GPU unless distributed aggregation is enabled.
```

### 建议修改方案 B：实现 distributed all_reduce

如果需要多卡统计，则需要在 `GaussianStatAggregator` 中新增 `sync_distributed()` 方法，对以下累加器执行 `dist.all_reduce`：

```text
total_frames
total_gaussians
sum_mean_scale
sum_nsr
sum_ligr
vol_sum
vol_count
total_cov_covered
total_cov_total
purity 相关累加器
_dist_count
_dist_sum_scale
_dist_sum_ar
_dist_nsr_count
_dist_nsr_total
_cat_count
_cat_sum_scale
_cat_sum_ar
_cat_nsr_count
_cat_nsr_total
_cat_sum_purity
_cat_purity_frames
_dcov_covered
_dcov_total
```

注意：如果继续保存 `_acc_s_hat`、`_acc_ars`、`_acc_volumes` 用于 percentile，多卡下这些 list 也需要 gather 或改成 histogram 近似统计。

---

## 1.5 `exclude_classes` 的语义需要拆分或明确

### 当前问题

当前 `exclude_classes` 只过滤 predicted Gaussian class：

```python
keep = ~torch.isin(pred_class, self._exclude_classes_tensor)
means = means[keep]
scales = scales[keep]
rotations = rotations[keep]
pred_class = pred_class[keep]
```

但是它不会过滤 GT voxel label。

这意味着：

```text
exclude_classes=[0]
```

当前含义是：

```text
不让 class 0 的 Gaussian 参与 coverage/purity。
```

但 class 0 的 GT voxels 仍然会参与统计。

这个语义容易混淆。

### 建议修改

将参数拆成两个：

```python
exclude_gaussian_classes
exclude_voxel_classes
```

分别表示：

```text
exclude_gaussian_classes: 排除某些预测类别的 Gaussian primitive；
exclude_voxel_classes: 排除某些 GT 语义类别的 voxel。
```

如果只是为了排除 empty/free voxels，不建议使用 `exclude_classes`，而应使用：

```python
empty_label=17
ignore_empty=True
```

---

# 2. 建议修改项

## 2.1 入口脚本增加显式统计口径参数

建议在 `gaussian_statistic.py` 中新增参数：

```python
parser.add_argument('--num-classes', type=int, default=None,
                    help='语义类别数；None 表示从模型或 config 自动推断')
parser.add_argument('--empty-label', type=int, default=17,
                    help='GT occupancy 中 empty/free 类别标签，默认 17')
parser.add_argument('--ignore-empty', action='store_true', default=True,
                    help='Coverage/Purity 是否排除 empty voxels')
parser.add_argument('--coverage-scope', type=str, default='non_empty_visible',
                    choices=['visible', 'non_empty_visible'],
                    help='Coverage/Purity 的 voxel 统计范围')
```

初始化 aggregator 时传入：

```python
aggregator = GaussianStatAggregator(
    ...,
    empty_label=args.empty_label,
    ignore_empty=args.ignore_empty,
    coverage_scope=args.coverage_scope,
)
```

---

## 2.2 `num_classes` 推断需要更稳健

当前入口脚本默认：

```python
num_classes = 17
```

并优先从 `raw_model.head.num_classes` 或 `cfg.model.head.num_classes` 推断。

建议增加 `--num-classes` 参数，并优先使用用户指定值：

```python
if args.num_classes is not None:
    num_classes = args.num_classes
elif hasattr(raw_model, 'head') and hasattr(raw_model.head, 'num_classes'):
    num_classes = raw_model.head.num_classes
elif hasattr(cfg.model, 'head') and 'num_classes' in cfg.model.head:
    num_classes = cfg.model.head.num_classes
else:
    num_classes = 17
```

同时建议在日志中打印：

```python
logger.info(f'num_classes for Gaussian statistics: {num_classes}')
logger.info(f'empty_label for occupancy statistics: {args.empty_label}')
```

---

## 2.3 Category-wise Statistics 增加 Gaussian 分配比例

### 当前输出

当前 category-wise 表格包括：

```text
Class | Count | MeanScale | MeanAR | NSR | Purity
```

### 建议新增

增加：

```text
Ratio
```

计算方式：

```python
category_ratio = cat_count[c] / total_gaussians
```

输出改为：

```text
Class | Count | Ratio | MeanScale | MeanAR | NSR | Purity
```

这个指标可以更直接地支撑论文中关于：

```text
primitive budget 是否根据类别复杂度进行合理分配
```

的分析。

---

## 2.4 Reporter 主摘要区应突出 Mean AR

当前 `reporter.py` 的主摘要区打印：

```text
Mean Scale
Near-Spherical Ratio
LIGR
Mean Coverage
Mean Purity
```

建议加入：

```text
Mean AR
```

推荐主摘要区改为：

```text
Mean Scale
Mean AR
Near-Spherical Ratio
LIGR
Mean Coverage
Mean Purity(valid)
Mean Purity(penalized)
Unused Gaussian Ratio
```

其中 `Mean Scale + Mean AR` 是论文中分析 primitive shape allocation 的核心证据。

---

## 2.5 Reporter 支持新版 purity 字段

如果 aggregator 按照建议输出：

```python
mean_purity_valid
mean_purity_penalized
unused_gaussian_ratio
```

则 reporter 应兼容打印：

```python
if 'mean_purity_valid' in stats:
    logger.info(f"  Mean Purity(valid):     {stats['mean_purity_valid']:.6f}")
if 'mean_purity_penalized' in stats:
    logger.info(f"  Mean Purity(penalized): {stats['mean_purity_penalized']:.6f}")
if 'unused_gaussian_ratio' in stats:
    logger.info(f"  Unused Gaussian Ratio:  {stats['unused_gaussian_ratio']:.6f}")
else:
    logger.info(f"  Mean Purity:            {stats['mean_purity']:.6f}")
```

---

## 2.6 Coverage 输出中明确统计范围

建议在 `stats` 中增加：

```python
'coverage_voxel_scope': 'non_empty_visible'
```

Reporter 中打印：

```python
logger.info(f"  Coverage Scope:        {stats.get('coverage_voxel_scope', 'unknown')}")
```

Distance-wise coverage 标题建议改为：

```text
[Distance-wise Non-empty Visible Coverage]
```

或者根据 `coverage_voxel_scope` 动态决定标题。

---

# 3. 可选优化项

## 3.1 Percentile 统计不建议长期保存在 GPU list 中

当前 `aggregator.py` 中为了计算 percentile，会保存所有帧的：

```python
self._acc_s_hat.append(s_hat.detach())
self._acc_ars.append(ar.detach())
self._acc_volumes.append(vol.detach())
```

如果验证集约 6000 帧，每帧 25600 个 Gaussian，总数约 154M。三个 float tensor 合计约 1.85GB，如果都留在 GPU 上，会增加显存压力。

### 方案 A：转移到 CPU

```python
self._acc_s_hat.append(s_hat.detach().cpu())
self._acc_ars.append(ar.detach().cpu())
self._acc_volumes.append(vol.detach().cpu())
```

### 方案 B：每帧采样

```python
max_sample_per_frame = 4096
idx = torch.randperm(G, device=device)[:max_sample_per_frame]
self._acc_s_hat.append(s_hat[idx].detach().cpu())
self._acc_ars.append(ar[idx].detach().cpu())
self._acc_volumes.append(vol[idx].detach().cpu())
```

### 方案 C：使用 histogram 近似 percentile

适合多卡统计或极大验证集，但实现更复杂。

---

## 3.2 `compute_category_stats()` 中 Python for-loop 应向量化

当前 `calculator.py` 中：

```python
for g in range(G):
    t = pt[g].item()
    ppg[g] = 1.0 if t == 0 else pm[g].item() / t
```

存在大量 `.item()` 同步，效率较低。

建议改为：

```python
valid_g = pt > 0
ppg = torch.zeros(G, dtype=torch.float32, device=scales.device)
ppg[valid_g] = pm[valid_g].float() / pt[valid_g].float().clamp(min=1)
```

如果需要 unused ratio：

```python
unused_ratio = (~valid_g).float().mean().item()
```

---

## 3.3 `scatter_add_` 前建议检查 pred_class 范围

当前 category-wise scatter 直接使用：

```python
self._cat_count.scatter_add_(0, pred_class, ones_g)
```

如果 `pred_class.max() >= num_classes`，会报错。

建议加入保护：

```python
valid_cls = (pred_class >= 0) & (pred_class < self.num_classes)
pc = pred_class[valid_cls]

self._cat_count.scatter_add_(0, pc, ones_g[valid_cls])
self._cat_sum_scale.scatter_add_(0, pc, s_hat[valid_cls])
self._cat_sum_ar.scatter_add_(0, pc, ar[valid_cls])
self._cat_nsr_count.scatter_add_(0, pc, nsr_mask[valid_cls])
self._cat_nsr_total.scatter_add_(0, pc, ones_g[valid_cls])
```

---

## 3.4 `compute_distancewise_coverage()` 中 precomp 与过滤后 Gaussian 可能不匹配

当前 `calculator.py` 中，如果 `exclude_classes` 过滤了 Gaussian，但调用者传入了过滤前的 `precomp`，则可能出现 shape 不匹配或索引错位。

建议在过滤后同步过滤 precomp：

```python
precomp = {
    'cov_inv': precomp['cov_inv'][keep],
    'R': precomp['R'][keep],
    'scales_vec': precomp['scales_vec'][keep],
    'search_radius': precomp['search_radius'][keep],
    'pred_class': precomp['pred_class'][keep],
}
```

或者在过滤后重新计算 precomp。

---

## 3.5 Reporter 增强字段兼容性

当前 `reporter.py` 假设所有字段都存在。如果未来 stats 字段发生变化，可能出现 `KeyError`。

建议使用兼容式写法：

```python
if 'scale_percentiles' in stats:
    ...
if 'category_stats' in stats:
    ...
```

对于 category key，建议兼容 int / str：

```python
for c, v in sorted(stats['category_stats'].items(), key=lambda x: int(x[0])):
    c_int = int(c)
```

---

## 3.6 JSON 写入建议指定 encoding

当前：

```python
with open(json_path, 'w') as f:
    json.dump(serializable, f, indent=2, ensure_ascii=False)
```

建议改为：

```python
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(serializable, f, indent=2, ensure_ascii=False)
```

---

## 3.7 入口脚本中若 config 无 `load_from` / `print_freq` 可能不稳

当前：

```python
elif cfg.load_from:
```

建议改为：

```python
elif cfg.get('load_from', None):
```

当前：

```python
print_freq = cfg.print_freq
```

但 `print_freq` 后续未使用，建议删除，避免 config 中没有该字段时报错。

---

# 4. 建议修改后的指标输出结构

最终建议 `gaussian_statistic_result.json` 至少包含以下字段：

```json
{
  "num_frames": 6019,
  "num_gaussians": 154086400,
  "coverage_voxel_scope": "non_empty_visible",

  "mean_scale": 0.34,
  "anisotropy_ratio": {
    "mean_ar": 1.34,
    "median_ar": 1.20,
    "p75_ar": 1.50,
    "p90_ar": 2.10
  },
  "near_spherical_ratio": 0.66,
  "ligr": 0.0,

  "mean_coverage": 0.52,
  "mean_purity_valid": 0.XX,
  "mean_purity_penalized": 0.XX,
  "unused_gaussian_ratio": 0.XX,

  "scale_percentiles": {
    "P50": 0.33,
    "P75": 0.35,
    "P90": 0.40,
    "P95": 0.45
  },

  "category_stats": {
    "0": {
      "count": 2849,
      "ratio": 0.018,
      "mean_scale": 0.33,
      "mean_ar": 1.28,
      "near_spherical_ratio": 0.72,
      "mean_purity_valid": 0.XX,
      "unused_gaussian_ratio": 0.XX
    }
  },

  "distance_stats": {
    "(0,10)": {
      "count": 4668309,
      "mean_scale": 0.34,
      "mean_ar": 1.20,
      "near_spherical_ratio": 0.80
    }
  },

  "distancewise_coverage": {
    "(0,10)": 0.85,
    "(10,20)": 0.72
  }
}
```

---

# 5. 按文件划分的修改建议

## 5.1 `gaussian_statistic.py`

### 必须修改

- 多卡统计需要实现 all_reduce，或明确只支持单卡完整统计。
- 增加 `--empty-label`、`--ignore-empty`、`--num-classes` 等显式参数。

### 建议修改

- `cfg.load_from` 改为 `cfg.get('load_from', None)`。
- 删除未使用的 `print_freq = cfg.print_freq`。
- 在日志中打印 `num_classes`、`empty_label`、`coverage_scope`。
- 明确 `exclude_classes` 的含义，或拆成 `exclude_gaussian_classes` / `exclude_voxel_classes`。

---

## 5.2 `aggregator.py`

### 必须修改

- 增加 coverage 一致性自检：`sum(_dcov_covered) / sum(_dcov_total)` 必须等于 `total_cov_covered / total_cov_total`。
- 在 JSON 中输出 `coverage_debug` 和 `distancewise_coverage_debug`，用于定位 distance-wise coverage 全为 1.0 的问题。
- Coverage/Purity 使用 `occ_cam_mask & occ_label != empty_label`。
- Purity 不再将 unused Gaussian 记为 1.0。
- 增加 `unused_gaussian_ratio`、`mean_purity_valid`、`mean_purity_penalized`。
- 如果使用 `exclude_classes`，需要修正 purity 分母，不能继续用 `total_gaussians`。

### 建议修改

- Category-wise stats 增加 `ratio`。
- Percentile tensor 可转 CPU 或采样，降低显存压力。
- 对 `pred_class` 范围做保护。
- 为多卡统计预留 `sync_distributed()`。

---

## 5.3 `calculator.py`

### 必须修改

- 确认 `cov_threshold` 与可视化使用的 Gaussian 半径一致，建议报告 Coverage@1σ / Coverage@2σ / Coverage@3σ。
- `compute_distance_stats()` 使用 BEV Chebyshev distance，而不是 3D Euclidean distance。
- Coverage/Purity 相关函数加入 empty voxel 过滤。
- `compute_category_stats()` 不再将 unused Gaussian purity 记为 1.0。

### 建议修改

- 将 `compute_category_stats()` 中逐 Gaussian for-loop 向量化。
- 修复 `compute_distancewise_coverage()` 中 precomp 与过滤后 Gaussian 不匹配的问题。
- 增加 `build_valid_voxel_mask()` 辅助函数，统一 valid voxel 口径。
- 确认旋转矩阵方向是否与模型中 Gaussian covariance 定义一致。

---

## 5.4 `reporter.py`

### 必须修改

- 主摘要区加入 `Mean AR`。
- 打印 `Coverage Threshold τ`、`Coverage Scope` 和 `coverage_debug` 中的 counter consistency 信息。
- 支持 `mean_purity_valid`、`mean_purity_penalized`、`unused_gaussian_ratio`。
- Coverage 输出中明确 `coverage_voxel_scope`。

### 建议修改

- Category-wise 表格加入 `Ratio`。
- Category-wise purity 字段命名为 `PurityValid` 或 `Purity(FrameAvg)`，避免误解。
- JSON 写入加 `encoding='utf-8'`。
- 对缺失字段增加兼容处理。

---

# 6. 推荐修改优先级

## P0：必须先改，否则论文统计口径可能不可信

1. 修复 coverage 统计异常：加入 `coverage_debug`，确认 `mean_coverage` 与 distance-wise coverage 的加权平均一致；
2. 检查 `distancewise_coverage` 全为 1.0 的原因，重点排查 `cov_threshold`、`_dcov_total` 分母、旧 JSON 混用和 `covered` 是否全 True；
3. Coverage/Purity 排除 empty voxels；
4. Purity 不再将 unused Gaussian 记为 1.0；
5. 增加 `unused_gaussian_ratio`；
6. 统一 distance-wise 距离定义；
7. 多卡统计问题：要么单卡运行，要么实现 all_reduce。

## P1：强烈建议修改，能增强论文表达

1. Reporter 主摘要区加入 Mean AR；
2. Category-wise stats 增加 Gaussian ratio；
3. Coverage 输出明确是 `non_empty_visible`；
4. 入口参数显式加入 `empty_label` 和 `coverage_scope`；
5. `exclude_classes` 拆分为 Gaussian 过滤和 voxel 过滤。

## P2：工程优化，可后续逐步完善

1. Percentile tensor 转 CPU 或采样；
2. `compute_category_stats()` 向量化；
3. Reporter 增强字段兼容；
4. JSON 写入指定 encoding；
5. DDP 日志多 rank 写同一文件的问题；
6. `cfg.load_from` 和 `cfg.print_freq` 的鲁棒性修正。

---

# 7. 最终建议

当前统计代码已经具备完整的分析框架，可以作为 Gaussian allocation problem 论文实验分析的基础。后续最关键的不是增加更多指标，而是统一统计口径，避免 coverage 和 purity 的定义被审稿人质疑。

建议最终论文中主打以下指标组合：

```text
Mean Scale + Mean AR
→ 证明 primitive shape adaptation 不足。

Distance-wise Scale/Shape + Distance-wise Coverage
→ 证明不同空间区域 allocation 不均衡。

Category-wise Count/Ratio + Category-wise Scale/AR/Purity
→ 证明不同语义类别 allocation 不合理。

Non-empty Visible Coverage + Valid Purity + Unused Gaussian Ratio
→ 证明现有 Gaussian allocation 同时存在 under-coverage、semantic impurity 和 primitive redundancy。
```

完成本文档中的 P0 和 P1 修改后，该统计系统将更适合作为论文中 “发现问题 → 指标分析 → 方法设计动机” 的实验支撑。
