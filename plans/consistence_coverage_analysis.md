# Coverage 统计方式一致性分析

## 用户期望的语义

距离 bins 划分 `[0, 10, 20, 30, 40, 50]` 对应的空间范围为：

| Bin | xy 平面范围（用户描述） | 对应区域 | z 范围 |
|:---:|:---|------|:---:|
| (0,10)  | `[-10, 10]` | 中心 20m × 20m 的正方形 | [-5, 3] |
| (10,20) | `[-20,-10]` & `[10,20]` | 内环 20-40m 的正方形环 | [-5, 3] |
| (20,30) | `[-30,-20]` & `[20,30]` | 中环 40-60m 的正方形环 | [-5, 3] |
| (30,40) | `[-40,-30]` & `[30,40]` | 外环 60-80m 的正方形环 | [-5, 3] |
| (40,50) | `[-50,-40]` & `[40,50]` | 最外环 80-100m 的正方形环 | [-5, 3] |

即：使用 **Chebyshev 距离** `d = max(|x|, |y|)` 将体素分配到不同的同心方环中。

Coverage 的定义：每个 bin 内，**被 Gaussian 至少一个 3D 椭球覆盖的 GT 非空网格数 / 该 bin 内所有 GT 非空网格数**。

---

## 当前代码实现 vs 用户预期

### 1. 距离计算方式 ✅ 一致

| 项目 | 当前代码 | 用户预期 |
|------|---------|---------|
| 距离公式 | [`aggregator.py:184`](gaussian_statistic/aggregator.py:184)<br>`torch.max(torch.abs(v[..., :2]), dim=-1).values` | `max(\|x\|, \|y\|)` |
| 几何含义 | 同心正方形环 | 同心正方形环 |
| bin 范围 | `(lo, hi]` via `bucketize` | `[lo, hi)` |

**验证**：体素 (8, 8) → `max(8, 8) = 8` → bin (0,10) ✅  
体素 (15, 3) → `max(15, 3) = 15` → bin (10,20) ✅

### 2. 体素筛选：occ_cam_mask ⚠️ 需要确认

当前代码使用 `occ_cam_mask` 筛选"GT 非空网格"：

```python
mask_flat = occ_cam_mask.reshape(-1).bool()
valid_xyz = occ_flat_xyz[mask_flat]
```

`occ_cam_mask` 在 [`LoadOccupancySurroundOcc`](dataset/transform_3d.py:623) 中的定义为：

```python
mask = new_label != 0
```

SurroundOcc 的 label 编码为：
| Label | 含义 |
|:---:|------|
| 0 | others/background（可能有语义但需要确认） |
| 1-16 | 实际语义类（barrier, car, ...） |
| 17 | empty/ignore（填充值） |

因此 `new_label != 0` 的 mask 排除了 label 0，但**包含了 label 17（empty）的体素**。如果 label 17 表示 free space（空），那么它们会被错误地计入分母，导致覆盖率系统性偏低。

**建议确认** `occ_cam_mask` 的语义。如果 label 17 确实表示 empty/free space，应将 mask 改为 `new_label != 17` 或 `(new_label > 0) & (new_label < 17)`。

### 3. 分桶统计逻辑 ✅ 一致

Distance-wise Coverage 的累加逻辑（[`aggregator.py:196-205`](gaussian_statistic/aggregator.py:196-205)）：

```python
bin_idx = torch.bucketize(valid_dist, bin_edges) - 1
valid_b = (bin_idx >= 0) & (bin_idx < n_bins)
bc = bin_idx.clamp(0, n_bins - 1)

self._dcov_total.scatter_add_(0, bc, ones_n * valid_b.float())
self._dcov_covered.scatter_add_(0, bc, covered.float() * valid_b.float())
```

- 每个体素根据 `max(|x|,|y|)` 分配到对应 bin
- `valid_b` 排除距离 < 0 和 ≥ 50 的体素（保持与用户定义一致）
- 最终 `coverage[bin_i] = dcov_covered[i] / dcov_total[i]`

**结果**：每个 bin 的 coverage = 该 ring 内被覆盖的体素数 / 该 ring 内所有体素数 ✅

### 4. 全局 Mean Coverage ✅ 一致（已修正）

[`aggregator.py:207-209`](gaussian_statistic/aggregator.py:207-209)：

```python
self.total_cov_covered += (covered & valid_b).sum().item()
self.total_cov_total += valid_b.sum().item()
```

- 仅统计距离分桶 `[0, 50)` 内的体素（与 distance-wise 口径一致）
- **mean_coverage = 所有 bin 中被覆盖体素总数 / 所有 bin 中体素总数**
- 等价于 distance-wise coverage 按体素数的加权平均 ✅

### 5. 几何覆盖判定 ✅ 一致

`compute_coverage_and_purity()`（[`calculator.py:124-170`](gaussian_statistic/calculator.py:124-170)）：

```python
dists = torch.cdist(chunk_v, means)                          # 体素到 Gaussian 中心的距离
cand_mask = dists <= search_radius[None, :]                   # τ·max(scale) 候选过滤
d_rot = torch.bmm(R[gj], diff.unsqueeze(-1)).squeeze(-1)     
mahal = (d_rot / scales_vec[gj].clamp(min=1e-8)).pow(2).sum(dim=-1)  # 马氏距离
hit = mahal <= tau_sq                                         # τ² 阈值判定
covered[start + hv] = True                                    # 标记被覆盖的体素
```

判定一个体素是否被 Gaussian "几何包裹住" 的逻辑：
1. 体素必须在 Gaussian 的 `τ·max(s)` 距离内（候选过滤）
2. 体素必须在 Gaussian 的 `τ` 马氏距离椭球内（精确判定）

这两个条件等价于：体素位于 Gaussian 的 τ-椭球内部。✅

---

## 总结

| 项目 | 状态 | 说明 |
|------|:----:|------|
| 距离度量（Chebyshev xy） | ✅ | `max(\|x\|, \|y\|)` 与用户期望一致 |
| 同心方环分桶 | ✅ | 各 bin 对应不同半径的方环区域 |
| GT 非空网格筛选 | ⚠️ | `occ_cam_mask = (label != 0)`，需确认 label 17（empty）是否被正确排除 |
| 被覆盖判定（椭球） | ✅ | 马氏距离 ≤ τ² 判定准确 |
| Per-bin coverage | ✅ | 被覆盖数 / 总非空网格数 |
| Mean coverage | ✅ | 全部分桶的加权平均，口径与 distance-wise 一致 |

**唯一待确认的点**是 `occ_cam_mask` 的定义是否正确地过滤了 empty/free space 体素。如果 `mask = new_label != 0` 包含了 label=17 的空体素，则应改为排除 label 17。
