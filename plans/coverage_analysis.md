# Coverage 统计结果分析

## 问题描述

### 问题 1：原始 Distance-wise Coverage 全为 1.0

在运行 `gaussian_statistic.py` 对验证集进行统计时，【Distance-wise Coverage】在所有距离分桶下的输出均为 **1.0000**（即 100%），无论距离远近。从日志 [`out/img_only_statistic/20260609_152217.log`](out/img_only_statistic/20260609_152217.log:2432) 可以看到：

```
  [Distance-wise Coverage]
           (0,10): 1.0000
          (10,20): 1.0000
          (20,30): 1.0000
          (30,40): 1.0000
          (40,50): 1.0000
```

### 问题 2：修改后 Mean Coverage = 0.1 与可视化不符

将 `--cov-threshold` 从 3.0 改为 1.0（新默认值）后，全局 Mean Coverage 仅约 0.1，与渲染可视化结果不一致。

### 问题 3：Distance-wise Coverage 仍有差异

即使在修改后，Distance-wise Coverage 各分桶之间的结果与预期仍有差距。

---

## 代码追踪

### 1. Coverage 计算流程

核心调用链如下：

1. [`aggregator.py`](gaussian_statistic/aggregator.py:84) `add_frame()` → 每帧收集 Gaussian 参数与 GT occupancy
2. [`aggregator.py`](gaussian_statistic/aggregator.py:149) `_accumulate_cov_purity()` → 提取可见体素并计算 coverage
3. [`calculator.py`](gaussian_statistic/calculator.py:124) `compute_coverage_and_purity()` → 体素分块 + 候选过滤 + 马氏距离判定
4. [`calculator.py`](gaussian_statistic/calculator.py:52) `precompute_frame_data()` → 预计算协方差逆、搜索半径等

### 2. 关键逻辑

在 `compute_coverage_and_purity()` 中，对一个体素点判断是否"被覆盖"的逻辑为：

```python
dists = torch.cdist(chunk_v, means)                    # 体素到所有 Gaussian 中心的距离
cand_mask = dists <= search_radius[None, :]             # 候选过滤：距离 ≤ 搜索半径
# ... 对候选取出后计算马氏距离 ...
mahal = (d_rot / scales_vec[gj])^2  # 快速马氏距离
hit = mahal <= tau_sq                                   # τ²
```

其中搜索半径定义为（见 [`calculator.py`](gaussian_statistic/calculator.py:65)）：

```python
search_radius = cov_threshold * scales.max(dim=-1).values  # τ * max(s_x, s_y, s_z)
```

---

## 根因 1：Empty Gaussians 并不在模型输出中

通过追踪模型输出流程可以确认：

- 模型 forward 调用链：`BEVSegmentor.forward()` → `GaussianOccEncoder.forward()` → `GaussianHead.forward()`
- Encoder 输出 `representation[-1]['gaussian']` 是 `GaussianPrediction` NamedTuple，包含 `means`、`scales`、`rotations`、`opacities`、`semantics`
- Head 中的 `prepare_gaussian_args()` 虽然会 **在渲染时** 添加 empty Gaussians（`scale=[100,100,8]`），但其仅用于局部聚合渲染，**不影响返回的 `gaussian` 对象**
- 返回的 `gaussian` 来自 `return {'gaussian': representation[-1]['gaussian']}`，是原始 encoder 输出

因此 **empty Gaussians 并非原因**。详见 [`gaussian_head.py`](model/head/gaussian_head.py:200-202) 和 [`gaussian_head.py`](model/head/gaussian_head.py:95-107)。

---

## 根因 2：cov_threshold 对覆盖率的敏感度分析（核心）

### 2.1 椭球体积 vs 体素大小

体素网格：200 × 200 × 16 = 640,000 个体素，分辨率 0.5m，每个体素体积 `0.125 m³`。

Gaussian scale 范围 [0.08, 0.64]，均值 ≈ 0.34。

对于不同 `τ` 值，单个 Gaussian 的椭球覆盖体积和体素覆盖数：

| τ | 椭球半径 (max) | 椭球体积 | 覆盖体素数 | 25,600 个 Gaussian 覆盖的体素数 |
|:---:|:---:|:---:|:---:|:---:|
| 1.0 | ~0.34 m | **0.074 m³** | **~0.6** | **~15,360 (2.4%)** |
| 1.5 | ~0.51 m | 0.25 m³ | ~2.0 | ~51,200 (8%) |
| 2.0 | ~0.68 m | **0.59 m³** | **~4.7** | **~120,320 (18.8%)** |
| 2.5 | ~0.85 m | 1.16 m³ | ~9.3 | ~238,080 (37.2%) |
| 3.0 | ~1.02 m | **2.0 m³** | **~16** | **~409,600 (64%)** |

> 注意：τ=1.0 时，单个 1σ 椭球的体积（0.074 m³）**小于一个体素**（0.125 m³）。

### 2.2 不同 τ 下的覆盖率估计

场景中可见体素（`occ_cam_mask` = True）大约占总体的 20-40%，即 128,000-256,000 个体素。

| τ | 总覆盖体素数 | 可见体素覆盖率估计 | 预期 Mean Coverage |
|:---:|:---:|:---:|:---:|
| 1.0 | ~15,360 | 6-12% | **~0.1** ✅ |
| 2.0 | ~120,320 | 47-94% | ~0.5-0.9 |
| 3.0 | ~409,600 | 160-320% | **1.0** ✅ |

**结论：Mean Coverage = 0.1 对于 τ=1.0 是完全正确的结果。** τ=1.0 的椭球太小，每个 Gaussian 连一个完整的体素都覆盖不了。

### 2.3 可视化 vs 硬几何覆盖率的差异

Gaussian 渲染可视化使用的是 **opacity alpha 合成**，每个 Gaussian 对所有体素都有贡献（即使距离很远，通过 opacity 衰减），这与硬几何椭球覆盖（严格二值判定）有本质区别：

| 度量方式 | 判定标准 | 特点 |
|---------|---------|------|
| **渲染可视化** | Alpha 合成，soft 贡献 | Gaussian 远距离也有贡献，视觉上"覆盖整个场景" |
| **硬几何覆盖** | 体素中心是否在椭球内 | 二值化判定，τ 越小越严格 |

所以 **Mean Coverage = 0.1 与可视化不一致是正常的**，因为二者的定义完全不同。

### 2.4 τ 的物理意义

| τ | 概率质量 | 语义含义 |
|:---:|:-------:|---------|
| 1.0 | 68.3% | Gaussian 核心区域，过于严格 |
| **2.0** | **95.4%** | **Gaussian 主体区域，与实际语义范围接近** |
| 3.0 | 99.7% | Gaussian 几乎全部概率质量，过于宽松 |

**建议默认值设为 τ=2.0**，这既给出了有区分度的覆盖率（不会全为 1.0），又与实际渲染效果更接近。

---

## 根因 3：全局 Mean Coverage 与 Distance-wise Coverage 的统计口径不一致

### 3.1 代码对比

Distance-wise Coverage（[`aggregator.py`](gaussian_statistic/aggregator.py:202-203)）：
```python
self._dcov_total.scatter_add_(0, bc, ones_n * valid_b.float())
self._dcov_covered.scatter_add_(0, bc, covered.float() * valid_b.float())
```
> `valid_b` 过滤掉了距离 > 50m 和距离 < 0 的体素。只有体素到原点的距离在 `[0, 50)` 范围内才被计入。

全局 Mean Coverage（[`aggregator.py`](gaussian_statistic/aggregator.py:206-207)）：
```python
self.total_cov_covered += covered.sum().item()
self.total_cov_total += valid_xyz.shape[0]
```
> **所有**可见体素（`occ_cam_mask=True`）都被计入，无论距离多远。

### 3.2 影响

场景范围是 `[-50, 50] × [-50, 50] × [-5, 3]`（各向 100m）。体素到原点的最大距离为 `√(50² + 50²) ≈ 70.7m`。

**距离 > 50m 的体素占总体的比例估算：**
- 总格子数：200 × 200 × 16 = 640,000
- 50m 半径内格子数：π × 50² × 8 / 0.125 ≈ 502,654
- 50m 半径外格子数：640,000 - 502,654 = **137,346 (21.4%)**

这些 > 50m 的角落体素通常距离 Gaussians 中心较远，覆盖率更低。由于 Distance-wise Coverage 只统计 `[0, 50)` 范围内的体素，而全局 Mean Coverage 包含所有体素（包括角落中低覆盖率的体素），因此 **全局 Mean Coverage 会系统性低于 Distance-wise Coverage 各分桶的均值**。

场景示意图（俯视图）：

```
       50m
    ┌──────────────┐
    │  ████████████ │  ← 距离原点 > 50m 的角落体素
    │  ██  [0,50) ██│     - 占比 ~21%
    │  ██  圆柱体 ██│     - 通常覆盖率较低
    │  ████████████ │
    └──────────────┘
       ← 100m →
```

---

## 修复建议

### 建议 1：调整默认 cov_threshold 为 2.0

将 `--cov-threshold` 默认值从 **1.0 改为 2.0**。τ=2.0 的 2σ 椭球：
- 包含约 95.4% 的概率质量，与实际语义范围匹配
- 覆盖 ~4.7 个体素/Gaussian，覆盖率约 47-94%（有区分度）
- 与渲染可视化的效果更接近

### 建议 2：统一全局 Mean Coverage 与 Distance-wise 的统计口径

将全局 Mean Coverage 改为 **只统计距离在 `[0, 50)` 范围内的可见体素**，与 Distance-wise Coverage 保持一致。

修改位置：[`aggregator.py`](gaussian_statistic/aggregator.py:205-207)：
```python
# 当前（所有可见体素）：
self.total_cov_covered += covered.sum().item()
self.total_cov_total += valid_xyz.shape[0]

# 建议修改（只统计距离分桶内的体素，与 distance-wise 一致）：
self.total_cov_covered += (covered & valid_b).sum().item()
self.total_cov_total += valid_b.sum().item()
```

### 建议 3：在报告中增加 cov_threshold 使用说明

在 [`reporter.py`](gaussian_statistic/reporter.py) 输出的 Coverage 部分注明当前使用的 τ 值，便于用户理解结果含义。

### 建议 4：增加多 τ 对比功能（可选）

允许一次运行中同时计算 τ=1.0、2.0、3.0 的 Coverage，便于调试和对比分析。这需要在 `compute_coverage_and_purity` 中支持多阈值。

---

## 总结

| 问题 | 原因 | 建议 |
|------|------|------|
| 原始 τ=3 全为 1.0 | τ=3 椭球太大，每帧覆盖体积超过场景 | 默认 τ 改为 2.0 |
| 新 τ=1 仅 0.1 | τ=1 椭球比体素还小 | 默认 τ 改为 2.0 |
| 全局 vs 分桶不一致 | 全局包含 >50m 远角体素 | 统一统计口径 |
| 与可视化不符 | 硬几何覆盖 vs soft alpha 渲染本质不同 | 文档说明差异 |
