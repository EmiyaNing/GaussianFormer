# Gaussian Statistic 系统技术文档

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     gaussian_statistic.py (主入口)                    │
│  加载配置 → 构建模型 → 加载权重 → 遍历验证集 → 逐帧统计 → 汇总输出   │
└──────────────────┬──────────────────────────────────────┬───────────┘
                   │                                      │
         ┌─────────▼──────────┐             ┌─────────────▼──────────┐
         │  GaussianStat       │             │  report_statistics()   │
         │  Aggregator         │             │  (reporter.py)         │
         │  (aggregator.py)    │             │  格式化输出 + JSON      │
         │  跨帧累加器          │             │                        │
         └─────────┬───────────┘             └────────────────────────┘
                   │
         ┌─────────▼───────────┐
         │  calculator.py       │
         │  纯数学计算函数       │
         │  (无状态, 无 Module) │
         └─────────────────────┘
```

## 2. 数据流图

### 2.1 主循环数据流

```
val_dataset_loader
    │
    ▼
data = { 'img': Tensor(B,N,3,H,W), 'occ_xyz': Tensor(B,200,200,16,4),
         'occ_label': Tensor(B,200,200,16), 'occ_cam_mask': Tensor(B,200,200,16),
         'projection_mat': ..., 'image_wh': ... }
    │
    ├── input_imgs = data.pop('img')          # (1, 6, 3, 864, 1600)
    └── metas = data                          # 剩余所有 key
         │
         ▼
    result_dict = my_model(imgs=input_imgs, metas=metas)
         │
         ├── gaussian = result_dict['gaussian']  # GaussianPrediction
         └── 其他: 'pred_occ', 'final_occ', ...
              │
              ▼
    aggregator.add_frame(gaussian, metas)     # 核心统计调用
```

### 2.2 add_frame 内部数据流

```
gaussian (GaussianPrediction)
  ├── .means     (1, 25600, 3)    → means    (25600, 3)
  ├── .scales    (1, 25600, 3)    → scales   (25600, 3)
  ├── .rotations (1, 25600, 4)    → rotations(25600, 4)
  └── .semantics (1, 25600, 17)   → semantics(25600, 17)
                                      → pred_class(25600,) argmax
                                      → s_hat(25600,) mean(dim=-1)
                                      → ar(25600,) max/min
                                      → nsr_mask(25600,) ar < t_sphere
                                      → vol(25600,) (4/3)π·sx·sy·sz

metas
  ├── occ_xyz      (1,200,200,16,4) → (200,200,16,3) → (640000,3)
  ├── occ_label    (1,200,200,16)   → (200,200,16)   → (640000,)
  └── occ_cam_mask (1,200,200,16)   → (200,200,16)   → (640000,) bool
       │
       └── valid_xyz = occ_xyz[occ_cam_mask]   # (N_visible, 3)
```

## 3. 输入数据 shape 与格式

### 3.1 GaussianPrediction (model/encoder/gaussian_encoder/utils.py:62)

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `means` | `(1, 25600, 3)` | float32 | Gaussian 中心坐标 (x,y,z)，范围 [-50,50]×[-50,50]×[-5,3] |
| `scales` | `(1, 25600, 3)` | float32 | Gaussian 三轴尺度 s_x, s_y, s_z，范围 [0.08, 0.64] |
| `rotations` | `(1, 25600, 4)` | float32 | 旋转四元数 (w,x,y,z) |
| `opacities` | `(1, 25600, 1)` | float32 | 不透明度 |
| `semantics` | `(1, 25600, 17)` | float32 | 语义 logits，经 softplus 激活，17 类 |

### 3.2 GT Occupancy (来自数据集)

| 字段 | Shape (batch) | Shape (squeezed) | 说明 |
|------|:---:|:---:|------|
| `occ_xyz` | `(1, 200, 200, 16, 4)` | `(640000, 3)` | 体素世界坐标 (x,y,z)，homogeneous 第 4 维被丢弃 |
| `occ_label` | `(1, 200, 200, 16)` | `(640000,)` | 语义标签 (0-16 语义类, 17=empty) |
| `occ_cam_mask` | `(1, 200, 200, 16)` | `(640000,)` bool | True=可见非空体素 |

坐标系统：`use_ego=False`，体素坐标由 meshgrid 生成：
```
xxx = [i * 0.5 + 0.25 - 50 for i in range(200)]  # [-49.75, ..., 49.75]
yyy = [j * 0.5 + 0.25 - 50 for j in range(200)]  # [-49.75, ..., 49.75]
zzz = [k * 0.5 + 0.25 - 5  for k in range(16)]   # [-4.75, ..., 2.75]
```

## 4. 统计指标计算逻辑

### 4.1 基础 Gaussian 属性统计

| 指标 | 计算公式 | 代码位置 |
|------|---------|---------|
| Mean Scale | `mean(scales.mean(dim=-1))` | [`aggregator.py:111`](gaussian_statistic/aggregator.py:111) |
| Scale Percentiles | `quantile(scales.mean(dim=-1), [50,75,90,95]%)` | [`calculator.py:78-84`](gaussian_statistic/calculator.py:78-84) |
| Scale Volume | `(4/3)π·sx·sy·sz` | [`calculator.py:86-90`](gaussian_statistic/calculator.py:86-90) |
| Anisotropy Ratio | `max(s) / min(s)` | [`calculator.py:92-96`](gaussian_statistic/calculator.py:92-96) |
| Near-Spherical Ratio | `mean(AR < t_sphere)` | [`calculator.py:98-99`](gaussian_statistic/calculator.py:98-99) |
| LIGR | `mean((s_hat > t_scale) & (AR < t_sphere))` | [`calculator.py:101-103`](gaussian_statistic/calculator.py:101-103) |

### 4.2 Distance-wise Scale/Shape

**目的**：按 Gaussian 到原点的距离分桶，统计每个距离环内 Gaussians 的尺度和形状。

**计算逻辑** ([`aggregator.py:129-143`](gaussian_statistic/aggregator.py:129-143))：

```python
dist = max(|means_x|, |means_y|)        # Chebyshev 距离
bin_idx = bucketize(dist, bins) - 1     # 分配到 [0,10),[10,20),...
valid = (bin_idx >= 0) & (bin_idx < n_bins)

# GPU scatter_add 累加
dist_count[bin] += 1                    # Gaussian 计数
dist_sum_scale[bin] += s_hat            # scale 累加
dist_sum_ar[bin] += ar                  # AR 累加
dist_nsr_count[bin] += nsr_mask         # 近球计数
```

**输出**：每个 bin 的 `{count, mean_scale, mean_ar, near_spherical_ratio}`

### 4.3 Category-wise Statistics

**目的**：按 Gaussian 预测的语义类别分组统计。

**计算逻辑** ([`aggregator.py:145-150`](gaussian_statistic/aggregator.py:145-150))：

```python
pred_class = semantics.argmax(dim=-1)   # (G,) 预测类别 [0, 16]

# GPU scatter_add 累加
cat_count[class] += 1
cat_sum_scale[class] += s_hat
cat_sum_ar[class] += ar
cat_nsr_count[class] += nsr_mask
```

**注意**：这里统计的是所有 Gaussians（包括空/背景类），`num_classes` 默认从 model head 获取。

### 4.4 Coverage（几何覆盖率）

#### 4.4.1 覆盖判定算法 ([`calculator.py:124-170`](gaussian_statistic/calculator.py:124-170))

给定 N 个可见体素 `voxel_xyz` 和 G 个 Gaussians，判断每个体素是否被至少一个 Gaussian 的 τ-椭球覆盖：

```
for each voxel chunk (chunk_size=30000):
    dists = cdist(chunk_voxels, means)                    # (chunk_N, G) 欧氏距离
    cand_mask = dists <= τ * max(scales)                  # 候选过滤
    
    for each candidate pair (voxel_i, gaussian_j):
        d_rot = R[j] @ (voxel - mean[j])                  # 旋转到局部坐标系
        mahal² = Σ(d_rot[k] / scales[j][k])²              # 马氏距离平方
        if mahal² <= τ²:  → covered[voxel] = True         # 被覆盖
```

**核心参数**：
- `τ = --cov-threshold`（默认 1.0）：马氏距离阈值
- `search_radius = τ * max(scales)`：候选搜索半径
- `tau_sq = τ²`：精确判定阈值

> 数学等价性：`mahal² ≤ τ²` 等价于体素位于 Gaussian 的 τ-标准差椭球内部。

#### 4.4.2 Distance-wise Coverage ([`aggregator.py:196-205`](gaussian_statistic/aggregator.py:196-205))

**目的**：统计每个同心方环区域内，可见体素被 Gaussian 覆盖的比例。

```python
valid_dist = max(|voxel_x|, |voxel_y|)     # Chebyshev 距离
bin_idx = bucketize(valid_dist, bins) - 1
valid_b = (bin_idx >= 0) & (bin_idx < n_bins)

# GPU scatter_add 累加
dcov_total[bin] += 1                         # 该环内体素总数
dcov_covered[bin] += covered                 # 该环内被覆盖体素数
```

**最终输出**：每个 bin 的 `coverage = dcov_covered[i] / dcov_total[i]`

#### 4.4.3 全局 Mean Coverage ([`aggregator.py:207-209`](gaussian_statistic/aggregator.py:207-209))

```python
total_cov_covered += (covered & valid_b).sum()   # 所有 bin 的被覆盖体素数
total_cov_total += valid_b.sum()                  # 所有 bin 的体素总数
```

等价于 distance-wise coverage 按体素数的加权平均。

### 4.5 Purity（语义纯度）

**目的**：每个 Gaussian 覆盖的体素中，语义标签与 Gaussian 预测类别一致的比例。

**计算逻辑** ([`calculator.py:167-168`](gaussian_statistic/calculator.py:167-168))：

```python
# 对每个被覆盖的体素-高斯对：
p_total[gaussian] += 1                                              # 该 Gaussian 覆盖的体素数
p_match[gaussian] += (voxel_label == gaussian_pred_class)           # 语义匹配数

# Per-Gaussian Purity：
purity[g] = p_match[g] / p_total[g]  # 若 p_total[g]=0 则视为 1.0
```

**两种聚合方式**：

1. **Per-class Purity** ([`aggregator.py:212-222`](gaussian_statistic/aggregator.py:212-222))：
   - 每帧先按类别累加 `p_match_cls` 和 `p_total_cls`
   - 计算每帧每个类别的 purity = p_match_cls / p_total_cls
   - 跨帧平均（每帧贡献一次，避免帧间体素数不均衡）

2. **全局 Mean Purity** ([`aggregator.py:224-229`](gaussian_statistic/aggregator.py:224-229))：
   - 跨所有 Gaussian、所有帧计算：`total_purity_sum / total_gaussians`
   - 每帧的 per-Gaussian purity = p_match / p_total（p_total=0 时视为 1.0）

## 5. CLI 参数

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--py-config` | `config/nuscenes_gs25600_voxel.py` | 模型配置 |
| `--work-dir` | `./out/gaussian_statistic` | 输出目录 |
| `--resume-from` | `''` | 权重路径 |
| `--seed` | `42` | 随机种子 |
| `--t-sphere` | `2.0` | 近球判定阈值 (AR < t_sphere) |
| `--t-scale` | `0.5` | 大 Gaussian 阈值 (s_hat > t_scale) |
| `--cov-threshold` | `1.0` | Coverage 马氏距离阈值 τ |
| `--distance-bins` | `[0,10,20,30,40,50]` | 距离分桶边界（米） |
| `--percentiles` | `[50,75,90,95]` | 分位数列表 |
| `--stat-freq` | `100` | 中间快照打印频率（帧） |
| `--chunk-size` | `30000` | Coverage 计算 chunk 大小 |
| `--exclude-classes` | `None` | 排除的语义类别索引 |

## 6. 输出结构

### 6.1 终端输出

```
============================================================
====  Gaussian Statistic Results  ====
============================================================
  Frames processed:   XXX
  Total Gaussians:      XXXXXXXXXX
────────────────────────────────────────────────────────────
  Mean Scale:          0.XXXXXX
  Near-Spherical Ratio: 0.XXXXXX
  LIGR:                0.XXXXXX
  Mean Coverage:       0.XXXXXX    ← 全局平均覆盖率
  Mean Purity:         0.XXXXXX
────────────────────────────────────────────────────────────
  [Scale Percentiles]
    P50: 0.XXXXXX
    ...
────────────────────────────────────────────────────────────
  [Scale Volume]
    mean_volume: 0.XXXXXX
    ...
────────────────────────────────────────────────────────────
  [Anisotropy Ratio]
    mean_ar: 1.XXXXXX
    ...
────────────────────────────────────────────────────────────
  [Category-wise Statistics]
     Class |    Count |  MeanScale |   MeanAR |      NSR |   Purity
     -----------------------------------------------------------------
         0 |      XXXX |   0.XXXXXX |   X.XXXX |   0.XXXX |   0.XXXX
         ...
────────────────────────────────────────────────────────────
  [Distance-wise Scale/Shape]
              Bin |    Count |  MeanScale |   MeanAR |      NSR
     ----------------------------------------------------------
          (0,10) |    XXXXX |   0.XXXXXX |   X.XXXX |   0.XXXX
         ...
────────────────────────────────────────────────────────────
  [Distance-wise Coverage]
           (0,10): X.XXXX
          ...
============================================================
```

### 6.2 JSON 输出 (`gaussian_statistic_result.json`)

```json
{
  "num_frames": 6019,
  "num_gaussians": 154086400,
  "mean_scale": 0.34,
  "near_spherical_ratio": 0.66,
  "ligr": 0.0,
  "mean_coverage": 0.52,
  "mean_purity": 0.003,
  "scale_percentiles": { "P50": 0.33, "P75": 0.35, ... },
  "scale_volume": { "mean_volume": 0.15, ... },
  "anisotropy_ratio": { "mean_ar": 1.34, ... },
  "category_stats": {
    "0": { "count": 2849, "mean_scale": 0.33, ... },
    "1": { ... },
    ...
  },
  "distance_stats": {
    "(0,10)": { "count": 4668309, "mean_scale": 0.34, ... },
    "(10,20)": { ... },
    ...
  },
  "distancewise_coverage": {
    "(0,10)": 0.85,
    "(10,20)": 0.72,
    ...
  }
}
```

## 7. GPU 累积器状态

所有 GPU 端累加器均在 `_init_gpu_state()` 中初始化（[`aggregator.py:62-86`](gaussian_statistic/aggregator.py:62-86)）：

| 累加器 | Shape | 用途 |
|--------|:-----:|------|
| `_dist_count` | `(n_bins,)` | 每个距离 bin 的 Gaussian 计数 |
| `_dist_sum_scale` | `(n_bins,)` | 每个距离 bin 的 scale 累加 |
| `_dist_sum_ar` | `(n_bins,)` | 每个距离 bin 的 AR 累加 |
| `_dist_nsr_count` | `(n_bins,)` | 每个距离 bin 的近球 Gaussian 计数 |
| `_dist_nsr_total` | `(n_bins,)` | 每个距离 bin 的 Gaussian 总计数 |
| `_cat_count` | `(num_classes,)` | 每个类别的 Gaussian 计数 |
| `_cat_sum_scale` | `(num_classes,)` | 每个类别的 scale 累加 |
| `_cat_sum_ar` | `(num_classes,)` | 每个类别的 AR 累加 |
| `_cat_nsr_count` | `(num_classes,)` | 每个类别的近球 Gaussian 计数 |
| `_cat_nsr_total` | `(num_classes,)` | 每个类别的 Gaussian 总计数 |
| `_cat_sum_purity` | `(num_classes,)` | 每个类别的 purity 累加（per-frame） |
| `_cat_purity_frames` | `(num_classes,)` | 每个类别有 purity 数据的帧数 |
| `_dcov_covered` | `(n_bins,)` | 每个 bin 被覆盖体素数 |
| `_dcov_total` | `(n_bins,)` | 每个 bin 体素总数 |

## 8. 标量累加器

| 累加器 | 类型 | 用途 |
|--------|:----:|------|
| `total_frames` | int | 处理帧数 |
| `total_gaussians` | int | 所有帧 Gaussian 总数 |
| `sum_mean_scale` | float | 每帧 mean_scale 累加 |
| `sum_nsr` | float | 每帧 NSR 累加 |
| `sum_ligr` | float | 每帧 LIGR 累加 |
| `vol_sum` | float | 所有 Gaussian 体积累加 |
| `vol_count` | int | 所有 Gaussian 计数 |
| `total_purity_sum` | float | 所有 Gaussian 的 per-G purity 累加 |
| `total_cov_covered` | float | 所有帧被覆盖体素总数 |
| `total_cov_total` | int | 所有帧参与统计体素总数 |

> **设计原则**：标量累加器使用 `.item()` 每帧同步一次（可接受），GPU tensor 累加器则在整个验证集遍历完成后一次性 `cpu().tolist()` 同步。

## 9. 代码模块关系图

```
gaussian_statistic.py
    │
    ├── gaussian_statistic/__init__.py
    │       └── 导出所有 calculator 函数 + aggregator + reporter
    │
    ├── gaussian_statistic/aggregator.py
    │       GaussianStatAggregator
    │       ├── __init__()        ← 参数初始化 + GPU 累加器声明
    │       ├── _init_gpu_state() ← GPU tensor 延迟初始化
    │       ├── add_frame()       ← 每帧入口
    │       │   ├── 基础属性统计 (s_hat, ar, nsr, vol)
    │       │   ├── Distance-wise 分桶 (means → Chebyshev dist)
    │       │   ├── Category-wise 分桶 (pred_class)
    │       │   └── _accumulate_cov_purity()
    │       │       ├── 类别过滤 (exclude_classes)
    │       │       ├── precompute_frame_data() → calculator
    │       │       ├── compute_coverage_and_purity() → calculator
    │       │       ├── Distance-wise Coverage 累加
    │       │       ├── 全局 Mean Coverage 累加
    │       │       └── Category-wise Purity 累加
    │       ├── get_snapshot()    ← 中间快照
    │       └── finalize()        ← GPU→CPU 汇总
    │
    ├── gaussian_statistic/calculator.py
    │       ├── _get_rotation_matrix()    ← 四元数 → 旋转矩阵
    │       ├── _build_cov_inv()           ← scales+rot → 协方差逆
    │       ├── precompute_frame_data()    ← 每帧预计算
    │       ├── compute_coverage_and_purity() ← 核心覆盖判定
    │       ├── compute_distancewise_coverage() ← 独立函数版
    │       ├── compute_category_stats()   ← 独立函数版
    │       ├── compute_mean_scale()       ← 单帧 mean scale
    │       ├── compute_scale_percentiles()
    │       ├── compute_scale_volume()
    │       ├── compute_anisotropy_ratio()
    │       ├── compute_near_spherical_ratio()
    │       ├── compute_ligr()
    │       ├── compute_distance_stats()
    │       └── mask_flat_valid()
    │
    └── gaussian_statistic/reporter.py
            report_statistics()       ← 格式化输出 + JSON 写入
            _make_serializable()      ← numpy → Python 原生类型转换
```
