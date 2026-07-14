# gaussian_statistic 更多统计功能设计

## 1. 目标概览

本次扩展目标是在当前 `gaussian_statistic` 已有的 category-wise / distance-wise / coverage / purity 统计基础上，增加两个更细粒度指标：

| 新指标 | 输出位置 | 核心问题 | 推荐主口径 |
|---|---|---|---|
| Category-wise Coverage | `category_stats` | 每个语义类别的非空可见体素是否被 Gaussian 覆盖 | GT-class geometric coverage |
| Distance-wise Non_empty_visible Purity | 新增 `distancewise_purity` | 不同距离范围内，被 Gaussian 覆盖到的非空可见体素语义是否纯 | hit-pair semantic purity |

当前代码已经在一次覆盖判定中产出：

| 中间结果 | 含义 | 现有用途 |
|---|---|---|
| `covered` | 每个有效体素是否被至少一个 Gaussian 覆盖 | global coverage / distance-wise coverage |
| `purity_match` | 每个 Gaussian 覆盖的 GT 语义匹配次数 | global purity / category-wise purity |
| `purity_total` | 每个 Gaussian 覆盖的体素总次数 | global purity / category-wise purity |

因此本次设计的重点不是重复计算覆盖关系，而是在同一次 `compute_coverage_and_purity()` 的 hit-pair 遍历中额外收集 category / distance 维度的计数器。

---

## 2. 当前统计链路

### 2.1 模块关系

```text
gaussian_statistic.py
    |
    | 逐帧取出 model 输出的 GaussianPrediction 与 metas
    v
GaussianStatAggregator.add_frame()
    |
    | 基础 Gaussian 属性:
    | mean scale / AR / NSR / LIGR / distance stats / category stats
    |
    v
GaussianStatAggregator._accumulate_cov_purity()
    |
    | 构建 non_empty_visible 体素集合
    | 预计算 Gaussian covariance / rotation / search radius
    |
    v
calculator.compute_coverage_and_purity()
    |
    | 分 chunk 枚举 voxel-Gaussian hit pairs
    |
    +--> covered
    +--> purity_match
    +--> purity_total
    |
    v
Aggregator GPU counters
    |
    v
finalize() -> reporter.py -> log + gaussian_statistic_result.json
```

### 2.2 当前有效体素口径

当前 `_accumulate_cov_purity()` 中的 coverage / purity 统计体素为：

```text
valid_voxel = occ_cam_mask
if ignore_empty:
    valid_voxel = valid_voxel and occ_label != empty_label
if exclude_voxel_classes is not None:
    valid_voxel = valid_voxel and occ_label not in exclude_voxel_classes
```

当默认 `--ignore-empty=True` 时，输出日志中的 `Coverage Scope` 为：

```text
non_empty_visible
```

新指标全部沿用这个 scope，避免 global coverage、distance-wise coverage、category-wise coverage、distance-wise purity 使用不同分母。

---

## 3. 新指标一：Category-wise Coverage

### 3.1 需求解释

用户需求是：

```text
在当前 category-wise statistics 的基础上额外添加 coverage，
统计每个类别的 Gaussian 的 coverage。
```

这里有两个容易混淆的口径：

| 口径 | 问题 | 优点 | 风险 |
|---|---|---|---|
| GT-class geometric coverage | GT 类别 c 的非空可见体素，有多少被任意 Gaussian 覆盖 | 与 global coverage 严格一致，解释为“该类别空间结构是否被覆盖” | 不要求 Gaussian 预测类别正确 |
| Class-aligned Gaussian coverage | GT 类别 c 的非空可见体素，有多少被预测为 c 的 Gaussian 覆盖 | 同时体现 coverage 与 semantic alignment | 更像 coverage+purity 的组合指标 |
| Pred-class footprint coverage | 预测类别 c 的 Gaussian 覆盖了多少非空可见体素 | 反映某类 Gaussian 的空间占用强度 | 分母不好定义，类别间可重叠，不适合作为主表 ratio |

推荐设计：在 `category_stats` 中新增两个 coverage 字段。

| 字段 | 推荐程度 | 定义 |
|---|---:|---|
| `coverage` | 主字段 | GT 类别 c 的 non_empty_visible 体素，被任意 Gaussian 覆盖的比例 |
| `aligned_coverage` | 辅助字段 | GT 类别 c 的 non_empty_visible 体素，被预测类别也为 c 的 Gaussian 覆盖的比例 |

这样当前 category-wise table 会同时表达：

```text
这个类别有多少 Gaussian、这些 Gaussian 形状如何、纯度如何；
这个类别的 GT 空间是否被覆盖；
这个类别的 GT 空间是否被同类别 Gaussian 覆盖。
```

### 3.2 数学定义

设：

| 符号 | 含义 |
|---|---|
| `V` | 当前统计 scope 下所有 non_empty_visible 体素 |
| `V_c` | `V` 中 GT label 为 c 的体素 |
| `G` | 当前参与 coverage 的 Gaussian |
| `G_c` | `G` 中预测类别为 c 的 Gaussian |
| `Covered(v, G)` | 体素 v 是否被任意 Gaussian 覆盖 |
| `Covered(v, G_c)` | 体素 v 是否被预测类别为 c 的 Gaussian 覆盖 |

#### 3.2.1 Category Coverage

```text
category_coverage[c]
  = count(v in V_c and Covered(v, G))
    / count(v in V_c)
```

含义：类别 c 的真实非空可见体素，有多少几何上被 Gaussian 表达到了。

#### 3.2.2 Category Aligned Coverage

```text
category_aligned_coverage[c]
  = count(v in V_c and Covered(v, G_c))
    / count(v in V_c)
```

含义：类别 c 的真实非空可见体素，有多少不仅被覆盖，而且是由预测为同类别 c 的 Gaussian 覆盖。

### 3.3 数据流图

```text
valid_xyz, valid_label
    |
    | valid_label -> GT class c
    v
compute_coverage_and_purity()
    |
    | hit pair: voxel_i <-> gaussian_j
    | pred_class_j = argmax(semantics_j)
    |
    +-------------------------------+
    |                               |
    v                               v
Any-Gaussian covered[v]         Same-class covered[v, c]
    |                               |
    | group by GT label             | group by GT label
    v                               v
cat_cov_covered[c]              cat_aligned_cov_covered[c]
cat_cov_total[c]                cat_cov_total[c]
    |                               |
    v                               v
coverage[c]                    aligned_coverage[c]
```

### 3.4 推荐计数器

在 `GaussianStatAggregator` 中新增 GPU 累加器：

| 累加器 | Shape | dtype | 含义 |
|---|---:|---|---|
| `_cat_cov_total` | `(num_classes,)` | int64 | 每个 GT 类别的 non_empty_visible 体素数 |
| `_cat_cov_covered` | `(num_classes,)` | int64 | 每个 GT 类别中，被任意 Gaussian 覆盖的体素数 |
| `_cat_aligned_cov_covered` | `(num_classes,)` | int64 | 每个 GT 类别中，被同预测类别 Gaussian 覆盖的体素数 |

注意：

```text
_cat_cov_total 以 GT label 分组。
_cat_count / _cat_sum_scale / _cat_sum_ar 仍然以 Gaussian pred_class 分组。
```

这两个分组维度不同，但都放在 `category_stats[c]` 下是合理的，因为最终都以语义类 c 为索引。

### 3.5 计算策略

#### 3.5.1 category coverage total

在 `_accumulate_cov_purity()` 构建出 `valid_label` 后，直接按 `valid_label` 累加：

```text
cat_cov_total[c] += count(valid_label == c)
```

只统计满足 `0 <= c < num_classes` 的标签。

#### 3.5.2 category coverage covered

`compute_coverage_and_purity()` 返回 `covered` 后，在 aggregator 中按 GT label 累加：

```text
cat_cov_covered[c] += count(valid_label == c and covered == True)
```

这个不需要修改 calculator。

#### 3.5.3 aligned coverage covered

`aligned_coverage` 必须知道某个体素是否被同类别 Gaussian hit 到，仅靠最终的 `covered` 不够。因此需要扩展 calculator，在 hit-pair 循环里额外记录：

```text
当 voxel label == gaussian pred_class:
    aligned_covered[voxel] = True
```

然后在 aggregator 中按 GT label 累加：

```text
cat_aligned_cov_covered[c] += count(valid_label == c and aligned_covered == True)
```

推荐让 calculator 通过可选参数返回 `aligned_covered`，避免默认调用者被迫处理额外结果。

### 3.6 输出结构

在 `finalize()` 中，`category_stats` 每个类别新增：

```text
category_stats[c]:
    count
    ratio
    mean_scale
    mean_ar
    near_spherical_ratio
    mean_purity
    coverage
    aligned_coverage
    coverage_total
    coverage_covered
    aligned_coverage_covered
```

推荐 reporter 表格新增两列：

```text
Class | Count | Ratio | MeanScale | MeanAR | NSR | Purity | Cov | AlignCov
```

---

## 4. 新指标二：Distance-wise Non_empty_visible Purity

### 4.1 需求解释

当前已有：

| 指标 | 当前状态 |
|---|---|
| Distance-wise Coverage | 已有，按 voxel distance 分桶 |
| Global Purity(valid / penalized) | 已有，按 Gaussian 聚合 |
| Category-wise Purity | 已有，按 Gaussian pred_class 聚合 |
| Distance-wise Purity | 缺失 |

新需求中的 `Distance-wise Non_empty_visible Purity` 建议定义为：

```text
在每个距离 bin 内，所有 non_empty_visible voxel-Gaussian hit pairs 中，
GT label 与 Gaussian pred_class 一致的比例。
```

这是最自然的 distance-wise purity，因为 distance bin 是体素空间区域，而不是 Gaussian 自身属性区域。

### 4.2 数学定义

设距离分桶 `B_k = [d_k, d_{k+1})`，体素距离沿用当前代码的 Chebyshev 距离：

```text
dist(v) = max(abs(v_x), abs(v_y))
```

对所有 hit pairs：

```text
Hit(v, g) = True
```

当且仅当体素 v 落在 Gaussian g 的 tau 椭球内。

Distance-wise Non_empty_visible Purity 定义为：

```text
distancewise_purity[k]
  = count(Hit(v, g) and dist(v) in B_k and label(v) == pred_class(g))
    / count(Hit(v, g) and dist(v) in B_k)
```

### 4.3 为什么使用 hit-pair purity

| 方案 | 定义 | 推荐度 | 原因 |
|---|---|---:|---|
| Hit-pair purity | bin 内匹配 hit pairs / 全部 hit pairs | 高 | 与现有 `purity_match/purity_total` 概念一致，易实现 |
| Covered-voxel purity | bin 内被覆盖体素是否存在同类 Gaussian | 中 | 更接近 aligned coverage，不能表达多个错误 Gaussian 污染 |
| Gaussian-wise distance purity | 按 Gaussian 中心距离分桶后平均 per-G purity | 低 | 与 “non_empty_visible” voxel scope 不一致 |

推荐主输出用 hit-pair purity，并在文档和 reporter 中明确：

```text
Distance-wise purity is computed over voxel-Gaussian hit pairs in non_empty_visible voxels.
```

### 4.4 数据流图

```text
valid_xyz
    |
    | distance = max(abs(x), abs(y))
    | bucketize(distance, distance_bins)
    v
voxel_bin_idx
    |
    v
compute_coverage_and_purity()
    |
    | hit pair: voxel_i <-> gaussian_j
    | bin = voxel_bin_idx[voxel_i]
    |
    +--> dist_purity_total[bin] += 1
    |
    +--> if valid_label[voxel_i] == pred_class[gaussian_j]:
             dist_purity_match[bin] += 1
    |
    v
distancewise_purity[bin] = match / total
```

### 4.5 推荐计数器

在 `GaussianStatAggregator` 中新增 GPU 累加器：

| 累加器 | Shape | dtype | 含义 |
|---|---:|---|---|
| `_dpurity_match` | `(n_bins,)` | int64 | 每个距离 bin 中语义匹配的 hit-pair 数 |
| `_dpurity_total` | `(n_bins,)` | int64 | 每个距离 bin 中所有 hit-pair 数 |

### 4.6 calculator 扩展策略

当前 `compute_coverage_and_purity()` 内部已经有完整 hit-pair 信息：

```text
voxel index
gaussian index
voxel label
gaussian pred_class
hit mask
```

新增 distance-wise purity 不应该在外部重新计算覆盖关系。推荐改造为：

```text
输入可选:
    voxel_bin_idx
    n_bins
    return_extra_stats

输出可选:
    distance_purity_match
    distance_purity_total
    aligned_covered
```

模块结构：

```text
compute_coverage_and_purity()
    |
    | 原有输出:
    | covered, purity_match, purity_total
    |
    | 新增可选输出:
    | extra_stats
    |   - aligned_covered
    |   - distance_purity_match
    |   - distance_purity_total
```

这样可以保持旧调用兼容：未请求 extra stats 时仍返回原有三项。

### 4.7 输出结构

`finalize()` 中新增：

```text
distancewise_purity:
    "(0,10)": value
    "(10,20)": value
    ...

distancewise_purity_debug:
    "(0,10)":
        match
        total
        purity
    ...
```

Reporter 中新增区域：

```text
[Distance-wise Non_empty_visible Purity]
      (0,10): 0.xxxx
     (10,20): 0.xxxx
     ...
```

---

## 5. 需要改动的文件

### 5.1 `gaussian_statistic/calculator.py`

#### 修改点 A：扩展 `compute_coverage_and_purity()`

目标：

```text
在不重复 coverage 计算的前提下，从 hit-pair 循环中额外提取:
1. aligned_covered
2. distance_purity_match
3. distance_purity_total
```

新增输入建议：

| 参数 | 类型 | 默认 | 用途 |
|---|---|---|---|
| `voxel_bin_idx` | Tensor or None | None | 每个有效体素的 distance bin |
| `n_bins` | int or None | None | distance bin 数 |
| `return_extra_stats` | bool | False | 是否返回扩展统计 |

新增输出建议：

```text
return_extra_stats == False:
    covered, purity_match, purity_total

return_extra_stats == True:
    covered, purity_match, purity_total, extra_stats
```

`extra_stats` 结构：

```text
extra_stats:
    aligned_covered: per-voxel bool vector
    distance_purity_match: per-bin int64 vector
    distance_purity_total: per-bin int64 vector
```

#### 修改点 B：保持独立函数兼容

`compute_distancewise_coverage()` 和 `compute_category_stats()` 仍然可以使用旧三元返回。若不传 `return_extra_stats`，行为不变。

### 5.2 `gaussian_statistic/aggregator.py`

#### 修改点 A：初始化新增累加器

在 `__init__()` 中声明：

```text
category coverage counters:
    _cat_cov_total
    _cat_cov_covered
    _cat_aligned_cov_covered

distance purity counters:
    _dpurity_match
    _dpurity_total
```

在 `_init_gpu_state()` 中按 device 初始化，dtype 使用 int64。

#### 修改点 B：在 `_accumulate_cov_purity()` 中计算 voxel bin

当前已经计算：

```text
valid_dist = max(abs(valid_xyz_x), abs(valid_xyz_y))
bin_idx = bucketize(valid_dist, distance_bins) - 1
valid_b = bin_idx inside range
```

这组结果应复用给：

| 统计项 | 使用 |
|---|---|
| distance-wise coverage | 已有 |
| global coverage denominator | 已有 |
| distance-wise purity | 新增 |

#### 修改点 C：调用 calculator 获取 extra stats

将 `voxel_bin_idx` 和 `n_bins` 传入 calculator，并请求 extra stats。

得到：

```text
aligned_covered
distance_purity_match
distance_purity_total
```

然后累加：

```text
_dpurity_match += distance_purity_match
_dpurity_total += distance_purity_total
```

#### 修改点 D：累加 category coverage

用 `valid_label` 和 `covered` 累加：

```text
_cat_cov_total[c] += count(valid_label == c and valid_b)
_cat_cov_covered[c] += count(valid_label == c and covered and valid_b)
_cat_aligned_cov_covered[c] += count(valid_label == c and aligned_covered and valid_b)
```

这里建议加上 `valid_b`，让 category-wise coverage 的空间范围与 global mean coverage 完全一致：只统计落在配置 distance bins 内的体素。

#### 修改点 E：finalize 输出新字段

在 GPU 到 CPU 同步时读取新增计数器，并在 `result` 中写入：

```text
category_stats[c].coverage
category_stats[c].aligned_coverage
category_stats[c].coverage_total
category_stats[c].coverage_covered
category_stats[c].aligned_coverage_covered

distancewise_purity
distancewise_purity_debug
```

#### 修改点 F：新增一致性 debug

推荐新增 category coverage debug：

```text
category_coverage_debug:
    sum_cat_cov_total
    scalar_cov_total
    category_total_matches_scalar
```

期望：

```text
sum(_cat_cov_total) == total_cov_total
```

如果不一致，说明 category-wise coverage 和 global coverage 使用了不同体素范围。

### 5.3 `gaussian_statistic/reporter.py`

#### 修改点 A：Category-wise 表格新增列

当前表格：

```text
Class | Count | Ratio | MeanScale | MeanAR | NSR | Purity
```

推荐变更为：

```text
Class | Count | Ratio | MeanScale | MeanAR | NSR | Purity | Cov | AlignCov
```

列宽需要略微调整，避免日志换行。

#### 修改点 B：新增 Distance-wise Purity 区块

建议放在 Distance-wise Coverage 之后：

```text
[Distance-wise Non_empty_visible Coverage]
...

[Distance-wise Non_empty_visible Purity]
...
```

如果 `coverage_voxel_scope` 不是 `non_empty_visible`，标题随 scope 自动变化：

```text
[Distance-wise Visible Purity]
```

#### 修改点 C：debug warning

如果新增的 category coverage consistency check 不通过，输出 warning。

### 5.4 `gaussian_statistic.py`

主入口原则上不需要新增 CLI 参数。

原因：

```text
两个新指标完全复用已有参数:
--cov-threshold
--distance-bins
--empty-label
--ignore-empty
--exclude-gaussian-classes
--exclude-voxel-classes
```

可选新增参数：

| 参数 | 是否推荐 | 用途 |
|---|---:|---|
| `--disable-extra-stat` | 暂不推荐 | 当显存或耗时明显增加时关闭新增统计 |
| `--category-coverage-mode` | 暂不推荐 | 在 any / aligned / both 中选择输出 |

当前建议先默认开启，因为新增统计复用 hit-pair 结果，主要额外开销是少量 int64 scatter 和 bool vector。

### 5.5 `gaussian_statistic/__init__.py`

通常不需要修改。

只有当新增 calculator helper 被外部直接导出时才需要更新。

---

## 6. 输出 JSON 结构设计

### 6.1 新增后的 `category_stats`

```text
category_stats:
    "0":
        count
        ratio
        mean_scale
        mean_ar
        near_spherical_ratio
        mean_purity
        coverage
        aligned_coverage
        coverage_total
        coverage_covered
        aligned_coverage_covered
        scale_sanity_warning
    "1":
        ...
```

### 6.2 新增 `distancewise_purity`

```text
distancewise_purity:
    "(0,10)": 0.xxxx
    "(10,20)": 0.xxxx
    "(20,30)": 0.xxxx
    "(30,40)": 0.xxxx
    "(40,50)": 0.xxxx
```

### 6.3 新增 `distancewise_purity_debug`

```text
distancewise_purity_debug:
    "(0,10)":
        match: integer
        total: integer
        purity: float
    "(10,20)":
        ...
```

### 6.4 新增 `category_coverage_debug`

```text
category_coverage_debug:
    coverage_scope: non_empty_visible
    sum_category_total: integer
    scalar_cov_total: integer
    category_total_matches_scalar: bool
```

---

## 7. 日志输出样式

### 7.1 Category-wise Statistics

```text
[Category-wise Statistics]
 Class |    Count |    Ratio |  MeanScale |  MeanAR |    NSR | Purity |    Cov | AlignCov
 ---------------------------------------------------------------------------------------
     0 |        0 | 0.000000 |   0.000000 |  0.0000 | 0.0000 | 0.0000 | 0.0000 |  0.0000
     1 |   572262 | 0.005739 |   0.573379 |  1.5245 | 0.5471 | 0.1341 | 0.xxxx |  0.xxxx
```

### 7.2 Distance-wise Purity

```text
[Distance-wise Non_empty_visible Coverage]
      (0,10): 0.9988
     (10,20): 0.9942
     ...

[Distance-wise Non_empty_visible Purity]
      (0,10): 0.xxxx
     (10,20): 0.xxxx
     ...
```

---

## 8. 指标关系与一致性检查

### 8.1 Coverage 一致性

已有检查：

```text
weighted_mean(distancewise_coverage) == mean_coverage
```

新增检查：

```text
sum(category_stats[c].coverage_total for c) == total_cov_total
```

当两个检查都成立时，可以确认：

```text
global coverage
distance-wise coverage
category-wise coverage
```

三者使用的是同一批有效体素。

### 8.2 Category Coverage 与 Aligned Coverage 的关系

对每个类别 c，理论上：

```text
0 <= aligned_coverage[c] <= coverage[c] <= 1
```

如果出现 `aligned_coverage > coverage`，说明 aligned hit 的体素集合和 any hit 的体素集合没有使用相同 scope，或者 aligned 计数重复计入了同一体素。

### 8.3 Distance-wise Purity 与 Coverage 的关系

二者没有必然大小关系：

```text
coverage 高，不代表 purity 高。
purity 高，也不代表 coverage 高。
```

例子：

| 情况 | Coverage | Purity | 解释 |
|---|---:|---:|---|
| 大量 Gaussian 覆盖所有区域但类别混乱 | 高 | 低 | 几何覆盖强，语义差 |
| 少量 Gaussian 只覆盖很少区域但类别正确 | 低 | 高 | 语义准，覆盖不足 |
| 远距离区域 coverage/purity 同时下降 | 低 | 低 | 远距离几何和语义都困难 |

---

## 9. 性能与显存评估

### 9.1 额外显存

新增常驻 GPU 累加器很小：

```text
category counters: 3 * num_classes int64
distance purity counters: 2 * n_bins int64
```

对于 `num_classes=18`、`n_bins=5` 几乎可以忽略。

### 9.2 单帧临时显存

可能增加的临时变量：

| 临时量 | Shape | 说明 |
|---|---:|---|
| `aligned_covered` | `(N_valid,)` bool | 每个有效体素是否被同类 Gaussian 覆盖 |
| `voxel_bin_idx` | `(N_valid,)` int64 | 每个有效体素的距离 bin |

其中 `N_valid` 是 non_empty_visible 体素数量，通常显著小于完整 `200*200*16`。

### 9.3 计算开销

新增开销主要发生在 hit-pair 已经筛出来之后：

```text
semantic match 判断
按 bin scatter_add
aligned_covered 写入
```

不会新增 `torch.cdist`，也不会重复计算 Mahalanobis distance，因此相对原 coverage/purity 计算开销应较小。

---

## 10. 实现顺序建议

### Step 1：扩展 calculator extra stats

目标：

```text
compute_coverage_and_purity() 保持旧接口兼容，
新增 optional extra stats。
```

验收：

```text
旧调用仍能跑通。
新调用能拿到 aligned_covered / distance purity counters。
```

### Step 2：扩展 aggregator counters

目标：

```text
新增 category coverage 和 distance purity 累加器。
```

验收：

```text
finalize() JSON 中出现新字段。
coverage consistency debug 通过。
```

### Step 3：扩展 reporter

目标：

```text
日志表格展示 Cov / AlignCov / Distance-wise Purity。
```

验收：

```text
日志列对齐，数值不换行，JSON 正常保存。
```

### Step 4：小样本验证

建议先用少量验证帧运行：

```text
stat_freq 较小
验证前 10~50 帧
```

检查：

```text
1. coverage_counter_consistent == True
2. category_total_matches_scalar == True
3. all aligned_coverage <= coverage
4. all distancewise_purity in [0, 1]
5. all category coverage in [0, 1]
```

### Step 5：全验证集运行

全量运行后重点观察：

```text
近距离 bin 的 coverage 与 purity 是否高于远距离；
小目标类别的 coverage 是否明显低于大面积类别；
aligned_coverage 与 category purity 是否呈现合理相关性。
```

---

## 11. 预期分析价值

### 11.1 Category-wise Coverage

可以回答：

```text
哪些类别的真实空间结构被 Gaussian 表达不足？
小目标类别是否 coverage 低？
大面积类别是否靠过多 Gaussian 获得高 coverage？
```

结合已有 category-wise purity：

```text
coverage 高、purity 高：该类别表达较好
coverage 高、purity 低：覆盖到了但语义混乱
coverage 低、purity 高：只覆盖了一小部分但类别较准
coverage 低、purity 低：该类别整体困难
```

### 11.2 Distance-wise Non_empty_visible Purity

可以回答：

```text
远距离区域是几何覆盖掉得快，还是语义纯度掉得快？
模型是否在远距离产生大量错误类别 Gaussian 覆盖？
coverage 与 purity 是否随距离同步衰减？
```

结合已有 distance-wise coverage：

```text
Distance bin | Coverage | Purity | 解释
near         | high     | high   | 表达充分且语义正确
middle       | high     | low    | 覆盖仍在，但语义开始混乱
far          | low      | high   | 只覆盖容易区域
far          | low      | low    | 几何和语义都退化
```

---

## 12. 最终推荐字段命名

| 字段 | 推荐名称 | 说明 |
|---|---|---|
| 每类 GT coverage | `category_stats[c].coverage` | 类别 c 的 non_empty_visible GT 体素被任意 Gaussian 覆盖比例 |
| 每类同类 Gaussian coverage | `category_stats[c].aligned_coverage` | 类别 c 的 GT 体素被预测类别 c 的 Gaussian 覆盖比例 |
| 距离 purity | `distancewise_purity` | 每个距离 bin 的 non_empty_visible hit-pair purity |
| 距离 purity debug | `distancewise_purity_debug` | 每个 bin 的 match / total / purity |
| 类别 coverage debug | `category_coverage_debug` | category coverage 与 global coverage 的分母一致性 |

---

## 13. 本次设计的核心原则

```text
1. 不重复计算 coverage，复用已有 hit-pair traversal。
2. 所有新增指标沿用 non_empty_visible scope。
3. category coverage 以 GT 类别为分母，避免预测类别分母不清。
4. aligned coverage 单独输出，避免把几何 coverage 和语义正确性混成一个不可解释指标。
5. distance-wise purity 以 voxel distance 分桶，和 distance-wise coverage 使用同一 bin 口径。
6. 所有大计数器使用 int64，避免之前 float32 饱和问题再次出现。
7. 保持 calculator 旧接口兼容，降低对现有独立函数和脚本的影响。
```
