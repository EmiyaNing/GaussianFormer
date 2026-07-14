# Gaussian Statistic 新增 Mixed-Gaussian 与 Sem-Sup 设计方案

## 1. 目标

在现有 `gaussian_statistic.py` 的 coverage / purity 统计基础上增加两个指标：

1. **Mixed-Gaussian**：所有参与统计的 Gaussian 中，purity 严格小于手动阈值 `rho` 的 Gaussian 比例。
2. **Sem-Sup**：被 Mixed-Gaussian 几何覆盖的 GT occupied voxel，占全部有效 GT occupied voxel 的比例。

新增统计应满足：

- 与现有 purity 使用完全相同的 Gaussian-voxel 椭球命中定义；
- 与现有 `ignore_empty`、可见区域和类别排除参数保持一致；
- voxel 重复覆盖必须去重；
- 不把未覆盖任何 GT voxel 的 Gaussian 错误解释为 mixed；
- 使用 `int64` 计数，支持完整验证集流式累积；
- 不保存全数据集 Gaussian 或 Gaussian-voxel pair。

## 2. 指标定义

### 2.1 单 Gaussian purity

对一帧中第 `g` 个 Gaussian，沿用当前实现：

```text
T_g = Gaussian g 几何覆盖的有效 GT occupied voxel 数
M_g = 上述 voxel 中 GT label 等于 Gaussian argmax semantic class 的数量
```

当 `T_g > 0` 时：

```text
purity_g = M_g / T_g
```

几何覆盖仍由 `cov_threshold=tau` 控制：

```text
Mahalanobis(voxel, Gaussian) <= tau^2
```

因此 `rho` 是语义 purity 阈值，`tau` 是几何覆盖阈值，两者必须分别记录，不能混用。

### 2.2 Mixed-Gaussian

定义：

```text
valid_g = (T_g > 0)
mixed_g = valid_g and (purity_g < rho)
```

用户要求的主指标使用全部参与 purity 统计的 Gaussian 为分母：

```text
Mixed-Gaussian = sum_g mixed_g / G_evaluated
```

其中 `G_evaluated` 是经过 `exclude_gaussian_classes` 过滤后进入 purity 计算的 Gaussian 数量。

同时建议输出条件比例：

```text
Mixed-Gaussian(valid) = sum_g mixed_g / sum_g valid_g
```

原因是 `T_g = 0` 时 purity 在数学上未定义。主指标中 unused Gaussian 保留在分母，但不进入 mixed 分子；valid-only 指标用于区分：

- mixed Gaussian 确实较多；
- 只是 unused Gaussian 较多导致主比例被稀释。

禁止采用以下定义：

```text
unused purity = 0 -> unused 全部判为 mixed
```

否则 Mixed-Gaussian 会退化为 unused ratio 的变体，重现旧 purity 口径错位问题。

阈值比较严格遵循需求使用 `< rho`，不是 `<= rho`。

### 2.3 Sem-Sup

令当前帧有效 GT occupied voxel 集合为 `V_occ`。对每个 voxel `v`：

```text
mixed_covered(v) = 是否存在 mixed Gaussian g，使 g 在几何上覆盖 v
```

则：

```text
Sem-Sup = |{v in V_occ : mixed_covered(v)}| / |V_occ|
```

统计规则：

- 只关心几何覆盖，不要求 Gaussian semantic class 与 voxel GT label 一致；
- 一个 voxel 被多个 Mixed-Gaussian 覆盖仍只计一次；
- 一个 Mixed-Gaussian 覆盖多个 voxel 时分别计数；
- 分母是全部有效 GT occupied voxel，而不是只统计已经被任意 Gaussian 覆盖的 voxel；
- 默认 scope 为 `occ_cam_mask & (label != empty_label)`；
- 如果指定 `exclude_voxel_classes`，被排除类别同时离开分子和分母。

`Sem-Sup` 表示低 purity Gaussian 的几何影响范围，不等价于 mIoU，也不表示这些 voxel 一定预测错误。

## 3. 关于“场景”的聚合口径

当前 `add_frame()` 每次处理一个 dataloader sample，现有代码没有可靠的 scene boundary 聚合逻辑。因此建议同时输出：

```text
sem_sup                        # 数据集级 micro ratio
mean_frame_sem_sup             # frame ratio 的算术平均
```

定义为：

```text
sem_sup = 所有帧 mixed-covered voxel 总数 / 所有帧有效 occupied voxel 总数
mean_frame_sem_sup = mean(frame_mixed_covered / frame_occupied)
```

主结果使用 `sem_sup`，因为它不受每帧 occupied voxel 数量差异影响。文档和日志中应称第二项为 frame-wise，而不要把 frame 误称为完整 NuScenes scene。

如果后续需要真正的 scene-wise 指标，应另行基于 `scene_token` 分组，不能直接复用 `mean_frame_sem_sup` 的名称。

## 4. 参数设计

在 `gaussian_statistic.py` 增加：

```python
parser.add_argument(
    '--purity-threshold-rho',
    type=float,
    default=0.5,
    help='Mixed-Gaussian purity 阈值，严格使用 purity < rho',
)
```

启动时校验：

```python
if not 0.0 <= args.purity_threshold_rho <= 1.0:
    parser.error('--purity-threshold-rho must be in [0, 1]')
```

并传入：

```python
GaussianStatAggregator(
    ...,
    purity_threshold_rho=args.purity_threshold_rho,
)
```

日志开头明确打印：

```text
Purity threshold rho: 0.5
Mixed Gaussian rule: valid purity < rho
```

不建议复用 `t_scale`、`t_sphere` 或 `cov_threshold`，避免参数语义混乱。

## 5. Calculator 实现方案

### 5.1 为什么不能在现有单遍 chunk 内直接算 Sem-Sup

当前 `compute_coverage_and_purity()` 按 voxel chunk 流式更新 `p_match[g]` 和 `p_total[g]`。Gaussian 的最终 purity 只有所有 voxel chunk 处理结束后才能确定。

处理前面的 chunk 时还不知道该 Gaussian 最终是否满足 `purity < rho`，因此不能当场可靠地生成 `mixed_covered`。若保存所有命中 pair 到最后再判断，会产生不可控显存或 CPU 内存开销。

### 5.2 推荐的 bounded two-pass 实现

保留现有第一遍计算：

```python
covered, purity_match, purity_total, extra_stats = compute_coverage_and_purity(...)
```

第一遍结束后构造：

```python
valid_g = purity_total > 0
purity = torch.zeros_like(purity_total, dtype=torch.float32)
purity[valid_g] = (
    purity_match[valid_g].float()
    / purity_total[valid_g].float()
)
mixed_mask = valid_g & (purity < rho)
```

随后新增 calculator 函数：

```python
compute_subset_coverage(
    means,
    precomp,
    voxel_xyz,
    gaussian_mask=mixed_mask,
    cov_threshold=tau,
    chunk_size=chunk_size,
    max_pair_elements=max_pair_elements,
) -> Tensor[N]  # bool
```

第二遍只针对 Mixed-Gaussian 子集计算几何 union coverage：

```text
N x G  ->  N x G_mixed
```

helper 必须用同一个 `gaussian_mask` 同步切片以下张量，避免 Gaussian index 错位：

```text
means
precomp['R']
precomp['scales_vec']
precomp['search_radius']
```

实现继续使用动态 chunk：

```python
effective_chunk = min(
    chunk_size,
    max(1, max_pair_elements // max(G_mixed, 1)),
)
```

第二遍不需要：

- GT semantic comparison；
- `p_match` / `p_total` scatter；
- distance-wise purity；
- aligned coverage。

只需输出每个 voxel 是否被至少一个 mixed Gaussian 命中，因此代码应独立为轻量 coverage-only helper，避免递归调用完整 purity 逻辑。

边界情况：

```python
if mixed_mask.sum() == 0:
    return torch.zeros(N, dtype=torch.bool, device=device)
```

### 5.3 不采用的方案

不建议在第一遍保存全部 `(voxel_index, gaussian_index)` hit pair：

- 大尺度 Gaussian 和 `tau=3` 会产生大量 pair；
- pair 数与场景密度有关，没有稳定上界；
- 会抵消当前 `max_pair_elements` 和流式统计的显存优化。

因此 two-pass 会增加计算量，但保持内存有界，更适合离线统计脚本。

## 6. Aggregator 改动

### 6.1 初始化字段

新增并统一采用以下字段：

```python
self.purity_threshold_rho = float(purity_threshold_rho)

self.total_mixed_gaussians = 0
self.total_mixed_evaluated_gaussians = 0
self.total_purity_defined_gaussians = 0

self.total_sem_sup_covered = 0
self.total_sem_sup_voxels = 0
self.sum_frame_sem_sup = 0.0
self.sem_sup_frame_count = 0
```

### 6.2 每帧累积

在 `_accumulate_cov_purity()` 获得 `purity_match` 和 `purity_total` 后：

```python
valid_g = purity_total > 0
purity_g = torch.zeros_like(purity_total, dtype=torch.float32)
purity_g[valid_g] = (
    purity_match[valid_g].float()
    / purity_total[valid_g].float().clamp_min(1)
)
mixed_g = valid_g & (purity_g < self.purity_threshold_rho)
```

累积 Gaussian 计数：

```python
self.total_mixed_gaussians += mixed_g.sum().item()
self.total_mixed_evaluated_gaussians += purity_total.numel()
self.total_purity_defined_gaussians += valid_g.sum().item()
```

计算 Mixed-Gaussian union coverage：

```python
mixed_covered = calculator.compute_subset_coverage(
    means,
    precomp,
    valid_xyz,
    gaussian_mask=mixed_g,
    cov_threshold=self.cov_threshold,
    chunk_size=self.chunk_size,
    max_pair_elements=self.max_pair_elements,
)
```

累积 Sem-Sup：

```python
frame_total = valid_xyz.shape[0]
frame_covered = mixed_covered.sum().item()

self.total_sem_sup_covered += frame_covered
self.total_sem_sup_voxels += frame_total
self.sum_frame_sem_sup += frame_covered / frame_total
self.sem_sup_frame_count += 1
```

这里 Sem-Sup 分母不应使用 `covered.sum()`，否则指标会变成“已覆盖 voxel 中有多少被 mixed Gaussian 覆盖”，与需求不符。

### 6.3 distance bin 口径

主 Sem-Sup 分母推荐使用全部 `valid_xyz`，不受 `distance_bins` 截断影响，因为需求写的是“全部 GT occupied voxels”。

现有 global coverage 使用 `valid_b` 限制在 distance bins 内。为了避免两个指标看似同 scope 实际不同，输出元数据必须明确：

```text
sem_sup_voxel_scope = non_empty_visible_all_distance
```

若项目希望 Sem-Sup 与现有 global coverage 严格同分母，则统一改为：

```python
frame_mask = valid_b
frame_total = valid_b.sum()
frame_covered = (mixed_covered & valid_b).sum()
```

两种定义不可混用。基于当前文字需求，本文推荐“全部有效 occupied voxel、不按 distance bins 截断”。

## 7. Snapshot、Finalize 与 JSON

### 7.1 实时 snapshot

`get_snapshot()` 新增：

```python
'mixed_gaussian_ratio': _safe_div(
    self.total_mixed_gaussians,
    self.total_mixed_evaluated_gaussians,
),
'mixed_gaussian_valid_ratio': _safe_div(
    self.total_mixed_gaussians,
    self.total_purity_defined_gaussians,
),
'sem_sup': _safe_div(
    self.total_sem_sup_covered,
    self.total_sem_sup_voxels,
),
```

实时日志增加：

```text
Mixed-G(rho=0.50): 0.xxxx | Sem-Sup: 0.xxxx
```

### 7.2 Finalize JSON

推荐结构化输出：

```json
{
  "mixed_gaussian": {
    "purity_threshold_rho": 0.5,
    "comparison": "purity < rho",
    "unused_policy": "excluded_from_numerator_kept_in_all_denominator",
    "mixed_count": 123,
    "evaluated_gaussian_count": 1000,
    "purity_defined_gaussian_count": 600,
    "ratio_all": 0.123,
    "ratio_valid": 0.205
  },
  "sem_sup": {
    "purity_threshold_rho": 0.5,
    "covered_voxel_count": 456,
    "occupied_voxel_count": 2000,
    "ratio": 0.228,
    "mean_frame_ratio": 0.231,
    "voxel_scope": "non_empty_visible_all_distance",
    "coverage_rule": "union_of_geometric_hits_from_mixed_gaussians"
  }
}
```

保留原始 count 很重要，可用于排查不同 run 的 scope、过滤类别和数据量是否一致。

### 7.3 Reporter

最终日志增加：

```text
Mixed-Gaussian (rho=0.500, all):   0.xxxxxx
Mixed-Gaussian (valid-only):       0.xxxxxx
Sem-Sup:                           0.xxxxxx
Mean Frame Sem-Sup:                0.xxxxxx
```

不要只打印一个 `Mixed-Gaussian`，否则无法判断 unused Gaussian 对分母的影响。

## 8. 一致性检查

`finalize()` 增加以下 sanity 字段：

```python
mixed_count <= purity_defined_gaussian_count
purity_defined_gaussian_count <= evaluated_gaussian_count
sem_sup_covered_voxel_count <= sem_sup_occupied_voxel_count
0.0 <= mixed_gaussian_ratio <= 1.0
0.0 <= sem_sup <= 1.0
```

还应验证：

1. `rho = 0` 时，Mixed-Gaussian 和 Sem-Sup 必须为 0。
2. 增大 `rho` 时，Mixed-Gaussian 与 Sem-Sup 应单调不减。
3. Sem-Sup 必须小于等于使用相同 voxel scope 的普通 geometric coverage。
4. `mixed_count + non_mixed_valid_count + unused_count = evaluated_gaussian_count`。
5. 同一 voxel 被多个 mixed Gaussian 覆盖时，Sem-Sup numerator 只能增加 1。
6. 指定 `exclude_gaussian_classes` 后，Gaussian 三类计数的总和必须使用过滤后的 denominator。
7. 指定 `exclude_voxel_classes` 后，Sem-Sup 分子和分母必须同时变化。

如果主 Sem-Sup 采用 all-distance scope，而普通 coverage 采用 distance-bin scope，第 3 条检查必须先统一 mask 后再比较，不能直接比较两个最终 scalar。

## 9. 测试方案

建议新增 CPU 单元测试，使用轴对齐、rotation 为单位四元数的人工 Gaussian：

### Case A：纯 Gaussian

- 一个 Gaussian 覆盖两个同类 voxel；
- `purity=1`；
- `rho=0.5`；
- Mixed-Gaussian=0，Sem-Sup=0。

### Case B：mixed Gaussian

- 一个 class-1 Gaussian 覆盖一个 class-1 和一个 class-2 voxel；
- `purity=0.5`；
- `rho=0.6`；
- Mixed-Gaussian=1，Sem-Sup=1。

### Case C：严格小于

- purity 恰好为 `0.5`；
- `rho=0.5`；
- 该 Gaussian 不应判为 mixed。

### Case D：unused Gaussian

- 两个 Gaussian，其中一个不覆盖任何 voxel；
- unused 不进入 mixed 分子；
- `ratio_all` 与 `ratio_valid` 不同；
- Sem-Sup 不受 unused Gaussian 影响。

### Case E：voxel union 去重

- 两个 mixed Gaussian 覆盖同一个 voxel；
- mixed_count=2；
- Sem-Sup covered count=1。

### Case F：过滤口径

- 排除一个 Gaussian class 和一个 voxel class；
- 检查 Mixed-Gaussian denominator 与 Sem-Sup numerator/denominator同步更新。

### Case G：rho 单调性

- 对同一批输入测试 `rho=[0.2, 0.5, 0.8]`；
- 两个指标都不得下降。

## 10. 文件改动清单

实施阶段预计修改：

| 文件 | 改动 |
| --- | --- |
| `gaussian_statistic.py` | 增加 `--purity-threshold-rho`、参数校验、传入 aggregator、snapshot 日志 |
| `gaussian_statistic/aggregator.py` | Mixed-Gaussian/Sem-Sup 计数、snapshot、finalize 和 sanity check |
| `gaussian_statistic/calculator.py` | 增加 bounded `compute_subset_coverage()` |
| `gaussian_statistic/reporter.py` | 输出两个新指标及原始计数 |
| `tests/test_gaussian_statistic.py` | 人工几何与边界条件测试；若项目暂无 tests 目录则新建 |

## 11. 推荐实施顺序

1. 先实现并测试单帧 `mixed_mask` 定义。
2. 实现 coverage-only 的 `compute_subset_coverage()`，验证 voxel union 去重。
3. 接入 aggregator 的流式 count。
4. 接入 CLI、snapshot、reporter 和 JSON。
5. 用多个 `rho` 在同一 checkpoint 上做单调性验证。
6. 最后跑完整验证集，对比原有 coverage、unused ratio、Mixed-Gaussian 和 Sem-Sup。

## 12. 最终推荐口径

默认建议采用：

```text
rho = 0.5（允许命令行覆盖）
mixed = (purity_total > 0) and (purity < rho)
Mixed-Gaussian 主分母 = 过滤后的全部 Gaussian
Mixed-Gaussian(valid) 分母 = purity 有定义的 Gaussian
Sem-Sup voxel 分母 = visible、non-empty、未被排除的全部 GT occupied voxel
Sem-Sup voxel 分子 = 被至少一个 mixed Gaussian 几何覆盖的 voxel union
```

这套定义既符合需求，也避免将 unused Gaussian、重复 voxel hit 和 distance-bin 截断混入指标含义。
