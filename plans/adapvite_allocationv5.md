# AdaptiveAllocationV5 设计文档

> 目标实现文件：`model/encoder/gaussian_encoder/topk_module/adaptive_allocationv5.py`
>
> 核心定位：V5 先用 TopK 选择最值得被操作的 Gaussian；只有这些 TopK Gaussian 进入 adaptive risky operation bank。进入 bank 后不再允许 keep，只在 `clone / split / attenuation` 三类风险操作之间进行 soft contribution 分配。

## 1. 设计结论

V5 的最终设计选择如下：

| 设计点 | V5 方案 |
| --- | --- |
| 操作对象 | 不是所有 Gaussian，而是 allocation TopK 选出的高价值 Gaussian |
| 非 TopK Gaussian | 旁路保留，不进入 risky operation bank |
| TopK Gaussian | 必须进入风险操作集合，不允许 router 选择 keep |
| operation classes | 只包含 `clone / split / attenuation` 三类 |
| operation probability | 三类 soft probability 直接调制 candidate opacity / contribution |
| 训练/推理一致性 | 训练和推理都使用同一套 soft risky candidate bank |
| auxiliary loss | 不新增 auxiliary loss |
| V4 命名兼容 | 不要求保留 V4 模块名或 head 名 |
| 可视化 | 统计 TopK 内 clone/split/atten 的 soft expected contribution |

一句话概括：

```text
V5 = TopK risky Gaussian selector + clone/split/atten-only soft candidate bank
```

## 2. 与上一版 V5 方案的差异

上一版 V5 仍保留了 soft keep candidate，并倾向兼容 V4 的模块命名。现在根据新的需求，V5 做以下收紧：

```text
去掉 keep operation。
去掉 p_keep。
去掉 V4 模块命名兼容要求。
去掉 V4 checkpoint 结构兼容要求。
只对 TopK Gaussian 生成 risky candidate bank。
TopK 内只允许 clone / split / attenuation。
```

需要特别区分两个概念：

```text
non-TopK pass-through:
  没有被选为值得操作的 Gaussian，直接旁路保留。
  这是预算筛选结果，不是 router 选择 keep。

TopK risky operation:
  被认为值得操作的 Gaussian。
  必须在 clone / split / attenuation 中分配贡献。
```

因此，V5 中不再存在：

```text
keep as an operation class
```

只存在：

```text
non-selected pass-through
```

## 3. 设计目标

### 3.1 功能目标

1. 保持 forward 主输入输出形式与现有 densify layer 尽量兼容。
2. 先通过 TopK 选择最值得被操作的 Gaussian。
3. 只对 TopK Gaussian 构造 risky candidate bank。
4. risky bank 中只包含 `clone / split / attenuation`。
5. operation probability 必须进入最终 Gaussian 输出计算图。
6. 训练和推理不切换 hard routing，始终使用 soft candidate bank。
7. 不新增 auxiliary loss。
8. 不要求 V4 模块命名或 checkpoint 参数名兼容。
9. 支持可视化 soft operation probability 和 candidate contribution。

### 3.2 非目标

V5 不追求：

- 不对所有 Gaussian 生成完整 bank；
- 不把 keep 作为 operation router 的一类；
- 不做 hard operation id 路由；
- 不做 soft-to-hard 训练/推理切换；
- 不新增 router auxiliary loss；
- 不强制 clone/split/atten 均匀分布；
- 不保留 V4 的模块命名约束。

## 4. 高层流程

```text
Input:
  instance_feature
  anchor
  gaussian

        |
        v

Build allocation state
  feature cue + geometry cue + semantic cue

        |
        v

Compute allocation score
  score means "how worth risky operation"

        |
        v

TopK risky Gaussian selection
  selected_mask

        |
        +-------------------------------+
        |                               |
        v                               v
Non-TopK Gaussian                 TopK Gaussian
pass-through original             enter risky operation bank
                                  clone / split / attenuation only

        |                               |
        +---------------+---------------+
                        |
                        v

Assemble output Gaussian bank

        |
        v

Return:
  result_anchors
  result_gaussian
  result_features
```

## 5. 模块层级图

```text
AdaptiveAllocationV5
|
|-- Allocation State Encoder
|   |-- feature cue
|   |-- geometry cue
|   |-- semantic cue
|   |-- allocation state
|
|-- Risky Gaussian Selector
|   |-- allocation score predictor
|   |-- TopK selector
|   |-- selected_mask
|
|-- Risky Operation Gate
|   |-- clone logit
|   |-- split logit
|   |-- attenuation logit
|   |-- clone/split/atten probability
|
|-- Risky Candidate Bank
|   |-- clone parent candidate
|   |-- clone child candidate
|   |-- split child 1 candidate
|   |-- split child 2 candidate
|   |-- attenuation candidate
|
|-- Pass-through Bank
|   |-- non-TopK original Gaussian
|
|-- Output Assembler
|   |-- probability-weighted opacity
|   |-- candidate feature / semantic / geometry
|   |-- batch padding if needed
|
|-- Visualization / Statistics Cache
    |-- selected_mask
    |-- risky operation probability
    |-- expected clone / split / attenuation contribution
    |-- output candidate type labels
```

## 6. 输入输出接口

V5 的 forward 输入保持三项：

```text
instance_feature
anchor
gaussian
```

V5 的 forward 主输出保持三元组：

```text
result_anchors
result_gaussian
result_features
```

这样可以尽量减少对 `GaussianOccEncoder` 和后续 occupancy head 的影响。

V5 内部可以缓存诊断信息：

```text
latest_selected_mask
latest_risky_operation_prob
latest_risky_operation_stats
latest_output_candidate_labels
```

这些缓存只用于可视化、统计和 debug，不改变主 forward 返回值。

## 7. 配置项设计

推荐配置项：

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `allocation_ratio` | 进入 risky bank 的 TopK 比例 | 必须显式配置 |
| `router_temperature` | clone/split/atten softmax 温度 | 1.0 起步 |
| `selector_hidden_dim` | selector/state encoder hidden 维度 | 根据实现需要设置 |
| `operation_hidden_dim` | risky operation gate hidden 维度 | 根据实现需要设置 |
| `cache_router_stats` | 是否缓存可视化统计 | True |
| `opacity_floor` | candidate 最小 opacity contribution，防止极低概率候选完全无梯度 | 可选 |

不再需要：

```text
keep class weight
p_keep
non_candidate_keep_prob
growth_ratio
keep_ceiling_topk
router_aux_loss_weight
use_soft_training
use_budget_router_eval
```

## 8. TopK Risky Gaussian Selector

### 8.1 Selector 的语义

`allocation_score` 不再表示“是否进入一个可能 keep 的候选集合”，而是表示：

```text
当前 Gaussian 是否值得执行风险操作
```

风险操作包括：

```text
clone
split
attenuation
```

TopK selector 的职责是：

```text
从输入 Gaussian 中选出最值得被主动修改的一部分。
```

### 8.2 非 TopK Gaussian 的处理

非 TopK Gaussian 直接旁路输出：

```text
non-TopK Gaussian:
  output original Gaussian
  no clone candidate
  no split candidate
  no attenuation candidate
  no operation probability
```

这不是 keep operation，而是：

```text
not selected for risky operation
```

### 8.3 TopK Gaussian 的处理

TopK Gaussian 必须进入 risky bank：

```text
TopK Gaussian:
  generate clone candidate
  generate split candidate
  generate attenuation candidate
  combine them by clone/split/atten probability
```

TopK Gaussian 不再拥有 keep 分支。

## 9. Risky Operation Gate

### 9.1 三类操作概率

operation gate 只输出三类概率：

```text
p_clone
p_split
p_atten
```

它们在 TopK Gaussian 内归一化：

```text
p_clone + p_split + p_atten = 1
```

这三个概率不是离散选择，而是 risky candidate contribution 的连续权重。

### 9.2 三类操作语义

```text
clone:
  parent contribution remains
  plus one clone child
  suitable for coverage expansion

split:
  parent is replaced by two split children
  suitable for large / mixed / anisotropic Gaussian

attenuation:
  reduce opacity contribution
  suitable for uncertain, redundant, or harmful Gaussian
```

### 9.3 不允许 keep

TopK 内不再有：

```text
p_keep
keep logit
keep candidate
keep_from_router
```

如果一个 Gaussian 被 TopK selector 选中，它必须通过 risky operation bank 对输出产生变化。

## 10. Risky Soft Candidate Bank

### 10.1 Bank 组成

对每个 TopK Gaussian，V5 构造以下 candidates：

```text
clone parent candidate:
  original Gaussian contribution weighted by p_clone

clone child candidate:
  clone branch generated child weighted by p_clone

split child 1 candidate:
  split branch generated child 1 weighted by p_split

split child 2 candidate:
  split branch generated child 2 weighted by p_split

attenuation candidate:
  attenuated Gaussian weighted by p_atten
```

注意：

```text
clone parent candidate 不是 keep。
```

它属于 clone 操作的语义，因为 clone 操作本来就是：

```text
parent retained + child appended
```

### 10.2 输出规模

设输入 Gaussian 数为 N，TopK 数为 K。

输出有效数量为：

```text
non-TopK pass-through:
  N - K

TopK clone parent candidates:
  K

TopK clone child candidates:
  K

TopK split child 1 candidates:
  K

TopK split child 2 candidates:
  K

TopK attenuation candidates:
  K

total:
  N - K + 5K = N + 4K
```

该输出数量上升是 V5 的设计结果。当前阶段不把 Gaussian 增多作为主要风险处理，因为 nuScenes LiDAR 场景中更核心的问题是信息不足。

### 10.3 Candidate contribution

每个 risky candidate 的有效贡献由对应 operation probability 调制：

```text
clone parent opacity:
  original opacity weighted by p_clone

clone child opacity:
  clone child opacity weighted by p_clone

split child 1 opacity:
  split child 1 opacity weighted by p_split

split child 2 opacity:
  split child 2 opacity weighted by p_split

attenuation opacity:
  attenuated opacity weighted by p_atten
```

这样主 occupancy loss 可以通过 candidate opacity 回传到 risky operation gate。

## 11. Candidate 属性策略

### 11.1 几何属性

每个 candidate 使用自身的几何属性：

```text
clone parent:
  original means / scales / rotations

clone child:
  clone branch generated means / scales / rotations

split child 1:
  split branch generated means / scales / rotations

split child 2:
  split branch generated means / scales / rotations

attenuation:
  original means / scales / rotations
```

不建议对 geometry 做概率加权混合。原因：

```text
mean / scale / rotation 的混合会引入额外设计复杂度；
quaternion rotation 混合尤其容易出问题；
直接输出 candidate bank 更清晰，也更容易可视化。
```

### 11.2 Opacity 属性

opacity 是 operation probability 进入计算图的主要通道：

```text
candidate effective opacity
  = candidate raw opacity contribution
  weighted by operation probability
```

这使得：

```text
occupancy loss
  -> local aggregation / rendering
  -> candidate opacity
  -> operation probability
  -> risky operation gate
```

成为稳定的端到端路径。

### 11.3 Feature 与 Semantic 属性

feature 和 semantic 使用各 branch 生成的 candidate 属性：

```text
clone parent:
  original feature / semantic

clone child:
  clone branch feature / semantic

split child 1 / 2:
  split branch feature / semantic

attenuation:
  original feature / semantic
```

第一版不需要额外 feature mixer 或 semantic mixer。

## 12. 与主 Loss 的关系

V5 不新增 auxiliary loss。

router 训练信号来自主任务：

```text
occupancy loss
  -> Gaussian output bank
  -> probability-weighted candidate opacity
  -> p_clone / p_split / p_atten
  -> risky operation gate
```

因此 V5 的必要约束是：

```text
每个 TopK risky candidate 的有效 opacity 必须由对应 operation probability 调制。
```

如果只生成 probability 但不调制输出 contribution，V5 会重新退化成 V4 的问题。

## 13. 模块命名与 V4 的关系

本版 V5 不要求保留 V4 的模块名。

可以复用 V4 的思想：

```text
geometry cue
semantic cue
allocation score
clone branch
split branch
attenuation branch
```

但不要求复用 V4 的具体命名：

```text
state_encoder
score_head
operation_head
clone_dir_head
split_dir_head
...
```

V5 的实现可以采用更符合新语义的命名，例如：

```text
risky_selector
risky_gate
clone_expert
split_expert
atten_expert
```

这样可以避免后续阅读代码时误以为 V5 与 V4 的 hard-router 语义一致。

## 14. 最小实现建议

为了减少代码量，V5 第一版可以遵循以下原则：

```text
只新增 adaptive_allocationv5.py。
不改 V4。
不改 loss。
不改 encoder 主流程。
不引入新的通用 MoE 框架。
不实现 hard route。
不实现 keep expert。
```

实现内部可以借鉴 V4 的 branch 计算逻辑，但用新的类名和 helper 名组织。

建议实现顺序：

```text
1. 构造 allocation state。
2. 用 risky selector 得到 TopK selected_mask。
3. 对 selected Gaussian 计算 risky operation probability。
4. 对 selected Gaussian 生成 clone / split / attenuation candidates。
5. 用 p_clone / p_split / p_atten 调制 candidate opacity。
6. 拼接 non-TopK pass-through Gaussian 与 TopK risky candidates。
7. 缓存 soft stats 和 candidate labels。
```

## 15. 可视化与统计

### 15.1 统计口径变化

V4 统计回答：

```text
多少输入 Gaussian 被分到 keep / clone / split / attenuation？
```

V5 统计回答：

```text
多少 Gaussian 被 TopK selector 选中？
TopK 内 clone / split / attenuation 的期望贡献是多少？
输出 bank 中各 candidate type 的有效 opacity 贡献是多少？
```

V5 不再统计：

```text
keep_from_router
routed_keep
p_keep
```

V5 应统计：

```text
input_gaussians
selected_topk
non_topk_pass_through

expected_clone
expected_split
expected_atten

topk_p_clone_mean
topk_p_split_mean
topk_p_atten_mean

effective_opacity_clone_parent
effective_opacity_clone_child
effective_opacity_split_child_1
effective_opacity_split_child_2
effective_opacity_atten

output_candidate_count
```

### 15.2 可视化模式

推荐两种可视化模式。

模式 A：candidate type color

```text
non-TopK pass-through:
  灰色

clone parent / clone child:
  蓝色

split child 1 / split child 2:
  红色

attenuation candidate:
  黄色
```

模式 B：TopK soft risky color

```text
TopK Gaussian color
  = p_clone * clone_color
  + p_split * split_color
  + p_atten * atten_color
```

该模式只用于展示被 TopK 选中的 Gaussian 的 risky operation 倾向。

### 15.3 与现有可视化 wrapper 的关联

第一阶段不强制修改现有 wrapper。

V5 模块内部缓存：

```text
latest_selected_mask
latest_risky_operation_prob
latest_output_candidate_labels
latest_risky_operation_stats
```

后续可视化 wrapper 只需扩展识别：

```text
AdaptiveAllocationV5
```

并读取上述缓存，即可绘制 V5 的 soft risky operation 分布。

## 16. 训练与推理一致性

V5 的训练和推理流程一致：

```text
TopK risky selection
clone/split/atten probability
risky candidate generation
probability-weighted opacity contribution
output candidate bank
```

不包含：

```text
training soft / inference hard
hard operation id
hard budget routing
keep fallback inside TopK
```

## 17. 风险与处理策略

### 17.1 TopK 选错 Gaussian

如果 selector 早期选错 Gaussian，risky bank 可能操作不该操作的目标。

缓解思路：

```text
使用较保守的 allocation_ratio 起步；
观察 selected_topk 的空间分布；
可视化 selected_mask；
必要时降低 selector 学习率或短期冻结部分主干。
```

### 17.2 clone/split/atten 概率塌缩

V5 不新增 auxiliary loss，因此三类 risky probability 仍可能偏向某一类。

但与 V4 不同：

```text
三类 probability 都直接调制输出 candidate opacity；
只要某类 candidate 对 occupancy 有帮助，主 loss 可以直接调整 risky gate。
```

如果出现过早塌缩，可考虑模块内部非 loss 策略：

```text
提高 router_temperature；
使用较小 opacity floor；
降低 risky gate 初期 sharpness。
```

### 17.3 输出 Gaussian 数量增加

输出数量约为：

```text
N + 4K
```

这是 V5 的显式设计，不作为第一阶段主要风险处理。当前任务中，nuScenes LiDAR 信息不足更关键，更多 candidate 有助于补充可能的覆盖。

如果后续确实出现显存或速度瓶颈，再考虑：

```text
降低 allocation_ratio；
对极低有效 opacity candidate 做轻量过滤；
只在可视化时保留完整 candidate labels。
```

## 18. 实现检查清单

实现 `adaptive_allocationv5.py` 后，建议检查：

```text
接口检查:
  forward 输入输出与现有 densify layer 兼容。

TopK 检查:
  selected_topk 数量等于 allocation_ratio * N。
  non_topk_pass_through 数量正确。

Operation 检查:
  TopK 内没有 keep probability。
  risky probability 只有 clone / split / attenuation 三类。
  p_clone + p_split + p_atten = 1。

计算图检查:
  risky operation probability 参与 candidate opacity。
  risky gate 梯度不为 None。
  risky gate 梯度均值不是长期接近 0。

输出规模检查:
  有效输出数量约为 N + 4K。
  batch padding 逻辑正常。

统计检查:
  expected clone / split / attenuation 能输出。
  non_topk_pass_through 能输出。
  不再出现 keep_from_router。

可视化检查:
  selected_mask 可显示。
  candidate type color 可显示。
  TopK soft risky color 可显示。
```

## 19. 最终方案摘要

V5 应坚持以下原则：

```text
1. TopK 先选择最值得被风险操作的 Gaussian。
2. 非 TopK Gaussian 旁路保留，不属于 keep operation。
3. TopK Gaussian 只能进入 clone / split / attenuation soft bank。
4. 不设计 p_keep，不设计 keep expert，不统计 keep_from_router。
5. operation probability 必须调制 candidate opacity，进入主计算图。
6. 不新增 auxiliary loss。
7. 不要求保留 V4 模块名。
```

最终结构是：

```text
input Gaussian
  -> risky TopK selector
  -> selected Gaussian enter clone/split/atten bank
  -> p_clone / p_split / p_atten weight candidate contribution
  -> output pass-through + risky candidate Gaussian bank
```

这样可以直接避免 V4 中 router 大量选择 keep 的问题，因为 V5 的 operation router 语义中已经没有 keep 这一类。

