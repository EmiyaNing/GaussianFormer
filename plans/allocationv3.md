# AllocationV3：Budget-constrained Hard Routing Adaptive Gaussian Allocation Module

## 1. 模块定位

本文档整理 `AllocationV3` 模块的代码设计方案。该模块面向 **Learning Adaptive Semantic Gaussian Allocation for 3D Occupancy Prediction / SAGFormer**，目标是在给定 semantic 3D Gaussian primitives 的基础上，根据每个 Gaussian 的几何、语义和覆盖状态，选择固定比例的 Gaussian 进行自适应分配，并通过硬路由决定其执行 `keep`、`clone`、`split` 或 `opacity-modulate` 操作。

该版本的核心设计原则是：

1. **Budget-constrained TopK selection**：由用户手动设置 allocation ratio，例如 40% 或 60%，模块只选择 TopK Gaussian 进入 adaptive allocation，从而保证 primitive budget 可控。
2. **Hard routing consistency**：训练和推理均使用相同的 hard TopK + hard operation routing，不再使用 soft-hard alignment，避免训练-推理路由偏差。
3. **Operation-specific allocation**：明确区分 `clone` 和 `split` 的功能。`clone` 用于提升 under-covered 区域的覆盖能力，`split` 用于提升 semantically impure 或 geometrically over-expanded Gaussian 的语义纯度。
4. **5N candidate bank implementation**：为每个 Gaussian 预先生成最多五类候选结果，并通过 active mask 选择实际生效的 candidates，保证 GPU 并行友好和多卡训练中的 shape 稳定。

---

## 2. 输入与输出定义

假设当前场景中有 `N` 个 semantic Gaussian primitives：

```python
G = {
    "mean":      Tensor[B, N, 3],      # Gaussian center, mu
    "scale":     Tensor[B, N, 3],      # Gaussian scale, sx, sy, sz
    "rotation":  Tensor[B, N, 4],      # Quaternion or rotation representation
    "opacity":   Tensor[B, N, 1],      # Gaussian opacity
    "semantic":  Tensor[B, N, C],      # Semantic logits or semantic embedding
    "feature":   Tensor[B, N, D],      # Gaussian hidden feature
}
```

模块输出为新的 Gaussian candidate bank 以及 active mask：

```python
G_bank = {
    "mean":      Tensor[B, 5N, 3],
    "scale":     Tensor[B, 5N, 3],
    "rotation":  Tensor[B, 5N, 4],
    "opacity":   Tensor[B, 5N, 1],
    "semantic":  Tensor[B, 5N, C],
    "feature":   Tensor[B, 5N, D],
}

active_mask = Tensor[B, 5N]  # bool or float mask
```

后续 Gaussian splatting / voxel rendering 仅对 `active_mask == 1` 的 candidates 生效。为了避免动态 shape，建议在实现中始终保持 `5N` 个 candidates，并通过 `active_mask` 或 opacity mask 控制有效项。

---

## 3. 总体流程

`AllocationV3` 的前向流程如下：

```text
Input: Gaussian set G, allocation ratio rho
Output: Candidate Gaussian bank G_bank and active_mask

1. Extract multi-cue allocation states for each Gaussian.
2. Predict allocation score a_i for each Gaussian.
3. Select K = floor(rho * N) Gaussians using budget-constrained TopK.
4. For each selected Gaussian, predict operation distribution over:
   {keep, clone, split, opacity-modulate}.
5. Use hard argmax to determine the operation.
6. Generate operation-specific candidates.
7. Build a 5N candidate bank and active mask.
8. Feed active candidates into Gaussian splatting / occupancy prediction.
```

训练和推理阶段完全使用同一套流程：

```text
Training:  hard TopK + hard operation routing
Inference: hard TopK + hard operation routing
```

不引入 soft routing，不使用 soft-hard alignment。

---

## 4. Multi-cue Allocation State Encoding

每个 Gaussian 的 allocation state 由四部分组成：

```python
z_i = concat(
    gaussian_feature_i,
    geometry_cue_i,
    semantic_cue_i,
    coverage_cue_i,
)
```

### 4.1 Geometry cues

Geometry cues 用于衡量 Gaussian 的几何形态是否合理：

```python
mean_scale = scale.mean(dim=-1, keepdim=True)          # [B, N, 1]
max_scale = scale.max(dim=-1, keepdim=True).values     # [B, N, 1]
min_scale = scale.min(dim=-1, keepdim=True).values     # [B, N, 1]
anisotropy_ratio = max_scale / (min_scale + eps)       # [B, N, 1]
log_volume = torch.log(scale.prod(dim=-1, keepdim=True) + eps)
```

推荐 geometry cue：

```python
geometry_cue = concat(
    scale,              # [B, N, 3]
    mean_scale,         # [B, N, 1]
    log_volume,         # [B, N, 1]
    anisotropy_ratio,   # [B, N, 1]
)
```

这些特征主要用于判断：

- Gaussian 是否 scale 过大；
- Gaussian 是否 anisotropy 不合理；
- Gaussian 是否可能覆盖多个结构区域；
- 当前 Gaussian 是否适合执行 `split`。

### 4.2 Semantic cues

Semantic cues 用于衡量 Gaussian 的语义不确定性和语义混杂程度：

```python
prob = semantic_logits.softmax(dim=-1)                 # [B, N, C]
entropy = -(prob * (prob + eps).log()).sum(dim=-1, keepdim=True)
top2_prob = prob.topk(k=2, dim=-1).values              # [B, N, 2]
margin = top2_prob[..., 0:1] - top2_prob[..., 1:2]      # [B, N, 1]
confidence = top2_prob[..., 0:1]                       # [B, N, 1]
```

推荐 semantic cue：

```python
semantic_cue = concat(
    entropy,
    margin,
    confidence,
)
```

这些特征主要用于判断：

- Gaussian 是否语义不确定；
- Gaussian 是否可能跨越语义边界；
- 当前 Gaussian 是否适合执行 `split`；
- 当前 Gaussian 是否适合被 `keep`。

### 4.3 Coverage cues

Coverage cues 用于衡量 Gaussian 附近的局部覆盖需求。注意：**推理阶段不能使用 GT coverage**。因此 controller 的输入必须使用 inference-available proxy，例如：

- 当前 Gaussian 渲染得到的局部 accumulated opacity；
- 局部 occupancy confidence；
- 局部 uncertainty；
- 局部 residual feature；
- 距离或投影邻域中的 predicted coverage score。

推荐接口：

```python
coverage_cue = coverage_encoder(
    gaussian_mean=mean,
    gaussian_feature=feature,
    rendered_opacity=rendered_opacity_proxy,
    occupancy_confidence=occupancy_confidence_proxy,
    local_uncertainty=local_uncertainty_proxy,
)
```

如果当前阶段暂时没有可靠 coverage proxy，可以先将 coverage cue 设计为可选输入：

```python
if coverage_cue is None:
    coverage_cue = torch.zeros(B, N, cov_dim, device=feature.device)
```

Coverage cues 主要用于判断：

- Gaussian 周围是否存在 under-covered 区域；
- 当前 Gaussian 是否适合执行 `clone`；
- 当前 Gaussian 是否应该进入 TopK allocation candidates。

---

## 5. Allocation Score Predictor

将多源 cue 编码为 allocation feature：

```python
z = concat(feature, geometry_cue, semantic_cue, coverage_cue)
h = allocation_state_encoder(z)  # [B, N, D_alloc]
```

预测 allocation score：

```python
allocation_score = sigmoid(score_head(h))  # [B, N, 1]
```

该 score 表示每个 Gaussian 进入 adaptive allocation candidate set 的必要性。

---

## 6. Budget-constrained TopK Selector

用户设置 allocation ratio：

```python
rho = 0.4  # or 0.6, manually configured by user
K = int(rho * N)
```

使用 hard TopK 选择 candidates：

```python
topk_score, topk_idx = allocation_score.squeeze(-1).topk(K, dim=1)
selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
selected_mask.scatter_(dim=1, index=topk_idx, value=True)
```

该设计保证：

1. 每个 batch 中进入 adaptive allocation 的 Gaussian 数量固定；
2. 不会出现无限制 clone / split；
3. 可以公平比较不同 allocation strategies；
4. 训练和推理的 candidate selection 行为一致。

---

## 7. Operation Router

对每个 Gaussian 预测 operation distribution：

```python
op_logits = operation_head(h)              # [B, N, 4]
op_prob = op_logits.softmax(dim=-1)        # [B, N, 4]
op_id = op_prob.argmax(dim=-1)             # [B, N]
```

操作 ID 约定：

```python
KEEP = 0
CLONE = 1
SPLIT = 2
ATTEN = 3  # opacity-modulate / attenuation
```

对于未进入 TopK 的 Gaussian，强制执行 `KEEP`：

```python
op_id = torch.where(selected_mask, op_id, torch.full_like(op_id, KEEP))
```

### 7.1 Hard routing consistency

该版本不使用 soft-hard alignment。训练和推理均采用：

```text
hard TopK + hard argmax operation routing
```

如果担心 hard argmax 阻碍 operation router 的训练，可以采用 surrogate gradient 技术，但前向行为必须保持 hard routing。例如：

```python
# Optional straight-through one-hot operation mask
op_onehot_hard = F.one_hot(op_id, num_classes=4).float()
op_onehot_soft = op_prob
op_onehot = op_onehot_hard.detach() - op_onehot_soft.detach() + op_onehot_soft
```

该设计的前向仍然是 hard operation，反向可从 soft probability 获得梯度。

---

## 8. Operation-specific Candidate Generation

每个 Gaussian 预先生成五个 candidate slots：

```text
slot 0: keep / original
slot 1: clone child
slot 2: split positive child
slot 3: split negative child
slot 4: opacity-modulated candidate
```

因此总 candidate bank 大小为 `5N`。

---

## 9. Keep Operation

### 9.1 适用场景

`keep` 用于 well-allocated primitives：

- semantic purity 高；
- coverage 已经充分；
- scale / anisotropy 合理；
- 当前 Gaussian 对 occupancy prediction 有稳定贡献。

### 9.2 操作定义

```python
G_keep = G_i
```

为了突出 allocation 而非 refinement，建议 `keep` 默认不改变 Gaussian 参数。

---

## 10. Clone Operation：Coverage-oriented Expansion

### 10.1 适用场景

`clone` 用于 **semantically consistent but under-covered regions**。

典型场景：

- small object 周围 coverage 不足；
- distant region 表示稀疏；
- occluded but semantically consistent structure；
- 当前 Gaussian 语义较纯，但周围仍有 coverage demand。

### 10.2 操作目标

`clone` 的目标是增加局部覆盖，而不是分解语义混杂。因此：

- 原 Gaussian 保留；
- 新 Gaussian 在邻近区域生成；
- clone child 的 scale 不应大幅缩小；
- clone child 的 semantic / feature 与原 Gaussian 接近，但允许 residual adjustment。

### 10.3 参数生成

```python
clone_dir = normalize(clone_dir_head(h))                # [B, N, 3]
clone_dist = sigmoid(clone_dist_head(h)) * mean_scale   # [B, N, 1]
clone_scale_ratio = 0.7 + 0.3 * sigmoid(clone_scale_head(h))
```

### 10.4 Candidate 定义

```python
mean_clone = mean + clone_dist * clone_dir
scale_clone = scale * clone_scale_ratio
rotation_clone = rotation
opacity_clone = opacity * clone_opacity_ratio
semantic_clone = semantic + semantic_residual_clone
feature_clone = feature + feature_residual_clone
```

其中 `clone_scale_ratio` 推荐范围为 `[0.7, 1.0]`。

### 10.5 输出规则

如果某 Gaussian 执行 `clone`，则激活：

```text
original keep slot + clone child slot
```

净增加 1 个 Gaussian。

---

## 11. Split Operation：Purity-oriented Decomposition

### 11.1 适用场景

`split` 用于 **semantically impure or geometrically over-expanded primitives**。

典型场景：

- 一个 Gaussian 覆盖多个 semantic regions；
- semantic entropy 高；
- semantic purity 低；
- Gaussian scale 过大；
- Gaussian shape 与局部结构不匹配；
- semantic boundary 附近需要更细粒度 primitives。

### 11.2 操作目标

`split` 的目标是提升语义纯度和边界表达能力。因此：

- 原 Gaussian 不保留；
- 生成两个更小的 child Gaussians；
- 两个 child 沿 split direction 分离；
- child scale 应明显小于原 Gaussian。

### 11.3 Split direction

Split direction 可以结合 Gaussian 最大尺度方向与学习方向：

```python
axis_dir = get_max_scale_axis(rotation, scale)       # [B, N, 3]
learned_dir = normalize(split_dir_head(h))           # [B, N, 3]
split_dir = normalize(axis_dir + learned_dir)
```

其中 `axis_dir` 用于提供几何先验，`learned_dir` 用于适配语义边界和局部上下文。

### 11.4 参数生成

```python
split_dist = sigmoid(split_dist_head(h)) * mean_scale
split_scale_ratio = 0.45 + 0.30 * sigmoid(split_scale_head(h))
```

`split_scale_ratio` 推荐范围为 `[0.45, 0.75]`。

### 11.5 Candidate 定义

```python
mean_split_pos = mean + split_dist * split_dir
mean_split_neg = mean - split_dist * split_dir

scale_split_pos = scale * split_scale_ratio
scale_split_neg = scale * split_scale_ratio

rotation_split_pos = rotation
rotation_split_neg = rotation

opacity_split_pos = opacity * split_opacity_ratio_pos
opacity_split_neg = opacity * split_opacity_ratio_neg

semantic_split_pos = semantic + semantic_residual_pos
semantic_split_neg = semantic + semantic_residual_neg

feature_split_pos = feature + feature_residual_pos
feature_split_neg = feature + feature_residual_neg
```

### 11.6 输出规则

如果某 Gaussian 执行 `split`，则激活：

```text
split positive child slot + split negative child slot
```

原 Gaussian 不激活。净增加 1 个 Gaussian。

### 11.7 与 clone 的区别

`clone` 与 `split` 必须在设计和论文表述中明确区分：

| Operation | 核心目标 | 原 Gaussian | Child scale | 适用问题 |
|---|---|---|---|---|
| Clone | 提升 coverage | 保留 | 轻微缩小或接近原 scale | under-covered but semantically consistent regions |
| Split | 提升 purity | 替换 | 明显缩小 | semantically impure or geometrically over-expanded primitives |

关键表述：

> Clone increases local coverage while preserving semantic consistency, whereas split decomposes semantically ambiguous or geometrically over-expanded primitives into smaller and purer ones.

---

## 12. Opacity-modulate Operation：Redundancy Suppression

### 12.1 适用场景

`opacity-modulate` 用于 redundant or low-contribution primitives：

- simple regions 中过度分配的 Gaussian；
- 低贡献或 noisy primitives；
- 预测不稳定但不适合直接删除的 Gaussian。

### 12.2 操作定义

不引入额外 existence score，只调制 opacity：

```python
m = m_min + (1 - m_min) * sigmoid(opacity_mod_head(h))
opacity_atten = opacity * m
```

推荐：

```python
m_min = 0.05 or 0.1
```

### 12.3 Candidate 定义

```python
mean_atten = mean
scale_atten = scale
rotation_atten = rotation
opacity_atten = opacity * m
semantic_atten = semantic
feature_atten = feature
```

### 12.4 输出规则

如果某 Gaussian 执行 `opacity-modulate`，则激活：

```text
opacity-modulated slot
```

原 Gaussian 不激活。Gaussian 数量不增加。

---

## 13. Candidate Bank 与 Active Mask

### 13.1 Candidate bank layout

对所有 Gaussian 构建 `5N` candidate bank：

```python
bank_mean = torch.cat([
    mean_keep,
    mean_clone,
    mean_split_pos,
    mean_split_neg,
    mean_atten,
], dim=1)  # [B, 5N, 3]
```

其他属性同理：

```python
bank_scale
bank_rotation
bank_opacity
bank_semantic
bank_feature
```

### 13.2 Active mask 生成

每个 Gaussian 对应五个 slot：

```python
keep_mask  = (op_id == KEEP)
clone_mask = (op_id == CLONE)
split_mask = (op_id == SPLIT)
atten_mask = (op_id == ATTEN)
```

激活规则：

```python
active_keep      = keep_mask | clone_mask
active_clone     = clone_mask
active_split_pos = split_mask
active_split_neg = split_mask
active_atten     = atten_mask
```

最终：

```python
active_mask = torch.cat([
    active_keep,
    active_clone,
    active_split_pos,
    active_split_neg,
    active_atten,
], dim=1)  # [B, 5N]
```

对于未激活 candidates，可以在 splatting 前处理：

```python
bank_opacity = bank_opacity * active_mask.unsqueeze(-1).float()
```

或在 Gaussian rendering / splatting 中直接使用 `active_mask`。

---

## 14. 输出 Gaussian 数量与计算预算

虽然 candidate bank 为 `5N`，但实际生效 Gaussian 数量上界为：

```text
N + K = (1 + rho)N
```

因为：

- `keep`：不增加；
- `clone`：原 Gaussian + clone child，净增加 1；
- `split`：两个 child 替换原 Gaussian，净增加 1；
- `opacity-modulate`：不增加。

示例：

| allocation ratio rho | TopK 数量 | 有效 Gaussian 数量上界 |
|---:|---:|---:|
| 0.2 | 0.2N | 1.2N |
| 0.4 | 0.4N | 1.4N |
| 0.6 | 0.6N | 1.6N |
| 0.8 | 0.8N | 1.8N |

临时 `5N` bank 主要用于统一实现，并不代表最终所有 candidates 都参与渲染。

---

## 15. Training Objectives

### 15.1 Occupancy loss

基础 occupancy loss：

```python
L_occ = L_ce + L_lovasz
```

### 15.2 Coverage-guided objective

Coverage objective 用于提升 scene-level Gaussian coverage，尤其是 under-covered occupied regions。

建议形式：

```python
L_cov = coverage_loss(predicted_coverage, target_coverage)
```

其中 target coverage 可以在训练阶段基于 GT occupied voxels 构造，推理阶段不使用 GT。

### 15.3 Semantic-purity objective

Purity objective 用于减少一个 Gaussian 覆盖多个 semantic regions 的情况。

建议形式：

```python
L_pur = semantic_purity_loss(gaussian_semantic_distribution, local_voxel_labels)
```

或者使用 soft semantic consistency：

```python
L_pur = entropy_or_kl_regularization(local_semantic_distribution)
```

### 15.4 Allocation ranking loss

由于 hard TopK 不可导，建议使用轻量 ranking loss 训练 allocation score。

构造 training-only allocation demand target：

```python
d_i = lambda_cov * d_cov_i + lambda_sem * d_sem_i + lambda_geo * d_geo_i
```

其中：

- `d_cov_i`：局部 under-coverage 程度；
- `d_sem_i`：semantic ambiguity / low purity 程度；
- `d_geo_i`：scale / anisotropy mismatch 程度。

Listwise ranking loss：

```python
L_rank = KL(
    softmax(d / tau),
    softmax(allocation_score / tau)
)
```

或者 pairwise ranking loss：

```python
L_rank = max(0, margin - (score_i - score_j)), if d_i > d_j
```

### 15.5 Operation prior loss（可选）

为了强化 clone 和 split 的语义分工，可加入弱 operation prior：

```python
t_clone ∝ d_cov * purity_proxy
t_split ∝ d_sem + d_geo
t_atten ∝ redundancy_score
t_keep  ∝ 1 - d_total
```

得到 soft target：

```python
t_op = normalize([t_keep, t_clone, t_split, t_atten])
```

使用：

```python
L_op = KL(t_op, op_prob)
```

该 loss 权重建议较小，例如：

```python
lambda_op = 0.05 ~ 0.2
```

### 15.6 Score-operation consistency loss（可选）

为了避免 TopK 选入的 Gaussian 大量执行 `keep`，可加入：

```python
L_cons = |allocation_score - (1 - p_keep)|
```

### 15.7 总损失

推荐主文版本：

```python
L_total = (
    L_occ
    + lambda_cov * L_cov
    + lambda_pur * L_pur
    + lambda_rank * L_rank
)
```

完整实现版本：

```python
L_total = (
    L_occ
    + lambda_cov * L_cov
    + lambda_pur * L_pur
    + lambda_rank * L_rank
    + lambda_op * L_op
    + lambda_cons * L_cons
)
```

---

## 16. 推荐 PyTorch 类结构

```python
class AllocationV3(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        semantic_dim: int,
        coverage_dim: int,
        hidden_dim: int = 256,
        allocation_ratio: float = 0.4,
        m_min: float = 0.05,
        use_straight_through: bool = True,
    ):
        super().__init__()
        self.allocation_ratio = allocation_ratio
        self.m_min = m_min
        self.use_straight_through = use_straight_through

        geo_dim = 3 + 1 + 1 + 1  # scale + mean_scale + log_volume + AR
        sem_cue_dim = 3           # entropy + margin + confidence
        input_dim = feature_dim + geo_dim + sem_cue_dim + coverage_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.score_head = nn.Linear(hidden_dim, 1)
        self.operation_head = nn.Linear(hidden_dim, 4)

        self.clone_dir_head = nn.Linear(hidden_dim, 3)
        self.clone_dist_head = nn.Linear(hidden_dim, 1)
        self.clone_scale_head = nn.Linear(hidden_dim, 1)
        self.clone_opacity_head = nn.Linear(hidden_dim, 1)
        self.clone_feature_head = nn.Linear(hidden_dim, feature_dim)
        self.clone_semantic_head = nn.Linear(hidden_dim, semantic_dim)

        self.split_dir_head = nn.Linear(hidden_dim, 3)
        self.split_dist_head = nn.Linear(hidden_dim, 1)
        self.split_scale_head = nn.Linear(hidden_dim, 1)
        self.split_opacity_head = nn.Linear(hidden_dim, 2)
        self.split_feature_head = nn.Linear(hidden_dim, 2 * feature_dim)
        self.split_semantic_head = nn.Linear(hidden_dim, 2 * semantic_dim)

        self.opacity_mod_head = nn.Linear(hidden_dim, 1)

    def forward(self, gaussian, coverage_cue=None):
        # 1. Extract gaussian attributes
        mean = gaussian["mean"]
        scale = gaussian["scale"]
        rotation = gaussian["rotation"]
        opacity = gaussian["opacity"]
        semantic = gaussian["semantic"]
        feature = gaussian["feature"]

        B, N, _ = mean.shape

        # 2. Build cues
        geometry_cue = self.build_geometry_cue(scale)
        semantic_cue = self.build_semantic_cue(semantic)
        if coverage_cue is None:
            coverage_cue = torch.zeros(B, N, self.coverage_dim, device=mean.device)

        z = torch.cat([feature, geometry_cue, semantic_cue, coverage_cue], dim=-1)
        h = self.state_encoder(z)

        # 3. Allocation score and TopK
        score = torch.sigmoid(self.score_head(h)).squeeze(-1)
        K = int(self.allocation_ratio * N)
        _, topk_idx = torch.topk(score, k=K, dim=1)
        selected_mask = torch.zeros(B, N, dtype=torch.bool, device=mean.device)
        selected_mask.scatter_(1, topk_idx, True)

        # 4. Operation routing
        op_logits = self.operation_head(h)
        op_prob = op_logits.softmax(dim=-1)
        op_id = op_prob.argmax(dim=-1)
        op_id = torch.where(selected_mask, op_id, torch.zeros_like(op_id))

        # 5. Generate candidate bank
        bank, active_mask = self.generate_candidates(
            gaussian=gaussian,
            h=h,
            op_id=op_id,
        )

        return {
            "gaussian_bank": bank,
            "active_mask": active_mask,
            "allocation_score": score,
            "operation_logits": op_logits,
            "operation_prob": op_prob,
            "operation_id": op_id,
            "selected_mask": selected_mask,
        }
```

---

## 17. 实现注意事项

### 17.1 多卡训练稳定性

不要在不同 GPU 上动态生成不同长度的 Gaussian list。推荐始终使用：

```python
[B, 5N, D]
```

的 padded candidate bank，并使用：

```python
[B, 5N]
```

的 active mask 控制有效 Gaussian。

### 17.2 避免 GT leakage

Controller 输入中的 coverage cue 不能使用 GT。GT coverage 只能用于：

- training objective；
- allocation demand target；
- ranking loss；
- 训练期辅助监督。

推理阶段所有 controller 输入必须由模型当前预测或可观测输入构造。

### 17.3 Hard routing 与梯度

如果直接 hard argmax 导致 operation router 难以学习，可以使用 straight-through trick。但前向必须保持 hard routing，以维持训练推理一致。

### 17.4 Ratio 设置

建议默认：

```python
allocation_ratio = 0.4 or 0.6
```

并在实验中报告 sensitivity：

```text
rho = 0.2, 0.4, 0.6, 0.8
```

### 17.5 Clone / split 输出数量

`clone` 和 `split` 都最多使 Gaussian 数量净增加 1，因此有效 Gaussian 数量上界为：

```python
N + floor(rho * N)
```

但由于使用 candidate bank，实现上无需动态拼接。

---

## 18. 建议消融实验

### 18.1 TopK ratio ablation

| Ratio rho | Effective Gaussian upper bound | mIoU | IoU | Purity | Coverage |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 1.2N | - | - | - | - |
| 0.4 | 1.4N | - | - | - | - |
| 0.6 | 1.6N | - | - | - | - |
| 0.8 | 1.8N | - | - | - | - |

### 18.2 Operation ablation

| Variant | Clone | Split | Opacity-modulate | mIoU | Purity | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| only clone | ✓ | ✗ | ✗ | - | - | - |
| only split | ✗ | ✓ | ✗ | - | - | - |
| clone + split | ✓ | ✓ | ✗ | - | - | - |
| full | ✓ | ✓ | ✓ | - | - | - |

### 18.3 Routing consistency ablation

| Variant | Training routing | Inference routing | mIoU | Notes |
|---|---|---|---:|---|
| soft train / hard test | soft | hard | - | train-test discrepancy |
| soft-hard alignment | mixed | hard | - | possible inference bias |
| hard train / hard test | hard | hard | - | proposed |

### 18.4 Heuristic baseline comparison

| Method | Selection | Operation | Learned? | mIoU | Purity | Coverage |
|---|---|---|---:|---:|---:|---:|
| Random TopK | random | learned or fixed | partially | - | - | - |
| Scale TopK | scale | heuristic | ✗ | - | - | - |
| Entropy TopK | entropy | heuristic | ✗ | - | - | - |
| Coverage TopK | coverage proxy | heuristic | ✗ | - | - | - |
| AllocationV3 | learned score | learned hard routing | ✓ | - | - | - |

---

## 19. 推荐论文表述

可以在方法部分这样描述该模块：

> Given a set of semantic Gaussian primitives, AllocationV3 first predicts an allocation necessity score for each primitive by jointly encoding its geometric, semantic, and coverage cues. A budget-constrained TopK selector then selects a fixed ratio of primitives as adaptive allocation candidates, ensuring controllable primitive growth and fair comparison under the same allocation budget. For each selected primitive, an operation router determines one of four actions, including keep, clone, split, and opacity attenuation. Unlike soft routing strategies that introduce discrepancies between training and inference, AllocationV3 adopts deterministic hard TopK selection and hard operation routing in both stages. Moreover, clone and split are explicitly decoupled: clone expands semantically consistent but under-covered regions, while split decomposes semantically ambiguous or geometrically over-expanded primitives into smaller and purer ones.

---

## 20. 总结

`AllocationV3` 是一个面向 semantic Gaussian occupancy prediction 的 budget-constrained hard routing adaptive allocation module。它通过 learned allocation score 选择固定比例 Gaussian 进入 adaptive allocation，再通过 hard operation routing 决定具体操作，并通过 5N candidate bank 实现稳定、高效、GPU 友好的并行计算。

该模块的关键优势包括：

1. **预算可控**：通过 allocation ratio 控制参与自适应分配的 Gaussian 数量。
2. **训练推理一致**：使用 hard TopK + hard argmax，避免 soft-hard alignment 带来的推理偏差。
3. **操作语义清晰**：clone 负责 coverage，split 负责 purity，opacity-modulate 负责 redundancy suppression。
4. **实现稳定**：使用 5N candidate bank 和 active mask，避免 DDP 下动态 shape 问题。
5. **论文叙事强**：能够自然支撑 “semantic Gaussian allocation is a learnable, budget-constrained, operation-specific decision process” 这一核心观点。
