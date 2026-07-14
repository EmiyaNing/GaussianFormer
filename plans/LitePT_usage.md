# LitePT 接入 GaussianVoxelLearnear 的分析

本文分析当前仓库中新加入的 `LitePT/` 代码如何接入现有 GaussianFormer 工程，目标是用 LitePT 替换或增强 [`model/lifter/voxel_gaussian_lifter.py`](model/lifter/voxel_gaussian_lifter.py) 中的 `GaussianVoxelLearnear` 点云分支，构造新的 gaussian lifter。

> 注意：当前代码注册的类名是 `GaussianVoxelLearnear`，拼写里是 `Learnear`，不是 `GaussianVoxelLearner`。配置文件目前也使用 `type='GaussianVoxelLearnear'`，如果新增类或改名，需要同步更新 `model/lifter/__init__.py` 和 config。

## 1. 当前 GaussianVoxelLearnear 的接口

现有 `GaussianVoxelLearnear` 的流程是：

1. 从 `metas['lidar_points']` 逐 batch 读取点云，每个点为 `[x, y, z, intensity]`。
2. 使用 `VoxelGeneratorWrapper` 做体素化，体素大小为 `voxel_size / 8`，默认 `0.0625m`。
3. 通过 `MeanVFE` 聚合体素内点特征。
4. 通过 `VoxelResBackBone8x(4, embed_dims, [128, 1600, 1600])` 得到稀疏体素特征 `out_tensor`。
5. 将 `out_tensor.indices/features/spatial_shape` 解码为：
   - `representation`: gaussian anchor，形状期望为 `[B, G, 10 + opa + semantic_dim]`
   - `rep_features`: anchor feature，形状期望为 `[B, G, embed_dims]`
   - `anchor_init`: `representation.clone()`

下游 `GaussianOccEncoder` 直接把这两个张量当作 dense batch 张量使用：

- `SparseGaussian3DEncoder` 读取 `anchor[..., :3]`, `anchor[..., 3:6]`, `anchor[..., 6:10]`, opacity 和 semantic。
- `SparseGaussian3DRefinementModule` 在 anchor 空间内预测增量，并把 xyz/scale 通过 sigmoid 映射到真实世界。
- `SparseConv3D` 也假设 `instance_feature` 和 `anchor` 是 `[B, G, ...]`。

因此 lifter 输出必须满足固定 batch 维和固定 anchor 数 `G`，否则当前 encoder/head 无法直接工作。

当前实现里有一个重要隐患：`decode_anchors_from_voxel()` 每个 batch 返回的 voxel 数可能不同，但 forward 里直接 `torch.stack(anchors_list)`。这要求每个样本最终 sparse voxel 数完全一致，实际数据中通常不成立。若之前没有报错，可能是特定配置、batch size 或中间输出恰好规避了该问题。LitePT 接入时需要正面处理变长点集。

## 2. LitePT 模型的输入输出

LitePT 位于 [`LitePT/litept/model.py`](LitePT/litept/model.py)，核心类包括：

- `Point`: Pointcept 风格的点云字典封装。
- `LitePT`: sparse conv + serialized attention 的点云主干。
- `GridPooling/GridUnpooling`: 基于 `grid_coord` 聚合和上采样。
- `PointROPEAttention`: 使用 `libs.pointrope.PointROPE` 和 `flash_attn.flash_attn_varlen_qkvpacked_func`。

LitePT 的 forward 输入是一个扁平化后的 batched 点云字典，至少包含：

```python
dict(
    feat=...,        # [N, C_in]
    coord=...,       # [N, 3] 原始点坐标
    grid_coord=...,  # [N, 3] 整数 voxel 坐标，或提供 coord + grid_size 让内部生成
    batch=...,       # [N] 每个点属于哪个 batch
    # 或 offset=...  # [B] 每个 batch 的累积点数
)
```

LitePT 输出仍是 `Point` 对象，关键字段是：

- `point.feat`: `[N_out, C_out]`
- `point.coord`: `[N_out, 3]`
- `point.grid_coord`: `[N_out, 3]`
- `point.batch`: `[N_out]`
- `point.offset`: `[B]`

如果 `enc_mode=False`，LitePT 会走 encoder-decoder，输出通常回到较高分辨率点集；如果 `enc_mode=True`，输出是 encoder 最深层或最后 pooling 后的稀疏点集，点数更少、语义更强。

## 3. 推荐集成路线

建议不要直接修改旧 `GaussianVoxelLearnear` 的所有逻辑，而是新增一个类，例如 `LitePTGaussianVoxelLearnear`，先保持旧实现可回退。推荐路线如下。

### 3.1 新增 LitePT lifter

新增文件可以放在：

```text
model/lifter/litept_gaussian_lifter.py
```

并在 [`model/lifter/__init__.py`](model/lifter/__init__.py) 注册导入。

类结构建议继承 `BaseLifter`，输出保持与当前 lifter 完全一致：

```python
return {
    "rep_features": instance_feature,  # [B, G, embed_dims]
    "representation": anchor,          # [B, G, 10 + opa + semantic_dim]
    "anchor_init": anchor.clone(),
}
```

这样可以复用现有 `GaussianOccEncoder`、refine、head 和 loss。

### 3.2 点云输入桥接

LitePT 不需要当前 `VoxelGeneratorWrapper + MeanVFE + VoxelResBackBone8x` 这套输入。桥接代码应直接从 `metas['lidar_points']` 生成 LitePT 所需的扁平字典：

```python
coords = []
feats = []
grid_coords = []
batches = []
for b, pts in enumerate(metas["lidar_points"]):
    pts = pts.to(device)
    xyz = pts[:, :3]
    intensity = pts[:, 3:4] if pts.shape[1] > 3 else xyz.new_zeros(xyz.shape[0], 1)
    mask = in_pc_range(xyz, pc_range)
    xyz = xyz[mask]
    intensity = intensity[mask]
    grid_coord = torch.floor((xyz - pc_min) / litept_grid_size).int()
    coords.append(xyz)
    grid_coords.append(grid_coord)
    feats.append(torch.cat([xyz, intensity], dim=-1))  # [x, y, z, intensity], 对应 in_channels=4
    batches.append(torch.full((xyz.shape[0],), b, device=device, dtype=torch.long))
data_dict = dict(
    coord=torch.cat(coords, 0),
    grid_coord=torch.cat(grid_coords, 0),
    feat=torch.cat(feats, 0),
    batch=torch.cat(batches, 0),
)
```

更稳妥的输入特征是 `[x, y, z, intensity]`，对应 LitePT 默认 `in_channels=4`。如果希望坐标归一化或加入 height、range 等派生特征，需要同步调整 `in_channels`。

### 3.3 LitePT 输出转 gaussian anchor

LitePT 输出是变长点集，当前 GaussianFormer 下游需要固定 `[B, G, ...]`。推荐先采用“采样/补齐到固定 `num_anchor`”的策略，而不是第一版就改 encoder 支持 ragged。

对每个 batch：

1. 从 `point.batch == b` 取当前样本的 `cur_coord` 和 `cur_feat`。
2. 若点数 `N_b > num_anchor`：
   - 训练阶段可随机采样或按某个 score/topk 采样。
   - 更推荐先做 deterministic sampling，例如 farthest point sampling、体素均匀采样或按 `cur_feat.norm()` topk，便于复现。
3. 若点数 `N_b < num_anchor`：
   - 用可学习 fallback anchors/features 补齐，或从已有点重复采样补齐。
   - 建议保留一组 learnable `fallback_anchor` 和 `fallback_feature`，类似 `GaussianVoxelLifter` 的随机 anchor，避免空点云或稀疏帧导致训练崩溃。
4. 将坐标映射到 anchor 的 xyz 表达：
   - 当前 refine 配置默认 `xyz_activation="sigmoid"`，anchor xyz 应是 inverse-sigmoid 空间。
   - 真实坐标 `xyz` 先归一化：`xyz_norm = (xyz - pc_min) / (pc_max - pc_min)`。
   - clamp 到 `[eps, 1 - eps]` 后调用 `safe_inverse_sigmoid(xyz_norm)`。
5. 用小 head 从 LitePT feature 预测 scale/rot/opacity/semantic：
   - `scale_learner: Linear(C_out, 3)`，输出经过 sigmoid 后再 inverse-sigmoid，保持和现有 lifter 一致。
   - `rot_learner: Linear(C_out, 4)` 更合理；当前旧代码是 3 维再前面补 0，后续 refine 会 normalize，但初始四元数第一维为 0 不太理想。建议改为直接预测 4 维，或初始化为 `[1, 0, 0, 0] + delta`。
   - `opa_learner: Linear(C_out, 1)`。注意当前旧 lifter 把 `sigmoid(density)` 直接放进 anchor；而 `SparseGaussian3DEncoder` 只是编码该值，`SparseGaussian3DRefinementModule` 最终又对输出 opacity 做 sigmoid。这里可以沿用旧行为，但文档上应明确它不是 inverse-sigmoid opacity。
   - `semantic_learner: Linear(C_out, semantic_dim)`。

最终拼接：

```python
anchor = torch.cat([xyz_logit, scale_logit, rot, opacity, semantic], dim=-1)
instance_feature = out_proj(cur_feat)  # 若 LitePT C_out != embed_dims
```

### 3.4 LitePT 配置建议

LitePT 默认 `enc_channels=(36, 72, 144, 252, 504)`，而当前 GaussianFormer 常用 `embed_dims=128`。两种做法：

- 保持 LitePT 默认通道，新增 `feature_proj = nn.Linear(litept_out_channels, embed_dims)`。
- 调整 LitePT 通道使最终输出就是 `embed_dims`，例如让 decoder 输出 `128`。这需要仔细匹配 `enc_channels/dec_channels/num_head`，并保证每层 `channels % num_heads == 0`。

第一版推荐使用 `feature_proj`，对 LitePT 原实现侵入最小。

`enc_mode` 的选择：

- `enc_mode=False`：输出点数更接近输入点/高分辨率点，适合生成较多 gaussian anchors，但显存和时间更高。
- `enc_mode=True`：输出更稀疏，特征语义更强，适合少量 anchors 或作为 proposal 特征，但需要更强的补齐/采样策略。

对于当前 `num_anchor=25600`，建议第一版使用 `enc_mode=False`，然后用采样固定到 `num_anchor`。

## 4. 可能遇到的问题

### 4.1 包导入路径

`LitePT/litept/model.py` 使用：

```python
from libs.pointrope import PointROPE
from .serialization import encode
```

如果从仓库根目录直接 `from LitePT.litept.model import LitePT`，`libs.pointrope` 可能找不到，因为它是相对 `LitePT/` 目录的顶层包。可选处理：

1. 把 `LitePT/` 加入 `PYTHONPATH`。
2. 在工程内改成相对导入，例如 `from LitePT.libs.pointrope import PointROPE`，但这会改 LitePT 原代码。
3. 在新 lifter 内做局部 path 注入，不太优雅但改动小。

推荐优先使用环境变量或安装 LitePT 包路径，减少源码改动。

### 4.2 依赖和算子

LitePT 除 `libs/pointrope` 外还依赖：

- `flash_attn`
- `spconv.pytorch`
- `torch_scatter`
- `timm.layers.DropPath`
- `addict`

即使 `LitePT/libs/pointrope` 已安装，也需要确认这些依赖和当前 PyTorch/CUDA 版本一致。`PointROPE` 有 PyTorch fallback，但 `flash_attn` 没有 fallback；启用 attention 的阶段必须能正常 import 和运行 flash attention。

### 4.3 grid 坐标范围和 serialization depth

`Point.serialization()` 有限制：

```python
assert depth * 3 + len(offset).bit_length() <= 63
assert depth <= 16
```

如果 `grid_coord.max()` 太大，serialization depth 会超过 16。以 nuScenes 范围 `[-50, -50, -5, 50, 50, 3]` 为例：

- `grid_size=0.01m` 时 x/y 最大约 10000，depth 约 14，可接受但点数非常大。
- `grid_size=0.0625m` 时 x/y 最大约 1600，depth 约 11，更安全。
- `grid_size=0.5m` 时 x/y 最大约 200，depth 约 8，速度更好但空间分辨率低。

建议 LitePT 的初始 `grid_size` 与当前 lifter 的 `voxel_size / 8 = 0.0625m` 或稍粗一些的 `0.1m/0.2m` 起步测试。若输入点数过多，优先在进入 LitePT 前做 grid sampling 或随机下采样。

### 4.4 batch 内变长点数

LitePT 支持变长 batch，但 GaussianFormer 当前 encoder 不支持 ragged。必须在 lifter 内固定成 `[B, num_anchor, ...]`。这一步会影响训练稳定性：

- topk 采样可能偏向高强度/高密度区域，远处和稀疏物体可能少。
- 随机采样训练噪声更大，但覆盖更均衡。
- fallback learnable anchors 会引入非 LiDAR anchors，需要避免数量过多时主导训练。

建议先实现 deterministic 策略，并统计每帧 `N_out`、补齐比例、截断比例。

### 4.5 坐标系一致性

当前 dataset 在 `occ3d=True` 时会把 lidar 点变换到 ego/lidar 相关坐标并按 `pc_range` 裁剪。LitePT lifter 应沿用 `metas['lidar_points']` 的实际坐标语义，不要再重复做错误的全局/ego变换。

接入前建议用一个 batch 打印或可视化：

- `metas['lidar_points'][b][:, :3].min/max`
- 是否落在 `pc_range`
- 生成的 `grid_coord.min/max`
- anchor 经过 sigmoid 和 `pc_range` 还原后的 xyz 是否与点云范围一致

### 4.6 feature 维度和 encoder 配置

当前 config 中 `embed_dims=128`，`ffn`、`SparseGaussian3DEncoder`、`SparseGaussian3DRefinementModule` 都按这个维度构建。LitePT 输出通道如果不是 128，必须在 lifter 内投影到 `embed_dims`。

另外，`operation_order` 中如果包含 `"spconv"`，`SparseConv3D` 会根据 anchor xyz 再次稀疏卷积。LitePT 已经是点云主干，第一版可以先关闭 encoder 内的 `spconv_layer`，只保留：

```python
operation_order=[
    "deformable",
    "ffn",
    "norm",
    "refine",
] * num_decoder
```

等 LitePT lifter 跑通后，再评估是否保留 `"spconv"`。

### 4.7 当前旧 lifter 的若干 bug/不一致

接入时建议顺手规避这些问题：

- `GaussianVoxelLearnear.__init__()` 接收 `num_anchor`，但没有保存 `self.num_anchor`。在 `include_opa=False` 或 `semantics=False` 分支里用到了 `self.num_anchor`，会潜在报错。
- `process_lidar_data()` 里键名是 `all_num_ponts`，拼写错误但内部一致，不影响功能。
- `coords` 拼 batch index 时用 `torch.ones(...) * i`，dtype 可能是 float；spconv indices 通常需要 int。当前后续是否正常取决于 `VoxelResBackBone8x` 内部处理。
- `normalized_xyz = cur_voxel / spatial_shape` 使用的是 voxel index，不是 voxel center；最好加 `0.5` 得到中心。
- `opacity` 当前用 sigmoid 后值域 `[0,1]`，而初始随机 GaussianLifter 使用 inverse sigmoid。两类 lifter 的 anchor opacity 语义不完全一致。
- `torch.stack(anchors_list)` 对变长 voxel 不安全。

## 5. 建议的最小实现步骤

1. 新增 `LitePTGaussianVoxelLearnear`，不要直接覆盖旧类。
2. 在 lifter 内完成 `metas['lidar_points'] -> LitePT data_dict`。
3. 实例化 `LitePT(in_channels=4, enc_mode=False, ...)`。
4. 将 LitePT 输出按 batch 切分，并采样/补齐到 `num_anchor`。
5. 增加 `feature_proj`、`scale_learner`、`rot_learner`、`opa_learner`、`semantic_learner`。
6. 输出完全兼容当前 `GaussianOccEncoder`。
7. 新增一个 config，例如 `config/img_point_lite/litept_gaussian.py`，只替换 lifter，其他模块先尽量不动。
8. 用小 batch 做 smoke test：
   - import 是否成功
   - forward 是否成功
   - `representation.shape == [B, num_anchor, 10 + int(include_opa) + semantic_dim]`
   - `rep_features.shape == [B, num_anchor, embed_dims]`
   - loss 能反传

## 6. 后续优化方向

第一版跑通后，可以继续做更深入的融合：

- 让 LitePT 输出多尺度点特征，填充 `multi_stride_features`，供 encoder 中的 `"query"` 或其它 cross-attention 模块使用。
- 用 LitePT 特征预测 anchor quality，再用 topk 选择 anchors。
- 把 GaussianFormer encoder 改造成支持 ragged point set，避免强制 padding 到固定 `num_anchor`。
- 用历史帧点云构造 LitePT 输入，但要明确历史点已经变换到当前帧坐标，避免重复坐标变换。
- 预训练 LitePT 权重迁移：需要处理 key 命名、输入特征定义和输出通道不一致。

## 7. 推荐结论

LitePT 适合作为当前 `GaussianVoxelLearnear` 中 `VoxelGeneratorWrapper + MeanVFE + VoxelResBackBone8x` 的替代点云主干，但不能简单替换一行 backbone。真正需要实现的是一个适配层：把 LitePT 的变长 `Point` 输出转换成 GaussianFormer 下游需要的固定 `[B, num_anchor, ...]` gaussian anchor 和 feature。

最稳妥的第一版方案是新增 `LitePTGaussianVoxelLearnear`，保持下游 encoder/head 不变，在 lifter 内完成固定 anchor 数、坐标 logit 化、feature 投影和 gaussian 属性预测。这样风险集中在 lifter 内，旧配置可回退，也便于逐步比较旧 spconv lifter 与 LitePT lifter 的效果。
