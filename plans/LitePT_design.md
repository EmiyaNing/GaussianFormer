# LitePTGaussianLearner 设计与代码方案

本文给出 `LitePTGaussianLearner` 的第一版实现方案。目标是按 [`LitePT_usage.md`](LitePT_usage.md) 的最小实现思路接入 LitePT，但根据新的约束调整为：

- 新类命名为 `LitePTGaussianLearner`。
- 使用 4 次 LitePT encoder 下采样到 `1.0m`，再由 `LitePT_Ours` 根据 decoder 参数执行 1 次 decoder unpool 回 `0.5m`。
- 输入 voxelize 尺寸固定为 `0.0625m`。
- LitePT 输出 voxel 尺寸固定为 `0.5m`，即最终有效下采样倍率为 `8x`。
- 输出接口与当前 `GaussianVoxelLearnear` 保持一致：返回 batched dense `rep_features`、`representation`、`anchor_init`，尽量不改 encoder/head。
- 不再对齐到历史固定 `num_anchor=25600`，但会在当前 batch 内对齐 LitePT 输出点数，以便 `torch.stack` 成 `[B, G, ...]`。
- `num_anchor` 只作为旧 config 兼容参数接收，不参与核心逻辑。

## 1. 关键结论

LitePT 当前默认配置接近我们要的结构：

```python
stride=(2, 2, 2, 2)
enc_depths=(2, 2, 2, 6, 2)
enc_channels=(36, 72, 144, 252, 504)
enc_mode=False
```

这会在 encoder 端下采样 `2 * 2 * 2 * 2 = 16x`。若输入 grid size 为 `0.0625m`，只看 encoder 最深层时 voxel size 会变成：

```text
0.0625m * 16 = 1.0m
```

这比目标 `0.5m` 更粗。因此当前采用的方案是：保留 4 次 encoder 下采样得到 `1.0m` bottleneck，再通过 LitePT decoder 的第一层 `GridUnpooling` 回到 stage3 的 `0.5m` 分辨率，并停止在该层输出。

LitePT 原始 forward 会在 `enc_mode=False` 时跑完所有 decoder stage，最终回到接近输入的 `0.0625m` 分辨率。为了只回到 `0.5m`，在 `LitePT/litept` 下新增 `LitePT_Ours`：它复用 LitePT 的基础模块，但 forward 天然支持“只执行指定数量的 decoder module”。本方案中 `num_decoders=1`，即只执行 `self.dec[0]`，注册名对应 `dec3`。

完整展开后的底层配置等价于：

```python
stride=(2, 2, 2, 2)
enc_depths=(2, 2, 2, 6, 2)
enc_channels=(36, 72, 144, 252, 504)
enc_num_head=(2, 4, 8, 14, 28)
enc_patch_size=(1024, 1024, 1024, 1024, 1024)
enc_conv=(True, True, True, False, False)
enc_attn=(False, False, False, True, True)

# decoder 只需要第一个 module: dec[0]/dec3, stage4(1.0m) -> stage3(0.5m)
dec_depths=(0, 0, 0, 0)
dec_channels=(72, 72, 144, 252)
dec_num_head=(4, 4, 8, 14)
dec_patch_size=(1024, 1024, 1024, 1024)
dec_conv=(False, False, False, False)
dec_attn=(False, False, False, False)
enc_mode=False
```

尺寸路径为：

```text
input:  0.0625m
enc1:   0.125m
enc2:   0.25m
enc3:   0.5m
enc4:   1.0m
dec[0]/dec3: 0.5m
```

这里 `self.litept.dec[0]` 指 LitePT 构造 decoder 时 `reversed(range(self.num_stages - 1))` 的第一项，对应 `s=3`，module 名为 `dec3`，也就是从最深 stage4 unpool 到 encoder stage3。由于 `dec_depths[3]=0`，这一层只做 `GridUnpooling` 和线性/BN/GELU 投影，不额外增加 attention block。

为了简化上层配置，`LitePTGaussianLearner` 不直接暴露上面所有 tuple 参数，而是调用 `LitePT_Ours` 的少量高层参数：

```python
litept=dict(
    in_channels=4,
    preset="litept_base",
    input_grid_size=0.0625,
    target_grid_size=0.5,
    num_decoders=1,
    decoder_block_depths=(0,),
)
```

`LitePT_Ours` 内部根据 `input_grid_size`、`target_grid_size` 和 `num_decoders` 推导需要停止的位置，并记录 `out_channels=252`、`out_stride=8`、`out_grid_size=0.5`。

## 2. LitePT_Ours 设计

允许在 `LitePT/litept` 下新增文件：

```text
LitePT/litept/litept_ours.py
```

其中实现：

```python
class LitePT_Ours(PointModule):
    ...
```

也可以新增或更新 `LitePT/litept/__init__.py` 并导出：

```python
from .litept_ours import LitePT_Ours
```

### 2.1 设计目标

`LitePT_Ours` 的目标不是重写 LitePT，而是把 [`LitePT/litept/model.py`](LitePT/litept/model.py) 中已有的 `Point`、`PointSequential`、`Embedding`、`GridPooling`、`GridUnpooling`、`Block` 等模块复用起来，并解决两个工程问题：

1. 原版 `LitePT.forward()` 只能选择 encoder-only 或完整 decoder，不能自然停在中间 decoder 分辨率。
2. 原版参数 tuple 较多，lifter 配置会非常冗长，容易配错 `enc_channels/dec_channels/num_heads`。

### 2.2 推荐构造接口

推荐 `LitePT_Ours` 暴露少量高层参数：

```python
class LitePT_Ours(PointModule):
    def __init__(
        self,
        in_channels=4,
        preset="litept_base",
        input_grid_size=0.0625,
        target_grid_size=0.5,
        num_decoders=1,
        decoder_block_depths=(0,),
        drop_path=0.3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders=True,
        **kwargs,
    ):
        ...
```

参数含义：

- `preset`: 选择一组内置 LitePT 结构。第一版只需要 `litept_base`。
- `input_grid_size`: 输入 voxel size，本项目固定为 `0.0625`。
- `target_grid_size`: 期望输出 voxel size，本项目固定为 `0.5`。
- `num_decoders`: forward 中执行多少个 decoder module。当前取 `1`，即 `1.0m -> 0.5m`。
- `decoder_block_depths`: 每个已执行 decoder module 内部额外接多少个 `Block`。当前取 `(0,)`，只做 `GridUnpooling`。

`LitePT_Ours` 仍允许通过 `**kwargs` 覆盖底层 tuple 参数，但第一版 lifter config 不建议暴露这些细节。

### 2.3 preset 展开规则

`preset="litept_base"` 展开为：

```python
stride=(2, 2, 2, 2)
enc_depths=(2, 2, 2, 6, 2)
enc_channels=(36, 72, 144, 252, 504)
enc_num_head=(2, 4, 8, 14, 28)
enc_patch_size=(1024, 1024, 1024, 1024, 1024)
enc_conv=(True, True, True, False, False)
enc_attn=(False, False, False, True, True)
enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0)

dec_channels=(72, 72, 144, 252)
dec_num_head=(4, 4, 8, 14)
dec_patch_size=(1024, 1024, 1024, 1024)
dec_conv=(False, False, False, False)
dec_attn=(False, False, False, False)
dec_rope_freq=(100.0, 100.0, 100.0, 100.0)
```

`decoder_block_depths=(0,)` 会被扩展到 LitePT 内部需要的 `dec_depths=(0, 0, 0, 0)`，但 forward 只执行前 `num_decoders=1` 个 decoder module。若未来希望执行两个 decoder module，例如输出 `0.25m`，可以设置：

```python
target_grid_size=0.25
num_decoders=2
decoder_block_depths=(0, 0)
```

### 2.4 forward 行为

`LitePT_Ours.forward()` 接收与原 LitePT 一致的 `data_dict`：

```python
point = Point(data_dict)
if self.enc_attn[0]:
    point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
point.sparsify()
point = self.embedding(point)
point = self.enc(point)
for i in range(self.num_decoders):
    point = self.dec[i](point)
return point
```

它不需要 `enc_mode` 参数。是否执行 decoder 由 `num_decoders` 控制：

- `num_decoders=0`: 输出 encoder bottleneck，当前为 `1.0m`。
- `num_decoders=1`: 输出 `0.5m`，本项目默认。
- `num_decoders=2`: 输出 `0.25m`。
- `num_decoders=4`: 等价于完整 decoder，输出接近 `0.0625m`。

### 2.5 输出元信息

`LitePT_Ours` 应在初始化后提供：

```python
self.out_channels = dec_channels[-1]   # num_decoders=1 时为 252
self.out_stride = 8                    # 0.5 / 0.0625
self.out_grid_size = 0.5
```

更一般地：

```python
encoder_stride = prod(stride)          # 16
decoder_factor = 2 ** num_decoders     # 2
out_stride = encoder_stride // decoder_factor
out_grid_size = input_grid_size * out_stride
```

实现时应 assert：

```python
assert abs(out_grid_size - target_grid_size) < 1e-6
```

这样 `LitePTGaussianLearner` 可以直接写：

```python
self.litept = LitePT_Ours(...)
self.feature_proj = nn.Linear(self.litept.out_channels, embed_dims)
```

## 3. 新 lifter 的位置与注册

建议新增文件：

```text
model/lifter/litept_gaussian_lifter.py
```

新增类：

```python
@MODELS.register_module()
class LitePTGaussianLearner(BaseLifter):
    ...
```

并在 [`model/lifter/__init__.py`](model/lifter/__init__.py) 中加入：

```python
from .litept_gaussian_lifter import LitePTGaussianLearner
```

配置中使用：

```python
lifter=dict(
    type="LitePTGaussianLearner",
    embed_dims=128,
    semantics=True,
    semantic_dim=17,
    include_opa=True,
    pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
    voxel_size=0.5,
    litept_grid_size=0.0625,
    output_grid_size=0.5,
    litept=dict(
        preset="litept_base",
        num_decoders=1,
        decoder_block_depths=(0,),
    ),
)
```

`num_anchor` 可以继续保留在配置里，但类内只做兼容接收：

```python
def __init__(self, num_anchor=None, ..., **kwargs):
    self.num_anchor = num_anchor  # 仅记录，不用于 pad/sample
```

## 4. 输入 voxelize 设计

当前旧 `GaussianVoxelLearnear` 使用 `VoxelGeneratorWrapper(vsize_xyz=[voxel_size / 8] * 3)`。如果 `voxel_size=0.5`，输入体素尺寸就是：

```text
0.5 / 8 = 0.0625m
```

LitePT 自身不需要 `VoxelGeneratorWrapper`，它需要的是 `grid_coord`。因此新 lifter 直接从 `metas["lidar_points"]` 构造 LitePT 的输入字典：

```python
pc_min = torch.tensor(pc_range[:3], device=device)
pc_max = torch.tensor(pc_range[3:], device=device)
grid = 0.0625

all_coord = []
all_grid_coord = []
all_feat = []
all_batch = []

for b, pts in enumerate(metas["lidar_points"]):
    pts = pts.to(device)
    xyz = pts[:, :3]
    intensity = pts[:, 3:4] if pts.shape[-1] > 3 else xyz.new_zeros(xyz.shape[0], 1)

    mask = ((xyz >= pc_min) & (xyz < pc_max)).all(dim=-1)
    xyz = xyz[mask]
    intensity = intensity[mask]

    grid_coord = torch.floor((xyz - pc_min) / grid).to(torch.int32)
    feat = torch.cat([xyz, intensity], dim=-1)

    all_coord.append(xyz)
    all_grid_coord.append(grid_coord)
    all_feat.append(feat)
    all_batch.append(torch.full((xyz.shape[0],), b, device=device, dtype=torch.long))

data_dict = dict(
    coord=torch.cat(all_coord, dim=0),
    grid_coord=torch.cat(all_grid_coord, dim=0),
    feat=torch.cat(all_feat, dim=0),
    batch=torch.cat(all_batch, dim=0),
)
```

### 是否需要预聚合

LitePT 的 `Point.sparsify()` 会用 `grid_coord` 和 `feat` 构造 `spconv.SparseConvTensor`，但如果同一个 `grid_coord` 内存在多个原始点，spconv 对重复 indices 的行为需要谨慎。为了稳定，第一版建议在进入 LitePT 前做一次 `0.0625m` grid pooling，把同一输入 voxel 内的点聚合成一个点：

- `coord`: voxel 内 xyz 均值，或 voxel center。
- `feat`: `[mean_x, mean_y, mean_z, mean_intensity]`，或 `[center_x, center_y, center_z, mean_intensity]`。
- `grid_coord`: 唯一的 `[ix, iy, iz]`。
- `batch`: 当前 batch id。

实现上可以复用 `torch.unique(..., return_inverse=True)` 和 `torch_scatter.scatter_mean`。仓库中 LitePT 已依赖 `torch_scatter`，所以这里不用再引入新依赖。

推荐第一版做预聚合。这样可以明确保证“输入 voxelize 尺寸为 `0.0625m`”，并避免 duplicate sparse indices。

## 5. LitePTGaussianLearner 中的 LitePT_Ours 配置

在 `LitePTGaussianLearner.__init__()` 中不再直接构建原始 `LitePT`，而是构建 `LitePT_Ours`：

```python
self.input_grid_size = litept_grid_size  # 0.0625
self.output_grid_size = output_grid_size # 0.5
self.downsample_stride = int(round(output_grid_size / litept_grid_size)) # 8
assert abs(litept_grid_size * 8 - output_grid_size) < 1e-6

self.litept = LitePT_Ours(
    in_channels=4,
    preset="litept_base",
    input_grid_size=litept_grid_size,
    target_grid_size=output_grid_size,
    num_decoders=1,
    decoder_block_depths=(0,),
    drop_path=0.3,
)
self.feature_proj = nn.Linear(self.litept.out_channels, embed_dims)
```

此时 `LitePTGaussianLearner.forward()` 可以直接调用 `point = self.litept(data_dict)`。`LitePT_Ours` 内部已经保证只跑 4 次 encoder 和 1 次 decoder unpool，不会完整上采样回 `0.0625m`。

这里 `self.litept.out_channels` 在默认配置下为 `252`，而 GaussianFormer 的 `embed_dims` 通常是 `128`，因此 `feature_proj` 仍然是必要的。

如果希望更省显存，也可以把最后通道改成 `128`：

```python
enc_channels=(32, 64, 128, 192, 256)
enc_num_head=(2, 4, 8, 12, 16)
dec_channels=(64, 64, 128, 128)
```

但第一版建议保持 LitePT 较接近原始通道风格，再投影到 `embed_dims`，减少对 LitePT 结构的猜测。

## 6. 输出格式设计：保持 GaussianVoxelLearnear 兼容

为了尽量少改当前工程，`LitePTGaussianLearner` 的对外输出应与 [`model/lifter/voxel_gaussian_lifter.py`](model/lifter/voxel_gaussian_lifter.py) 中的 `GaussianVoxelLearnear` 保持一致：

- `rep_features`: `[B, G, embed_dims]`
- `representation`: `[B, G, 10 + int(include_opa) + semantic_dim]`
- `anchor_init`: `[B, G, 10 + int(include_opa) + semantic_dim]`

其中 `G` 不再是配置里的固定 `num_anchor`，而是 LitePT 在当前 batch 中产生的 `0.5m` 下采样点经过 batch 对齐后的数量。这样后续 `GaussianOccEncoder`、`SparseGaussian3DEncoder`、`SparseGaussian3DRefinementModule`、`GaussianHead` 仍然看到原来的 dense batch 张量，第一版不需要新增 ragged encoder/head。

LitePT 原始输出仍是 flattened point：

```python
feat_flat = self.feature_proj(point.feat)      # [N_out, embed_dims]
anchor_flat = self.decode_anchor(point.coord, feat_flat)
batch = point.batch.long()                     # [N_out]
```

然后在 lifter 内按 batch 切分并对齐：

```python
features_list = []
anchors_list = []
counts = torch.bincount(batch, minlength=batch_size)
target_g = counts.min().item()

for b in range(batch_size):
    mask = batch == b
    cur_feat = feat_flat[mask]
    cur_anchor = anchor_flat[mask]

    # 默认 crop_to_min，避免构造无效 padded gaussian。
    # 可选按 feature norm / opacity score 排序后保留 target_g。
    keep = self.select_points(cur_feat, cur_anchor, target_g)
    features_list.append(cur_feat[keep])
    anchors_list.append(cur_anchor[keep])

instance_feature = torch.stack(features_list, dim=0)
anchor = torch.stack(anchors_list, dim=0)
```

默认推荐 `align_mode="crop_to_min"`，原因是它不需要下游 mask，也不会引入 padded fake gaussian。代价是当 batch 内样本点数差异很大时，会丢弃点数较多样本的一部分 LitePT 输出。若训练主要使用 `batch_size=1`，则不会发生裁剪。

`select_points()` 建议第一版保持确定性，例如按 `cur_feat.norm(dim=-1)` 或预测 opacity 分数取 top-k，避免随机裁剪造成复现实验困难。如果 `target_g == 0`，说明当前 batch 至少有一个样本没有有效 LitePT 输出；此时应回退到一个很小的 learnable fallback，例如 `min_output_points=1` 或 `8`，否则现有 encoder/head 很可能无法处理空 gaussian 集合。

可选模式：

- `crop_to_min`: 默认，最少改动，无需下游 mask。
- `pad_to_max`: 保留所有点并 pad 到当前 batch 最大点数，但 padded gaussian 会进入现有 encoder/head；如果没有 mask，可能产生额外噪声。
- `fixed_topk`: 设置一个新的 `max_output_points`，按 score 截断/补齐到固定值。它类似 `num_anchor`，但语义是 LitePT 输出上限，不建议第一版默认使用。

第一版返回值保持为：

```python
return {
    "rep_features": instance_feature,
    "representation": anchor,
    "anchor_init": anchor.clone(),
}
```

如果需要调试，可以临时返回 `litept_counts`、`litept_grid_size` 等辅助 key，但默认 config 不应依赖这些 key。

## 7. Gaussian anchor 解码方案

`LitePT_Ours` 输出 `point` 后：

```python
point = self.litept(data_dict)
litept_feat = point.feat              # [N_out, C_litept]
coord = point.coord                   # [N_out, 3]
grid_coord = point.grid_coord         # [N_out, 3], 已经是 0.5m grid
batch = point.batch                   # [N_out]
instance_feature = self.feature_proj(litept_feat)
```

由于 `GridPooling` 会把 `coord` 用 mean 聚合，所以 `point.coord` 是真实坐标下的聚合中心。gaussian xyz 建议直接用 `coord`，再转换到当前 refine 使用的 logit 空间：

```python
xyz_norm = (coord - pc_min) / (pc_max - pc_min)
xyz_norm = xyz_norm.clamp(eps, 1.0 - eps)
xyz_anchor = safe_inverse_sigmoid(xyz_norm)
```

scale 初始化应对应输出 voxel 的物理尺寸 `0.5m`。但当前 refine 的 scale 是归一化/logit 后再映射到 `scale_range`。建议第一版使用 feature head 预测，初始化 bias 让初始值接近 `0.5m`：

```python
scale_logits = self.scale_learner(instance_feature)
scale_prob = torch.sigmoid(scale_logits)
scale_anchor = safe_inverse_sigmoid(scale_prob.clamp(eps, 1 - eps))
```

更稳定的初始化策略：

- 若 `scale_range=[0.08, 0.64]`，目标 `0.5` 对应归一化概率 `(0.5 - 0.08) / (0.64 - 0.08) = 0.75`。
- 可以把 `scale_learner.bias` 初始化到 `safe_inverse_sigmoid(0.75)` 附近。

rotation 建议直接预测 4 维并归一化：

```python
rot_raw = self.rot_learner(instance_feature)
rot = F.normalize(rot_raw + identity_quat_bias, dim=-1)
```

其中 `identity_quat_bias=[1, 0, 0, 0]` 可以通过 bias 初始化实现。

opacity 和 semantic：

```python
opacity = torch.sigmoid(self.opa_learner(instance_feature))  # include_opa=True
semantic = self.semantic_learner(instance_feature)           # semantics=True
anchor = torch.cat([xyz_anchor, scale_anchor, rot, opacity, semantic], dim=-1)
```

## 8. 下游 encoder 的代码方案

因为 lifter 内已经把 LitePT 输出整理回 `[B, G, ...]`，下游可以继续沿用当前 dense 版本：

- `SparseGaussian3DEncoder` 不需要改。
- `AsymmetricFFN` / `LN` 不需要改。
- `SparseGaussian3DRefinementModule` 不需要改。
- `GaussianHead` 不需要因 LitePT 输出形态而改。
- `SparseConv3D` 和 deformable 图像交互也可以继续接收 `[B, G, C]`，但第一版仍建议先关闭，缩小变量。

因此第一版配置可以继续使用 `GaussianOccEncoder`：

```python
encoder=dict(
    type="GaussianOccEncoder",
    ...
    deformable_model=None,
    spconv_layer=None,
    operation_order=[
        "ffn",
        "norm",
        "refine",
    ] * num_decoder,
)
```

等 LitePT lifter 跑通后，再恢复 `deformable_model` 或 `spconv_layer` 评估收益。关键是先验证只替换 lifter 时训练 forward/loss 能稳定运行。

## 9. 文件级改动计划

第一阶段实现 `LitePT_Ours` 和 LitePT lifter，并保持现有 encoder/head 不变：

```text
LitePT/litept/litept_ours.py
model/lifter/litept_gaussian_lifter.py
model/lifter/__init__.py
config/img_point_lite/litept_gaussian_partial_decoder.py
```

第二阶段才考虑是否暴露 `litept_counts`、`litept_grid_coord` 或做 ragged 版本 encoder。第一版不要新增 `RaggedGaussianOccEncoder`，避免把接口改动扩散到整个 pipeline。

## 10. 代码骨架

### 10.1 LitePT_Ours

```python
class LitePT_Ours(PointModule):
    def __init__(
        self,
        in_channels=4,
        preset="litept_base",
        input_grid_size=0.0625,
        target_grid_size=0.5,
        num_decoders=1,
        decoder_block_depths=(0,),
        drop_path=0.3,
        **kwargs,
    ):
        super().__init__()
        cfg = build_litept_preset(preset, decoder_block_depths, **kwargs)
        self.num_decoders = num_decoders
        self.input_grid_size = input_grid_size
        self.out_stride = compute_out_stride(cfg["stride"], num_decoders)
        self.out_grid_size = input_grid_size * self.out_stride
        assert abs(self.out_grid_size - target_grid_size) < 1e-6

        # Build embedding, encoder and decoder modules following LitePT.
        ...
        self.out_channels = resolve_out_channels(cfg, num_decoders)

    def forward(self, data_dict):
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        for i in range(self.num_decoders):
            point = self.dec[i](point)
        return point
```

`resolve_out_channels()` 应按实际执行的输出 stage 决定通道数：`num_decoders=0` 时为 encoder bottleneck 通道 `504`，`num_decoders=1` 时为第一个 decoder module 输出通道 `252`，`num_decoders=2` 时为 `144`。不要直接依赖负索引猜测。

### 10.2 LitePTGaussianLearner

```python
@MODELS.register_module()
class LitePTGaussianLearner(BaseLifter):
    def __init__(
        self,
        embed_dims,
        num_anchor=None,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        pc_range=(-50, -50, -5, 50, 50, 3),
        voxel_size=0.5,
        litept_grid_size=0.0625,
        output_grid_size=0.5,
        align_mode="crop_to_min",
        min_output_points=1,
        litept=None,
        **kwargs,
    ):
        super().__init__()
        assert abs(litept_grid_size * 8 - output_grid_size) < 1e-6
        self.embed_dims = embed_dims
        self.num_anchor = num_anchor
        self.pc_range = pc_range
        self.input_grid_size = litept_grid_size
        self.output_grid_size = output_grid_size
        self.align_mode = align_mode
        self.min_output_points = min_output_points
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.include_opa = include_opa
        self.semantics = semantics
        self.semantic_dim = semantic_dim if semantics else 0

        litept_cfg = dict(litept or {})
        self.litept = LitePT_Ours(
            in_channels=litept_cfg.pop("in_channels", 4),
            preset=litept_cfg.pop("preset", "litept_base"),
            input_grid_size=litept_grid_size,
            target_grid_size=output_grid_size,
            num_decoders=litept_cfg.pop("num_decoders", 1),
            decoder_block_depths=litept_cfg.pop("decoder_block_depths", (0,)),
            **litept_cfg,
        )
        self.feature_proj = nn.Linear(self.litept.out_channels, embed_dims)
        self.scale_learner = nn.Linear(embed_dims, 3)
        self.rot_learner = nn.Linear(embed_dims, 4)
        if include_opa:
            self.opa_learner = nn.Linear(embed_dims, 1)
        if semantics:
            self.semantic_learner = nn.Linear(embed_dims, semantic_dim)
```

### 10.3 forward

```python
def forward(self, imgs=None, metas=None, **kwargs):
    data_dict = self.build_litept_input(metas)
    point = self.litept(data_dict)

    feat_flat = self.feature_proj(point.feat)
    anchor_flat = self.decode_anchor(point.coord, feat_flat)
    feat, anchor = self.align_as_dense_batch(
        feat_flat,
        anchor_flat,
        point.batch.long(),
        batch_size=len(metas["lidar_points"]),
    )

    return {
        "rep_features": feat,
        "representation": anchor,
        "anchor_init": anchor.clone(),
    }
```

## 11. 验证清单

实现后先做以下检查：

1. 打印输入预聚合后的 `grid_coord.max()`，确认输入 grid 是 `0.0625m`。
2. 打印 LitePT 输出 `point.grid_coord.max()` 和 `point.coord` 范围。
3. 验证输出 voxel 尺寸：
   - 输入 grid size: `0.0625`
   - encoder stride product: `16`
   - partial decoder unpool factor: `2`
   - final effective stride: `8`
   - 输出 grid size: `0.5`
4. 检查 shape：
   - `rep_features`: `[B, G, embed_dims]`
   - `representation`: `[B, G, 10 + int(include_opa) + semantic_dim]`
   - `anchor_init`: `[B, G, 10 + int(include_opa) + semantic_dim]`
5. 检查每个 batch 的点数：
   - `torch.bincount(point.batch)`
   - 对齐后的 `G` 不要求等于 `num_anchor`
6. 先关闭 deformable/spconv，只跑 `ffn/norm/refine`。
7. 直接使用现有 `GaussianOccEncoder` 和 `GaussianHead` 验证 loss/反传。

## 12. 风险点

- 当前下游假设 `[B, G, C]`，因此 LitePT 的 flattened 输出必须在 lifter 内对齐回 dense batch。
- `crop_to_min` 会丢弃 batch 内点数较多样本的一部分 LitePT 输出；如果 batch size 为 1，则没有该问题。
- `pad_to_max` 虽然保留所有点，但如果下游不使用 mask，padded gaussian 可能影响训练。第一版不建议默认使用。
- `Point.sparsify()` 对重复 sparse indices 的处理不应依赖隐式行为，因此建议显式做 `0.0625m` 输入预聚合。
- `LitePT/litept/model.py` 使用 `from libs.pointrope import PointROPE`，需要保证 `LitePT/` 在 `PYTHONPATH` 中，或在实现时修正导入策略。
- stage3 和 stage4 启用 attention 时依赖 `flash_attn` 和 `PointROPE`，需要在 smoke test 中确认 CUDA 算子可用。
- `enc_attn=(False, False, False, True, True)` 会在 0.5m 和 1.0m 两个较低分辨率 stage re-serialization，比较符合显存预期。
- `LitePTGaussianLearner` 应调用 `LitePT_Ours(data_dict)`，不要再手动拆原版 `LitePT.embedding/enc/dec`；partial decoder 停止逻辑应集中在 `LitePT_Ours` 内。
- `LitePT_Ours` 的 `num_decoders`、`target_grid_size`、`out_stride` 必须互相校验，否则很容易无意输出 `1.0m` 或 `0.25m`。
- 如果 `enc_channels/dec_channels` 与 LitePT 预训练权重不匹配，加载预训练会有 key/shape mismatch。第一版可先随机初始化。

## 13. 推荐实施顺序

1. 新增 `LitePT/litept/litept_ours.py`，实现 `LitePT_Ours`、preset 展开、`num_decoders` 控制和输出元信息。
2. 新增 `LitePTGaussianLearner`，实现 input voxelize、调用 `LitePT_Ours`、anchor decode、batch dense 对齐。
3. 写一个临时脚本或 debug 分支，单独跑 lifter forward，确认 `0.0625 -> 1.0 -> 0.5`。
4. 确认 lifter 返回 key 与 `GaussianVoxelLearnear` 一致，且 shape 为 `[B, G, ...]`。
5. 新增 config，继续使用现有 `GaussianOccEncoder` 和 `GaussianHead`，先关闭 `deformable_model` 和 `spconv_layer` 跑通训练 forward/loss。
6. 再考虑恢复图像 deformable 交互或多尺度点特征。
