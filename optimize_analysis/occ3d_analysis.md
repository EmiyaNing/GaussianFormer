# Occ3D 迁移分析：从 `img_voxel_react.py` 到可稳定训练的 Occ3D 配置

目标配置：`config/img_point_lite_react/img_voxel_react.py`

相关实现：

- `dataset/dataset.py`
- `dataset/transform_3d.py`
- `model/lifter/voxel_gaussian_lifter.py`
- `model/head/gaussian_head.py`
- `train.py`
- `eval.py`

## 1. 核心结论

推荐把 Occ3D 迁移做成 **全 ego 坐标主轴**：

```text
Occ3D label grid: ego
occ_xyz:          ego
lidar_points:     ego
Gaussian means:   ego
projection_mat:   ego2img
head pc_min:      Occ3D ego pc_range min
```

也就是说，Occ3D 不应该继续沿用 SurroundOcc 的 `lidar2img + lidar pc_range` 语义。当前代码里已有一半逻辑实际上已经在 ego 坐标下工作，但另一半仍按 lidar 坐标投影，这会造成隐性错位。

最需要优先修正的点有四个：

1. `LoadOccupancyOcc3D(use_ego=False)` 当前返回的 `occ_xyz` 实际仍是 Occ3D ego grid；但 Occ3D config 又使用 `NuScenesAdaptor(use_ego=False)`，导致 ego anchors 被当成 lidar 点投影到图像。
2. `NuScenesDataset(occ3d=True)` 会把 `lidar_points` 从 lidar 转到 ego，但 config 没有把 Occ3D 的 `pc_range=[-40,-40,-1,40,40,5.4]` 传进 dataset，点云预过滤仍用默认 SurroundOcc range，尤其会丢掉 `z>3` 的点。
3. 现有 Occ3D config 的图像增强不合理：`input_shape=(1600,864)` 但 `resize_lim=(0.5,0.5)`，训练时会把 1600x900 图像缩到 800x450 后再 crop 到 1600x864，大面积 padding 会严重破坏图像特征。迁移 `img_voxel_react.py` 时应先保留其 `resize_lim=(1.0,1.0)`。
4. `occ_mask` 的语义需要重命名和解耦。`semantics != 17` 是 non-empty mask，不是 “no camera mask eval” 的 mask。Occ3D eval 不使用 cam mask 时应该是不传 mask 或使用 all-true mask，而不是只评估非空体素。

因此，合理迁移不是简单把 loader 换成 `LoadOccupancyOcc3D`，而是要统一以下不变量：

```text
同一个 batch 内，occ_xyz / lidar_points / Gaussian means / projection_mat / head grid
必须全部处于同一个 3D 坐标系。
```

## 2. 当前 SurroundOcc 配置为什么能正常工作

`img_voxel_react.py` 的关键设定是：

```python
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
grid_size = 0.5
H, W, D = 200, 200, 16
LoadOccupancySurroundOcc(..., use_ego=False)
NuScenesAdaptor(use_ego=False)
```

这条链路在 SurroundOcc 上是自洽的：

| 模块 | 坐标语义 |
| --- | --- |
| `LoadOccupancySurroundOcc` | label 与 `occ_xyz` 是 lidar grid |
| `NuScenesDataset` | `lidar_points` 是 lidar 点 |
| `GaussianVoxelLearnear` | 由 lidar 点体素化生成 anchor |
| `NuScenesAdaptor` | 输出 `lidar2img` |
| encoder deformable attention | 用 lidar Gaussian 投影到 image |
| `GaussianHead` | 在 lidar grid 上 render 到 `occ_xyz` |
| `localagg_react` | `pc_min=[-50,-50,-5]`, `grid_size=0.5` |

所以 SurroundOcc 的成功不是某个单点设置，而是整条链路都以 lidar 坐标为共同语言。

## 3. 当前 Occ3D 实现的错位点

### 3.1 `LoadOccupancyOcc3D` 的 `use_ego` 语义不适合直接复用

`LoadOccupancySurroundOcc` 中的 meshgrid 是 lidar grid，所以：

```python
use_ego=False -> occ_xyz 是 lidar
use_ego=True  -> occ_xyz 从 lidar 变到 ego
```

但 `LoadOccupancyOcc3D` 的 meshgrid 本身就是 Occ3D ego grid：

```python
xyz = self.get_meshgrid(pc_range, [200, 200, 16], grid_size)
```

因此对 Occ3D 来说：

```python
use_ego=False -> occ_xyz 保持 ego，这是正确的
use_ego=True  -> 又乘了一次 lidar2ego，反而错误
```

这说明 Occ3D loader 不应该继续使用 `use_ego` 这个从 SurroundOcc 继承来的开关名。更清晰的设计是显式引入：

```python
label_coord = "ego"       # Occ3D 固定是 ego
model_coord = "ego"       # 推荐
```

如果未来确实要支持 lidar 主轴，再写成：

```python
model_coord = "lidar"
```

并在 loader 中明确做 `ego -> lidar` 的坐标转换，而不是复用 `use_ego`。

### 3.2 `lidar_points` 已经被转到 ego，但 projection 仍可能是 lidar2img

`NuScenesDataset.get_data_info()` 里有：

```python
if self.occ3d:
    lidar_points[:, 3] = 1.0
    lidar_points = lidar2ego @ lidar_points
```

所以在 `occ3d=True` 时，传给 `GaussianVoxelLearnear` 的 `metas["lidar_points"]` 已经是 ego 坐标。当前 `GaussianVoxelLearnear` 不使用 `lidar_pose`，只是直接体素化这些点，因此初始 anchors 也是 ego 坐标。

但现有 Occ3D config 里仍是：

```python
dict(type="NuScenesAdaptor", use_ego=False, num_cams=6)
```

这会把 `projection_mat` 设置成 `lidar2img`。结果是：

```text
ego anchor -> lidar2img -> image
```

这在几何上是错的。正确路线是：

```python
dict(type="NuScenesAdaptor", use_ego=True, num_cams=6)
```

让 deformable attention 使用：

```text
ego anchor -> ego2img -> image
```

### 3.3 dataset 的 `pc_range` 没有传入，Occ3D 点云预过滤仍用默认 SurroundOcc 范围

`NuScenesDataset.__init__()` 默认：

```python
pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
```

现有 `config/occ3d/occ3d_no_mask.py` 和 `config/img_point_fusion/img_voxel_lite_occ3d.py` 定义了：

```python
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
```

但构造 dataset 时没有传：

```python
train_dataset_config = dict(
    ...
    occ3d=True,
    phase='train'
)
```

这会让 ego lidar 点先按 `[-50,-50,-5,50,50,3]` 过滤。后续 lifter 的 voxelizer 会再按 `[-40,-40,-1,40,40,5.4]` 过滤，所以 xy 上只是多保留了一些点再丢掉，但 z 上会提前丢掉 `3.0 < z < 5.4` 的点。Occ3D 的竖直范围本来就只有 16 层，这个错误会伤到较高物体和建筑结构。

应改为：

```python
train_dataset_config = dict(
    ...,
    pc_range=pc_range,
    occ3d=True,
)
val_dataset_config = dict(
    ...,
    pc_range=pc_range,
    occ3d=True,
)
```

### 3.4 图像增强配置从旧 Occ3D config 继承会直接破坏训练图像

当前能工作的 `img_voxel_react.py` 使用：

```python
input_shape = (1600, 864)
resize_lim = (1.0, 1.0)
H = 900
W = 1600
```

这表示原始 nuScenes 图像基本保持 1600 宽，只从 900 高 crop 到 864 高。

但已有 Occ3D config 使用：

```python
input_shape = (1600, 864)
resize_lim = (0.5, 0.5)
H = 900
W = 1600
```

训练时 `_sample_augmentation()` 会得到 resized image 约 `800x450`，然后 `ResizeCropFlipImage` 再 crop 成 `1600x864`。PIL 对越界 crop 会产生 padding，这等于训练时大面积黑边或空区域。这个问题和坐标系无关，但足以导致性能很差。

迁移 `img_voxel_react.py` 时，第一版 Occ3D baseline 应保持：

```python
resize_lim = (1.0, 1.0)
final_dim = (864, 1600)
```

如果显存压力太大，应同步降低 `input_shape` 和 `resize_lim`，例如真正使用 `input_shape=(800,450)`，而不是只把 resize 改成 0.5。

### 3.5 `occ_mask` 当前语义混乱

`LoadOccupancyOcc3D` 当前写入：

```python
results['occ_mask'] = semantics != 17
results['occ_cam_mask'] = mask_camera
results['occ_lidar_mask'] = mask_lidar
```

这里 `occ_mask` 只是 non-empty mask，不是 eval no-mask，也不是训练可见域 mask。

这会带来两个风险：

1. 如果把 `occ_mask` 加进 `return_keys`，并在 `eval_mask_flag=False` 时使用它，就会变成只评估非空体素，mIoU 定义错误。
2. 如果把它作为 loss mask，就会让模型只在非空体素上训练，free-space 几何会被削弱。

更合理的字段设计见第 5 节。

### 3.6 `train.py` 与 `eval.py` 的 Occ3D mask 行为不一致

`train.py` 中 Occ3D validation 仍用 `MeanIoU`，并按 `cfg.eval_mask_flag` 选择：

```python
if cfg.eval_mask_flag:
    occ_mask = result_dict['occ_cam_mask']
else:
    occ_mask = result_dict['occ_mask']
```

但独立 `eval.py` 中是：

```python
Metric_mIoU(use_lidar_mask=False, use_image_mask=True)
```

也就是永远使用 camera mask，`eval_mask_flag` 实际没有生效。后续如果要比较 cam mask / no cam mask，两条 eval 路径必须统一，否则训练日志和独立评估不可比。

## 4. 推荐坐标系设计

### 4.1 推荐路线 A：全 ego 坐标

这是 Occ3D 最自然、改动最少、最不容易踩坑的路线。

| 项 | 设置 |
| --- | --- |
| `pc_range` | `[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]` |
| `grid_size` | `0.4` |
| `H,W,D` | `200,200,16` |
| `LoadOccupancyOcc3D` | 直接输出 ego `occ_xyz` |
| `NuScenesDataset(occ3d=True)` | lidar 点转 ego，并按 Occ3D `pc_range` 过滤 |
| `NuScenesAdaptor` | `use_ego=True` |
| `projection_mat` | `ego2img` |
| `GaussianVoxelLearnear` | `pc_range=Occ3D pc_range`, `voxel_size=0.4` |
| `SparseConv3D` | `pc_range=Occ3D pc_range`, `grid_size=[0.4,0.4,0.4]` |
| `GaussianHead.cuda_kwargs` | `pc_min=[-40,-40,-1]`, `grid_size=0.4` |

这条路线下，Occ3D label/mask 不需要重采样。`occ_label[i,j,k]` 对应的物理中心就是：

```text
x = -40 + (i + 0.5) * 0.4
y = -40 + (j + 0.5) * 0.4
z =  -1 + (k + 0.5) * 0.4
```

它同时也是 renderer 使用的规则网格坐标。

### 4.2 备选路线 B：保留 lidar 坐标主轴

这条路线只有在强依赖 lidar 坐标预训练或与 SurroundOcc 完全共用模块时才建议考虑。它不是首选。

原因是 `localagg_react` 会把 query point 转成规则体素索引：

```python
points_int = ((pts - pc_min) / grid_size).to(torch.int)
```

CUDA forward 再用 `points_int` 查每个 voxel 的 Gaussian range。也就是说 head 假设 query points 与 `pc_min/grid_size/H/W/D` 定义的规则轴对齐网格一致。

如果只把 Occ3D ego grid 的 `occ_xyz` 逐点变到 lidar 坐标，而 label/mask 仍按 ego grid 展平，理论上每个点和 label 仍一一对应；但变换后的点不再是 Occ3D ego 规则网格，也不一定对齐 lidar grid。这样会让 renderer 的 `points_int` 出现 floor 误差、重复格或空洞。由于 nuScenes lidar 到 ego 有平移和可能的旋转，这条路线风险较高。

如果坚持走 lidar 主轴，更稳妥的做法是：

1. 把 Occ3D label/mask 从 ego grid 重采样到 lidar 规则 grid；
2. `occ_xyz` 使用 lidar 规则 grid；
3. `lidar_points` 保持 lidar；
4. `projection_mat` 使用 `lidar2img`；
5. eval 前再把预测从 lidar grid 映射回 Occ3D ego grid。

这会引入离散重采样误差和额外工程复杂度。考虑到 Occ3D 官方标注本来就在 ego grid，下游评估也按 ego grid，第一版不建议这样做。

## 5. Cam mask / no cam mask 的设计

Occ3D 的 mask 至少应分成四类字段，不要再用一个 `occ_mask` 承载所有含义。

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `occ_label` | `(200,200,16)` semantic label, `17=free` | 监督与评估 |
| `occ_xyz` | `(200,200,16,3)` query center | renderer query |
| `occ_cam_mask` | Occ3D `mask_camera` | camera-visible eval / 可选训练 mask |
| `occ_lidar_mask` | Occ3D `mask_lidar` | lidar-visible eval / 可选训练 mask |
| `occ_nonempty_mask` | `occ_label != 17` | 统计或 debug，不应用作 no-mask eval |
| `occ_train_mask` 或 `occ_loss_mask` | 由 config 选择出的训练 mask | loss sampling / loss filtering |

建议 loader 支持：

```python
train_mask_type = "none"      # "none" | "camera" | "lidar" | "camera_lidar" | "nonempty"
eval_mask_type = "camera"     # "camera" | "none" | "lidar"
```

其中：

- `eval_mask_type="camera"`：使用 `mask_camera` 评估，适合和 camera-visible benchmark 对齐。
- `eval_mask_type="none"`：不传 mask 或传 all-true mask，评估完整 Occ3D grid。
- `eval_mask_type="lidar"`：只在 lidar-visible 区域评估，主要用于诊断，不建议作为主结果。

### 5.1 eval 使用 cam mask

配置语义：

```python
dataset_name_flag = "occ3d"
occ3d_eval_mask = "camera"
```

metric 行为：

```python
Metric_mIoU(
    num_classes=18,
    use_image_mask=True,
    use_lidar_mask=False,
)
```

调用：

```python
miou_metric.add_batch(
    pred_occ,
    gt_occ,
    mask_lidar=None,
    mask_camera=occ_cam_mask,
)
```

### 5.2 eval 不使用 cam mask

配置语义：

```python
occ3d_eval_mask = "none"
```

metric 行为：

```python
Metric_mIoU(
    num_classes=18,
    use_image_mask=False,
    use_lidar_mask=False,
)
```

调用时可以传 mask，但 metric 不使用：

```python
miou_metric.add_batch(pred_occ, gt_occ, None, None)
```

注意：这里不是使用 `semantics != 17`。no-mask eval 应该让 free voxels 也参与 confusion matrix，否则 free/occupied false positive 的惩罚会消失。

### 5.3 训练 mask 怎么选

建议第一版 baseline 做两组：

| 训练 mask | 评估 mask | 目的 |
| --- | --- | --- |
| `train_mask_type="none"` | `eval_mask_type="none"` 和 `"camera"` 都报 | 建立完整 grid 监督 baseline |
| `train_mask_type="camera"` | `eval_mask_type="camera"` | 对齐 camera-visible eval，减少不可见区域噪声 |

如果训练使用 camera mask，最好让 head 在 sampling 阶段就只 render camera-visible query，而不是先 render 全部 640k voxel 后再在 loss 里过滤。当前 `GaussianHead.forward()` 调用：

```python
sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
```

它会始终 render 全量 grid。后续即使 loss 收到 mask，也省不了 renderer 显存。建议改成：

```python
occ_loss_mask = metas.get("occ_loss_mask", None)
sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, occ_loss_mask)
```

同时明确 `final_occ` 在只采样 mask 时不再能直接 reshape 成 `(200,200,16)`。所以实现上可以分两档：

1. baseline：训练和 eval 都 render 全量 grid，只在 loss/metric 里 mask，工程最稳。
2. 优化版：训练 render masked query，eval render 全量 grid。此时 head 需要区分 train/eval sampling。

第一版迁移建议先用 baseline，等坐标和指标稳定后再做 masked render 优化。

## 6. `img_voxel_react.py` 迁移配置建议

第一版 Occ3D config 不建议引入额外结构变化。应尽量保留 `img_voxel_react.py` 已经稳定的模型结构，只替换数据集空间定义。

关键差异：

```python
dataset_name_flag = "occ3d"
occ3d_eval_mask = "camera"  # 或 "none"

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
grid_size = 0.4
input_shape = (1600, 864)
resize_lim = (1.0, 1.0)
```

pipeline 应改为：

```python
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadOccupancyOcc3D",
        occ3d_path=occ3d_path,
        semantic=True,
        pc_range=pc_range,
        grid_size=0.4,
        model_coord="ego",
        train_mask_type="none",
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=True, num_cams=6),
]
```

dataset config 应显式传 `pc_range` 与 `return_keys`：

```python
occ3d_return_keys = [
    "img",
    "projection_mat",
    "image_wh",
    "occ_label",
    "occ_xyz",
    "occ_cam_mask",
    "occ_lidar_mask",
    "occ_nonempty_mask",
    "occ_loss_mask",
    "ori_img",
    "cam_positions",
    "focal_positions",
    "lidar_points",
    "lidar_pose",
    "ego_pose",
]

train_dataset_config = dict(
    type="NuScenesDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    pc_range=pc_range,
    occ3d=True,
    phase="train",
    return_keys=occ3d_return_keys,
)
```

模型中所有空间相关参数同步改成 Occ3D：

```python
model = dict(
    lifter=dict(
        type="GaussianVoxelLearnear",
        pc_range=pc_range,
        voxel_size=0.4,
        ...
    ),
    encoder=dict(
        deformable_model=dict(
            kps_generator=dict(pc_range=pc_range, scale_range=scale_range),
        ),
        refine_layer=dict(pc_range=pc_range, scale_range=scale_range),
        densify_layer=dict(
            type="DensifyOnly",
            pc_range=pc_range,
            scale_range=scale_range,
            ...
        ),
        spconv_layer=dict(
            pc_range=pc_range,
            grid_size=[0.4, 0.4, 0.4],
        ),
    ),
    head=dict(
        use_localagg_react=True,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200,
            W=200,
            D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4,
        ),
    ),
)
```

`empty_args` 建议与 Occ3D z-range 对齐：

```python
empty_args=dict(
    mean=[0, 0, 2.2],
    scale=[80, 80, 6.4],
)
```

如果为了复用旧 ckpt 保守起见，也可以先保留 `scale=[100,100,8]`，但 `mean=[0,0,2.2]` 比复制 SurroundOcc 的 `[0,0,-1]` 更合理。

## 7. 需要配套修改的代码设计

### 7.1 重构 `LoadOccupancyOcc3D`

建议把 `use_ego` 替换为更明确的参数：

```python
class LoadOccupancyOcc3D:
    def __init__(
        self,
        occ3d_path,
        semantic=True,
        pc_range=[-40, -40, -1, 40, 40, 5.4],
        grid_size=0.4,
        model_coord="ego",
        train_mask_type="none",
        perturb=False,
    ):
        assert model_coord in ["ego", "lidar"]
        assert train_mask_type in ["none", "camera", "lidar", "camera_lidar", "nonempty"]
```

加载 label 时：

```python
semantics = labels["semantics"].astype(np.int64)
mask_camera = labels["mask_camera"].astype(bool)
mask_lidar = labels["mask_lidar"].astype(bool)
mask_nonempty = semantics != 17
```

mask 选择：

```python
if train_mask_type == "none":
    occ_loss_mask = np.ones_like(mask_camera, dtype=bool)
elif train_mask_type == "camera":
    occ_loss_mask = mask_camera
elif train_mask_type == "lidar":
    occ_loss_mask = mask_lidar
elif train_mask_type == "camera_lidar":
    occ_loss_mask = mask_camera & mask_lidar
elif train_mask_type == "nonempty":
    occ_loss_mask = mask_nonempty
```

输出字段：

```python
results["occ_label"] = semantics
results["occ_cam_mask"] = mask_camera
results["occ_lidar_mask"] = mask_lidar
results["occ_nonempty_mask"] = mask_nonempty
results["occ_loss_mask"] = occ_loss_mask
```

坐标输出：

```python
xyz = self.xyz.copy()  # Occ3D ego grid

if self.model_coord == "ego":
    occ_xyz = xyz[..., :3]
elif self.model_coord == "lidar":
    # only if the rest of the model is explicitly lidar-based
    ego2lidar = results["ego2lidar"]
    occ_xyz = ego2lidar[None, None, None] @ xyz[..., None]
    occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
```

注意：推荐路线不使用 `model_coord="lidar"`。

路径定位也可以简化。当前 loader 通过 lidar 文件名 timestamp 去 `sample.json` 查 sample token，再去 `scene.json` 查 scene name。local sanity check 显示前 100 个训练样本都能匹配，但 pkl 里的 `info` 已经有 `token`、`scene_token`、`occ_path`。更稳的设计是让 dataset 把 `sample_idx`、`scene_token`、`occ_path` 传给 loader，优先直接构造路径，timestamp 逻辑只作为 fallback。

### 7.2 修正 `NuScenesDataset` 的 Occ3D 坐标输出

如果 `occ3d=True`，当前把 `lidar_points` 转到 ego 是对推荐路线有利的，但还需要同步三件事：

1. dataset config 必须传入 Occ3D `pc_range`。
2. `cam_positions/focal_positions` 如果后续模块使用，也应在 ego 坐标下计算。
3. `lidar_pose` 字段名容易误导；当点已经是 ego 坐标时，应额外提供 `points_pose=ego2global`，或者在 Occ3D 模式下让依赖点云 pose 的模块读取 `ego_pose`。

`cam_positions/focal_positions` 的 ego 版本可以按现有写法平移：

```python
img2ego = np.linalg.inv(ego2global) @ img2global
cam_position = img2ego @ viewpad @ [0, 0, 0, 1]
focal_position = img2ego @ viewpad @ [0, 0, f, 1]
```

当前 `img_voxel_react.py` 主链路主要使用 `projection_mat`，但把这些字段一起修正可以避免未来换 lifter 或 history 模块时再次错位。

### 7.3 `GaussianHead` 支持独立 loss mask

建议不要再让 `loss_input_convertion` 读取 `occ_mask` 这个含义模糊的名字。可以改为：

```python
loss_input_convertion = dict(
    pred_occ="pred_occ",
    gaussian="gaussian",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_loss_mask",
)
```

`GaussianHead.forward()` 返回：

```python
return {
    ...
    "occ_loss_mask": occ_loss_mask,
    "occ_cam_mask": occ_cam_mask,
    "occ_lidar_mask": occ_lidar_mask,
    "occ_nonempty_mask": occ_nonempty_mask,
}
```

第一版可以仍然 full render：

```python
sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
```

只在 loss 里用 `occ_loss_mask` 过滤。等 baseline 正常后，再做 train-only masked render。

### 7.4 统一 `train.py` 和 `eval.py` 的 Occ3D metric

建议用字符串配置替代布尔配置：

```python
occ3d_eval_mask = "camera"  # "camera" | "none" | "lidar"
```

构建 metric：

```python
use_image_mask = cfg.occ3d_eval_mask == "camera"
use_lidar_mask = cfg.occ3d_eval_mask == "lidar"

miou_metric = Metric_mIoU(
    num_classes=18,
    use_lidar_mask=use_lidar_mask,
    use_image_mask=use_image_mask,
)
```

add batch：

```python
mask_camera = result_dict.get("occ_cam_mask", None)
mask_lidar = result_dict.get("occ_lidar_mask", None)

miou_metric.add_batch(
    pred_occ,
    gt_occ,
    mask_lidar.squeeze(0).cpu().numpy() if mask_lidar is not None else None,
    mask_camera.squeeze(0).cpu().numpy() if mask_camera is not None else None,
)
```

这样 `occ3d_eval_mask="none"` 时 metric 会自然评估完整 grid。

## 8. 为什么现有 `LoadOccupancyOcc3D` 跑出来性能可能不满意

优先级从高到低：

1. **坐标投影错位**：`lidar_points` 和 `occ_xyz` 实际是 ego，但 `projection_mat` 是 lidar2img。deformable attention 采图像特征的位置会系统性偏移。
2. **训练图像被错误 crop/padding**：旧 Occ3D config 的 `resize_lim=0.5` 与 `input_shape=1600x864` 不匹配，训练图像质量会大幅下降。
3. **点云 z 范围被提前截断**：dataset 没传 Occ3D `pc_range`，导致 ego 点云中 `z>3` 的点在进入 lifter 前被丢弃。
4. **mask 语义混乱**：`occ_mask=semantics!=17` 容易被误用为 no-mask eval 或 train mask。no-mask eval 不应过滤 free；train mask 也不应默认 non-empty。
5. **eval 路径不一致**：`train.py` 看起来有 `eval_mask_flag`，但 `eval.py` 对 Occ3D 永远 `use_image_mask=True`，导致不同入口的结果不可比。
6. **`use_ego` 命名误导**：Occ3D loader 的 `use_ego=True` 反而会把 ego grid 再变换一次；这会让后续 config 调参非常容易踩坑。
7. **class weight 没有针对 Occ3D 重新统计**：当前手动权重来自现有 NuScenes/SurrOcc 风格配置。Occ3D 的 label 分布不同，建议统计 train split 的 18 类频率后重算，至少比较一次。
8. **empty Gaussian 先验仍是 SurroundOcc 风格**：`empty_args.mean=[0,0,-1]` 对 SurroundOcc z-range 是中心，但对 Occ3D z-range 不是中心。虽然 scale 很大能覆盖全局，但更合理的是改到 `[0,0,2.2]`。
9. **loader 路径查找过重**：每个 loader 初始化读 `sample.json/scene.json` 可以工作，但没有利用 pkl 里已有 token 信息。它主要影响鲁棒性和启动时间，不是 mIoU 的首要问题。

本地对一个 Occ3D sample 的 `.npz` header 做过 sanity check：

```text
shape:          (200, 200, 16)
total voxels:   640000
semantics dtype uint8, label range 0..17
mask_camera:   74354 true
mask_lidar:    82661 true
non-empty:     26776
non-empty cam: 17154
```

这说明 camera mask 不是 non-empty mask；它包含大量 free-space 可见体素。把 `semantics != 17` 当成 eval mask 会改变指标定义。

## 9. 推荐实验顺序

### E0：数据与坐标 sanity check

在训练前抽一个 batch 检查：

```text
occ_xyz min/max ~= [-39.8,-39.8,-0.8] 到 [39.8,39.8,5.2]
lidar_points min/max 在 Occ3D pc_range 内
projection_mat 来自 ego2img
points_int min/max 在 [0,199]x[0,199]x[0,15]
```

同时可视化一帧：

```text
ego lidar points
Occ3D non-empty voxel centers
Gaussian initial anchors
```

三者应该在同一个 ego frame 内重合。

### E1：全 ego + full grid loss

目的：先排除坐标错误。

```python
train_mask_type = "none"
occ3d_eval_mask = "camera"
```

同时在独立 eval 再跑：

```python
occ3d_eval_mask = "none"
```

这样能看到 camera-visible 与 full-grid 的差距。

### E2：全 ego + camera mask loss

目的：对齐 camera-visible metric。

```python
train_mask_type = "camera"
occ3d_eval_mask = "camera"
```

第一版仍 full render，只在 loss 过滤，避免引入 masked render 的 reshape 问题。

### E3：统计 Occ3D class weight

统计 train split 中 `occ_loss_mask` 范围内的 label 频率，重算 18 类 CE weight。至少比较：

```text
现有 manual_class_weight
Occ3D full-grid frequency weight
Occ3D camera-mask frequency weight
```

### E4：masked render 优化

等 E1/E2 正常后再做：

```text
train: 只 render occ_loss_mask=True 的 query
eval:  render full grid
```

这可以降低训练显存和时间，但需要 head 返回 full-grid prediction 的逻辑与 train loss sampled prediction 分开。

## 10. 迁移验收标准

满足以下条件后，才认为 Occ3D 迁移是干净的：

1. `occ_xyz`、`lidar_points`、Gaussian means、`projection_mat` 坐标系一致。
2. `NuScenesAdaptor(use_ego=True)` 在 Occ3D ego 路线下被使用。
3. dataset 显式传入 Occ3D `pc_range`。
4. 图像增强不会产生非预期 padding。
5. `occ_cam_mask`、`occ_lidar_mask`、`occ_nonempty_mask`、`occ_loss_mask` 语义分离。
6. `occ3d_eval_mask="camera"` 和 `"none"` 两种模式都能从同一个 eval 入口稳定运行。
7. no-mask eval 不使用 `semantics != 17` 作为 mask。
8. 第一版模型结构尽量保持 `img_voxel_react.py`，不同时引入 TopK densify、不同输入分辨率、不同 mask 策略等额外变量。

## 11. 最小可行改动摘要

最小可行 Occ3D baseline：

```text
1. 新建 img_voxel_react_occ3d.py，复制 img_voxel_react.py。
2. pc_range 改为 [-40,-40,-1,40,40,5.4]，grid_size 改为 0.4。
3. pipeline 使用 LoadOccupancyOcc3D(model_coord="ego")。
4. NuScenesAdaptor(use_ego=True)。
5. train/val dataset_config 传 pc_range=pc_range, occ3d=True。
6. data_aug_conf 保持 resize_lim=(1.0,1.0)，不要复用旧 Occ3D 的 0.5 resize。
7. lifter / encoder / spconv / head 全部同步 Occ3D pc_range 与 grid_size。
8. mask 字段分离，eval 用 occ3d_eval_mask 字符串控制。
9. 先 full render + full grid loss 跑通，再做 camera mask loss 和 masked render。
```

这条路线能最大限度隔离变量：如果 E1 仍然差，再分析模型容量、class weight、pretrain 或 Occ3D 标注分布；但在此之前，当前代码最明显的问题仍是坐标与数据管线没有完全对齐。
