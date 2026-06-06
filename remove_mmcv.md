# 剔除 mmcv / mmdet / mmdet3d / mmsegmentation 依赖 — 工程分析报告

> **目标：** 将当前代码库中除 `mmengine` 之外的所有 OpenMMLab 依赖全部移除，仅保留 `mmengine`。
>
> **当前依赖（来自 [`docs/installation.md`](docs/installation.md:17)）：**
> ```
> mmcv==2.0.1
> mmdet==3.0.0
> mmsegmentation==1.0.0
> mmdet3d==1.1.1
> ```
>
> **保留依赖：** `mmengine`（目前已大量使用其 Config / Registry / BaseModule / MMLogger 等）。

---

## 1. 总览

| 库 | 引用文件数 | 难度 | 预估工时（人天） |
|---|:---:|---|:---:|
| **mmcv** | ~15 | 中等 | 3–5 |
| **mmdet** | 2 | 低 | 0.5 |
| **mmdet3d** | 2 | 低 | 0.5–1 |
| **mmsegmentation** | ~22 | 高 | 7–10 |
| **合计** | ~35 | — | **11–17 人天** |

> 以下按库逐一分析每个引用点的具体用途、替换方案和工作量。

---

## 2. mmcv 依赖分析（~15 文件，3–5 人天）

### 2.1 [`mmcv.cnn.Scale`](model/encoder/gaussian_encoder/refine_module.py:3) — 8 处

**影响文件：**
- [`model/encoder/gaussian_encoder/refine_module.py`](model/encoder/gaussian_encoder/refine_module.py:3)
- [`model/encoder/gaussian_encoder/refine_module_v2.py`](model/encoder/gaussian_encoder/refine_module_v2.py:3)
- [`model/encoder/gaussian_encoder/refine_densify_module.py`](model/encoder/gaussian_encoder/refine_densify_module.py:3)
- [`model/encoder/gaussian_encoder/refine_densify_module_machine.py`](model/encoder/gaussian_encoder/refine_densify_module_machine.py:3)
- [`model/encoder/gaussian_encoder/densify_module.py`](model/encoder/gaussian_encoder/densify_module.py:3)
- [`model/encoder/gaussian_encoder/densify_low_cost.py`](model/encoder/gaussian_encoder/densify_low_cost.py:3)
- [`model/encoder/gaussian_encoder/topk_densify_module.py`](model/encoder/gaussian_encoder/topk_densify_module.py:3)
- [`model/encoder/gaussian_encoder/topk_module/topk_densify_v2.py`](model/encoder/gaussian_encoder/topk_module/topk_densify_v2.py:4)
- [`model/encoder/gaussian_encoder/topk_module/topk_semantic_densify.py`](model/encoder/gaussian_encoder/topk_module/topk_semantic_densify.py:4)

**用途：** 可学习的逐通道缩放因子，初始化值为 1.0。

```python
from mmcv.cnn import Scale
self.layers = nn.Sequential(
    *linear_relu_ln(embed_dims, 2, 2),
    nn.Linear(self.embed_dims, self.output_dim),
    Scale([1.0] * self.output_dim))
```

**替换方案（极简单，每处 5 分钟）：**
```python
# 直接用 nn.Parameter 实现同等功能
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value))

    def forward(self, x):
        return x * self.scale
```
将此实现放入 `model/utils/` 并全局替换 import 即可。

**工作量：** 0.3 天

---

### 2.2 [`mmcv.cnn.ConvModule`](model/backbone_img/cp_fpn.py:9) — 1 处

**影响文件：** [`model/backbone_img/cp_fpn.py`](model/backbone_img/cp_fpn.py:9)

**用途：** `ConvModule` 是 `Conv2d + Norm + Activation` 的组合封装。`CPFPN` 类中使用它来构建 lateral convolution 和 FPN convolution。

```python
l_conv = ConvModule(
    in_channels[i], out_channels, 1,
    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
```

**替换方案（简单，0.5 天）：**
直接用 `nn.Sequential(nn.Conv2d(...), norm_layer, activation)` 替换，或将 `ConvModule` 精简实现移入本地 utils。需注意 `conv_cfg` / `norm_cfg` / `act_cfg` 这些配置字典的解析逻辑。

**工作量：** 0.5 天

---

### 2.3 [`mmcv.cnn.build_activation_layer / build_norm_layer / build_dropout`](model/encoder/gaussian_encoder/ffn_module.py:3) — 1 处

**影响文件：** [`model/encoder/gaussian_encoder/ffn_module.py`](model/encoder/gaussian_encoder/ffn_module.py:3)

**用途：** 根据配置字典动态构建激活层、归一化层和 Dropout 层。

```python
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout

self.activate = build_activation_layer(act_cfg)
self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]
self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()
```

**替换方案（中等，1 天）：**
- `build_activation_layer`：直接用 `getattr(nn, act_cfg['type'])(**kwargs)` 替换（仅使用 `ReLU`）
- `build_norm_layer`：直接用 `nn.LayerNorm` 或 `nn.BatchNorm1d` 替换（当前只用 `LN`）
- `build_dropout`：直接用 `nn.Dropout` 替换

实际上当前代码中 `act_cfg` 只用 `ReLU`，`pre_norm` 只用 `LN`，所以可以直接硬编码为 `nn.ReLU` / `nn.LayerNorm` / `nn.Dropout`。

**工作量：** 0.5 天

---

### 2.4 [`mmcv.image.io.imread`](dataset/dataset_nusc_surroundocc_stream.py:5) — 2 处

**影响文件：**
- [`dataset/dataset_nusc_surroundocc_stream.py`](dataset/dataset_nusc_surroundocc_stream.py:5)
- [`dataset/transform_3d.py`](dataset/transform_3d.py:329)

**用途：** 读取图像文件，支持 `color_type` 参数（如 `'unchanged'`）。

```python
from mmcv.image.io import imread
img = mmcv.imread(fname, self.color_type)
```

**替换方案（简单，0.5 天）：**
使用 `cv2.imread` 或 `PIL.Image.open` + `np.array()` 替换。`mmcv.imread` 本质是对 OpenCV 的薄封装。已存在的 `transform_3d.py` 中也在用 `PIL.Image`，可以统一。

```python
import cv2
img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
```

**工作量：** 0.3 天

---

### 2.5 [`mmcv.imnormalize`](dataset/transform_3d.py:169) — 1 处

**影响文件：** [`dataset/transform_3d.py`](dataset/transform_3d.py:169)

**用途：** `NormalizeMultiviewImage` 类中对图像进行均值/标准差归一化。

```python
results["img"] = [
    mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
    for img in results["img"]
]
```

**替换方案（极简单，10 分钟）：**
```python
def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    img = (img - mean) / std
    if to_rgb:
        img = img[..., ::-1]  # BGR to RGB
    return img
```

**工作量：** 0.1 天

---

### 2.6 [`mmcv.bgr2hsv / mmcv.hsv2bgr`](dataset/transform_3d.py:247) — 2 处

**影响文件：** [`dataset/transform_3d.py`](dataset/transform_3d.py:247)

**用途：** `PhotoMetricDistortionMultiViewImage` 中的颜色空间转换。

```python
img = mmcv.bgr2hsv(img)
# ... random saturation/hue ...
img = mmcv.hsv2bgr(img)
```

**替换方案（简单，0.3 天）：**
使用 OpenCV 或自行实现：
```python
import cv2
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
```

**工作量：** 0.2 天

---

### 2.7 [`mmcv.ops.sigmoid_focal_loss / softmax_focal_loss`](loss/occupancy_loss.py:275) — 1 处

**影响文件：** [`loss/occupancy_loss.py`](loss/occupancy_loss.py:275)

**用途：** CUDA 加速的 Focal Loss 实现。当前代码中 CPU fallback 是被 `assert False` 阻止的，**实际上只用 CUDA 路径**。

```python
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmcv.ops import softmax_focal_loss as _softmax_focal_loss
```

**关键发现：** 文件中已经包含了完整的 **纯 PyTorch 参考实现**：
- `py_sigmoid_focal_loss()`（第 280 行）
- `py_focal_loss_with_prob()`（第 330 行）

当前 `CustomFocalLoss.forward()` 在 CUDA 可用时走 `mmcv.ops`，CPU 时 `assert False`（第 558 行）。替换时只需：
1. 将 `sigmoid_focal_loss()` 函数中的 `_sigmoid_focal_loss` 调用替换为 `py_sigmoid_focal_loss`
2. 将 `softmax_focal_loss()` 函数中的 `_softmax_focal_loss` 调用替换为纯 PyTorch 实现
3. 删除 `from mmcv.ops import ...` 和 `from mmdet.models.losses.utils import weight_reduce_loss`

**注意事项：** `mmcv.ops` 的 focal loss 支持 `cls_weight` 参数，纯 PyTorch 版本需支持同等语义。

**工作量：** 0.5 天

---

### 2.8 [`import mmcv`](train.py:6) — 无实际使用

**影响文件：** [`train.py`](train.py:6)

在 [`train.py`](train.py:6) 顶部有 `import mmcv`，但分析完整文件后，未发现除 mmcv 子模块外的直接调用。可能是残留 import。

**工作量：** 直接删除即可，0 天。

---

### 2.9 配置文件中的 `type='BN2d'` — 所有配置文件

分布在 85+ 个配置字典中：`norm_cfg=dict(type='BN2d', requires_grad=False)`。

`BN2d` 在 `mmcv` 中注册，本质上就是 `nn.BatchNorm2d`。替换方案：在 `model/utils/` 中注册一个别名为 `BN2d` 的包装即可，或在配置中将 `BN2d` 改为 `BatchNorm2d`。

**工作量：** 0.2 天（批量替换）

---

## 3. mmdet 依赖分析（2 文件，0.5 人天）

### 3.1 [`mmdet.models.losses.utils.weight_reduce_loss`](loss/occupancy_loss.py:277)

**影响文件：** [`loss/occupancy_loss.py`](loss/occupancy_loss.py:277)

**用途：** `py_focal_loss_with_prob()` 函数中使用，用于对 loss 进行加权和归约。

**替换方案（简单，0.3 天）：**
`weight_reduce_loss` 的逻辑非常简单：
```python
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean':
        loss = loss.mean() if avg_factor is None else loss.sum() / avg_factor
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
```
直接内联或用 `F.binary_cross_entropy` 自带的 `weight` 和 `reduction` 参数替代。

**工作量：** 0.2 天

---

## 4. mmdet3d 依赖分析（2 文件，0.5–1 人天）

### 4.1 [`mmdet3d.registry.MODELS`](model/segmentor/base_segmentor.py:2)

**影响文件：**
- [`model/segmentor/base_segmentor.py`](model/segmentor/base_segmentor.py:2)
- [`model/lifter/gaussian_initializer/resnet_secondfpn.py`](model/lifter/gaussian_initializer/resnet_secondfpn.py:4)

**用途：**

**a) [`base_segmentor.py`](model/segmentor/base_segmentor.py:28)：** 作为 `img_neck` 的 fallback builder，当 `mmseg.models.builder.build_neck` 失败时尝试用 `mmdet3d.registry.MODELS.build`。

```python
try:
    self.img_neck = builder.build_neck(img_neck)
except:
    self.img_neck = MODELS.build(img_neck)   # MODELS = mmdet3d.registry.MODELS
```

**b) [`resnet_secondfpn.py`](model/lifter/gaussian_initializer/resnet_secondfpn.py:21)：** 用于构建 `SECONDFPN` neck。

```python
self.img_neck = mmdet3dMODELS.build(neck_confifg)
```

**替换方案（中等，1 天）：**

`mmdet3d.registry.MODELS` 本质上也是一个 `Registry`。该代码库已经有自己的注册机制：
- [`model/segmentor/base_segmentor.py`](model/segmentor/base_segmentor.py:1) 使用 `mmseg.models.SEGMENTORS`
- [`model/lifter/base_lifter.py`](model/lifter/base_lifter.py:1) 使用 `mmseg.registry.MODELS`
- [`model/encoder/base_encoder.py`](model/encoder/base_encoder.py:1) 使用 `mmseg.registry.MODELS`

有两种策略：
1. 如果 `SECONDFPN` 类型在 `mmseg.models.necks` 中也有注册（通过 [`model/neck/__init__.py`](model/neck/__init__.py:1) 的 `from mmseg.models.necks import *`），则可以直接走 `mmseg` 的 registry
2. 如果需要彻底剔除 `mmdet3d`，则在自己的代码库中实现一个 `SECONDFPN`（这是一个相对标准的 FPN 变体）

**工作量：** 0.5–1 天

---

## 5. mmsegmentation 依赖分析（~22 文件，7–10 人天）

这是 **最复杂、工作量最大** 的部分，因为代码库的模型架构和训练流程深度依赖 mmseg 的注册机制和组件。

### 5.1 注册机制依赖（Registry & SEGMENTORS/HEADS/MODELS）— 核心架构

**影响文件（~18 个）：**

| 用途 | 引入方式 | 影响文件 |
|------|---------|---------|
| `SEGMENTORS.register_module()` | `from mmseg.models import SEGMENTORS` | [`base_segmentor.py`](model/segmentor/base_segmentor.py:1), [`bev_segmentor.py`](model/segmentor/bev_segmentor.py:2) |
| `HEADS.register_module()` | `from mmseg.models import HEADS` | [`base_head.py`](model/head/base_head.py:2) |
| `MODELS.register_module()` | `from mmseg.registry import MODELS` | [`base_encoder.py`](model/encoder/base_encoder.py:1), [`base_lifter.py`](model/lifter/base_lifter.py:1), 7 个 lifter 文件, [`convnext.py`](model/backbone_img/convnext.py:11), [`vovnetcp.py`](model/backbone_img/vovnetcp.py:20), [`cp_fpn.py`](model/backbone_img/cp_fpn.py:12), [`gaussian_encoder.py`](model/encoder/gaussian_encoder/gaussian_encoder.py:4), [`voxel_query_module.py`](model/encoder/gaussian_encoder/voxel_query_module.py:7) |
| `builder.build_backbone` | `from mmseg.models import builder` | [`base_segmentor.py`](model/segmentor/base_segmentor.py:21), [`resnet_secondfpn.py`](model/lifter/gaussian_initializer/resnet_secondfpn.py:20) |
| `builder.build_head` | `from mmseg.models import builder` | [`base_segmentor.py`](model/segmentor/base_segmentor.py:32) |
| `builder.build_neck` | `from mmseg.models import builder` | [`base_segmentor.py`](model/segmentor/base_segmentor.py:26) |
| `build_segmentor` | `from mmseg.models import build_segmentor` | [`train.py`](train.py:12), [`eval.py`](eval.py:13), [`eval_stream.py`](eval_stream.py:12), [`visualize.py`](visualize.py:25) |
| `build_backbone` | `from mmseg.models import build_backbone` | [`bev_segmentor.py`](bev_segmentor.py:3) |
| `mmseg.models.backbones.*` | wildcard import | [`model/backbone/__init__.py`](model/backbone/__init__.py:1) |
| `mmseg.models.necks.*` | wildcard import | [`model/neck/__init__.py`](model/neck/__init__.py:1) |

**分析：**

当前架构使用 `mmseg` 作为 **全局 Registry 提供者**。关键观察：

1. 代码库已经有自己的 Registry：`OPENOCC_DATASET`、`OPENOCC_LOSS`、`OPENOCC_TRANSFORMS`（基于 `mmengine.registry.Registry`）
2. `mmseg.registry.MODELS` 被用作组件注册中心，虽然代码库的组件都注册在上面，但 **Registry 本身来自 mmseg**
3. 配置文件中的 `type='ResNet'`、`type='FPN'` 等依赖 `mmseg` 的 Registry 来解析这些类型名
4. `build_segmentor()` 是 `mmseg` 提供的便利函数，本质是 `SEGMENTORS.build(cfg)`

**替换方案（核心架构改造，3–4 天）：**

这是整个迁移的核心难点，需要系统性地重构：

**Step 1：创建自己的全局 Registry（0.5 天）**
```python
# 在 model/__init__.py 或新文件 model/registry.py 中
from mmengine.registry import Registry
MODELS = Registry('gaussianformer_models')
BACKBONES = Registry('gaussianformer_backbones')
NECKS = Registry('gaussianformer_necks')
HEADS = Registry('gaussianformer_heads')
SEGMENTORS = Registry('gaussianformer_segmentors')
```

**Step 2：替换所有 `@XXX.register_module()` 装饰器（1 天）**
- `@SEGMENTORS.register_module()` → `@SEGMENTORS.register_module()`（只是 import 来源变）
- `@HEADS.register_module()` → `@HEADS.register_module()`
- `@MODELS.register_module()` → `@MODELS.register_module()`
- 涉及 ~18 个文件的 import 行修改

**Step 3：替换 `builder.build_*` 调用（0.5 天）**
- `builder.build_backbone(cfg)` → `BACKBONES.build(cfg)` 或 `MODELS.build(cfg)`
- `builder.build_neck(cfg)` → `NECKS.build(cfg)`
- `builder.build_head(cfg)` → `MODELS.build(cfg)`
- `build_segmentor(cfg)` → `SEGMENTORS.build(cfg)`
- `build_backbone(cfg)` → `BACKBONES.build(cfg)`

**Step 4：重构 wildcard import（1 天）**
- [`model/backbone/__init__.py`](model/backbone/__init__.py:1)：`from mmseg.models.backbones import *` 需要改为：显式导入需要的 backbone 类（`ResNet`、`VoVNetCP`、`ConvNeXt` 等）
- [`model/neck/__init__.py`](model/neck/__init__.py:1)：`from mmseg.models.necks import *` 需要改为：显式导入 `FPN`、`SECONDFPN` 等

**工作量：** 3–4 天

---

### 5.2 具体组件依赖 — Backbone (ResNet)

**现状：** 配置文件大量使用 `type='ResNet'`，该 ResNet 实现来自 [`model/backbone/__init__.py`](model/backbone/__init__.py:1) 的 wildcard import → `mmseg.models.backbones.ResNet`。

**关键参数依赖 mmcv：** 配置中使用 `norm_cfg=dict(type='BN2d', requires_grad=False)`、`dcn=dict(type='DCNv2', ...)`。

**替换方案（1–2 天）：**
- **方案 A（推荐）：** 使用 `torchvision.models.resnet50 / resnet101`，自行封装为适配当前接口的 `ResNet` 类，注册到自己的 `BACKBONES` Registry
- **方案 B：** 从 `mmseg` 源码中提取 `ResNet` 实现（代码量大，不推荐）

需要处理的细节：
- `frozen_stages` 逻辑
- `out_indices` 中间层输出
- `with_cp`（checkpoint）支持
- `DCNv2` 可选支持（仅在部分配置中使用 `dcn=dict(type='DCNv2', ...)`）—— 这可能还需要 `mmcv.ops.DeformConv2d`，是深层依赖。若不使用 DCN 配置则可跳过

**工作量：** 1.5 天

---

### 5.3 具体组件依赖 — Neck (FPN)

**现状：** 配置中使用 `type='FPN'`，来自 `mmseg.models.necks.FPN`。

**替换方案（1 天）：**
- 标准 FPN 的 PyTorch 实现非常多。可以直接从 `torchvision.ops` 或自行实现一个简化版 FPN
- 注意：`CPFPN`（[`model/backbone_img/cp_fpn.py`](model/backbone_img/cp_fpn.py:18)）已在代码库中独立实现，只需解耦 `ConvModule` 依赖
- `SECONDFPN`（来自 `mmdet3d`）需要单独处理

**工作量：** 1 天

---

### 5.4 具体组件依赖 — Loss (DiceLoss)

**影响文件：**
- [`loss/occupancy_loss.py`](loss/occupancy_loss.py:4)
- [`loss/gaussian_semantic_loss.py`](loss/gaussian_semantic_loss.py:6)
- [`loss/gaussian_semantic_mult_loss.py`](loss/gaussian_semantic_mult_loss.py:6)

**用途：** Dice Loss 用于语义分割。

```python
from mmseg.models.losses import DiceLoss
self.dice_loss = DiceLoss(class_weight=self.class_weights, loss_weight=2.0)
```

**替换方案（简单，0.5 天）：**
DiceLoss 是标准损失函数，可以直接实现：
```python
class DiceLoss(nn.Module):
    def __init__(self, class_weight=None, loss_weight=1.0, ...):
        ...
    def forward(self, pred, target):
        # 标准 Dice 实现
        ...
```

**工作量：** 0.3 天

---

### 5.5 训练/评估入口文件

**影响文件：** [`train.py`](train.py:12), [`eval.py`](eval.py:13), [`eval_stream.py`](eval_stream.py:12), [`visualize.py`](visualize.py:25)

这些文件的改动较小——仅需将 `from mmseg.models import build_segmentor` 替换为 `from model import SEGMENTORS; my_model = SEGMENTORS.build(cfg.model)`。

**工作量：** 0.3 天

---

### 5.6 配置文件中的类型引用

所有 `config/*.py` 中使用 `type='ResNet'`、`type='FPN'`、`type='SECONDFPN'` 的地方都需要确认新 Registry 中已注册同名组件。如果组件名保持不变（推荐），配置文件无需修改。

**工作量：** 验证，0.5 天

---

## 6. 汇总：按模块的改造清单

### 6.1 新建文件

| 文件 | 内容 | 工时 |
|------|------|:---:|
| `model/registry.py` | 全局 Registry 定义（MODELS, BACKBONES, NECKS, HEADS, SEGMENTORS） | 0.3 |
| `model/utils/mmcv_utils.py` | `Scale`, `ConvModule`, `imnormalize`, `build_activation_layer` 等替代实现 | 0.5 |
| `model/backbone/resnet.py` | 基于 torchvision 的自有 ResNet 封装 | 1.0 |
| `model/neck/fpn.py` | 自有 FPN 实现 | 0.5 |
| `loss/dice_loss.py` | 自有 DiceLoss 实现 | 0.2 |
| `loss/focal_loss.py` | 自有 FocalLoss（迁移自 `occupancy_loss.py` 中的 py_* 函数） | 0.3 |

### 6.2 需修改的核心文件（按优先级）

| 优先级 | 文件 | 改动内容 | 工时 |
|:---:|------|------|:---:|
| 🔴 P0 | [`model/__init__.py`](model/__init__.py:1) | 替换为自有 Registry 导入 | 0.1 |
| 🔴 P0 | [`model/backbone/__init__.py`](model/backbone/__init__.py:1) | 去掉 wildcard import，显式导入自有组件 | 0.3 |
| 🔴 P0 | [`model/neck/__init__.py`](model/neck/__init__.py:1) | 同上 | 0.2 |
| 🔴 P0 | [`model/segmentor/base_segmentor.py`](model/segmentor/base_segmentor.py:1) | 替换 Registry 和 builder 调用 | 0.3 |
| 🔴 P0 | [`model/segmentor/bev_segmentor.py`](model/segmentor/bev_segmentor.py:2) | 替换 Registry 和 build_backbone | 0.2 |
| 🔴 P0 | [`model/head/base_head.py`](model/head/base_head.py:1) | 替换 Registry | 0.1 |
| 🔴 P0 | [`model/lifter/base_lifter.py`](model/lifter/base_lifter.py:1) | 替换 Registry | 0.1 |
| 🔴 P0 | [`model/encoder/base_encoder.py`](model/encoder/base_encoder.py:1) | 替换 Registry | 0.1 |
| 🔴 P0 | [`train.py`](train.py:12) | 替换 build_segmentor，去掉 `import mmcv` | 0.2 |
| 🔴 P0 | [`eval.py`](eval.py:13) | 替换 build_segmentor | 0.1 |
| 🔴 P0 | [`eval_stream.py`](eval_stream.py:12) | 替换 build_segmentor | 0.1 |
| 🔴 P0 | [`visualize.py`](visualize.py:25) | 替换 build_segmentor | 0.1 |
| 🟡 P1 | 所有 `mmseg.registry.MODELS` → 自有 `MODELS` | 批量替换 ~12 个 lifter/encoder 文件 | 0.5 |
| 🟡 P1 | 所有 `mmcv.cnn.Scale` → 自有实现 | ~9 个文件 | 0.3 |
| 🟡 P1 | [`model/encoder/gaussian_encoder/ffn_module.py`](model/encoder/gaussian_encoder/ffn_module.py:3) | 替换 build_activation/norm/dropout | 0.3 |
| 🟡 P1 | [`model/backbone_img/cp_fpn.py`](model/backbone_img/cp_fpn.py:9) | 替换 ConvModule | 0.3 |
| 🟡 P1 | [`dataset/transform_3d.py`](dataset/transform_3d.py:5) | 替换 mmcv.imread/imnormalize/bgr2hsv/hsv2bgr | 0.5 |
| 🟡 P1 | [`dataset/dataset_nusc_surroundocc_stream.py`](dataset/dataset_nusc_surroundocc_stream.py:5) | 替换 mmcv.imread | 0.2 |
| 🟡 P1 | [`loss/occupancy_loss.py`](loss/occupancy_loss.py:275) | 替换 mmcv.ops focal loss 和 mmdet weight_reduce_loss | 0.5 |
| 🟡 P1 | [`model/lifter/gaussian_initializer/resnet_secondfpn.py`](model/lifter/gaussian_initializer/resnet_secondfpn.py:3) | 替换 mmdet3d.MODELS 和 mmseg.builder | 0.5 |
| 🟢 P2 | 所有配置文件 `norm_cfg=dict(type='BN2d')` | 替换为自有注册名 | 0.2 |
| 🟢 P2 | [`docs/installation.md`](docs/installation.md:19) | 更新安装文档 | 0.2 |
| 🟢 P2 | 测试与验证 | 完整训练 + 推理流程测试 | 2.0 |

---

## 7. 风险评估

| 风险 | 等级 | 说明 |
|------|:---:|------|
| **Registry 迁移导致组件找不到** | 高 | 17+ 个文件的 `@XXX.register_module()` 装饰器需要全部迁移，遗漏一个就会导致运行时 `KeyError`。建议分批迁移并写好单元测试 |
| **ResNet/FPN 权重不兼容** | 中 | 当前使用 `mmseg` 的 ResNet 可能有自定义的预训练权重加载逻辑。换用 `torchvision` 后需确保权重正确加载 |
| **DCNv2 依赖** | 中 | 部分配置使用 `dcn=dict(type='DCNv2', ...)`，如果剔除 mmcv，需要额外处理 DCNv2（它是 `mmcv.ops.DeformConv2d`）。如果实际不使用 DCN 配置则可忽略 |
| **SECONDFPN** | 低 | 仅在 `config/prob/` 下的 3 个配置中使用。可以自己实现或直接去掉相关配置 |
| **Focal Loss CUDA 性能回退** | 低 | `mmcv.ops` 的 CUDA focal loss 速度快。纯 PyTorch 版本在功能上等价但略慢。考虑到这是 loss 计算（非主要瓶颈），性能影响可接受 |
| **隐性 API 依赖** | 中 | `mmseg` 的 `BaseModule` / `ResNet` / `FPN` 等类可能在其他地方被通过 `isinstance` 或 `type()` 检查，迁移后需要全面回归测试 |

---

## 8. 建议的迁移策略

### 推荐顺序（分 4 个阶段）：

```
阶段 1（2–3 天）：工具层替换
├── 创建 model/registry.py（自有 Registry）
├── 创建 model/utils/mmcv_utils.py（Scale, imnormalize, bgr2hsv 等）
├── 替换所有 mmcv 图像处理函数
└── 替换 mmcv.cnn.Scale

阶段 2（3–4 天）：注册机制迁移
├── 替换所有 @MODELS/@SEGMENTORS/@HEADS 装饰器的 import 来源
├── 替换 builder.build_* 调用
├── 去掉 backbone/neck __init__.py 的 wildcard import
└── 替换 train.py / eval.py 中的 build_segmentor

阶段 3（3–5 天）：组件自有实现
├── 实现自有 ResNet（基于 torchvision）
├── 实现自有 FPN
├── 实现自有 DiceLoss
├── 替换 FocalLoss 为纯 PyTorch 版本
└── 处理 SECONDFPN 和 mmdet3d 依赖

阶段 4（2–3 天）：清理与测试
├── 更新所有配置文件（BN2d 等）
├── 更新安装文档
├── 完整训练流程测试
├── 推理流程测试
└── 性能对比验证
```

---

## 9. 总结

| 指标 | 数值 |
|------|:---:|
| **需修改/新建的文件总数** | ~40–45 |
| **预估总工时** | **11–17 人天** |
| **核心难度** | mmseg 的 Registry + 组件（ResNet/FPN）替换 |
| **最大单项工时** | Registry 迁移（3–4 天）+ 自有 ResNet 实现（1.5 天） |
| **可并行化程度** | 中等（阶段 1 和阶段 3 的部分工作可并行） |
| **回退风险** | 低（每个阶段完成后可单独验证） |

**核心结论：** 剔除 mmcv/mmdet/mmdet3d/mmseg 的工程量属于 **中等偏大**，主要是因为 `mmsegmentation` 渗透进了整个模型构建流程（Registry、组件注册、builder 模式）。但由于 `mmengine` 已经提供了 `Registry` 和 `BaseModule` 等基础设施，**不需要从零搭建注册体系**，只需要将注册中心从 `mmseg` 迁移到自己的命名空间即可。最大风险在于 ResNet/FPN 这些具体模型的迁移，建议使用 `torchvision` 作为替代。
