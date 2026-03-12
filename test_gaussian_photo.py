import torch
import sys
sys.path.insert(0, '.')
from model.encoder.gaussian_encoder.gaussian_photo import GaussianPhoto

# 模拟数据
B = 1  # batch size
N = 6  # number of cameras
C = 3  # channels
H, W = 256, 256  # image size
num_pts = 100  # number of Gaussians

# 随机图像和掩码
img = torch.randn(B, N, C, H, W)  # 归一化后的图像可能包含负值
mask_img = torch.rand(B, N, 1, H, W) > 0.5  # 二值掩码

# 高斯参数
means = torch.randn(B, num_pts, 3)
quats = torch.randn(B, num_pts, 4)
scales = torch.rand(B, num_pts, 3) * 0.1
opacities = torch.rand(B, num_pts, 1)
semantics = torch.randint(0, 17, (B, num_pts, 1))

# 创建 GaussianPrediction 对象（使用命名元组）
from model.encoder.gaussian_encoder.utils import GaussianPrediction
gaussians = GaussianPrediction(
    means=means,
    scales=scales,
    rotations=quats,
    opacities=opacities,
    semantics=semantics,
    original_means=None,
    delta_means=None
)

# 元数据
projection_mat = torch.randn(B, N, 4, 4)  # 假设的投影矩阵
intrinsic = torch.randn(B, N, 3, 3)  # 内参矩阵
image_wh = torch.tensor([[[W, H]]]).repeat(B, N, 1).float()  # 图像宽高

metas = {
    'projection_mat': projection_mat,
    'intrinsic': intrinsic,
    'image_wh': image_wh
}

# 实例化模型
model = GaussianPhoto()

# 前向传播
print("Running forward pass...")
rendered = model(img, mask_img, gaussians, metas)
print(f"Rendered shape: {rendered.shape}")
print(f"Rendered min/max: {rendered.min().item():.3f}, {rendered.max().item():.3f}")
print(f"Rendered mean: {rendered.mean().item():.3f}")

# 检查是否全为背景颜色（接近背景颜色值）
background_color = torch.tensor([-2.117, -2.035, -1.804], dtype=torch.float32)
# 计算差异
diff = (rendered - background_color.view(1,1,3,1,1)).abs().mean()
print(f"Average diff from background color: {diff.item():.3f}")

# 如果差异很小，说明渲染结果全是背景色
if diff < 0.1:
    print("WARNING: Rendered output appears to be mostly background color!")
else:
    print("Rendered output seems to contain other colors.")