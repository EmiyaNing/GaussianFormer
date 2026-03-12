import torch
import sys
sys.path.insert(0, '.')
from model.encoder.gaussian_encoder.gaussian_photo import GaussianPhoto
from model.encoder.gaussian_encoder.utils import GaussianPrediction

# 模拟数据
B = 1
N = 6
C = 3
H, W = 256, 256
num_pts = 100

# 图像和掩码：使用随机值
img = torch.randn(B, N, C, H, W)  # 归一化后的图像
mask_img = (torch.rand(B, N, 1, H, W) > 0.5).float()

# 高斯参数：将点放在相机前方，并确保它们在图像内
# 在相机坐标系中，将点放在 z=10 附近，x 和 y 在 [-5,5] 范围内，投影后应落在图像中心附近
means = torch.randn(B, num_pts, 3)
# 调整位置：使深度为正，且 x,y 在合理范围内
means[:, :, 2] = means[:, :, 2].abs() + 5  # 深度 5-?
means[:, :, 0] = means[:, :, 0] * 2  # 水平方向
means[:, :, 1] = means[:, :, 1] * 2  # 垂直方向

quats = torch.randn(B, num_pts, 4)
scales = torch.rand(B, num_pts, 3) * 0.1
opacities = torch.rand(B, num_pts, 1)
semantics = torch.randint(0, 17, (B, num_pts, 1))

gaussians = GaussianPrediction(
    means=means,
    scales=scales,
    rotations=quats,
    opacities=opacities,
    semantics=semantics,
    original_means=None,
    delta_means=None
)

# 构建投影矩阵（lidar2cam 外参矩阵）：假设为单位矩阵（lidar 和相机坐标系对齐）
lidar2cam = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)  # (B,N,4,4)
# 内参矩阵：简单的针孔相机模型
fx, fy = 500., 500.
cx, cy = W/2, H/2
intrinsic = torch.tensor([[[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]]], dtype=torch.float32).repeat(B, N, 1, 1)
# 投影矩阵：lidar2img = intrinsic @ lidar2cam[:3]
# 但代码中 projection_mat 应该是 lidar2img（4x4 齐次坐标）。我们构造一个简单的。
# 我们将构造一个将 3D 点投影到图像平面的矩阵。
# 使用 intrinsic 作为 3x3 矩阵，并添加一行 [0,0,0,1]？
# 实际上 projection_mat 的形状是 (B,N,4,4)，所以我们需要一个 4x4 矩阵，其中最后一行是 [0,0,0,1]。
# 我们将 intrinsic 放在左上角 3x3，平移列设置为零。
proj = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
proj[:, :, :3, :3] = intrinsic[:, :, :3, :3]  # 设置内参
# 注意：外参是单位矩阵，所以投影矩阵就是内参矩阵（扩展为 4x4）。
# 但投影矩阵需要将齐次坐标 (x,y,z,1) 变换到 (x', y', z', w')，其中 x' = fx*x + cx*z, 等等。
# 实际上，标准的投影矩阵是 [fx, 0, cx, 0; 0, fy, cy, 0; 0,0,1,0]（3x4）。
# 我们使用 4x4 矩阵，其中第三行是 [0,0,1,0]，第四行是 [0,0,0,1]。
proj[:, :, 0, 0] = fx
proj[:, :, 0, 2] = cx
proj[:, :, 1, 1] = fy
proj[:, :, 1, 2] = cy
proj[:, :, 2, 2] = 1.0
proj[:, :, 2, 3] = 0.0
proj[:, :, 3, 2] = 0.0
proj[:, :, 3, 3] = 1.0
projection_mat = proj

image_wh = torch.tensor([[[W, H]]], dtype=torch.float32).repeat(B, N, 1)

metas = {
    'projection_mat': projection_mat,
    'intrinsic': intrinsic,
    'image_wh': image_wh
}

# 实例化模型
model = GaussianPhoto()

print("Running forward pass with points in view...")
rendered = model(img, mask_img, gaussians, metas)
print(f"Rendered shape: {rendered.shape}")
print(f"Rendered min/max: {rendered.min().item():.3f}, {rendered.max().item():.3f}")
print(f"Rendered mean: {rendered.mean().item():.3f}")

# 检查 mask 是否全为 False
# 我们可以在 forward 方法中添加调试代码，但暂时先这样。

# 计算与背景颜色的差异
background_color = torch.tensor([-2.117, -2.035, -1.804], dtype=torch.float32)
diff = (rendered - background_color.view(1,1,3,1,1)).abs().mean()
print(f"Average diff from background color: {diff.item():.3f}")

if diff < 0.1:
    print("WARNING: Rendered output appears to be mostly background color!")
else:
    print("Rendered output seems to contain other colors.")

# 保存渲染图像以供可视化
import matplotlib.pyplot as plt
for i in range(N):
    img_np = rendered[0, i].permute(1,2,0).detach().cpu().numpy()
    # 归一化到 [0,1] 以便显示
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    plt.imsave(f'rendered_view_{i}.png', img_np)
    print(f"Saved rendered view {i}")