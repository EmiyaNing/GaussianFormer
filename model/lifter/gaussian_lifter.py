import torch, torch.nn as nn, math, os
import numpy as np
from einops import rearrange
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from ..utils.sampler import DistributionSampler
from .lidar_processor import LidarVoxelProcessor

# try:
#     from pointops import farthest_point_sampling
# except:
#     print("farthest_point_sampling import error.")

try:
    from pointops import furthestsampling as farthest_point_sampling
except ImportError:
    print("furthestsampling import error, using alternative method.")
    def farthest_point_sampling(points, num_points):
        if points.shape[0] <= num_points:
            return torch.arange(points.shape[0], device=points.device)
        else:
            return torch.randperm(points.shape[0], device=points.device)[:num_points]

@MODELS.register_module()
class GaussianLifterV2(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",

        num_samples=64,
        pc_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=0.5,
        occ_resolution=[200, 200, 16],
        empty_label=17,
        anchors_per_pixel=1,
        random_sampling=True,
        projection_in=None,
        initializer=None,
        initializer_img_downsample=None,
        pretrained_path=None,
        deterministic=True,
        random_samples=0,

        use_lidar_init=True,           # 使用LiDAR初始化
        lidar_fusion_frames=1,            # 融合帧数
        lidar_min_points=1,        # 体素内最小点数
        max_anchors=25600,               # 最大锚点数量
        lidar_voxel_size=0.15,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.include_opa = include_opa
        self.semantics = semantics
        self.semantic_dim = semantic_dim

        self.use_lidar_init = use_lidar_init
        self.lidar_fusion_frames = lidar_fusion_frames
        self.lidar_min_points = lidar_min_points
        self.max_anchors = max_anchors

        print(f"=== GaussianLifterV2 初始化 ===")
        print(f"use_lidar_init: {self.use_lidar_init}")
        print(f"max_anchors: {self.max_anchors}")

        self.lidar_processor = LidarVoxelProcessor(
            pc_range=pc_range,
            lidar_voxel_size=lidar_voxel_size,
            max_points_per_voxel=50,
            num_sweeps=lidar_fusion_frames
        )
        self.lidar_voxel_size=lidar_voxel_size

        self.random_samples = random_samples
        if random_samples > 0:
            self.random_anchors = self.init_random_anchors()
                    
        if not self.use_lidar_init:
            print("使用图像初始化模式")
            scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
            if scale_activation == "sigmoid":
                scale = safe_inverse_sigmoid(scale)

            rots = torch.zeros(num_anchor, 4, dtype=torch.float)
            rots[:, 0] = 1

            if include_opa:
                opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
            else:
                opacity = torch.ones((num_anchor, 0), dtype=torch.float)

            if semantics:
                assert semantic_dim is not None
            else:
                semantic_dim = 0
            semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
            anchor = torch.cat([scale, rots, opacity, semantic], dim=-1)

            self.num_anchor = num_anchor
            self.anchor = nn.Parameter(
                torch.tensor(anchor, dtype=torch.float32),
                requires_grad=anchor_grad,
            )
            self.instance_feature = nn.Parameter(
                torch.zeros([num_anchor + random_samples, self.embed_dims]),
                requires_grad=feat_grad,
            )
            projection_in = embed_dims * 4 if projection_in is None else projection_in
            self.projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(projection_in, num_samples + 1),
            )
            self.sampler = DistributionSampler()
            self.num_samples = num_samples
            self.register_buffer("depth_bins", torch.linspace(
                1.0, 72.0, self.num_samples, dtype=torch.float), persistent=False)
            self.register_buffer("pc_start", torch.tensor(
                pc_range[:3], dtype=torch.float), persistent=False)
            self.anchors_per_pixel = anchors_per_pixel
            self.random_sampling = random_sampling
            if initializer is not None:
                self.initialize_backbone = MODELS.build(initializer)
            else:
                self.initialize_backbone = None
            self.initializer_img_downsample = initializer_img_downsample
        
        else:
            print("使用雷达初始化模式")
            self.num_anchor = num_anchor  # 保持兼容性，但实际使用max_anchors
            param_dim = 3 + 3 + 4 + (1 if include_opa else 0) + (semantic_dim if semantics else 0)
            self.anchor = nn.Parameter(
                torch.zeros(num_anchor, param_dim, dtype=torch.float32),
                requires_grad=anchor_grad,
            )

            self.instance_feature = nn.Parameter(
                torch.zeros([max_anchors + random_samples, self.embed_dims]),
                requires_grad=feat_grad,
            )
            self.projection = None
            self.sampler = None
            self.depth_bins = None
            self.initialize_backbone = None
            self.initializer_img_downsample = None

        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.occ_resolution = occ_resolution
        self.empty_label = empty_label
        
        self.pretrained_path = pretrained_path
        self.deterministic = deterministic
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            if 'instance_feature' in ckpt:
                del ckpt['instance_feature']
            if 'anchor' in ckpt:
                del ckpt['anchor']
            print(self.load_state_dict(ckpt, strict=False))
            print("Gaussian Initializer Weight Loaded Successfully.")



    def init_random_anchors(self):
        num_anchor = self.random_samples

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)
        
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        if self.scale_act == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if self.include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if self.semantics:
            semantic_dim = self.semantic_dim
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)
        anchor = nn.Parameter(anchor, True)
        return anchor


    def init_weights(self):
        if self.pretrained_path is not None:
            return
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)


    def process_lidar_data(self, metas):
        """处理LiDAR数据，返回体素化结果"""
        #print("=== 进入 process_lidar_data ===")
        
        batch_size = len(metas['lidar_points'])

        device = next(self.parameters()).device  # 获取模型所在的设备
        
        all_voxel_centers = []
        all_point_counts = []
        
        for i in range(batch_size):
            # 获取当前帧LiDAR数据
            current_lidar = metas['lidar_points'][i]  # [N, 4] 或 [N, 3]
            current_pose = metas['lidar_pose'][i]     # 当前帧位姿
            current_lidar = current_lidar[:, :3]

            
            # # 获取历史帧LiDAR数据（从sweeps中获取）
            # previous_lidars = []
            # previous_poses = []
            
            # if 'lidar_sweeps' in metas and i < len(metas['lidar_sweeps']):
            #     sweeps = metas['lidar_sweeps'][i]
            #     # 取最近的前几帧（不包括当前帧）
            #     num_previous = min(self.lidar_fusion_frames - 1, len(sweeps))
            #     for j in range(num_previous):
            #         sweep_data = sweeps[j]
            #         previous_lidars.append(sweep_data['points'])
            #         previous_poses.append(sweep_data['pose'])

            # print(f"Batch {i}:")
            # print(f"  LiDAR数据形状: {current_lidar.shape if hasattr(current_lidar, 'shape') else 'None'}")
            # print(f"  位姿数据形状: {current_pose.shape if hasattr(current_pose, 'shape') else 'None'}")
            
            if current_lidar is None or len(current_lidar) == 0:
                print(f" Batch {i} 的LiDAR数据为空")
            
            # 使用LiDAR处理器进行融合和体素化
            try:

                
                self.lidar_processor = self.lidar_processor.to(device)

                lidar_result = self.lidar_processor(
                    current_lidar=current_lidar,
                    current_pose=current_pose,
                    previous_lidars=None,
                    previous_poses=None
                )
                
                # print(f"  非空体素中心形状: {lidar_result['voxel_centers'].shape}")
                # print(f"  点计数形状: {lidar_result['point_counts'].shape}")
                print(f"  非空体素数量: {lidar_result['num_voxels']}")
                
                all_voxel_centers.append(lidar_result['voxel_centers'])
                all_point_counts.append(lidar_result['point_counts'])
                
            except Exception as e:
                print(f"  LiDAR处理失败: {e}")
                all_voxel_centers.append(torch.zeros((0, 3)))
                all_point_counts.append(torch.zeros((0,)))
        
        return all_voxel_centers, all_point_counts



    def init_anchors_from_lidar(self, voxel_centers_list, point_counts_list, batch_size):
        """根据LiDAR体素化结果初始化高斯锚点"""
        anchors_list = []
        actual_anchor_counts = []  # 记录每个batch实际使用的锚点数
        
        for b in range(batch_size):
            voxel_centers = voxel_centers_list[b]
            point_counts = point_counts_list[b]
            
            if len(voxel_centers) == 0:
                # 如果没有体素，使用默认初始化
                print("无体素，默认初始化")
                default_anchor = self._init_default_anchors(1)[0]
                anchors_list.append(default_anchor)
                actual_anchor_counts.append(1)
                continue
            
            num_voxels = len(voxel_centers)
            # 动态确定实际锚点数：不超过max_anchors
            num_anchors_actual = min(num_voxels, self.max_anchors)
            actual_anchor_counts.append(num_anchors_actual)
            
            # 采样或选择策略
            if num_voxels > self.max_anchors:
                # 加权采样：点越多的体素，被选中的概率越大
                #print("采样")
                weights = point_counts.float() / point_counts.sum()
                indices = torch.multinomial(weights, self.max_anchors, replacement=False)
                selected_centers = voxel_centers[indices]
                selected_counts = point_counts[indices]
            else:
                # 使用全部体素
                #print("体素不足，使用全部")
                selected_centers = voxel_centers
                selected_counts = point_counts
            
            # 归一化坐标到 [0, 1]
            pc_range_tensor = torch.tensor(self.pc_range, device=selected_centers.device)
            normalized_xyz = (selected_centers - pc_range_tensor[:3]) / (pc_range_tensor[3:] - pc_range_tensor[:3])
            
            if self.xyz_act == "sigmoid":
                normalized_xyz = safe_inverse_sigmoid(normalized_xyz)
            
            # 初始化尺度（基于体素尺寸）
            scale = torch.ones(num_anchors_actual, 3, device=selected_centers.device) * self.lidar_voxel_size
            if self.scale_act == "sigmoid":
                scale = safe_inverse_sigmoid(scale)
            
            # 初始化旋转（单位四元数）
            rots = torch.zeros(num_anchors_actual, 4, device=selected_centers.device)
            rots[:, 0] = 1
            
            # 初始化不透明度（基于点云密度）
            if self.include_opa:
                # 使用点数量作为密度指标，归一化到0-1
                density = selected_counts.float() / selected_counts.max().clamp(min=1)
                opacity = safe_inverse_sigmoid(density.unsqueeze(1))
            else:
                opacity = torch.zeros((num_anchors_actual, 0), device=selected_centers.device)
            
            # 初始化语义特征
            if self.semantics:
                semantic_dim = self.semantic_dim
                semantic = torch.randn(num_anchors_actual, semantic_dim, device=selected_centers.device)
            else:
                semantic = torch.zeros((num_anchors_actual, 0), device=selected_centers.device)
            
            # 组合锚点参数
            anchor = torch.cat([normalized_xyz, scale, rots, opacity, semantic], dim=-1)
            #print(f"anchor特征: {anchor}")
            anchors_list.append(anchor)
        
        return anchors_list, actual_anchor_counts

    
    
    def _init_default_anchors(self, batch_size):
        """默认初始化方法（备用）"""
        device = self.anchor.device
        
        # 在点云范围内均匀分布
        xyz = torch.rand(self.num_anchor, 3, device=device)
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)

        scale = torch.ones(self.num_anchor, 3, device=device) * 0.5
        if self.scale_act == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(self.num_anchor, 4, device=device)
        rots[:, 0] = 1

        if self.include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((self.num_anchor, 1), device=device))
        else:
            opacity = torch.ones((self.num_anchor, 0), device=device)

        if self.semantics:
            semantic_dim = self.semantic_dim
            semantic = torch.randn(self.num_anchor, semantic_dim, device=device)
        else:
            semantic = torch.zeros((self.num_anchor, 0), device=device)
        
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)
        return anchor.unsqueeze(0).repeat(batch_size, 1, 1)




    def forward(self, metas, **kwargs):
        #原有方法
        if not self.use_lidar_init:
            return self._forward_image_based(metas, **kwargs)
        
        # 处理LiDAR数据
        if 'lidar_points' not in metas:
            print("默认初始化")
            return self._forward_fallback(metas, **kwargs)
            
        voxel_centers_list, point_counts_list = self.process_lidar_data(metas)
        batch_size = len(voxel_centers_list)
        
        # 根据LiDAR体素初始化锚点
        lidar_anchors_list, actual_anchor_counts = self.init_anchors_from_lidar(
            voxel_centers_list, point_counts_list, batch_size)
        
        # 处理每个batch的锚点和特征
        anchors_list = []
        features_list = []
        
        #max_anchors_in_batch = max(actual_anchor_counts) if actual_anchor_counts else 0
        
        for b, (lidar_anchors, actual_count) in enumerate(zip(lidar_anchors_list, actual_anchor_counts)):
            # 如果实际锚点数小于最大锚点数，填充到最大数量
            if actual_count < self.max_anchors:
                #print("not enough anchor，random")
                num_random_needed = self.max_anchors - actual_count
                random_anchors = self._init_random_anchors_batch(num_random_needed, device=lidar_anchors.device)
                
                final_anchors = torch.cat([lidar_anchors, random_anchors], dim=0)
            
                #print(f"填补完成: LiDAR锚点 {actual_count} + 随机锚点 {num_random_needed} = 总计 {final_anchors.shape[0]}")

            else:
                final_anchors = lidar_anchors
            
            anchors_list.append(final_anchors)
            
            # 对应的instance_feature
            current_features = self.instance_feature[:self.max_anchors, :]
            features_list.append(current_features)
        
        # 堆叠成batch
        anchor = torch.stack(anchors_list)
        instance_feature = torch.stack(features_list)
        
        # 处理随机样本
        if self.random_samples > 0:
            random_anchors = torch.tile(self.random_anchors[None], (batch_size, 1, 1))
            anchor = torch.cat([anchor, random_anchors], dim=1)
            # 对应的随机特征
            random_features = self.instance_feature[self.max_anchors:self.max_anchors+self.random_samples, :]
            random_features = torch.tile(random_features[None], (batch_size, 1, 1))
            instance_feature = torch.cat([instance_feature, random_features], dim=1)
        
        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': anchor[0].clone(),
            'pixel_logits': None,  # LiDAR初始化不需要深度估计
            'pixel_gt': None,      # LiDAR初始化不需要深度GT
        }


    def _init_random_anchors_batch(self, num_anchors, device):
        """初始化指定数量的随机锚点"""
        xyz = torch.rand(num_anchors, 3, device=device)
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)
    
        scale = torch.ones(num_anchors, 3, device=device) * 0.5
        if self.scale_act == "sigmoid":
            scale = safe_inverse_sigmoid(scale)
    
        rots = torch.zeros(num_anchors, 4, device=device)
        rots[:, 0] = 1
    
        if self.include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchors, 1), device=device))
        else:
            opacity = torch.ones((num_anchors, 0), device=device)
    
        if self.semantics:
            semantic_dim = self.semantic_dim
            semantic = torch.randn(num_anchors, semantic_dim, device=device)
        else:
            semantic = torch.zeros((num_anchors, 0), device=device)
    
        return torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)
    


    def _forward_image_based(self, metas, **kwargs):
        if self.initialize_backbone is not None:
            b, n = kwargs["imgs"].shape[:2]
            initialize_input = kwargs["imgs"].flatten(0, 1)
            if self.initializer_img_downsample is not None:
                initialize_input = nn.functional.interpolate(
                    initialize_input, scale_factor=self.initializer_img_downsample, 
                    mode='bilinear', align_corners=True)
            secondfpn_out = self.initialize_backbone(initialize_input)
            secondfpn_out = secondfpn_out.unflatten(0, (b, n))
        else:
            secondfpn_out = kwargs["secondfpn_out"]
        
        b, n, _, h, w = secondfpn_out.shape
        feature = rearrange(secondfpn_out, 'b n c h w -> b n h w c')
        logits = self.projection(feature) # b, n, h, w, d + 1

        projection_mat = metas["projection_mat"].inverse() # img2lidar
        u = (torch.arange(w, dtype=feature.dtype, device=feature.device) + 0.5) / w
        v = (torch.arange(h, dtype=feature.dtype, device=feature.device) + 0.5) / h
        uv = torch.stack([
            u[None, :].expand(h, w), v[:, None].expand(h, w)], dim=-1) # h, w, 2
        uv = uv[None, None].expand(b, n, h, w, 2) * metas['image_wh'][:, :, None, None] # b, n, h, w, 2
        uvd = uv.unsqueeze(4).expand(b, n, h, w, self.num_samples, 2)
        uvd1 = torch.cat([uvd, torch.ones_like(uvd)], dim=-1) # b, n, h, w, d, 4
        uvd1[..., :3] = uvd1[..., :3] * self.depth_bins.view(1, 1, 1, 1, -1, 1)
        anchor_pts = projection_mat[:, :, None, None, None] @ uvd1[..., None]
        anchor_pts = anchor_pts.squeeze(-1)[..., :3]
        if kwargs.get("benchmarking", False):
            anchor_gt = None
        else:
            oob_mask = (anchor_pts[..., 0] < self.pc_range[0]) | (anchor_pts[..., 0] >= self.pc_range[3]) | \
                       (anchor_pts[..., 1] < self.pc_range[1]) | (anchor_pts[..., 1] >= self.pc_range[4]) | \
                       (anchor_pts[..., 2] < self.pc_range[2]) | (anchor_pts[..., 2] >= self.pc_range[5])
            #体素索引
            anchor_idx = (anchor_pts - self.pc_start.view(1, 1, 1, 1, 1, 3)) / self.voxel_size
            anchor_idx = anchor_idx.to(torch.int)
            anchor_idx[..., 0].clamp_(0, self.occ_resolution[0] - 1)
            anchor_idx[..., 1].clamp_(0, self.occ_resolution[1] - 1)
            anchor_idx[..., 2].clamp_(0, self.occ_resolution[2] - 1)
            #获取对应占据标签
            occupancy = metas["occ_label"]
            valid_mask = metas["occ_cam_mask"]
            anchor_occ = torch.stack([occ[idx[..., 0], idx[..., 1], idx[..., 2]] for occ, idx in zip(occupancy, anchor_idx)])
            anchor_occ[oob_mask] = self.empty_label
            anchor_valid = torch.stack([occ[idx[..., 0], idx[..., 1], idx[..., 2]] for occ, idx in zip(valid_mask, anchor_idx)])
            anchor_valid[oob_mask] = False
            anchor_gt = (anchor_occ != self.empty_label) & anchor_valid
            anchor_gt = torch.cat([anchor_gt, ~torch.any(anchor_gt, dim=-1, keepdim=True)], dim=-1)
        
        pdfs = torch.softmax(logits, dim=-1)
        deterministic = getattr(self, 'deterministic', True)
        index, pdf_i = self.sampler.sample(pdfs, deterministic, self.anchors_per_pixel) # b, n, h, w, a
        disable_mask = (pdfs.argmax(dim=-1, keepdim=True) == self.num_samples).expand(
            -1, -1, -1, -1, self.anchors_per_pixel)
        # disable_mask = index == self.num_samples
        sampled_anchor = self.sampler.gather(index.clamp(max=(self.num_samples-1)), anchor_pts) # b, n, h, w, a, 3
        
        anchor_xyz = []
        for i in range(b):
            cur_sampled_anchor = sampled_anchor[i][~disable_mask[i]]
            cur_oob_mask = (cur_sampled_anchor[..., 0] < self.pc_range[0]) | (cur_sampled_anchor[..., 0] >= self.pc_range[3]) | \
                   (cur_sampled_anchor[..., 1] < self.pc_range[1]) | (cur_sampled_anchor[..., 1] >= self.pc_range[4]) | \
                   (cur_sampled_anchor[..., 2] < self.pc_range[2]) | (cur_sampled_anchor[..., 2] >= self.pc_range[5])
            scan = cur_sampled_anchor[~cur_oob_mask]
            
            if self.random_sampling:
                if scan.shape[0] < self.num_anchor:
                    multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                    scan_ = scan.repeat(multi, 1)
                    scan_ = scan_ + torch.randn_like(scan_) * 0.1
                    scan_ = scan_[np.random.choice(scan_.shape[0], self.num_anchor - scan.shape[0], False)]
                    scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                    scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                    scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                    scan = torch.cat([scan, scan_], 0)
                else:
                    scan = scan[np.random.choice(scan.shape[0], self.num_anchor, False)]
            else:
                if scan.shape[0] < self.num_anchor:
                    multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                    scan_ = scan.repeat(multi, 1)
                    scan_ = scan_ + torch.randn_like(scan_) * 0.1
                    scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                    scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                    scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                    scan = torch.cat([scan, scan_], 0)
                # breakpoint()
                if kwargs.get("benchmarking", False):
                    scan = scan[np.random.permutation(scan.shape[0])]
                    num_subsets = 3
                    sublens = torch.linspace(0, scan.shape[0], num_subsets + 1, dtype=torch.int, device=scan.device)[1:]
                    new_sublens = torch.linspace(0, self.num_anchor, num_subsets + 1, dtype=torch.int, device=scan.device)[1:]
                    scanidx = farthest_point_sampling(scan, sublens, new_sublens)
                else:
                    # breakpoint()
                    scanidx = farthest_point_sampling(
                        scan, 
                        torch.tensor([scan.shape[0]], device=scan.device, dtype=torch.int),
                        torch.tensor([self.num_anchor], device=scan.device, dtype=torch.int))
                scan = scan[scanidx, :]
            
            anchor_xyz.append(scan)

            if os.environ.get("DEBUG", 'false') == 'true':
                prefix = 'kitti-'
                #### save pred scan
                np.save(f'{prefix}pred_scan.npy', scan.detach().cpu().numpy())
                #### save gt scan
                np.save('gt_scan_occ.npy', anchor_occ.detach().cpu().numpy())
                np.save('gt_scan_pts.npy', anchor_pts.detach().cpu().numpy())
                #### save gt occupancy
                np.save('gt_occ.npy', metas['occ_label'].detach().cpu().numpy())
                np.save('gt_pts.npy', metas['occ_xyz'].detach().cpu().numpy())

                #### obtain depth
                # occ_depth = anchor_gt.float().argmax(dim=-1) # b, n, h, w
                # oob_mask = occ_depth == 128
                # occ_depth = occ_depth.clamp_max(127)
                
                # occ_from_occ_depth = torch.gather(
                #     anchor_idx, -2, occ_depth[..., None, None].expand(-1, -1, -1, -1, -1, 3))
                # occ_from_occ_depth = occ_from_occ_depth[~oob_mask].reshape(-1, 3)
                # pred_occ = torch.zeros_like(occupancy[0], dtype=torch.bool)
                # pred_occ[
                #     occ_from_occ_depth[:, 0], 
                #     occ_from_occ_depth[:, 1], 
                #     occ_from_occ_depth[:, 2]] = True
                
                # occ = occupancy[i]
                # scan_idx = ((scan - self.pc_start.view(1, 3)) / self.voxel_size).int()
                # pred_occ = torch.zeros_like(occ, dtype=torch.bool)
                # pred_occ[scan_idx[..., 0], scan_idx[..., 1], scan_idx[..., 2]] = True
                # gt_occ = (occ != self.empty_label).bool()
                # correct = (pred_occ & gt_occ).sum()
                # recall = correct / gt_occ.sum()
                # print(f"recall: {recall}")
                # miou = correct / (gt_occ.sum() + pred_occ.sum() - correct)
                # print(f"miou: {miou}")
                # precision = correct / pred_occ.sum()
                # print(f"precision: {precision}")
                # # scan_occ = occ[scan_idx[..., 0], scan_idx[..., 1], scan_idx[..., 2]]
                # # precision = (scan_occ != self.empty_label).sum() / scan_occ.numel()
                # # print(f"precision: {precision}")
                breakpoint()
        
        anchor_xyz = torch.stack(anchor_xyz)
        anchor_xyz[..., 0] = (anchor_xyz[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        anchor_xyz[..., 1] = (anchor_xyz[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        anchor_xyz[..., 2] = (anchor_xyz[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(anchor_xyz)
        anchor = torch.cat([
            xyz, torch.tile(self.anchor[None], (b, 1, 1))], dim=-1)
        
        if self.random_samples > 0:
            random_anchors = torch.tile(self.random_anchors[None], (b, 1, 1))
            anchor = torch.cat([anchor, random_anchors], dim=1)

        instance_feature = torch.tile(
            self.instance_feature[None], (b, 1, 1)
        )
        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': anchor[0].clone(),
            'pixel_logits': logits,
            'pixel_gt': anchor_gt,
        }
    


    def _forward_fallback(self, metas, **kwargs):
        """回退方法（当LiDAR初始化失败时使用）"""
        batch_size = len(metas.get('img', [1]))  # 估计batch_size

        # 使用默认初始化
        default_anchor = self._init_default_anchors(batch_size)
        
        if self.random_samples > 0:
            random_anchors = torch.tile(self.random_anchors[None], (batch_size, 1, 1))
            anchor = torch.cat([default_anchor, random_anchors], dim=1)
        else:
            anchor = default_anchor
            
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        
        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': anchor[0].clone(),
            'pixel_logits': None,
            'pixel_gt': None,
        }