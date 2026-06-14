"""Gaussian Statistic Aggregator —— 跨帧/跨场景数据聚合器（v5 低同步版）。

v5 核心优化：
- 所有 per-bin/per-class 累加在 GPU 端完成，消除 ~116 次 .item() 同步/帧
- 使用 scatter_add_ / bincount 矢量化分类和分桶
- finalize() 时一次性 GPU→CPU 传输
- count/covered/total 使用 int64，sum 使用 float64，避免千万级统计的 float32 精度饱和
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from . import calculator


class GaussianStatAggregator:
    """跨帧聚合器：GPU 端累积，finalize 一次性同步。"""

    def __init__(self, t_sphere=2.0, t_scale=0.5, distance_bins=None,
                 percentiles=None, num_classes=17, cov_threshold=3.0,
                 chunk_size=10000, exclude_classes: Optional[List[int]] = None,
                 empty_label=17, ignore_empty=True,
                 exclude_gaussian_classes: Optional[List[int]] = None,
                 exclude_voxel_classes: Optional[List[int]] = None,
                 scale_range: Optional[List[float]] = None):
        self.t_sphere = t_sphere; self.t_scale = t_scale
        self.distance_bins = distance_bins or [0, 10, 20, 30, 40, 50]
        self.percentiles = percentiles or [50, 75, 90, 95]
        self.num_classes = num_classes; self.cov_threshold = cov_threshold
        self.chunk_size = chunk_size
        self.empty_label = empty_label
        self.ignore_empty = ignore_empty
        self.scale_range = scale_range
        # 拆分 exclude_classes 为 Gaussian 过滤和 Voxel 过滤
        self.exclude_gaussian_classes = exclude_gaussian_classes if exclude_gaussian_classes is not None else exclude_classes
        self.exclude_voxel_classes = exclude_voxel_classes
        self._exclude_gaussian_classes_tensor = None
        self._exclude_voxel_classes_tensor = None
        n_bins = len(self.distance_bins) - 1

        # ---- 全局标量 ----
        self.total_frames = 0; self.total_gaussians = 0
        self.sum_mean_scale = 0.0; self.sum_nsr = 0.0
        self.sum_ligr = 0.0; self.vol_sum = 0.0; self.vol_count = 0
        # Purity 累加器（新版）
        self.total_purity_sum = 0.0       # 旧版兼容（所有 Gaussian，unused=1.0）
        self.total_valid_purity_sum = 0.0  # 仅覆盖了体素的 Gaussian
        self.total_valid_purity_count = 0
        self.total_penalized_purity_sum = 0.0  # 全局惩罚版（unused=0.0）
        self.total_unused_gaussians = 0
        self.total_purity_gaussians = 0
        # Coverage 累加器
        self.total_cov_covered = 0.0
        self.total_cov_total = 0

        # GPU tensor 累积（延后 CPU 传输）
        self._acc_s_hat: List[torch.Tensor] = []
        self._acc_ars: List[torch.Tensor] = []
        self._acc_volumes: List[torch.Tensor] = []

        # GPU 端累加器（消除 .item() 同步）
        self.device = None  # 延迟初始化
        self._dist_count = None       # (n_bins,) int64
        self._dist_sum_scale = None
        self._dist_sum_ar = None
        self._dist_nsr_count = None
        self._dist_nsr_total = None
        self._cat_count = None        # (num_classes,) int64
        self._cat_sum_scale = None
        self._cat_sum_ar = None
        self._cat_nsr_count = None
        self._cat_nsr_total = None
        self._cat_sum_purity = None
        self._cat_purity_frames = None  # per-class: 有多少帧贡献了 purity, int64

        # Distance-wise Coverage GPU 累加器
        self._dcov_covered = None     # (n_bins,) int64
        self._dcov_total = None

    def _init_gpu_state(self, device):
        """延迟初始化 GPU 累加器。"""
        if self.device is not None:
            return
        self.device = device
        n_bins = len(self.distance_bins) - 1
        nc = self.num_classes
        self._dist_count = torch.zeros(n_bins, device=device, dtype=torch.long)
        self._dist_sum_scale = torch.zeros(n_bins, device=device, dtype=torch.float64)
        self._dist_sum_ar = torch.zeros(n_bins, device=device, dtype=torch.float64)
        self._dist_nsr_count = torch.zeros(n_bins, device=device, dtype=torch.long)
        self._dist_nsr_total = torch.zeros(n_bins, device=device, dtype=torch.long)
        self._cat_count = torch.zeros(nc, device=device, dtype=torch.long)
        self._cat_sum_scale = torch.zeros(nc, device=device, dtype=torch.float64)
        self._cat_sum_ar = torch.zeros(nc, device=device, dtype=torch.float64)
        self._cat_nsr_count = torch.zeros(nc, device=device, dtype=torch.long)
        self._cat_nsr_total = torch.zeros(nc, device=device, dtype=torch.long)
        self._cat_sum_purity = torch.zeros(nc, device=device, dtype=torch.float64)
        self._cat_purity_frames = torch.zeros(nc, device=device, dtype=torch.long)
        self._dcov_covered = torch.zeros(n_bins, device=device, dtype=torch.long)
        self._dcov_total = torch.zeros(n_bins, device=device, dtype=torch.long)
        # 初始化排除类别的 GPU tensor
        if self.exclude_gaussian_classes is not None:
            self._exclude_gaussian_classes_tensor = torch.tensor(
                self.exclude_gaussian_classes, device=device, dtype=torch.long)
        if self.exclude_voxel_classes is not None:
            self._exclude_voxel_classes_tensor = torch.tensor(
                self.exclude_voxel_classes, device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def add_frame(self, gaussian, metas: dict) -> None:
        means = gaussian.means.reshape(-1, 3)
        scales = gaussian.scales.reshape(-1, 3)
        rotations = gaussian.rotations.reshape(-1, 4)
        semantics = gaussian.semantics.reshape(-1, gaussian.semantics.shape[-1])
        G = means.shape[0]
        if G == 0: return

        device = means.device
        self._init_gpu_state(device)
        self.total_frames += 1; self.total_gaussians += G

        # ---- 基础属性（GPU 端） ----
        s_hat = scales.mean(dim=-1)                              # (G,)
        ar = scales.max(dim=-1).values / (scales.min(dim=-1).values + 1e-8)
        pred_class = semantics.argmax(dim=-1)
        nsr_mask = (ar < self.t_sphere).float()

        # 标量累加（仅 4 次 .item()，可接受）
        self.sum_mean_scale += s_hat.mean().item()
        self.sum_nsr += nsr_mask.mean().item()
        self.sum_ligr += ((s_hat > self.t_scale) & (ar < self.t_sphere)).float().mean().item()
        vol = (4.0/3.0)*torch.pi*scales[:,0]*scales[:,1]*scales[:,2]
        self.vol_sum += vol.sum().item(); self.vol_count += G

        # GPU tensor 累积
        self._acc_s_hat.append(s_hat.detach())
        self._acc_ars.append(ar.detach())
        self._acc_volumes.append(vol.detach())

        # ---- 提取 GT 数据 ----
        occ_xyz = metas.get('occ_xyz'); occ_label = metas.get('occ_label')
        occ_cam_mask = metas.get('occ_cam_mask')
        if occ_xyz is not None and occ_xyz.dim() == 5: occ_xyz = occ_xyz[0]
        if occ_label is not None and occ_label.dim() == 4: occ_label = occ_label[0]
        if occ_cam_mask is not None and occ_cam_mask.dim() == 4: occ_cam_mask = occ_cam_mask[0]

        # ---- Distance-wise：GPU 端 bucketize + scatter_add ----
        # 使用 Chebyshev 距离（max(|x|, |y|)），对应 [-L,L]×[-L,L] 的方形区域
        dist = torch.max(torch.abs(means[..., :2]), dim=-1).values
        n_bins = len(self.distance_bins) - 1
        bin_edges = torch.tensor(self.distance_bins, device=device)
        bin_idx = torch.bucketize(dist, bin_edges) - 1          # 0-based
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        bc = bin_idx.clamp(0, n_bins - 1)
        ones_g_long = torch.ones(G, device=device, dtype=torch.long)
        valid_long = valid.to(torch.long)
        valid_f64 = valid.to(torch.float64)

        self._dist_count.scatter_add_(0, bc, valid_long)
        self._dist_sum_scale.scatter_add_(0, bc, s_hat.to(torch.float64) * valid_f64)
        self._dist_sum_ar.scatter_add_(0, bc, ar.to(torch.float64) * valid_f64)
        self._dist_nsr_count.scatter_add_(0, bc, (nsr_mask > 0).to(torch.long) * valid_long)
        self._dist_nsr_total.scatter_add_(0, bc, ones_g_long * valid_long)

        # ---- Category-wise：GPU 端 scatter_add（带 pred_class 范围保护） ----
        valid_cls = (pred_class >= 0) & (pred_class < self.num_classes)
        pc = pred_class[valid_cls]
        ones_g_valid = ones_g_long[valid_cls]
        self._cat_count.scatter_add_(0, pc, ones_g_valid)
        self._cat_sum_scale.scatter_add_(0, pc, s_hat[valid_cls].to(torch.float64))
        self._cat_sum_ar.scatter_add_(0, pc, ar[valid_cls].to(torch.float64))
        self._cat_nsr_count.scatter_add_(0, pc, (nsr_mask[valid_cls] > 0).to(torch.long))
        self._cat_nsr_total.scatter_add_(0, pc, ones_g_valid)

        # ---- 合并 Coverage + Purity ----
        if (occ_xyz is not None and occ_cam_mask is not None
                and calculator.mask_flat_valid(occ_cam_mask)):
            self._accumulate_cov_purity(means, scales, rotations, semantics,
                                        pred_class, occ_xyz, occ_label, occ_cam_mask)

    def _accumulate_cov_purity(self, means, scales, rotations, semantics,
                                pred_class, occ_xyz, occ_label, occ_cam_mask):
        """GPU 端累加 Coverage + Purity（消除 46 次 .item()）。

        修改说明（2025-06）:
        - 排除 empty/free voxels（label == empty_label）
        - Purity 不再将 unused Gaussian（未覆盖任何体素）记为 1.0
        - exclude_classes 拆分为 exclude_gaussian_classes / exclude_voxel_classes
        """
        device = means.device
        # ---- 按语义类别过滤 Gaussian ----
        if self._exclude_gaussian_classes_tensor is not None:
            keep = ~torch.isin(pred_class, self._exclude_gaussian_classes_tensor)
            if keep.any():
                means = means[keep]
                scales = scales[keep]
                rotations = rotations[keep]
                semantics = semantics[keep]
                pred_class = pred_class[keep]
            else:
                return  # 所有 Gaussian 都被过滤，跳过此帧

        # ---- 构建有效体素 mask（occ_cam_mask + 排除 empty + 排除 voxel 类别） ----
        occ_flat_xyz = occ_xyz.reshape(-1, 3)
        mask_flat = occ_cam_mask.reshape(-1).bool()

        if occ_label is not None:
            label_flat = occ_label.reshape(-1).long()
            if self.ignore_empty:
                mask_flat = mask_flat & (label_flat != self.empty_label)
            if self._exclude_voxel_classes_tensor is not None:
                mask_flat = mask_flat & ~torch.isin(label_flat, self._exclude_voxel_classes_tensor)
        else:
            label_flat = None

        valid_xyz = occ_flat_xyz[mask_flat]
        if valid_xyz.shape[0] == 0: return

        # 使用 Chebyshev 距离（max(|x|, |y|)），对应方柱体区域
        valid_dist = torch.max(torch.abs(valid_xyz[..., :2]), dim=-1).values
        precomp = calculator.precompute_frame_data(means, scales, rotations, semantics,
                                                    cov_threshold=self.cov_threshold)
        if label_flat is not None:
            valid_label = label_flat[mask_flat]
        else:
            valid_label = torch.zeros(valid_xyz.shape[0], dtype=torch.long, device=device)

        covered, purity_match, purity_total = calculator.compute_coverage_and_purity(
            means, precomp, valid_xyz, valid_label,
            cov_threshold=self.cov_threshold, chunk_size=self.chunk_size)

        # Distance-wise Coverage：GPU 端 bucketize + scatter_add
        n_bins = len(self.distance_bins) - 1
        bin_edges = torch.tensor(self.distance_bins, device=device)
        bin_idx = torch.bucketize(valid_dist, bin_edges) - 1
        valid_b = (bin_idx >= 0) & (bin_idx < n_bins)
        bc = bin_idx.clamp(0, n_bins - 1)
        valid_b_long = valid_b.to(torch.long)
        ones_n_long = torch.ones(valid_xyz.shape[0], device=device, dtype=torch.long)

        self._dcov_total.scatter_add_(0, bc, ones_n_long * valid_b_long)
        self._dcov_covered.scatter_add_(0, bc, covered.to(torch.long) * valid_b_long)

        # ---- 全局 Mean Coverage（仅统计距离分桶内的体素，与 Distance-wise 口径一致） ----
        self.total_cov_covered += (covered & valid_b).sum().item()
        self.total_cov_total += valid_b.sum().item()

        # ---- Category-wise Purity（GPU 端 scatter_add） ----
        if pred_class.shape[0] > 0:
            valid_pred_cls = (pred_class >= 0) & (pred_class < self.num_classes)
            p_match_cls = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
            p_total_cls = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
            p_match_cls.scatter_add_(0, pred_class[valid_pred_cls], purity_match[valid_pred_cls].to(torch.float64))
            p_total_cls.scatter_add_(0, pred_class[valid_pred_cls], purity_total[valid_pred_cls].to(torch.float64))
            # Per-frame per-class purity ratio
            valid_cls = p_total_cls > 0
            frame_purity = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
            frame_purity[valid_cls] = p_match_cls[valid_cls] / p_total_cls[valid_cls].clamp(min=1)
            self._cat_sum_purity += frame_purity
            self._cat_purity_frames += valid_cls.to(torch.long)

            # ---- 新版 Purity 累加（P0-2） ----
            # 将所有 Gaussian 标记为 unused 或 valid
            n_g = purity_total.shape[0]
            valid_g = purity_total > 0
            n_valid = valid_g.sum().item()
            n_unused = n_g - n_valid

            # 旧版兼容：unused Gaussian 视为 1.0
            p_g_purity_old = torch.where(
                valid_g,
                purity_match.float() / purity_total.float().clamp(min=1),
                torch.ones_like(purity_match, dtype=torch.float32))
            self.total_purity_sum += p_g_purity_old.sum().item()

            # 新版 valid purity：仅统计覆盖了体素的 Gaussian
            if n_valid > 0:
                valid_purity = purity_match[valid_g].float() / purity_total[valid_g].float().clamp(min=1)
                self.total_valid_purity_sum += valid_purity.sum().item()
                self.total_valid_purity_count += n_valid

            # 新版 penalized purity：unused Gaussian 视为 0.0
            p_g_purity_penalized = torch.zeros(n_g, dtype=torch.float32, device=device)
            if n_valid > 0:
                p_g_purity_penalized[valid_g] = (
                    purity_match[valid_g].float() / purity_total[valid_g].float().clamp(min=1))
            self.total_penalized_purity_sum += p_g_purity_penalized.sum().item()

            self.total_unused_gaussians += n_unused
            self.total_purity_gaussians += n_g

    # ------------------------------------------------------------------
    # 快照（仍需要少量 .item()，但仅在 stat_freq 时调用）
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        if self.total_frames == 0: return {}
        def _s(a, b): return a/b if b else 0.0
        if self._acc_ars:
            mean_ar = torch.cat(self._acc_ars).mean().item()
        else:
            mean_ar = 0.0
        return {'frames': self.total_frames, 'gaussians': self.total_gaussians,
                'mean_scale': _s(self.sum_mean_scale, self.total_frames),
                'nsr': _s(self.sum_nsr, self.total_frames),
                'ligr': _s(self.sum_ligr, self.total_frames),
                'mean_vol': _s(self.vol_sum, self.vol_count), 'mean_ar': mean_ar,
                'mean_coverage': _s(self.total_cov_covered, self.total_cov_total),
                'mean_purity_old': _s(self.total_purity_sum, self.total_gaussians),
                'mean_purity_valid': _s(self.total_valid_purity_sum, self.total_valid_purity_count),
                'mean_purity_penalized': _s(self.total_penalized_purity_sum, self.total_purity_gaussians),
                'unused_ratio': _s(self.total_unused_gaussians, self.total_purity_gaussians)}

    # ------------------------------------------------------------------
    # 最终聚合（一次性 GPU→CPU）
    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        if self.total_frames == 0: return {'error': 'No frames processed'}

        # 一次性 GPU→CPU 传输
        all_s_hat = torch.cat(self._acc_s_hat).cpu().tolist() if self._acc_s_hat else []
        all_ars   = torch.cat(self._acc_ars).cpu().tolist() if self._acc_ars else []
        all_vols  = torch.cat(self._acc_volumes).cpu().tolist() if self._acc_volumes else []

        dist_count = self._dist_count.cpu().tolist()
        dist_sum_scale = self._dist_sum_scale.cpu().tolist()
        dist_sum_ar = self._dist_sum_ar.cpu().tolist()
        dist_nsr_count = self._dist_nsr_count.cpu().tolist()
        dist_nsr_total = self._dist_nsr_total.cpu().tolist()
        cat_count = self._cat_count.cpu().tolist()
        cat_sum_scale = self._cat_sum_scale.cpu().tolist()
        cat_sum_ar = self._cat_sum_ar.cpu().tolist()
        cat_nsr_count = self._cat_nsr_count.cpu().tolist()
        cat_nsr_total = self._cat_nsr_total.cpu().tolist()
        cat_sum_purity = self._cat_sum_purity.cpu().tolist()
        cat_purity_frames = self._cat_purity_frames.cpu().tolist()
        dcov_covered = self._dcov_covered.cpu().tolist()
        dcov_total = self._dcov_total.cpu().tolist()

        def _s(a, b): return a/b if b else 0.0
        def _p(arr, q): return float(np.percentile(arr, q)) if arr else 0.0

        result = {'num_frames': self.total_frames, 'num_gaussians': self.total_gaussians,
                  'mean_scale': self.sum_mean_scale/self.total_frames,
                  'near_spherical_ratio': self.sum_nsr/self.total_frames,
                  'ligr': self.sum_ligr/self.total_frames,
                  'mean_coverage': _s(self.total_cov_covered, self.total_cov_total),
                  # Purity 系列
                  'mean_purity': _s(self.total_purity_sum, self.total_gaussians),
                  'mean_purity_valid': _s(self.total_valid_purity_sum, self.total_valid_purity_count),
                  'mean_purity_penalized': _s(self.total_penalized_purity_sum, self.total_purity_gaussians),
                  'unused_gaussian_ratio': _s(self.total_unused_gaussians, self.total_purity_gaussians),
                  # 元数据
                  'cov_threshold': self.cov_threshold,
                  'coverage_voxel_scope': 'non_empty_visible' if self.ignore_empty else 'visible'}

        observed_min_scale = min(all_s_hat) if all_s_hat else 0.0
        observed_max_scale = max(all_s_hat) if all_s_hat else 0.0
        result['scale_sanity'] = {
            'configured_scale_range': self.scale_range,
            'observed_min_s_hat': float(observed_min_scale),
            'observed_max_s_hat': float(observed_max_scale),
        }

        result['scale_percentiles'] = {f'P{p}': _p(all_s_hat, p) for p in self.percentiles}
        result['scale_volume'] = {'mean_volume': _s(self.vol_sum, self.vol_count),
            'p90_volume': _p(all_vols, 90), 'p95_volume': _p(all_vols, 95)}

        ar_arr = np.array(all_ars, dtype=np.float64)
        result['anisotropy_ratio'] = {'mean_ar': float(ar_arr.mean()) if len(ar_arr) else 0,
            'median_ar': float(np.median(ar_arr)) if len(ar_arr) else 0,
            'p75_ar': _p(all_ars, 75), 'p90_ar': _p(all_ars, 90)}

        n_bins = len(self.distance_bins) - 1
        total_g = self.total_gaussians
        result['category_stats'] = {c: {'count': int(cat_count[c]),
            'ratio': _s(cat_count[c], total_g),
            'mean_scale': _s(cat_sum_scale[c], cat_count[c]),
            'mean_ar': _s(cat_sum_ar[c], cat_count[c]),
            'near_spherical_ratio': _s(cat_nsr_count[c], cat_nsr_total[c]),
            'mean_purity': _s(cat_sum_purity[c], cat_purity_frames[c]),
            'scale_sanity_warning': (
                cat_count[c] > 0 and _s(cat_sum_scale[c], cat_count[c]) > observed_max_scale + 1e-6)}
            for c in range(self.num_classes)}

        result['distance_stats'] = {
            f'({self.distance_bins[i]},{self.distance_bins[i+1]})': {
                'count': int(dist_count[i]),
                'mean_scale': _s(dist_sum_scale[i], dist_count[i]),
                'mean_ar': _s(dist_sum_ar[i], dist_count[i]),
                'near_spherical_ratio': _s(dist_nsr_count[i], dist_nsr_total[i]),
                'scale_sanity_warning': (
                    dist_count[i] > 0 and _s(dist_sum_scale[i], dist_count[i]) > observed_max_scale + 1e-6)}
            for i in range(n_bins)}

        result['distancewise_coverage'] = {
            f'({self.distance_bins[i]},{self.distance_bins[i+1]})':
                _s(dcov_covered[i], dcov_total[i])
            for i in range(n_bins)}

        # ---- Coverage debug 自检（P0-3） ----
        sum_dcov_covered = float(sum(dcov_covered))
        sum_dcov_total = float(sum(dcov_total))
        scalar_cov_covered = float(self.total_cov_covered)
        scalar_cov_total = float(self.total_cov_total)
        coverage_from_bins = _s(sum_dcov_covered, sum_dcov_total)
        coverage_from_scalar = _s(scalar_cov_covered, scalar_cov_total)
        result['coverage_debug'] = {
            'cov_threshold': self.cov_threshold,
            'coverage_scope': 'non_empty_visible' if self.ignore_empty else 'visible',
            'sum_dcov_covered': sum_dcov_covered,
            'sum_dcov_total': sum_dcov_total,
            'scalar_cov_covered': scalar_cov_covered,
            'scalar_cov_total': scalar_cov_total,
            'coverage_from_bins': coverage_from_bins,
            'coverage_from_scalar': coverage_from_scalar,
            'coverage_counter_abs_diff': abs(coverage_from_bins - coverage_from_scalar),
            'coverage_counter_consistent': abs(coverage_from_bins - coverage_from_scalar) < 1e-6,
        }
        result['distancewise_coverage_debug'] = {
            f'({self.distance_bins[i]},{self.distance_bins[i+1]})': {
                'covered': float(dcov_covered[i]),
                'total': float(dcov_total[i]),
                'coverage': _s(dcov_covered[i], dcov_total[i]),
            }
            for i in range(n_bins)}

        return result
