"""Gaussian Statistic Calculator —— 纯数学统计函数，不依赖 torch Module。

优化说明（v5 低同步版）：
- search_radius 直接用 scales.max() 替代 eigvalsh（省 ~50ms/帧）
- compute_coverage_and_purity() voxel 分块 + 矢量化，chunk_size=10000
- _build_cov_inv() return_R 避免重复计算旋转矩阵
- Mahalanobis R+scales+bmm 快速路径
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def _get_rotation_matrix(rotations: torch.Tensor) -> torch.Tensor:
    """将四元数 (..., 4) 转换为旋转矩阵 (..., 3, 3)。"""
    tensor = F.normalize(rotations, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]; mat1[..., 0, 1] = -tensor[..., 1]
    mat1[..., 0, 2] = -tensor[..., 2]; mat1[..., 0, 3] = -tensor[..., 3]
    mat1[..., 1, 0] = tensor[..., 1]; mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = -tensor[..., 3]; mat1[..., 1, 3] = tensor[..., 2]
    mat1[..., 2, 0] = tensor[..., 2]; mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]; mat1[..., 2, 3] = -tensor[..., 1]
    mat1[..., 3, 0] = tensor[..., 3]; mat1[..., 3, 1] = -tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]; mat1[..., 3, 3] = tensor[..., 0]
    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]; mat2[..., 0, 1] = -tensor[..., 1]
    mat2[..., 0, 2] = -tensor[..., 2]; mat2[..., 0, 3] = -tensor[..., 3]
    mat2[..., 1, 0] = tensor[..., 1]; mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]; mat2[..., 1, 3] = -tensor[..., 2]
    mat2[..., 2, 0] = tensor[..., 2]; mat2[..., 2, 1] = -tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]; mat2[..., 2, 3] = tensor[..., 1]
    mat2[..., 3, 0] = tensor[..., 3]; mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = -tensor[..., 1]; mat2[..., 3, 3] = tensor[..., 0]
    mat2 = torch.conj(mat2).transpose(-1, -2)
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


def _build_cov_inv(scales: torch.Tensor, rotations: torch.Tensor, return_R: bool = False):
    G = scales.shape[0]
    S = torch.zeros(G, 3, 3, dtype=scales.dtype, device=scales.device)
    S[:, 0, 0] = scales[:, 0]; S[:, 1, 1] = scales[:, 1]; S[:, 2, 2] = scales[:, 2]
    R = _get_rotation_matrix(rotations)
    M = torch.matmul(S, R)
    Cov = torch.matmul(M.transpose(-1, -2), M)
    CovInv = torch.inverse(Cov)
    return (CovInv, R) if return_R else CovInv


def precompute_frame_data(means, scales, rotations, semantics, cov_threshold=3.0):
    """预计算每帧共享数据。v5: scales.max() 替代 eigvalsh。"""
    G = scales.shape[0]
    device, dtype = scales.device, scales.dtype
    if G == 0:
        return {
            'cov_inv': torch.empty(0, 3, 3, device=device, dtype=dtype),
            'R': torch.empty(0, 3, 3, device=device, dtype=dtype),
            'scales_vec': torch.empty(0, 3, device=device, dtype=dtype),
            'search_radius': torch.empty(0, device=device, dtype=dtype),
            'pred_class': torch.empty(0, device=device, dtype=torch.long),
        }
    cov_inv, R = _build_cov_inv(scales, rotations, return_R=True)
    search_radius = cov_threshold * scales.max(dim=-1).values
    pred_class = semantics.argmax(dim=-1)
    return {'cov_inv': cov_inv, 'R': R, 'scales_vec': scales,
            'search_radius': search_radius, 'pred_class': pred_class}


# ---------------------------------------------------------------------------
# 单帧统计函数
# ---------------------------------------------------------------------------

def compute_mean_scale(scales): 
    return scales.mean(dim=-1).mean().item() if scales.numel() > 0 else 0.0

def compute_scale_percentiles(scales, percentiles=None):
    if percentiles is None: percentiles = [50, 75, 90, 95]
    if scales.numel() == 0: return {f'P{p}': 0.0 for p in percentiles}
    s_hat = scales.mean(dim=-1)
    q = torch.tensor(percentiles, dtype=torch.float32, device=scales.device)
    r = torch.quantile(s_hat.float(), q / 100.0)
    return {f'P{p}': r[i].item() for i, p in enumerate(percentiles)}

def compute_scale_volume(scales):
    if scales.numel() == 0: return {'mean_volume': 0.0, 'p90_volume': 0.0, 'p95_volume': 0.0}
    vol = (4.0/3.0)*torch.pi*scales[:,0]*scales[:,1]*scales[:,2]
    return {'mean_volume': vol.mean().item(), 'p90_volume': torch.quantile(vol.float(), 0.90).item(),
            'p95_volume': torch.quantile(vol.float(), 0.95).item()}

def compute_anisotropy_ratio(scales, eps=1e-8):
    if scales.numel() == 0: return {'mean_ar':0,'median_ar':0,'p75_ar':0,'p90_ar':0}
    ar = scales.max(dim=-1).values/(scales.min(dim=-1).values+eps)
    return {'mean_ar': ar.mean().item(), 'median_ar': ar.median().item(),
            'p75_ar': torch.quantile(ar.float(),0.75).item(), 'p90_ar': torch.quantile(ar.float(),0.90).item()}

def compute_near_spherical_ratio(ar, t_sphere):
    return (ar < t_sphere).float().mean().item() if ar.numel() > 0 else 0.0

def compute_ligr(scales, ar, t_scale, t_sphere):
    if scales.numel() == 0: return 0.0
    return ((scales.mean(dim=-1) > t_scale) & (ar < t_sphere)).float().mean().item()

def compute_distance_stats(means, scales, t_sphere, distance_bins):
    """Distance-wise stats using BEV Chebyshev distance: max(|x|, |y|)."""
    if means.numel() == 0: return {}
    dist = torch.max(torch.abs(means[..., :2]), dim=-1).values
    s_hat = scales.mean(dim=-1)
    ar = scales.max(dim=-1).values/(scales.min(dim=-1).values+1e-8)
    r = {}
    for i in range(len(distance_bins)-1):
        lo, hi = distance_bins[i], distance_bins[i+1]
        m = (dist>=lo)&(dist<hi); c = m.sum().item()
        r[f'({lo}, {hi})'] = {'count':c, 'mean_scale':s_hat[m].mean().item()if c else 0,
            'mean_ar':ar[m].mean().item()if c else 0,
            'near_spherical_ratio':(ar[m]<t_sphere).float().mean().item()if c else 0}
    return r


# ---------------------------------------------------------------------------
# Coverage + Purity（v5 voxel-chunked 矢量化 + 低同步）
# ---------------------------------------------------------------------------

def compute_coverage_and_purity(
    means, precomp, voxel_xyz, voxel_label,
    cov_threshold=3.0, chunk_size=10000, use_fast_mahalanobis=True,
    voxel_bin_idx=None, n_bins=None, return_extra_stats=False,
    max_pair_elements=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxel-chunked coverage and purity with a bounded dense-distance peak."""
    N, G = voxel_xyz.shape[0], means.shape[0]
    device = means.device
    if N == 0 or G == 0:
        base = (torch.zeros(N, dtype=torch.bool, device=device),
                torch.zeros(G, dtype=torch.int64, device=device),
                torch.zeros(G, dtype=torch.int64, device=device))
        if not return_extra_stats:
            return base
        n_extra_bins = n_bins or 0
        extra_stats = {
            'aligned_covered': torch.zeros(N, dtype=torch.bool, device=device),
            'distance_purity_match': torch.zeros(n_extra_bins, dtype=torch.int64, device=device),
            'distance_purity_total': torch.zeros(n_extra_bins, dtype=torch.int64, device=device),
        }
        return (*base, extra_stats)

    cov_inv = precomp['cov_inv']; R = precomp['R']
    scales_vec = precomp['scales_vec']; search_radius = precomp['search_radius']
    pred_class = precomp['pred_class']; tau_sq = cov_threshold**2

    chunk_size = max(1, int(chunk_size))
    if max_pair_elements is not None:
        # ``torch.cdist`` materializes a dense [chunk, G] tensor. Bound the
        # element count so that a larger history-derived G cannot cause a
        # sudden multi-GB allocation.
        chunk_size = min(
            chunk_size,
            max(1, int(max_pair_elements) // max(G, 1)),
        )

    covered = torch.zeros(N, dtype=torch.bool, device=device)
    p_match = torch.zeros(G, dtype=torch.int64, device=device)
    p_total = torch.zeros(G, dtype=torch.int64, device=device)
    if return_extra_stats:
        if voxel_bin_idx is None or n_bins is None:
            raise ValueError('voxel_bin_idx and n_bins are required when return_extra_stats=True')
        aligned_covered = torch.zeros(N, dtype=torch.bool, device=device)
        distance_purity_match = torch.zeros(n_bins, dtype=torch.int64, device=device)
        distance_purity_total = torch.zeros(n_bins, dtype=torch.int64, device=device)

    def process_chunk(start, end):
        """Keep dense chunk intermediates scoped to one helper invocation."""
        chunk_v = voxel_xyz[start:end]
        chunk_l = voxel_label[start:end]

        dists = torch.cdist(chunk_v, means)
        cand_mask = dists <= search_radius[None, :]
        del dists
        pairs = cand_mask.nonzero(as_tuple=False)
        del cand_mask
        if pairs.shape[0] == 0:
            return

        vi, gj = pairs[:, 0], pairs[:, 1]
        diff = chunk_v[vi] - means[gj]

        if use_fast_mahalanobis:
            d_rot = torch.bmm(R[gj], diff.unsqueeze(-1)).squeeze(-1)
            mahal = (d_rot / scales_vec[gj].clamp(min=1e-8)).pow(2).sum(dim=-1)
        else:
            mahal = torch.einsum('ki,kij,kj->k', diff, cov_inv[gj], diff)

        hit = mahal <= tau_sq
        if not hit.any():
            return

        hv, hg = vi[hit], gj[hit]
        covered[start + hv] = True
        hit_match = chunk_l[hv] == pred_class[hg]
        p_total.scatter_add_(
            0, hg, torch.ones(hg.shape[0], dtype=torch.int64, device=device))
        p_match.scatter_add_(0, hg, hit_match.long())

        if return_extra_stats:
            global_hv = start + hv
            aligned_covered[global_hv[hit_match]] = True
            hit_bins = voxel_bin_idx[global_hv]
            valid_bins = (hit_bins >= 0) & (hit_bins < n_bins)
            if valid_bins.any():
                vb = hit_bins[valid_bins]
                distance_purity_total.scatter_add_(
                    0, vb, torch.ones(vb.shape[0], dtype=torch.int64, device=device))
                distance_purity_match.scatter_add_(0, vb, hit_match[valid_bins].long())

    for start in range(0, N, chunk_size):
        process_chunk(start, min(start + chunk_size, N))

    if not return_extra_stats:
        return covered, p_match, p_total
    extra_stats = {
        'aligned_covered': aligned_covered,
        'distance_purity_match': distance_purity_match,
        'distance_purity_total': distance_purity_total,
    }
    return covered, p_match, p_total, extra_stats


def compute_subset_coverage(
    means, precomp, voxel_xyz, gaussian_mask,
    cov_threshold=3.0, chunk_size=10000, use_fast_mahalanobis=True,
    max_pair_elements=None,
) -> torch.Tensor:
    """Return the voxel union covered by a selected Gaussian subset.

    Purity is only known after the first coverage pass.  This bounded second
    pass avoids retaining all voxel-Gaussian hit pairs while still allowing
    coverage to be attributed to the final low-purity Gaussian subset.
    """
    N, G = voxel_xyz.shape[0], means.shape[0]
    device = means.device
    if gaussian_mask.shape != (G,):
        raise ValueError(
            f'gaussian_mask must have shape ({G},), got {tuple(gaussian_mask.shape)}')

    gaussian_mask = gaussian_mask.to(device=device, dtype=torch.bool)
    selected_count = int(gaussian_mask.sum().item())
    if N == 0 or selected_count == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)

    subset_means = means[gaussian_mask]
    subset_cov_inv = precomp['cov_inv'][gaussian_mask]
    subset_R = precomp['R'][gaussian_mask]
    subset_scales = precomp['scales_vec'][gaussian_mask]
    subset_search_radius = precomp['search_radius'][gaussian_mask]
    tau_sq = cov_threshold ** 2

    chunk_size = max(1, int(chunk_size))
    if max_pair_elements is not None:
        chunk_size = min(
            chunk_size,
            max(1, int(max_pair_elements) // selected_count),
        )

    covered = torch.zeros(N, dtype=torch.bool, device=device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_v = voxel_xyz[start:end]
        dists = torch.cdist(chunk_v, subset_means)
        candidate_pairs = (dists <= subset_search_radius[None, :]).nonzero(
            as_tuple=False)
        del dists
        if candidate_pairs.shape[0] == 0:
            continue

        voxel_index, gaussian_index = candidate_pairs[:, 0], candidate_pairs[:, 1]
        diff = chunk_v[voxel_index] - subset_means[gaussian_index]
        if use_fast_mahalanobis:
            rotated = torch.bmm(
                subset_R[gaussian_index], diff.unsqueeze(-1)).squeeze(-1)
            mahal = (
                rotated / subset_scales[gaussian_index].clamp(min=1e-8)
            ).pow(2).sum(dim=-1)
        else:
            mahal = torch.einsum(
                'ki,kij,kj->k', diff, subset_cov_inv[gaussian_index], diff)

        hit = mahal <= tau_sq
        if hit.any():
            covered[start + voxel_index[hit]] = True

    return covered


def compute_distancewise_coverage(means, scales, rotations, occ_xyz, occ_cam_mask,
    distance_bins, cov_threshold=3.0, precomp=None, chunk_size=10000,
    pred_class=None, exclude_classes=None):
    if means.numel() == 0: return {}
    # 按语义类别过滤 Gaussian
    if exclude_classes is not None and pred_class is not None:
        excl = torch.tensor(exclude_classes, device=means.device, dtype=torch.long)
        keep = ~torch.isin(pred_class, excl)
        if keep.any():
            means = means[keep]; scales = scales[keep]; rotations = rotations[keep]
        else:
            return {f'({distance_bins[i]},{distance_bins[i+1]})':0.0 for i in range(len(distance_bins)-1)}
    occ_flat = occ_xyz.reshape(-1, 3)
    mf = occ_cam_mask.reshape(-1).bool()
    if not mf.any(): return {f'({distance_bins[i]},{distance_bins[i+1]})':0.0 for i in range(len(distance_bins)-1)}
    vx = occ_flat[mf]; vd = torch.max(torch.abs(vx[..., :2]), dim=-1).values
    if precomp is None:
        ci, Rm = _build_cov_inv(scales, rotations, return_R=True)
        sr = cov_threshold * scales.max(dim=-1).values
        dp = {'cov_inv':ci,'R':Rm,'scales_vec':scales,'search_radius':sr,
              'pred_class':torch.zeros(means.shape[0],dtype=torch.long,device=means.device)}
    else:
        dp = {'cov_inv':precomp['cov_inv'],'R':precomp['R'],'scales_vec':precomp['scales_vec'],
              'search_radius':precomp['search_radius'],
              'pred_class':torch.zeros(means.shape[0],dtype=torch.long,device=means.device)}
    covered, _, _ = compute_coverage_and_purity(means, dp, vx,
        torch.zeros(vx.shape[0], dtype=torch.long, device=vx.device),
        cov_threshold=cov_threshold, chunk_size=chunk_size)
    r = {}
    for i in range(len(distance_bins)-1):
        lo, hi = distance_bins[i], distance_bins[i+1]
        bm = (vd>=lo)&(vd<hi); t = bm.sum().item()
        r[f'({lo},{hi})'] = covered[bm].sum().item()/t if t else 0.0
    return r


def compute_category_stats(scales, semantics, rotations, means, occ_xyz, occ_label,
    occ_cam_mask, t_sphere, num_classes, cov_threshold=3.0, precomp=None, chunk_size=10000,
    empty_label=17, ignore_empty=True):
    G = scales.shape[0]
    if G == 0: return {}
    s_hat = scales.mean(dim=-1)
    ar = scales.max(dim=-1).values/(scales.min(dim=-1).values+1e-8)
    pc = semantics.argmax(dim=-1)
    ofx = occ_xyz.reshape(-1,3); ofl = occ_label.reshape(-1).long()
    mf = occ_cam_mask.reshape(-1).bool()
    # 排除 empty voxels
    if ignore_empty and ofl is not None:
        mf = mf & (ofl != empty_label)
    ppg = torch.zeros(G, dtype=torch.float32, device=scales.device)
    if mf.any():
        vx = ofx[mf]; vl = ofl[mf]
        if precomp is None:
            ci, Rm = _build_cov_inv(scales, rotations, return_R=True)
            sr = cov_threshold * scales.max(dim=-1).values
            lp = {'cov_inv':ci,'R':Rm,'scales_vec':scales,'search_radius':sr,'pred_class':pc}
        else:
            lp = precomp
        _, pm, pt = compute_coverage_and_purity(means, lp, vx, vl,
            cov_threshold=cov_threshold, chunk_size=chunk_size)
        # 向量化：unused Gaussian purity = 0.0
        valid_g = pt > 0
        ppg[valid_g] = pm[valid_g].float() / pt[valid_g].float().clamp(min=1)
        # unused Gaussian 保持 ppg=0.0
    r = {}
    for c in range(num_classes):
        m = pc == c; cnt = m.sum().item()
        r[c] = {'count':cnt, 'mean_scale':s_hat[m].mean().item()if cnt else 0,
                'mean_ar':ar[m].mean().item()if cnt else 0,
                'near_spherical_ratio':(ar[m]<t_sphere).float().mean().item()if cnt else 0,
                'mean_purity':ppg[m].mean().item()if cnt else 0}
    return r


def mask_flat_valid(occ_cam_mask):
    """检查 cam_mask 是否有任何 True 值。"""
    return occ_cam_mask.reshape(-1).bool().any().item()
