"""Auxiliary supervision and observability for AdaptiveAllocationV6."""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


def _max_scale_axis(
    rotation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    max_scale_dim = scale.argmax(dim=-1, keepdim=True)
    w, x, y, z = rotation.unbind(dim=-1)
    columns = torch.stack([
        torch.stack([
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
        ], dim=-1),
        torch.stack([
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        ], dim=-1),
        torch.stack([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ], dim=-1),
    ], dim=-2)
    index = max_scale_dim.unsqueeze(-1).expand(*max_scale_dim.shape, 3)
    return F.normalize(
        columns.gather(dim=-2, index=index).squeeze(-2),
        dim=-1,
        eps=1e-6,
    )


def _radius_neighbors(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    radius: float,
) -> torch.Tensor:
    """Return bounded-radius neighbor indices for CUDA training and CPU tests."""
    k = min(int(k), reference.shape[1])
    if k <= 0:
        return torch.empty(
            *query.shape[:2], 0,
            dtype=torch.long,
            device=query.device,
        )

    if query.is_cuda:
        try:
            import frnn

            _, index, _, _ = frnn.frnn_grid_points(
                query.float(),
                reference.float(),
                K=k,
                r=radius,
                return_nn=True,
            )
            return index.long()
        except (ImportError, RuntimeError):
            pass

    outputs = []
    query_chunk = 256
    reference_chunk = 4096
    for batch_id in range(query.shape[0]):
        batch_outputs = []
        for query_start in range(0, query.shape[1], query_chunk):
            current_query = query[
                batch_id,
                query_start:query_start + query_chunk,
            ].float()
            best_distance = current_query.new_full(
                (current_query.shape[0], k),
                float("inf"),
            )
            best_index = torch.full(
                (current_query.shape[0], k),
                -1,
                dtype=torch.long,
                device=query.device,
            )
            for ref_start in range(0, reference.shape[1], reference_chunk):
                current_reference = reference[
                    batch_id,
                    ref_start:ref_start + reference_chunk,
                ].float()
                distance = torch.cdist(current_query, current_reference)
                distance = distance.masked_fill(distance > radius, float("inf"))
                ref_index = torch.arange(
                    ref_start,
                    ref_start + current_reference.shape[0],
                    device=query.device,
                ).expand(current_query.shape[0], -1)
                merged_distance = torch.cat([best_distance, distance], dim=1)
                merged_index = torch.cat([best_index, ref_index], dim=1)
                best_distance, order = merged_distance.topk(
                    k,
                    largest=False,
                    dim=1,
                )
                best_index = merged_index.gather(1, order)
            best_index = best_index.masked_fill(
                ~torch.isfinite(best_distance),
                -1,
            )
            batch_outputs.append(best_index)
        outputs.append(torch.cat(batch_outputs, dim=0))
    return torch.stack(outputs, dim=0)


def _gather(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(value.shape[0], device=value.device)[:, None, None]
    return value[batch, index.clamp_min(0)]


def _entropy(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = logits.softmax(dim=-1)
    entropy = -(probability * (probability + eps).log()).sum(dim=-1)
    return entropy / max(math.log(logits.shape[-1]), eps)


def _anisotropy(scales: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return scales.amax(dim=-1) / scales.amin(dim=-1).clamp_min(eps)


def _weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(eps)


@OPENOCC_LOSS.register_module()
class AdaptiveAllocationV6Loss(nn.Module):
    """Train V6's selector/router and report Gaussian allocation metrics.

    Four proxy targets are generated from occupancy GT and local Gaussian
    statistics.  They supervise risk ranking and operation behavior; no
    explicit four-channel diagnostic head is introduced.
    """

    def __init__(
        self,
        weight=1.0,
        empty_label=17,
        semantic_dim=17,
        support_k=27,
        support_radius=2.0,
        coverage_k=8,
        coverage_radius=3.0,
        max_coverage_voxels=65536,
        coverage_threshold=0.5,
        coverage_temperature=0.1,
        semantic_gt_mix=0.8,
        risk_bce_weight=0.1,
        risk_ranking_weight=0.1,
        operation_kl_weight=0.02,
        budget_weight=0.05,
        outcome_weight=0.02,
        ranking_temperature=0.1,
        split_entropy_margin=0.02,
        split_min_js=0.02,
        **kwargs,
    ):
        super().__init__()
        self.weight = float(weight)
        self.empty_label = int(empty_label)
        self.semantic_dim = int(semantic_dim)
        self.support_k = int(support_k)
        self.support_radius = float(support_radius)
        self.coverage_k = int(coverage_k)
        self.coverage_radius = float(coverage_radius)
        self.max_coverage_voxels = int(max_coverage_voxels)
        self.coverage_threshold = float(coverage_threshold)
        self.coverage_temperature = float(coverage_temperature)
        self.semantic_gt_mix = float(semantic_gt_mix)
        self.risk_bce_weight = float(risk_bce_weight)
        self.risk_ranking_weight = float(risk_ranking_weight)
        self.operation_kl_weight = float(operation_kl_weight)
        self.budget_weight = float(budget_weight)
        self.outcome_weight = float(outcome_weight)
        self.ranking_temperature = float(ranking_temperature)
        self.split_entropy_margin = float(split_entropy_margin)
        self.split_min_js = float(split_min_js)

    def _semantic_geometry_targets(
        self,
        aux: Dict,
        sampled_xyz: torch.Tensor,
        sampled_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gaussian = aux["input_gaussian"]
        means = gaussian.means.detach().float()
        scales = gaussian.scales.detach().float().clamp_min(1e-4)
        rotations = gaussian.rotations.detach().float()
        semantics = gaussian.semantics.detach().float()
        xyz = sampled_xyz.detach().float()
        labels = sampled_label.detach().long()

        index = _radius_neighbors(
            means,
            xyz,
            self.support_k,
            self.support_radius,
        )
        valid_index = index >= 0
        neighbor_labels = _gather(labels.unsqueeze(-1), index).squeeze(-1)
        occupied = (
            valid_index
            & (neighbor_labels != self.empty_label)
            & (neighbor_labels >= 0)
            & (neighbor_labels < self.semantic_dim)
        )
        occupied_f = occupied.float()
        support_count = occupied_f.sum(dim=-1)

        histogram = means.new_zeros(
            means.shape[0],
            means.shape[1],
            self.semantic_dim,
        )
        histogram.scatter_add_(
            2,
            neighbor_labels.clamp(0, self.semantic_dim - 1),
            occupied_f,
        )
        distribution = histogram / support_count.unsqueeze(-1).clamp_min(1.0)
        gt_entropy = -(
            distribution * (distribution + 1e-6).log()
        ).sum(dim=-1) / math.log(self.semantic_dim)
        pred_probability = semantics.softmax(dim=-1)
        pred_entropy = -(
            pred_probability * (pred_probability + 1e-6).log()
        ).sum(dim=-1) / math.log(self.semantic_dim)
        semantic_target = torch.where(
            support_count > 0,
            self.semantic_gt_mix * gt_entropy
            + (1 - self.semantic_gt_mix) * pred_entropy,
            pred_entropy,
        ).clamp(0, 1)

        predicted_class = semantics.argmax(dim=-1, keepdim=True)
        predicted_support = histogram.gather(2, predicted_class).squeeze(-1)
        support_error = torch.where(
            support_count > 0,
            1.0 - predicted_support / support_count.clamp_min(1.0),
            torch.zeros_like(support_count),
        )

        neighbor_xyz = _gather(xyz, index)
        weight = occupied_f.unsqueeze(-1)
        local_mean = (neighbor_xyz * weight).sum(dim=2) / weight.sum(
            dim=2,
        ).clamp_min(1.0)
        centered = (neighbor_xyz - local_mean.unsqueeze(2)) * weight
        covariance = torch.einsum(
            "bnki,bnkj->bnij",
            centered,
            centered,
        ) / support_count[..., None, None].clamp_min(1.0)
        identity = torch.eye(3, device=means.device)[None, None]
        eigenvalue, eigenvector = torch.linalg.eigh(covariance + identity * 1e-5)
        local_axis = eigenvector[..., -1]
        gaussian_axis = _max_scale_axis(rotations, scales)
        axis_mismatch = 1.0 - (
            local_axis * gaussian_axis
        ).sum(dim=-1).abs().clamp(0, 1)

        gaussian_ratio = _anisotropy(scales)
        local_ratio = torch.sqrt(
            eigenvalue[..., -1].clamp_min(1e-5)
            / eigenvalue[..., 0].clamp_min(1e-5)
        )
        ratio_mismatch = (
            (gaussian_ratio.log() - local_ratio.log()).abs()
            / (gaussian_ratio.log().abs() + local_ratio.log().abs() + 1e-6)
        ).clamp(0, 1)
        shape_mismatch = 0.5 * (axis_mismatch + ratio_mismatch)
        shape_mismatch = torch.where(
            support_count >= 3,
            shape_mismatch,
            torch.zeros_like(shape_mismatch),
        )
        raw_anisotropy = (
            gaussian_ratio.log() / math.log(16.0)
        ).clamp(0, 1)
        geometry_target = (
            raw_anisotropy * torch.maximum(support_error, shape_mismatch)
        ).clamp(0, 1)
        return semantic_target, geometry_target, support_error

    def _coverage_targets(
        self,
        aux: Dict,
        sampled_xyz: torch.Tensor,
        sampled_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gaussian = aux["input_gaussian"]
        means = gaussian.means.detach().float()
        scales = gaussian.scales.detach().float().clamp_min(1e-4)
        opacities = gaussian.opacities.detach().float().squeeze(-1)
        coverage_target = means.new_zeros(means.shape[:2])
        direction_target = means.new_zeros(means.shape)

        for batch_id in range(means.shape[0]):
            batch_label = sampled_label[batch_id].detach()
            occupied_index = torch.where(
                (batch_label >= 0) & (batch_label < self.semantic_dim),
            )[0]
            if occupied_index.numel() == 0:
                continue
            if occupied_index.numel() > self.max_coverage_voxels:
                sample_position = torch.linspace(
                    0,
                    occupied_index.numel() - 1,
                    self.max_coverage_voxels,
                    device=occupied_index.device,
                ).long()
                occupied_index = occupied_index[sample_position]

            points = sampled_xyz[
                batch_id:batch_id + 1,
                occupied_index,
            ].detach().float()
            index = _radius_neighbors(
                points,
                means[batch_id:batch_id + 1],
                self.coverage_k,
                self.coverage_radius,
            )[0]
            valid = index >= 0
            safe_index = index.clamp_min(0)
            neighbor_means = means[batch_id, safe_index]
            neighbor_scales = scales[batch_id, safe_index]
            neighbor_opacity = opacities[batch_id, safe_index]
            delta = points[0, :, None] - neighbor_means
            mahalanobis = (
                delta.square() / neighbor_scales.square().clamp_min(1e-6)
            ).sum(dim=-1)
            affinity = torch.exp(-0.5 * mahalanobis) * neighbor_opacity
            affinity = affinity * valid.float()
            coverage = affinity.sum(dim=-1)
            uncovered = torch.sigmoid(
                (self.coverage_threshold - coverage)
                / max(self.coverage_temperature, 1e-6),
            )
            assignment = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(
                1e-6,
            )
            assignment = assignment * valid.float()
            weighted_uncovered = uncovered[:, None] * assignment

            numerator = means.new_zeros(means.shape[1])
            denominator = means.new_zeros(means.shape[1])
            numerator.scatter_add_(
                0,
                safe_index.reshape(-1),
                weighted_uncovered.reshape(-1),
            )
            denominator.scatter_add_(
                0,
                safe_index.reshape(-1),
                assignment.reshape(-1),
            )
            coverage_target[batch_id] = (
                numerator / denominator.clamp_min(1e-6)
            ).clamp(0, 1)

            direction_numerator = means.new_zeros(means.shape[1], 3)
            for axis in range(3):
                direction_numerator[:, axis].scatter_add_(
                    0,
                    safe_index.reshape(-1),
                    (weighted_uncovered * delta[..., axis]).reshape(-1),
                )
            direction_target[batch_id] = F.normalize(
                direction_numerator,
                dim=-1,
                eps=1e-6,
            )
        return coverage_target, direction_target

    def _build_targets(
        self,
        aux: Dict,
        sampled_xyz: torch.Tensor,
        sampled_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            semantic, geometry, _ = self._semantic_geometry_targets(
                aux,
                sampled_xyz,
                sampled_label,
            )
            coverage, clone_direction = self._coverage_targets(
                aux,
                sampled_xyz,
                sampled_label,
            )
            local = aux["local_cue"].float()
            overlap = local[..., 2].clamp(0, 1)
            agreement = local[..., 3].clamp(0, 1)
            density = local[..., 4].clamp(0, 1)
            opacity = aux["input_gaussian"].opacities.detach().float().squeeze(-1)
            marginal_redundancy = 1.0 - opacity / (
                opacity + density + 1e-6
            )
            redundancy = (
                overlap
                * agreement
                * (1.0 - coverage)
                * (1.0 - semantic)
                * marginal_redundancy.clamp(0, 1)
            ).clamp(0, 1)
            risk = torch.stack([
                semantic,
                geometry,
                coverage,
                redundancy,
            ], dim=-1).amax(dim=-1)
            operation = torch.stack([
                coverage,
                torch.maximum(semantic, geometry),
                redundancy,
            ], dim=-1)
            operation = (operation + 1e-4) / (
                operation.sum(dim=-1, keepdim=True) + 3e-4
            )
            return {
                "semantic": semantic,
                "geometry": geometry,
                "coverage": coverage,
                "redundancy": redundancy,
                "risk": risk,
                "operation": operation,
                "clone_direction": clone_direction,
            }

    def _risk_ranking_loss(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        count = max(1, score.shape[1] // 4)
        high_target, high_index = target.topk(count, dim=1)
        low_target, low_index = target.topk(count, dim=1, largest=False)
        high_score = score.gather(1, high_index)
        low_score = score.gather(1, low_index)
        target_gap = (high_target - low_target).clamp_min(0)
        ranking = F.softplus(
            -(high_score - low_score)
            / max(self.ranking_temperature, 1e-6),
        )
        return (ranking * target_gap).sum() / target_gap.sum().clamp_min(1e-6)

    def _outcome_loss(
        self,
        aux: Dict,
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = aux["risk_score"].sum() * 0.0
        clone_losses = []
        split_losses = []
        atten_losses = []
        clone_alignment = []
        purity_gain = []
        anisotropy_gain = []

        for batch_id, record in enumerate(aux["records"]):
            clone_index = record["clone_index"]
            if clone_index.numel() > 0:
                parent = record["clone_parent"]
                child = record["clone_child"]
                predicted_direction = F.normalize(
                    child.means - parent.means,
                    dim=-1,
                    eps=1e-6,
                )
                target_direction = targets["clone_direction"][
                    batch_id,
                    clone_index,
                ]
                alignment = (
                    predicted_direction * target_direction
                ).sum(dim=-1).clamp(-1, 1)
                weight = targets["coverage"][batch_id, clone_index]
                clone_losses.append(_weighted_mean(1.0 - alignment, weight))
                clone_alignment.append(_weighted_mean(alignment, weight))

            split_index = record["split_index"]
            if split_index.numel() > 0:
                parent = record["split_parent"]
                child_1 = record["split_child_1"]
                child_2 = record["split_child_2"]
                parent_entropy = _entropy(parent.semantics)
                child_entropy = 0.5 * (
                    _entropy(child_1.semantics) + _entropy(child_2.semantics)
                )
                split_weight = torch.maximum(
                    targets["semantic"][batch_id, split_index],
                    targets["geometry"][batch_id, split_index],
                )
                purity = F.relu(
                    child_entropy - parent_entropy + self.split_entropy_margin,
                )

                probability_1 = child_1.semantics.softmax(dim=-1)
                probability_2 = child_2.semantics.softmax(dim=-1)
                mixture = 0.5 * (probability_1 + probability_2)
                js = 0.5 * (
                    (
                        probability_1
                        * ((probability_1 + 1e-6) / (mixture + 1e-6)).log()
                    ).sum(dim=-1)
                    + (
                        probability_2
                        * ((probability_2 + 1e-6) / (mixture + 1e-6)).log()
                    ).sum(dim=-1)
                )
                diversity = F.relu(self.split_min_js - js)

                parent_anisotropy = _anisotropy(parent.scales)
                child_anisotropy = 0.5 * (
                    _anisotropy(child_1.scales)
                    + _anisotropy(child_2.scales)
                )
                geometry_weight = targets["geometry"][batch_id, split_index]
                anisotropy = F.relu(
                    child_anisotropy - 0.95 * parent_anisotropy,
                )
                split_losses.append(
                    _weighted_mean(purity + diversity, split_weight)
                    + _weighted_mean(anisotropy, geometry_weight)
                )
                purity_gain.append(
                    _weighted_mean(parent_entropy - child_entropy, split_weight)
                )
                anisotropy_gain.append(
                    _weighted_mean(
                        parent_anisotropy - child_anisotropy,
                        geometry_weight,
                    )
                )

            atten_index = record["atten_index"]
            if atten_index.numel() > 0:
                factor = record["atten_factor"].squeeze(-1)
                redundancy = targets["redundancy"][batch_id, atten_index]
                target_factor = 1.0 - 0.9 * redundancy
                atten_losses.append(F.smooth_l1_loss(factor, target_factor))

        def average(values: List[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values).mean() if values else zero

        outcome = average(clone_losses) + average(split_losses) + average(
            atten_losses,
        )
        metrics = {
            "coverage_gain_after_clone": average(clone_alignment),
            "purity_gain_after_split": average(purity_gain),
            "anisotropy_gain_after_split": average(anisotropy_gain),
        }
        return outcome, metrics

    @staticmethod
    def _soft_auc(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = score.detach().float().flatten()
        target = target.detach().float().flatten().clamp(0, 1)
        if score.numel() == 0:
            return score.new_tensor(float("nan"))
        order = score.argsort()
        positive = target[order]
        negative = 1.0 - positive
        cumulative_negative = negative.cumsum(dim=0) - negative
        denominator = positive.sum() * negative.sum()
        if denominator <= 1e-6:
            return score.new_tensor(float("nan"))
        return (positive * cumulative_negative).sum() / denominator

    @staticmethod
    def _correlation(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = score.detach().float().flatten()
        target = target.detach().float().flatten()
        score = score - score.mean()
        target = target - target.mean()
        denominator = score.square().sum().sqrt() * target.square().sum().sqrt()
        if denominator <= 1e-6:
            return score.new_zeros(())
        return (score * target).sum() / denominator

    def _metrics(
        self,
        aux: Dict,
        targets: Dict[str, torch.Tensor],
        outcome_metrics: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        risk = aux["risk_score"]
        operation = aux["operation_prob"]
        selected = aux["selected_mask"]
        metrics = {
            "auc_risk_semantic": self._soft_auc(risk, targets["semantic"]),
            "auc_risk_geometry": self._soft_auc(risk, targets["geometry"]),
            "auc_risk_coverage": self._soft_auc(risk, targets["coverage"]),
            "auc_risk_redundancy": self._soft_auc(risk, targets["redundancy"]),
            "auc_risk_combined": self._soft_auc(risk, targets["risk"]),
            "corr_clone_coverage": self._correlation(
                operation[..., 0],
                targets["coverage"],
            ),
            "corr_split_semantic_geometry": self._correlation(
                operation[..., 1],
                torch.maximum(targets["semantic"], targets["geometry"]),
            ),
            "corr_atten_redundancy": self._correlation(
                operation[..., 2],
                targets["redundancy"],
            ),
        }
        for name in ("semantic", "geometry", "coverage", "redundancy", "risk"):
            value = targets[name]
            selected_mean = value[selected].mean() if selected.any() else value.new_zeros(())
            metrics[f"topk_enrichment_{name}"] = (
                (selected_mean + 1e-6) / (value.mean() + 1e-6)
            )

        input_count = aux["input_count"].sum()
        output_count = aux["output_count"].sum()
        selected_count = selected.sum().to(input_count.dtype)
        full_bank_count = input_count + 4.0 * selected_count
        r_input = aux["estimated_r_input"].sum()
        r_output = aux["estimated_r_output"].sum()
        metrics.update({
            "p_out_over_p_in": output_count / input_count.clamp_min(1),
            "estimated_r_out_over_r_in": r_output / r_input.clamp_min(1),
            "estimated_r_per_gaussian": r_output / output_count.clamp_min(1),
            "materialized_ops_per_topk": input_count.new_tensor(1.0),
            "dropped_candidate_ratio": 1.0 - output_count / full_bank_count.clamp_min(1),
            "expected_budget_ratio": (
                aux["expected_output_cost"] / aux["cost_budget"].clamp_min(1)
            ).mean(),
        })
        if torch.cuda.is_available():
            metrics["peak_memory_gib"] = input_count.new_tensor(
                torch.cuda.max_memory_allocated() / (1024 ** 3),
            )
        metrics.update(outcome_metrics)
        return metrics

    def _single_aux_loss(
        self,
        aux: Dict,
        sampled_xyz: torch.Tensor,
        sampled_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        targets = self._build_targets(aux, sampled_xyz, sampled_label)
        risk_score = aux["risk_score"].clamp(1e-6, 1 - 1e-6)
        risk_bce = F.binary_cross_entropy_with_logits(
            aux["risk_logits"].float(),
            targets["risk"].float(),
        )
        ranking = self._risk_ranking_loss(risk_score, targets["risk"])
        operation_kl = (
            targets["operation"]
            * (
                (targets["operation"] + 1e-6).log()
                - (aux["operation_prob"] + 1e-6).log()
            )
        ).sum(dim=-1).mean()
        budget_excess = F.relu(
            aux["expected_output_cost"]
            / aux["cost_budget"].clamp_min(1.0)
            - 1.0,
        ).mean()
        outcome, outcome_metrics = self._outcome_loss(aux, targets)

        loss = (
            self.risk_bce_weight * risk_bce
            + self.risk_ranking_weight * ranking
            + self.operation_kl_weight * operation_kl
            + self.budget_weight * budget_excess
            + self.outcome_weight * outcome
        )
        metrics = self._metrics(aux, targets, outcome_metrics)
        metrics.update({
            "loss_risk_bce": risk_bce.detach(),
            "loss_risk_ranking": ranking.detach(),
            "loss_operation_kl": operation_kl.detach(),
            "loss_budget": budget_excess.detach(),
            "loss_outcome": outcome.detach(),
        })
        return loss, metrics

    @staticmethod
    def _zero_loss(inputs: Dict) -> torch.Tensor:
        predictions = inputs.get("pred_occ", [])
        if predictions:
            return predictions[-1].sum() * 0.0
        return torch.tensor(0.0)

    def forward(self, inputs: Dict):
        allocation_aux = inputs.get("allocation_aux", [])
        if isinstance(allocation_aux, dict):
            allocation_aux = [allocation_aux]
        if not allocation_aux:
            return self._zero_loss(inputs), {"missing_allocation_aux": 1.0}

        sampled_xyz = inputs["sampled_xyz"]
        sampled_label = inputs["sampled_label"]
        losses = []
        metric_lists: Dict[str, List[torch.Tensor]] = {}
        for aux in allocation_aux:
            loss, metrics = self._single_aux_loss(
                aux,
                sampled_xyz,
                sampled_label,
            )
            losses.append(loss)
            for name, value in metrics.items():
                metric_lists.setdefault(name, []).append(
                    torch.as_tensor(value, device=loss.device).float(),
                )

        total = self.weight * torch.stack(losses).mean()
        averaged_metrics = {
            name: torch.stack(values).nanmean().detach()
            for name, values in metric_lists.items()
        }
        return total, averaged_metrics
