"""Inference-only, child-Gaussian semantic routing between fixed forecast paths.

One perception pass is shared. The T-A0 forecast modules and trained G1 modules
unfold independently; selected outputs never feed either path. Slot equality
is an engineering alignment, not a claim of physical object correspondence.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from .pmg_query_update_gate_world import PMGQueryUpdateGateWorld
from .sparse_opus_full_horizon_baseline import SparseOpusFullHorizonBaseline


FORECAST_MODULES = ('ego_cross_attn', 'position_encoder', 'reg_branch',
                    'vel_branch', 'cls_branch', 'traj_head')
# Explicitly excludes traffic_cone (8), which the legacy velocity mask includes.
MOVABLE_CLASS_IDS = (2, 3, 4, 5, 6, 7, 9, 10)


def semantic_mask(logits, class_ids=MOVABLE_CLASS_IDS):
    """Hard route from reference raw class logits, without using GT or scores as probabilities."""
    labels = logits.detach().argmax(-1)
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for index in class_ids:
        selected |= labels == index
    return selected


def select_children(reference, candidate, mask):
    """Select a complete legal Gaussian state per child, never mix only its means."""
    if type(reference) is not type(candidate):
        raise TypeError('Gaussian state types differ')
    fields = []
    for left, right in zip(reference, candidate):
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError('Optional Gaussian field differs between paths')
            fields.append(None)
            continue
        if left.shape != right.shape or left.shape[:mask.ndim] != mask.shape:
            raise ValueError('Reference/candidate child slots do not align')
        expanded = mask.reshape(*mask.shape, *((1,) * (left.ndim - mask.ndim)))
        fields.append(torch.where(expanded, right, left))
    return type(reference)(*fields)


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGSemanticGaussianRouteWorld(PMGQueryUpdateGateWorld):
    """A fixed-weight output intervention, deliberately not a trained gate model."""

    def __init__(self, route_class_ids=MOVABLE_CLASS_IDS, **kwargs):
        super().__init__(**kwargs)
        self.route_class_ids = tuple(int(value) for value in route_class_ids)
        if self.route_class_ids != MOVABLE_CLASS_IDS:
            raise ValueError('This single probe uses the predeclared eight-class set')
        self.reference_forecast_modules = nn.ModuleDict({
            name: deepcopy(getattr(self, name)) for name in FORECAST_MODULES})
        self.route_mode = 'semantic'
        self.collect_route_statistics = False
        self._route_statistics = {}
        self.last_route_audit = []

    @contextmanager
    def reference_modules(self):
        original = {name: getattr(self, name) for name in FORECAST_MODULES}
        try:
            for name in FORECAST_MODULES:
                setattr(self, name, self.reference_forecast_modules[name])
            yield
        finally:
            for name, module in original.items():
                setattr(self, name, module)

    def _forecast(self, outputs, ego_feature, metas):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('This fixed-weight diagnostic requires eval and no_grad')
        with self.reference_modules():
            reference = SparseOpusFullHorizonBaseline._forecast(self, outputs, ego_feature, metas)
        candidate = super()._forecast(outputs, ego_feature, metas)
        self._reference_forecast = reference
        self._candidate_forecast = candidate
        self._route_masks = []
        points, classes = [], []
        for ref_points, ref_logits, cand_points, cand_logits in zip(
                reference[0], reference[1], candidate[0], candidate[1]):
            assert ref_points.shape == cand_points.shape and ref_logits.shape == cand_logits.shape
            mask = semantic_mask(ref_logits, self.route_class_ids)
            if self.route_mode == 'baseline':
                mask = torch.zeros_like(mask)
            elif self.route_mode == 'candidate':
                mask = torch.ones_like(mask)
            elif self.route_mode != 'semantic':
                raise ValueError('Unknown route mode')
            self._route_masks.append(mask)
            points.append(torch.where(mask[..., None], cand_points, ref_points))
            classes.append(torch.where(mask[..., None], cand_logits, ref_logits))
        # Child-level attribute selection happens after Gaussian decoding. A
        # single query feature cannot encode the mixed children. These remain
        # reference contexts; the renderer below uses both actual paths.
        context = candidate[2] if self.route_mode == 'candidate' else reference[2]
        return points, classes, context, reference[3], reference[4]

    def reset_route_statistics(self, enabled=True):
        self._route_statistics = {}
        self.collect_route_statistics = bool(enabled)

    def route_statistics(self):
        rows = []
        for step, record in sorted(self._route_statistics.items()):
            value = record.cpu().tolist()
            total, selected, disagree, distance, maximum = value[:5]
            rows.append(dict(seconds=step*.5, child_evaluations=int(total),
                selected_children=int(selected), fraction_selected=selected/max(total,1),
                selected_semantic_disagreement_fraction=disagree/max(selected,1),
                selected_center_displacement_mean_m=distance/max(selected,1),
                selected_center_displacement_max_m=maximum,
                reference_predicted_class_counts=[int(x) for x in value[5:22]],
                selected_candidate_predicted_class_counts=[int(x) for x in value[22:39]]))
        return dict(rows=rows, route_class_ids=list(self.route_class_ids),
            scope='Predicted reference classes per child slot; not GT motion, objects, or contribution ownership')

    @torch.no_grad()
    def _record_route(self, step, reference, candidate, mask):
        if not self.collect_route_statistics:
            return
        labels = reference.semantics.argmax(-1)
        candidate_labels = candidate.semantics.argmax(-1)
        distance = (candidate.means-reference.means).double().norm(dim=-1)
        selected = distance[mask]
        first = distance.new_tensor([mask.numel(), int(mask.sum()),
            int(((labels != candidate_labels) & mask).sum())])
        second = torch.stack((selected.sum(), selected.max() if selected.numel() else distance.new_zeros(())))
        record = torch.cat((first, second,
            torch.bincount(labels.flatten(),minlength=17).to(distance),
            torch.bincount(candidate_labels[mask],minlength=17).to(distance)))
        if step not in self._route_statistics:
            self._route_statistics[step] = record
        else:
            previous = self._route_statistics[step]
            maximum = torch.maximum(previous[4], record[4])
            previous += record
            previous[4] = maximum

    def _render_occupancy(self, outputs, forecast_points, forecast_classes,
                          forecast_features, current_queries, sampled_xyz,
                          batch_size, render_all=True):
        if self.training or not render_all:
            raise RuntimeError('Semantic routing is an evaluation-only intervention')
        gaussian = outputs['representation'][-1]['gaussian']
        current = type(gaussian)(*(value[:, current_queries] if value is not None else None for value in gaussian))
        logits_by_horizon = {0:self.gaussian_head.render_state(current,sampled_xyz)}
        grids = [logits_by_horizon[0].argmax(1).view(batch_size,200,200,16)]
        encoder = self.pts_bbox_head.transformer.encoder
        self.last_route_audit = []
        for index, mask in enumerate(self._route_masks):
            packs = []
            for path in (self._reference_forecast,self._candidate_forecast):
                points,classes,features = path[0][index],path[1][index],path[2][index]
                valid = torch.ones(features.shape[:2],dtype=torch.bool,device=features.device)
                packs.append(encoder._make_gaussian(features,points,classes,valid,len(encoder.layers)-1))
            reference,candidate = packs
            selected = select_children(reference,candidate,mask)
            # Retained for the engineering audit only, not full-run large caches.
            if getattr(self,'audit_route_fields',False):
                self.last_route_audit.append(dict(
                    retained_reference_exact=all(torch.equal(x[~mask],r[~mask]) for x,r in zip(selected,reference) if x is not None),
                    selected_candidate_exact=all(torch.equal(x[mask],c[mask]) for x,c in zip(selected,candidate) if x is not None)))
            self._record_route(index+1,reference,candidate,mask)
            logits = self.gaussian_head.render_state(selected,sampled_xyz)
            grids.append(logits.argmax(1).view(batch_size,200,200,16))
        return logits_by_horizon,grids
