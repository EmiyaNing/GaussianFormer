"""A small set-prediction detector reading a frozen Gaussian representation.

SparseBEV-style box coding, sparse query refinement and one-to-one matching.
The detector's spatial pooling and decoder belong entirely to this new head.
No image features or ground-truth proposals enter the frozen encoder.
"""

import math
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


DETECTION_CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                     'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')


def pool_gaussian_queries(features, gaussian, pc_range, resolution=1., *, detach=True):
    """Pool queries, optionally retaining feature/semantic/position gradients.

    Frozen readouts retain the original detached default. Joint training uses
    detach=False; grid membership stays discrete, while pooled values and
    within-cell positions carry gradients to the original Gaussian encoder.
    """
    with torch.no_grad() if detach else nullcontext():
        return _pool_gaussian_queries(features, gaussian, pc_range, resolution)


def _pool_gaussian_queries(features, gaussian, pc_range, resolution):
    assert features.shape[0] == 1 and features.shape[:2] == gaussian.means.shape[:2]
    h = round((pc_range[4]-pc_range[1]) / resolution)
    w = round((pc_range[3]-pc_range[0]) / resolution)
    xyz = gaussian.means[0].float()
    cell = torch.floor((xyz[:, :2]-xyz.new_tensor(pc_range[:2])) / resolution).long()
    keep = (cell[:, 0] >= 0) & (cell[:, 0] < w) & (cell[:, 1] >= 0) & (cell[:, 1] < h)
    ids = cell[keep, 1]*w + cell[keep, 0]
    data = torch.cat([features[0].float(), gaussian.semantics[0].float(), xyz], -1)[keep]
    pooled = data.new_zeros((h*w, data.shape[-1]))
    counts = data.new_zeros((h*w, 1))
    pooled.index_add_(0, ids, data)
    counts.index_add_(0, ids, torch.ones_like(data[:, :1]))
    mean = pooled/counts.clamp_min(1)
    # Only fill empty cells from nearby observed cells. Occupied cells retain
    # their own averaged features, and count is an explicit validity cue.
    raw = pooled.T.reshape(1, -1, h, w)
    count_map = counts.T.reshape(1, 1, h, w)
    local_count = F.avg_pool2d(count_map, 3, 1, 1)
    local = F.avg_pool2d(raw, 3, 1, 1)/local_count.clamp_min(1e-6)
    mean = mean.T.reshape(1, -1, h, w)
    mean = torch.where(count_map > 0, mean, local)
    yy, xx = torch.meshgrid(torch.arange(h, device=xyz.device), torch.arange(w, device=xyz.device), indexing='ij')
    center = torch.stack([(xx+.5)*resolution+pc_range[0],
                           (yy+.5)*resolution+pc_range[1]], 0)[None]
    xy = torch.where(local_count > 0, mean[:, -3:-1]-center, 0.)
    mean = torch.cat([mean[:, :-3], xy, mean[:, -1:]], 1)
    return torch.cat([mean, count_map.log1p()], 1)


class SparseBoxLayer(nn.Module):
    def __init__(self, dim, pc_range):
        super().__init__()
        self.register_buffer('lower', torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer('upper', torch.tensor(pc_range[3:], dtype=torch.float32))
        self.position = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.self_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.offset = nn.Linear(dim, 9*2)
        self.weights = nn.Linear(dim, 9)
        self.value = nn.Linear(dim, dim)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
        self.classification = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 10))
        self.regression = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 10))
        offsets = torch.cartesian_prod(torch.tensor([-2., 0., 2.]), torch.tensor([-2., 0., 2.]))
        self.register_buffer('base_offsets', offsets)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        nn.init.constant_(self.classification[-1].bias, -math.log(99.))
        nn.init.zeros_(self.regression[-1].weight)
        with torch.no_grad():
            self.regression[-1].bias.copy_(torch.tensor(
                [0, 0, 0, math.log(1.8), math.log(4.), math.log(1.6), 0, 1, 0, 0]))

    def forward(self, query, references, feature_map):
        position = self.position((references-self.lower)/(self.upper-self.lower))
        update = self.self_attention(query+position, query+position, query, need_weights=False)[0]
        query = self.norms[0](query+update)
        offsets = self.base_offsets + self.offset(query).reshape(*query.shape[:2], 9, 2).tanh()*4.
        locations = references[:, :, None, :2] + offsets
        grid = (locations-self.lower[:2])/(self.upper[:2]-self.lower[:2])*2-1
        samples = F.grid_sample(feature_map, grid, align_corners=False).permute(0, 2, 3, 1)
        update = (self.value(samples)*self.weights(query).softmax(-1)[..., None]).sum(2)
        query = self.norms[1](query+update)
        query = self.norms[2](query+self.ffn(query))
        raw = self.regression(query)
        center = torch.maximum(torch.minimum(references+raw[..., :3], self.upper-1e-4), self.lower+1e-4)
        boxes = torch.cat([center, raw[..., 3:]], -1)
        return query, self.classification(query), boxes


class LightGaussianBBoxHead(nn.Module):
    def __init__(self, pc_range=(-40, -40, -1, 40, 40, 5.4), dim=128, queries=600,
                 layers=2, resolution=1., input_channels=149):
        super().__init__()
        self.pc_range = tuple(pc_range)
        self.queries = queries
        self.resolution = resolution
        self.projection = nn.Sequential(nn.Conv2d(input_channels, dim, 1), nn.GroupNorm(8, dim), nn.ReLU())
        self.context = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                                     nn.Conv2d(dim, dim, 1), nn.GroupNorm(8, dim), nn.ReLU())
        self.proposal = nn.Conv2d(dim, 10, 1)
        self.label_embedding = nn.Embedding(10, dim)
        self.layers = nn.ModuleList([SparseBoxLayer(dim, pc_range) for _ in range(layers)])
        nn.init.constant_(self.proposal.bias, -math.log(99.))

    def forward(self, pooled):
        assert pooled.shape[0] == 1
        feature_map = self.projection(pooled.float())
        feature_map = feature_map+self.context(feature_map)
        dense = self.proposal(feature_map)
        b, _, h, w = dense.shape
        probability = dense.detach().sigmoid()
        keep = probability == F.max_pool2d(probability, 3, 1, 1)
        ranks = probability.masked_fill(~keep, 0).flatten(1)
        _, selected = ranks.topk(min(self.queries, ranks.shape[-1]), dim=-1)
        cells, labels = selected % (h*w), selected // (h*w)
        references = torch.stack([
            (cells % w + .5)*self.resolution+self.pc_range[0],
            (cells // w + .5)*self.resolution+self.pc_range[1],
            pooled[:, -2].reshape(b, -1).gather(1, cells)], -1)
        query = feature_map.flatten(2).transpose(1, 2).gather(
            1, cells[..., None].expand(-1, -1, feature_map.shape[1]))
        query = query+self.label_embedding(labels)
        stages = []
        for layer in self.layers:
            query, classification, boxes = layer(query, references, feature_map)
            stages.append((classification, boxes))
            references = boxes[..., :3].detach()
        return dict(stages=stages, proposal_logits=dense, proposal_cells=cells)

    @torch.no_grad()
    def predict(self, output, max_boxes=300):
        classification, codes = output['stages'][-1]
        scores, indices = classification[0].sigmoid().flatten().topk(
            min(max_boxes, classification.shape[1]*10))
        query, labels = indices // 10, indices % 10
        code = codes[0, query]
        boxes = torch.cat([code[:, :3], code[:, 3:6].clamp(-2.3, 3.4).exp(),
                           torch.atan2(code[:, 6:7], code[:, 7:8]), code[:, 8:10]], -1)
        return dict(boxes=boxes.cpu().numpy(), scores=scores.cpu().numpy(), labels=labels.cpu().numpy())


def encode_boxes(boxes):
    return torch.cat([boxes[:, :3], boxes[:, 3:6].clamp_min(.05).log(),
                      boxes[:, 6:7].sin(), boxes[:, 6:7].cos(), boxes[:, 7:9]], -1)


def sigmoid_focal(logits, targets):
    probability = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = probability*targets+(1-probability)*(1-targets)
    alpha = .25*targets+.75*(1-targets)
    return (alpha*(1-pt).square()*ce).sum()


class SparseDetectionLoss(nn.Module):
    def __init__(self, pc_range=(-40, -40, -1, 40, 40, 5.4), resolution=1.):
        super().__init__()
        self.pc_range = tuple(pc_range)
        self.resolution = resolution
        self.register_buffer('code_weights', torch.tensor([2., 2., 1., 1., 1., 1., 1., 1., .2, .2]))

    def heatmap_target(self, logits, boxes, labels):
        target = torch.zeros_like(logits)
        h, w = logits.shape[-2:]
        for box, label in zip(boxes, labels):
            x = int(torch.floor((box[0]-self.pc_range[0])/self.resolution))
            y = int(torch.floor((box[1]-self.pc_range[1])/self.resolution))
            if not (0 <= x < w and 0 <= y < h):
                continue
            radius = max(1, min(4, int(min(float(box[3]), float(box[4]))/self.resolution/2)))
            yy, xx = torch.meshgrid(torch.arange(max(0, y-radius), min(h, y+radius+1), device=logits.device),
                                    torch.arange(max(0, x-radius), min(w, x+radius+1), device=logits.device), indexing='ij')
            gaussian = torch.exp(-((xx-x).square()+(yy-y).square())/(2*(radius/2+.5)**2))
            target[0, label, yy, xx] = torch.maximum(target[0, label, yy, xx], gaussian)
        return target

    def forward(self, output, boxes, labels):
        assert boxes.ndim == 2 and labels.ndim == 1
        encoded = encode_boxes(boxes)
        finite = torch.isfinite(encoded)
        encoded = encoded.nan_to_num()
        classification_losses, box_losses, matched_distances = [], [], []
        for scores, codes in output['stages']:
            scores, codes = scores[0], codes[0]
            target = torch.zeros_like(scores)
            if len(boxes):
                probability = scores.detach().sigmoid().clamp(1e-6, 1-1e-6)
                positive = -.25*(1-probability).square()*probability.log()
                negative = -.75*probability.square()*(1-probability).log()
                class_cost = positive[:, labels]-negative[:, labels]
                # Matching ignores unavailable motion and uses geometry/yaw.
                box_cost = torch.cdist(codes.detach()[:, :8]*self.code_weights[:8],
                                       encoded[:, :8]*self.code_weights[:8], p=1)
                rows, columns = linear_sum_assignment((2*class_cost+.25*box_cost).cpu().numpy())
                rows = torch.as_tensor(rows, device=scores.device)
                columns = torch.as_tensor(columns, device=scores.device)
                target[rows, labels[columns]] = 1
                difference = (codes[rows]-encoded[columns]).abs()*finite[columns]*self.code_weights
                box_losses.append(difference.sum()/max(1, len(boxes)))
                matched_distances.append((codes[rows, :2]-boxes[columns, :2]).norm(dim=-1).mean().detach())
            else:
                box_losses.append(codes.sum()*0)
            classification_losses.append(sigmoid_focal(scores, target)/max(1, len(boxes)))
        logits = output['proposal_logits']
        heatmap = self.heatmap_target(logits, boxes, labels)
        p = logits.sigmoid().clamp(1e-5, 1-1e-5)
        center = heatmap.eq(1)
        dense = -(p.log()*(1-p).square()*center +
                  (1-p).log()*p.square()*(1-heatmap).pow(4)*~center).sum()/center.sum().clamp_min(1)
        cls = torch.stack(classification_losses).mean()
        reg = torch.stack(box_losses).mean()
        total = 2*cls+.25*reg+dense
        return total, dict(classification=float(cls.detach()), regression=float(reg.detach()),
                           proposal=float(dense.detach()),
                           matched_center_distance=float(torch.stack(matched_distances).mean()) if matched_distances else 0.)
