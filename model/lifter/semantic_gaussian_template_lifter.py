"""Learnable semantic-Gaussian templates for the Phase-A Gaussian-OPUS model."""
import torch
from torch import nn
from mmseg.registry import MODELS

from .base_lifter import BaseLifter
from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.utils.safe_ops import safe_inverse_sigmoid


@MODELS.register_module()
class SemanticGaussianTemplateLifter(BaseLifter):
    """Initialise fixed-cardinality OPUS queries as semantic Gaussian templates.

    Means and scales exposed to the encoder/head are in world coordinates.
    Their learnable parameters are stored as sigmoid logits to keep templates
    inside the configured scene and scale interval at initialisation.
    """

    def __init__(self, num_templates=600, embed_dims=256, semantic_dim=17,
                 pc_range=(-40., -40., -1., 40., 40., 5.4),
                 scale_range=((0.15, 0.15, 0.15), (8., 8., 4.)),
                 initial_scale=(1.6, 1.6, 1.0), initial_opacity=0.1,
                 query_grad=True, feature_grad=True, init_cfg=None):
        super().__init__(init_cfg)
        self.num_templates = num_templates
        self.embed_dims = embed_dims
        self.semantic_dim = semantic_dim
        self.pc_range = tuple(pc_range)
        self.register_buffer('scale_min', torch.tensor(scale_range[0], dtype=torch.float32))
        self.register_buffer('scale_max', torch.tensor(scale_range[1], dtype=torch.float32))

        mean_prob = torch.empty(num_templates, 3).uniform_(0.05, 0.95)
        scale = torch.tensor(initial_scale, dtype=torch.float32).expand(num_templates, -1)
        scale_prob = (scale - self.scale_min) / (self.scale_max - self.scale_min)
        self.mean_logits = nn.Parameter(safe_inverse_sigmoid(mean_prob), requires_grad=query_grad)
        self.scale_logits = nn.Parameter(
            safe_inverse_sigmoid(scale_prob.clamp(1e-4, 1 - 1e-4)), requires_grad=query_grad)
        rotation = torch.zeros(num_templates, 4)
        rotation[:, 0] = 1.
        self.rotation = nn.Parameter(rotation, requires_grad=query_grad)
        self.opacity_logits = nn.Parameter(
            safe_inverse_sigmoid(torch.full((num_templates, 1), initial_opacity)),
            requires_grad=query_grad)
        self.semantic_logits = nn.Parameter(torch.zeros(num_templates, semantic_dim),
                                            requires_grad=query_grad)
        self.query_features = nn.Parameter(torch.empty(num_templates, embed_dims),
                                           requires_grad=feature_grad)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.query_features)

    def _template(self, batch_size):
        lower = self.mean_logits.new_tensor(self.pc_range[:3])
        extent = self.mean_logits.new_tensor(self.pc_range[3:]) - lower
        means = lower + self.mean_logits.sigmoid() * extent
        scales = self.scale_min + self.scale_logits.sigmoid() * (self.scale_max - self.scale_min)
        rotations = torch.nn.functional.normalize(self.rotation, dim=-1, eps=1e-6)
        return GaussianPrediction(
            means=means.unsqueeze(0).expand(batch_size, -1, -1),
            scales=scales.unsqueeze(0).expand(batch_size, -1, -1),
            rotations=rotations.unsqueeze(0).expand(batch_size, -1, -1),
            opacities=self.opacity_logits.sigmoid().unsqueeze(0).expand(batch_size, -1, -1),
            semantics=self.semantic_logits.unsqueeze(0).expand(batch_size, -1, -1),
        )

    def forward(self, ms_img_feats, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        template = self._template(batch_size)
        return {
            'query_features': self.query_features.unsqueeze(0).expand(batch_size, -1, -1),
            'query_templates': template,
            # Kept for debugging and generic segmentor compatibility only.
            'query_points': template.means,
        }
