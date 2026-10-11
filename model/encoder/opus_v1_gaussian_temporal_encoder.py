"""Independent temporal Gaussian decoder with activation recomputation."""
import torch
from torch.utils.checkpoint import checkpoint
from mmengine.registry import MODELS

from .opus_v1_gaussian_softplus_encoder import OPUSV1GaussianSoftplusEncoder


@MODELS.register_module()
class OPUSV1GaussianTemporalEncoder(OPUSV1GaussianSoftplusEncoder):
    def __init__(self, decoder_checkpoint=True, **kwargs):
        super().__init__(**kwargs)
        self.decoder_checkpoint = bool(decoder_checkpoint)

    def forward(self, query_features, query_points, ms_img_feats, metas,
                query_valid_mask=None, **kwargs):
        batch = query_features.shape[0]
        if query_valid_mask is None:
            query_valid_mask = torch.ones(query_features.shape[:2],
                                          device=query_features.device, dtype=torch.bool)
        grouped = []
        for feature in ms_img_feats:
            _, cameras, channels, height, width = feature.shape
            if cameras != self.num_frames * self.num_views:
                raise ValueError('all OPUS levels must contain T * 6 images')
            grouped.append(feature.reshape(
                batch, self.num_frames, self.num_views, self.num_groups,
                channels // self.num_groups, height, width
            ).permute(0, 1, 3, 2, 5, 6, 4).reshape(
                batch * self.num_frames * self.num_groups, self.num_views,
                height, width, channels // self.num_groups).contiguous())
        points = query_points.unsqueeze(2)
        representation = []
        for stage, layer in enumerate(self.layers):
            if self.training and torch.is_grad_enabled() and self.decoder_checkpoint:
                # Bind each layer at construction: backward recomputation must
                # not capture the loop's final layer.
                def run(p, q, *features, decoder=layer):
                    return decoder(p, q, features, metas)
                query_features, logits, points, visible = checkpoint(
                    run, points, query_features, *grouped,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                query_features, logits, points, visible = layer(
                    points, query_features, grouped, metas)
            representation.append(dict(
                query_features=query_features, query_points=points,
                opus_logits=logits,
                gaussian=self._make_gaussian(query_features, points, logits,
                                             query_valid_mask, stage),
                query_valid_mask=query_valid_mask, visible=visible))
            points = points.detach()
        return {'representation': representation}
