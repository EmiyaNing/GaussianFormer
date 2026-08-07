"""GaussianFormer runner adapter for the strict official OPUSv2 head."""
from mmseg.models import SEGMENTORS

from .base_segmentor import CustomBaseSegmentor


@SEGMENTORS.register_module()
class OfficialOPUSV2Segmentor(CustomBaseSegmentor):
    """Keep the official detector's backbone -> FPN -> OPUSv2 flow.

    OPUSv2 owns its initial queries and transformer inside ``head``.  This is
    deliberate: it preserves the official ``pts_bbox_head.*`` state hierarchy
    under a single local ``head.*`` prefix.
    """

    def forward(self, imgs, metas, **kwargs):
        features = self.extract_img_feat(imgs=imgs)
        return self.head(ms_img_feats=features['ms_img_feats'], metas=metas, **kwargs)
