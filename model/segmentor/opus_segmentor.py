from mmseg.models import SEGMENTORS
from .bev_segmentor import BEVSegmentor


@SEGMENTORS.register_module()
class OPUSSegmentor(BEVSegmentor):
    """Image-query OPUS segmentor using the repository's established pipeline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
