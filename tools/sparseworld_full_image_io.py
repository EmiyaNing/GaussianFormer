"""Lossless RGB input optimizations, before the unchanged resize/normalization."""
from collections import OrderedDict
import numpy as np


def optimize_image_loaders(ds):
    names=[type(t).__name__ for t in ds.pipeline]
    assert ds.test_mode and ds.data_aug_conf is not None
    assert 'PhotoMetricDistortionMultiViewImage' not in names
    assert names.index('ResizeCropFlipImage')<names.index('NormalizeMultiviewImage')
    assert 'ori_img' not in ds.return_keys
    # ResizeCropFlipImage casts to uint8 before creating the PIL image. Loading
    # JPEG as uint8 directly removes the exact uint8 -> float32 -> uint8 roundtrip.
    # Its output and NormalizeMultiviewImage still use original FP32 operations.
    for transform in ds.pipeline:
        if type(transform).__name__ in ('LoadMultiViewImageFromFiles','LoadMultiViewImageHistory','LoadMultiViewImageFuture'):
            transform.to_float32=False
    return ds


def install_decoded_image_cache(capacity=132):
    import mmcv
    original=mmcv.imread;cache=OrderedDict()
    def read(filename,*args,**kwargs):
        # Array inputs retain the original API, with no cache or type change.
        if not isinstance(filename,(str,bytes)):return original(filename,*args,**kwargs)
        key=(filename,args,tuple(sorted(kwargs.items())))
        if key in cache:
            cache.move_to_end(key);return cache[key].copy()
        image=original(filename,*args,**kwargs)
        if image is not None:
            assert image.dtype==np.uint8,'Only lossless uint8 camera decoding is authorized for this run'
            cache[key]=image.copy()
            while len(cache)>capacity:cache.popitem(last=False)
        return image
    mmcv.imread=read
    return cache
