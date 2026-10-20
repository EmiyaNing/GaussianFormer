"""Regression coverage for BF16 backbone features entering an FP32 FPN."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from mmseg.models.necks import FPN

from model.segmentor.opus_temporal_segmentor import OPUSTemporalSegmentor


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine = nn.Conv2d(3, 4, 3, padding=1)
        self.coarse = nn.Conv2d(4, 8, 3, stride=2, padding=1)

    def forward(self, images):
        fine = self.fine(images)
        return {'fine': fine, 'coarse': self.coarse(fine)}


def _make_segmentor(device, image_dtype='bfloat16', checkpoint=True,
                    freeze_backbone=False, freeze_neck=False):
    # Exercise the real temporal extraction methods without constructing
    # unrelated decoder/renderer CUDA extensions or loading dataset assets.
    model = OPUSTemporalSegmentor.__new__(OPUSTemporalSegmentor)
    nn.Module.__init__(model)
    model.img_backbone = _Backbone()
    model.img_neck = FPN(in_channels=[4, 8], out_channels=4, num_outs=2)
    model.img_backbone_out_indices = [0, 1]
    model.image_amp_dtype = image_dtype
    model.image_chunk_size = 2
    model.image_checkpoint = checkpoint
    model.freeze_img_backbone = freeze_backbone
    model.freeze_img_neck = freeze_neck
    model.img_backbone.requires_grad_(not freeze_backbone)
    model.img_neck.requires_grad_(not freeze_neck)
    return model.to(device)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            pytest.skip('CUDA BF16 is unavailable')
    return torch.device(request.param)


@pytest.fixture
def reject_bf16_resize(monkeypatch):
    # Emulate the user's old CUDA kernel on newer PyTorch too, so this test
    # fails if BF16 reaches FPN interpolation even where BF16 is supported.
    original = F.interpolate
    calls = []

    def checked_interpolate(inputs, *args, **kwargs):
        if inputs.dtype == torch.bfloat16:
            raise RuntimeError(
                '"upsample_nearest2d_out_frame" not implemented for \'BFloat16\'')
        calls.append(inputs.dtype)
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(F, 'interpolate', checked_interpolate)
    return calls


@pytest.mark.parametrize('image_dtype', ['bfloat16', 'float32'])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('training', [False, True])
def test_temporal_fpn_precision_and_history_gradients(
        device, reject_bf16_resize, image_dtype, checkpoint, training):
    model = _make_segmentor(device, image_dtype, checkpoint)
    model.train(training)
    backbone_dtypes, neck_dtypes = [], []
    model.img_backbone.register_forward_hook(
        lambda module, args, output: backbone_dtypes.extend(
            feature.dtype for feature in output.values()))
    model.img_neck.register_forward_pre_hook(
        lambda module, args: neck_dtypes.extend(
            feature.dtype for feature in args[0]))
    images = torch.randn(1, 5, 3, 8, 8, device=device, requires_grad=True)

    # A runner's outer autocast must not recast the neck's lateral convolutions.
    with torch.set_grad_enabled(training):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = model.extract_img_feat(images)['ms_img_feats']
    assert [tuple(feature.shape) for feature in features] == [
        (1, 5, 4, 8, 8), (1, 5, 4, 4, 4)]
    assert all(feature.dtype == torch.float32 for feature in features)
    expected = torch.bfloat16 if image_dtype == 'bfloat16' else torch.float32
    assert backbone_dtypes and set(backbone_dtypes) == {expected}
    assert neck_dtypes and set(neck_dtypes) == {torch.float32}
    assert reject_bf16_resize and set(reject_bf16_resize) == {torch.float32}

    if training:
        sum(feature.square().mean() for feature in features).backward()
        assert torch.isfinite(images.grad).all()
        # Each camera/history image, including the final partial chunk, trains.
        assert torch.all(images.grad.flatten(2).abs().sum(dim=-1) > 0)
        for module in (model.img_backbone, model.img_neck):
            assert all(parameter.grad is not None and
                       torch.isfinite(parameter.grad).all()
                       for parameter in module.parameters())
    else:
        assert all(not feature.requires_grad for feature in features)

    # Checkpointed extraction agrees with eager extraction at the same precision.
    with torch.no_grad():
        reference = model.extract_img_feat(images)['ms_img_feats']
    for actual, expected_feature in zip(features, reference):
        torch.testing.assert_close(actual, expected_feature)


@pytest.mark.parametrize('freeze_backbone', [False, True])
@pytest.mark.parametrize('freeze_neck', [False, True])
def test_temporal_fp32_neck_preserves_freeze_contract(
        freeze_backbone, freeze_neck, reject_bf16_resize):
    model = _make_segmentor(torch.device('cpu'),
                            freeze_backbone=freeze_backbone,
                            freeze_neck=freeze_neck)
    model.train()
    images = torch.randn(1, 3, 3, 8, 8, requires_grad=True)
    features = model.extract_img_feat(images)['ms_img_feats']
    readout = nn.Conv2d(4, 1, 1)
    sum(readout(feature.flatten(0, 1)).square().mean()
        for feature in features).backward()
    # A frozen neck runs under no_grad, just as it did before this fix.
    assert all((parameter.grad is not None) == (not freeze_neck)
               for parameter in model.img_neck.parameters())
    assert all((parameter.grad is not None) ==
               (not freeze_backbone and not freeze_neck)
               for parameter in model.img_backbone.parameters())
