"""Optional image-only execution changes; checkpoint parameter names are stable."""
import torch
from torch import nn
from torch.autograd.function import once_differentiable


class _FixedAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias):
        from ..encoder.gaussian_encoder.ops import deformable_aggregation_ext as ext
        if not hasattr(ext, 'image_affine_forward'):
            raise RuntimeError('Rebuild the DAF extension for image_frozen_bn=True')
        channels_last = x.is_contiguous(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        ctx.channels_last = channels_last
        ctx.save_for_backward(scale)
        return ext.image_affine_forward(x, scale, bias, channels_last)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        from ..encoder.gaussian_encoder.ops import deformable_aggregation_ext as ext
        scale, = ctx.saved_tensors
        grad = grad.contiguous(memory_format=torch.channels_last if ctx.channels_last else torch.contiguous_format)
        return ext.image_affine_backward(grad, scale, ctx.channels_last), None, None


class FixedImageBatchNorm2d(nn.BatchNorm2d):
    """Fixed-stat BN affine in one kernel, including BF16 input/output.

    Subclass BatchNorm2d so ResNet's norm_eval/frozen_stages still apply.
    Retain all original state keys. Derived affine buffers are nonpersistent.
    """
    def __init__(self, source):
        if (not source.track_running_stats or
                any(p.requires_grad for p in source.parameters())):
            raise ValueError('image_frozen_bn requires fixed statistics and affine parameters')
        super().__init__(source.num_features, source.eps, source.momentum,
                         source.affine, source.track_running_stats)
        self.load_state_dict(source.state_dict())
        self.requires_grad_(False)
        self.register_buffer('_scale', None, persistent=False)
        self.register_buffer('_bias', None, persistent=False)
        self._versions = None

    def forward(self, x):
        if self.training:
            raise RuntimeError('FixedImageBatchNorm2d requires image backbone norm_eval=True')
        tensors = [self.running_mean, self.running_var]
        if self.affine:
            tensors += [self.weight, self.bias]
        versions = tuple((t.data_ptr(), t._version, t.dtype) for t in tensors)
        if self._versions != versions or self._scale is None:
            # A cache first populated in inference_mode must also be usable
            # by a later training forward (autograd saves the fixed scale).
            with torch.inference_mode(False), torch.no_grad():
                scale = (self.running_var.float() + self.eps).rsqrt()
                if self.affine:
                    scale = scale * self.weight.float()
                bias = -self.running_mean.float() * scale
                if self.affine:
                    bias = bias + self.bias.float()
                self._scale, self._bias = scale.contiguous(), bias.contiguous()
            self._versions = versions
        return _FixedAffine.apply(x, self._scale, self._bias)


def convert_image_batch_norm(module):
    if not getattr(module, 'norm_eval', False):
        raise ValueError('image_frozen_bn requires norm_eval=True')
    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d) and not isinstance(child, FixedImageBatchNorm2d):
                converted = FixedImageBatchNorm2d(child)
                converted.train(child.training)
                setattr(parent, name, converted)
            else:
                replace(child)
    replace(module)


def convert_sync_batchnorm_preserving_fixed(module, process_group=None):
    """Retain fixed image affines when the training runner enables SyncBN.

    PyTorch's converter replaces every _BatchNorm subclass, including fixed
    image affines. Other BN modules keep the standard SyncBN conversion.
    """
    if isinstance(module, FixedImageBatchNorm2d):
        return module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    for name, child in list(module.named_children()):
        module.add_module(name, convert_sync_batchnorm_preserving_fixed(child, process_group))
    return module
