"""Isolated training controls; no production model or loss registration."""
import math
import torch
from torch import nn
from model.segmentor.sparseworld_predictability_finetune_segmentor import SparseWorldPredictabilityFinetuneSegmentor


class TrainingProtocolStudent(SparseWorldPredictabilityFinetuneSegmentor):
    """Same numerical forward, optionally restoring detached future points."""
    detach_future_geometry = False

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.detach_future_geometry:
            output['future_pred_points'] = [p.detach() for p in output['future_pred_points']]
        return output


def set_training_mode(instance, fixed_bn):
    # BN affine weights remain trainable. Only running statistics / use of
    # batch statistics changes; attention and other dropout stay in train mode.
    instance.train()
    if fixed_bn:
        for module in instance.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def schedule_factor(step, warmup, total):
    if step < warmup:
        return min((step + 1) / warmup, 1.)
    return .05 + .95 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def bn_snapshot(instance):
    return {name: value.detach().cpu().clone() for name, value in instance.state_dict().items()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def bn_unchanged(instance, original):
    return all(torch.equal(instance.state_dict()[name].cpu(), value) for name, value in original.items())
