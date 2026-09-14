from copy import deepcopy

import torch


class ModelEMA:
    """Maintain an exponential moving average copy of a model.

    The shadow model is kept in eval mode and is intentionally not registered
    with the optimizer. Floating-point parameters are averaged after each
    successful optimizer update; buffers are copied so that BatchNorm and
    other running state remain coherent with the latest training model.
    """

    def __init__(self, model, decay=0.9998, update_buffers=True):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f'EMA decay must be in [0, 1), got {decay}')
        self.module = deepcopy(model).eval()
        self.decay = float(decay)
        self.update_buffers = bool(update_buffers)
        self.num_updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_parameters = dict(self.module.named_parameters())
        model_parameters = dict(model.named_parameters())
        if ema_parameters.keys() != model_parameters.keys():
            raise RuntimeError('EMA and training model parameters do not match')

        one_minus_decay = 1.0 - self.decay
        for name, ema_parameter in ema_parameters.items():
            source = model_parameters[name].detach()
            if ema_parameter.is_floating_point() or ema_parameter.is_complex():
                ema_parameter.mul_(self.decay).add_(
                    source, alpha=one_minus_decay)
            else:
                ema_parameter.copy_(source)

        if self.update_buffers:
            ema_buffers = dict(self.module.named_buffers())
            model_buffers = dict(model.named_buffers())
            if ema_buffers.keys() != model_buffers.keys():
                raise RuntimeError('EMA and training model buffers do not match')
            for name, ema_buffer in ema_buffers.items():
                ema_buffer.copy_(model_buffers[name].detach())

        self.num_updates += 1

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)
