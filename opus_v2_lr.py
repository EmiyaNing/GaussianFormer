"""Config-scoped LR scheduler matching the official OPUSv2 recipe."""
import math


class OPUSV2WarmupCosineLR:
    """Linear iteration warmup followed by cosine annealing.

    The multiplier is applied to every parameter group's own initial LR, so
    backbone/sampling_offset 0.1x groups preserve their ratio during warmup.
    """

    def __init__(self, optimizer, total_steps, warmup_iters=500,
                 warmup_ratio=1 / 3, min_lr_ratio=1e-3, last_step=-1):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.last_step = last_step
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        for group, lr in zip(optimizer.param_groups, self.base_lrs):
            group.setdefault('initial_lr', lr)
        # The runner advances schedulers after optimizer.step(); initialize
        # step zero here so the very first update already uses warmup LR.
        self.step_update(max(last_step + 1, 0))

    def _factor(self, step):
        if self.warmup_iters > 0 and step < self.warmup_iters:
            return self.warmup_ratio + (1. - self.warmup_ratio) * step / self.warmup_iters
        denominator = max(self.total_steps - self.warmup_iters, 1)
        progress = min(max((step - self.warmup_iters) / denominator, 0.), 1.)
        return self.min_lr_ratio + (1. - self.min_lr_ratio) * (
            1. + math.cos(math.pi * progress)) / 2.

    def step_update(self, step):
        self.last_step = step
        factor = self._factor(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base_lr * factor

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
