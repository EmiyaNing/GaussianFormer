from torch.optim import Optimizer


class OneCycleLR:
    """ Sets the learing rate of each parameter group by the one cycle learning rate policy
    proposed in https://arxiv.org/pdf/1708.07120.pdf.

    It is recommended that you set the max_lr to be the learning rate that achieves
    the lowest loss in the learning rate range test, and set min_lr to be 1/10 th of max_lr.

    So, the learning rate changes like min_lr -> max_lr -> min_lr -> final_lr,
    where final_lr = min_lr * reduce_factor.

    Supports multiple parameter groups. Each group's learning rate is scaled relative
    to the base learning rate of the first group (param_groups[0]), preserving the
    relative ratios defined by paramwise_cfg or similar mechanisms.

    Args:
        optimizer:             (Optimizer) against which we apply this scheduler
        num_steps:             (int) of total number of steps/iterations
        lr_range:              (tuple) of min and max values of learning rate
        momentum_range:        (tuple) of min and max values of momentum
        annihilation_frac:     (float), fracion of steps to annihilate the learning rate
        reduce_factor:         (float), denotes the factor by which we annihilate the learning rate at the end
        last_step:             (int), denotes the last step. Set to -1 to start training from the beginning

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = OneCycleLR(optimizer, num_steps=num_steps, lr_range=(0.1, 1.))
        >>> for epoch in range(epochs):
        >>>     for step in train_dataloader:
        >>>         train(...)
        >>>         scheduler.step()

    Useful resources:
        https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6
        https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0
    """

    def __init__(self,
                 optimizer: Optimizer,
                 num_steps: int,
                 lr_range: tuple = (0.1, 1.),
                 momentum_range: tuple = (0.85, 0.95),
                 annihilation_frac: float = 0.1,
                 reduce_factor: float = 0.01,
                 last_step: int = -1):
        # Sanity check
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.num_steps = num_steps

        self.min_lr, self.max_lr = lr_range[0], lr_range[1]
        assert self.min_lr < self.max_lr, \
            "Argument lr_range must be (min_lr, max_lr), where min_lr < max_lr"

        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        assert self.min_momentum < self.max_momentum, \
            "Argument momentum_range must be (min_momentum, max_momentum), where min_momentum < max_momentum"

        self.num_cycle_steps = int(num_steps * (1. - annihilation_frac))  # Total number of steps in the cycle
        self.final_lr = self.min_lr * reduce_factor

        # Compute per-group lr scales relative to the first group's base lr.
        # This preserves the relative learning rate ratios across parameter groups
        # (e.g. from paramwise_cfg) while still applying the one-cycle schedule uniformly.
        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.lr_scales = []
        for group in self.optimizer.param_groups:
            if 'lr' in group:
                self.lr_scales.append(group['lr'] / self.base_lr)
            else:
                self.lr_scales.append(1.0)

        # Compute per-group momentum scales, same rationale as lr_scales.
        self.base_momentum = self.optimizer.param_groups[0].get('momentum', None)
        self.momentum_scales = []
        for group in self.optimizer.param_groups:
            if 'momentum' in group and self.base_momentum is not None and self.base_momentum != 0:
                self.momentum_scales.append(group['momentum'] / self.base_momentum)
            else:
                self.momentum_scales.append(1.0)

        self.last_step = last_step

        if self.last_step == -1:
            self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        """Returns the learning rate of the first parameter group.
        For all groups, use get_last_lr().
        """
        return self.optimizer.param_groups[0]['lr']

    def get_last_lr(self):
        """Returns a list of learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def get_momentum(self):
        """Returns the momentum of the first parameter group.
        For all groups, use get_last_momentum().
        """
        return self.optimizer.param_groups[0]['momentum']

    def get_last_momentum(self):
        """Returns a list of momentum values for all parameter groups."""
        return [group.get('momentum', None) for group in self.optimizer.param_groups]

    def _apply_to_all_groups(self, lr, momentum):
        """Apply the computed lr and momentum to all parameter groups,
        scaling by each group's lr_scale and momentum_scale respectively.
        """
        for i, group in enumerate(self.optimizer.param_groups):
            if 'lr' in group:
                group['lr'] = lr * self.lr_scales[i]
            if momentum is not None and 'momentum' in group:
                group['momentum'] = momentum * self.momentum_scales[i]

    def step(self):
        """Conducts one step of learning rate and momentum update
        """
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step <= self.num_cycle_steps // 2:
            # Scale up phase
            scale = current_step / (self.num_cycle_steps // 2)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            # Scale down phase
            scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_steps:
            # Annihilation phase: only change lr
            scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        else:
            # Exceeded given num_steps: do nothing
            return

        self._apply_to_all_groups(lr, momentum)

    def step_update(self, current_step):
        """Conducts one step of learning rate and momentum update
        """
        self.last_step = current_step

        if current_step <= self.num_cycle_steps // 2:
            # Scale up phase
            scale = current_step / (self.num_cycle_steps // 2)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            # Scale down phase
            scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_steps:
            # Annihilation phase: only change lr
            scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        else:
            # Exceeded given num_steps: do nothing
            return

        self._apply_to_all_groups(lr, momentum)