"""Independent entry point for the four full Occ3D temporal Gaussian runs.

Reuse the established loss/evaluation runner; scale warmup and cosine floors
per parameter group so backbone/sampling/bias learning-rate multipliers hold
throughout training. The original train.py and previous runs are unchanged.
"""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from timm.scheduler import CosineLRScheduler
import train


class GroupRatioCosineScheduler(CosineLRScheduler):
    def __init__(self, optimizer, t_initial, lr_min, warmup_t,
                 warmup_lr_init, t_in_epochs=False, **kwargs):
        base_lr = 2e-4
        self.group_warmup_ratio = warmup_lr_init / base_lr
        self.group_min_ratio = lr_min / base_lr
        super().__init__(optimizer, t_initial=t_initial, lr_min=lr_min,
                         warmup_t=warmup_t, warmup_lr_init=warmup_lr_init,
                         t_in_epochs=t_in_epochs, **kwargs)
        self.update_groups([v * self.group_warmup_ratio for v in self.base_values])

    def _get_lr(self, t):
        if t < self.warmup_t:
            ratio = self.group_warmup_ratio + (1 - self.group_warmup_ratio) * t / self.warmup_t
        else:
            fraction = min(t / self.t_initial, 1.0)
            ratio = self.group_min_ratio + (1 - self.group_min_ratio) * (
                1 + math.cos(math.pi * fraction)) / 2
        return [value * ratio for value in self.base_values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume-from', default='')
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--seed', type=int, default=928)
    args = parser.parse_args()
    args.dataset = 'nuscenes'
    args.iter_resume = False
    args.gpus = torch.cuda.device_count()
    if args.gpus != 1:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to the single training GPU')
    train.CosineLRScheduler = GroupRatioCosineScheduler
    original_build = train.build_segmentor
    networks = []

    def build(*a, **kw):
        network = original_build(*a, **kw)
        networks.append(network)
        return network

    train.build_segmentor = build
    try:
        train.main(0, args)
    finally:
        for network in networks:
            network.flush_memory_phase()


if __name__ == '__main__':
    main()
