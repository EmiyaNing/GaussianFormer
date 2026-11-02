"""Temporal CUDA sampling with packed dropout and selective frame gradients."""

import torch
from torch.autograd.function import once_differentiable

from . import extension


def packed_dropout(reference, count, probability):
    return extension().packed_dropout(reference, count, probability)


class _TemporalSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, projection, image_wh, camera_valid, shapes, starts,
                dropout, query, camera, *frames):
        if projection.requires_grad or image_wh.requires_grad:
            raise ValueError('Temporal sampling expects fixed camera metadata')
        values = (points.contiguous(), projection.contiguous(), image_wh.contiguous(),
                  camera_valid.contiguous(), shapes.contiguous(), starts.contiguous(),
                  dropout.contiguous(), query.contiguous(), camera.contiguous())
        frames = tuple(frame.contiguous() for frame in frames)
        output, statistics = extension().temporal_sampling_forward(*values, list(frames))
        ctx.save_for_backward(*values, output, statistics, *frames)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        points, projection, image_wh, valid, shapes, starts, dropout, query, camera, output, statistics, *frames = ctx.saved_tensors
        needs = ctx.needs_input_grad
        result = extension().temporal_sampling_backward(
            points, projection, image_wh, valid, shapes, starts, dropout, query, camera,
            frames, output, statistics, gradient.contiguous(),
            needs[0], needs[7], needs[8], list(needs[9:]))
        dpoints, dquery, dcamera, *dframes = result
        return (dpoints if needs[0] else None, None, None, None, None, None, None,
                dquery if needs[7] else None, dcamera if needs[8] else None,
                *(grad if needed else None for grad, needed in zip(dframes, needs[9:])))


def temporal_sampling(points, projection, image_wh, camera_valid, shapes, starts,
                      query, camera, frames, dropout_probability=0.):
    dropout = packed_dropout(query, query.numel() * camera.shape[1], dropout_probability)
    return _TemporalSampling.apply(points, projection, image_wh, camera_valid, shapes,
                                   starts, dropout, query, camera, *frames)
