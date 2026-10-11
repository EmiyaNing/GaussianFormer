"""Sequential, memory-bounded OPUSv1 teachers for all six horizons."""

import hashlib
import os

import torch
from mmengine import Config
from mmseg.models import build_segmentor

from misc.checkpoint_util import select_model_state_dict


def file_sha256(filename, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(filename, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


class MultiHorizonFrozenOPUSV1Teacher:
    """Run one frozen OPUS model at t+1 ... t+6 without retaining graphs.

    The loader layout is ``[t, t-1, ..., t-6, t+1, ..., t+6]``.  For every
    horizon, the selected eight-frame window and every projection matrix are
    expressed in that horizon's native future-ego coordinate system.
    """

    def __init__(self, config, checkpoint, device, history_frames=6,
                 num_horizons=6, num_cams=6, prefer_ema=True,
                 expected_sha256=None, amp=False):
        for filename in (config, checkpoint):
            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
        self.config_path = config
        self.checkpoint_path = checkpoint
        self.checkpoint_sha256 = file_sha256(checkpoint)
        if (expected_sha256 and self.checkpoint_sha256 !=
                expected_sha256.lower()):
            raise RuntimeError(
                'teacher checkpoint SHA256 mismatch: expected '
                f'{expected_sha256}, got {self.checkpoint_sha256}')
        self.device = torch.device(device)
        self.history_frames = int(history_frames)
        self.num_horizons = int(num_horizons)
        self.num_cams = int(num_cams)
        self.amp = bool(amp)

        import model  # noqa: F401
        cfg = Config.fromfile(config)
        self.model = build_segmentor(cfg.model)
        checkpoint_data = torch.load(checkpoint, map_location='cpu')
        state, self.state_key = select_model_state_dict(
            checkpoint_data, prefer_ema=prefer_ema)
        incompat = self.model.load_state_dict(state, strict=True)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError(f'teacher strict load failed: {incompat}')
        self.model.requires_grad_(False)
        self.model.eval().to(self.device)

    def _frame_index(self, relative_index):
        """Map a relative keyframe offset to the packed image frame index."""
        relative_index = int(relative_index)
        if relative_index == 0:
            return 0
        if relative_index < 0:
            return -relative_index
        return 1 + self.history_frames + relative_index - 1

    def teacher_frame_indices(self, horizon):
        horizon = int(horizon)
        if not 1 <= horizon <= self.num_horizons:
            raise ValueError(f'invalid teacher horizon: {horizon}')
        return [self._frame_index(relative) for relative in range(
            horizon, horizon - 8, -1)]

    def _select_metas(self, metas, horizon, frame_indices, camera_indices):
        selected = dict(metas)
        projections = metas['projection_mat'][:, camera_indices].clone().float()
        transforms = metas.get('future_ego_to_current')
        if transforms is None or transforms.shape[1] < horizon:
            raise ValueError(
                'future_ego_to_current does not contain the teacher target')
        transforms = transforms.float()
        current_from_target = transforms[:, horizon - 1]

        for slot, frame_index in enumerate(frame_indices):
            relative = self._relative_offset(frame_index)
            if relative <= 0:
                # Current/history projection: image <- current ego.
                frame_from_target = current_from_target
            else:
                # Future projection: image <- its own native future ego.
                current_from_frame = transforms[:, relative - 1]
                frame_from_target = (
                    torch.linalg.inv(current_from_frame) @
                    current_from_target)
            begin, end = slot * self.num_cams, (slot + 1) * self.num_cams
            projections[:, begin:end] = (
                projections[:, begin:end] @ frame_from_target[:, None])
        selected['projection_mat'] = projections
        selected['image_wh'] = metas['image_wh'][:, camera_indices]
        return selected

    def _relative_offset(self, frame_index):
        if frame_index == 0:
            return 0
        if frame_index <= self.history_frames:
            return -frame_index
        return frame_index - self.history_frames

    @torch.no_grad()
    def extract_horizon(self, imgs, metas, horizon):
        self.model.eval()
        frame_indices = self.teacher_frame_indices(horizon)
        total_frames = imgs.shape[1] // self.num_cams
        if imgs.shape[1] % self.num_cams or max(frame_indices) >= total_frames:
            raise ValueError(
                'training input is missing the required teacher frames')
        camera_indices = [camera for frame in frame_indices
                          for camera in range(frame * self.num_cams,
                                              (frame + 1) * self.num_cams)]
        teacher_imgs = imgs[:, camera_indices].contiguous()
        teacher_metas = self._select_metas(
            metas, horizon, frame_indices, camera_indices)
        with torch.cuda.amp.autocast(enabled=self.amp):
            representation = self.model(
                imgs=teacher_imgs, metas=teacher_metas, rep_only=True)
        final = representation[-1]
        pc_range = teacher_imgs.new_tensor(self.model.head.pc_range).float()
        points = final['query_points'].float()
        points_metric = points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
        point_logits = final['opus_logits'].float()
        result = dict(
            query_points_metric=points_metric.detach(),
            query_features=final['query_features'].float().detach(),
            point_logits=point_logits.detach(),
            semantic_logits=point_logits.mean(2).detach(),
            coordinate_system='native_future_ego',
            horizon_index=int(horizon))
        del representation, final, teacher_imgs, teacher_metas
        return result

    @torch.no_grad()
    def extract_all(self, imgs, metas):
        # Sequential execution bounds memory to one teacher forward at a time.
        return [self.extract_horizon(imgs, metas, horizon)
                for horizon in range(1, self.num_horizons + 1)]

    def manifest(self):
        return dict(
            config=self.config_path,
            checkpoint=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            state_key=self.state_key,
            history_frames=self.history_frames,
            num_horizons=self.num_horizons,
            teacher_windows=[self.teacher_frame_indices(horizon)
                             for horizon in range(1, self.num_horizons + 1)],
            input_layout='[t,t-1,...,t-6,t+1,...,t+6]',
            projection_coordinate='per_horizon_native_future_ego',
            output_coordinate='per_horizon_native_future_ego')
