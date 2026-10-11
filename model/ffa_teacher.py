"""Construction and memory-bounded execution of the frozen OPUSv1 teacher."""

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


class FrozenOPUSV1Teacher:
    """Own a non-registered teacher outside the student checkpoint/DDP graph."""

    def __init__(self, config, checkpoint, device, target_horizon_index=6,
                 num_student_frames=5, num_cams=6, prefer_ema=True,
                 expected_sha256=None, amp=False):
        if not os.path.isfile(config):
            raise FileNotFoundError(config)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        self.config_path = config
        self.checkpoint_path = checkpoint
        self.checkpoint_sha256 = file_sha256(checkpoint)
        if (expected_sha256 and
                self.checkpoint_sha256 != expected_sha256.lower()):
            raise RuntimeError(
                'teacher checkpoint SHA256 mismatch: expected '
                f'{expected_sha256}, got {self.checkpoint_sha256}')
        self.device = torch.device(device)
        self.target_horizon_index = int(target_horizon_index)
        self.num_student_frames = int(num_student_frames)
        self.num_cams = int(num_cams)
        self.amp = bool(amp)

        # Import registers all local OPUS model classes before construction.
        import model  # noqa: F401
        cfg = Config.fromfile(config)
        self.model = build_segmentor(cfg.model)
        checkpoint_data = torch.load(checkpoint, map_location='cpu')
        state, state_key = select_model_state_dict(
            checkpoint_data, prefer_ema=prefer_ema)
        incompat = self.model.load_state_dict(state, strict=True)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError(f'teacher strict load failed: {incompat}')
        self.state_key = state_key
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)

    def _frame_index(self, relative_index):
        if relative_index == 0:
            return 0
        if relative_index < 0:
            return -relative_index
        return self.num_student_frames + relative_index - 1

    @property
    def teacher_frame_indices(self):
        # OPUS history loader order is current, nearest history, ..., oldest.
        return [self._frame_index(relative) for relative in range(
            self.target_horizon_index,
            self.target_horizon_index - 8, -1)]

    def _select_metas(self, metas, frame_indices, flat_camera_indices):
        result = dict(metas)
        projections = metas['projection_mat'][:, flat_camera_indices].clone()
        transforms = metas.get('future_ego_to_current')
        if transforms is None:
            raise KeyError(
                'future_ego_to_current is required to rebase teacher views')
        target_step = self.target_horizon_index - 1
        if transforms.shape[1] <= target_step:
            raise ValueError('future ego transforms do not contain teacher target')

        # Every OPUS view must project points expressed in one common frame:
        # the teacher's current (target-future) ego frame. History matrices
        # map current-student ego to image, whereas future matrices are native
        # to their own future ego frame, so they need different right factors.
        current_from_target = transforms[:, target_step].float()
        for slot, frame in enumerate(frame_indices):
            begin = slot * self.num_cams
            end = begin + self.num_cams
            if frame < self.num_student_frames:
                frame_from_target = current_from_target
            else:
                future_horizon = frame - self.num_student_frames + 1
                current_from_frame = transforms[:, future_horizon - 1].float()
                frame_from_target = (
                    torch.linalg.inv(current_from_frame) @
                    current_from_target)
            projections[:, begin:end] = (
                projections[:, begin:end].float() @
                frame_from_target[:, None])
        result['projection_mat'] = projections
        result['image_wh'] = metas['image_wh'][:, flat_camera_indices]
        return result

    @torch.no_grad()
    def extract(self, imgs, metas):
        self.model.eval()
        frame_indices = self.teacher_frame_indices
        if min(frame_indices) < 0:
            raise ValueError('teacher window extends before available history')
        if max(frame_indices) >= imgs.shape[1] // self.num_cams:
            raise ValueError('training input is missing teacher future frames')
        camera_indices = []
        for frame in frame_indices:
            camera_indices.extend(range(
                frame * self.num_cams, (frame + 1) * self.num_cams))
        teacher_imgs = imgs[:, camera_indices].contiguous()
        teacher_metas = self._select_metas(
            metas, frame_indices, camera_indices)
        with torch.cuda.amp.autocast(enabled=self.amp):
            representation = self.model(
                imgs=teacher_imgs, metas=teacher_metas, rep_only=True)
        final = representation[-1]
        pc_range = teacher_imgs.new_tensor(self.model.head.pc_range).float()
        points = final['query_points'].float()
        points_metric = points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
        point_logits = final['opus_logits'].float()
        state = dict(
            points_metric=points_metric.detach(),
            query_features=final['query_features'].detach(),
            point_logits=point_logits.detach(),
            semantic_logits=point_logits.mean(2).detach())
        del representation, final, teacher_imgs
        return state

    def manifest(self):
        return dict(
            config=self.config_path,
            checkpoint=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            state_key=self.state_key,
            teacher_frame_indices=self.teacher_frame_indices,
            projection_coordinate='target_future_ego',
            output_coordinate='target_future_ego',
            target_horizon_index=self.target_horizon_index)
