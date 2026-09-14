"""Stage-one future sparse latent query prediction for OPUSv2."""
import os

import torch
import torch.nn.functional as F
from mmseg.models import SEGMENTORS

from model.sparse_latent_dynamics import (
    HungarianSparseMatcher, SharedSparseLatentPredictor)
from .base_segmentor import CustomBaseSegmentor


@SEGMENTORS.register_module()
class SparseLatentWorld(CustomBaseSegmentor):
    """Frozen OPUS teacher/decoder plus a shared sparse-state predictor."""
    def __init__(self, num_history=5, num_future=6, num_cams=6,
                 predictor=None, matcher=None,
                 current_occupancy_checkpoint=None, **kwargs):
        super().__init__(**kwargs)
        self.num_history = int(num_history)
        self.num_future = int(num_future)
        self.num_cams = int(num_cams)
        self.current_occupancy_checkpoint = current_occupancy_checkpoint
        self._checkpoint_loaded = False
        predictor = dict(predictor or {})
        predictor.setdefault('num_horizons', self.num_future)
        self.future_predictor = SharedSparseLatentPredictor(**predictor)
        self.matcher = HungarianSparseMatcher(**dict(matcher or {}))
        self._freeze_teacher()

    @staticmethod
    def _adapt_mixing_parameter(value, target, module):
        groups = module.n_groups
        matrix_size = module.m_parameters
        out_points = module.out_points
        source_per_group = value.shape[0] // groups
        target_per_group = target.shape[0] // groups
        source_points = (source_per_group - matrix_size) // out_points
        target_points = (target_per_group - matrix_size) // out_points
        if (source_points % target_points or
                source_points * out_points + matrix_size != source_per_group or
                target_points * out_points + matrix_size != target_per_group):
            raise RuntimeError('incompatible OPUS temporal mixing parameter')
        tail = value.shape[1:]
        value = value.reshape(groups, source_per_group, *tail)
        channel = value[:, :matrix_size]
        spatial = value[:, matrix_size:].reshape(
            groups, out_points, source_points // target_points,
            target_points, *tail).sum(2)
        return torch.cat([channel, spatial.flatten(1, 2)], 1).reshape_as(target)

    def init_weights(self):
        super().init_weights()
        filename = self.current_occupancy_checkpoint
        if not filename or self._checkpoint_loaded:
            return
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        checkpoint = torch.load(filename, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        own = self.state_dict()
        scoped = {key: value for key, value in state.items()
                  if key.startswith(('img_backbone.', 'img_neck.', 'head.'))}
        expected = {key for key in own
                    if key.startswith(('img_backbone.', 'img_neck.', 'head.'))}
        for index, layer in enumerate(
                self.head.transformer.decoder.decoder_layers):
            prefix = (f'head.transformer.decoder.decoder_layers.{index}.'
                      'mixing.parameter_generator.')
            for suffix in ('weight', 'bias'):
                key = prefix + suffix
                if scoped[key].shape != own[key].shape:
                    scoped[key] = self._adapt_mixing_parameter(
                        scoped[key], own[key], layer.mixing)
        missing, unexpected = expected - set(scoped), set(scoped) - expected
        bad = [key for key, value in scoped.items()
               if key not in own or value.shape != own[key].shape]
        if missing or unexpected or bad:
            raise RuntimeError(
                f'teacher checkpoint mismatch: missing={sorted(missing)}, '
                f'unexpected={sorted(unexpected)}, bad_shapes={bad}')
        result = self.load_state_dict(scoped, strict=False)
        if result.unexpected_keys or any(
                key in expected for key in result.missing_keys):
            raise RuntimeError(f'teacher checkpoint load failed: {result}')
        self._checkpoint_loaded = True
        self._freeze_teacher()

    def _freeze_teacher(self):
        for module in (self.img_backbone, self.img_neck, self.head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_teacher()
        return self

    @property
    def pc_range(self):
        return self.head.pc_range

    def _decode_points(self, points):
        value = self.pc_range.to(points)
        return points * (value[3:] - value[:3]) + value[:3]

    def _encode_points(self, points):
        value = self.pc_range.to(points)
        return (points - value[:3]) / (value[3:] - value[:3])

    @staticmethod
    def _frame_metas(metas, camera_slice):
        result = dict(metas)
        result['projection_mat'] = metas['projection_mat'][:, camera_slice]
        result['image_wh'] = metas['image_wh'][:, camera_slice]
        return result

    def _encode_frame(self, images, metas):
        with torch.no_grad():
            features = self.extract_img_feat(images)['ms_img_feats']
            output = self.head(ms_img_feats=features, metas=metas)
        return output

    def _state(self, output):
        return dict(
            points=self._detach(output['all_refine_pts'][-1]),
            points_metric=self._detach(self._decode_points(
                output['all_refine_pts'][-1])),
            query_feats=self._detach(output['all_query_feats'][-1]),
            point_feats=self._detach(output['all_point_feats'][-1]),
            semantic_logits=self._detach(output['query_semantic_logits']))

    @staticmethod
    def _detach(value):
        return value.detach()

    def _alignment(self, predicted, target):
        matches = self.matcher(predicted, target)
        reference = predicted['points_metric']
        position = reference.sum() * 0.
        feature = position
        semantic = position
        existence = position
        center_error = []
        class_agreement = []
        matched_count = 0
        for batch_index, (pred_index, target_index) in enumerate(zip(
                matches.prediction, matches.target)):
            existence_target = torch.zeros_like(
                predicted['existence_logits'][batch_index])
            existence_target[pred_index] = 1.
            existence = existence + F.binary_cross_entropy_with_logits(
                predicted['existence_logits'][batch_index], existence_target)
            if pred_index.numel() == 0:
                continue
            pred_points = predicted['points_metric'][batch_index, pred_index]
            target_points = target['points_metric'][batch_index, target_index]
            pred_center, target_center = pred_points.mean(1), target_points.mean(1)
            center_term = F.smooth_l1_loss(
                pred_center, target_center, beta=.5)
            pair_distance = torch.cdist(pred_points.float(), target_points.float())
            chamfer = .5 * (pair_distance.min(-1).values.mean() +
                            pair_distance.min(-2).values.mean())
            position = position + center_term + chamfer
            pred_query = predicted['query_feats'][batch_index, pred_index]
            target_query = target['query_feats'][batch_index, target_index]
            query_loss = F.smooth_l1_loss(pred_query, target_query)
            query_loss = query_loss + (1. - F.cosine_similarity(
                pred_query.float(), target_query.float(), dim=-1)).mean()
            point_loss = F.smooth_l1_loss(
                predicted['point_feats'][batch_index, pred_index],
                target['point_feats'][batch_index, target_index])
            feature = feature + query_loss + point_loss
            teacher_probability = target['semantic_logits'][
                batch_index, target_index].float().softmax(-1)
            semantic = semantic + F.kl_div(
                predicted['semantic_logits'][batch_index, pred_index].float(
                ).log_softmax(-1), teacher_probability,
                reduction='batchmean')
            center_error.append((pred_center - target_center).norm(dim=-1).mean())
            class_agreement.append((
                predicted['semantic_logits'][batch_index, pred_index].argmax(-1) ==
                target['semantic_logits'][batch_index, target_index].argmax(-1)
            ).float().mean())
            matched_count += int(pred_index.numel())
        batch = max(reference.shape[0], 1)
        metrics = dict(
            matched_queries=reference.new_tensor(matched_count / batch),
            unmatched_prediction_ratio=reference.new_tensor(
                1. - matched_count / (batch * reference.shape[1])),
            center_l2_m=(torch.stack(center_error).mean() if center_error
                         else reference.new_tensor(float('nan'))),
            class_agreement=(torch.stack(class_agreement).mean()
                             if class_agreement else reference.new_tensor(float('nan'))))
        return dict(position=position / batch, feature=feature / batch,
                    semantic=semantic / batch, existence=existence / batch,
                    metrics=metrics)

    def _predicted_output(self, predicted):
        normalized = self._encode_points(predicted['points_metric']).clamp(0., 1.)
        return self.head.decode_sparse_state(
            normalized, predicted['point_feats'], predicted['query_feats'])

    def forward(self, imgs, metas, global_iter=0,
                oracle_future_latent=False, **kwargs):
        observed_frames = self.num_history + 1
        frame_images = imgs.reshape(
            imgs.shape[0], -1, self.num_cams, *imgs.shape[2:])
        if frame_images.shape[1] < observed_frames:
            raise ValueError('SparseLatentWorld input is missing observed frames')

        # Loader order is current, nearest history -> farthest history.
        observed_states, current_output = [], None
        for frame_index in range(observed_frames - 1, -1, -1):
            start = frame_index * self.num_cams
            output = self._encode_frame(
                frame_images[:, frame_index],
                self._frame_metas(metas, slice(start, start + self.num_cams)))
            observed_states.append(self._state(output))
            if frame_index == 0:
                current_output = output

        selected_horizon = None
        if 'future_horizon' in metas:
            values = torch.as_tensor(
                metas['future_horizon'], device=imgs.device).reshape(-1)
            if torch.unique(values).numel() != 1:
                raise ValueError('one batch must select one common future horizon')
            selected_horizon = int(values[0])
            horizons = [selected_horizon]
        else:
            horizons = list(range(min(
                self.num_future, frame_images.shape[1] - observed_frames)))
        if self.training and not horizons:
            raise ValueError('training requires at least one future teacher frame')

        predictions = []
        for local_index, horizon in enumerate(horizons):
            future_frame = observed_frames + local_index
            projection_start = future_frame * self.num_cams
            teacher_output = self._encode_frame(
                frame_images[:, future_frame], self._frame_metas(
                    metas, slice(projection_start,
                                 projection_start + self.num_cams)))
            target_state = self._state(teacher_output)
            transform_index = 0 if selected_horizon is not None else horizon
            future_to_current = metas['future_ego_to_current'][:, transform_index]
            current_to_future = torch.linalg.inv(future_to_current.float())
            predicted_state = self.future_predictor(
                observed_states, current_to_future, horizon)
            predicted_output = self._predicted_output(predicted_state)
            alignment = self._alignment(predicted_state, target_state)
            predictions.append(dict(
                horizon=horizon,
                opus=(teacher_output if oracle_future_latent
                      else predicted_output),
                predicted_opus=predicted_output,
                teacher_opus=teacher_output,
                predicted_state=predicted_state,
                teacher_state=target_state,
                alignment=alignment))

        stage1_output = dict(
            current_opus=current_output,
            future_predictions=predictions,
            future_prediction_coordinate='native_future')
        result = dict(stage1_output=stage1_output)
        result.update(current_output)
        return result
