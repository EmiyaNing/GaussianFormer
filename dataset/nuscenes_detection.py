"""nuScenes detection targets and coordinate transforms for frozen readouts."""

import json
from pathlib import Path
import pickle
import random

import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from . import OPENOCC_DATASET
from .dataset import NuScenesDataset


@OPENOCC_DATASET.register_module()
class NuScenesDetectionDataset(NuScenesDataset):
    """Original inputs/occupancy with deterministic per-sample augmentation.

    Detection targets are joined by sample_idx outside feature extraction.
    """
    def __getitem__(self, index):
        if not hasattr(self, 'augmentation_seed'):
            return super().__getitem__(index)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        seed = (self.augmentation_seed+int(index)) % (2**32)
        np.random.seed(seed)
        random.seed(seed)
        try:
            return super().__getitem__(index)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


DETECTION_CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                     'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
CLASS_RANGES = dict(car=50., truck=50., construction_vehicle=50., bus=50., trailer=50.,
                    barrier=30., motorcycle=40., bicycle=40., pedestrian=40., traffic_cone=30.)


def annotation_velocity(annotation, annotations, sample_times):
    previous = annotations.get(annotation['prev'])
    following = annotations.get(annotation['next'])
    if previous is None and following is None:
        return np.full(3, np.nan)
    first = previous if previous is not None else annotation
    last = following if following is not None else annotation
    delta = (sample_times[last['sample_token']]-sample_times[first['sample_token']])/1e6
    limit = 3. if previous is not None and following is not None else 1.5
    if delta <= 0 or delta > limit:
        return np.full(3, np.nan)
    return (np.asarray(last['translation'])-np.asarray(first['translation']))/delta


def inside_racks(points, racks):
    keep = np.zeros(len(points), dtype=bool)
    for rack in racks:
        box = Box(rack[:3], rack[3:6], Quaternion(rack[6:10]))
        keep |= points_in_box(box, np.asarray(points).T)
    return keep


def scope_mask(record, pc_range, roi=True):
    centers = record['boxes'][:, :3]
    global_delta = record['global_boxes'][:, :3]-record['ego_translation']
    limits = np.asarray([CLASS_RANGES[DETECTION_CLASSES[int(label)]] for label in record['labels']])
    mask = np.linalg.norm(global_delta[:, :2], axis=-1) < limits
    if roi:
        region = np.asarray(pc_range)
        mask &= ((centers >= region[:3]) & (centers < region[3:])).all(-1)
    return mask


def read_keyframes(imageset):
    with open(imageset, 'rb') as stream:
        data = pickle.load(stream)
    keyframes = sorted(data['metadata'], key=lambda value: value[0]+str(value[1]).zfill(3))
    records = []
    for index, (scene, frame) in enumerate(keyframes):
        info = data['infos'][scene][frame]
        pose = info['data']['LIDAR_TOP']['pose']
        records.append(dict(index=index, sample_token=info['token'], scene_token=scene,
                            annotation_tokens=info['anns'], ego_translation=pose['translation'],
                            ego_rotation=pose['rotation']))
    return records


def prepare_detection_metadata(cfg, destination):
    """Build compact targets once; raw metadata is released before GPU work."""
    root = Path(cfg.data_root) / 'v1.0-trainval'
    with (root / 'category.json').open() as stream:
        categories = {value['token']: value['name'] for value in json.load(stream)}
    with (root / 'instance.json').open() as stream:
        instances = {value['token']: categories[value['category_token']] for value in json.load(stream)}
    with (root / 'attribute.json').open() as stream:
        attributes = {value['token']: value['name'] for value in json.load(stream)}
    with (root / 'sample.json').open() as stream:
        times = {value['token']: value['timestamp'] for value in json.load(stream)}
    with (root / 'sample_annotation.json').open() as stream:
        annotations = {value['token']: value for value in json.load(stream)}
    metadata = dict(classes=DETECTION_CLASSES, pc_range=list(cfg.pc_range), splits={})
    for split in ['train', 'val']:
        records = read_keyframes(cfg[split+'_dataset_config'].imageset)
        for record in records:
            ego = Quaternion(record['ego_rotation'])
            rotation = ego.rotation_matrix
            translation = np.asarray(record['ego_translation'], dtype=np.float64)
            boxes, global_boxes, labels, attrs, num_points, racks = [], [], [], [], [], []
            for token in record.pop('annotation_tokens'):
                ann = annotations[token]
                name = instances[ann['instance_token']]
                if name == 'static_object.bicycle_rack':
                    racks.append(ann['translation']+ann['size']+ann['rotation'])
                detection = category_to_detection_name(name)
                count = ann['num_lidar_pts']+ann['num_radar_pts']
                if detection is None or count <= 0:
                    continue
                position = (np.asarray(ann['translation'])-translation) @ rotation
                orientation = ego.inverse*Quaternion(ann['rotation'])
                yaw = np.arctan2(orientation.rotation_matrix[1, 0], orientation.rotation_matrix[0, 0])
                velocity = annotation_velocity(ann, annotations, times)
                velocity[2] = 0.  # nuScenes scores planar global velocity.
                local_velocity = velocity @ rotation
                boxes.append(np.r_[position, ann['size'], yaw, local_velocity[:2]])
                global_boxes.append(np.r_[ann['translation'], ann['size'], ann['rotation'], velocity[:2]])
                labels.append(DETECTION_CLASSES.index(detection))
                attrs.append(attributes[ann['attribute_tokens'][0]] if ann['attribute_tokens'] else '')
                num_points.append(count)
            record.update(boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 9),
                          global_boxes=np.asarray(global_boxes, dtype=np.float64).reshape(-1, 12),
                          labels=np.asarray(labels, dtype=np.int64), attributes=np.asarray(attrs),
                          num_points=np.asarray(num_points), racks=np.asarray(racks).reshape(-1, 10),
                          ego_translation=translation, ego_rotation=np.asarray(record['ego_rotation']))
            # Official evaluation ignores cycles whose centers lie in racks.
            cycles = np.isin(record['labels'], [6, 7])
            ignored = cycles & inside_racks(record['global_boxes'][:, :3], record['racks'])
            for key in ['boxes', 'global_boxes', 'labels', 'attributes', 'num_points']:
                record[key] = record[key][~ignored]
        metadata['splits'][split] = records
    assert set(r['scene_token'] for r in metadata['splits']['train']).isdisjoint(
        r['scene_token'] for r in metadata['splits']['val'])
    with open(destination, 'wb') as stream:
        pickle.dump(metadata, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return metadata


def prediction_attribute(name, velocity):
    moving = np.linalg.norm(velocity) > .2
    if name in ['car', 'truck', 'construction_vehicle', 'bus', 'trailer']:
        return 'vehicle.moving' if moving else ('vehicle.stopped' if name == 'bus' else 'vehicle.parked')
    if name in ['bicycle', 'motorcycle']:
        return 'cycle.with_rider' if moving else 'cycle.without_rider'
    if name == 'pedestrian':
        return 'pedestrian.moving' if moving else 'pedestrian.standing'
    return ''


def prediction_to_global(prediction, record):
    rotation = Quaternion(record['ego_rotation'])
    matrix = rotation.rotation_matrix
    result = []
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        name = DETECTION_CLASSES[int(label)]
        position = matrix @ box[:3]+record['ego_translation']
        velocity = matrix @ np.r_[box[7:9], 0.]
        quaternion = rotation*Quaternion(axis=[0, 0, 1], radians=float(box[6]))
        result.append(dict(sample_token=record['sample_token'], translation=position.tolist(),
                           size=box[3:6].astype(float).tolist(), rotation=quaternion.elements.tolist(),
                           velocity=velocity[:2].tolist(), detection_name=name,
                           detection_score=float(score), attribute_name=prediction_attribute(name, box[7:9])))
    return result
