"""Controlled teacher-query substitution without changing SparseWorld rollout."""

import torch


ARMS = ('task_only', 'teacher_feature', 'teacher_position',
        'teacher_joint', 'teacher_joint_shuffled_feature')


def inject_teacher_queries(state, teacher, arm, pc_range):
    """Replace only matched persistent slots, retaining all future-born slots.

    ``teacher['points_for_student']`` is already locally reordered to each
    Student point index. Both point sets are in the target future ego frame.
    The shuffled control keeps correct teacher geometry but permutes teacher
    features across matched queries, preserving the feature distribution.
    """
    if arm not in ARMS:
        raise ValueError(f'Unknown teacher intervention: {arm}')
    query = state['query'].clone()
    points = state['normalized_points'].clone()
    rows = teacher['student_rows'].to(query.device).long()
    indices = teacher['teacher_rows'].to(query.device).long()
    born = state['born_mask'].to(query.device).bool()
    if born.ndim != 1 or born.shape[0] != query.shape[1]:
        raise ValueError('Invalid future-born mask')
    if rows.numel() and (bool(born[rows].any()) or bool((rows >= 720).any())):
        raise ValueError('Teacher replacement would touch a future-born query')
    if rows.numel() != indices.numel() or rows.unique().numel() != rows.numel():
        raise ValueError('Invalid one-to-one Student matching')
    if arm == 'task_only' or not rows.numel():
        return query, points
    if arm in ('teacher_feature', 'teacher_joint',
               'teacher_joint_shuffled_feature'):
        features = teacher['query_features'][0, indices].to(query)
        if arm == 'teacher_joint_shuffled_feature':
            features = features.roll(1, 0)
        query[0, rows] = features
    if arm in ('teacher_position', 'teacher_joint',
               'teacher_joint_shuffled_feature'):
        metric = teacher['points_for_student'].to(points)
        lower = points.new_tensor(pc_range[:3])
        extent = points.new_tensor(pc_range[3:]) - lower
        points[0, rows] = (metric - lower) / extent
    if not torch.equal(query[:, born], state['query'][:, born].to(query)):
        raise RuntimeError('Future-born features were altered')
    if not torch.equal(points[:, born], state['normalized_points'][:, born].to(points)):
        raise RuntimeError('Future-born geometry was altered')
    return query, points
