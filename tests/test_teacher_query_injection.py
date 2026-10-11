import pytest
import torch

from model.teacher_query_injection import ARMS, inject_teacher_queries


def fixture():
    query = torch.arange(725 * 4, dtype=torch.float32).view(1, 725, 4)
    points = torch.full((1, 725, 2, 3), .5)
    born = torch.zeros(725, dtype=torch.bool)
    born[720:] = True
    state = dict(query=query, normalized_points=points, born_mask=born)
    teacher = dict(student_rows=torch.tensor([1, 3]),
                   teacher_rows=torch.tensor([0, 2]),
                   query_features=torch.tensor([[[11., 12., 13., 14.],
                                                 [21., 22., 23., 24.],
                                                 [31., 32., 33., 34.]]]),
                   points_for_student=torch.tensor([[[1., 2., 3.], [4., 5., 6.]],
                                                    [[2., 3., 4.], [5., 6., 7.]]]))
    return state, teacher


@pytest.mark.parametrize('arm', ARMS)
def test_only_matched_persistent_slots_can_be_substituted(arm):
    state, teacher = fixture()
    query, points = inject_teacher_queries(state, teacher, arm, (0., 0., 0., 10., 10., 10.))
    torch.testing.assert_close(query[:, 720:], state['query'][:, 720:])
    torch.testing.assert_close(points[:, 720:], state['normalized_points'][:, 720:])
    if arm == 'task_only':
        torch.testing.assert_close(query, state['query'])
        torch.testing.assert_close(points, state['normalized_points'])
    elif arm == 'teacher_feature':
        torch.testing.assert_close(points, state['normalized_points'])
        torch.testing.assert_close(query[0, 1], teacher['query_features'][0, 0])
    elif arm == 'teacher_position':
        torch.testing.assert_close(query, state['query'])
        torch.testing.assert_close(points[0, 3], teacher['points_for_student'][1] / 10)
    else:
        torch.testing.assert_close(points[0, 1], teacher['points_for_student'][0] / 10)
        index = 2 if arm.endswith('shuffled_feature') else 0
        torch.testing.assert_close(query[0, 1], teacher['query_features'][0, index])


def test_future_born_match_is_rejected():
    state, teacher = fixture()
    teacher['student_rows'] = torch.tensor([1, 720])
    with pytest.raises(ValueError, match='future-born'):
        inject_teacher_queries(state, teacher, 'teacher_joint', (0., 0., 0., 10., 10., 10.))
