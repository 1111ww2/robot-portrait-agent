import math

import numpy as np
import pytest

from leica_tracker.physical_direction_mapping import direction_in_base

SIGNS = {cue: 1 for cue in ('MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN',
                            'MOVE_CLOSER', 'MOVE_FARTHER')}


@pytest.mark.parametrize('cue,expected', [
    ('MOVE_LEFT', [1, 0, 0]), ('MOVE_RIGHT', [-1, 0, 0]),
    ('MOVE_UP', [0, -1, 0]), ('MOVE_DOWN', [0, 1, 0]),
    ('MOVE_CLOSER', [0, 0, -1]), ('MOVE_FARTHER', [0, 0, 1]),
])
@pytest.mark.parametrize('mirrored', [True, False])
def test_participant_cues_are_independent_of_preview_mirroring(cue, expected, mirrored):
    result = direction_in_base(cue, camera_to_tcp=np.eye(3), tcp_rotation_vector=[0, 0, 0],
                               preview_mirrored=mirrored, cue_frame='participant', gesture_signs=SIGNS)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('mirrored,expected', [(False, [-1, 0, 0]), (True, [1, 0, 0])])
def test_preview_left_changes_sign_when_mirrored(mirrored, expected):
    result = direction_in_base('MOVE_LEFT', camera_to_tcp=np.eye(3),
                               tcp_rotation_vector=[0, 0, 0],
                               preview_mirrored=mirrored, cue_frame='preview', gesture_signs=SIGNS)
    assert np.allclose(result, expected)


def test_camera_to_tcp_and_tcp_to_base_are_both_applied_in_order():
    # 相机+X->TCP+Y->基座+Z（90度围绕TCP Z和基座X）。
    camera_to_tcp = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = direction_in_base('MOVE_LEFT', camera_to_tcp=camera_to_tcp,
                               tcp_rotation_vector=[math.pi / 2, 0, 0],
                               preview_mirrored=True, cue_frame='participant', gesture_signs=SIGNS)
    assert np.allclose(result, [0, 0, 1], atol=1e-12)


@pytest.mark.parametrize('rotation', [
    [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 反思不是一个僵化的取向
    [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
    [[1, 0, 0], [0, 1, 0], [0, 0, float('nan')]],
    [[1, 0], [0, 1]],
])
def test_invalid_calibration_fails(rotation):
    with pytest.raises(ValueError):
        direction_in_base('MOVE_LEFT', camera_to_tcp=rotation, tcp_rotation_vector=[0, 0, 0],
                          preview_mirrored=True, cue_frame='participant', gesture_signs=SIGNS)


@pytest.mark.parametrize('pose', [[0, 0], [0, 0, float('inf')], [0, 0, True]])
def test_bad_tcp_orientation_is_rejected(pose):
    with pytest.raises(ValueError):
        direction_in_base('MOVE_LEFT', camera_to_tcp=np.eye(3), tcp_rotation_vector=pose,
                          preview_mirrored=True, cue_frame='participant', gesture_signs=SIGNS)


@pytest.mark.parametrize('cue,mirror,frame', [
    ('HOLD', True, 'participant'), ('MOVE_LEFT', 1, 'participant'),
    ('MOVE_LEFT', True, 'unreviewed'),
])
def test_ambiguous_requests_are_rejected(cue, mirror, frame):
    with pytest.raises(ValueError):
        direction_in_base(cue, camera_to_tcp=np.eye(3), tcp_rotation_vector=[0, 0, 0],
                          preview_mirrored=mirror, cue_frame=frame, gesture_signs=SIGNS)


@pytest.mark.parametrize('cue', list(SIGNS))
def test_each_gesture_sign_can_reverse_person_direction(cue):
    opts = dict(camera_to_tcp=np.eye(3), tcp_rotation_vector=[0, 0, 0],
                preview_mirrored=True, cue_frame='participant')
    person_vector = direction_in_base(cue, gesture_signs=SIGNS, **opts)
    opposite = direction_in_base(cue, gesture_signs={**SIGNS, cue: -1}, **opts)
    assert np.allclose(opposite, -np.asarray(person_vector))


@pytest.mark.parametrize('signs', [{}, {**SIGNS, 'MOVE_LEFT': 0},
                                   {**SIGNS, 'MOVE_LEFT': True},
                                   {**SIGNS, 'MOVE_LEFT': 1.0},
                                   {**SIGNS, 'EXTRA': 1}])
def test_incomplete_or_ambiguous_sign_configuration_is_rejected(signs):
    with pytest.raises(ValueError):
        direction_in_base('MOVE_LEFT', camera_to_tcp=np.eye(3), tcp_rotation_vector=[0, 0, 0],
                          preview_mirrored=True, cue_frame='participant', gesture_signs=signs)
