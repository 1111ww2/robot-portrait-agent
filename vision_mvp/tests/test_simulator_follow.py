import importlib
import math

import pytest
from leica_tracker.horizontal_follow import VisionSamples


def adapter():
    name = 'leica_tracker.simulator_follow'
    assert importlib.util.find_spec(name) is not None, 'Simulator adapter is missing'
    return importlib.import_module(name)


def test_synthetic_stream_uses_the_existing_vision_contract():
    import json
    samples = VisionSamples()
    for i in range(4):
        now = 100. + i / 10
        event = adapter().synthetic_event(now, 3. + i / 10)
        samples.accept(json.dumps(event), now)
    assert samples.latest[2] == 4
    assert 0 < samples.latest[0] <= 1
    event = adapter().synthetic_event(101., 21.)
    samples.accept(json.dumps(event), 101.)
    assert samples.latest[0] is None
    assert samples.latest[3] == 1


def test_simulator_velocity_is_horizontal_bounded_and_signed():
    neutral = [-.49, -.13, .21, 0., 1., 0.]
    velocity = adapter().target_velocity(neutral, neutral, (1., 10., 3, 0), 10.)
    assert velocity == [-.08, 0., 0., 0., 0., 0.]
    pose = [-.587, -.13, .21, 0., 1., 0.]
    assert adapter().target_velocity(neutral, pose, (1., 10., 3, 0), 10.) == [0.]*6


@pytest.mark.parametrize('sample', [(1., 9., 3, 0), (None, 10., 0, 1),
                                   (1., 10., 2, 0), (1., 11., 3, 0)])
def test_invalid_or_stale_source_holds_without_returning_home(sample):
    neutral = [-.49, -.13, .21, 0., 1., 0.]
    pose = [-.54, -.13, .21, 0., 1., 0.]
    assert adapter().target_velocity(neutral, pose, sample, 10.) == [0.]*6


def test_non_simulator_identity_is_rejected():
    valid = dict(target=dict(kind='simulator', authority_origin='simulation',
                             instance_id='abc'))
    assert adapter().verify_identity(valid) == 'abc'
    valid['target']['kind'] = 'robot'
    with pytest.raises(ValueError):
        adapter().verify_identity(valid)


def test_bad_pose_fails_closed():
    with pytest.raises(ValueError):
        adapter().target_velocity([0.]*6, [math.nan]*6, (1., 10., 3, 0), 10.)


@pytest.mark.parametrize('right', [-1., 1.])
def test_large_virtual_range_continues_past_old_limit_and_stops_at_new_target(right):
    neutral = [-.49, -.13, .21, 0., 1., 0.]
    pose = neutral.copy()
    pose[0] -= right*.12
    assert adapter().target_velocity(neutral, pose, (right, 10., 3, 0), 10., range_m=.25) == [-right*.08, 0., 0., 0., 0., 0.]
    pose[0] = neutral[0]-right*.247
    assert adapter().target_velocity(neutral, pose, (right, 10., 3, 0), 10., range_m=.25) == [0.]*6
    assert adapter().target_velocity(neutral, pose, (right, 9., 3, 0), 10., range_m=.25) == [0.]*6
    pose[0] = neutral[0]-right*.252
    with pytest.raises(ValueError, match='horizontal range'):
        adapter().target_velocity(neutral, pose, (right, 10., 3, 0), 10., range_m=.25)


@pytest.mark.parametrize('range_m', [0., -.1, .009, .251, math.nan, math.inf])
def test_invalid_virtual_range_is_rejected(range_m):
    with pytest.raises(ValueError, match='range'):
        adapter().target_velocity([0.]*6, [0.]*6, (1., 10., 3, 0), 10., range_m=range_m)
