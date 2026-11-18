"""重启观察器使用独立只读通道的原始样本，不使用运动通道。"""
import copy

import pytest

from leica_tracker.operator_design import selected_profile
from leica_tracker.operator_design_capture import capture_selected_reference
from test_operator_design import design
from test_operator_design_capture import setup
from test_ursim_profile import snapshots


def test_observer_gets_each_advancing_sample_including_motion_with_original_receipt_time():
    def moving(kind, frame, stamp, now):
        if kind == 'feedback' and 10.2 <= now < 10.4:
            frame['joint_speed'][0] = .01
    profile, options, clock, objects = setup(change=moving)
    observed = []
    def observe(primary, native, sampled_at):
        observed.append((copy.deepcopy(primary), copy.deepcopy(native), sampled_at))
        assert sampled_at == min(primary['mode']['timestamp'] / 1e6, native['timestamp'])
        return 'ignored callback return'
    primary, native = capture_selected_reference(profile, sample_observer=observe, **options)
    assert len(observed) >= 8
    assert any(item[1]['joint_speed'][0] > 0 for item in observed)
    for previous, current in zip(observed, observed[1:]):
        assert current[0]['mode']['timestamp'] > previous[0]['mode']['timestamp']
        assert current[1]['timestamp'] > previous[1]['timestamp']
        assert current[2] > previous[2]
    assert clock.now >= 11.0
    assert primary['pose'] == native['tcp_pose']
    assert all(device.closed for device in objects['channels'])


def test_observer_cannot_mutate_the_admitted_reference():
    profile, options, _, _ = setup()
    def observe(primary, native, sampled_at):
        primary['pose'][0] = 99.
        native['tcp_pose'][0] = 99.
    primary, native = capture_selected_reference(profile, sample_observer=observe, **options)
    assert primary['pose'][0] == native['tcp_pose'][0] == .4


def test_observer_exception_retires_both_receive_only_channels():
    profile, options, _, objects = setup()
    def observe(*_):
        raise RuntimeError('old reference proof rejected')
    with pytest.raises(RuntimeError, match='proof rejected'):
        capture_selected_reference(profile, sample_observer=observe, **options)
    assert len(objects['channels']) == 2 and all(device.closed for device in objects['channels'])


def test_stop_during_observer_prevents_capture_and_retires_channels():
    profile, options, _, objects = setup()
    def observe(*_):
        objects['stop'] = True
    with pytest.raises(RuntimeError, match='stopped'):
        capture_selected_reference(profile, sample_observer=observe, **options)
    assert all(device.closed for device in objects['channels'])


def test_epoch_change_during_observer_is_checked_before_admission():
    changed = [False]
    profile, options, _, objects = setup(
        guest_change=lambda state, _: state.update(epoch=99) if changed[0] else None)
    def observe(*_):
        changed[0] = True
    with pytest.raises(RuntimeError, match='Guest control changed'):
        capture_selected_reference(profile, sample_observer=observe, **options)
    assert all(device.closed for device in objects['channels'])


def test_blocking_observer_cannot_freshen_an_expired_sample():
    profile, options, clock, objects = setup()
    def observe(*_):
        clock.sleep(.2)
    with pytest.raises(RuntimeError, match='expired|stale|fresh'):
        capture_selected_reference(profile, sample_observer=observe, **options)
    assert all(device.closed for device in objects['channels'])


def test_invalid_observer_is_rejected_before_opening_devices():
    profile, options, _, objects = setup()
    with pytest.raises(ValueError, match='observer'):
        capture_selected_reference(profile, sample_observer=123, **options)
    assert not objects['channels'] and not objects['dashboard']


@pytest.mark.parametrize('with_observer', [False, True])
def test_selected_profile_forwards_observer_only_when_supplied(with_observer):
    calls = []
    def provider(profile, **arguments):
        calls.append(arguments)
        return snapshots()
    profile = selected_profile(design(), capture_provider=provider)
    observer = (lambda *_: None) if with_observer else None
    bound = profile.bind_before_motion(reader=object(), control={'epoch': 1},
        should_stop=lambda: False, sample_observer=observer)
    assert bound.reference_bound
    assert ('sample_observer' in calls[0]) is with_observer
    if with_observer:
        assert calls[0]['sample_observer'] is observer
