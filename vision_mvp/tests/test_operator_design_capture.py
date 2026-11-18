"""只读捕获访问使用假数据；这些测试不会打开任何接口。"""
import copy
import math
from types import SimpleNamespace

import pytest

from leica_tracker.operator_design_capture import capture_selected_reference
from leica_tracker.physical_errors import SessionCleanupError


CONTROL = {'instance': 'selected-test', 'epoch': 7, 'started_ns': 5000}


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now = round(self.now + seconds, 8)


def setup(*, change=None, guest_change=None, stop=lambda: False):
    clock = Clock()
    objects = {'dashboard': [], 'channels': [], 'stop': False, 'reader_calls': 0}
    profile = SimpleNamespace(host='192.0.2.1', cfg=SimpleNamespace(robot_model='ur5e'))
    pose = [.4, -.2, .6, 0., 0., 0.]
    joints = [.3, -1.2, -1.4, -.8, 1.4, .2]

    class Device:
        def __init__(self, kind):
            self.kind = kind; self.failure = None; self.closed = False
            self.close_error = False; self.close_delay = 0.
            objects['channels'].append(self)

        @property
        def latest(self):
            rate = 10 if self.kind == 'primary' else 50
            stamp = math.floor((clock.now + 1e-7) * rate) / rate
            if self.kind == 'primary':
                frame = dict(pose=list(pose), tcp_offset=[0., 0., .2786, 0., 0., 0.],
                    mode=dict(timestamp=round(stamp * 1e6), mode=7, connected=True,
                        enabled=True, power=True, emergency=False, protective=False,
                        paused=False, running=False))
            else:
                frame = dict(timestamp=stamp, tcp_pose=list(pose), joints=list(joints),
                    tcp_speed=[0.] * 6, joint_speed=[0.] * 6)
            if change is not None:
                replacement = change(self.kind, frame, stamp, clock.now)
                if replacement is not None: return replacement
            return frame, stamp

        def close(self):
            self.closed = True; clock.sleep(self.close_delay)
            if self.close_error: raise RuntimeError('injected read-only cleanup failure')

        def send(self, *_): raise AssertionError('Read-only capture never writes targets')

    def state_factory(host):
        assert host == profile.host
        objects['primary'] = Device('primary')
        return objects['primary']

    def rtde_factory(host, **kwargs):
        assert host == profile.host
        assert kwargs['spatial'] is True and kwargs['receive_only'] is True
        assert 0 < kwargs['startup_timeout'] <= 8.
        objects['rtde_options'] = kwargs
        objects['rtde'] = Device('feedback')
        return objects['rtde']

    def dashboard_check(host, **kwargs):
        assert host == profile.host
        assert kwargs == {'idle': True, 'robot_model': 'ur5e'}
        objects['dashboard'].append(clock.now)

    def reader():
        objects['reader_calls'] += 1
        value = dict(CONTROL, active=True, control_fresh=True,
            camera_connected=True, camera_fresh=True, control_action='start')
        if guest_change is not None: guest_change(value, clock.now)
        return value

    deps = dict(state_factory=state_factory, rtde_factory=rtde_factory,
        dashboard_check=dashboard_check, clock=clock, sleep=clock.sleep)
    return profile, dict(reader=reader, control=dict(CONTROL),
        should_stop=lambda: objects['stop'] or stop(), dependencies=deps), clock, objects


def run_setup(**kwargs):
    profile, options, clock, objects = setup(**kwargs)
    result = capture_selected_reference(profile, **options)
    return result, clock, objects


def test_capture_proves_advancing_stillness_and_uses_only_receive_channels():
    (primary, feedback), clock, objects = run_setup()
    assert clock.now >= 10.6
    assert primary['pose'] == feedback['tcp_pose'] == [.4, -.2, .6, 0., 0., 0.]
    assert primary['tcp_offset'] == [0., 0., .2786, 0., 0., 0.]
    assert feedback['joints'] == [.3, -1.2, -1.4, -.8, 1.4, .2]
    assert objects['dashboard'] and objects['reader_calls'] >= 3
    assert len(objects['channels']) == 2 and all(c.closed for c in objects['channels'])
    assert objects['rtde_options']['receive_only'] is True


@pytest.mark.parametrize('field,value', [('instance', 'other'), ('epoch', 8),
    ('started_ns', 6000), ('active', False), ('control_fresh', False),
    ('camera_connected', False), ('camera_fresh', False), ('stale', True)])
def test_wrong_or_inactive_guest_cannot_open_native_receivers(field, value):
    profile, options, _, objects = setup(guest_change=lambda x, _: x.update({field: value}))
    with pytest.raises((ValueError, RuntimeError), match='Guest|guest|control|Control'):
        capture_selected_reference(profile, **options)
    assert objects['channels'] == []


def test_real_guest_projection_can_omit_stale_when_fresh_flags_are_true():
    # LocalGuestControlReader不会投影过时的字段。
    _, _, objects = run_setup()
    assert all(c.closed for c in objects['channels'])


@pytest.mark.parametrize('field,value', [('mode', 5), ('running', True), ('paused', True),
    ('protective', True), ('emergency', True), ('connected', False),
    ('enabled', False), ('power', False)])
def test_unready_native_mode_is_rejected(field, value):
    def change(kind, frame, stamp, now):
        if kind == 'primary': frame['mode'][field] = value
    profile, options, _, objects = setup(change=change)
    with pytest.raises((ValueError, RuntimeError), match='mode|idle|ready'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


@pytest.mark.parametrize('channel', ['primary', 'feedback'])
@pytest.mark.parametrize('offset', [-.2, 1.])
def test_stale_or_future_native_receipt_is_never_a_reference(channel, offset):
    def change(kind, frame, stamp, now):
        if kind == channel: return frame, now + offset
    profile, options, _, objects = setup(change=change)
    with pytest.raises((ValueError, RuntimeError), match='fresh|stale|clock'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


@pytest.mark.parametrize('channel', ['primary', 'feedback'])
def test_repeated_producer_clock_cannot_be_freshened_by_new_receipt_times(channel):
    def change(kind, frame, stamp, now):
        if kind == channel:
            if kind == 'primary': frame['mode']['timestamp'] = 10000000
            else: frame['timestamp'] = 10.
    profile, options, clock, objects = setup(change=change)
    with pytest.raises((ValueError, RuntimeError), match='advanc|stationary|reference|timed out'):
        capture_selected_reference(profile, **options)
    assert clock.now <= 15.1 and all(c.closed for c in objects['channels'])


@pytest.mark.parametrize('field,index,value', [('joint_speed', 0, .003),
    ('tcp_speed', 0, .0005), ('tcp_speed', 3, .002)])
def test_continuous_motion_cannot_be_captured(field, index, value):
    def change(kind, frame, stamp, now):
        if kind == 'feedback': frame[field][index] = value
    profile, options, clock, objects = setup(change=change)
    with pytest.raises((ValueError, RuntimeError), match='stationary|reference|moving|timed out'):
        capture_selected_reference(profile, **options)
    assert clock.now <= 15.1 and all(c.closed for c in objects['channels'])


def test_a_motion_sample_restarts_the_full_stillness_dwell():
    def change(kind, frame, stamp, now):
        if kind == 'feedback' and 10.25 <= now < 10.45:
            frame['joint_speed'][0] = .01
    _, clock, _ = run_setup(change=change)
    assert clock.now >= 11.05


@pytest.mark.parametrize('angular', [False, True])
def test_sources_must_agree_before_a_reference_is_captured(angular):
    def change(kind, frame, stamp, now):
        if kind == 'primary': frame['pose'][3 if angular else 0] += .01
    profile, options, _, objects = setup(change=change)
    with pytest.raises((ValueError, RuntimeError), match='agree|reference|stationary|timed out'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


def test_tcp_configuration_change_cannot_be_folded_into_the_same_proof():
    def change(kind, frame, stamp, now):
        if kind == 'primary' and now >= 10.3: frame['tcp_offset'][2] += .001
    profile, options, _, objects = setup(change=change)
    with pytest.raises((ValueError, RuntimeError), match='TCP.*chang|TCP.*stable'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


@pytest.mark.parametrize('kind,field', [('primary', 'pose'), ('primary', 'tcp_offset'),
    ('feedback', 'joints'), ('feedback', 'tcp_pose'), ('feedback', 'tcp_speed'),
    ('feedback', 'joint_speed')])
def test_nonfinite_vectors_are_rejected(kind, field):
    def change(channel, frame, stamp, now):
        if channel == kind: frame[field][0] = float('nan')
    profile, options, _, objects = setup(change=change)
    with pytest.raises((ValueError, RuntimeError), match='finite|vector|Invalid'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


def test_stop_during_rtde_bootstrap_closes_both_owned_readers_before_returning():
    profile, options, _, objects = setup()
    factory = options['dependencies']['rtde_factory']
    def stopped(host, **kwargs):
        device = factory(host, **kwargs); objects['stop'] = True; return device
    options['dependencies']['rtde_factory'] = stopped
    with pytest.raises(RuntimeError, match='stop|Stop'):
        capture_selected_reference(profile, **options)
    assert len(objects['channels']) == 2 and all(c.closed for c in objects['channels'])


def test_stop_on_entry_does_not_open_dashboard_or_receivers():
    profile, options, _, objects = setup(stop=lambda: True)
    with pytest.raises(RuntimeError, match='stop|Stop'):
        capture_selected_reference(profile, **options)
    assert objects['channels'] == [] and objects['dashboard'] == []


def test_changed_control_during_capture_cannot_produce_a_reference():
    profile, options, _, objects = setup(guest_change=lambda x, now: x.update(epoch=8) if now >= 10.3 else None)
    with pytest.raises(RuntimeError, match='Guest|guest|control|Control'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


def test_wrong_model_dashboard_failure_precedes_receiver_creation():
    profile, options, _, objects = setup()
    def bad(*_, **__): raise RuntimeError('Physical robot model mismatch')
    options['dependencies']['dashboard_check'] = bad
    with pytest.raises(RuntimeError, match='model mismatch'):
        capture_selected_reference(profile, **options)
    assert objects['channels'] == []


def test_receiver_factory_failure_still_closes_the_primary():
    profile, options, _, objects = setup()
    def bad(*_, **__): raise RuntimeError('injected RTDE startup failure')
    options['dependencies']['rtde_factory'] = bad
    with pytest.raises(RuntimeError, match='RTDE startup'):
        capture_selected_reference(profile, **options)
    assert len(objects['channels']) == 1 and objects['primary'].closed


def test_all_readers_are_retired_even_if_one_close_fails():
    profile, options, _, objects = setup()
    factory = options['dependencies']['rtde_factory']
    def bad_close(host, **kwargs):
        device = factory(host, **kwargs); device.close_error = True; return device
    options['dependencies']['rtde_factory'] = bad_close
    with pytest.raises(SessionCleanupError, match='cleanup'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


def test_capture_expiring_during_cleanup_is_not_returned_as_fresh():
    profile, options, _, objects = setup()
    factory = options['dependencies']['rtde_factory']
    def slow_close(host, **kwargs):
        device = factory(host, **kwargs); device.close_delay = .2; return device
    options['dependencies']['rtde_factory'] = slow_close
    with pytest.raises(RuntimeError, match='expired|fresh|cleanup'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


def test_fixed_guest_error_text_does_not_expose_control_identity():
    profile, options, _, _ = setup(guest_change=lambda x, _: x.update(instance='sensitive-other'))
    with pytest.raises(RuntimeError) as exc: capture_selected_reference(profile, **options)
    assert 'selected-test' not in str(exc.value) and 'sensitive-other' not in str(exc.value)


def test_final_snapshot_callback_runs_once_with_readers_open_then_reproves_full_dwell():
    profile, options, clock, objects = setup()
    callback_finished = []
    def check_identity():
        assert len(objects['channels']) == 2
        assert all(not c.closed for c in objects['channels'])
        clock.sleep(.4)  # 比不变的150毫秒原生新鲜度预算更长。
        callback_finished.append(clock.now)
    options['dependencies']['before_final_snapshot'] = check_identity
    primary, feedback = capture_selected_reference(profile, **options)
    assert len(callback_finished) == 1
    assert clock.now - callback_finished[0] >= .6
    assert primary['mode']['timestamp'] / 1e6 >= callback_finished[0] + .5
    assert feedback['timestamp'] >= callback_finished[0] + .58
    assert all(c.closed for c in objects['channels'])


def test_final_snapshot_callback_cannot_extend_the_original_proof_deadline():
    profile, options, clock, objects = setup()
    called = []
    def slow_identity():
        called.append(clock.now); clock.sleep(4.9)
    options['dependencies']['before_final_snapshot'] = slow_identity
    with pytest.raises(RuntimeError, match='reference.*timed out'):
        capture_selected_reference(profile, **options)
    assert len(called) == 1
    assert all(c.closed for c in objects['channels'])


def test_final_snapshot_callback_failure_retires_both_receivers_without_a_reference():
    profile, options, _, objects = setup()
    def changed_identity(): raise RuntimeError('Docker identity changed')
    options['dependencies']['before_final_snapshot'] = changed_identity
    with pytest.raises(RuntimeError, match='Docker identity changed'):
        capture_selected_reference(profile, **options)
    assert len(objects['channels']) == 2 and all(c.closed for c in objects['channels'])


def test_final_snapshot_callback_stop_retires_both_receivers_without_a_reference():
    profile, options, clock, objects = setup()
    def stop_during_identity():
        clock.sleep(.4); objects['stop'] = True
    options['dependencies']['before_final_snapshot'] = stop_during_identity
    with pytest.raises(RuntimeError, match='stopped'):
        capture_selected_reference(profile, **options)
    assert all(c.closed for c in objects['channels'])


def test_final_snapshot_callback_keeps_tcp_change_detection_across_the_check():
    checked = []
    def change(kind, frame, stamp, now):
        if kind == 'primary' and checked: frame['tcp_offset'][2] += .001
    profile, options, clock, objects = setup(change=change)
    def changed_tcp(): checked.append(clock.now); clock.sleep(.4)
    options['dependencies']['before_final_snapshot'] = changed_tcp
    with pytest.raises(RuntimeError, match='TCP.*changed'):
        capture_selected_reference(profile, **options)
    assert len(checked) == 1 and all(c.closed for c in objects['channels'])


@pytest.mark.parametrize('channel', ['primary', 'feedback'])
def test_final_snapshot_callback_keeps_producer_clock_regression_detection(channel):
    checked = []
    def change(kind, frame, stamp, now):
        if kind == channel and checked:
            if kind == 'primary': frame['mode']['timestamp'] = 9000000
            else: frame['timestamp'] = 9.
    profile, options, clock, objects = setup(change=change)
    def clock_reset(): checked.append(clock.now); clock.sleep(.4)
    options['dependencies']['before_final_snapshot'] = clock_reset
    with pytest.raises(RuntimeError, match='clock stopped advancing'):
        capture_selected_reference(profile, **options)
    assert len(checked) == 1 and all(c.closed for c in objects['channels'])


def test_final_snapshot_callback_does_not_relabel_old_receipts_as_fresh():
    checked = []
    def change(kind, frame, stamp, now):
        if checked: return frame, checked[0]
    profile, options, clock, objects = setup(change=change)
    def blocked_receivers(): checked.append(clock.now); clock.sleep(.4)
    options['dependencies']['before_final_snapshot'] = blocked_receivers
    with pytest.raises(RuntimeError, match='receipt stale'):
        capture_selected_reference(profile, **options)
    assert len(checked) == 1 and all(c.closed for c in objects['channels'])


def test_final_snapshot_callback_must_be_callable_before_any_receiver_is_opened():
    profile, options, _, objects = setup()
    options['dependencies']['before_final_snapshot'] = 'not a callback'
    with pytest.raises(ValueError, match='callback|callable|Callable'):
        capture_selected_reference(profile, **options)
    assert objects['channels'] == [] and objects['dashboard'] == []
