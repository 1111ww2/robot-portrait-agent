import json
import struct

import pytest

from leica_tracker.horizontal_follow import HorizontalLimits, reviewed_horizontal_limits, read_target, next_target, VisionSamples, MotionSupervisor, CommandChannel
from leica_tracker.robot_state import parse_state, ProducerClock, text_marker


class _DashboardStream:
    def __init__(self, lines): self.lines=iter(lines)
    def readline(self, _limit): return next(self.lines)


class _DashboardSocket:
    def __init__(self, lines): self.stream=_DashboardStream(lines);self.sent=[]
    def __enter__(self): return self
    def __exit__(self,*_args): pass
    def settimeout(self,_value): pass
    def makefile(self,_mode): return self.stream
    def sendall(self,value): self.sent.append(value.decode().strip())


def test_ur10e_dashboard_identity_is_checked_before_physical_use(monkeypatch):
    from leica_tracker import horizontal_follow as module
    fake=_DashboardSocket([b'Connected\n',b'Safetystatus: NORMAL\n',b'true\n',
                           b'Program running: false\n',b'UR10\n',
                           b'URSoftware 5.13.0.113898\n'])
    monkeypatch.setattr(module.socket,'create_connection',lambda *_a,**_k:fake)

    module.check_dashboard('192.0.2.1',idle=True,robot_model='ur10e')

    assert fake.sent==['safetystatus','is in remote control','running',
                       'get robot model','PolyscopeVersion']


def test_wrong_or_unsupported_robot_model_is_rejected(monkeypatch):
    from leica_tracker import horizontal_follow as module
    monkeypatch.setattr(module.socket,'create_connection',
                        lambda *_a,**_k:pytest.fail('unsupported model must fail before socket'))
    with pytest.raises(ValueError,match='robot model'):
        module.check_dashboard('192.0.2.1',robot_model='ur5')

    fake=_DashboardSocket([b'Connected\n',b'Safetystatus: NORMAL\n',b'true\n',
                           b'UR5\n',b'URSoftware 5.13.0.113898\n'])
    monkeypatch.setattr(module.socket,'create_connection',lambda *_a,**_k:fake)
    with pytest.raises(RuntimeError,match='model'):
        module.check_dashboard('192.0.2.1',robot_model='ur10e')


def limits():
    return HorizontalLimits(neutral=(-0.135, -0.506, 0.683, 0.0, -2.85, 1.31), sign=-1,range_m=.10)


def calibration(range_m=.5,reviewed=.5):
    return {'status':'HORIZONTAL_ONLY_CONFIRMED','base_axis':'X',
            'neutral_tcp_m_rad':list(limits().neutral),'camera_right_sign':-1,
            'range_m':range_m,'reviewed_range_m':reviewed}


def test_no_implicit_ten_centimetre_limit_and_configured_range_can_exceed_it():
    with pytest.raises(TypeError):HorizontalLimits(limits().neutral,-1)
    cfg=reviewed_horizontal_limits(calibration())
    assert cfg.range_m==.5
    pose=list(cfg.neutral);pose[0]-=.2
    target=next_target(cfg,pose,1.,age=.1)
    assert target[0]==pytest.approx(pose[0]-.005)
    assert next_target(cfg,pose,1.,age=.6) is None


@pytest.mark.parametrize('value',[0.,-.1,.003,float('nan'),float('inf'),True])
def test_range_rejects_invalid_or_too_small_for_existing_margin(value):
    with pytest.raises(ValueError):HorizontalLimits(limits().neutral,-1,range_m=value)


@pytest.mark.parametrize('value',[True,False,0,1.0])
def test_direction_sign_must_be_an_explicit_integer(value):
    with pytest.raises(ValueError):HorizontalLimits(limits().neutral,value,range_m=.5)


def test_review_is_bound_to_the_configured_range_before_connection():
    cal=calibration();cal.pop('reviewed_range_m')
    with pytest.raises(ValueError,match='review'):reviewed_horizontal_limits(cal)
    with pytest.raises(ValueError,match='review'):reviewed_horizontal_limits(calibration(.5,.1))


@pytest.mark.parametrize('module_name',['leica_tracker.horizontal_follow','leica_tracker.resident_follow'])
def test_legacy_range_requires_matching_review_before_robot_connection(module_name,tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module(module_name)
    monkeypatch.setattr(module,'check_dashboard',lambda *a,**k:pytest.fail('Robot connection attempted'))
    cal=calibration(.5,.1);cal.update(robot_ip='unused.invalid',camera=1,tcp_offset_m_rad=[0.]*6)
    path=tmp_path/'old-calibration.json';path.write_text(json.dumps(cal))
    with pytest.raises(ValueError,match='review'):
        module.main(['--calibration',str(path),'--execute'])


def event(right=0.8, tracking=True):
    return json.dumps({"event": "tracking_target", "tracking": tracking,
        "timestamp_ms": 1000, "command": {"right": right, "down": 0, "toward_subject": 0},
        "motion_intent_m": {"hold": not tracking}})


def test_reversed_camera_mapping_moves_only_base_x_in_bounded_segments():
    cfg = limits()
    target = next_target(cfg, cfg.neutral, read_target(event()), age=0.1)
    assert target[0] == pytest.approx(cfg.neutral[0] - cfg.segment_m)
    assert target[1:] == list(cfg.neutral[1:])
    pose = list(cfg.neutral)
    for _ in range(100):
        candidate = next_target(cfg, pose, 1.0, age=0.1)
        if candidate is not None:
            assert abs(candidate[0] - pose[0]) <= cfg.segment_m + 1e-12
            pose = candidate
    assert pose[0] == pytest.approx(cfg.neutral[0] - cfg.range_m)


@pytest.mark.parametrize('sample,age', [(None, 0.0), (0.8, 0.6)])
def test_loss_and_stale_samples_never_generate_a_target(sample, age):
    assert next_target(limits(), limits().neutral, sample, age) is None


@pytest.mark.parametrize('sample', [event(tracking=False), event(float('nan')), event(1.1), '{}'])
def test_invalid_or_lost_vision_is_rejected(sample):
    assert read_target(sample) is None


def test_nonhorizontal_camera_output_is_rejected():
    sample = json.loads(event())
    sample['command']['down'] = 0.2
    assert read_target(json.dumps(sample)) is None


@pytest.mark.parametrize('axis,offset', [(0,0.11),(1,0.002),(2,0.002),(3,0.02)])
def test_robot_outside_confirmed_path_is_rejected(axis, offset):
    pose = list(limits().neutral)
    pose[axis] += offset
    with pytest.raises(ValueError):
        next_target(limits(), pose, 0.8, age=0)


def test_native_state_parses_47_byte_mode_without_legacy_offset_error():
    mode = struct.pack('!IBQ7?BB3dB',47,0,123,True,True,True,False,False,False,False,7,0,.39,0.,1.,0)
    cart = struct.pack('!IB12d',101,4,*limits().neutral,0,0,.2786,0,0,0)
    state = parse_state(mode + cart)
    assert state['mode']['fraction'] == pytest.approx(.39)
    assert state['pose'] == list(limits().neutral)
    assert state['tcp_offset'][2] == pytest.approx(.2786)


@pytest.mark.parametrize('packet', [b'', b'abc', struct.pack('!IB',101,4)+b'abc'])
def test_native_state_rejects_truncated_or_missing_fields(packet):
    with pytest.raises(ValueError):
        parse_state(packet)


def source_event(stamp, tracking=True):
    data = json.loads(event(tracking=tracking))
    data['source_monotonic_ns'] = int(stamp * 1_000_000_000)
    return json.dumps(data)


def test_old_frames_do_not_become_fresh_and_loss_cannot_be_overwritten():
    samples = VisionSamples()
    samples.accept(source_event(1), now=2)
    assert samples.latest[0] is None
    assert samples.latest[1] == 1
    for i in range(4):
        samples.accept(source_event(2 + i / 100), now=2 + i / 100)
    assert samples.latest[2] == 4
    assert samples.latest[3] == 1  # 恢复后损失仍然可见


def test_delayed_increasing_robot_timestamps_retain_backlog_age():
    clock = ProducerClock()
    assert clock.observe(1_000_000, 10) == 10
    assert clock.observe(1_100_000, 10.1) == pytest.approx(10.1)
    source_time = clock.observe(1_200_000, 12)
    assert 12 - source_time == pytest.approx(1.8)


def test_stop_requires_matching_ack_and_post_ack_idle_before_resume():
    controller = MotionSupervisor(limits())
    state = {'pose': list(limits().neutral), 'mode': {'running': False}}
    fresh = lambda now: (.8, now, 4, 0)
    assert controller.step(10, state, 9.99, fresh(10), (None, 0))[0] == 'stop'
    marker = controller.pending[1]
    assert controller.step(10.1, state, 9.99, fresh(10.1), (None, 0)) is None
    assert controller.step(10.2, state, 10.05, fresh(10.2), (marker, 10.1)) is None
    assert controller.pending is not None
    assert controller.step(10.3, state, 10.25, fresh(10.3), (marker, 10.1)) is None
    assert controller.pending is None
    assert controller.step(10.4, state, 10.35, fresh(10.4), (marker, 10.1))[0] == 'move'
    # 丢失的帧后跟着有效的帧仍然会中断当前的运动。
    assert controller.step(10.5, state, 10.45, (.8, 10.5, 4, 1), (marker, 10.1))[0] == 'stop'


def test_missing_stop_ack_fails_closed_without_motion():
    controller = MotionSupervisor(limits())
    state = {'pose': list(limits().neutral), 'mode': {'running': False}}
    controller.step(10, state, 10, (.8, 10, 4, 0), (None, 0))
    with pytest.raises(RuntimeError, match='acknowledgement'):
        controller.step(14, state, 14, (.8, 14, 4, 0), (None, 0))


def test_primary_text_ack_parser_requires_text_message():
    body = struct.pack('!Qbb', 123, -2, 0) + b'LEICA_ACK_test'
    assert text_marker(body) == 'LEICA_ACK_test'
    assert text_marker(body[:9] + b'\x03' + body[10:]) is None


def test_partial_write_poisoned_connection_cannot_accept_cleanup_stop():
    class PartialSocket:
        writes = 0
        closed = False
        def sendall(self, data):
            self.writes += 1
            raise TimeoutError('partial write')
        def close(self):
            self.closed = True
    class FinishedThread:
        def join(self, timeout):
            pass
    channel = CommandChannel.__new__(CommandChannel)
    channel.sock = PartialSocket()
    channel.thread = FinishedThread()
    channel.failure = None
    channel.closed = False
    with pytest.raises(TimeoutError):
        channel.send('def incomplete():\n  stopl(0.1)\nend')
    assert channel.closed and channel.sock.closed
    with pytest.raises(RuntimeError, match='partial write'):
        channel.send('stopl(0.1)')
    assert channel.sock.writes == 1
