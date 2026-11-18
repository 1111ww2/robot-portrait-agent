import json

import pytest

from leica_tracker import physical_status as status


class Receiver:
    def __init__(self, sample, stamp=100., failure=None):
        self.latest = (sample, stamp)
        self.failure = failure
        self.closed = False

    def close(self):
        self.closed = True


def test_only_read_only_receivers_and_correct_current_status():
    primary = Receiver({'mode': {'mode': 7, 'control': 1, 'connected': True,
                                  'power': True, 'emergency': False,
                                  'protective': False, 'running': False,
                                  'paused': False},
                        'pose': [1, 2, 3, 0, 0, 0],
                        'tcp_offset': [0, 0, .2786, 0, 0, 0]})
    rtde = Receiver({'tcp_pose': [1, 2, 3, 0, 0, 0],
                     'tcp_speed': [0, 0, 0, 0, 0, 0],
                     'joints': [0, 1, 2, 3, 4, 5]})
    calls = []
    def rtde_factory(host, **kwargs):
        calls.append((host, kwargs))
        return rtde
    report = status.inspect_robot('192.168.1.100', state_factory=lambda _: primary,
        rtde_factory=rtde_factory, dashboard_reader=lambda _: {'safety_status': 'NORMAL',
        'robot_mode': 'RUNNING', 'program_running': False, 'remote_control': True},
        clock=lambda: 100., sleep=lambda _: None)
    assert calls == [('192.168.1.100', {'startup_timeout': 3., 'spatial': True, 'receive_only': True})]
    assert primary.closed and rtde.closed
    assert report['primary']['tcp_offset_m_rad'] == [0, 0, .2786, 0, 0, 0]
    assert report['rtde']['tcp_speed_m_s_rad_s'] == [0, 0, 0, 0, 0, 0]
    assert report['dashboard']['program_running'] is False
    assert report['dashboard']['remote_control'] is True
    assert report['connected'] is True
    assert all(report[name]['fresh'] and report[name]['age_seconds'] == 0
               for name in ('primary','rtde','dashboard'))


def test_stale_samples_and_failures_cannot_look_current():
    primary = Receiver({'mode': {'connected': True}, 'pose': [0]*6, 'tcp_offset': [0]*6}, stamp=90.)
    rtde = Receiver({'tcp_pose': [0]*6, 'tcp_speed': [0]*6, 'joints': [0]*6}, stamp=90.)
    def failed(_): raise OSError('private details')
    report = status.inspect_robot('192.168.1.100', state_factory=lambda _: primary,
        rtde_factory=lambda *a, **k: rtde, dashboard_reader=failed, clock=lambda: 100., sleep=lambda _: None,
        primary_wait=0.)
    assert report['connected'] is False
    assert report['primary']['tcp_offset_m_rad'] is None
    assert report['rtde']['tcp_speed_m_s_rad_s'] is None
    assert report['dashboard']['safety_status'] is None
    assert 'private details' not in json.dumps(report)


def test_dashboard_requests_only_explicit_read_queries():
    class Sock:
        def __init__(self):
            self.sent = []
            self.lines = iter([b'Connected\n', b'Safetystatus: NORMAL\n',
                b'Robotmode: RUNNING\n', b'Program running: false\n', b'true\n'])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): assert value <= 1
        def recv(self, n): return next(self.lines)
        def sendall(self, value): self.sent.append(value)
    sock = Sock()
    result = status.read_dashboard('192.168.1.100', connect=lambda *a, **k: sock)
    assert sock.sent == [b'safetystatus\n', b'robotmode\n', b'running\n', b'is in remote control\n']
    assert result == {'safety_status': 'NORMAL', 'robot_mode': 'RUNNING',
                      'program_running': False, 'remote_control': True}


def test_unrecognized_dashboard_responses_remain_unknown():
    class Sock:
        def __init__(self): self.lines = iter([b'Connected\n', b'Unsupported\n', b'Unknown\n', b'not sure\n', b'not sure\n'])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): pass
        def recv(self, n): return next(self.lines)
        def sendall(self, value): pass
    assert status.read_dashboard('192.168.1.100', connect=lambda *a, **k: Sock()) == {
        'safety_status': None, 'robot_mode': None, 'program_running': None, 'remote_control': None}


def test_dashboard_slow_peer_cannot_extend_deadline():
    class Sock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): assert value <= .7
        def recv(self, n): return b'x'
        def sendall(self, value): raise AssertionError('greeting incomplete')
    ticks = iter([0, .5, 1., 2., 3., 3.6])
    with pytest.raises(TimeoutError):
        status.read_dashboard('192.168.1.100', connect=lambda *a, **k: Sock(), clock=lambda: next(ticks))


def test_no_connections_are_attempted_when_ip_invalid(monkeypatch):
    def fail(*args, **kwargs): raise AssertionError('must not connect')
    monkeypatch.setattr(status.socket, 'create_connection', fail)
    with pytest.raises(ValueError): status.inspect_robot('not-an-address')


def test_rtde_unavailable_does_not_hide_primary_or_claim_velocity():
    primary = Receiver({'mode': {'connected': True}, 'pose': [0]*6, 'tcp_offset': [0]*6})
    def fail(*args, **kwargs): raise ImportError('missing module')
    report = status.inspect_robot('192.168.1.100', state_factory=lambda _: primary,
        rtde_factory=fail, dashboard_reader=lambda _: {}, clock=lambda: 100., sleep=lambda _: None)
    assert report['connected'] is True and report['primary']['available'] is True
    assert report['rtde']['available'] is False and report['rtde']['tcp_speed_m_s_rad_s'] is None
    assert report['dashboard']['available'] is False


def test_unavailable_rtde_registers_leave_other_channels_visible():
    primary = Receiver({'mode': {'connected': True}, 'pose': [0]*6, 'tcp_offset': [0]*6})
    def no_registers(*args, **kwargs): raise RuntimeError('register 40 unsupported')
    dashboard = {'safety_status': 'NORMAL', 'robot_mode': 'RUNNING',
                 'program_running': False, 'remote_control': True}
    report = status.inspect_robot('192.168.1.100', state_factory=lambda _: primary,
        rtde_factory=no_registers, dashboard_reader=lambda _: dashboard,
        clock=lambda: 100., sleep=lambda _: None)
    assert report['connected'] is True
    assert report['primary']['available'] and report['dashboard']['available']
    assert report['rtde']['available'] is False and report['rtde']['tcp_speed_m_s_rad_s'] is None
    assert 'not necessarily a disconnected robot' in report['note']
    assert 'single fresh sample cannot rule out queued data' in report['note']
    assert 'more than two seconds' in report['note']


def test_independent_samples_are_marked_stale_if_dashboard_takes_time():
    now = [100.]
    primary = Receiver({'mode': {'connected': True}, 'pose': [0]*6, 'tcp_offset': [0]*6})
    rtde = Receiver({'tcp_pose': [0]*6, 'tcp_speed': [0]*6, 'joints': [0]*6})
    def slow_dashboard(_):
        now[0] = 102.
        return {'safety_status': 'NORMAL', 'robot_mode': 'RUNNING',
                'program_running': False, 'remote_control': True}
    result = status.inspect_robot('192.168.1.100', state_factory=lambda _: primary,
        rtde_factory=lambda *a, **k: rtde, dashboard_reader=slow_dashboard,
        clock=lambda: now[0], sleep=lambda _: None)
    assert all(result[name]['available'] and not result[name]['fresh']
               and result[name]['age_seconds'] == 2 for name in ('primary','rtde','dashboard'))
    assert 'not simultaneously' in result['note']
    assert result['elapsed_seconds'] == 2


@pytest.mark.parametrize('address', ['example.com', '127.0.0.1/24', '', '1.2.3.999'])
def test_explicit_ip_required_without_network(address):
    with pytest.raises(ValueError):
        status.inspect_robot(address)
