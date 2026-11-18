"""物理引导测试使用注入设备；无网络访问。"""
import pytest

from leica_tracker.physical_resident_bootstrap import open_resident
from leica_tracker.physical_resident_engine import Installation
from test_physical_resident_engine import reviewed


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


def setup(*, stop_during_upload=False):
    clock = Clock(); cfg = Installation(reviewed()); objects = {}
    class State:
        failure = None
        latest = ({'mode': {'connected': True, 'enabled': True, 'power': True, 'mode': 7,
                            'emergency': False, 'protective': False, 'paused': False, 'running': False},
                   'tcp_offset': list(cfg.tcp)}, 10.)
        def close(self): pass
    class RTDE:
        failure = None
        latest = ({'session': 0, 'state': 2, 'sequence': 0,
                   'tcp_pose': list(cfg.neutral), 'tcp_speed': [0.]*6,
                   'joints': list(cfg.joints)}, 10.)
        def close(self): pass
        def send(self, *args): raise AssertionError('No target before bootstrap completion')
    class Command:
        failure = None
        def __init__(self): self.sent = []
        def send(self, text):
            self.sent.append(text)
            if text.startswith('def leica_resident_xyz'):
                objects['rtde'].latest[0]['session'] = 123
                objects['state'].latest[0]['mode']['running'] = True
                if stop_during_upload: objects['watch'].close()
        def close(self): pass
    class Gate:
        def __init__(self, channel): self.channel = channel; self.can_send = False
        def stop(self, reason): self.can_send = False
        def send(self, *args): return self.channel.send(*args)
    class Watch:
        def __init__(self, reader, gate, **kwargs): self.gate = gate; objects['watch'] = self
        def start(self): self.gate.can_send = True
        def wait_ready(self, timeout): return self.gate.can_send
        def close(self): self.gate.stop('closed')
    def rtde(host, **kwargs):
        device = RTDE()
        if not kwargs.get('receive_only'): objects['rtde'] = device
        return device
    def command(host): objects['command'] = Command(); return objects['command']
    def state(host): objects['state'] = State(); return objects['state']
    def dashboard_check(*args, **kwargs): objects.setdefault('dashboard_checks', []).append((args, kwargs))
    deps = dict(state_factory=state, rtde_factory=rtde,
                command_factory=command, dashboard_check=dashboard_check,
                gate_factory=Gate, watch_factory=Watch, clock=clock, sleep=clock.sleep,
                session_factory=lambda: 123)
    return cfg, deps, objects


def test_startup_uploads_held_resident_only_without_sending_a_target():
    cfg, deps, objects = setup()
    session = open_resident('192.0.2.1', cfg, lambda: None, dependencies=deps)
    assert len(objects['command'].sent) == 1
    assert 'armed = False' in objects['command'].sent[0]
    assert session.engine.state == 'HANDSHAKE'
    assert session.feedback[0]['session'] == 123
    assert objects['dashboard_checks'] and all(call[1].get('robot_model') == 'ur10e'
                                                for call in objects['dashboard_checks'])
    session.stop()
    assert session.engine.state == 'STOPPED'
    assert not session.closed
    assert session.reader.latest[0]['tcp_speed'] == [0.]*6
    session.close()
    assert len(objects['command'].sent) == 1


def test_stop_during_upload_leaves_gate_closed_without_overwriting_a_program():
    cfg, deps, objects = setup(stop_during_upload=True)
    with pytest.raises(RuntimeError, match='stopped'):
        open_resident('192.0.2.1', cfg, lambda: None, dependencies=deps)
    assert len(objects['command'].sent) == 1
    assert not objects['watch'].gate.can_send


def test_draft_never_opens_device():
    cfg, deps, objects = setup()
    with pytest.raises(ValueError):
        open_resident('192.0.2.1', reviewed(), lambda: None, dependencies=deps)
    assert not objects


@pytest.mark.parametrize('failure', ['tcp', 'protective', 'stale', 'offset'])
def test_changed_installation_or_unsafe_state_never_uploads(failure):
    cfg, deps, objects = setup()
    make_state = deps['state_factory']
    def unsafe(host):
        result = make_state(host)
        data, stamp = result.latest
        if failure == 'tcp': data['tcp_offset'][2] += .01
        if failure == 'protective': data['mode']['protective'] = True
        if failure == 'stale': result.latest = (data, 9.)
        if failure == 'offset':
            # 更改实际的 RTDE 参考，而不是提供的审查。
            make_rtde = deps['rtde_factory']
            def displaced(host, **kwargs):
                receiver = make_rtde(host, **kwargs)
                receiver.latest[0]['tcp_pose'][0] += .01
                return receiver
            deps['rtde_factory'] = displaced
        return result
    deps['state_factory'] = unsafe
    with pytest.raises((ValueError, RuntimeError)):
        open_resident('192.0.2.1', cfg, lambda: None, dependencies=deps)
    assert 'command' not in objects or not objects['command'].sent
