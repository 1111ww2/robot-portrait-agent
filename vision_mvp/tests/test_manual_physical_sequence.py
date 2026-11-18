import pytest

from leica_tracker.manual_physical_sequence import ManualControl, open_manual_sequence
from test_physical_intro_bootstrap import setup as intro_setup


def setup():
    profile, deps, objects = intro_setup()
    base = deps['gate_factory']
    class Gate(base):
        def arm(self):
            if self.close_finished.is_set():
                raise RuntimeError('stopped')
            self.can_send = True
    deps['gate_factory'] = Gate
    return profile, deps, objects


def test_manual_sequence_uses_operator_control_without_camera_or_guest():
    profile, deps, objects = setup()
    control = ManualControl()
    opened = open_manual_sequence('192.0.2.1', profile, control, dependencies=deps)
    assert len(objects['command'].sent) == 1
    assert objects['writer'].sent == []  # 上传图像本身不能启用运动。
    assert opened.gate.can_send
    control.stop('Terminal Stop')
    assert not opened.gate.can_send
    opened.close()


def test_stop_before_start_cannot_be_rearmed():
    profile, deps, objects = setup()
    control = ManualControl()
    control.stop('Operator cancelled')
    with pytest.raises(RuntimeError):
        open_manual_sequence('192.0.2.1', profile, control, dependencies=deps)
    assert 'command' not in objects


def test_unreviewed_route_does_not_connect():
    profile, deps, objects = setup()
    profile['review']['path'] = False
    with pytest.raises(ValueError):
        open_manual_sequence('192.0.2.1', profile, ManualControl(), dependencies=deps)
    assert objects == {}


def test_manual_stop_during_upload_cannot_enable():
    profile, deps, objects = setup()
    control = ManualControl()
    factory = deps['command_factory']
    def command(host):
        result = factory(host)
        original = result.send
        def send(script):
            original(script)
            control.stop('Operator Stop during upload')
        result.send = send
        return result
    deps['command_factory'] = command
    with pytest.raises(RuntimeError):
        open_manual_sequence('192.0.2.1', profile, control, dependencies=deps)
    assert objects['writer'].sent == []
