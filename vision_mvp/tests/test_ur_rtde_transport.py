"""离线验证接口约定，不连接任何机器人端点。"""
import importlib
import math
import time

import pytest


def module():
    name = 'leica_tracker.ur_rtde_transport'
    assert importlib.util.find_spec(name) is not None, 'ur-rtde adapter is missing'
    return importlib.import_module(name)


class Registers:
    """模拟寄存器写入分别送达的情况，包括部分写入失败。"""
    def __init__(self, fail_at=0):
        self.values = {42: 0, 43: 0, 44: 123, 'x': -.135}
        self.snapshots = []
        self.fail_at = fail_at

    def write(self, key, value):
        if len(self.snapshots) + 1 == self.fail_at:
            return False
        self.values[key] = value
        self.snapshots.append(dict(self.values))
        return True

    def setInputIntRegister(self, key, value):
        assert key in (42, 43, 44)
        return self.write(key, value)

    def setInputDoubleRegister(self, key, value):
        assert key == 42
        return self.write('x', value)


def test_only_a_complete_target_gets_a_positive_commit_sequence():
    io = Registers()
    module().write_target(io, (7, 1, 456, -.2), 20., clock=lambda: 10.)
    assert all(s[42] <= 0 for s in io.snapshots[:-1])
    assert io.snapshots[-1] == {42: 7, 43: 1, 44: 456, 'x': -.2}


def test_scalar_recovery_hold_commits_its_flag_without_enabling_motion():
    io = Registers()
    module().write_target(io, (7, 3, 456, -.2), 20., clock=lambda: 10.)
    assert all(s[42] <= 0 for s in io.snapshots[:-1])
    assert io.snapshots[-1] == {42: 7, 43: 3, 44: 456, 'x': -.2}


def test_scalar_recovery_hold_deadline_still_prevents_positive_commit():
    io = Registers(); ticks = iter([10., 10.01, 10.02, 10.03, 10.3])
    with pytest.raises(TimeoutError):
        module().write_target(io, (7, 3, 456, -.2), 10.15, clock=lambda: next(ticks))
    assert io.values[42] < 0


@pytest.mark.parametrize('fail_at', [1, 2, 3, 4, 5])
def test_partial_library_write_never_commits_target(fail_at):
    io = Registers(fail_at)
    with pytest.raises(RuntimeError):
        module().write_target(io, (7, 1, 456, -.2), 20., clock=lambda: 10.)
    assert io.values[42] <= 0


@pytest.mark.parametrize('update', [
    (True, 1, 2, 0.), (1, True, 2, 0.), (1, 1, False, 0.),
    (1, 1, 2, math.nan), (0, 1, 2, 0.), (1, 2, 2, 0.),
])
def test_invalid_target_cannot_write_any_register(update):
    io = Registers()
    with pytest.raises(ValueError):
        module().write_target(io, update, 20., clock=lambda: 10.)
    assert not io.snapshots


def test_expiry_during_payload_write_prevents_commit():
    io = Registers()
    ticks = iter([10., 10.01, 10.02, 10.03, 10.3])
    with pytest.raises(TimeoutError):
        module().write_target(io, (7, 1, 456, -.2), 10.15, clock=lambda: next(ticks))
    assert io.values[42] < 0


class Receiver:
    def __init__(self, ticks):
        self.ticks = iter(ticks)

    def getTimestamp(self):
        return next(self.ticks)

    def getOutputIntRegister(self, register):
        return {42: 1, 43: 7, 40: 456}[register]

    def getActualTCPSpeed(self):
        return [.003, 0., 0., 0., 0., 0.]


def test_feedback_does_not_mix_two_controller_packets():
    assert module().read_feedback(Receiver([12., 12.02])) is None
    result = module().read_feedback(Receiver([12., 12.]))
    assert result == dict(timestamp=12., state=1, sequence=7, session=456,
                          tcp_speed=[.003, 0., 0., 0., 0., 0.])


def test_parent_kills_a_worker_that_never_finishes_library_startup(monkeypatch):
    adapter = module()
    real_popen = adapter.subprocess.Popen
    children = []

    def stalled_worker(command, **kwargs):
        # 验证实际进程隔离，不创建机器人客户端。
        process = real_popen([command[0], '-u', '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(adapter.subprocess, 'Popen', stalled_worker)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        adapter.RTDEChannel('unused.invalid', startup_timeout=.15)
    assert time.monotonic() - started < 3
    assert children[0].poll() is not None


def test_frozen_library_cache_cannot_refresh_parent_telemetry(monkeypatch):
    adapter = module()
    real_popen = adapter.subprocess.Popen
    program = ('import json,time; '
               'print(json.dumps({"feedback":{"timestamp":1.,"state":0,"sequence":0,'
               '"session":123,"tcp_speed":[0.]*6},"sampled_at":time.monotonic()}),flush=True); '
               'time.sleep(30)')
    monkeypatch.setattr(adapter.subprocess, 'Popen',
                        lambda command, **kwargs: real_popen([command[0], '-u', '-c', program], **kwargs))
    channel = adapter.RTDEChannel('unused.invalid')
    try:
        stamp = channel.latest[1]
        time.sleep(.2)
        assert channel.latest[1] == stamp
        with pytest.raises(RuntimeError, match='stale'):
            channel.send(1, 0, 123, 0.)
    finally:
        channel.close()


class SpatialRegisters(Registers):
    def setInputDoubleRegister(self,key,value):
        assert key in (42,43,44)
        return self.write(('xyz',key),value)


def test_xyz_recovery_hold_is_encoded_as_hold_with_all_coordinates_then_commit():
    io = SpatialRegisters()
    module().write_spatial_target(io, (7, 3, 456, [.1,.2,.3]), 20., clock=lambda: 10.)
    assert all(s[42] < 0 for s in io.snapshots[:-1])
    assert io.values[42] == 7 and io.values[43] == 3 and io.values[44] == 456
    assert [io.values[('xyz',r)] for r in (42,43,44)] == [.1,.2,.3]


def test_xyz_recovery_hold_expiry_still_leaves_invalid_sequence():
    io = SpatialRegisters(); ticks = iter([1.,1.,1.,1.,1.,1.,2.])
    with pytest.raises(TimeoutError):
        module().write_spatial_target(io,(7,3,456,[.1,.2,.3]),1.5,clock=lambda:next(ticks))
    assert io.values[42] < 0


def test_batched_xyz_worker_encodes_recovery_hold_with_same_atomic_staging():
    import struct
    from leica_tracker.atomic_rtde_transport import encode_spatial_batch
    data=encode_spatial_batch(2,(7,3,456,[.1,.2,.3]),module().validate_spatial_target)
    # 两条完整传输记录都保留 HOLD3，区别仅在提交序号的符号。
    records=[]
    while data:
        size,kind=struct.unpack('!HB',data[:3]); assert kind==85
        records.append(struct.unpack('!Biiiddd',data[3:size])); data=data[size:]
    assert records==[(2,-7,3,456,.1,.2,.3),(2,7,3,456,.1,.2,.3)]


@pytest.mark.parametrize('fail_at',range(1,8))
def test_xyz_partial_update_never_commits(fail_at):
    adapter=module()
    assert hasattr(adapter,'write_spatial_target'), 'Missing atomic XYZ writer'
    io=SpatialRegisters(fail_at)
    with pytest.raises(RuntimeError):
        adapter.write_spatial_target(io,(7,1,456,[.1,.2,.3]),20.,clock=lambda:10.)
    assert io.values[42]<=0


def test_xyz_commit_happens_after_all_three_coordinates():
    adapter=module()
    assert hasattr(adapter,'write_spatial_target'), 'Missing atomic XYZ writer'
    io=SpatialRegisters()
    adapter.write_spatial_target(io,(7,1,456,[.1,.2,.3]),20.,clock=lambda:10.)
    assert all(s[42]<0 for s in io.snapshots[:-1])
    assert io.snapshots[-1][42]==7
    assert [io.values[('xyz',r)] for r in (42,43,44)]==[.1,.2,.3]


def test_nonfinite_xyz_never_invalidates_current_packet():
    adapter=module()
    assert hasattr(adapter,'write_spatial_target'), 'Missing atomic XYZ writer'
    io=SpatialRegisters()
    with pytest.raises(ValueError):
        adapter.write_spatial_target(io,(7,1,456,[.1,float('nan'),.3]),20.,clock=lambda:10.)
    assert io.snapshots==[]


def test_read_only_worker_never_constructs_io(monkeypatch):
    import sys
    from types import SimpleNamespace
    adapter=module()
    import inspect
    assert 'receive_only' in inspect.signature(adapter.worker).parameters
    calls=[]
    class ReadOnly:
        def __init__(self,*args):calls.append('receive')
        def disconnect(self):calls.append('disconnect')
    monkeypatch.setitem(sys.modules,'rtde_receive',SimpleNamespace(RTDEReceiveInterface=ReadOnly))
    monkeypatch.setitem(sys.modules,'rtde_io',SimpleNamespace(RTDEIOInterface=lambda *args: calls.append('io')))
    monkeypatch.setattr(sys,'stdin',iter([]))
    assert adapter._sdk_worker('unused',spatial=True,receive_only=True)==0
    assert calls==['receive','disconnect']


def test_xyz_commit_deadline_and_read_only_send_are_enforced():
    adapter=module();io=SpatialRegisters()
    times=iter([1.,1.,1.,1.,1.,1.,2.])
    with pytest.raises(TimeoutError):
        adapter.write_spatial_target(io,(7,1,456,[.1,.2,.3]),1.5,clock=lambda:next(times))
    assert io.values[42]<0
    channel=object.__new__(adapter.RTDEChannel);channel.receive_only=True
    with pytest.raises(RuntimeError,match='Read-only'):channel.send(1,1,123,[0,0,0])


def test_xyz_feedback_is_coherent_and_uses_supported_session_register():
    class SpatialReceiver(Receiver):
        def getActualTCPPose(self):return [0,0,.5,0,0,0]
        def getActualQ(self):return [0]*6
    adapter=module()
    assert adapter.OUTPUT_SESSION_REGISTER==40
    assert adapter.read_feedback(SpatialReceiver([1.,2.]),spatial=True) is None
    result=adapter.read_feedback(SpatialReceiver([1.,1.]),spatial=True)
    assert result['session']==456 and result['tcp_pose'][2]==.5


def test_read_only_feedback_includes_coherent_joint_velocity():
    class SpatialReceiver(Receiver):
        def getActualTCPPose(self): return [0, 0, .5, 0, 0, 0]
        def getActualQ(self): return [0]*6
        def getActualQd(self): return [.01, 0., 0., 0., 0., 0.]
    adapter = module()
    result = adapter.read_feedback(SpatialReceiver([1., 1.]), spatial=True, joint_speeds=True)
    assert result['joint_speed'] == [.01, 0., 0., 0., 0., 0.]
    assert adapter.read_feedback(SpatialReceiver([1., 2.]), spatial=True, joint_speeds=True) is None


def test_invalid_joint_velocity_cannot_prove_standstill():
    class SpatialReceiver(Receiver):
        def getActualTCPPose(self): return [0, 0, .5, 0, 0, 0]
        def getActualQ(self): return [0]*6
        def getActualQd(self): return [float('nan')]*6
    with pytest.raises(ValueError):
        module().read_feedback(SpatialReceiver([1., 1.]), spatial=True, joint_speeds=True)
