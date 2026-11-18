"""验证工作线程观察到的确认回执，所有输入和接口均离线。"""
import json
import math
import sys
from types import SimpleNamespace

import pytest

from leica_tracker import ur_rtde_transport as transport


def receipt_clock():
    assert hasattr(transport, 'AckReceiptClock'), 'First worker ACK receipt is not retained'
    return transport.AckReceiptClock()


def feedback(session=123, sequence=7, state=1, timestamp=4.0):
    return dict(session=session, sequence=sequence, state=state, timestamp=timestamp,
                tcp_speed=[0.0] * 6)


def test_newer_feedback_and_state_change_retain_first_matching_ack_receipt():
    clock = receipt_clock()
    original = feedback()
    first = clock.observe(original, 10.0)
    later = clock.observe(feedback(state=0, timestamp=4.02), 10.14)
    assert first['ack_sampled_at'] == 10.0
    assert first['worker_sampled_at'] == 10.0
    assert later['ack_sampled_at'] == 10.0
    assert later['worker_sampled_at'] == 10.14
    assert later['state'] == 0 and later['timestamp'] == 4.02
    assert 'ack_sampled_at' not in original and 'worker_sampled_at' not in original


@pytest.mark.parametrize('next_pair', [(123, 8), (456, 7), (0, 0)])
def test_new_sequence_or_session_gets_its_own_first_observation(next_pair):
    clock = receipt_clock()
    clock.observe(feedback(), 10.0)
    result = clock.observe(feedback(*next_pair), 10.02)
    assert result['ack_sampled_at'] == 10.02


@pytest.mark.parametrize('pair', [
    (True, 7), (123, False), (123.0, 7), (123, 7.0),
    (-1, 7), (123, -1), (2**31, 7), (123, 2**31),
])
def test_invalid_ack_identity_cannot_create_a_receipt(pair):
    clock = receipt_clock()
    with pytest.raises(ValueError):
        clock.observe(feedback(*pair), 10.0)


@pytest.mark.parametrize('sampled_at', [True, '10', -1.0, math.nan, math.inf, -math.inf])
def test_invalid_observation_clock_cannot_create_a_receipt(sampled_at):
    with pytest.raises(ValueError):
        receipt_clock().observe(feedback(), sampled_at)


def test_backwards_clock_is_rejected_without_replacing_first_receipt():
    clock = receipt_clock()
    clock.observe(feedback(), 10.0)
    with pytest.raises(ValueError):
        clock.observe(feedback(sequence=8), 9.99)
    assert clock.observe(feedback(timestamp=4.02), 10.02)['ack_sampled_at'] == 10.0


def test_equal_windows_ticks_do_not_invent_an_earlier_or_later_receipt():
    clock = receipt_clock()
    assert clock.observe(feedback(), 10.0)['ack_sampled_at'] == 10.0
    assert clock.observe(feedback(timestamp=4.02), 10.0)['ack_sampled_at'] == 10.0
    assert clock.observe(feedback(sequence=8, timestamp=4.04), 10.0)['ack_sampled_at'] == 10.0


@pytest.mark.parametrize('field', ['ack_sampled_at', 'worker_sampled_at'])
def test_caller_cannot_supply_fabricated_worker_metadata(field):
    clock = receipt_clock()
    forged = dict(feedback(), **{field: 1.0})
    with pytest.raises(ValueError):
        clock.observe(forged, 10.0)
    assert clock.observe(feedback(), 10.1)['ack_sampled_at'] == 10.1


@pytest.mark.parametrize('receive_only', [False, True])
def test_actual_worker_attaches_first_receipt_only_to_coherent_advancing_feedback(
        monkeypatch, capsys, receive_only):
    # 运行实际工作逻辑及在读数前后核对时间戳的读取器。只替换原生 SDK 和调度边界，避免创建套接字或线程。
    frames = [
        (4.00, 0, 2, 10.00, 4.00),
        (4.02, 1, 0, 10.04, 4.02),
        (4.02, 1, 0, 10.06, 4.02),  # 来自生产端缓存的数据包。
        (4.03, 2, 0, 10.08, 4.04),  # 分次读取导致内容不一致的数据包不能证明序号 2 已获得确认。
        (4.04, 1, 1, 10.14, 4.04),
        (4.06, 2, 0, 10.16, 4.06),
    ]
    position = [0]
    timestamp_calls = [0]
    calls = []

    class Event:
        def is_set(self): return position[0] >= len(frames)
        def set(self): position[0] = len(frames)
        def wait(self, seconds):
            assert seconds == .005
            position[0] += 1
            timestamp_calls[0] = 0

    class Thread:
        def __init__(self, *, target, daemon):
            assert callable(target) and daemon is True
        def start(self): pass

    class Receive:
        def __init__(self, host, frequency, outputs, verbose, upper):
            assert host == 'unused.invalid' and frequency == 50.0
            assert verbose is False and upper is True
            assert outputs[-1] == ('actual_qd' if receive_only else 'actual_q')
            calls.append('receive')
        def isConnected(self): return True
        def getTimestamp(self):
            timestamp_calls[0] += 1
            return frames[position[0]][0 if timestamp_calls[0] == 1 else 4]
        def getOutputIntRegister(self, register):
            return {40: 123, 42: frames[position[0]][2], 43: frames[position[0]][1]}[register]
        def getActualTCPSpeed(self): return [0.0] * 6
        def getActualTCPPose(self): return [0.0, 0.0, .5, 0.0, 0.0, 0.0]
        def getActualQ(self): return [0.0] * 6
        def getActualQd(self): return [0.0] * 6
        def disconnect(self): calls.append('receive_closed')

    class IO:
        def __init__(self, host, verbose, upper):
            assert host == 'unused.invalid' and verbose is False and upper is True
            calls.append('io')
        def isConnected(self): return True
        def disconnect(self): calls.append('io_closed')

    monkeypatch.setitem(sys.modules, 'rtde_receive', SimpleNamespace(RTDEReceiveInterface=Receive))
    monkeypatch.setitem(sys.modules, 'rtde_io', SimpleNamespace(RTDEIOInterface=IO))
    monkeypatch.setattr(transport, 'threading', SimpleNamespace(Event=Event, Thread=Thread))
    monkeypatch.setattr(transport, 'time', SimpleNamespace(monotonic=lambda: frames[position[0]][3]))
    assert transport._sdk_worker('unused.invalid', spatial=True, receive_only=receive_only) == 0
    emitted = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item['sampled_at'] for item in emitted] == [10.0, 10.04, 10.14, 10.16]
    assert [item['feedback'].get('ack_sampled_at') for item in emitted] == [10.0, 10.04, 10.04, 10.16]
    assert [item['feedback'].get('worker_sampled_at') for item in emitted] == [10.0, 10.04, 10.14, 10.16]
    assert calls == (['receive', 'receive_closed'] if receive_only else
                     ['receive', 'io', 'io_closed', 'receive_closed'])
