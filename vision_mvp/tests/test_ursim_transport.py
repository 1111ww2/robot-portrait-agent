"""仅测试 URSim 传输逻辑，不创建实际套接字、Docker 进程、SDK 工作线程或运动。"""
import io
import json
import struct
import threading
from types import SimpleNamespace

import pytest


def module():
    from leica_tracker import ursim_transport
    return ursim_transport


@pytest.mark.parametrize('host', ['127.0.0.1', 'localhost', '192.0.2.1', '::1', None])
def test_sim_factories_reject_every_other_host_before_native_construction(monkeypatch, host):
    sim = module()
    monkeypatch.setattr(sim.StateReceiver, '__init__', lambda *_a, **_k: pytest.fail('native state opened'))
    monkeypatch.setattr(sim.RTDEChannel, '__init__', lambda *_a, **_k: pytest.fail('native RTDE opened'))
    for factory in (sim.SimStateReceiver, sim.SimRTDEChannel):
        with pytest.raises(ValueError, match='127.0.0.2'):
            factory(host)


def test_sim_factories_reuse_native_lifecycle_with_exact_arguments(monkeypatch):
    sim = module()
    calls = []
    monkeypatch.setattr(sim.StateReceiver, '__init__', lambda _self, host: calls.append(('state', host)))
    monkeypatch.setattr(sim.RTDEChannel, '__init__',
                        lambda _self, host, **kwargs: calls.append(('rtde', host, kwargs)))
    sim.SimStateReceiver('127.0.0.2')
    sim.SimRTDEChannel('127.0.0.2', startup_timeout=3., spatial=True, receive_only=True)
    assert calls == [('state', '127.0.0.2'), ('rtde', '127.0.0.2',
                      {'startup_timeout': 3., 'spatial': True, 'receive_only': True})]
    assert sim.SimRTDEChannel.send is sim.RTDEChannel.send
    assert sim.SimRTDEChannel._write is sim.RTDEChannel._write
    assert sim.SimRTDEChannel.close is sim.RTDEChannel.close
    assert sim.SimStateReceiver.close is sim.StateReceiver.close


def test_sim_arrival_clock_allows_slow_simulation_without_refreshing_repeated_samples():
    clock = module().ArrivalClock()
    assert clock.observe(1_000_000, 10.) == 10.
    assert clock.observe(1_010_000, 10.1) == 10.1
    assert clock.observe(1_020_000, 10.2) == 10.2
    with pytest.raises(ValueError, match='advanc'):
        clock.observe(1_020_000, 10.3)


def test_physical_producer_clock_keeps_accumulated_delay_behavior():
    from leica_tracker.robot_state import ProducerClock
    module()
    native = ProducerClock()
    assert native.observe(1_000_000, 10.) == 10.
    assert native.observe(1_010_000, 10.1) == pytest.approx(10.01)


@pytest.mark.parametrize('producer,arrival', [(1_000_001, 9.999), (999_999, 10.1),
                                            (float('nan'), 10.1), (1_000_001, float('inf'))])
def test_arrival_clock_rejects_invalid_or_nonadvancing_samples(producer, arrival):
    clock = module().ArrivalClock()
    clock.observe(1_000_000, 10.)
    with pytest.raises(ValueError):
        clock.observe(producer, arrival)


def test_advancing_producer_can_share_coarse_arrival_without_faking_a_new_timestamp():
    clock = module().ArrivalClock()
    first = clock.observe(2_594_046_000, 148571.187)
    second = clock.observe(2_594_066_000, 148571.187)
    assert first == second == 148571.187
    with pytest.raises(ValueError):
        clock.observe(2_594_086_000, 148571.186)
    with pytest.raises(ValueError, match='advanc'):
        clock.observe(2_594_066_000, 148571.203)


def rtde_reader(messages, monkeypatch, now):
    sim = module()
    monkeypatch.setattr(sim, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    channel = object.__new__(sim.SimRTDEChannel)
    lines = ''.join(json.dumps(message)+'\n' for message in messages)
    channel.process = SimpleNamespace(stdout=io.StringIO(lines))
    channel.closed = False
    channel.failure = None
    channel.latest = (None, 0.)
    channel.ready = threading.Event()
    channel._read()
    return channel


def feedback(timestamp, sampled):
    return {'feedback': {'timestamp': timestamp, 'state': 100, 'sequence': 0,
                         'session': 7, 'tcp_speed': [0.]*6}, 'sampled_at': sampled}


def test_sim_rtde_stamps_worker_actual_arrival_and_accepts_slow_clock(monkeypatch):
    channel = rtde_reader([feedback(1., 10.), feedback(1.01, 10.1)], monkeypatch, [10.12])
    assert channel.latest[0]['timestamp'] == 1.01 and channel.latest[1] == 10.1
    assert channel.failure == 'ur-rtde worker exited'


@pytest.mark.parametrize('sampled', [9.849, 10.001, float('nan')])
def test_sim_rtde_rejects_stale_queued_or_future_worker_arrival(monkeypatch, sampled):
    channel = rtde_reader([feedback(1., sampled)], monkeypatch, [10.])
    assert channel.latest == (None, 0.)
    assert channel.ready.is_set() and 'clock' in channel.failure


def test_sim_rtde_repeated_producer_cannot_refresh_latest(monkeypatch):
    channel = rtde_reader([feedback(1., 10.), feedback(1., 10.1)], monkeypatch, [10.12])
    assert channel.latest[1] == 10. and 'advanc' in channel.failure


def test_sim_rtde_keeps_same_actual_receipt_for_two_advancing_coarse_clock_packets(monkeypatch):
    channel = rtde_reader([feedback(1., 10.), feedback(1.02, 10.)], monkeypatch, [10.12])
    assert channel.latest[0]['timestamp'] == 1.02 and channel.latest[1] == 10.
    assert channel.failure == 'ur-rtde worker exited'


@pytest.mark.parametrize('second_timestamp,expected_stamp', [(1_010_000,10.1), (1_000_000,10.)])
def test_sim_primary_requires_advancing_producer_and_stamps_actual_arrival(monkeypatch, second_timestamp, expected_stamp):
    sim = module()
    arrival = iter([10.,10.1])
    states = iter([{'mode':{'timestamp':1_000_000}}, {'mode':{'timestamp':second_timestamp}}])
    chunks = iter([struct.pack('!IB',6,16), b'a', struct.pack('!IB',6,16), b'b'])
    source = object.__new__(sim.SimStateReceiver)
    source.closed=False;source.failure=None;source.latest=(None,0.);source.sock=FakeSocket()
    def exact(_size):
        try:return next(chunks)
        except StopIteration:raise EOFError('fake stream ended') from None
    source._exact=exact
    monkeypatch.setattr(sim,'parse_state',lambda _body:next(states))
    monkeypatch.setattr(sim,'time',SimpleNamespace(monotonic=lambda:next(arrival)))
    source._read()
    assert source.latest[1] == expected_stamp
    assert 'runtime' in source.failure
    if second_timestamp == 1_000_000:
        assert 'advanc' in source.failure


def test_dependencies_retain_native_controls_and_add_dashboard_factory():
    from leica_tracker.physical_resident_bootstrap import _dependencies
    from leica_tracker.horizontal_follow import DashboardReceiver
    sim = module()
    native = _dependencies()
    adapted = sim.dependencies()
    assert adapted['state_factory'] is sim.SimStateReceiver
    assert adapted['rtde_factory'] is sim.SimRTDEChannel
    assert adapted['dashboard_factory'] is DashboardReceiver
    for key in ('command_factory', 'dashboard_check', 'gate_factory', 'watch_factory', 'clock', 'sleep'):
        assert adapted[key] is native[key]
    assert callable(adapted['session_factory'])


class FakeSocket:
    def __init__(self, fail_port=None):
        self.fail_port = fail_port
        self.bound = None
        self.closed = False
        self.shutdowns = []
        self.options = []
    def setsockopt(self, *args): self.options.append(args)
    def settimeout(self, _value): pass
    def bind(self, address):
        self.bound = address
        if address[1] == self.fail_port:
            raise OSError('occupied')
    def listen(self, _backlog): pass
    def shutdown(self, how): self.shutdowns.append(how)
    def close(self): self.closed = True


class IdleThread:
    def __init__(self, **kwargs): self.target = kwargs['target']; self.started = False
    def start(self): self.started = True
    def join(self, timeout): pass
    def is_alive(self): return False


def fake_gateway_runtime(monkeypatch, fail_port=None):
    sim = module()
    sockets = []
    def socket_factory(*_args):
        result = FakeSocket(fail_port)
        sockets.append(result)
        return result
    monkeypatch.setattr(sim, 'socket', SimpleNamespace(socket=socket_factory, AF_INET=2,
                       SOCK_STREAM=1, SHUT_RDWR=2, timeout=TimeoutError))
    monkeypatch.setattr(sim, 'threading', SimpleNamespace(Event=threading.Event, Lock=threading.Lock,
                       Thread=IdleThread, current_thread=threading.current_thread))
    return sim, sockets


@pytest.mark.parametrize('kwargs', [{'bind_host':'127.0.0.1'}, {'bind_host':'0.0.0.0'},
                                  {'ports':{29999:29999}}, {'ports':{29999:39999}}])
def test_gateway_rejects_nonisolated_bind_or_changed_ports_before_socket(monkeypatch, kwargs):
    sim, sockets = fake_gateway_runtime(monkeypatch)
    with pytest.raises(ValueError):
        sim.LoopbackGateway(**kwargs)
    assert sockets == []


def test_gateway_binds_all_four_exact_loopback_ports_and_closes_owned_sockets(monkeypatch):
    sim, sockets = fake_gateway_runtime(monkeypatch)
    gateway = sim.LoopbackGateway()
    assert sockets == []
    with gateway:
        assert {item.bound for item in sockets} == {('127.0.0.2', port) for port in (29999, 30002, 30004, 30011)}
        assert dict(gateway.ports) == {29999:39999, 30002:40002, 30004:40004, 30011:40011}
        client, upstream = FakeSocket(), FakeSocket()
        assert gateway._own(client) and gateway._own(upstream)
    assert all(item.closed for item in [*sockets, client, upstream])
    late = FakeSocket()
    assert not gateway._own(late) and late.closed
    with pytest.raises(RuntimeError):
        gateway.__enter__()


def test_gateway_partial_bind_failure_closes_every_listener_without_fallback(monkeypatch):
    sim, sockets = fake_gateway_runtime(monkeypatch, fail_port=30004)
    gateway = sim.LoopbackGateway()
    with pytest.raises(OSError, match='occupied'):
        gateway.__enter__()
    assert [item.bound for item in sockets] == [('127.0.0.2',29999), ('127.0.0.2',30002), ('127.0.0.2',30004)]
    assert all(item.closed for item in sockets)


def test_gateway_relay_connects_only_to_fixed_local_destination_and_copies_both_directions(monkeypatch):
    sim = module()
    gateway = sim.LoopbackGateway()
    calls = []
    class Stream(FakeSocket):
        def __init__(self, chunks): super().__init__(); self.chunks = iter(chunks); self.sent = []
        def recv(self, _size): return next(self.chunks)
        def sendall(self, data): self.sent.append(data)
    client, upstream = Stream([b'command',b'']), Stream([b'feedback'])
    assert gateway._own(client)
    def connect(address, timeout): calls.append(address); return upstream
    monkeypatch.setattr(sim, 'socket', SimpleNamespace(create_connection=connect, SHUT_RDWR=2))
    ready = iter([([client,upstream],[],[]), ([client],[],[])])
    monkeypatch.setattr(sim, 'select', SimpleNamespace(select=lambda *_args: next(ready)))
    gateway._relay(client, 40002)
    assert calls == [('127.0.0.1',40002)]
    assert client.sent == [b'feedback'] and upstream.sent == [b'command']
    import socket
    assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in client.options
    assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in upstream.options
    assert client.closed and upstream.closed


def test_gateway_cleanup_attempts_every_owned_socket_and_reports_close_failure(monkeypatch):
    sim, _sockets = fake_gateway_runtime(monkeypatch)
    gateway = sim.LoopbackGateway()
    class BrokenClose(FakeSocket):
        def close(self):
            self.closed = True
            raise OSError('fake close failed')
    broken, remaining = BrokenClose(), FakeSocket()
    gateway._listeners = [broken, remaining]
    with pytest.raises(RuntimeError, match='cleanup incomplete'):
        gateway.close()
    assert broken.closed and remaining.closed


def test_gateway_relay_rejects_unknown_destination_without_connecting(monkeypatch):
    sim = module()
    gateway = sim.LoopbackGateway()
    client = FakeSocket()
    gateway._own(client)
    monkeypatch.setattr(sim, 'socket', SimpleNamespace(SHUT_RDWR=2,
                        create_connection=lambda *_a, **_k:pytest.fail('unexpected destination')))
    gateway._relay(client, 30002)
    assert client.closed and 'destination' in gateway.failure


def test_gateway_thread_start_failure_preserves_error_and_closes_all_bound_listeners(monkeypatch):
    sim, sockets = fake_gateway_runtime(monkeypatch)
    class FailedStart(IdleThread):
        count = 0
        def start(self):
            type(self).count += 1
            if self.count == 2:
                raise RuntimeError('thread capacity')
            super().start()
        def join(self, timeout):
            if not self.started:
                raise RuntimeError('unstarted joined')
    sim.threading.Thread = FailedStart
    with pytest.raises(RuntimeError, match='thread capacity'):
        sim.LoopbackGateway().__enter__()
    assert len(sockets) == 4 and all(item.closed for item in sockets)


@pytest.mark.parametrize('retired_side', ['client_reset', 'upstream_reset', 'broken_pipe'])
def test_retired_sdk_connection_closes_only_its_pair_and_keeps_gateway_usable(monkeypatch, retired_side):
    sim = module()
    gateway = sim.LoopbackGateway()
    listeners = [FakeSocket() for _ in range(4)]
    gateway._listeners = listeners
    unrelated = FakeSocket()
    gateway._own(unrelated)
    class Stream(FakeSocket):
        def __init__(self, side): super().__init__(); self.side = side
        def recv(self, _size):
            if retired_side == self.side+'_reset':
                raise ConnectionResetError('SDK connection retired')
            return b'packet'
        def sendall(self, _data):
            if retired_side == 'broken_pipe':
                raise BrokenPipeError('SDK connection retired')
    client, upstream = Stream('client'), Stream('upstream')
    gateway._own(client)
    monkeypatch.setattr(sim, 'socket', SimpleNamespace(SHUT_RDWR=2,
                        create_connection=lambda *_a, **_k:upstream))
    ready = upstream if retired_side == 'upstream_reset' else client
    monkeypatch.setattr(sim, 'select', SimpleNamespace(select=lambda *_args:([ready],[],[])))
    gateway._relay(client,40002)
    assert client.closed and upstream.closed
    assert not unrelated.closed and not any(listener.closed for listener in listeners)
    assert gateway.failure is None and not gateway._closed.is_set()
    next_client = FakeSocket()
    assert gateway._own(next_client)
    gateway.close()
    assert unrelated.closed and next_client.closed


def test_gateway_refused_upstream_still_latches_failure_and_closes_all_owned_sockets(monkeypatch):
    sim = module()
    gateway = sim.LoopbackGateway()
    listener, client = FakeSocket(), FakeSocket()
    gateway._listeners = [listener]
    gateway._own(client)
    def refused(*_args, **_kwargs):
        raise ConnectionRefusedError('fixed upstream unavailable')
    monkeypatch.setattr(sim,'socket',SimpleNamespace(SHUT_RDWR=2,create_connection=refused))
    gateway._relay(client,40002)
    assert gateway._closed.is_set() and 'upstream unavailable' in gateway.failure
    assert client.closed and listener.closed
