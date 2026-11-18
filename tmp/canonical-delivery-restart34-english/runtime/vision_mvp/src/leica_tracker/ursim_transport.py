"""为明确封锁的官方URSim单独设置仅限回环传输。此模块不会发现或识别Docker容器。
其调用者必须在打开网关和每次会话之前验证所选的官方控制器、型号和隔离端口发布。这里没有任何东西可以针对物理地址。
原生工作线程、输入验证、截止日期和停止门保持不变；
只有模拟器的推进缓慢生产者时钟根据实际样本接收而不是实时漂移进行解释。"""
import json
import math
import select
import socket
from socket import IPPROTO_TCP, TCP_NODELAY
import struct
import threading
import time
from types import MappingProxyType

from .robot_state import StateReceiver, parse_state, text_marker
from .ur_rtde_transport import RTDEChannel

SIM_HOST = '127.0.0.2'
GATEWAY_HOST = '127.0.0.1'
FORWARD_PORTS = MappingProxyType({29999:39999, 30002:40002, 30004:40004, 30011:40011})


def _require_sim_host(host):
    if host != SIM_HOST:
        raise ValueError('URSim transport requires exact isolated host 127.0.0.2')


class ArrivalClock:
    """升级模拟器样本保留了实际的收据时间戳。"""
    def __init__(self):
        self.previous = -1
        self.received = -1.

    def observe(self, microseconds, received):
        if (type(microseconds) not in (int,float) or not math.isfinite(microseconds)
                or microseconds < 0 or type(received) not in (int,float)
                or not math.isfinite(received) or received < 0):
            raise ValueError('Invalid URSim producer or arrival clock')
        if microseconds <= self.previous:
            raise ValueError('URSim producer timestamp stopped advancing')
        if received < self.received:
            raise ValueError('URSim arrival clock moved backwards')
        # Windows GetTickCount64可能会给两个前进的数据包提供相同的15.625毫秒接收计数。
        # 保持实际值不变：消费者仍然等待更新的印章，并强制执行相同的过期日期。
        self.previous, self.received = microseconds, received
        return received


class SimStateReceiver(StateReceiver):
    """具有仅模拟器接收时间表的原生主接收器生命周期。"""
    def __init__(self, host):
        _require_sim_host(host)
        super().__init__(host)

    def _read(self):
        clock = ArrivalClock()
        try:
            while not self.closed:
                size, kind = struct.unpack('!IB', self._exact(5))
                if not 5 <= size <= 65536:
                    raise ValueError('Invalid state message length')
                body = self._exact(size-5)
                if kind == 20:
                    marker = text_marker(body)
                    if marker and marker.startswith('LEICA_FAULT_'):
                        self.notice = marker
                    elif marker:
                        self.ack = marker, time.monotonic()
                if kind != 16:
                    continue
                current = parse_state(body)
                stamp = clock.observe(current['mode']['timestamp'], time.monotonic())
                self.latest = current, stamp
                self.sock.settimeout(.5)
        except Exception as error:
            if not self.closed:
                phase = 'startup' if self.latest[0] is None else 'runtime'
                self.failure = f'Robot state {phase}: {error}'


class SimRTDEChannel(RTDEChannel):
    """原生一次性工作线程/写入/清理路径；受限的接收新鲜度。"""
    def __init__(self, host, startup_timeout=8., spatial=False, receive_only=False):
        _require_sim_host(host)
        super().__init__(host, startup_timeout=startup_timeout,
                         spatial=spatial, receive_only=receive_only)

    def _read(self):
        clock = ArrivalClock()
        try:
            while line := self.process.stdout.readline(8193):
                if len(line)>8192 or not line.endswith('\n'):
                    raise ValueError('Invalid RTDE worker output')
                if not line.startswith('{'):
                    continue
                message = json.loads(line)
                if 'error' in message:
                    raise RuntimeError(message['error'])
                feedback, sampled = message['feedback'], message['sampled_at']
                now = time.monotonic()
                if (type(sampled) not in (int,float) or not math.isfinite(sampled)
                        or not 0 <= now-sampled <= .15):
                    raise ValueError('Invalid or stale URSim worker clock')
                stamp = clock.observe(round(feedback['timestamp']*1_000_000), sampled)
                self.latest = feedback, stamp
                self.ready.set()
            if not self.closed:
                raise RuntimeError('ur-rtde worker exited')
        except Exception as error:
            if not self.closed:
                self.failure = str(error)
            self.ready.set()


def dependencies():
    """原生受保护的引导工厂只有两个模拟器接收器。"""
    from .physical_resident_bootstrap import _dependencies
    from .horizontal_follow import DashboardReceiver
    result = _dependencies()
    result.update(state_factory=SimStateReceiver, rtde_factory=SimRTDEChannel,
                  dashboard_factory=DashboardReceiver)
    return result


class LoopbackGateway:
    """拥有四个确切的监听器，并仅转发到固定的本地端口。构造不执行I/O。在接受任何连接之前，必须将每个监听器绑定。
    占用的端口会导致整个网关失败；没有地址/端口备用、重新连接或自动控制器命令。"""
    def __init__(self, bind_host=SIM_HOST, ports=None):
        _require_sim_host(bind_host)
        if ports is not None and ports != FORWARD_PORTS:
            raise ValueError('URSim gateway requires the exact isolated four-port mapping')
        self._lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleanup_errors = []
        self._closed = threading.Event()
        self._started = False
        self._listeners = []
        self._connections = set()
        self._threads = []
        self.failure = None

    @property
    def bind_host(self):
        return SIM_HOST

    @property
    def ports(self):
        return FORWARD_PORTS

    def __enter__(self):
        if self._started or self._closed.is_set():
            raise RuntimeError('URSim gateway cannot be reopened')
        self._started = True
        try:
            for local_port, remote_port in FORWARD_PORTS.items():
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._listeners.append(listener)
                if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind((SIM_HOST, local_port))
                listener.listen(8)
                listener.settimeout(.2)
            with self._lock:
                for listener, remote_port in zip(self._listeners, FORWARD_PORTS.values()):
                    thread = threading.Thread(target=self._accept, args=(listener,remote_port), daemon=True)
                    thread.start()
                    self._threads.append(thread)
            return self
        except BaseException:
            self.close()
            raise

    def _own(self, connection):
        with self._lock:
            if not self._closed.is_set():
                self._connections.add(connection)
                return True
        self._close_socket(connection)
        return False

    def _close_socket(self, connection):
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        except Exception as error:
            with self._cleanup_lock:
                self._cleanup_errors.append(error)
        finally:
            try:
                connection.close()
            except Exception as error:
                with self._cleanup_lock:
                    self._cleanup_errors.append(error)

    def _shutdown(self):
        self._closed.set()
        with self._lock:
            owned = [*self._listeners, *self._connections]
        for connection in owned:
            self._close_socket(connection)

    def _fail(self, error):
        with self._lock:
            if self.failure is None:
                self.failure = str(error)
        self._shutdown()

    def _accept(self, listener, remote_port):
        try:
            while not self._closed.is_set():
                try:
                    client, _address = listener.accept()
                except socket.timeout:
                    continue
                if not self._own(client):
                    return
                with self._lock:
                    if self._closed.is_set():
                        self._close_socket(client)
                        self._connections.discard(client)
                        return
                    thread = threading.Thread(target=self._relay, args=(client,remote_port), daemon=True)
                    thread.start()
                    self._threads.append(thread)
        except Exception as error:
            if not self._closed.is_set():
                self._fail(error)

    def _relay(self, client, remote_port):
        upstream = None
        try:
            if remote_port not in FORWARD_PORTS.values():
                raise ValueError('Unsupported URSim gateway destination')
            if self._closed.is_set():
                return
            upstream = socket.create_connection((GATEWAY_HOST, remote_port), timeout=.5)
            if not self._own(upstream):
                return
            # RTDE/注册流量由小型、时间敏感的数据包组成。两端都必须避免通过此额外的代理进行Nagle批处理。
            client.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
            upstream.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
            client.settimeout(.2)
            upstream.settimeout(.2)
            while not self._closed.is_set():
                readable, _, _ = select.select((client,upstream), (), (), .1)
                for source in readable:
                    payload = source.recv(65536)
                    if not payload:
                        return
                    (upstream if source is client else client).sendall(payload)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # 当输入/接收通道退出时，原生SDK工作者会被终止。
            # 它们的TCP重置相当于配对本地EOF，而不是监听器或无关/新会话通道的故障。
            # 如果上游在运行期间重置，关闭此配对仍然会使该会话的未更改的电测/ACK警报停止它。
            pass
        except Exception as error:
            if not self._closed.is_set():
                self._fail(error)
        finally:
            for connection in (client,upstream):
                if connection is not None:
                    self._close_socket(connection)
                    with self._lock:
                        self._connections.discard(connection)

    def close(self):
        self._shutdown()
        with self._lock:
            threads = tuple(self._threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1.)
        if any(thread is not threading.current_thread() and thread.is_alive() for thread in threads):
            raise RuntimeError('URSim gateway cleanup incomplete')
        if self._cleanup_errors:
            raise RuntimeError('URSim gateway cleanup incomplete') from self._cleanup_errors[0]

    def __exit__(self, *_exc):
        self.close()
