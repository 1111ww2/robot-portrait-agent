"""用于明确创建的RTDE居民的独立客机控制监督机构。这是应用程序级监督，而不是紧急停止或安全等级功能。
“ResidentMotionGate”必须是访问居民RTDE输入通道的*唯一*路径。
在客机停止、时钟刻度变化、缺失或被阻止的快照或发送失败时，它会锁定关闭，并异步关闭通道。
兼容的机器人居民程序必须单独在缺失心跳时停止；此模块无法证明它确实如此。
在使用它来控制机器人之前，请在实际安装中验证居民和硬件停止行为。
明确提供的通信恢复策略可能会暂停启用的写入，而无需关闭此通道。这需要具有native flag3持有的居民；
它既不会刷新所有者/相机时间戳，也不会放松机器人ACK监督。
停止共享写入/反馈通道也会终止其反馈：确认停止需要单独的独立仅接收遥测连接。
呼叫者拥有该通道，除通过此门户外，不得向其发送。"""
import math
import http.client
import io
import json
import socket
import sys
import threading
import time
from types import MappingProxyType


def _snapshot_diagnostics(value, require_camera):
    """仅限受限字段名称和数字标志；永远不要保留原始身份。"""
    data = value if isinstance(value, dict) else {}
    identity_valid = (isinstance(value, dict) and isinstance(data.get('instance'), str)
                      and 1 <= len(data['instance']) <= 128
                      and type(data.get('epoch')) is int and 0 <= data['epoch'] < 2**53)
    flags = {'identity_valid': int(identity_valid), 'require_camera': int(require_camera)}
    for field in ('active', 'control_fresh', 'camera_connected', 'camera_fresh'):
        current = data.get(field)
        flags[field] = (1 if current is True else 0 if current is False
                        else -1 if field not in data else -2)
    failed = [] if identity_valid else ['invalid_identity']
    required = ('active', 'control_fresh') + (('camera_connected', 'camera_fresh') if require_camera else ())
    failed.extend(field+'_not_true' for field in required if data.get(field) is not True)
    return failed, flags


class GuestSnapshotRequestError(RuntimeError):
    """仅固定分类；从不包括令牌、URL或原始正文。"""
    def __init__(self,category,phase,elapsed_ms,status=None):
        self.category=category;self.phase=phase;self.elapsed_ms=round(elapsed_ms,3)
        self.status=status
        fields=dict(category=category,phase=phase,elapsed_ms=self.elapsed_ms)
        if status is not None:fields['status']=status
        super().__init__('Local guest control snapshot unavailable '+json.dumps(fields,sort_keys=True))


def _remaining(deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Control request deadline expired')
    return remaining


class _DeadlineRaw(io.RawIOBase):
    def __init__(self,sock,deadline):
        super().__init__();self._socket=sock;self._deadline=deadline
        # 即使在HTTP在其响应正文被消耗之前选择Connection:close，makefile也会正确拥有套接字引用。
        self._raw=sock.makefile('rb',buffering=0)
    def readable(self):return True
    def readinto(self,buffer):
        self._socket.settimeout(_remaining(self._deadline))
        return self._raw.readinto(buffer)
    def close(self):
        try:self._raw.close()
        finally:super().close()


class _DeadlineSocket:
    def __init__(self,sock,deadline):self._socket=sock;self._deadline=deadline
    def makefile(self,mode):
        if mode!='rb':raise ValueError('Read-only HTTP response required')
        return io.BufferedReader(_DeadlineRaw(self._socket,self._deadline))


class _DeadlineConnection(http.client.HTTPConnection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.deadline=None
        self.response_class=self._response
    def _response(self,sock,*args,**kwargs):
        return http.client.HTTPResponse(_DeadlineSocket(sock,self.deadline),*args,**kwargs)
    def connect(self):
        self.timeout=_remaining(self.deadline);super().connect()
        self.sock.settimeout(_remaining(self.deadline))
    def send(self,data):
        if self.sock is None:self.connect()
        self.sock.settimeout(_remaining(self.deadline));super().send(data)


class LocalGuestControlReader:
    """通过IPv4循环回环读取配对的客户端控制快照，无需重定向。配对令牌保留在请求头中。没有URL或异常包含它。
    有限的HTTP超时和大小限制补充，而不是取代，独立的守护程序超时。它从不刷新客户端所有者的租约。"""
    def __init__(self, port, token, *, timeout):
        if (type(port) is not int or not 1 <= port <= 65535
                or not isinstance(token, str) or not 1 <= len(token) <= 512
                or any(ord(c) < 33 or ord(c) > 126 for c in token)
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Valid loopback port, token and timeout required')
        self._port = port
        self._token = token
        self._timeout = timeout
        self._local=threading.local();self._connections=set();self._connection_lock=threading.Lock()
        self._closed=False

    def close_current_thread(self):
        conn=getattr(self._local,'conn',None);self._local.conn=None
        if conn is not None:
            with self._connection_lock:self._connections.discard(conn)
            conn.close()

    def close(self):
        with self._connection_lock:
            self._closed=True;connections=tuple(self._connections);self._connections.clear()
        for conn in connections:
            if conn.sock is not None:
                try:conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
            conn.close()

    def __call__(self):
        started=time.monotonic();deadline=started+self._timeout;phase='request';retained=False
        with self._connection_lock:
            if self._closed:raise GuestSnapshotRequestError('closed',phase,0.)
            conn=getattr(self._local,'conn',None)
            if conn is None:
                conn=_DeadlineConnection('127.0.0.1',self._port,timeout=self._timeout)
                self._local.conn=conn;self._connections.add(conn)
        conn.deadline=deadline
        try:
            conn.request('GET', '/operator-state', headers={'X-Guest-Token': self._token,
                                                             'Connection': 'keep-alive'})
            phase='headers'
            if conn.sock is not None:conn.sock.settimeout(_remaining(deadline))
            response = conn.getresponse()
            if response.status != 200:
                raise GuestSnapshotRequestError('http_rejected',phase,(time.monotonic()-started)*1000,response.status)
            phase='body'
            body = response.read(262145)
            if len(body) > 262144:
                raise ValueError('Control response too large')
            if response.length not in (None,0):raise ValueError('Incomplete control response')
            if not response.isclosed() and (response.read(1) or not response.isclosed()):
                raise ValueError('Incomplete control response')
            value = json.loads(body)
            if (not isinstance(value, dict) or not isinstance(value.get('control'), dict)
                    or value.get('control_action') not in ('start', 'restart', 'stop')
                    or type(value.get('control_started_ns')) is not int or value['control_started_ns'] <= 0):
                raise ValueError('Control response invalid')
            _remaining(deadline)
            retained=not response.will_close and not self._closed
            snapshot = {'instance': value['control'].get('instance'),
                    'epoch': value['control'].get('epoch'), 'active': value.get('active'),
                    'camera_connected': value.get('camera_connected'),
                    'camera_fresh': value.get('camera_fresh'),
                    'control_fresh': value.get('control_fresh'),
                    'control_action': value['control_action'], 'started_ns': value['control_started_ns']}
            if 'communication' in value:
                snapshot['communication'] = value['communication']
            return snapshot
        except GuestSnapshotRequestError:raise
        except TimeoutError:
            raise GuestSnapshotRequestError('timeout',phase,(time.monotonic()-started)*1000) from None
        except OSError:
            raise GuestSnapshotRequestError('connection',phase,(time.monotonic()-started)*1000) from None
        except http.client.HTTPException:
            raise GuestSnapshotRequestError('protocol',phase,(time.monotonic()-started)*1000) from None
        except (ValueError,TypeError):
            raise GuestSnapshotRequestError('response',phase,(time.monotonic()-started)*1000) from None
        finally:
            if not retained:self.close_current_thread()


class ResidentMotionGate:
    """Gate RTDE写入，替换可选保留，或永久停止它们。
    RTDEChannel.send是一种受限的队列操作，而不是阻塞的网络写入。
    在构建此门槛之前，通道必须完全初始化。异步关闭故意不被视为机器人停止ACK。"""
    def __init__(self, channel):
        self._channel = channel
        self._lock = threading.Lock()
        self._armed = False
        self._stopped = False
        self._paused = False
        self.pause_reason = None
        self.reason = None
        self.close_error = None
        self.close_details = None
        self.close_finished = threading.Event()

    def arm(self):
        with self._lock:
            if self._stopped:
                raise RuntimeError('Resident motion gate is stopped')
            self._armed = True

    @property
    def can_send(self):
        """一个短暂的局部锁定状态，而不是机器人或相机安全证明。"""
        with self._lock:
            return self._armed and not self._stopped

    @property
    def is_paused(self):
        with self._lock:
            return self._paused

    def pause(self, reason):
        """在替换原生保留时，保持ACK/心跳传输保持活跃。"""
        with self._lock:
            if self._stopped:
                raise RuntimeError('Resident motion gate is stopped')
            self._paused = True
            self.pause_reason = str(reason)

    def resume(self):
        """只有运行时可以在原生保持ACK和静止后调用。"""
        with self._lock:
            if self._stopped:
                raise RuntimeError('Resident motion gate is stopped')
            self._paused = False
            self.pause_reason = None

    def _close_channel(self,completed=False):
        try:
            if completed:self._channel.close_completed()
            else:self._channel.close()
        except Exception as exc:
            self.close_error = str(exc)
            details=getattr(exc,'retirement_details',None)
            if (isinstance(details,dict) and set(details)=={'phase','category','duration_s','exit_code'}
                    and details['phase'] in ('writer','eof','process','reader','stdout')
                    and details['category'] in ('timeout','prior_failure','io_error','nonzero_exit','operation')
                    and type(details['duration_s']) in (int,float) and math.isfinite(details['duration_s'])
                    and 0<=details['duration_s']<=60
                    and (details['exit_code'] is None or type(details['exit_code']) is int
                         and -2**31<=details['exit_code']<2**32)):
                self.close_details=dict(details)
                print('RTDE_COMPLETED_RETIREMENT_FAILED '+json.dumps(self.close_details,sort_keys=True),
                      file=sys.stderr,flush=True)
        finally:
            self.close_finished.set()

    def _close_async(self,completed=False):
        threading.Thread(target=self._close_channel,args=(completed,),name='resident-channel-close',daemon=True).start()

    def retire_completed(self,reason):
        """在正常清理之前关闭进入；切勿升级更早的停止。"""
        with self._lock:
            if self._stopped or not self._armed:
                raise RuntimeError('Resident motion gate stopped before completed retirement')
            self._stopped=True
            self._armed=False
            self.reason=str(reason)
        self._close_async(completed=True)

    def stop(self, reason):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._armed = False
            self.reason = str(reason)
        self._close_async()

    def send(self, *args):
        failure = None
        with self._lock:
            if not self._armed or self._stopped:
                raise RuntimeError('Resident motion gate is not armed')
            try:
                if self._paused and len(args) >= 2 and args[1] in (1, 2):
                    args = (args[0], 3, *args[2:])
                self._channel.send(*args)
            except Exception as exc:
                self._stopped = True
                self._armed = False
                self.reason = 'RTDE input failed'
                failure = exc
        if failure is not None:
            self._close_async()
            raise failure


class GuestControlWatchdog:
    """独立于相机和模型循环观看访客快照。
    “read_snapshot”必须返回一个带有“instance”、“epoch”和“active”的原子映射。
    它不得刷新访客租约。使用了两个线程，因此卡住的快照无法阻止超时关闭门。在运动之前明确等待准备状态；
    一旦停止，就没有重置。
    共享的审查有限路径运行时明确使用仅控制所有权，并使用require_camera=False。
    新图像仍然允许新的视觉决策。其他调用者保留默认的相机要求。每个样本必须报告新的实际所有者控制。
    有了明确的恢复协调器，相同相遇的通信间隙将代替原生保留；只有运行时才能授权恢复。"""
    def __init__(self, read_snapshot, gate, *, interval, timeout, require_camera=True, recovery=None):
        if (not callable(read_snapshot) or not isinstance(gate, ResidentMotionGate)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (interval, timeout)) or interval >= timeout or type(require_camera) is not bool):
            raise ValueError('Snapshot reader, gate, and valid explicit intervals required')
        self._read = read_snapshot
        self._gate = gate
        self._interval = interval
        self._timeout = timeout
        self._require_camera = require_camera
        self._recovery = recovery
        self._lock = threading.Lock()
        self._started = False
        self._started_at = None
        self._last = None
        self._identity = None
        self._ready = threading.Event()
        self.stopped = threading.Event()
        self.failure = None
        self.failure_snapshot = None
        self._threads = []

    def _fail_locked(self, reason):
        if self.stopped.is_set():
            return
        self.failure = reason
        # 在发出停止监控信号之前，请锁定运动门。
        self._gate.stop(reason)
        self.stopped.set()

    def _sample_loop(self):
        try:self._sample_loop_owned()
        finally:
            close=getattr(self._read,'close_current_thread',None)
            if callable(close):close()

    def _sample_loop_owned(self):
        if self._recovery is not None:
            return self._recoverable_sample_loop()
        while not self.stopped.is_set():
            flags = None
            try:
                value = self._read()
                now = time.monotonic()
                failed, flags = _snapshot_diagnostics(value, self._require_camera)
                if failed:
                    raise ValueError('Guest snapshot invalid or inactive: '+', '.join(failed))
                identity = value['instance'], value['epoch']
                with self._lock:
                    if self.stopped.is_set():
                        return
                    if self._identity is not None and identity != self._identity:
                        self._fail_locked('Guest control epoch or instance changed')
                        return
                    if now - (self._last if self._last is not None else self._started_at) > self._timeout:
                        self._fail_locked('Guest control snapshot stale')
                        return
                    self._identity = identity
                    self._last = now
                    self._gate.arm()
                    self._ready.set()
            except Exception as exc:
                with self._lock:
                    if not self.stopped.is_set() and flags is not None:
                        self.failure_snapshot = MappingProxyType(flags)
                    self._fail_locked('Guest snapshot failed: ' + str(exc))
                return
            self.stopped.wait(self._interval)

    def _recoverable_sample_loop(self):
        while not self.stopped.is_set():
            try:
                value = self._read()
                with self._recovery.synchronized():
                    with self._lock:
                        if self.stopped.is_set():
                            return
                        status = self._recovery.observe(value)
                        now = time.monotonic()
                        self._identity = value['instance'], value['epoch']
                        self._last = now
                        if status in ('waiting', 'ready'):
                            self._gate.pause('Communication recovery')
                        self._gate.arm()
                        self._ready.set()
            except Exception as exc:
                with self._recovery.synchronized():
                    with self._lock:
                        if self.stopped.is_set():
                            return
                        try:
                            recoverable = self._recovery.transport_error(exc)
                        except Exception:
                            recoverable = False
                        if not recoverable:
                            self._fail_locked('Guest snapshot failed: ' + str(exc))
                            return
                        self._gate.pause('Communication recovery')
            self.stopped.wait(self._interval)

    def _timeout_loop(self):
        while not self.stopped.wait(min(self._interval, self._timeout / 4)):
            now = time.monotonic()
            if self._recovery is not None:
                with self._recovery.synchronized():
                    with self._lock:
                        if self.stopped.is_set():
                            return
                        try:
                            self._recovery.status
                            if now - (self._last if self._last is not None else self._started_at) > self._timeout:
                                error = GuestSnapshotRequestError('timeout', 'request', self._timeout * 1000)
                                if not self._recovery.transport_error(error):
                                    raise RuntimeError('Guest control recovery expired')
                                self._gate.pause('Communication recovery')
                        except Exception as exc:
                            self._fail_locked('Guest control snapshot stale: ' + str(exc))
                            return
                continue
            with self._lock:
                if now - (self._last if self._last is not None else self._started_at) > self._timeout:
                    self._fail_locked('Guest control snapshot stale')
                    return

    def start(self):
        with self._lock:
            if self._started:
                raise RuntimeError('Watchdog cannot be restarted')
            self._started = True
            self._started_at = time.monotonic()
            self._threads = [threading.Thread(target=self._sample_loop, name='guest-control-sample', daemon=True),
                             threading.Thread(target=self._timeout_loop, name='guest-control-timeout', daemon=True)]
            for thread in self._threads:
                thread.start()

    def wait_ready(self, timeout):
        if not self._ready.wait(timeout):
            return False
        with self._lock:
            return not self.stopped.is_set()

    def close(self):
        with self._lock:
            self._fail_locked('Watchdog closed')
        # 读者本身可能会被阻止。它是一个守护进程；无需加入即可抑制RTDE通道或停止此守护程序。

    def retire_completed(self):
        """在正常门清理之前，请将此显示器撤回，无需发出“停止”指令。"""
        with self._lock:
            if not self._started or self.stopped.is_set():
                raise RuntimeError('Guest watchdog stopped before completed retirement')
            self.stopped.set()
