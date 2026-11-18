"""将经过明确审查的XYZ事件序列化到实时参考门。此模块中不会发生连接、进程启动、居民上传或机器人调试。
呼叫者拥有所有资源，必须在推进电测时调用tick，而不阻塞，并且必须运行独立的客机/摄像头守护程序。
固定方向的居民不支持模拟器的倾斜打开。
HTTP门操作在守护程序工作器上运行，并具有有限的监督器截止日期，因此它们不能阻塞机器人心跳。
门ACK或这些应用程序检查都不是安全等级的停止或路径清除证明。关闭门无法召回已排队或正在传输的RTDE数据包。
居民的250毫秒心跳丢失停止尚未进行物理验证。"""
import http.client
import json
import math
import sys
import threading
import time

from .physical_resident_engine import ResidentEngine
from .reference_gate import rotation_vector_distance

_CUES = frozenset(('MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN',
                   'MOVE_CLOSER', 'MOVE_FARTHER'))

_REJECTION_REASONS=frozenset(('request_invalid','guest_inactive','guest_stop','guest_expired',
    'epoch_changed','display_closed','backend_missing','backend_heartbeat_expired',
    'backend_clock_invalid','control_heartbeat_expired','camera_stale','backend_fault','backend_revoked',
    'communication_timeout'))
_REFERENCE_REASONS=frozenset(('gate_blocked','wrong_session','stale_sample','return_unverified','invalid_reference',
    'inactive_reference','restart_stage','restart_not_begun','restart_already_begun','communication_waiting'))
_REFERENCE_OPS=frozenset(('begin','sample','release','restart_begin','restart_sample','restart_release'))
_JOB_KINDS=_REFERENCE_OPS|frozenset(('status','photo_source','photo_ready'))


def safe_reference_check(value):
    """允许列表数字参考诊断，无需原始电测/身份。"""
    if not isinstance(value,dict):return {}
    result={}
    if value.get('status') in ('unblocked','no_sample','freshness','position','orientation',
                               'moving','acknowledgement','dwell','ready'):
        result['status']=value['status']
    if type(value.get('acknowledged')) is bool:result['acknowledged']=value['acknowledged']
    for key in ('steady_ms','sample_age_ms','position_error_mm','rotation_error_rad',
                'linear_speed_mm_s','angular_speed_rad_s'):
        number=value.get(key)
        if type(number) in (int,float) and math.isfinite(number):result[key]=round(number,6)
    return result


class ReferenceRequestError(RuntimeError):
    """修复了本地传输分类；从不包含URL或原始正文。"""
    def __init__(self,category, *, phase,elapsed_ms,status=None,reason=None,diagnostics=None,
                 reference_reason=None,reference_op=None,reference_check=None):
        self.category=category if category in ('timeout','connection','protocol','response','http_rejected') else 'response'
        self.phase=phase if phase in ('request','headers','body','response') else 'response'
        self.status=status if type(status) is int and 100<=status<=599 else None
        self.reason=reason if isinstance(reason,str) and reason in _REJECTION_REASONS else 'request_invalid'
        self.reference_reason=reference_reason if isinstance(reference_reason,str) and reference_reason in _REFERENCE_REASONS else None
        self.reference_op=reference_op if isinstance(reference_op,str) and reference_op in _REFERENCE_OPS else None
        self.reference_check=safe_reference_check(reference_check)
        self.elapsed_ms=round(elapsed_ms,2) if type(elapsed_ms) in (int,float) and math.isfinite(elapsed_ms) else None
        self.diagnostics={}
        if isinstance(diagnostics,dict):
            for key in ('control_age_s','backend_age_s','camera_age_s'):
                value=diagnostics.get(key)
                if type(value) in (int,float) and math.isfinite(value):self.diagnostics[key]=round(value,6)
            if diagnostics.get('mode') in ('intro','guidance'):self.diagnostics['mode']=diagnostics['mode']
        fields={'category':self.category,'phase':self.phase,'elapsed_ms':self.elapsed_ms}
        if self.status is not None:fields['status']=self.status
        if self.category=='http_rejected':
            fields.update(reason=self.reason,**self.diagnostics)
            if self.reference_reason is not None:fields['reference_reason']=self.reference_reason
            if self.reference_op is not None:fields['reference_op']=self.reference_op
            if self.reference_check:fields['reference_check']=dict(self.reference_check)
        self._safe_details=dict(fields)
        super().__init__('Local reference service unavailable '+json.dumps(fields,sort_keys=True))

    def safe_details(self):
        result=dict(self._safe_details)
        if 'reference_check' in result:result['reference_check']=dict(result['reference_check'])
        return result


def _safe_job_error(kind,error):
    fields={'operation':kind if isinstance(kind,str) and kind in _JOB_KINDS else 'unknown'}
    if type(error) is ReferenceRequestError:fields.update(error.safe_details())
    else:fields['error_type']=type(error).__name__
    return json.dumps(fields,sort_keys=True,allow_nan=False)


class LocalReferenceGate:
    """已认证的参考操作；固定在本地主机，无重定向。"""
    def __init__(self, port, token, *, timeout, control, lease):
        if (type(port) is not int or not 1 <= port <= 65535
                or not isinstance(token, str) or not 1 <= len(token) <= 512
                or any(ord(c) < 33 or ord(c) > 126 for c in token)
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or not isinstance(control, dict) or not isinstance(control.get('instance'), str)
                or not 1 <= len(control['instance']) <= 128 or type(control.get('epoch')) is not int
                or not 0 <= control['epoch'] < 2**53 or not isinstance(lease, str)
                or not 1 <= len(lease) <= 512
                or type(control.get('started_ns', control.get('control_started_ns'))) is not int
                or control.get('started_ns', control.get('control_started_ns')) <= 0):
            raise ValueError('Valid local port, token and timeout required')
        self._port, self._token, self._timeout = port, token, timeout
        self._identity = {'instance': control['instance'], 'epoch': control['epoch']}
        self.started_ns = control.get('started_ns', control.get('control_started_ns'))
        self._lease = lease
        self._feedback_conn = None
        self._feedback_owner = None
        self._feedback_owner_lock = threading.Lock()
        self._restart_reference = None
        self._restart_reanchor = False

    @property
    def identity(self):
        return dict(self._identity)

    @property
    def restart_reference(self):
        return self._restart_reference

    def _request(self, method, path, body=None):
        conn = http.client.HTTPConnection('127.0.0.1', self._port, timeout=self._timeout)
        try:
            return self._exchange(conn, method, path, body)
        finally:
            conn.close()

    def _claim_feedback_transport(self):
        owner = threading.get_ident()
        with self._feedback_owner_lock:
            if self._feedback_owner is None:
                self._feedback_owner = owner
            elif self._feedback_owner != owner:
                raise RuntimeError('Feedback transport belongs to its sender thread')

    def feedback_request(self, body):
        """一个发送者拥有的持久连接；不会重播失败的POST。门户工作者继续使用独立的短寿命连接。
        反馈发送者必须在其线程的finally中关闭自己的传输。"""
        self._claim_feedback_transport()
        conn = self._feedback_conn
        self._feedback_conn = None
        if conn is None:
            conn = http.client.HTTPConnection('127.0.0.1', self._port, timeout=self._timeout)
        retained = False
        try:
            conn.timeout = self._timeout
            if conn.sock is not None:
                conn.sock.settimeout(self._timeout)
            result, reusable = self._exchange(conn, 'POST', '/physical-feedback', body, keep_alive=True)
            if reusable:
                self._feedback_conn = conn
                retained = True
            return result
        finally:
            if not retained:
                conn.close()

    def close_feedback_transport(self):
        self._claim_feedback_transport()
        conn, self._feedback_conn = self._feedback_conn, None
        if conn is not None:
            conn.close()

    def presentation_transport(self):
        """仅由显示器发件人拥有的独立同身份套接字。"""
        return LocalReferenceGate(self._port,self._token,timeout=self._timeout,
            control={**self._identity,'started_ns':self.started_ns},lease=self._lease)

    def control_transport(self, *, timeout):
        """关键发送者使用其现有的租约预算，而不是门禁RPC超时。"""
        return LocalReferenceGate(self._port,self._token,timeout=timeout,
            control={**self._identity,'started_ns':self.started_ns},lease=self._lease)

    def _exchange(self, conn, method, path, body, *, keep_alive=False):
        headers = {'X-Guest-Token': self._token, 'Connection': 'keep-alive' if keep_alive else 'close'}
        data = None
        if body is not None:
            data = json.dumps({**body, **self._identity, 'lease': self._lease}, allow_nan=False).encode()
            headers.update({'Origin': f'http://127.0.0.1:{self._port}',
                            'Content-Type': 'application/json',
                            'X-Guest-Client': 'physical-supervisor'})
        started=time.monotonic();phase='request'
        try:
            conn.request(method, path, body=data, headers=headers)
            phase='headers'
            response = conn.getresponse()
            phase='body'
            if response.status != 200:
                payload=response.read(4097)
                try:rejection=json.loads(payload) if len(payload)<=4096 else {}
                except (ValueError,TypeError):rejection={}
                if not isinstance(rejection,dict):rejection={}
                raise ReferenceRequestError('http_rejected',phase='response',
                    elapsed_ms=(time.monotonic()-started)*1000,status=response.status,
                    reason=rejection.get('reason'),diagnostics=rejection.get('physical_revocation'),
                    reference_reason=rejection.get('reference_reason'),reference_op=rejection.get('reference_op'),
                    reference_check=rejection.get('reference_check'))
            payload = response.read(262145)
            if len(payload) > 262144:
                raise ValueError('Reference response too large')
            if keep_alive:
                # HTTPResponse.read(limit)不一定会拒绝截断的内容长度响应。永远不要缓存该套接字。
                if response.length not in (None, 0):
                    raise ValueError('Incomplete reference response')
                if not response.isclosed():
                    if response.read(1) or not response.isclosed():
                        raise ValueError('Incomplete reference response')
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise ValueError('Invalid reference response')
            return (result, not response.will_close) if keep_alive else result
        except ReferenceRequestError:raise
        except TimeoutError:
            raise ReferenceRequestError('timeout',phase=phase,elapsed_ms=(time.monotonic()-started)*1000) from None
        except OSError:
            raise ReferenceRequestError('connection',phase=phase,elapsed_ms=(time.monotonic()-started)*1000) from None
        except http.client.HTTPException:
            raise ReferenceRequestError('protocol',phase=phase,elapsed_ms=(time.monotonic()-started)*1000) from None
        except (ValueError,TypeError):
            raise ReferenceRequestError('response',phase=phase,elapsed_ms=(time.monotonic()-started)*1000) from None

    def begin(self, reference):
        return self._request('POST', '/reference-gate', {'op': 'begin', 'reference': reference}).get('session')

    def status(self, session):
        state = self._request('GET', '/operator-state')
        if (state.get('active') is not True or state.get('control') != self._identity
                or state.get('control_started_ns') != self.started_ns
                or not isinstance(state.get('reference_gate'), dict)):
            raise RuntimeError('Guest control inactive')
        return state['reference_gate']

    def photo_source(self):
        state = self._request('GET', '/operator-state')
        if (state.get('active') is not True or state.get('control') != self._identity
                or state.get('control_started_ns') != self.started_ns
                or not isinstance(state.get('physical'), dict)
                or state['physical'].get('backend_available') is not True):
            raise RuntimeError('Physical photo session unavailable')
        return state['physical'].get('photo_source_ns')

    def photo_ready(self, source_ns):
        result = self._request('POST', '/physical-feedback', {'op': 'photo_ready', 'source_ns': source_ns})
        return result.get('ok') is True

    def fault(self):
        result = self._request('POST', '/physical-feedback', {'op': 'fault'})
        return result.get('ok') is True

    def sample(self, session, pose, speed, sampled_at):
        result=self._request('POST', '/reference-gate', {'op': 'sample', 'session': session,
                             'pose': pose, 'speed': speed, 'sampled_at': sampled_at})
        if (result.get('ok') is not True or type(result.get('return_verified')) is not bool
                or type(result.get('sampled_at')) not in (int,float)
                or not math.isfinite(result['sampled_at']) or result['sampled_at']!=sampled_at):
            raise RuntimeError('Reference sample receipt invalid')
        return {key:result[key] for key in ('ok','sampled_at','return_verified')}

    def release(self, session):
        result = self._request('POST', '/reference-gate', {'op': 'release', 'session': session})
        if result.get('ok') is not True:
            raise RuntimeError('Reference release rejected')
        return result.get('release_ns')

    def restart_begin(self, session, *, reanchor=False):
        """重新开始证明，可选使用所选的当前姿势策略。"""
        from uuid import uuid4
        if type(reanchor) is not bool:raise ValueError('Explicit restart reference policy required')
        body={'op': 'restart_begin','session': session, 'request_id': uuid4().hex}
        if reanchor:body['reference_policy']='capture_stationary_on_explicit_start'
        result = self._request('POST', '/reference-gate', body)
        if result.get('ok') is not True:
            raise RuntimeError('Reference restart verification rejected')
        reference = result.get('reference')
        if (not isinstance(reference, (list, tuple)) or len(reference) != 6
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in reference)):
            raise RuntimeError('Reference restart reference invalid')
        original = tuple(float(value) for value in reference)
        if self._restart_reference is not None and self._restart_reference != original:
            raise RuntimeError('Reference restart reference changed')
        if reanchor and result.get('reference_policy')!='capture_stationary_on_explicit_start':
            raise RuntimeError('Reference restart selection policy rejected')
        self._restart_reference = original
        self._restart_reanchor = reanchor
        return True

    def restart_sample(self, session, pose, speed, sampled_at):
        result = self._request('POST', '/reference-gate', {'op': 'restart_sample',
            'session': session, 'pose': pose, 'speed': speed, 'sampled_at': sampled_at})
        if (result.get('ok') is not True or type(result.get('return_verified')) is not bool
                or type(result.get('sampled_at')) not in (int, float)
                or not math.isfinite(result['sampled_at']) or result['sampled_at'] != sampled_at):
            raise RuntimeError('Reference restart sample receipt invalid')
        receipt={key: result[key] for key in ('ok', 'sampled_at', 'return_verified')}
        check=safe_reference_check(result.get('reference_check'))
        if check:receipt['reference_check']=check
        return receipt

    def restart_release(self, session, pose=None, speed=None, sampled_at=None):
        from uuid import uuid4
        body = {'op': 'restart_release', 'session': session, 'request_id': uuid4().hex}
        supplied = (pose, speed, sampled_at)
        if any(value is not None for value in supplied):
            if any(value is None for value in supplied):
                raise ValueError('Complete restart release sample required')
            body.update(pose=pose, speed=speed, sampled_at=sampled_at)
        result = self._request('POST', '/reference-gate', body)
        if result.get('ok') is not True:
            raise RuntimeError('Reference restart release rejected')
        if self._restart_reanchor:
            reference=result.get('reference')
            if (not isinstance(reference,(list,tuple)) or len(reference)!=6
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in reference)):
                raise RuntimeError('Committed selected reference missing or invalid')
            self._restart_reference=tuple(float(x) for x in reference)
        return result.get('release_ns')


class _Worker:
    def __init__(self, fn, *args):
        started_at = time.monotonic()
        self._done = threading.Event()
        self.value = None
        self.error = None
        self.elapsed = None
        self.recovery_accounting = None
        def run():
            owner = getattr(fn, '__self__', None)
            begin_accounting = getattr(owner, 'begin_worker_accounting', None)
            try:
                if callable(begin_accounting):
                    self.recovery_accounting = begin_accounting()
                self.value = fn(*args)
            except Exception as exc:
                self.error = exc
            finally:
                if self.recovery_accounting is not None:
                    try:
                        owner.end_worker_accounting(self.recovery_accounting)
                    except Exception as exc:
                        if self.error is None:
                            self.error = exc
                # 在请求开始前包括排程，并在广告完成前发布实际完成时间。
                self.elapsed = time.monotonic() - started_at
                self._done.set()
        self._thread = threading.Thread(target=run, name='physical-reference-request', daemon=True)
        self._thread.start()

    @property
    def done(self):
        return self._done.is_set()

    def close(self, timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('Finite nonnegative worker close timeout required')
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeError('Reference request worker did not close')


class PhysicalInteractionSupervisor:
    """一次故障关闭的居民相遇；所有公共方法都是序列化的。
    ``forward_paths``为每个提示提供经过审查的绝对TCP XYZ航点。
    主管将相同的路径反向返回经过审查的中立位置。路径审查必须包括两个方向和当前连接的设备。
    默认情况下不使用模拟器手势、速度、振幅或校准。"""
    def __init__(self, engine, reference_gate, forward_paths, *, reviewed_paths, hold_timeout,
                 action_timeout, gate_timeout, max_joint_speed, worker_factory=_Worker):
        if (not isinstance(engine, ResidentEngine) or not isinstance(forward_paths, dict)
                or set(forward_paths) != _CUES or not isinstance(reviewed_paths, dict)
                or set(reviewed_paths) != _CUES):
            raise ValueError('Resident and reviewed paths for all six cues required')
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
               for x in (hold_timeout, action_timeout, gate_timeout, max_joint_speed)):
            raise ValueError('Explicit positive finite timeouts and speed required')
        self.engine, self.reference = engine, reference_gate
        self.paths = {}
        for cue, waypoints in forward_paths.items():
            if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 20:
                raise ValueError('Each cue requires a reviewed forward path')
            forward = [engine.cfg.check_target(point) for point in waypoints]
            approved = reviewed_paths[cue]
            if not isinstance(approved, list) or [engine.cfg.check_target(point) for point in approved] != forward:
                raise ValueError('Path differs from recorded review')
            neutral = tuple(engine.cfg.neutral[:3])
            if any(point == neutral for point in forward):
                raise ValueError('Forward path must not include neutral')
            self.paths[cue] = forward + list(reversed(forward[:-1])) + [neutral]
            if engine.cfg.gesture is not None:self.paths[cue]=[forward[-1]]
        self.hold_timeout, self.action_timeout, self.gate_timeout = hold_timeout, action_timeout, gate_timeout
        self.max_joint_speed = max_joint_speed
        self.worker_factory = worker_factory
        self.state = 'OBSERVING'
        self.failure = None
        self._job = self._job_kind = self._job_started = None
        self._detached_workers = []
        self._session = None
        self._path = []
        self._deadline = None
        self._last_frame_ns = 0
        self._decision_ns = 0
        self._network_decision_pause_s = 0.
        self._release_ns = 0
        self._settle_since = None
        self._last_settle_sample = None
        self._last_joint_sample = None
        self._sample_submitted_at = None
        self._sample_payload_at = None
        self._sample_acknowledged_at = None
        self._reference_return_verified = False
        self._last_enabled_sequence = 0
        self._photo_source_ns = None
        self._photo_requested_at = None
        self._photo_still_since = None
        self._photo_last_sample = None
        self._revocation = None
        self._move_ready = None
        self._active_cue = None
        self._await_first_enabled = False
        self.recovering_expiry = False
        self._control_identity = None
        self._control_started_ns = None
        self._network_pause_started = None
        self._network_pause_intervals = []

    def pause_for_network(self, now):
        """冻结交互会计；运行时单独泵出原生保留。"""
        if type(now) not in (int, float) or not math.isfinite(now):
            self._fail('Invalid network pause clock')
        if self.state in ('FAULT', 'STOPPED'):
            raise RuntimeError('Interaction stopped')
        if self._network_pause_started is None:
            self._network_pause_started = now

    def resume_after_network(self, now, frame_floor):
        if (type(now) not in (int, float) or not math.isfinite(now)
                or type(frame_floor) is not int or frame_floor <= 0
                or frame_floor > round(now * 1e9)
                or self._network_pause_started is None or now < self._network_pause_started):
            self._fail('Invalid network resume proof')
        if self.state in ('FAULT', 'STOPPED'):
            raise RuntimeError('Interaction stopped')
        started = self._network_pause_started
        self._network_pause_intervals.append((started, now))
        self._network_pause_started = None
        if self._deadline is not None:
            self._deadline += now - started
        if self._decision_ns:
            self._network_decision_pause_s += max(0., now - max(started, self._decision_ns / 1e9))
        self._settle_since = self._last_settle_sample = None
        self._sample_acknowledged_at = None
        self._reference_return_verified = False
        self._sample_submitted_at = self._sample_payload_at = None
        self._release_ns = max(self._release_ns, frame_floor)
        self._last_frame_ns = max(self._last_frame_ns, frame_floor)
        self._detached_workers[:] = [worker for worker in self._detached_workers
                                    if getattr(worker, 'done', None) is not True]
        if (self._job_kind == 'sample' or self.state.startswith('PHOTO_')):
            # 超时请求仍然可以执行。
            # 丢弃其语义结果，但保留工作进程，以便在其旧的HTTP副作用仍然可能的情况下，关闭不能接受后续的启动。
            if self._job is not None and getattr(self._job, 'done', None) is not True:
                if not any(worker is self._job for worker in self._detached_workers):
                    self._detached_workers.append(self._job)
        if self._job_kind == 'sample':
            self._job = self._job_kind = self._job_started = None
            self._network_pause_intervals.clear()
            self.state = 'SETTLING'
        if self.state.startswith('PHOTO_'):
            self.state = 'OBSERVING'
            self._deadline = None
            self._photo_source_ns = self._photo_requested_at = None
            self._photo_still_since = self._photo_last_sample = None
            self._job = self._job_kind = self._job_started = None

    def _network_job_overlap(self, until, failed_attempts=(), *, since=None):
        start = self._job_started if since is None else max(self._job_started, since)
        intervals = list(self._network_pause_intervals) + list(failed_attempts)
        if self._network_pause_started is not None:
            intervals.append((self._network_pause_started, until))
        # 失败的尝试可能会与运行时检测到的暂停重叠。一次性排除它们的联合，永远不要为相同的墙面时间授予两个积分。
        clipped = sorted((max(start, begin), min(until, end)) for begin, end in intervals
                         if min(until, end) > max(start, begin))
        total = 0.
        end = start
        for begin, finish in clipped:
            total += max(0., finish - max(begin, end))
            end = max(end, finish)
        return total

    def bind_control_identity(self, control):
        """将语义框架绑定到客机时段及其源时间底线。"""
        if (self.state != 'OBSERVING' or self._last_frame_ns or not isinstance(control, dict)
                or not isinstance(control.get('instance'), str) or not 1 <= len(control['instance']) <= 128
                or type(control.get('epoch')) is not int or not 0 <= control['epoch'] < 2**53
                or type(control.get('started_ns')) is not int or control['started_ns'] <= 0):
            raise ValueError('Explicit active control identity and frame floor required')
        identity = {'instance': control['instance'], 'epoch': control['epoch']}
        ref_identity = getattr(self.reference, 'identity', None)
        if ref_identity != identity or getattr(self.reference, 'started_ns', None) != control['started_ns']:
            raise ValueError('Reference service identity mismatch')
        self._control_identity = identity
        self._control_started_ns = control['started_ns']

    def set_move_readiness(self, callback):
        """在动作之前安装一个匹配的本地接受的动作谓词。
        回调接收（提示、source_ns），必须是非阻塞的，并且必须仅在接受的动作和当前控制流中返回字面值True。
        演示确认不是动作租约。如果没有谓词，动作就无法启动。"""
        if not callable(callback) or self.state != 'OBSERVING' or self._last_frame_ns:
            raise ValueError('Install a readiness predicate before the first action')
        self._move_ready = callback

    def _revoke(self):
        if self._revocation is None:
            try:
                self._revocation = self.worker_factory(self.reference.fault)
            except Exception:
                # 关闭运动门仍然是主要的本地行动；失败的撤销不能被视为成功。
                self._revocation = False

    def _fail(self, reason):
        self.state = 'FAULT'
        self.failure = str(reason)
        self.engine.stop()
        self._revoke()
        raise RuntimeError(self.failure)

    def stop(self):
        self.state = 'STOPPED'
        self.engine.stop()
        self._revoke()

    def close_workers(self, timeout=.5):
        """在绝对的清理预算内退休未完成的HTTP工作。"""
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('Finite nonnegative worker cleanup timeout required')
        deadline = time.monotonic() + timeout
        seen = set()
        errors = []
        for worker in (self._job, self._revocation, *self._detached_workers):
            if worker is None or worker is False or id(worker) in seen:
                continue
            seen.add(id(worker))
            try:
                close = getattr(worker, 'close', None)
                if callable(close):
                    close(max(0., deadline - time.monotonic()))
                elif getattr(worker, 'done', None) is not True:
                    raise RuntimeError('Pending reference worker has no bounded close')
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('Reference worker cleanup incomplete') from errors[0]
        self._detached_workers.clear()

    def _submit(self, kind, fn, now, *args):
        if self._job is not None:
            self._fail('Reference operation already pending')
        self._job_kind, self._job_started = kind, now
        self._job = self.worker_factory(fn, *args)

    def _cancel_expired_direction(self, now, *, phase):
        """丢弃从未启用的方向，保留测量参考门。"""
        age = now - self._decision_ns / 1e9 - self._network_decision_pause_s
        if not math.isfinite(age) or age <= .3:
            self._fail('Direction decision clock invalid')
        if self._deadline is not None and now > self._deadline:
            self._fail('Interaction phase timed out')
        if not self._session or self._job is not None:
            self._fail('Expired direction reference session unavailable')
        if self.engine.state == 'MOVING':
            self.engine.cancel_before_enable()
        elif self.engine.state != 'READY':
            self._fail('Expired direction is not held')
        diagnostic = {'phase': phase, 'cue': self._active_cue,
                      'source_age_ms': round(age * 1000, 2),
                      'sequence': self.engine.sequence}
        print('PHYSICAL_DIRECTION_EXPIRED: ' + json.dumps(diagnostic, sort_keys=True),
              file=sys.stderr, flush=True)
        self._await_first_enabled = False
        self.recovering_expiry = True
        self._path.clear()
        self._active_cue = None
        self._decision_ns = 0
        self._settle_since = self._last_settle_sample = None
        self._sample_acknowledged_at = None
        self._reference_return_verified = False
        self._sample_submitted_at = self._sample_payload_at = None
        self._deadline = now + self.action_timeout
        self.state = 'SETTLING'

    def accept_event(self, event, *, now):
        if self.state in ('FAULT', 'STOPPED'):
            raise RuntimeError('Interaction stopped')
        if isinstance(event, dict) and event.get('event') == 'person_guidance':
            if (self._control_identity is None
                    or event.get('control_instance') != self._control_identity['instance']
                    or type(event.get('control_epoch')) is not int
                    or event['control_epoch'] != self._control_identity['epoch']):
                self._fail('Guidance control identity mismatch')
            source_ns = event.get('source_monotonic_ns')
            if type(source_ns) is not int or source_ns <= self._control_started_ns:
                return False
        if self.state != 'OBSERVING' or self.engine.state != 'READY':
            return False
        if (isinstance(event, dict) and event.get('event') == 'person_guidance'
                and event.get('coordinate_frame') == 'normalized_camera_image'
                and event.get('robot_actuated') is False and event.get('tracking') is False
                and event.get('phase') == 'SETTLING' and event.get('cue') == 'HOLD'
                and event.get('reason') == 'TARGET_PENDING_PHYSICAL_STOP'
                and type(event.get('source_monotonic_ns')) is int
                and type(now) in (int, float) and math.isfinite(now)
                and event['source_monotonic_ns'] > self._release_ns
                and 0 <= now - event['source_monotonic_ns'] / 1e9 <= .3):
            self._photo_requested_at = now
            self._deadline = now + self.action_timeout
            self.state = 'PHOTO_LOOKUP'
            try:
                self._submit('photo_source', self.reference.photo_source, now)
            except Exception as exc:
                if self.state != 'FAULT': self._fail(str(exc))
                raise
            return True
        if (not isinstance(event, dict) or event.get('event') != 'person_guidance'
                or event.get('coordinate_frame') != 'normalized_camera_image'
                or event.get('robot_actuated') is not False or event.get('tracking') is not True
                or not isinstance(event.get('cue'), str) or event.get('cue') not in _CUES
                or type(event.get('source_monotonic_ns')) is not int
                or type(now) not in (int, float) or not math.isfinite(now)):
            return False
        ns = event['source_monotonic_ns']
        if (ns <= self._last_frame_ns or ns <= self._release_ns
                or not 0 <= now - ns / 1e9 <= .3):
            return False
        self._last_frame_ns = ns
        self._decision_ns = ns
        self._network_decision_pause_s = 0.
        self._active_cue = event['cue']
        self._path = list(self.paths[event['cue']])
        self._deadline = now + self.hold_timeout
        self.state = 'BEGIN_PENDING'
        try:
            self._submit('begin', self.reference.begin, now, list(self.engine.cfg.neutral))
        except Exception as exc:
            if self.state != 'FAULT':
                self._fail(str(exc))
            raise
        return True

    def _finish_job(self, now):
        if self._job is None:
            return None, None
        done = self._job.done
        completed_elapsed = getattr(self._job, 'elapsed', None) if done else None
        # 在请求按时完成后，呼叫者可能会在很晚的时候被抽样。其记录的持续时间是权威的；
        # 待定/自定义工作线程保留原始保守的提交到抽样的截止日期。
        elapsed = now - self._job_started if completed_elapsed is None else completed_elapsed
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            self._fail('Reference operation clock invalid')
        accounting = getattr(self._job, 'recovery_accounting', None)
        failed_attempts = ()
        if accounting is not None:
            from .network_reference import ReferenceRecoveryAccounting
            if type(accounting) is not ReferenceRecoveryAccounting:
                self._fail('Reference recovery accounting invalid')
            failed_attempts = accounting.intervals
        completed_at = self._job_started + elapsed
        elapsed -= self._network_job_overlap(completed_at, failed_attempts)
        if elapsed > self.gate_timeout:
            timing = 'completion' if completed_elapsed is not None else 'poll'
            error=getattr(self._job,'error',None) if done else None
            diagnostic=' worker_error '+_safe_job_error(self._job_kind,error) if error is not None else ''
            self._fail(f'Reference operation timed out: {self._job_kind} {timing} '
                       f'{elapsed:.6f}s exceeds {self.gate_timeout:.6f}s'+diagnostic)
        if not done:
            return None, None
        if (self._job_kind in ('begin', 'status') and self._job.error is None
                and self._decision_ns and failed_attempts):
            # 在RPC中断之前，这种方向已被批准。
            # 信用仅在最初的决定之后对失败时间进行分类，而不是由resume_after_network已经累计的实际暂停时间。
            # 已完成的工作已移除到下面，因此以后的轮询不能再次累计。
            since = self._decision_ns / 1e9
            extra = (self._network_job_overlap(completed_at, failed_attempts, since=since)
                     - self._network_job_overlap(completed_at, since=since))
            self._network_decision_pause_s += max(0., extra)
        job, kind = self._job, self._job_kind
        self._job = self._job_kind = self._job_started = None
        self._network_pause_intervals.clear()
        if job.error is not None:
            # 失败的样本永远不会以过期的sampled_at重新播放。
            from .network_reference import ReferenceSampleInterrupted, ReferencePhotoInterrupted
            if kind == 'sample' and type(job.error) is ReferenceSampleInterrupted:
                self._settle_since = self._last_settle_sample = None
                self._sample_acknowledged_at = None
                self._reference_return_verified = False
                self._sample_submitted_at = self._sample_payload_at = None
                self.state = 'SETTLING'
                return None, None
            if kind in ('photo_source', 'photo_ready') and type(job.error) is ReferencePhotoInterrupted:
                self.state = 'OBSERVING'
                self._deadline = None
                self._photo_source_ns = self._photo_requested_at = None
                self._photo_still_since = self._photo_last_sample = None
                return None, None
            self._fail('Reference operation failed '+_safe_job_error(kind,job.error))
        return kind, job.value

    def _held_and_still(self, feedback, independent_speed, joints):
        speed = feedback['tcp_speed']
        return (self.engine.state == 'READY' and feedback.get('state') == 0
                and type(feedback.get('sequence')) is int
                and feedback['sequence'] > self._last_enabled_sequence
                and math.hypot(*speed[:3]) < .0005 and math.hypot(*speed[3:]) < .002
                and math.hypot(*independent_speed[:3]) < .0005
                and math.hypot(*independent_speed[3:]) < .002
                and max(abs(x) for x in joints) < self.max_joint_speed)

    def _observe_reference_stillness(self, feedback, independent_speed, joints, combined_at):
        pose = feedback['tcp_pose']
        neutral = self.engine.cfg.neutral
        still = (self._held_and_still(feedback, independent_speed, joints)
                 and math.dist(pose[:3], neutral[:3]) <= .0005
                 and rotation_vector_distance(pose[3:], neutral[3:]) <= .005)
        if (not still or self._last_settle_sample is not None
                and combined_at - self._last_settle_sample > .15):
            self._settle_since = None
            self._sample_acknowledged_at = None
            self._reference_return_verified = False
        if still:
            if self._settle_since is None: self._settle_since = combined_at
            self._last_settle_sample = combined_at
        else:
            self._last_settle_sample = None
        return still

    def tick(self, feedback, *, sampled_at, joint_feedback, joint_sampled_at, now):
        if self.state in ('FAULT', 'STOPPED'):
            raise RuntimeError('Interaction stopped')
        try:
            if not self.engine.gate.can_send:
                self._fail('Guest or motion watchdog stopped')
            joints = joint_feedback.get('joint_speed') if isinstance(joint_feedback, dict) else None
            independent_speed = joint_feedback.get('tcp_speed') if isinstance(joint_feedback, dict) else None
            if (type(joint_sampled_at) not in (int, float) or not math.isfinite(joint_sampled_at)
                    or type(now) not in (int, float) or not math.isfinite(now)
                    or not 0 <= now - joint_sampled_at <= .15
                    or self._last_joint_sample is not None and joint_sampled_at <= self._last_joint_sample
                    or not isinstance(joints, (list, tuple)) or len(joints) != 6
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in joints)
                    or not isinstance(independent_speed, (list, tuple)) or len(independent_speed) != 6
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in independent_speed)):
                self._fail('Independent joint feedback unavailable or stale')
            self._last_joint_sample = joint_sampled_at
            if self._await_first_enabled:
                if not 0 <= now - self._decision_ns / 1e9 - self._network_decision_pause_s <= .3:
                    self._cancel_expired_direction(now, phase='first_enabled_packet')
                elif (self._move_ready is None
                        or self._move_ready(self._active_cue, self._decision_ns) is not True):
                    self._fail('Directional feedback acknowledgement no longer available')
            self.engine.tick(feedback, sampled_at=sampled_at, now=now)
            if self.engine.pending is not None and self.engine.pending[2] != 0:
                self._last_enabled_sequence = max(self._last_enabled_sequence, self.engine.pending[0])
                self._await_first_enabled = False
            if self._deadline is not None and now > self._deadline:
                self._fail('Interaction phase timed out')
            # 检查每个电测时钟，包括HTTP样本或发布请求等待期间。后续样本不得隐藏在早期网络请求期间发生的运动。
            settling_observed = False
            combined_at = min(sampled_at, joint_sampled_at)
            if self.state in ('SETTLING', 'SAMPLE_PENDING', 'RELEASE_PENDING'):
                settling_observed = True
                still = self._observe_reference_stillness(feedback, independent_speed, joints, combined_at)
                if self.state == 'RELEASE_PENDING' and not still:
                    self._fail('Robot moved while reference release was pending')
            if self.state == 'PHOTO_PENDING' and not self._held_and_still(feedback, independent_speed, joints):
                self._fail('Robot moved while photo confirmation was pending')
            kind, value = self._finish_job(now)
            if kind == 'begin':
                if not isinstance(value, str) or not value:
                    self._fail('Reference gate session unavailable')
                self._session = value
                self.state = 'WAIT_ACK'
            elif kind == 'status':
                if (not isinstance(value, dict) or value.get('blocked') is not True
                        or value.get('session') != self._session
                        or type(value.get('acknowledged')) is not bool):
                    self._fail('Reference gate identity changed')
                if value['acknowledged']:
                    if not 0 <= now - self._decision_ns / 1e9 - self._network_decision_pause_s <= .3:
                        self._cancel_expired_direction(now, phase='reference_acknowledgement')
                    elif (self._move_ready is not None
                            and self._move_ready(self._active_cue, self._decision_ns) is True):
                        self.state = 'MOVING'
                        self._deadline = now + self.action_timeout
                        self._await_first_enabled = True
                        target=self._path.pop(0)
                        if self.engine.cfg.gesture is not None:self.engine.gesture(target,now=now)
                        else:self.engine.target(target, now=now)
                    else:
                        # 在等待期间，继续对实际参考门进行取样；过期的或不匹配的亮度ACK永远无法启用运动。
                        self.state = 'WAIT_ACK'
                else:
                    self.state = 'WAIT_ACK'
            elif kind == 'sample':
                if (not isinstance(value,dict) or value.get('ok') is not True
                        or type(value.get('return_verified')) is not bool
                        or type(value.get('sampled_at')) not in (int,float)
                        or not math.isfinite(value['sampled_at'])
                        or value['sampled_at']!=self._sample_payload_at):
                    self._fail('Reference sample rejected')
                self._sample_acknowledged_at = self._sample_submitted_at
                self._reference_return_verified = value['return_verified']
                self.state = 'SETTLING'
            elif kind == 'release':
                if type(value) is not int or value <= 0 or value > round(now * 1e9):
                    self._fail('Reference release timestamp invalid')
                self._release_ns = max(value, self._release_ns)
                self._deadline = None
                self._session = None
                self.state = 'OBSERVING'
                if self.recovering_expiry:
                    self.recovering_expiry = False
                    print('PHYSICAL_DIRECTION_REOBSERVING: reference standstill verified; '
                          'waiting for a post-release frame in the same encounter.',
                          file=sys.stderr, flush=True)
            elif kind == 'photo_source':
                if (type(value) is not int or value <= self._release_ns or value <= self._control_started_ns
                        or value > round(now * 1e9)):
                    self._fail('Physical photo source unavailable')
                self._photo_source_ns = value
                self._photo_still_since = self._photo_last_sample = None
                self.state = 'PHOTO_SETTLING'
            elif kind == 'photo_ready':
                if value is not True:
                    self._fail('Physical photo acknowledgement rejected')
                self._deadline = None
                self.state = 'PHOTO_READY'
            if self.state == 'WAIT_ACK' and self._job is None:
                self._submit('status', self.reference.status, now, self._session)
            elif self.state == 'MOVING' and self.engine.state == 'READY':
                if self._path:
                    self.engine.target(self._path.pop(0), now=now)
                else:
                    self._settle_since = self._last_settle_sample = None
                    self._sample_acknowledged_at = None
                    self._reference_return_verified = False
                    self.state = 'SETTLING'
            if self.state == 'SETTLING' and self._job is None:
                if not settling_observed:
                    self._observe_reference_stillness(feedback, independent_speed, joints, combined_at)
                # 独立的遥测和发送的TCP样本可能具有不同的时间戳/间隔。
                # 在释放之前，必须满足本地静止和服务器确切的接受样本停留时间。
                if (self._reference_return_verified and self._settle_since is not None
                        and self._sample_acknowledged_at is not None
                        and 0<=now-self._sample_payload_at<=.15
                        and self._sample_acknowledged_at - self._settle_since >= .6):
                    self.state = 'RELEASE_PENDING'
                    self._submit('release', self.reference.release, now, self._session)
                else:
                    self.state = 'SAMPLE_PENDING'
                    self._sample_submitted_at = combined_at
                    self._sample_payload_at = sampled_at
                    self._submit('sample', self.reference.sample, now, self._session,
                                 feedback['tcp_pose'], feedback['tcp_speed'], sampled_at)
            if self.state == 'PHOTO_SETTLING' and self._job is None:
                speed = feedback.get('tcp_speed') if isinstance(feedback, dict) else None
                if (not isinstance(speed, (list, tuple)) or len(speed) != 6
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in speed)):
                    self._fail('TCP speed unavailable for photo')
                combined_at = min(sampled_at, joint_sampled_at)
                held = (self._held_and_still(feedback, independent_speed, joints)
                        and combined_at > self._photo_requested_at)
                if not held or (self._photo_last_sample is not None
                                and combined_at - self._photo_last_sample > .15):
                    self._photo_still_since = None
                if held:
                    if self._photo_still_since is None: self._photo_still_since = combined_at
                    self._photo_last_sample = combined_at
                    if combined_at - self._photo_still_since >= .6:
                        self.state = 'PHOTO_PENDING'
                        self._submit('photo_ready', self.reference.photo_ready, now, self._photo_source_ns)
        except Exception as exc:
            if self.state not in ('FAULT', 'STOPPED'):
                self._fail(str(exc))
            raise
