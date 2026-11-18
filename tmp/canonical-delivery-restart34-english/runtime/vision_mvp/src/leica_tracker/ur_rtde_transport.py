"""为带保护的常驻寄存器协议提供单次使用的传输通道。实机和仿真的空间会话共用批量 RTDE v2 工作线程；
旧标量入口显式使用 SDK 工作线程，两种传输之间不自动回退。子进程阻塞不能妨碍监管器执行停止。
此处不创建 RTDEControlInterface。"""
import json
import math
import queue
import subprocess
import sys
import threading
import time

from .robot_state import ProducerClock
from .physical_errors import SessionCleanupError

SEQ_REGISTER = 42
ENABLE_REGISTER = 43
SESSION_REGISTER = 44
TARGET_REGISTER = 42
STATE_REGISTER = 42
ACK_REGISTER = 43
OUTPUT_SESSION_REGISTER = 40
OUTPUTS = ['timestamp', 'output_int_register_42', 'output_int_register_43',
           'output_int_register_40', 'actual_TCP_speed']


class AckReceiptClock:
    """保留工作线程首次一致观测到当前 ACK 对的时刻。上层轮询时原始反馈包可能已被覆盖，因此使用该回执保留观测时间；
    不能根据写入已排队就推断控制器已处理，也不能刷新内容未变的确认。"""
    def __init__(self):
        self.key = None
        self.first = None
        self.last = None

    def observe(self, feedback, sampled_at):
        if (not isinstance(feedback, dict) or
                any(name in feedback for name in ('ack_sampled_at', 'worker_sampled_at')) or
                any(type(feedback.get(name)) is not int or
                    not 0 <= feedback[name] < 2**31 for name in ('session', 'sequence')) or
                type(sampled_at) not in (int, float) or not math.isfinite(sampled_at) or
                sampled_at < 0 or self.last is not None and sampled_at < self.last):
            raise ValueError('Invalid worker ACK observation')
        key = feedback['session'], feedback['sequence']
        if key != self.key:
            self.key, self.first = key, sampled_at
        self.last = sampled_at
        return dict(feedback, ack_sampled_at=self.first, worker_sampled_at=sampled_at)


def validate_target(update):
    sequence, enable, session, target = update
    if (type(sequence) is not int or not 0 < sequence < 2**31
            or type(enable) is not int or enable not in (0, 1, 3)
            or type(session) is not int or not 0 < session < 2**31
            or type(target) not in (int, float) or not math.isfinite(target)):
        raise ValueError('Invalid RTDE target/control values')


def write_target(io, update, deadline, clock=time.monotonic):
    """依次使旧数据失效、暂存新值、提交；ur-rtde 会分别发送各个寄存器。"""
    validate_target(update)
    sequence, enable, session, target = update
    writes = ((io.setInputIntRegister, SEQ_REGISTER, -sequence),
              (io.setInputIntRegister, ENABLE_REGISTER, enable),
              (io.setInputIntRegister, SESSION_REGISTER, session),
              (io.setInputDoubleRegister, TARGET_REGISTER, target),
              (io.setInputIntRegister, SEQ_REGISTER, sequence))
    for method, register, value in writes:
        if not math.isfinite(deadline) or clock() >= deadline:
            raise TimeoutError('RTDE target expired before commit')
        if not method(register, value):
            raise RuntimeError('ur-rtde register write failed; restart required')


def validate_spatial_target(update):
    sequence, enable, session, target = update
    if (not isinstance(target, (list, tuple)) or len(target) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in target)):
        raise ValueError('Invalid XYZ target')
    # flag2 请求有限提示动作，flag3 表示完整开场或空间程序进入恢复 HOLD。
    # 旧常驻程序仍拒绝不支持的标志。
    if type(enable) is not int or enable not in (0,1,2,3):
        raise ValueError('Invalid XYZ control flag')
    validate_target((sequence, 1 if enable==2 else enable, session, target[0]))


def write_spatial_target(io, update, deadline, clock=time.monotonic):
    """七次写入依次用于失效标记、标志与会话、XYZ 载荷，最后提交正序号。"""
    validate_spatial_target(update)
    sequence, enable, session, target = update
    writes = [(io.setInputIntRegister, SEQ_REGISTER, -sequence),
              (io.setInputIntRegister, ENABLE_REGISTER, enable),
              (io.setInputIntRegister, SESSION_REGISTER, session)]
    writes += [(io.setInputDoubleRegister, TARGET_REGISTER+i, v) for i, v in enumerate(target)]
    writes += [(io.setInputIntRegister, SEQ_REGISTER, sequence)]
    for method, register, value in writes:
        if not math.isfinite(deadline) or clock() >= deadline:
            raise TimeoutError('XYZ target expired before commit')
        if not method(register, value):
            raise RuntimeError('XYZ register write failed; restart required')


def read_feedback(receive, spatial=False, joint_speeds=False):
    # 各读取函数分别访问库缓存。对比读取前后的时间戳，拒绝跨越两个数据包的快照，以免拼出并不真实的一致回显。
    stamp = receive.getTimestamp()
    feedback = dict(timestamp=stamp,
                    state=receive.getOutputIntRegister(STATE_REGISTER),
                    sequence=receive.getOutputIntRegister(ACK_REGISTER),
                    session=receive.getOutputIntRegister(OUTPUT_SESSION_REGISTER),
                    tcp_speed=receive.getActualTCPSpeed())
    if spatial:
        feedback.update(tcp_pose=receive.getActualTCPPose(), joints=receive.getActualQ())
    if joint_speeds:
        feedback['joint_speed'] = receive.getActualQd()
    if receive.getTimestamp() != stamp:
        return None
    if (not math.isfinite(stamp) or stamp < 0 or len(feedback['tcp_speed']) != 6
            or not all(math.isfinite(x) for x in feedback['tcp_speed'])):
        raise ValueError('Invalid ur-rtde feedback')
    if spatial and any(len(feedback[k]) != 6 or not all(math.isfinite(v) for v in feedback[k])
                       for k in ('tcp_pose', 'joints')):
        raise ValueError('Invalid XYZ feedback')
    if joint_speeds and (len(feedback['joint_speed']) != 6 or
                         not all(math.isfinite(v) for v in feedback['joint_speed'])):
        raise ValueError('Invalid joint-speed feedback')
    return feedback


def worker(host, spatial=False, receive_only=False):
    if spatial:
        from .atomic_rtde_transport import worker as spatial_worker
        return spatial_worker(host,receive_only=receive_only,ack_clock_factory=AckReceiptClock,
                              validate_update=validate_spatial_target)
    return _sdk_worker(host,spatial=spatial,receive_only=receive_only)


def _sdk_worker(host, spatial=False, receive_only=False):
    # 导入模块不联网，只有构造接口对象时才连接主机。
    from rtde_receive import RTDEReceiveInterface

    incoming = queue.Queue(maxsize=1)
    ended = threading.Event()
    read_errors = []

    def read_commands():
        try:
            for line in sys.stdin:
                if len(line) > 4096:
                    raise ValueError('Oversized RTDE command')
                incoming.put_nowait(json.loads(line))
        except Exception as error:
            read_errors.append(error)
        finally:
            ended.set()

    io = receive = None
    try:
        outputs = OUTPUTS + (['actual_TCP_pose', 'actual_q'] if spatial else [])
        if receive_only:
            outputs += ['actual_qd']
        receive = RTDEReceiveInterface(host, 50., outputs, False, True)
        if not receive_only:
            from rtde_io import RTDEIOInterface
            io = RTDEIOInterface(host, False, True)
        threading.Thread(target=read_commands, daemon=True).start()
        previous = -1.
        fresh_at = time.monotonic()
        ack_clock = AckReceiptClock()
        while not ended.is_set():
            if not receive.isConnected() or (io is not None and not io.isConnected()):
                raise RuntimeError('ur-rtde disconnected; explicit restart required')
            feedback = read_feedback(receive, spatial=spatial, joint_speeds=receive_only)
            now = time.monotonic()
            if feedback is not None and feedback['timestamp'] != previous:
                if feedback['timestamp'] < previous:
                    raise RuntimeError('Robot clock reset')
                previous = feedback['timestamp']
                fresh_at = now
                feedback = ack_clock.observe(feedback, now)
                print(json.dumps(dict(feedback=feedback, sampled_at=now)), flush=True)
            if now - fresh_at > .15:
                raise RuntimeError('ur-rtde cached telemetry is stale')
            if ended.is_set():break
            try:
                message = incoming.get_nowait()
            except queue.Empty:
                pass
            else:
                if ended.is_set():break
                if receive_only:
                    raise RuntimeError('Read-only worker cannot write targets')
                writer = write_spatial_target if spatial else write_target
                writer(io, message['update'], message['deadline'])
            ended.wait(.005)
        if read_errors:raise read_errors[0]
    except Exception as error:
        print(json.dumps({'error': str(error)}), flush=True)
        return 1
    finally:
        cleanup_errors=[]
        for resource in (io, receive):
            if resource is not None:
                try:resource.disconnect()
                except Exception as error:cleanup_errors.append(error)
        if cleanup_errors:raise cleanup_errors[0]
    return 0


class RTDEChannel:
    """上层与工作线程之间的桥接采用有限等待，不重连，也不重试运动写入。"""
    def __init__(self, host, startup_timeout=8., spatial=False, receive_only=False):
        self.spatial = spatial
        self.receive_only = receive_only
        if receive_only and not spatial:
            raise ValueError('Read-only mode requires spatial feedback')
        self.closed = False
        self.failure = None
        self.latest = (None, 0.)
        self.pending = queue.Queue(maxsize=1)
        self.ready = threading.Event()
        self.process = subprocess.Popen(
            [sys.executable, '-u', '-m', __name__,
             '--receive-worker' if receive_only else ('--spatial-worker' if spatial else '--worker'), host],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.reader.start()
        self.writer.start()
        try:
            if not self.ready.wait(startup_timeout):
                raise TimeoutError('ur-rtde library startup timed out')
            if self.failure:
                raise RuntimeError(self.failure)
        except BaseException:
            self.close()
            raise

    def _read(self):
        clock = ProducerClock()
        try:
            while line := self.process.stdout.readline(8193):
                if len(line) > 8192 or not line.endswith('\n'):
                    raise ValueError('Invalid RTDE worker output')
                # 原生库也可能把诊断信息输出到标准输出。
                if not line.startswith('{'):
                    continue
                message = json.loads(line)
                if 'error' in message:
                    raise RuntimeError(message['error'])
                feedback = message['feedback']
                sampled = message['sampled_at']
                if not math.isfinite(sampled) or sampled > time.monotonic() + .001:
                    raise ValueError('Invalid worker clock')
                stamp = clock.observe(round(feedback['timestamp']*1_000_000), sampled)
                self.latest = feedback, stamp
                self.ready.set()
            if not self.closed:
                raise RuntimeError('ur-rtde worker exited')
        except Exception as error:
            if not self.closed:
                self.failure = str(error)
            self.ready.set()

    def _write(self):
        try:
            while not self.closed:
                try:
                    message = self.pending.get(timeout=.05)
                except queue.Empty:
                    continue
                if self.closed:break
                if self.failure or time.monotonic() >= message['deadline']:
                    raise RuntimeError('RTDE target expired or transport failed')
                self.process.stdin.write(json.dumps(message) + '\n')
                self.process.stdin.flush()
        except Exception as error:
            if not self.closed:
                self.failure = str(error)

    def send(self, sequence, enable, session, target):
        if self.receive_only:
            raise RuntimeError('Read-only channel cannot send targets')
        if self.closed or self.failure:
            raise RuntimeError(self.failure or 'RTDE connection closed')
        update = (sequence, enable, session, target)
        (validate_spatial_target if self.spatial else validate_target)(update)
        now = time.monotonic()
        if not 0 <= now - self.latest[1] <= .15:
            self.failure = 'RTDE feedback stale'
            raise RuntimeError(self.failure)
        try:
            self.pending.put_nowait(dict(update=update, deadline=now+.15))
        except queue.Full:
            self.failure = 'RTDE command backpressure; restart required'
            raise RuntimeError(self.failure) from None

    def close_completed(self):
        """仅当开场经验证已完成时，才可请求对端配合正常结束数据流。"""
        if self.closed:raise SessionCleanupError('RTDE input already stopped before completed retirement')
        started=time.monotonic();deadline=started+1.8
        phase='writer';category='operation'
        def remaining(limit):
            value=deadline-time.monotonic()
            if value<=0:raise TimeoutError('Completed RTDE retirement deadline expired')
            return min(limit,value)
        self.closed=True
        try:
            self.writer.join(timeout=remaining(.2))
            if self.writer.is_alive():raise TimeoutError('RTDE parent writer did not quiesce')
            if self.failure:
                category='prior_failure'
                raise RuntimeError('RTDE worker had failed before completed retirement')
            phase='eof'
            eof_done=threading.Event();eof_errors=[]
            def close_input():
                try:self.process.stdin.close()
                except Exception as error:eof_errors.append(error)
                finally:eof_done.set()
            self._eof_closer=threading.Thread(target=close_input,name='completed-rtde-eof',daemon=True)
            self._eof_closer.start()
            if not eof_done.wait(remaining(.2)):raise TimeoutError('RTDE input EOF did not complete')
            if eof_errors:
                category='io_error'
                raise RuntimeError('RTDE input EOF failed') from eof_errors[0]
            self._eof_closer.join(timeout=remaining(.05))
            if self._eof_closer.is_alive():raise TimeoutError('RTDE EOF worker did not close')
            phase='process'
            if self.process.wait(timeout=remaining(1.5))!=0:
                category='nonzero_exit'
                raise RuntimeError('RTDE worker cleanup failed')
            phase='reader'
            self.reader.join(timeout=remaining(.3))
            if self.reader.is_alive():raise TimeoutError('RTDE parent reader did not close')
            phase='stdout'
            self.process.stdout.close()
            remaining(1.8)
        except Exception as error:
            details={'phase':phase,
                     'category':'timeout' if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else category,
                     'duration_s':round(min(60.,max(0.,time.monotonic()-started)),3),
                     'exit_code':self.process.poll()}
            try:self.close()
            except Exception as cleanup_error:
                failure=SessionCleanupError('Completed RTDE retirement and hard cleanup failed')
                failure.retirement_details=details
                raise failure from cleanup_error
            failure=SessionCleanupError('Completed RTDE retirement failed')
            failure.retirement_details=details
            raise failure from error

    def close(self):
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        self.reader.join(timeout=.3)
        self.writer.join(timeout=.3)
        eof_closer=getattr(self,'_eof_closer',None)
        if eof_closer is None or not eof_closer.is_alive():self.process.stdin.close()
        self.process.stdout.close()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('--worker', '--spatial-worker', '--receive-worker'):
        raise SystemExit('Internal ur-rtde worker; use leica_tracker.resident_follow')
    raise SystemExit(worker(sys.argv[2], spatial=sys.argv[1] != '--worker',
                          receive_only=sys.argv[1] == '--receive-worker'))
