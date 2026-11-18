"""按帧批量传输空间控制记录，使用 RTDE v2 协议；导入模块时不执行输入输出。
通过同一套接字调用一次 sendall，先发送负序号的完整暂存记录，再发送内容相同、序号为正的提交记录。
由常驻程序保护事务，不假定寄存器支持未公开的原子操作。不使用 SDK 控制、标量回退、重连或命令重放。"""
import json
import math
import queue
import socket
import struct
import sys
import threading
import time

INPUT_NAMES = ('input_int_register_42', 'input_int_register_43', 'input_int_register_44',
               'input_double_register_42', 'input_double_register_43', 'input_double_register_44')
INPUT_TYPES = 'INT32,INT32,INT32,DOUBLE,DOUBLE,DOUBLE'
OUTPUT_NAMES = ('timestamp', 'output_int_register_42', 'output_int_register_43',
                'output_int_register_40', 'actual_TCP_speed', 'actual_TCP_pose', 'actual_q')
OUTPUT_TYPES = 'DOUBLE,INT32,INT32,INT32,VECTOR6D,VECTOR6D,VECTOR6D'
MAX_PACKET = 8192


def packet(kind, body=b''):
    return struct.pack('!HB', len(body)+3, kind)+body


def _recipe(body, expected, *, phase='unknown',mode='unknown'):
    if not body or body[0] == 0 or body[1:] != expected.encode('ascii'):
        # 仅输出协议标记和已限制范围的位置数值，避免暴露控制器原文；严格遵循上面的验收条件。
        allowed={b'BOOL',b'UINT8',b'UINT32',b'UINT64',b'INT32',b'DOUBLE',
                 b'VECTOR3D',b'VECTOR6D',b'VECTOR6INT32',b'VECTOR6UINT32',
                 b'IN_USE',b'NOT_FOUND'}
        tokens=body[1:].split(b',') if len(body)>1 else []
        wanted=expected.encode('ascii').split(b',')
        mismatch=[index for index in range(min(16,max(len(tokens),len(wanted))))
                  if (tokens[index] if index<len(tokens) else None)
                  != (wanted[index] if index<len(wanted) else None)]
        if not body:reason='empty'
        elif b'IN_USE' in tokens:reason='in_use'
        elif b'NOT_FOUND' in tokens:reason='missing'
        elif not tokens or len(tokens)>len(wanted) or any(token not in allowed for token in tokens):
            reason='malformed'
        elif len(tokens)<len(wanted):reason='missing'
        elif body[0]==0:reason='zero'
        else:reason='mismatch'
        details={'phase':phase if phase in ('input','output') else 'unknown',
                 'mode':mode if mode in ('writer','receive_only') else 'unknown',
                 'reason':reason,'recipe_id':body[0] if body else None,
                 'token_count':len(tokens),'expected_count':len(wanted),
                 'tokens':[token.decode('ascii') if token in allowed else 'OTHER' for token in tokens[:16]],
                 'mismatch_indices':mismatch}
        raise RuntimeError('Spatial RTDE recipe rejected, unavailable or already owned '+
                           json.dumps(details,sort_keys=True,separators=(',',':')))
    return body[0]


def encode_spatial_batch(recipe, update, validate_update):
    if type(recipe) is not int or not 0 < recipe < 256 or not callable(validate_update):
        raise ValueError('Negotiated spatial recipe and shared validator required')
    validate_update(update)
    sequence, enable, session, target = update
    # 仅复制已验证的三个位置，避免二次遍历可变列表或子类迭代器，也避免读取无界迭代器。
    target = tuple(target[index] for index in range(3))
    validate_update((sequence, enable, session, target))
    # 两条配方记录都以序号开头。在有序流上，先完整处理负记录，再处理内容对应的正记录。
    return b''.join(packet(85, struct.pack('!Biiiddd', recipe, sign*sequence,
        enable, session, *target)) for sign in (-1, 1))


def parse_feedback(body, recipe, *, receive_only=False):
    values = 24 if receive_only else 18
    format_string = '!Bdiii'+str(values)+'d'
    if len(body) != struct.calcsize(format_string):
        raise ValueError('Spatial RTDE output frame length differs from its recipe')
    selected, stamp, state, sequence, session, *vectors = struct.unpack(format_string, body)
    if (selected != recipe or not math.isfinite(stamp) or stamp < 0 or
            not 0 <= sequence < 2**31 or not 0 <= session < 2**31 or
            not all(math.isfinite(value) for value in vectors)):
        raise ValueError('Invalid spatial RTDE output values or recipe')
    feedback = dict(timestamp=stamp, state=state, sequence=sequence, session=session,
        tcp_speed=vectors[:6], tcp_pose=vectors[6:12], joints=vectors[12:18])
    if receive_only:
        feedback['joint_speed'] = vectors[18:24]
    return feedback


def _text_message(body):
    # 验证第二版协议的文本分帧，不暴露任意控制器文本。
    if not body:
        raise ValueError('Invalid spatial RTDE text frame')
    first = 1+body[0]
    if first >= len(body):
        raise ValueError('Invalid spatial RTDE text frame')
    last = first+1+body[first]
    if last+1 != len(body) or body[last] > 3:
        raise ValueError('Invalid spatial RTDE text frame')
    if body[last] <= 1:
        raise RuntimeError('Controller reported a spatial RTDE protocol error')


class SpatialRTDEChannel:
    """显式管理单次连接的生命周期，仅解码完整帧。"""
    def __init__(self, host, *, receive_only=False, validate_update=None,
                 socket_factory=socket.create_connection, clock=time.monotonic):
        if type(receive_only) is not bool or not callable(clock) or (not receive_only and not callable(validate_update)):
            raise ValueError('Explicit spatial mode, clock and shared update validator required')
        self.receive_only, self.validate_update, self.clock = receive_only, validate_update, clock
        self.closed = False
        self.sock = None
        self.buffer = bytearray()
        self.startup_feedback = []
        self.write_lock = threading.Lock()
        try:
            self.sock = socket_factory((host, 30004), timeout=.5)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # 官方控制器的版本协商可能超过 500 毫秒。启动各阶段共用一个截止时间，分片读取和后续请求都不能延长预算。
            self.startup_deadline = self.clock()+3.
            if self._request(86, struct.pack('!H', 2)) != b'\x01':
                raise RuntimeError('Spatial RTDE protocol v2 rejected')
            names = OUTPUT_NAMES+(('actual_qd',) if receive_only else ())
            types = OUTPUT_TYPES+(',VECTOR6D' if receive_only else '')
            mode = 'receive_only' if receive_only else 'writer'
            self.output_recipe = _recipe(self._request(79,
                struct.pack('!d', 50.)+','.join(names).encode('ascii')), types,phase='output',mode=mode)
            self.input_recipe = None
            if not receive_only:
                self.input_recipe = _recipe(self._request(73, ','.join(INPUT_NAMES).encode('ascii')),
                                           INPUT_TYPES,phase='input',mode=mode)
            if self._request(83) != b'\x01':
                raise RuntimeError('Spatial RTDE synchronization rejected')
        except BaseException:
            self.close()
            raise

    def _remaining(self, deadline):
        remaining = deadline-self.clock()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError('Spatial RTDE absolute deadline expired')
        return remaining

    def _exact(self, size, deadline):
        data = bytearray()
        while len(data) < size:
            self.sock.settimeout(self._remaining(deadline))
            part = self.sock.recv(size-len(data))
            if not part:
                raise EOFError('Spatial RTDE connection closed')
            data.extend(part)
            self._remaining(deadline)
        return bytes(data)

    def _request(self, kind, body=b''):
        deadline = self.startup_deadline
        self.sock.settimeout(self._remaining(deadline))
        self.sock.sendall(packet(kind, body))
        self._remaining(deadline)
        while True:
            size, reply = struct.unpack('!HB', self._exact(3, deadline))
            if not 3 <= size <= MAX_PACKET:
                raise ValueError('Invalid spatial RTDE handshake frame length')
            payload = self._exact(size-3, deadline)
            if reply == kind:
                return payload
            if kind == 83 and reply == 85:
                self.startup_feedback.append(parse_feedback(payload, self.output_recipe,
                    receive_only=self.receive_only))
                continue
            if reply != 77:
                raise RuntimeError('Unexpected spatial RTDE handshake response')
            _text_message(payload)

    def poll(self, timeout=.005):
        if self.closed:
            raise RuntimeError('Spatial RTDE connection is closed')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= .15:
            raise ValueError('Bounded spatial RTDE poll timeout required')
        if self.startup_feedback:
            result, self.startup_feedback = self.startup_feedback, []
            return result
        self.sock.settimeout(timeout)
        try:
            part = self.sock.recv(MAX_PACKET)
        except socket.timeout:
            return []
        if not part:
            raise EOFError('Spatial RTDE connection closed')
        self.buffer.extend(part)
        feedback = []
        while len(self.buffer) >= 3:
            size, kind = struct.unpack('!HB', self.buffer[:3])
            if not 3 <= size <= MAX_PACKET:
                raise ValueError('Invalid spatial RTDE runtime frame length')
            if len(self.buffer) < size:
                break
            body = bytes(self.buffer[3:size])
            del self.buffer[:size]
            if kind == 85:
                feedback.append(parse_feedback(body, self.output_recipe, receive_only=self.receive_only))
            elif kind == 77:
                _text_message(body)
            else:
                raise RuntimeError('Unexpected spatial RTDE runtime frame type')
        return feedback

    def send(self, update, deadline):
        if self.receive_only:
            raise RuntimeError('Read-only spatial RTDE channel cannot send targets')
        if self.closed:
            raise RuntimeError('Spatial RTDE connection is closed')
        data = encode_spatial_batch(self.input_recipe, update, self.validate_update)
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('Finite parent commit deadline required')
        # 沿用更早的上层截止时间。等待工作线程、锁或部分套接字操作后，不重新分配写入时间。
        deadline = min(deadline, self.clock()+.15)
        if not self.write_lock.acquire(timeout=self._remaining(deadline)):
            self.close()
            raise TimeoutError('Spatial RTDE writer was busy beyond its commit deadline')
        try:
            if self.closed:
                raise RuntimeError('Spatial RTDE connection is closed')
            self.sock.settimeout(self._remaining(deadline))
            self.sock.sendall(data)
            self._remaining(deadline)
        except BaseException:
            self.close()
            raise
        finally:
            self.write_lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                self.sock.close()


def _command(message, validate_update):
    if (not isinstance(message, dict) or set(message) != {'update', 'deadline'} or
            type(message['deadline']) not in (int, float) or not math.isfinite(message['deadline'])):
        raise ValueError('Invalid bounded spatial worker command')
    validate_update(message['update'])
    return message


def worker(host, *, receive_only=False, ack_clock_factory, validate_update,
           channel_factory=SpatialRTDEChannel, input_stream=None, output_stream=None,
           clock=time.monotonic):
    """一次性使用的底层工作线程，共用注入的确认时钟和校验逻辑。"""
    if not callable(ack_clock_factory) or not callable(validate_update):
        raise ValueError('Shared spatial ACK clock and update validator required')
    source = sys.stdin if input_stream is None else input_stream
    output = sys.stdout if output_stream is None else output_stream
    incoming = queue.Queue(maxsize=1)
    ended = threading.Event()
    read_errors = []

    def read_commands():
        try:
            for line in source:
                if len(line) > 4096 or not line.endswith('\n'):
                    raise ValueError('Invalid spatial worker input line')
                if receive_only:
                    raise RuntimeError('Read-only spatial worker cannot accept targets')
                incoming.put_nowait(_command(json.loads(line), validate_update))
        except Exception as error:
            read_errors.append(error)
        finally:
            ended.set()

    channel = None
    try:
        channel = channel_factory(host, receive_only=receive_only, validate_update=validate_update, clock=clock)
        ack_clock = ack_clock_factory()
        threading.Thread(target=read_commands, daemon=True).start()
        previous = -1.
        fresh_at = clock()
        while not ended.is_set():
            for feedback in channel.poll(timeout=.005):
                now = clock()
                if feedback['timestamp'] < previous:
                    raise RuntimeError('Spatial RTDE robot clock reset')
                if feedback['timestamp'] != previous:
                    previous = feedback['timestamp']
                    fresh_at = now
                    observed = ack_clock.observe(feedback, now)
                    print(json.dumps(dict(feedback=observed, sampled_at=now)), file=output, flush=True)
            if clock()-fresh_at > .15:
                raise RuntimeError('Spatial RTDE telemetry is stale')
            if ended.is_set():break
            try:
                message = incoming.get_nowait()
            except queue.Empty:
                continue
            if ended.is_set():break
            channel.send(message['update'], message['deadline'])
        if read_errors:
            raise read_errors[0]
    except Exception as error:
        print(json.dumps(dict(error=str(error))), file=output, flush=True)
        return 1
    finally:
        if channel is not None:
            channel.close()
    return 0
