"""旧的原始 RTDE 传输仅供查看历史实验诊断。
实时常驻跟随使用 ur_rtde_transport（ur-rtde 1.6.5）。
两者不自动切换，旧通道不能和新寄存器协议共用会话。"""
import math
import socket
import struct
import threading
import time

from .robot_state import ProducerClock

INPUT_NAMES = 'input_int_register_24,input_int_register_25,input_int_register_26,input_double_register_24'
INPUT_TYPES = 'INT32,INT32,INT32,DOUBLE'
OUTPUT_NAMES = 'timestamp,output_int_register_24,output_int_register_25,output_int_register_26,actual_TCP_speed'
OUTPUT_TYPES = 'DOUBLE,INT32,INT32,INT32,VECTOR6D'


class PacketReader:
    def __init__(self, sock):
        self.sock = sock

    def exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise EOFError('RTDE connection closed')
            data.extend(chunk)
        return bytes(data)

    def read(self):
        size, kind = struct.unpack('!HB', self.exact(3))
        if not 3 <= size <= 8192:
            raise ValueError('Invalid RTDE packet length')
        return kind, self.exact(size - 3)


def packet(kind, body=b''):
    return struct.pack('!HB', len(body) + 3, kind) + body


def require_recipe(body, expected):
    if len(body) < 2 or body[0] == 0 or body[1:].decode('ascii') != expected:
        raise RuntimeError(f'RTDE recipe unavailable or already owned: {body!r}')
    return body[0]


def input_packet(recipe, sequence, enable, session, target):
    if (not math.isfinite(target) or enable not in (0, 1) or not 0 < sequence < 2**31
            or not 0 < session < 2**31):
        raise ValueError('Invalid RTDE target/control values')
    return packet(85, struct.pack('!Biiid', recipe, sequence, enable, session, target))


def parse_feedback(body, recipe):
    if len(body) != 69 or body[0] != recipe:
        raise ValueError('Invalid RTDE feedback recipe or size')
    _, stamp, state, sequence, session, *speed = struct.unpack('!Bdiii6d', body)
    if not all(math.isfinite(v) for v in [stamp, *speed]):
        raise ValueError('Invalid RTDE clock')
    return dict(timestamp=stamp, state=state, sequence=sequence, session=session, tcp_speed=speed)


class RTDEChannel:
    def __init__(self, host):
        self.sock = socket.create_connection((host, 30004), timeout=.5)
        self.sock.settimeout(.5)
        self.reader = PacketReader(self.sock)
        self.closed = False
        self.failure = None
        self.latest = (None, 0.0)
        self.thread = None
        try:
            if self._request(86, struct.pack('!H', 2)) != b'\x01':
                raise RuntimeError('RTDE v2 not accepted')
            self.output_recipe = require_recipe(self._request(79, struct.pack('!d', 50.) + OUTPUT_NAMES.encode()), OUTPUT_TYPES)
            self.input_recipe = require_recipe(self._request(73, INPUT_NAMES.encode()), INPUT_TYPES)
            if self._request(83) != b'\x01':
                raise RuntimeError('RTDE start rejected')
            self.sock.settimeout(.1)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except Exception:
            self.close()
            raise

    def _request(self, kind, body=b''):
        self.sock.sendall(packet(kind, body))
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline:
            reply, payload = self.reader.read()
            if reply == kind:
                return payload
            if reply != 77:
                raise RuntimeError(f'Unexpected RTDE setup response: {reply}')
        raise TimeoutError('RTDE setup timeout')

    def _read(self):
        clock = ProducerClock()
        try:
            while not self.closed:
                kind, body = self.reader.read()
                if kind == 85:
                    feedback = parse_feedback(body, self.output_recipe)
                    stamp = clock.observe(round(feedback['timestamp'] * 1_000_000), time.monotonic())
                    self.latest = (feedback, stamp)
        except Exception as error:
            if not self.closed:
                self.failure = str(error)

    def send(self, sequence, enable, session, target):
        if self.closed or self.failure:
            raise RuntimeError(self.failure or 'RTDE connection closed')
        data = input_packet(self.input_recipe, sequence, enable, session, target)
        try:
            self.sock.sendall(data)
        except OSError as error:
            self.failure = str(error)
            self.close()
            raise

    def close(self):
        self.closed = True
        self.sock.close()
        if self.thread:
            self.thread.join(timeout=.2)
