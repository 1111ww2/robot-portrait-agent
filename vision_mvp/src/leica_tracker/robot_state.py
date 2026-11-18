"""只读PolyScope 5主遥测，独立于运动传输。"""
import math
import socket
import struct
import threading
import time


class ProducerClock:
    """保留逐步累积的传输延迟，不刷新排队中样本的时间。"""
    def __init__(self):
        self.previous = -1
        self.offset = float("inf")

    def observe(self, microseconds, received):
        if microseconds <= self.previous:
            raise ValueError("Robot timestamp stopped advancing")
        self.previous = microseconds
        seconds = microseconds / 1_000_000
        self.offset = min(self.offset, received - seconds)
        return seconds + self.offset


def text_marker(body):
    if len(body) >= 10 and body[9] == 0:
        message = body[10:].decode("ascii", errors="replace").strip("\x00")
        if message.startswith(("LEICA_ACK_", "LEICA_FAULT_")):
            return message
    return None


def parse_state(body):
    result = {}
    offset = 0
    while offset < len(body):
        if len(body) - offset < 5:
            raise ValueError("Truncated state header")
        size, kind = struct.unpack_from("!IB", body, offset)
        if size < 5 or offset + size > len(body):
            raise ValueError("Invalid subpacket length")
        packet = body[offset:offset + size]
        if kind == 0:
            if size < 47:
                raise ValueError("Unsupported robot mode layout")
            values = struct.unpack_from("!Q7?BB3d", packet, 5)
            result["mode"] = dict(zip(
                ("timestamp", "connected", "enabled", "power", "emergency", "protective",
                 "running", "paused", "mode", "control", "fraction", "scaling", "limit"), values))
        elif kind == 4:
            if size < 101:
                raise ValueError("Truncated Cartesian data")
            values = struct.unpack_from("!12d", packet, 5)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite Cartesian data")
            result["pose"] = list(values[:6])
            result["tcp_offset"] = list(values[6:])
        offset += size
    if "mode" not in result or "pose" not in result:
        raise ValueError("Missing state fields")
    return result


class StateReceiver:
    def __init__(self, host):
        self.sock = socket.create_connection((host, 30011), timeout=2)
        # 连接设置期间，第一状态可能会延迟；运行时的新鲜度仍由主管独立检查，与此启动宽限期无关。
        self.sock.settimeout(2.0)
        self.latest = (None, 0.0)
        self.ack = (None, 0.0)
        self.notice = None
        self.failure = None
        self.closed = False
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _exact(self, size):
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise EOFError("Robot state stream closed")
            data += chunk
        return data

    def _read(self):
        clock = ProducerClock()
        try:
            while not self.closed:
                size, kind = struct.unpack("!IB", self._exact(5))
                if not 5 <= size <= 65536:
                    raise ValueError("Invalid state message length")
                body = self._exact(size - 5)
                if kind == 20:
                    marker = text_marker(body)
                    if marker and marker.startswith('LEICA_FAULT_'):
                        self.notice = marker
                    elif marker:
                        self.ack = (marker, time.monotonic())
                if kind != 16:
                    continue
                current = parse_state(body)
                timestamp = current["mode"]["timestamp"]
                self.latest = (current, clock.observe(timestamp, time.monotonic()))
                self.sock.settimeout(0.5)
        except Exception as error:
            if not self.closed:
                phase = 'startup' if self.latest[0] is None else 'runtime'
                self.failure = f'Robot state {phase}: {error}'

    def close(self):
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.thread.join(timeout=1)
