"""从拥有的摄像头日志中跟踪新记录，无需重播或重新盖章。调用者提供捕获的写入者的实时检查。
该模块既不会启动进程也不会打开网络连接，也不会编造事件时间。"""
import os
from pathlib import Path
import threading


class TailEventStream:
    """为 LatestEventSource 提供阻塞式文本 readline 接口，跳过打开前已有的数据。
    只要本任务的写入进程仍在运行，读到文件末尾只是暂时没有新数据；
    关闭、外部取消或写入进程退出时才返回真正的 EOF。
    size 用于识别无效记录，与 LatestEventSource 每次读取 8193 字节、单条记录最多 8192 字节（含换行）的约定一致。"""
    def __init__(self, sourcepath, *, is_alive, stop_event=None):
        if not callable(is_alive):
            raise ValueError('An owned camera writer liveness check is required')
        if stop_event is not None and not callable(getattr(stop_event, 'is_set', None)):
            raise ValueError('A cancellation event is required')
        self.path = Path(sourcepath)
        self.is_alive = is_alive
        self.stop_event = stop_event
        self._closed = threading.Event()
        self._read_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._buffer = bytearray()
        self._file = self.path.open('rb', buffering=0)
        try:
            stat = os.fstat(self._file.fileno())
            self._identity = (stat.st_dev, stat.st_ino)
            self._position = self._file.seek(0, os.SEEK_END)
            self._discard_initial_fragment = False
            if self._position:
                self._file.seek(-1, os.SEEK_END)
                self._discard_initial_fragment = self._file.read(1) != b'\n'
                self._file.seek(self._position)
            self._validate_file()
        except Exception:
            self._file.close()
            raise

    def _validate_file(self):
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise RuntimeError('Camera guidance log disappeared or was replaced') from exc
        if (stat.st_dev, stat.st_ino) != self._identity:
            raise RuntimeError('Camera guidance log was rotated or replaced')
        if stat.st_size < self._position:
            raise RuntimeError('Camera guidance log was truncated')

    def _stopped(self):
        return (self._closed.is_set()
                or self.stop_event is not None and self.stop_event.is_set()
                or not self.is_alive())

    def readline(self, size=8193):
        if type(size) is not int or not 1 <= size <= 8193:
            raise ValueError('Guidance record size must be from 1 to 8193 bytes')
        with self._read_lock:
            while True:
                if self._stopped():
                    return ''
                with self._file_lock:
                    if self._closed.is_set():
                        return ''
                    self._validate_file()
                    newline = self._buffer.find(b'\n')
                    if newline >= 0:
                        length = newline + 1
                        if length >= size:
                            raise ValueError('Camera guidance record exceeds its size limit')
                        line = bytes(self._buffer[:length])
                        del self._buffer[:length]
                        if self._discard_initial_fragment:
                            self._discard_initial_fragment = False
                            continue
                        return line.decode('utf-8', errors='strict')
                    if len(self._buffer) >= size:
                        raise ValueError('Camera guidance partial record exceeds its size limit')
                    data = self._file.read(size - len(self._buffer))
                    if data:
                        self._position += len(data)
                        self._buffer.extend(data)
                        continue
                # 文件读取不会在EOF处阻塞。稍等片刻，不要将空/部分记录作为新鲜的相机输入。
                self._closed.wait(.01)

    def close(self):
        """在EOF处解锁一个读者，并仅关闭此适配器的文件句柄。"""
        self._closed.set()
        with self._file_lock:
            self._file.close()
