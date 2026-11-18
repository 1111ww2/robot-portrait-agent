"""跟随本任务的相机日志，只读取新增记录，不重放或伪造更新时刻。"""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

import pytest

from leica_tracker.ursim_source import TailEventStream


@pytest.fixture
def source_path():
    root = Path(__file__).resolve().parents[2] / 'tmp/test-ursim-source'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='case-', dir=root) as directory:
        path = Path(directory) / 'camera.jsonl'
        path.write_bytes(b'')
        yield path


def append(path, data):
    with path.open('ab', buffering=0) as stream:
        stream.write(data)


def background_read(stream, size=8193):
    result = {}; done = threading.Event()
    def read():
        try: result['value'] = stream.readline(size)
        except Exception as exc: result['error'] = exc
        finally: done.set()
    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return result, done, thread


def test_old_records_are_ignored_and_only_new_complete_records_are_returned(source_path):
    source_path.write_bytes(b'{"old":true}\n')
    stream = TailEventStream(source_path, is_alive=lambda: True)
    result, done, thread = background_read(stream)
    try:
        assert not done.wait(.03)
        append(source_path, b'{"new":1}\n{"new":2}\n')
        assert done.wait(1.)
        assert result == {'value': '{"new":1}\n'}
        assert stream.readline() == '{"new":2}\n'
    finally:
        stream.close(); thread.join(1.)


def test_partial_record_waits_for_newline_instead_of_emitting_stale_fragment(source_path):
    stream = TailEventStream(source_path, is_alive=lambda: True)
    result, done, thread = background_read(stream)
    try:
        append(source_path, b'{"new":')
        assert not done.wait(.03)
        append(source_path, b'3}')
        assert not done.wait(.03)
        append(source_path, b'\n')
        assert done.wait(1.) and result == {'value': '{"new":3}\n'}
    finally:
        stream.close(); thread.join(1.)


def test_fragment_already_in_progress_at_attach_is_not_a_new_record(source_path):
    source_path.write_bytes(b'{"old":')
    stream = TailEventStream(source_path, is_alive=lambda: True)
    try:
        append(source_path, b'7}\n{"fresh":9}\n')
        assert stream.readline() == '{"fresh":9}\n'
    finally: stream.close()


@pytest.mark.parametrize('external', [False, True])
def test_close_or_stop_unblocks_a_live_writer_eof(source_path, external):
    stop = threading.Event()
    stream = TailEventStream(source_path, is_alive=lambda: True, stop_event=stop)
    result, done, thread = background_read(stream)
    try:
        assert not done.wait(.03)
        if external: stop.set()
        else: stream.close()
        assert done.wait(1.) and result == {'value': ''}
    finally:
        stream.close(); thread.join(1.)


def test_dead_writer_returns_eof_without_replaying_or_returning_a_partial_line(source_path):
    alive = threading.Event(); alive.set()
    stream = TailEventStream(source_path, is_alive=alive.is_set)
    result, done, thread = background_read(stream)
    try:
        append(source_path, b'{"incomplete":')
        assert not done.wait(.03)
        alive.clear()
        assert done.wait(1.) and result == {'value': ''}
    finally:
        stream.close(); thread.join(1.)


def test_truncation_is_rejected_instead_of_starting_again(source_path):
    stream = TailEventStream(source_path, is_alive=lambda: True)
    try:
        append(source_path, b'{"first":1}\n')
        assert stream.readline() == '{"first":1}\n'
        source_path.write_bytes(b'')
        with pytest.raises(RuntimeError, match='truncat'):
            stream.readline()
    finally: stream.close()


def test_replacement_file_identity_is_rejected(source_path, monkeypatch):
    stream = TailEventStream(source_path, is_alive=lambda: True)
    original = Path.stat
    def changed(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == source_path:
            return SimpleNamespace(st_dev=value.st_dev, st_ino=value.st_ino + 1, st_size=value.st_size)
        return value
    monkeypatch.setattr(Path, 'stat', changed)
    try:
        with pytest.raises(RuntimeError, match='replac|rotat'):
            stream.readline()
    finally: stream.close()


@pytest.mark.parametrize('newline', [False, True])
def test_oversize_input_is_rejected_with_bounded_partial_buffer(source_path, newline):
    stream = TailEventStream(source_path, is_alive=lambda: True)
    try:
        append(source_path, b'x' * 8193 + (b'\n' if newline else b''))
        with pytest.raises(ValueError, match='size|long'):
            stream.readline()
    finally: stream.close()


def test_readline_size_limits_the_record_including_newline(source_path):
    stream = TailEventStream(source_path, is_alive=lambda: True)
    try:
        append(source_path, b'{}\n')
        with pytest.raises(ValueError, match='size|long'):
            stream.readline(3)
    finally: stream.close()


def test_latest_event_source_stays_live_at_eof_and_never_restamps_records(source_path):
    import json
    import time
    from leica_tracker.physical_runtime import LatestEventSource
    stream = TailEventStream(source_path, is_alive=lambda: True)
    source = LatestEventSource(stream)
    try:
        time.sleep(.03)
        assert source.failure is None and source.take() is None
        event = {'event': 'physical_intro', 'source_monotonic_ns': 123}
        append(source_path, (json.dumps(event) + '\n').encode())
        deadline = time.monotonic() + 1.
        value = None
        while value is None and time.monotonic() < deadline:
            value = source.take(); time.sleep(.005)
        assert value == event
    finally:
        stream.close(); source.thread.join(1.)
    assert not source.thread.is_alive()
