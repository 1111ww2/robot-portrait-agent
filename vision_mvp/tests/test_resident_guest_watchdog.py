import threading
import time
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from leica_tracker.resident_guest_watchdog import (ResidentMotionGate, GuestControlWatchdog,
                                                   LocalGuestControlReader)


class Channel:
    def __init__(self):
        self.sent = []
        self.closed = threading.Event()

    def send(self, *args):
        self.sent.append(args)

    def close(self):
        self.closed.set()


def snapshot(epoch=1, active=True):
    return {'instance': 'guest-a', 'epoch': epoch, 'active': active,
            'camera_connected': True, 'camera_fresh': True, 'control_fresh': True}


@pytest.mark.parametrize('field', ['active','control_fresh','camera_connected','camera_fresh'])
def test_invalid_snapshot_reports_specific_boolean_and_preserves_first_numeric_flags(field):
    channel=Channel();gate=ResidentMotionGate(channel)
    state=snapshot();state[field]=False
    watchdog=GuestControlWatchdog(lambda:dict(state),gate,interval=.01,timeout=.1)
    watchdog.start()
    assert watchdog.stopped.wait(1)
    assert channel.closed.wait(1) and not gate.can_send
    failure=watchdog.failure
    assert failure.startswith('Guest snapshot failed: Guest snapshot invalid or inactive')
    assert field+'_not_true' in failure
    assert watchdog.failure_snapshot[field]==0
    assert watchdog.failure_snapshot['identity_valid']==1
    assert all(type(value) is int for value in watchdog.failure_snapshot.values())
    state.update(snapshot())
    watchdog.close()
    assert watchdog.failure==gate.reason==failure
    assert watchdog.failure_snapshot[field]==0


@pytest.mark.parametrize('bad', [None,{},snapshot(epoch=True),{**snapshot(),'instance':''}])
def test_invalid_identity_diagnostic_does_not_dump_snapshot_values(bad):
    channel=Channel();gate=ResidentMotionGate(channel)
    watchdog=GuestControlWatchdog(lambda:bad,gate,interval=.01,timeout=.1)
    watchdog.start()
    assert watchdog.stopped.wait(1)
    assert 'invalid_identity' in watchdog.failure
    assert watchdog.failure_snapshot['identity_valid']==0
    assert 'guest-a' not in watchdog.failure
    watchdog.close()


def test_snapshot_diagnostic_reports_all_failed_fields_together():
    channel=Channel();gate=ResidentMotionGate(channel)
    state={**snapshot(),'active':False,'control_fresh':False,
           'camera_connected':False,'camera_fresh':False}
    watchdog=GuestControlWatchdog(lambda:state,gate,interval=.01,timeout=.1)
    watchdog.start()
    assert watchdog.stopped.wait(1)
    assert all(field+'_not_true' in watchdog.failure for field in
               ('active','control_fresh','camera_connected','camera_fresh'))
    watchdog.close()


def test_stop_closes_real_interface_channel_and_latches_motion_gate():
    channel = Channel()
    gate = ResidentMotionGate(channel)
    assert not gate.can_send
    with pytest.raises(RuntimeError):
        gate.send(1, 1, 1, .1)
    gate.arm()
    assert gate.can_send
    gate.send(1, 1, 1, .1)
    gate.stop('guest stopped')
    assert not gate.can_send
    assert channel.closed.wait(1)
    with pytest.raises(RuntimeError):
        gate.send(2, 1, 1, .2)
    with pytest.raises(RuntimeError):
        gate.arm()
    assert channel.sent == [(1, 1, 1, .1)]


def test_epoch_change_stops_without_dependence_on_camera_loop():
    channel = Channel()
    gate = ResidentMotionGate(channel)
    state = snapshot()
    watchdog = GuestControlWatchdog(lambda: dict(state), gate, interval=.01, timeout=.12)
    watchdog.start()
    try:
        assert watchdog.wait_ready(1)
        gate.send(1, 1, 1, .1)
        state['epoch'] = 2
        assert watchdog.stopped.wait(1)
        assert channel.closed.wait(1)
        assert 'changed' in watchdog.failure
        with pytest.raises(RuntimeError):
            gate.send(2, 1, 1, .2)
    finally:
        watchdog.close()


def test_blocked_snapshot_does_not_block_watchdog_from_closing_channel():
    channel = Channel()
    gate = ResidentMotionGate(channel)
    blocked = threading.Event()
    entered = threading.Event()
    calls = []

    def read():
        calls.append(1)
        if len(calls) > 1:
            entered.set()
            blocked.wait(2)
        return snapshot()

    watchdog = GuestControlWatchdog(read, gate, interval=.01, timeout=.10)
    watchdog.start()
    try:
        assert watchdog.wait_ready(1)
        assert entered.wait(1)
        assert watchdog.stopped.wait(1)
        assert channel.closed.wait(1)
        assert 'stale' in watchdog.failure
    finally:
        blocked.set()
        watchdog.close()


@pytest.mark.parametrize('bad', [None, {}, snapshot(active=False), snapshot(epoch=True)])
def test_invalid_or_inactive_initial_state_never_arms_motion(bad):
    channel = Channel()
    gate = ResidentMotionGate(channel)
    watchdog = GuestControlWatchdog(lambda: bad, gate, interval=.01, timeout=.1)
    watchdog.start()
    try:
        assert watchdog.stopped.wait(1)
        assert channel.closed.wait(1)
        assert not watchdog.wait_ready(.05)
        with pytest.raises(RuntimeError):
            gate.send(1, 1, 1, 0.)
    finally:
        watchdog.close()


@pytest.mark.parametrize('field', ['camera_connected', 'camera_fresh', 'control_fresh'])
def test_camera_loss_or_stale_frame_stops_control(field):
    channel = Channel()
    gate = ResidentMotionGate(channel)
    state = snapshot()
    watchdog = GuestControlWatchdog(lambda: dict(state), gate, interval=.01, timeout=.12)
    watchdog.start()
    try:
        assert watchdog.wait_ready(1)
        state[field] = False
        assert watchdog.stopped.wait(1)
        assert channel.closed.wait(1)
        assert not gate.can_send
    finally:
        watchdog.close()


def test_explicit_intro_watchdog_tracks_real_control_even_when_images_drop():
    channel=Channel();gate=ResidentMotionGate(channel)
    state={**snapshot(),'control_fresh':True,'camera_fresh':False,'camera_connected':False}
    watchdog=GuestControlWatchdog(lambda:dict(state),gate,interval=.01,timeout=.12,require_camera=False)
    watchdog.start()
    try:
        assert watchdog.wait_ready(1) and gate.can_send
        state['control_fresh']=False
        assert watchdog.stopped.wait(1) and channel.closed.wait(1)
        assert not gate.can_send
    finally:watchdog.close()


def test_intro_watchdog_requires_explicit_control_fresh_field():
    channel=Channel();gate=ResidentMotionGate(channel)
    state=snapshot();state.pop('control_fresh')
    watchdog=GuestControlWatchdog(lambda:dict(state),gate,interval=.01,timeout=.1,require_camera=False)
    watchdog.start()
    try:
        assert watchdog.stopped.wait(1)
        assert not gate.can_send
    finally:watchdog.close()


@pytest.mark.parametrize('control_fresh', [False, None])
def test_guidance_watchdog_rejects_stale_or_missing_owner_control(control_fresh):
    channel=Channel();gate=ResidentMotionGate(channel)
    state={**snapshot(),'control_fresh':control_fresh}
    watchdog=GuestControlWatchdog(lambda:dict(state),gate,interval=.01,timeout=.1)
    watchdog.start()
    try:
        assert watchdog.stopped.wait(.2)
        assert channel.closed.wait(1)
        assert not gate.can_send
    finally:watchdog.close()


@pytest.mark.parametrize('change',[{'active':False},{'epoch':2}])
def test_intro_watchdog_still_stops_on_stop_or_new_epoch(change):
    channel=Channel();gate=ResidentMotionGate(channel)
    state={**snapshot(),'control_fresh':True}
    watchdog=GuestControlWatchdog(lambda:dict(state),gate,interval=.01,timeout=.1,require_camera=False)
    watchdog.start()
    try:
        assert watchdog.wait_ready(1)
        state.update(change)
        assert watchdog.stopped.wait(1) and not gate.can_send
    finally:watchdog.close()


def test_failed_channel_send_latches_and_closes():
    class Broken(Channel):
        def send(self, *args):
            raise RuntimeError('lost')

    channel = Broken()
    gate = ResidentMotionGate(channel)
    gate.arm()
    with pytest.raises(RuntimeError, match='lost'):
        gate.send(1, 1, 1, 0.)
    assert channel.closed.wait(1)
    with pytest.raises(RuntimeError):
        gate.send(2, 1, 1, 0.)


def test_blocked_channel_close_still_permanently_inhibits_new_sends():
    release = threading.Event()
    class SlowClose(Channel):
        def close(self):
            self.closed.set()
            release.wait(2)
    channel = SlowClose()
    gate = ResidentMotionGate(channel)
    gate.arm()
    gate.stop('stop')
    try:
        assert channel.closed.wait(1)
        assert not gate.close_finished.is_set()
        with pytest.raises(RuntimeError):
            gate.send(1, 1, 1, 0.)
    finally:
        release.set()
        assert gate.close_finished.wait(1)


def test_local_operator_reader_sends_token_header_and_extracts_only_control():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get('X-Guest-Token')))
            body = json.dumps({'active': True, 'control': {'instance': 'service', 'epoch': 4},
                               'control_action': 'start', 'control_started_ns': 42,
                               'camera_connected': True, 'camera_fresh': True, 'control_fresh': True,
                               'other': 'ignored'}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        read = LocalGuestControlReader(server.server_port, 'temporary-test-token', timeout=.2)
        assert read() == {'active': True, 'instance': 'service', 'epoch': 4,
                          'camera_connected': True, 'camera_fresh': True, 'control_action': 'start',
                          'started_ns':42,'control_fresh':True}
        assert seen == [('/operator-state', 'temporary-test-token')]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_local_operator_reader_sanitizes_error_and_never_follows_redirect():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header('Location', 'http://example.invalid/?secret=sentinel-secret')
            self.send_header('Content-Length', '0')
            self.end_headers()
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        read = LocalGuestControlReader(server.server_port, 'sentinel-secret', timeout=.2)
        with pytest.raises(RuntimeError) as error:
            read()
        assert 'sentinel-secret' not in str(error.value)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
