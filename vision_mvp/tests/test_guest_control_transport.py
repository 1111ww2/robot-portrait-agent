"""通过真实 HTTP 传输状态快照，不创建机器人接口。"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from leica_tracker.resident_guest_watchdog import LocalGuestControlReader,GuestControlWatchdog,ResidentMotionGate


@pytest.fixture
def endpoint():
    state=SimpleNamespace(connections=0,calls=[],finished=[],behavior='ok',active=True)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1';disable_nagle_algorithm=True
        def setup(self):
            super().setup();state.connections+=1;self.identity=state.connections
        def finish(self):
            try:super().finish()
            finally:state.finished.append(self.identity)
        def log_message(self,*args):pass
        def do_GET(self):
            state.calls.append((self.identity,self.path,self.headers.get('X-Guest-Token')))
            body=json.dumps(dict(active=state.active,control=dict(instance='fixture-instance',epoch=1),
                control_action='start',control_started_ns=1,camera_connected=True,camera_fresh=True,
                control_fresh=True)).encode()
            if state.behavior=='oversize':body=body[:-1]+b',"padding":"'+b'x'*262144+b'"}'
            if state.behavior in ('headers_delay','blocked','split_delay'):
                time.sleep(.15 if state.behavior=='headers_delay' else .5 if state.behavior=='blocked' else .12)
            self.send_response(302 if state.behavior=='redirect' else 200)
            if state.behavior=='redirect':self.send_header('Location','http://invalid.example/?sentinel-secret')
            if state.behavior!='unknown_length':
                self.send_header('Content-Length',str(len(body)+(5 if state.behavior=='truncate' else 0)))
            if state.behavior in ('truncate','unknown_length'):
                self.send_header('Connection','close');self.close_connection=True
            self.end_headers()
            try:
                if state.behavior=='split_delay':time.sleep(.12)
                self.wfile.write(body);self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    worker=threading.Thread(target=server.serve_forever,kwargs=dict(poll_interval=.01),daemon=True);worker.start()
    state.port=server.server_port
    try:yield state
    finally:server.shutdown();server.server_close();worker.join(1)


def reader(endpoint,timeout=.25):return LocalGuestControlReader(endpoint.port,'sentinel-secret',timeout=timeout)


def test_persistent_reader_fetches_new_snapshots_without_replaying_or_caching(endpoint):
    read=reader(endpoint)
    try:
        assert read()['active'] is True
        endpoint.active=False
        assert read()['active'] is False
        assert len(endpoint.calls)==2 and endpoint.calls[0][0]==endpoint.calls[1][0]
        assert all(path=='/operator-state' and token=='sentinel-secret' for _,path,token in endpoint.calls)
    finally:read.close()


def test_legitimate_150ms_response_within_250ms_budget_is_accepted(endpoint):
    endpoint.behavior='headers_delay';read=reader(endpoint)
    try:assert read()['active'] is True
    finally:read.close()


def test_headers_and_body_share_one_total_request_deadline(endpoint):
    endpoint.behavior='split_delay';read=reader(endpoint,timeout=.20)
    try:
        with pytest.raises(RuntimeError) as caught:read()
        assert caught.value.category=='timeout' and caught.value.phase=='body'
        assert len(endpoint.calls)==1
        assert 'sentinel-secret' not in str(caught.value)
    finally:read.close()


@pytest.mark.parametrize('behavior,category',[('truncate','response'),('oversize','response'),('redirect','http_rejected')])
def test_invalid_response_is_classified_and_connection_is_not_reused(endpoint,behavior,category):
    endpoint.behavior=behavior;read=reader(endpoint)
    try:
        with pytest.raises(RuntimeError) as caught:read()
        assert caught.value.category==category
        assert caught.value.phase in ('headers','body')
        assert 'sentinel-secret' not in str(caught.value)
        endpoint.behavior='ok';assert read()['active']
        assert endpoint.calls[0][0]!=endpoint.calls[1][0]
        assert len(endpoint.calls)==2
    finally:read.close()


def test_unknown_length_complete_response_is_accepted_but_not_cached(endpoint):
    endpoint.behavior='unknown_length';read=reader(endpoint)
    try:
        assert read()['active'];assert read()['active']
        assert endpoint.calls[0][0]!=endpoint.calls[1][0]
    finally:read.close()


def test_each_thread_owns_and_closes_its_own_connection(endpoint):
    read=reader(endpoint);results=[];barrier=threading.Barrier(2)
    def work():
        try:
            barrier.wait(1);results.append(read()['active']);results.append(read()['active'])
        finally:read.close_current_thread()
    threads=[threading.Thread(target=work) for _ in range(2)]
    try:
        for thread in threads:thread.start()
        for thread in threads:thread.join(2)
        assert results==[True]*4 and len(set(row[0] for row in endpoint.calls))==2
        deadline=time.monotonic()+1
        while len(endpoint.finished)<2 and time.monotonic()<deadline:time.sleep(.005)
        assert len(endpoint.finished)==2
    finally:read.close()


def test_independent_300ms_watchdog_still_stops_while_http_reader_is_blocked(endpoint):
    class Channel:
        def __init__(self):self.closed=threading.Event()
        def close(self):self.closed.set()
        def send(self,*args):pass
    read=reader(endpoint,timeout=.5);channel=Channel();gate=ResidentMotionGate(channel)
    watchdog=GuestControlWatchdog(read,gate,interval=.02,timeout=.3,require_camera=False)
    watchdog.start()
    try:
        assert watchdog.wait_ready(1) and gate.can_send
        endpoint.behavior='blocked'
        assert watchdog.stopped.wait(1) and channel.closed.wait(1)
        assert not gate.can_send and 'stale' in watchdog.failure
    finally:watchdog.close();read.close()
