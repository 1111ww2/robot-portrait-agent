"""仅在隔离的回环地址测试 TLS 和 HTTP 客户端，不使用实际 iPad 或运动服务。"""
import http.client
from pathlib import Path
import socket
import ssl
import tempfile
import threading

import pytest

from leica_tracker.guest_display import GuestDisplay
from leica_tracker.guest_https import GuestHTTPS, create_certificates


@pytest.fixture
def certificates():
    root = Path(__file__).resolve().parents[2] / 'tmp' / 'test-https-handshake'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield create_certificates(directory, '127.0.0.1')


def get_state(connection, token):
    connection.request('GET', '/state', headers={'X-Guest-Token': token})
    response = connection.getresponse()
    response.read()
    return response.status


def test_stalled_tls_handshake_preserves_established_and_new_https_clients(certificates, monkeypatch):
    """建立 TCP 后不发送 ClientHello 的客户端不能占住接受新连接的线程。"""
    accepted = threading.Event()
    original = ssl.SSLContext.wrap_socket

    def observed(context, sock, *args, **kwargs):
        if kwargs.get('server_side'):
            # 各平台都可通过已连接的对端判断套接字是否已接受连接。
            # macOS 不支持通过 getsockopt 读取 SO_ACCEPTCONN。
            try:
                sock.getpeername()
            except OSError:
                pass
            else:
                accepted.set()
        return original(context, sock, *args, **kwargs)

    monkeypatch.setattr(ssl.SSLContext, 'wrap_socket', observed)
    context = ssl.create_default_context(cafile=str(certificates['ca']))
    with GuestDisplay(port=0) as guest, GuestHTTPS(certificates, upstream_port=guest.port,
            host='127.0.0.1', port=0, setup_port=0, addresses=[]) as proxy:
        established = http.client.HTTPSConnection('127.0.0.1', proxy.port, context=context, timeout=.5)
        new = None
        stalled = None
        try:
            assert get_state(established, guest.token) == 200
            accepted.clear()
            stalled = socket.create_connection(('127.0.0.1', proxy.port), timeout=1)
            assert accepted.wait(1), 'the unhandshaken TCP client must actually be accepted'
            assert get_state(established, guest.token) == 200
            new = http.client.HTTPSConnection('127.0.0.1', proxy.port, context=context, timeout=.5)
            assert get_state(new, guest.token) == 200
        finally:
            if stalled is not None: stalled.close()
            if new is not None: new.close()
            established.close()


def test_unfinished_tls_handshake_times_out_and_closes_accepted_socket(certificates):
    with GuestDisplay(port=0) as guest, GuestHTTPS(certificates, upstream_port=guest.port,
            host='127.0.0.1', port=0, setup_port=0, addresses=[]) as proxy:
        proxy.server.handshake_timeout = .1
        with socket.create_connection(('127.0.0.1', proxy.port), timeout=1) as stalled:
            stalled.settimeout(1)
            assert stalled.recv(1) == b''


def test_plaintext_on_tls_port_is_closed_without_harming_next_tls_client(certificates):
    context = ssl.create_default_context(cafile=str(certificates['ca']))
    with GuestDisplay(port=0) as guest, GuestHTTPS(certificates, upstream_port=guest.port,
            host='127.0.0.1', port=0, setup_port=0, addresses=[]) as proxy:
        with socket.create_connection(('127.0.0.1', proxy.port), timeout=1) as invalid:
            invalid.sendall(b'GET /state HTTP/1.1\r\nHost: localhost\r\n\r\n')
            try: assert invalid.recv(1) == b''
            except ConnectionResetError: pass
        valid = http.client.HTTPSConnection('127.0.0.1', proxy.port, context=context, timeout=1)
        try: assert get_state(valid, guest.token) == 200
        finally: valid.close()


@pytest.mark.parametrize('failure', ['wrap', 'handshake', None])
def test_worker_closes_the_socket_that_owns_the_descriptor_at_each_stage(failure):
    from leica_tracker.guest_https import _ThreadingTLSServer
    calls = []

    class Socket:
        def __init__(self, name): self.name = name; self.timeout = None
        def settimeout(self, value): self.timeout = value; calls.append((self.name, 'timeout', value))
        def do_handshake(self):
            calls.append(('secured', 'handshake'))
            if failure == 'handshake': raise TimeoutError('fake incomplete ClientHello')

    raw, secured = Socket('raw'), Socket('secured')

    class Context:
        def wrap_socket(self, request, **kwargs):
            assert request is raw
            assert kwargs == {'server_side': True, 'do_handshake_on_connect': False}
            if failure == 'wrap': raise ssl.SSLError('fake invalid client')
            secured.timeout = request.timeout
            return secured

    server = object.__new__(_ThreadingTLSServer)
    server._tls_context = Context()
    server.shutdown_request = lambda request: calls.append(('close', request.name))
    def finish(request, address):
        assert request is secured and request.timeout == 4.
        calls.append(('http', address))
    server.finish_request = finish
    server.process_request_thread(raw, ('127.0.0.1', 12345))
    assert ('raw', 'timeout', 2.) in calls
    assert [item for item in calls if item[0] == 'close'] == [('close', 'raw' if failure == 'wrap' else 'secured')]
    assert any(item[0] == 'http' for item in calls) is (failure is None)
