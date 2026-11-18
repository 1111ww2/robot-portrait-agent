"""在局域网演示中提供短时、只读的照片分享，照片仅驻留内存。"""
import base64
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import ipaddress
from pathlib import Path
import re
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit

MAX_PHOTO_BYTES=4*1024*1024


def private_lan_address():
    """查询当前局域网出站路由，不发送数据包。"""
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as route:
        route.connect(('192.0.2.1',9))
        address=route.getsockname()[0]
    ip=ipaddress.ip_address(address)
    if ip.version!=4 or not ip.is_private or ip.is_unspecified:
        raise ValueError('A private local IPv4 address is required')
    return address


class PhotoShare:
    def __init__(self, *, host='0.0.0.0',address=None,ttl_seconds=300):
        if not 0 < ttl_seconds <= 300:raise ValueError('Invalid photo lifetime')
        self.address=address or private_lan_address()
        ip=ipaddress.ip_address(self.address)
        if ip.version!=4 or not ip.is_private or ip.is_unspecified:raise ValueError('Invalid LAN address')
        self.ttl=ttl_seconds;self.lock=threading.Lock();self.item=None;self.timer=None
        self.server=ThreadingHTTPServer((host,0),self._handler());self.server.daemon_threads=True
        self.port=self.server.server_address[1]
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*_):self.close()
    def close(self):
        self.clear();self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
    @property
    def empty(self):
        with self.lock:return self.item is None
    def clear(self):
        with self.lock:
            self.item=None
            if self.timer:self.timer.cancel();self.timer=None
    def discard(self,token):
        with self.lock:
            if self.item and secrets.compare_digest(self.item[0],token):
                self.item=None
                if self.timer:self.timer.cancel();self.timer=None

    def publish(self,jpeg):
        if not isinstance(jpeg,bytes) or not 4<=len(jpeg)<=MAX_PHOTO_BYTES or not jpeg.startswith(b'\xff\xd8\xff'):
            raise ValueError('Expected a bounded JPEG image')
        token=secrets.token_urlsafe(32)
        url=f'http://{self.address}:{self.port}/p/{token}'
        import qrcode
        output=io.BytesIO();qrcode.make(url).save(output,format='PNG')
        with self.lock:
            if self.timer:self.timer.cancel()
            self.item=(token,jpeg,time.monotonic()+self.ttl)
            timer=threading.Timer(self.ttl,self._expire,args=(token,));timer.daemon=True;self.timer=timer;timer.start()
        return {'qr_data_url':'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode('ascii'),
                'expires_seconds':self.ttl,'_url':url}
    def _expire(self,token):
        with self.lock:
            if self.item and secrets.compare_digest(self.item[0],token):self.item=None;self.timer=None
    def _get(self,token):
        with self.lock:
            if not self.item or time.monotonic()>=self.item[2]:
                self.item=None;return None
            return self.item[1] if secrets.compare_digest(self.item[0],token) else None
    def _handler(self):
        share=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_GET(self):
                try:local=ipaddress.ip_address(self.client_address[0]).is_private
                except ValueError:local=False
                match=re.fullmatch(r'/(p|image|logo)/([A-Za-z0-9_-]{43})',urlsplit(self.path).path)
                data=share._get(match.group(2)) if match and local else None
                if data is None:
                    self.send_response(404);self.send_header('Content-Length','0');self.send_header('Cache-Control','no-store');self.end_headers();return
                if match.group(1)=='p':
                    token=match.group(2)
                    data=(f'<!doctype html><html lang="en"><meta name="viewport" content="width=device-width,initial-scale=1">'
                          f'<meta name="referrer" content="no-referrer"><title>Your portrait</title>'
                          f'<style>body{{margin:0;background:#111;color:white;font:16px system-ui;text-align:center}}'
                          f'.brand{{display:block;width:48px;height:48px;margin:16px auto 6px}}'
                          f'h1{{font-size:24px;margin:8px 0}}p{{margin:10px auto}}'
                          f'.portrait{{display:block;max-width:100%;max-height:60vh;margin:auto}}'
                          f'.concept{{max-width:30em;color:#ccc;line-height:1.45;padding:0 18px}}'
                          f'</style><img class="brand" src="/logo/{token}" alt="Leica">'
                          f'<h1>Your portrait</h1><img class="portrait" src="/image/{token}" alt="Your portrait">'
                          f'<p>Touch and hold the photo to save it.</p>'
                          f'<p class="concept">The camera chooses where you stand. Who is framing whom?</p></html>').encode()
                    kind='text/html; charset=utf-8'
                elif match.group(1)=='logo':
                    try:data=Path(__file__).with_name('leica-logo.svg').read_bytes()
                    except OSError:
                        self.send_response(404);self.send_header('Content-Length','0');self.end_headers();return
                    kind='image/svg+xml'
                else:kind='image/jpeg'
                self.send_response(200)
                for k,v in {'Content-Type':kind,'Content-Length':str(len(data)),'Cache-Control':'no-store',
                            'Referrer-Policy':'no-referrer','X-Content-Type-Options':'nosniff',
                            'Content-Security-Policy':"default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'"}.items():self.send_header(k,v)
                self.end_headers()
                try:self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
        return Handler
