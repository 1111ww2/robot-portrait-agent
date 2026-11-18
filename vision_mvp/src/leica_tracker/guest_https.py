"""为已配对的访客服务提供本地 TLS 代理，不提供机器人或模型接口。"""
from datetime import datetime, timedelta, timezone
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import http.client
import ipaddress
import json
import math
import os
import plistlib
import re
import ssl
import threading
import time
import uuid


class HTTPSRequestTimings:
    """按需启用数值诊断，请求线程不读写文件。elapsed_ms 是从创建采集任务到完成的时长；
    计时不含 TLS 握手与 HTTP 标头解析。
    body_read_ms 从读取请求体前开始计时，由所有者循环完成记录。"""
    ROUTES = frozenset(('/camera/ticket', '/camera/frame', '/control-heartbeat'))
    NUMBERS = frozenset(('body_read_ms', 'upstream_request_ms', 'upstream_response_ms',
                         'server_total_ms', 'request_body_bytes', 'status', 'error'))
    def __init__(self, output, *, duration=30., clock=time.perf_counter):
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 1 <= duration <= 120:
            raise ValueError('HTTPS timing duration must be from 1 to 120 seconds')
        self.output = Path(output)
        if self.output.exists(): raise FileExistsError('Preserve existing HTTPS timing output')
        self.duration = float(duration); self.clock = clock; self.started = clock()
        self._events = deque(maxlen=4096); self._lock = threading.Lock(); self._finished = False
        self.write_error = 0

    def record(self, event):
        if not isinstance(event, dict) or event.get('route') not in self.ROUTES: return
        now = self.clock()
        if self._finished or not 0 <= now-self.started <= self.duration: return
        if not self._lock.acquire(blocking=False): return
        try:
            if self._finished: return
            row = {'route': event['route']}
            for key in self.NUMBERS:
                value = event.get(key)
                if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 4*1024*1024:
                    row[key] = value
            row['elapsed_ms'] = round((now-self.started)*1000, 3)
            self._events.append(row)
        finally: self._lock.release()

    def finish_if_due(self):
        if not self._finished and self.clock()-self.started < self.duration: return False
        return self.finish()

    def finish(self):
        """尝试一次私有、独占的写入；诊断失败不能中断 TLS 服务。"""
        with self._lock:
            if self._finished: return True
            self._finished = True
            payload = {'format_version': 1, 'duration_s': self.duration,
                       'event_limit': 4096, 'events': list(self._events)}
        try:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, separators=(',', ':'), allow_nan=False); stream.write('\n')
        except OSError: self.write_error = 1
        except Exception: self.write_error = 2
        return True


def drain_rejected_body(headers,stream,connection):
    """关闭 TLS 前在限定时间内读完被拒绝的小请求体。"""
    if headers.get('Transfer-Encoding'):return
    try:
        length=int(headers.get('Content-Length','0'))
        if 0<length<=2048:
            connection.settimeout(.25)
            stream.read(length)
    except (ValueError,OSError):
        pass


def _private_write(path, data):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data)


def load_pairing(live_path, cached_path):
    """相机删除当前证书文件后，仍可重新运行证书配置流程。"""
    live_path=Path(live_path);cached_path=Path(cached_path)
    source=live_path if live_path.exists() else cached_path
    pair=json.loads(source.read_text(encoding='utf-8'))
    if (type(pair.get('port')) is not int or not 1<=pair['port']<=65535 or
        not isinstance(pair.get('key'),str) or not re.fullmatch(r'[A-Za-z0-9_-]{24,128}',pair['key'])):
        raise ValueError('Invalid local guest pairing')
    pair={'port':pair['port'],'key':pair['key']}
    cached_path.parent.mkdir(parents=True,exist_ok=True)
    _private_write(cached_path,json.dumps(pair).encode())
    return pair


def create_certificates(directory,address):
    """复用项目 CA；局域网地址变化时只更新服务端叶证书。"""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    ip=ipaddress.ip_address(address)
    if ip.version!=4 or not ip.is_private:raise ValueError('Private IPv4 required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    files={name:directory/filename for name,filename in {
        'ca':'ca.pem','ca_key':'ca-key.pem','cert':'server.pem','key':'server-key.pem',
        'der':'leica-root.cer','profile':'leica-root.mobileconfig'}.items()}
    now=datetime.now(timezone.utc)
    def key_bytes(key):return key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption())
    if files['ca'].exists()!=files['ca_key'].exists():raise ValueError('Incomplete local CA; preserve and inspect existing files')
    if files['ca'].exists():
        ca=x509.load_pem_x509_certificate(files['ca'].read_bytes())
        ca_key=serialization.load_pem_private_key(files['ca_key'].read_bytes(),password=None)
        if ca.not_valid_after_utc<=now+timedelta(days=30):raise ValueError('Local CA needs renewal and iPad trust setup')
        if ca.public_key().public_numbers()!=ca_key.public_key().public_numbers():raise ValueError('CA key mismatch')
    else:
        ca_key=rsa.generate_private_key(public_exponent=65537,key_size=3072)
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Leica Local iPad Root')])
        ca=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=5)).not_valid_after(now+timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
            .add_extension(x509.KeyUsage(False,False,False,False,False,True,True,False,False),critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),critical=False)
            .sign(ca_key,hashes.SHA256()))
        _private_write(files['ca_key'],key_bytes(ca_key))
        _private_write(files['ca'],ca.public_bytes(serialization.Encoding.PEM))
    metadata=directory/'address.json'
    reuse=False
    if files['cert'].exists() and files['key'].exists() and metadata.exists():
        leaf=x509.load_pem_x509_certificate(files['cert'].read_bytes())
        reuse=json.loads(metadata.read_text()).get('address')==address and leaf.not_valid_after_utc>now+timedelta(days=7)
    if not reuse:
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        names=[x509.DNSName('localhost'),x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]
        if address!='127.0.0.1':names.append(x509.IPAddress(ip))
        leaf=(x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Leica iPad Local Server')]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-timedelta(minutes=5)).not_valid_after(min(now+timedelta(days=90),ca.not_valid_after_utc))
            .add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
            .add_extension(x509.SubjectAlternativeName(names),critical=False)
            .add_extension(x509.KeyUsage(True,False,True,False,False,False,False,False,False),critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),critical=False)
            .sign(ca_key,hashes.SHA256()))
        _private_write(files['key'],key_bytes(key))
        _private_write(files['cert'],leaf.public_bytes(serialization.Encoding.PEM)+ca.public_bytes(serialization.Encoding.PEM))
        metadata.write_text(json.dumps({'address':address}),encoding='utf-8')
    der=ca.public_bytes(serialization.Encoding.DER)
    _private_write(files['der'],der)
    profile={'PayloadContent':[{'PayloadCertificateFileName':'leica-root.cer','PayloadContent':der,
        'PayloadDescription':'Root certificate for this local Leica iPad prototype.',
        'PayloadDisplayName':'Leica Local iPad Root','PayloadIdentifier':'local.leica.ipad.root',
        'PayloadType':'com.apple.security.root','PayloadUUID':str(uuid.uuid4()),'PayloadVersion':1}],
        'PayloadDisplayName':'Leica Local iPad HTTPS','PayloadIdentifier':'local.leica.ipad.https',
        'PayloadType':'Configuration','PayloadUUID':str(uuid.uuid4()),'PayloadVersion':1,
        'PayloadDescription':'Installs one local root certificate. No device management, VPN or account settings.'}
    _private_write(files['profile'],plistlib.dumps(profile))
    return files


class _ThreadingTLSServer(ThreadingHTTPServer):
    """及时接受原始 TCP 连接，把有超时限制的 TLS 握手交给工作线程。
    若直接包装监听套接字，握手会发生在 accept() 内，先于 ThreadingHTTPServer 分发线程；
    一个空闲客户端就可能阻塞所有新连接。"""
    handshake_timeout = 2.

    def __init__(self, address, handler, *, context):
        self._tls_context = context
        super().__init__(address, handler)

    def process_request_thread(self, request, client_address):
        secured = None
        try:
            request.settimeout(self.handshake_timeout)
            secured = self._tls_context.wrap_socket(request, server_side=True,
                                                     do_handshake_on_connect=False)
            # SSLSocket 的超时约束整次握手，不会在每个握手分片到达后重新计时。
            secured.do_handshake()
            # 初始 HTTP 请求行也要限制读取时长，因为 Handler.handle_request 在解析完该行后才设置相同的超时。
            secured.settimeout(4.)
        except (OSError, ValueError):
            # wrap_socket 将描述符的所有权转移给 SSLSocket；此后只关闭原始套接字对象不足以释放连接。
            self.shutdown_request(secured if secured is not None else request)
            return
        except BaseException:
            self.shutdown_request(secured if secured is not None else request)
            raise
        # 由请求工作线程处理常规 HTTP 错误，并在处理程序退出时关闭 TLS 套接字；沿用原有认证和请求体规则。
        super().process_request_thread(secured, client_address)


class GuestHTTPS:
    def __init__(self,files,*,upstream_port=8767,host='0.0.0.0',port=8768,setup_port=8769,addresses,
                 timing_callback=None,timing_clock=time.perf_counter,upstream_clock=time.monotonic):
        self.files=files;self.upstream_port=upstream_port
        self.timing_callback=timing_callback;self.timing_clock=timing_clock
        self.upstream_clock=upstream_clock
        self.addresses=set(addresses)|{'127.0.0.1','localhost'}
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(files['cert'],files['key'])
        self.server=_ThreadingTLSServer((host,port),self._handler(False),context=context)
        try:
            self.setup=ThreadingHTTPServer((host,setup_port),self._handler(True))
        except Exception:self.server.server_close();raise
        self.port=self.server.server_address[1];self.setup_port=self.setup.server_address[1]
        self.threads=[]
        for server in (self.server,self.setup):
            server.daemon_threads=True
            self.threads.append(threading.Thread(target=server.serve_forever,daemon=True))

    def __enter__(self):
        for thread in self.threads:thread.start()
        return self

    def __exit__(self,*_):
        for server in (self.server,self.setup):server.shutdown();server.server_close()
        for thread in self.threads:thread.join(timeout=2)

    def _handler(self,setup):
        proxy=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            disable_nagle_algorithm=True
            def log_message(self,*_):pass
            def _close_upstream(self):
                connection=getattr(self,'_upstream',None)
                self._upstream=None;self._upstream_last_used=None
                if connection is not None:
                    try:connection.close()
                    except OSError:pass
            def _upstream_connection(self,timeout):
                # BaseHTTPRequestHandler 在一个线程内串行处理一个 TLS 连接。
                # 上游连接仅供该线程及其客户端使用。
                now=getattr(proxy,'upstream_clock',time.monotonic)()
                connection=getattr(self,'_upstream',None)
                last=getattr(self,'_upstream_last_used',None)
                if connection is not None and (last is None or not 0<=now-last<1.
                                                or getattr(connection,'sock',None) is None):
                    self._close_upstream();connection=None
                if connection is None:
                    connection=http.client.HTTPConnection('127.0.0.1',proxy.upstream_port,timeout=timeout)
                    self._upstream=connection
                connection.timeout=timeout
                if getattr(connection,'sock',None) is not None:connection.sock.settimeout(timeout)
                return connection
            def finish(self):
                # 包括客户端读到 EOF、解析失败和 TLS 断连。
                try:self._close_upstream()
                finally:super().finish()
            def _timing_start(self):
                return proxy.timing_clock() if getattr(self,'_request_timing',None) is not None else None
            def _timing_elapsed(self,key,started):
                if started is not None:self._request_timing[key]=round(max(0.,proxy.timing_clock()-started)*1000,3)
            def _timing_error(self,code):
                value=getattr(self,'_request_timing',None)
                if value is not None and not value['error']:value['error']=code
            def reply(self,status,data=b'',kind='text/plain; charset=utf-8',extra=None):
                if getattr(self,'_request_timing',None) is not None:self._request_timing['status']=status
                # 被拒绝的 POST 请求体可能尚未读完。不能把残余字节当作持久连接上的下一个请求。
                if status>=400:
                    self.close_connection=True;self._close_upstream()
                self.send_response(status)
                if self.close_connection:self.send_header('Connection','close')
                for key,value in {'Content-Type':kind,'Content-Length':str(len(data)),
                    'Cache-Control':'no-store','Referrer-Policy':'no-referrer','X-Content-Type-Options':'nosniff',**(extra or {})}.items():self.send_header(key,value)
                self.end_headers()
                try:self.wfile.write(data)
                except (OSError,ssl.SSLError):
                    self._timing_error(5);self.close_connection=True;self._close_upstream()
            def do_GET(self):self.handle_request()
            def do_POST(self):self.handle_request()
            def reject_unread(self,status):
                if self.command=='POST':drain_rejected_body(self.headers,self.rfile,self.connection)
                return self.reply(status)
            def handle_request(self):
                self._request_timing=None
                callback=proxy.timing_callback
                if callback is None or setup:return self._handle_request()
                route=urlsplit(self.path)
                if route.scheme or route.netloc or route.path not in HTTPSRequestTimings.ROUTES:
                    return self._handle_request()
                started=proxy.timing_clock()
                self._request_timing={'route':route.path,'body_read_ms':0.,'upstream_request_ms':0.,
                    'upstream_response_ms':0.,'server_total_ms':0.,'request_body_bytes':0,'status':0,'error':0}
                try:return self._handle_request()
                except BaseException:
                    self._timing_error(7);raise
                finally:
                    self._timing_elapsed('server_total_ms',started)
                    try:callback(dict(self._request_timing))
                    except Exception:pass
                    self._request_timing=None
            def _handle_request(self):
                self.connection.settimeout(4)
                port=proxy.setup_port if setup else proxy.port
                expected={f'{address}:{port}' for address in proxy.addresses}
                if not ipaddress.ip_address(self.client_address[0]).is_private or self.headers.get('Host') not in expected:
                    return self.reject_unread(403)
                if setup:
                    allowed={'/leica-root.cer':('der','application/x-x509-ca-cert'),
                             '/leica-root.mobileconfig':('profile','application/x-apple-aspen-config')}
                    if self.command!='GET' or self.path not in allowed:return self.reject_unread(404)
                    name,kind=allowed[self.path]
                    return self.reply(200,proxy.files[name].read_bytes(),kind)
                route=urlsplit(self.path)
                camera_routes=('/camera/start','/camera/ticket','/camera/frame','/camera/stop')
                if route.scheme or route.netloc or route.path not in ('/','/state','/frame.jpg','/cue-audio','/action','/control-heartbeat','/message','/transcribe','/photo')+camera_routes:
                    return self.reject_unread(404)
                body=None
                if self.command=='POST':
                    if route.path not in ('/action','/control-heartbeat','/message','/transcribe','/photo')+camera_routes:return self.reject_unread(404)
                    if self.headers.get('Origin')!='https://'+self.headers.get('Host',''):return self.reject_unread(403)
                    camera_frame=route.path=='/camera/frame'
                    kind='image/jpeg' if camera_frame or route.path=='/photo' else 'application/octet-stream' if route.path=='/transcribe' else 'application/json'
                    if self.headers.get_content_type()!=kind or self.headers.get('Transfer-Encoding'):return self.reject_unread(400)
                    try:
                        length=int(self.headers.get('Content-Length','0'))
                        if not 0<length<=(524288 if camera_frame else 4*1024*1024 if route.path=='/photo' else 2*1024*1024 if route.path=='/transcribe' else 2048):return self.reject_unread(413)
                        started=self._timing_start()
                        try:body=self.rfile.read(length)
                        finally:self._timing_elapsed('body_read_ms',started)
                        if self._request_timing is not None:self._request_timing['request_body_bytes']=len(body)
                        if len(body)!=length:
                            self._timing_error(2);return self.reply(400)
                    except (ValueError,OSError) as error:
                        self._timing_error(1 if isinstance(error,TimeoutError) else 6)
                        return self.reply(400)
                elif route.path not in ('/','/state','/frame.jpg','/cue-audio'):return self.reply(404)
                headers={k:self.headers[k] for k in ('X-Guest-Token','X-Guest-Client','X-Camera-Session','X-Camera-Ticket','X-Speech-Language') if self.headers.get(k)}
                if body is not None:headers.update({'Content-Type':kind,'Origin':f'http://127.0.0.1:{proxy.upstream_port}'})
                try:
                    connection=self._upstream_connection(45 if route.path=='/transcribe' else 3)
                    started=self._timing_start()
                    try:connection.request(self.command,self.path,body,headers)
                    finally:self._timing_elapsed('upstream_request_ms',started)
                    started=self._timing_start()
                    try:
                        response=connection.getresponse();data=response.read(2*1024*1024+1)
                        if len(data)<=2*1024*1024:
                            # 当对端关闭时，read(amt) 即使没有读满 Content-Length 也可能正常返回。
                            # 对于长度未知的响应，还须探测 EOF。
                            if not response.isclosed() and response.length is None:
                                if response.read(1):raise http.client.HTTPException('Incomplete upstream response')
                            if not response.isclosed() or response.length not in (None,0):
                                raise http.client.HTTPException('Incomplete upstream response')
                    finally:self._timing_elapsed('upstream_response_ms',started)
                    if len(data)>2*1024*1024:
                        self._close_upstream();return self.reply(502)
                    # 仅复用已完整读取且大小受限的响应连接。读写一旦失败，即使是 GET 也不重试。
                    if getattr(response,'will_close',True):self._close_upstream()
                    else:self._upstream_last_used=getattr(proxy,'upstream_clock',time.monotonic)()
                    extra={k:response.getheader(k) for k in ('X-Frame-Sequence','X-Frame-Feedback') if response.getheader(k)}
                    self.reply(response.status,data,response.getheader('Content-Type','application/octet-stream'),extra)
                except (OSError,http.client.HTTPException) as error:
                    self._close_upstream()
                    self._timing_error(3 if isinstance(error,TimeoutError) else 4)
                    self.reply(503,b'Camera service unavailable; check the laptop.')
        return Handler
