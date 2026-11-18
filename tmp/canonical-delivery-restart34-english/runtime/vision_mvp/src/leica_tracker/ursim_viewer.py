"""仅获取从围栏内的官方URSim中测量关节的实时演示。
这里不存在输入写入器、机器人程序、运动控制器或模拟器启动程序。标称DH几何学是演示，而不是校准或第二个运动流。
呼叫者必须已经拥有固定的模拟器回环网关。"""
import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from pathlib import Path
import threading
import time

from .physical_system import ROOT, signal_stop, workspace_path
from .guest_https import drain_rejected_body
from .ursim_system import verify_docker
from .ursim_transport import SIM_HOST, SimRTDEChannel, SimStateReceiver

HOST = '127.0.0.1'
PORT = 8779
DH_SOURCE = 'https://www.universal-robots.com/developer/hardware-and-motion/robot-motion-dh-parameters/'
# 标准DH：RotZ(q) TransZ(d) TransX(a) RotX(alpha)，米/度。
# 制造商标称表格，已验证2026年10月31日。实际校准不同。
DH = {
    'ur10e': ((0., -.6127, -.57155, 0., 0., 0.), (.1807, 0., 0., .17415, .11985, .11655)),
    'ur5e': ((0., -.425, -.3922, 0., 0., 0.), (.1625, 0., 0., .1333, .0997, .0996)),
}
ALPHA = (math.pi/2, 0., 0., math.pi/2, -math.pi/2, 0.)


def _vector(value):
    return (isinstance(value, (list, tuple)) and len(value) == 6
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def forward_kinematics(joints, model='ur10e'):
    """从测量q中返回基座和六个关节框架，没有任何运动。"""
    if model not in DH or not _vector(joints): raise ValueError('Known model and six finite measured joints required')
    a, d = DH[model]
    transform = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    frames = [copy.deepcopy(transform)]
    for q, link, offset, alpha in zip(joints, a, d, ALPHA):
        c,s,ca,sa = math.cos(q),math.sin(q),math.cos(alpha),math.sin(alpha)
        step = [[c,-s*ca,s*sa,link*c],[s,c*ca,-c*sa,link*s],[0.,sa,ca,offset],[0.,0.,0.,1.]]
        transform = [[sum(transform[i][k]*step[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        frames.append(transform)
    return {'frames': frames, 'origins': [[frame[i][3] for i in range(3)] for frame in frames]}


class MeasuredState:
    """保持最后测量到的姿势，切勿重新标记重复/无效的反馈。"""
    def __init__(self, receiver, primary, *, model='ur10e', clock=time.monotonic):
        if model not in DH: raise ValueError('Unknown nominal viewer model')
        self.receiver=receiver;self.primary=primary;self.model=model;self.clock=clock
        self._lock=threading.Lock();self._last=None;self._stamp=None;self._producer=None
        self._source_receiver=receiver;self._await_sample=True

    def attach(self,receiver,primary):
        with self._lock:
            # ReadOnlyTelemetry在附加新构建的仅接收通道之前验证相同的Docker身份。
            # 控制器的重新启动可能会重置该通道的生产者时钟。在合格的新电测取代之前，请保留旧的显示姿势/日期；
            # 重新附加相同的通道不得重置时钟验证。
            if receiver is not None and receiver is not self._source_receiver:
                self._source_receiver=receiver;self._await_sample=True
            self.receiver=receiver;self.primary=primary

    def snapshot(self):
        with self._lock:
            receiver,primary=self.receiver,self.primary
            feedback,stamp=receiver.latest if receiver is not None else (None,None)
            state,state_at=primary.latest if primary is not None else (None,None)
            now=self.clock()
            primary_age=now-state_at if type(state_at) in (int,float) and math.isfinite(state_at) else None
            mode=state.get('mode') if isinstance(state,dict) else None
            mode=mode if isinstance(mode,dict) else {}
            mode_fresh=(primary_age is not None and 0<=primary_age<=.5 and primary is not None
                        and isinstance(state,dict) and bool(mode) and not primary.failure)
            source_fresh=mode_fresh and receiver is not None and not receiver.failure
            invalid = False
            if isinstance(feedback, dict) and _vector(feedback.get('joints')) and _vector(feedback.get('tcp_pose')):
                producer=feedback.get('timestamp')
                if (source_fresh and type(stamp) in (int,float) and math.isfinite(stamp) and 0<=now-stamp<=.5
                        and type(producer) in (int,float) and math.isfinite(producer) and producer>=0):
                    if self._stamp is None or stamp>self._stamp and (self._await_sample or producer>self._producer):
                        q=list(feedback['joints']);tcp=list(feedback['tcp_pose'])
                        self._last={'q':q,'tcp_pose':tcp,'robot_timestamp':producer,
                                    **forward_kinematics(q,self.model)}
                        self._stamp=stamp;self._producer=producer
                        self._await_sample=False
                    elif stamp<self._stamp or not self._await_sample and producer<self._producer:invalid=True
                else:invalid=True
            elif feedback is not None:invalid=True
            age=now-self._stamp if self._stamp is not None else None
            fresh=(self._last is not None and age is not None and 0<=age<=.5 and mode_fresh
                   and receiver is not None and not receiver.failure and not self._await_sample and not invalid)
            result=copy.deepcopy(self._last) if self._last is not None else {'q':None,'tcp_pose':None,'origins':None,'frames':None,'robot_timestamp':None}
            result.update(model=self.model,source='Official URSim 5.13 / receive only',
                nominal_geometry=True,kinematic_source=DH_SOURCE,fresh=bool(fresh),
                sampled_at=self._stamp,sample_age_ms=round(max(0.,age)*1000,1) if age is not None else None,
                robot_mode=mode.get('mode') if mode_fresh and type(mode.get('mode')) is int else None,
                program_running=mode.get('running') if mode_fresh and type(mode.get('running')) is bool else None,
                status='Live measured pose' if fresh else 'Waiting for telemetry' if self._last is None else 'Stale — retaining last measured pose')
            return result


class ReadOnlyTelemetry:
    """重新连接仅在现有网关返回后才会收到仅接收的读卡器。"""
    def __init__(self,state,identity,*,docker=None):
        self.state=state;self.identity=identity;self.docker=docker
        self.receiver=self.primary=None;self.cleanup_failed=False
    def healthy(self):
        return bool(self.receiver is not None and self.primary is not None
                    and not self.receiver.failure and not self.primary.failure)
    def close(self):
        self.state.attach(None,None)
        items=(self.receiver,self.primary);self.receiver=self.primary=None;errors=[]
        for item in items:
            if item is not None:
                try:item.close()
                except Exception as exc:errors.append(exc)
        if errors:
            self.cleanup_failed=True
            raise RuntimeError('Viewer receive-only cleanup incomplete') from errors[0]
    def connect(self):
        self.close()
        if self.cleanup_failed:raise RuntimeError('Viewer receive-only cleanup incomplete')
        if verify_docker(self.docker)!=self.identity:raise RuntimeError('URSim viewer Docker identity changed')
        try:
            self.primary=SimStateReceiver(SIM_HOST)
            self.receiver=SimRTDEChannel(SIM_HOST,spatial=True,receive_only=True)
            self.state.attach(self.receiver,self.primary)
        except Exception:
            self.close();raise


def local_assets():
    base=Path(__file__).with_name('viewer_assets').resolve()
    names={'/assets/three.module.js':base/'three.module.js',
           '/assets/three.core.js':base/'three.core.js',
           '/assets/OrbitControls.js':base/'OrbitControls.js',
           '/assets/THREE-LICENSE.txt':base/'LICENSE'}
    return {name:path.resolve() for name,path in names.items() if path.is_file() and path.resolve().parent==base}


class ViewerServer:
    """固定回环呈现；每个非GET方法都被拒绝。"""
    def __init__(self, measured, *, root=ROOT, port=PORT):
        if port not in (0,PORT): raise ValueError('Viewer is fixed to loopback port 8779')
        self.measured=measured;self.assets=local_assets()
        self.page=Path(__file__).with_name('ursim_viewer.html').read_bytes()
        viewer=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1';disable_nagle_algorithm=True
            def log_message(self,*args):pass
            def reply(self,status,data=b'',kind='application/json'):
                self.send_response(status)
                for k,v in {'Content-Type':kind,'Content-Length':str(len(data)),'Cache-Control':'no-store',
                    'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'}.items():self.send_header(k,v)
                self.end_headers()
                try:self.wfile.write(data)
                except OSError:pass
            def do_GET(self):
                if not ipaddress.ip_address(self.client_address[0]).is_loopback:return self.reply(403)
                if self.headers.get('Host') not in (f'{HOST}:{viewer.port}',f'localhost:{viewer.port}'):return self.reply(403)
                if self.path=='/':return self.reply(200,viewer.page,'text/html; charset=utf-8')
                if self.path=='/state':return self.reply(200,json.dumps(viewer.measured.snapshot(),allow_nan=False).encode())
                path=viewer.assets.get(self.path)
                if path is not None:return self.reply(200,path.read_bytes(),
                    'text/plain; charset=utf-8' if self.path.endswith('.txt') else 'text/javascript; charset=utf-8')
                return self.reply(404)
            def refuse(self):
                drain_rejected_body(self.headers,self.rfile,self.connection)
                self.close_connection=True;self.send_response(405);self.send_header('Allow','GET')
                self.send_header('Content-Length','0');self.send_header('Connection','close');self.end_headers()
            do_POST=refuse;do_PUT=refuse;do_DELETE=refuse;do_PATCH=refuse;do_OPTIONS=refuse;do_HEAD=refuse
        self.server=ThreadingHTTPServer((HOST,port),Handler);self.server.daemon_threads=True
        self.port=self.server.server_port
        self.thread=threading.Thread(target=self.server.serve_forever,name='ursim-readonly-viewer',daemon=True)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2.)


def main(argv=None, *, root=ROOT):
    root=Path(root).resolve();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Open receive-only telemetry and the loopback viewer')
    parser.add_argument('--docker');parser.add_argument('--stop-file',type=Path)
    args=parser.parse_args(argv)
    stop_file=workspace_path(root,args.stop_file) if args.stop_file else None
    if stop_file is not None and stop_file.exists():parser.error('Stop file already exists')
    if not args.execute:
        print('Read-only URSim viewer validated offline; no telemetry or server opened.');return 0
    identity=verify_docker(args.docker)
    state=MeasuredState(None,None,model=identity.robot_model)
    owner=ReadOnlyTelemetry(state,identity,docker=args.docker)
    try:
        with ViewerServer(state,root=root) as server, signal_stop() as stopping:
            print(f'URSIM_VIEWER_READY http://{HOST}:{server.port} — receive only, no robot controls.',flush=True)
            next_attempt=0.;reported_wait=False
            while not stopping() and not (stop_file and stop_file.exists()):
                if not owner.healthy() and time.monotonic()>=next_attempt:
                    try:owner.connect();reported_wait=False
                    except Exception:
                        if owner.cleanup_failed:raise
                        if not reported_wait:
                            print('URSIM_VIEWER_WAITING: receive-only source unavailable; retained pose is stale.',flush=True)
                            reported_wait=True
                    next_attempt=time.monotonic()+2.
                time.sleep(.2)
    finally:owner.close()
    return 0
