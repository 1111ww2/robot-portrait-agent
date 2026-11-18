"""仅提供本地仿真的动作反馈，不连接实体机器人。"""
import hmac
import json
import math
import os
import socket
import threading
import time

# 这些目标值用于仿真动作编排，不代表实体机器人限值。
SIM_HORIZONTAL_M=.12
SIM_VERTICAL_M=.085
SIM_DEPTH_M=.06
MAX_AGE_NS=300_000_000
ENV_NAME='LEICA_SIM_FEEDBACK'


def photo_ready_from_motion(data, *, token, now_ns, source_ns):
    """仅接纳来自仿真器、与当前动作匹配且仍有效的到位就绪样本。"""
    return bool(isinstance(data,dict) and isinstance(data.get('token'),str)
                and hmac.compare_digest(data['token'],token)
                and type(source_ns) is int and source_ns>0
                and data.get('source_ns')==source_ns and data.get('cue')=='ARRIVED'
                and data.get('hold') is False and data.get('photo_ready') is True
                and type(data.get('time_ns')) is int and type(now_ns) is int
                and 0<=now_ns-data['time_ns']<=MAX_AGE_NS)


def intro_active_from_motion(data, *, token, now_ns, source_ns):
    return bool(isinstance(data,dict) and isinstance(data.get('token'),str)
                and hmac.compare_digest(data['token'],token)
                and type(source_ns) is int and source_ns>0
                and data.get('source_ns')==source_ns and data.get('profile')=='sim-intro'
                and data.get('hold') is False and type(data.get('time_ns')) is int
                and type(now_ns) is int and 0<=now_ns-data['time_ns']<=MAX_AGE_NS)


def intro_ready_from_motion(data, *, token, now_ns, source_ns):
    return intro_active_from_motion(data,token=token,now_ns=now_ns,source_ns=source_ns) and data.get('intro_ready') is True


class SimulatorIntroGate:
    """根据实测数据确认仿真器已停在采集的起始姿态。"""
    def __init__(self):self.source=None;self.started=None;self.last=None
    def reset(self):self.source=None;self.started=None;self.last=None
    def update(self,*,source_ns,cue,pattern,hold,elapsed_ms,duration_ms,joints,neutral,velocity,now):
        def vector(v):
            return isinstance(v,(list,tuple)) and len(v)==6 and all(type(x) in (int,float) and math.isfinite(x) for x in v)
        valid=(type(source_ns) is int and source_ns>0 and cue=='INVITE' and pattern=='curiosity_intro'
               and hold is False and type(elapsed_ms) in (int,float) and math.isfinite(elapsed_ms)
               and type(duration_ms) in (int,float) and math.isfinite(duration_ms) and elapsed_ms>=duration_ms>0
               and type(now) in (int,float) and math.isfinite(now) and vector(joints) and vector(neutral) and vector(velocity))
        if valid:valid=max(abs(a-b) for a,b in zip(joints,neutral))<=.002 and max(abs(v) for v in velocity)<=.003
        if not valid:self.reset();return False
        if self.source!=source_ns or self.last is None or not 0<=now-self.last<=.3:
            self.source=source_ns;self.started=now
        self.last=now
        return now-self.started>=.3


class SimulatorPhotoGate:
    """仅供本地仿真：确认后使用稳定的实测参考位。"""
    def __init__(self):
        self.source=None;self.started=None;self.last=None

    def reset(self):
        self.source=None;self.started=None;self.last=None

    def update(self, *, source_ns, cue, pattern, hold, elapsed_ms, duration_ms,
               joints, neutral, velocity, now):
        def vector(values):
            return (isinstance(values,(list,tuple)) and len(values)==6 and
                    all(type(v) in (int,float) and math.isfinite(v) for v in values))
        valid=(type(source_ns) is int and source_ns>0 and cue=='ARRIVED'
               and pattern=='confirm_retract_hold' and hold is False
               and type(elapsed_ms) in (int,float) and math.isfinite(elapsed_ms)
               and type(duration_ms) in (int,float) and math.isfinite(duration_ms)
               and duration_ms>0 and elapsed_ms>=duration_ms
               and type(now) in (int,float) and math.isfinite(now)
               and vector(joints) and vector(neutral) and vector(velocity))
        if valid:
            valid=(max(abs(a-b) for a,b in zip(joints,neutral))<=.002
                   and max(abs(v) for v in velocity)<=.003)
        if not valid:
            self.reset();return False
        if (self.source!=source_ns or self.last is None or not 0<=now-self.last<=.3):
            self.source=source_ns;self.started=now
        self.last=now
        return now-self.started>=.3


def feedback_from_motion(data, *, token, now_ns, source_ns, cue):
    if (not isinstance(data,dict) or not isinstance(data.get('token'),str)
            or not hmac.compare_digest(data['token'],token) or data.get('hold') is not False
            or type(data.get('time_ns')) is not int or type(data.get('source_ns')) is not int
            or data['source_ns']!=source_ns or data.get('cue')!=cue
            or not 0<=now_ns-data['time_ns']<=MAX_AGE_NS):
        return None
    amplitude=data.get('amplitude')
    if type(amplitude) not in (int,float) or not math.isfinite(amplitude) or not .15<=amplitude<=1.:
        return None
    delta=data.get('delta')
    if (not isinstance(delta,list) or len(delta)!=3 or
            any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>.5 for v in delta)):
        return None
    values={'MOVE_LEFT':('left',-delta[0],SIM_HORIZONTAL_M),
            'MOVE_RIGHT':('right',delta[0],SIM_HORIZONTAL_M),
            'MOVE_UP':('top',delta[2],SIM_VERTICAL_M),
            'MOVE_DOWN':('bottom',-delta[2],SIM_VERTICAL_M),
            'MOVE_CLOSER':('all',delta[1],SIM_DEPTH_M),
            'MOVE_FARTHER':('all',-delta[1],SIM_DEPTH_M)}
    if cue not in values:
        return {'passthrough':True}
    edge,displacement,scale=values[cue]
    fast=data.get('profile')=='sim-fast'
    # 快速参考动作会反转深度，使灯光与实际动作对应：MOVE_FARTHER 时机械臂靠近观看者，MOVE_CLOSER 时远离。
    if fast and cue in ('MOVE_CLOSER','MOVE_FARTHER'):
        displacement=-displacement
    # 在 sim-fast 配置中，实测位移跟随冻结计划的幅度；其他配置沿用原先的全幅缩放。
    motion_scale=amplitude if fast else 1.
    return {'edge':edge,'tone':'green','intensity':round(min(1.,max(0.,displacement/(scale*motion_scale))),5),
            'motion_mode':'near' if fast and amplitude<.825 else 'far'}


def parse_config(value):
    if not value:return None
    try:port_text,token=value.split(':',1);port=int(port_text)
    except (ValueError,TypeError):raise ValueError('Invalid simulation feedback configuration') from None
    if not 1024<=port<=65535 or len(token)!=32 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('Invalid simulation feedback configuration')
    return port,token


class MotionFeedbackReceiver:
    def __init__(self,config):
        self.port,self.token=parse_config(config)
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1',self.port))
        self.socket.settimeout(.1)
        self.lock=threading.Lock();self.last=None;self.stopped=threading.Event()
        self.thread=threading.Thread(target=self._receive,daemon=True)

    def start(self):self.thread.start()
    def close(self):
        self.stopped.set();self.thread.join(timeout=1);self.socket.close()
    def _receive(self):
        while not self.stopped.is_set():
            try:payload,address=self.socket.recvfrom(2048)
            except socket.timeout:continue
            except OSError:break
            if address[0]!='127.0.0.1':continue
            try:data=json.loads(payload)
            except (ValueError,UnicodeDecodeError):continue
            if not isinstance(data,dict) or not isinstance(data.get('token'),str):continue
            if not hmac.compare_digest(data['token'],self.token):continue
            stamp=data.get('time_ns');now=time.monotonic_ns()
            if type(stamp) is not int or not 0<=now-stamp<=MAX_AGE_NS:continue
            with self.lock:
                if self.last is None or stamp>self.last['time_ns']:self.last=data

    def feedback(self,source_ns,cue):
        with self.lock:data=dict(self.last) if self.last else None
        return feedback_from_motion(data,token=self.token,now_ns=time.monotonic_ns(),source_ns=source_ns,cue=cue)

    def photo_ready(self,source_ns):
        with self.lock:data=dict(self.last) if self.last else None
        return photo_ready_from_motion(data,token=self.token,now_ns=time.monotonic_ns(),source_ns=source_ns)

    def intro_active(self,source_ns):
        with self.lock:data=dict(self.last) if self.last else None
        return intro_active_from_motion(data,token=self.token,now_ns=time.monotonic_ns(),source_ns=source_ns)

    def intro_ready(self,source_ns):
        with self.lock:data=dict(self.last) if self.last else None
        return intro_ready_from_motion(data,token=self.token,now_ns=time.monotonic_ns(),source_ns=source_ns)


class MotionFeedbackSender:
    def __init__(self,config):
        self.port,self.token=parse_config(config)
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    def send(self,*,source_ns,cue,hold,delta,amplitude=1.,photo_ready=False,profile='semantic',intro_ready=False):
        data={'token':self.token,'time_ns':time.monotonic_ns(),'source_ns':source_ns,
              'cue':cue,'hold':hold,'delta':delta,'amplitude':amplitude,'photo_ready':photo_ready is True,'profile':profile,
              'intro_ready':intro_ready is True}
        self.socket.sendto(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),('127.0.0.1',self.port))
    def close(self):self.socket.close()
