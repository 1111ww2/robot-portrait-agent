"""浏览器上传的相机帧使用有限内存缓存，由服务器发放带时间约束的采集凭据。"""
import json
import math
import secrets
import sys
import threading
import time

import cv2
import numpy as np

MAX_JPEG_BYTES=524288
MAX_FRAME_AGE=.3
_CAMERA_INTEGER_DIAGNOSTICS={'camera_first_error_stage':4,'camera_first_error_kind':6,
                             'camera_first_error_status':599,'camera_error_count':240,
                             'direction_audio_played':240,'direction_audio_errors':240,'direction_audio_error':4}
_CAMERA_TIMING_DIAGNOSTICS=frozenset(('camera_first_error_ms','camera_max_ticket_ms',
    'camera_max_fresh_frame_ms','camera_max_encode_ms','camera_max_upload_ms','camera_max_upload_ack_gap_ms'))


def jpeg_dimensions(data):
    if not isinstance(data,bytes) or not 4<=len(data)<=MAX_JPEG_BYTES or data[:2]!=b'\xff\xd8':
        raise ValueError('Invalid JPEG')
    offset=2
    while offset+4<=len(data):
        if data[offset]!=255:raise ValueError('Invalid JPEG marker')
        while offset<len(data) and data[offset]==255:offset+=1
        if offset>=len(data):break
        marker=data[offset];offset+=1
        if marker in (0xd9,0xda):break
        if marker==1 or 0xd0<=marker<=0xd7:continue
        length=int.from_bytes(data[offset:offset+2],'big')
        if length<2 or offset+length>len(data):break
        if marker in (0xc0,0xc1,0xc2):
            if length<8:break
            height=int.from_bytes(data[offset+3:offset+5],'big')
            width=int.from_bytes(data[offset+5:offset+7],'big')
            if not (1<=width<=960 and 1<=height<=960):raise ValueError('JPEG dimensions too large')
            return width,height
        offset+=length
    raise ValueError('JPEG size unavailable')


class IPadCapture:
    def __init__(self,*,clock=time.monotonic,read_timeout=.25):
        self.clock=clock;self.read_timeout=read_timeout;self._condition=threading.Condition()
        self._owner=None;self._session=None;self._ticket=None;self._issued=0.;self._seen=0.
        self._frame=None;self._frame_time=0.;self._last_upload=None;self._closed=False
        self.source_monotonic_ns=0
        self._floor=float('-inf')
        self._last_received=None
        self._diagnostics={}
        self._voice_sequence=0;self._voice_session=0

    def begin(self,client):
        if not isinstance(client,str) or not 1<=len(client)<=128:raise ValueError('Invalid producer')
        with self._condition:
            if self._closed:raise ValueError('Camera closed')
            if self._owner and self._owner!=client and self.clock()-self._seen<2.5:raise ValueError('Camera is in use on another device')
            self._owner=client;self._session=secrets.token_urlsafe(24);self._seen=self.clock()
            self._frame=None;self._ticket=None;self._last_upload=None
            self._last_received=None
            self._voice_sequence=0;self._voice_session+=1
            return self._session

    def _check(self,client,session):
        if self._closed or not self._owner or client!=self._owner or session!=self._session:raise ValueError('Camera session expired')

    def ticket(self,client,session,diagnostics=None):
        with self._condition:
            self._check(client,session)
            if isinstance(diagnostics,dict):
                keys={'ticket_ms','fresh_frame_ms','encode_ms','upload_ms','error_stage','error_ms','voice_local','voice_processing','preview_fps','configured_fps','video_paused','track_muted','track_ended','hidden','version','photo_flow_version','photo_started','photo_state','photo_owner','photo_latched','photo_camera_ready','photo_active','voice_available','voice_armed','voice_listening','voice_starting','voice_blocked','voice_error','voice_results','voice_submits','voice_output','voice_spoken','voice_speaker_error','voice_audio','voice_sound','voice_speech','voice_level','voice_check'}
                keys.update({'control_hb_version','control_hb_expiry','control_hb_max_gap_ms',
                             'control_hb_last_rtt_ms','control_hb_max_rtt_ms','control_hb_pending',
                             'control_hb_max_pending','control_hb_sent','control_hb_settled','control_hb_aborted'})
                timing_keys={'ticket_ms','fresh_frame_ms','encode_ms','upload_ms','error_ms',
                             'control_hb_max_gap_ms','control_hb_last_rtt_ms','control_hb_max_rtt_ms'}
                self._diagnostics={k:v for k,v in diagnostics.items() if k in keys and type(v) in (int,float,bool)
                                   and math.isfinite(v) and 0<=v<=(10000 if k in timing_keys else 240)}
                for key, maximum in _CAMERA_INTEGER_DIAGNOSTICS.items():
                    value=diagnostics.get(key)
                    if type(value) is int and 0<=value<=maximum:self._diagnostics[key]=value
                for key in ('direction_audio_enabled','direction_audio_unlocked','direction_audio_playing'):
                    value=diagnostics.get(key)
                    if type(value) is bool:self._diagnostics[key]=value
                for key in _CAMERA_TIMING_DIAGNOSTICS:
                    value=diagnostics.get(key)
                    if type(value) in (int,float) and 0<=value<=10000 and math.isfinite(value):
                        self._diagnostics[key]=value
                # 生命周期事件仅保存受限数值，不含音频、转录、令牌或客户端标识。
                events=diagnostics.get('voice_events',[])
                if isinstance(events,list) and len(events)<=12:
                    for event in events:
                        if (not isinstance(event,list) or len(event)!=5 or
                                any(type(v) is not int or not 0<=v<=10**12 for v in event)):
                            continue
                        seq,elapsed,kind,reason,generation=event
                        if seq<=self._voice_sequence or not 1<=kind<=13 or not 0<=reason<=23:continue
                        self._voice_sequence=seq
                        print('VOICE_EVENT '+json.dumps({'session':self._voice_session,'received_ms':round(self.clock()*1000),
                            'seq':seq,'elapsed_ms':elapsed,'event':kind,'reason':reason,'generation':generation}),file=sys.stderr,flush=True)
            self._ticket=secrets.token_urlsafe(24);self._issued=self.clock();self._seen=self._issued
            return self._ticket

    def diagnostics(self):
        with self._condition:return dict(self._diagnostics)

    def accept(self,client,session,ticket,data):
        with self._condition:
            self._check(client,session)
            if not self._ticket or ticket!=self._ticket:raise ValueError('Invalid camera ticket')
            self._ticket=None;issued=self._issued
            if issued<self._floor or self.clock()-issued>MAX_FRAME_AGE:raise ValueError('Frame expired')
        width,height=jpeg_dimensions(data)
        frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2]!=(height,width):raise ValueError('JPEG decode failed')
        with self._condition:
            self._check(client,session)
            if issued<self._floor or self.clock()-issued>MAX_FRAME_AGE:raise ValueError('Frame expired')
            self._frame=frame;self._frame_time=issued;self._last_upload=issued;self._seen=self.clock()
            self._last_received=self.clock()
            self._condition.notify_all()

    def read(self):
        with self._condition:
            self._condition.wait_for(lambda:self._frame is not None or self._closed,timeout=self.read_timeout)
            frame=self._frame;self._frame=None
            if frame is None or self._closed or self.clock()-self._frame_time>MAX_FRAME_AGE:return False,None
            self.source_monotonic_ns=round(self._frame_time*1e9)
            return True,frame

    def fresh(self):
        with self._condition:return self._last_upload is not None and self.clock()-self._last_upload<=MAX_FRAME_AGE

    def stream_fresh(self):
        """在限定窗口内判断最近是否有上传，与队列中帧是否有效分开。参考位失效时丢弃图像，但保留相机会话。
        动作看门狗可配合 connected() 检查，避免把主动弃帧误认为相机断流。"""
        with self._condition:
            return (not self._closed and self._owner is not None and self._last_received is not None
                    and 0<=self.clock()-self._last_received<=MAX_FRAME_AGE)

    def connected(self):
        # 界面是否在线与动作控制对数据时效性的严格要求分别判断。
        with self._condition:
            return not self._closed and self._owner is not None and self._last_received is not None and self.clock()-self._last_received<=2.5

    def control_owner(self):
        """对明确委托的本地操作员动作，使用仅供服务器内部使用的所有者身份。"""
        with self._condition:
            if (not self._closed and self._owner is not None and self._last_received is not None
                    and 0<=self.clock()-self._last_received<=MAX_FRAME_AGE):
                return self._owner
            return None

    def invalidate_pending(self):
        with self._condition:
            self._floor=self.clock();self._ticket=None;self._frame=None;self._last_upload=None

    def end(self,client,session):
        with self._condition:
            self._check(client,session)
            self._owner=None;self._session=None;self._ticket=None;self._frame=None;self._last_upload=None
            self._condition.notify_all()

    def isOpened(self):return not self._closed

    def release(self):
        with self._condition:
            self._closed=True;self._frame=None;self._ticket=None;self._last_upload=None;self._condition.notify_all()
