"""面向已配对访客的局域网界面。相机帧保存在内存中，页面不提供机器人控制接口。"""
from collections import deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import hashlib,ipaddress,json,math,os,secrets,sys,threading,time
import cv2
from .gesture_feedback import edge_feedback_for
from .local_speech import LocalSpeech, MAX_AUDIO_BYTES
from .reference_gate import ReferenceGate, RestartReferenceProof

_REFERENCE_FAILURE_REASONS={
    'Gate already blocked or invalid time':'gate_blocked',
    'Wrong reference session':'wrong_session',
    'Stale, repeated or invalid robot sample':'stale_sample',
    'Reference return and standstill not verified':'return_unverified',
    'Invalid six-axis telemetry':'invalid_reference',
    'Invalid rotation vector':'invalid_reference',
    'Invalid rotation magnitude':'invalid_reference',
    'Physical reference operation belongs to an inactive encounter':'inactive_reference',
    'Reference restart requires a new Start before intro':'restart_stage',
    'Reference restart verification not begun':'restart_not_begun',
    'Reference restart verification already begun':'restart_already_begun',
    'Communication hold has not resumed':'communication_waiting',
}


class GuestDisplay:
    def __init__(self, *, host='127.0.0.1',port=8767,lease_seconds=2.5,token=None,debug_overlay=False,remote_capture=None,physical_feedback=False):
        self.remote_capture=remote_capture
        if type(physical_feedback) is not bool or physical_feedback and remote_capture is None:
            raise ValueError('Physical feedback requires an iPad camera source')
        self._physical_feedback_enabled=physical_feedback
        self._system_run_id=os.environ.get('LEICA_SYSTEM_RUN_ID')
        if self._system_run_id is not None and (len(self._system_run_id)!=32
                or any(c not in '0123456789abcdef' for c in self._system_run_id)):
            raise ValueError('Invalid system startup identity')
        self._physical_action=None;self._physical_last_id=0;self._physical_last_epoch=None
        self._physical_backend_epoch=None;self._physical_backend_at=None
        self._physical_backend_mode='guidance'
        self._physical_revocation=None
        self._physical_lease=None;self._physical_registered_epoch=None;self._physical_revoked=False
        self._recovery_window_s=0.;self._communication_deadline=None
        self._communication_stable_since=None;self._communication_checked_at=None
        self._pending_recovery_epoch=None
        self._physical_transactions={}
        self._physical_intro=None;self._physical_photo=None
        self._runtime_status=None
        self._last_stop=None
        self.speech=LocalSpeech()
        if token is not None and (not isinstance(token,str) or not 24<=len(token)<=128 or
                any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in token)):
            raise ValueError('Invalid guest session token')
        self.token=token or secrets.token_urlsafe(24)
        self._lock=threading.Lock();self._actions=deque();self._active=False
        self._control_instance=secrets.token_hex(16);self._control_epoch=0
        self._control_action='stop'
        self._control_protocol=0;self._control_heartbeat_sequence=0
        self._control_started_ns=round(time.monotonic()*1e9)
        self._reference_gate=ReferenceGate()
        self._reference_restart=None
        self._messages=deque(maxlen=8);self._dialogue={}
        self._frame_ready=threading.Condition(self._lock)
        self._sequence=0;self._frame_times=deque(maxlen=60);self._closed=False
        self._gesture=None;self._gesture_id=0;self._debug_overlay=debug_overlay
        self._gesture_source_ns=None
        feedback_config=os.environ.get('LEICA_SIM_FEEDBACK')
        if feedback_config and physical_feedback:raise ValueError('Choose simulation or physical feedback, not both')
        if feedback_config:
            from .simulator_feedback import MotionFeedbackReceiver
            self._sim_feedback=MotionFeedbackReceiver(feedback_config)
        else:self._sim_feedback=None
        self._frame_feedback={**edge_feedback_for(None,0),'gesture_id':0}
        self._owner=None;self._seen=0.;self._lease=lease_seconds
        self._jpeg=None;self._updated=0.;self._state={};self._direction=None;self._operator={}
        self._photo_share=None;self._photo_host=host;self._photo_lock=threading.Lock()
        self._photo_latched=False
        self._last_publish=0.;self._server=ThreadingHTTPServer((host,port),self._handler())
        self._server.daemon_threads=True
        self.port=self._server.server_address[1]
        self.origin=f'http://127.0.0.1:{self.port}'
        self._thread=threading.Thread(target=self._server.serve_forever,daemon=True)

    def __enter__(self):
        if self._sim_feedback:self._sim_feedback.start()
        self._thread.start();return self
    def __exit__(self,*_):self.close()
    def close(self):
        with self._frame_ready:
            if self._physical_registered_epoch==self._control_epoch:
                self._record_physical_revocation('display_closed',time.monotonic())
            self._closed=True;self._frame_ready.notify_all()
        self._server.shutdown();self._server.server_close();self._thread.join(timeout=2)
        if self._sim_feedback:self._sim_feedback.close()
        with self._photo_lock:
            if self._photo_share:self._photo_share.close();self._photo_share=None

    def _current_feedback(self):
        """调用者须持有显示锁；没有仿真遥测时熄灭提示灯。"""
        if self._physical_feedback_enabled:
            empty={**edge_feedback_for(None,0),'gesture_id':0}
            data=self._physical_action;now=time.monotonic()
            if (not data or not self._physical_backend_available_locked(now)
                    or data['epoch']!=self._control_epoch):
                communication=self._communication_locked(now)
                retained=bool(data and data['epoch']==self._control_epoch and self._active
                    and not self._closed and not self._physical_revoked
                    and communication['recoverable'] and communication['status'] in ('waiting','ready'))
                if not retained:self._physical_action=None
                return empty
            if data['feedback'] is None:return empty
            if not 0<=now-data['captured_at']<=.3:
                # 缺少显示样本时熄灭提示灯，但保留已接受动作的标识，以便下一条实测数据到达时恢复显示。
                return empty
            return {**data['feedback'],'gesture_id':data['action_id']}
        if self._sim_feedback is None:return self._state.get('feedback',self._frame_feedback)
        empty={**edge_feedback_for(None,0),'gesture_id':self._gesture_id}
        if (not self._active or not self._state.get('tracking') or self._gesture is None
                or not self._updated or time.monotonic()-self._updated>1.
                or (self.remote_capture and not self.remote_capture.fresh())):
            return empty
        observed=self._sim_feedback.feedback(self._gesture_source_ns,self._gesture.cue)
        if observed is None:return empty
        if observed.get('passthrough'):return self._state.get('feedback',empty)
        return {**observed,'gesture_id':self._gesture_id}

    def _current_motion(self):
        """发布与视觉跟踪独立的实测动作状态。
        在 _current_feedback 清除旧动作标识后，持有 _lock 调用本方法。
        结果不包含租约、所有者标识或源时间戳。"""
        empty={'active':False,'fresh':False}
        data=self._physical_action
        if not self._physical_feedback_enabled or not data or data['feedback'] is None:return empty
        now=time.monotonic();captured=data['captured_at']
        if (not self._physical_backend_available_locked(now) or data['epoch']!=self._control_epoch
                or captured is None or not 0<=now-captured<=.3):return empty
        return {'active':True,'fresh':True,'cue':data['cue'],'action_id':data['action_id'],
                'started':data.get('motion_started',False),
                'phase':data.get('phase','moving'),'progress':data['feedback']['intensity'],
                'sample_age_ms':round((now-captured)*1000,3)}

    def _physical_feedback_update(self,body):
        """处理经过认证的本地实测反馈，调用者须持有 _lock。"""
        from .physical_feedback import physical_edge_feedback
        if not self._physical_feedback_enabled or not isinstance(body,dict):raise ValueError('Physical feedback unavailable')
        now=time.monotonic();op=body.get('op')
        self._expire()
        if not self._active or self._closed:
            self._physical_action=None;raise ValueError('Guest or camera inactive')
        if (body.get('instance')!=self._control_instance or type(body.get('epoch')) is not int
                or body['epoch']!=self._control_epoch):raise ValueError('Invalid physical backend identity')
        if op=='fault':
            lease=body.get('lease')
            if (not isinstance(lease,str) or not isinstance(self._physical_lease,str)
                    or not secrets.compare_digest(lease,self._physical_lease)
                    or self._physical_registered_epoch!=self._control_epoch):
                raise ValueError('Invalid physical backend lease')
            self._record_physical_revocation('backend_fault',now)
            self._physical_revoked=True;self._physical_action=None;self._physical_intro=None;self._physical_photo=None
            return
        if op=='register':
            mode=body.get('mode','guidance')
            if mode not in ('intro','guidance'):raise ValueError('Invalid physical backend mode')
            window=body.get('recovery_window_s')
            if 'recovery_window_s' in body and (type(window) not in (int,float) or window!=3.0):
                raise ValueError('Invalid communication recovery window')
            pending=getattr(self,'_pending_recovery_epoch',None)==self._control_epoch
            if pending and window!=3.0:
                raise ValueError('Pending encounter requires matching recovery negotiation')
            cached=self._transaction_locked(op,body)
            if cached is not None:
                if not self._physical_credentials_locked(now):raise ValueError('Physical registration expired')
                return cached
            if (not self._camera_stream_fresh() or not self._control_stream_fresh(now)) and not pending:
                raise ValueError('Guest or camera inactive')
            if self._physical_registered_epoch==self._control_epoch:
                raise ValueError('Physical backend already registered for this encounter')
            self._physical_registered_epoch=self._control_epoch
            self._physical_lease=secrets.token_urlsafe(24);self._physical_revoked=False
            self._physical_backend_epoch=self._control_epoch;self._physical_backend_at=now
            self._physical_backend_mode=mode
            self._physical_revocation=None
            pending_deadline=self._communication_deadline if pending else None
            self._reset_communication_locked(window=window or 0.)
            self._communication_deadline=pending_deadline
            self._physical_transactions={}
            self._physical_action=None;self._physical_intro=None;self._physical_photo=None
            self._reference_restart=None
            result={'lease':self._physical_lease}
            self._transaction_locked(op,body,result)
            return result
        lease=body.get('lease')
        if (not isinstance(lease,str) or not isinstance(self._physical_lease,str)
                or not secrets.compare_digest(lease,self._physical_lease)
                or self._physical_registered_epoch!=self._control_epoch):
            raise ValueError('Invalid physical backend lease')
        if op=='resume':
            if (not self._recovery_enabled_locked() or not self._physical_credentials_locked(now)
                    or self._communication_locked(now)['status'] not in ('healthy','ready')):
                raise ValueError('Communication is not ready to resume')
            self._reset_communication_locked(window=self._recovery_window_s)
            return {'communication':self._communication_locked(now)}
        allowed=(self._physical_credentials_locked(now) if op=='heartbeat' and self._recovery_enabled_locked()
                 else self._physical_backend_available_locked(now))
        if not allowed:
            communication=self._communication_locked(now)
            temporary=(communication['recoverable'] and communication['status'] in ('waiting','ready')
                       and self._physical_credentials_locked(now))
            # 原生控制处于 HOLD 时拒绝在途的显示数据包，同时保留之后需要继续执行的动作。
            if not temporary:self._physical_action=None
            raise ValueError('Physical backend lease expired or revoked')
        if self._physical_backend_mode=='intro' and op not in ('clear','heartbeat','intro'):
            raise ValueError('Physical guidance requires completed intro')
        if op=='clear':
            action_id=body.get('action_id')
            if action_id is not None:
                if type(action_id) is not int or action_id<=0:raise ValueError('Invalid physical clear identity')
                if self._physical_action is not None and self._physical_action['action_id']!=action_id:
                    return {'cleared':False}
            self._physical_action=None
            return {'cleared':True}
        if op in ('heartbeat','intro','photo_ready'):
            if op=='heartbeat':
                if 'intro_id' in body or 'intro_status' in body:
                    # 先验证完整的阶段转换，再续期控制租约。通过一次有超时限制的关键 RPC 同时提交两项结果。
                    self._accept_physical_intro_locked(body.get('intro_id'),body.get('intro_status'),now)
                self._physical_backend_epoch=self._control_epoch;self._physical_backend_at=now
                return {'communication':self._communication_locked(now)}
            if op=='intro':
                self._accept_physical_intro_locked(body.get('intro_id'),body.get('status'),now)
                return
            photo=self._physical_photo
            if not photo or photo['epoch']!=self._control_epoch or body.get('source_ns')!=photo['source_ns']:
                raise ValueError('Photo stop confirmation mismatch')
            photo['ready_at']=now;return
        if op=='begin':
            action_id=body.get('action_id');source_ns=body.get('source_ns')
            # 动作监管器先接纳视觉输入，再将提示灯元数据入队。传输延迟不能使已接受的有限轨迹被重复接纳或过期；
            # 保留原始且不晚于当前时间的源标识。
            if (body.get('instance')!=self._control_instance or type(body.get('epoch')) is not int
                    or body['epoch']!=self._control_epoch or type(action_id) is not int or action_id<=0
                    or type(source_ns) is not int or source_ns<=0 or source_ns/1e9>now):
                raise ValueError('Invalid physical action identity')
            if self._physical_last_epoch!=self._control_epoch:
                self._physical_last_id=0;self._physical_last_epoch=self._control_epoch
            if action_id<=self._physical_last_id:
                raise ValueError('Replayed physical action')
            reference=body.get('reference_position');direction=body.get('direction');planned=body.get('planned_displacement')
            initial=physical_edge_feedback(cue=body.get('cue'),action_id=action_id,expected_action_id=action_id,
                started_at=now,captured_at=now,now=now,max_age=.3,active=True,
                reference_position=reference,current_position=reference,direction=direction,planned_displacement=planned)
            if initial is None:
                raise ValueError('Invalid physical action feedback parameters')
            self._physical_last_id=action_id
            self._physical_action={'epoch':self._control_epoch,'action_id':action_id,'source_ns':source_ns,
                'cue':body['cue'],'started_at':now,'reference':reference,'direction':direction,
                'planned':planned,'captured_at':None,'feedback':None,'phase':'moving','motion_started':False}
            return
        data=self._physical_action
        if (op!='sample' or not data or data['epoch']!=self._control_epoch
                or type(body.get('action_id')) is not int or body['action_id']!=data['action_id']
                or type(body.get('source_ns')) is not int or body['source_ns']!=data['source_ns']
                or body.get('cue')!=data['cue']):
            # 迟到的旧样本不能覆盖新的有效反馈。如果样本引用了尚未声明的后续动作，先熄灭旧方向提示，待该动作正式开始再显示。
            if (data and type(body.get('action_id')) is int
                    and body['action_id']>data['action_id']):self._physical_action=None
            raise ValueError('Physical action feedback mismatch')
        captured=body.get('captured_at')
        phase=body.get('phase','moving')
        started=body.get('motion_started',False)
        if type(started) is not bool:
            raise ValueError('Invalid physical motion start evidence')
        if phase not in ('moving','returning','settling'):
            raise ValueError('Invalid physical motion phase')
        if data['captured_at'] is not None and (type(captured) not in (int,float) or captured<=data['captured_at']):
            raise ValueError('Physical telemetry repeated')
        result=physical_edge_feedback(cue=data['cue'],action_id=body['action_id'],expected_action_id=data['action_id'],
            started_at=data['started_at'],captured_at=captured,now=now,max_age=.3,active=True,
            reference_position=data['reference'],current_position=body.get('current_position'),
            direction=data['direction'],planned_displacement=data['planned'])
        if result is None:
            raise ValueError('Physical telemetry invalid or stale')
        data['captured_at']=captured;data['feedback']=result;data['phase']=phase
        data['motion_started']=data.get('motion_started',False) or started

    def _accept_physical_intro_locked(self,intro_id,status,now):
        if type(intro_id) is not int or intro_id<=0 or status not in ('active','ready'):
            raise ValueError('Invalid physical intro status')
        prior=self._physical_intro
        if status=='ready' and (not prior or prior['id']!=intro_id or prior['epoch']!=self._control_epoch):
            raise ValueError('Physical intro has not started')
        if status=='ready':
            # 这是唯一允许的前向状态转换。视觉引导已经完成后，外部模式覆盖不能将流程退回开场。
            self._physical_backend_mode='guidance'
        self._physical_intro={'id':intro_id,'epoch':self._control_epoch,'status':status,'at':now}

    @staticmethod
    def _stop_origin(body):
        value=body.get('stop_origin',0)
        return value if type(value) is int and 0<=value<=10 else 0

    def _record_control_stop_locked(self,body,client,route):
        """锁定活跃会话中第一次被接受的停止记录，仅保留数值字段。"""
        if not self._active:return
        if self._last_stop is not None and self._last_stop['epoch']==self._control_epoch:return
        now=time.monotonic();expected=body.get('control')
        def age(value):
            return round(now-value,6) if type(value) in (int,float) and math.isfinite(value) else None
        camera=self.remote_capture;stamp=None
        condition=getattr(camera,'_condition',None)
        if condition is not None:
            with condition:stamp=getattr(camera,'_last_received',None)
        elif camera is not None:stamp=getattr(camera,'_last_received',None)
        self._last_stop={'epoch':self._control_epoch,'at_s':round(now,6),
            'stop_origin':self._stop_origin(body),'route':route,
            'owner_matches':self._owner==client,
            'control_matches':bool(isinstance(expected,dict) and expected.get('instance')==self._control_instance
                and type(expected.get('epoch')) is int and expected['epoch']==self._control_epoch),
            'control_age_s':age(self._seen),'backend_age_s':age(getattr(self,'_physical_backend_at',None)),
            'camera_age_s':age(stamp)}
        print('CONTROL_STOP_ACCEPTED '+json.dumps(self._last_stop,sort_keys=True),file=sys.stderr,flush=True)

    def _record_physical_revocation(self,reason,now):
        """只保留首次触发的原因和各项经过时间。调用者须持有显示锁。"""
        if self._physical_revocation is not None:return
        def age(stamp):
            return round(now-stamp,6) if type(stamp) in (int,float) and math.isfinite(stamp) else None
        camera=self.remote_capture;camera_stamp=None
        condition=getattr(camera,'_condition',None)
        if condition is not None:
            with condition:camera_stamp=getattr(camera,'_last_received',None)
        elif camera is not None:camera_stamp=getattr(camera,'_last_received',None)
        self._physical_revocation={'reason':reason,'epoch':self._control_epoch,
            'mode':self._physical_backend_mode,'control_age_s':age(self._seen),
            'backend_age_s':age(self._physical_backend_at),'camera_age_s':age(camera_stamp)}
        print('PHYSICAL_CONTROL_REVOKED '+json.dumps(self._physical_revocation,sort_keys=True),
              file=sys.stderr,flush=True)

    def _physical_backend_available_locked(self,now):
        # 电机控制权由所有者与后端控制流决定。仅在接纳源帧时检查画面新鲜度，已经被接纳的轨迹及其实测完成反馈不再依赖画面。
        if self._recovery_enabled_locked():
            communication=self._communication_locked(now)
            return (self._physical_credentials_locked(now) and communication['status']=='healthy'
                    and self._control_stream_fresh(now) and self._camera_stream_fresh()
                    and self._physical_backend_at is not None and 0<=now-self._physical_backend_at<=.3)
        available=bool(self._physical_feedback_enabled and self._active and not self._closed
                    and self._physical_backend_epoch==self._control_epoch
                    and self._physical_registered_epoch==self._control_epoch and not self._physical_revoked
                    and self._physical_lease is not None
                    and self._physical_backend_at is not None
                    and 0<=now-self._physical_backend_at<=.3
                    and self._control_stream_fresh(now))
        if not available and self._physical_registered_epoch==self._control_epoch:
            if self._closed:reason='display_closed'
            elif not self._active:reason='guest_inactive'
            elif self._physical_revoked:reason='backend_revoked'
            elif self._physical_backend_epoch!=self._control_epoch:reason='epoch_changed'
            elif self._physical_lease is None or self._physical_backend_at is None:reason='backend_missing'
            elif now-self._physical_backend_at<0:reason='backend_clock_invalid'
            elif now-self._physical_backend_at>.3:reason='backend_heartbeat_expired'
            elif not self._control_stream_fresh(now):reason='control_heartbeat_expired'
            else:reason='backend_unavailable'
            self._record_physical_revocation(reason,now)
            self._physical_revoked=True;self._physical_action=None
        return available

    def _recovery_enabled_locked(self):
        return bool(getattr(self,'_recovery_window_s',0.)==3.0
                    and (self._physical_registered_epoch==self._control_epoch
                         or getattr(self,'_pending_recovery_epoch',None)==self._control_epoch))

    def _restart_reference_locked(self,body,now):
        """新开场前重新验证已停止会话的原参考位。入口已经检查当前控制凭据和租约；
        此处不替换参考位，没有新的实测停留证据也不能放行。"""
        op=body['op'];gate=self._reference_gate
        if (not self._physical_feedback_enabled or self._physical_backend_mode!='intro'
                or self._control_action not in ('start','restart') or self._physical_intro is not None):
            raise ValueError('Reference restart requires a new Start before intro')
        if op in ('restart_begin','restart_release'):
            if 'request_id' not in body:raise ValueError('Restart request identity required')
            cached=self._transaction_locked(op,body)
            if cached is not None:return cached
        session=body.get('session')
        if not gate.blocked or not isinstance(session,str) or session!=gate.session:
            raise ValueError('Wrong reference session')
        if op=='restart_begin':
            if self._reference_restart is not None:
                raise ValueError('Reference restart verification already begun')
            policy=body.get('reference_policy')
            if policy not in (None,'capture_stationary_on_explicit_start'):
                raise ValueError('Invalid restart reference policy')
            gate.steady_since=gate.last_sample=None
            proof=RestartReferenceProof(gate) if policy is not None else None
            if proof is not None:gate.acknowledged=False
            self._reference_restart={'epoch':self._control_epoch,'session':session,'began_at':now,
                                     'proof':proof,'reference_policy':policy}
            result={'ok':True,'reference':list(gate.reference)}
            if policy is not None:result['reference_policy']=policy
            self._transaction_locked(op,body,result)
            return result
        restart=self._reference_restart
        if (not isinstance(restart,dict) or restart['epoch']!=self._control_epoch
                or restart['session']!=session):
            raise ValueError('Reference restart verification not begun')
        proof=restart.get('proof') or gate
        fields=('pose','speed','sampled_at')
        atomic_release=op=='restart_release' and any(field in body for field in fields)
        if op=='restart_sample' or atomic_release:
            if not all(field in body for field in fields):
                raise ValueError('Complete restart release sample required')
            # 样本被接收后若响应丢失，重试不能重放样本或重置其有效期和停留计时。已完成的释放请求由上面的结果缓存处理。
            applied=(self._transaction_locked('restart_release_sample',body)
                     if atomic_release else None)
            if applied is None:
                stamp=body['sampled_at']
                if (type(stamp) not in (int,float) or not math.isfinite(stamp)
                        or stamp<restart['began_at']):
                    raise ValueError('Stale, repeated or invalid robot sample')
                proof.sample(session,body['pose'],body['speed'],stamp,now)
                if atomic_release:
                    self._transaction_locked('restart_release_sample',body,{'sample_applied':True})
            if op=='restart_sample':
                return {'ok':True,'sampled_at':stamp,'return_verified':proof.return_verified(session,now),
                        'reference_check':proof.check_details(now)}
        if op=='restart_release':
            reference=proof.reference
            proof.release(session,now)
            result={'ok':True,'release_ns':gate.release_ns}
            if restart.get('proof') is not None:result['reference']=list(reference)
            self._transaction_locked(op,body,result)
            return result
        raise ValueError('Invalid reference restart operation')

    def _reference_check_locked(self,now):
        restart=self._reference_restart
        proof=(restart.get('proof') if isinstance(restart,dict)
               and restart.get('session')==self._reference_gate.session else None)
        return (proof or self._reference_gate).check_details(now)

    def _transaction_locked(self,op,body,result=None):
        """对提交结果不确定的开始或释放请求，在限定范围内重用同一载荷的回复。"""
        if 'request_id' not in body:return None
        request_id=body['request_id']
        if (not isinstance(request_id,str) or len(request_id)!=32
                or any(c not in '0123456789abcdef' for c in request_id)):
            raise ValueError('Invalid request identity')
        fingerprint=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()
        transactions=getattr(self,'_physical_transactions',{})
        key=(op,request_id)
        cached=transactions.get(key)
        if cached is not None:
            if cached[0]!=fingerprint:raise ValueError('Request identity payload changed')
            return dict(cached[1])
        if result is not None:
            if len(transactions)>=32:transactions.pop(next(iter(transactions)))
            transactions[key]=(fingerprint,dict(result));self._physical_transactions=transactions
        return None

    def _reset_communication_locked(self, *, window=0.):
        self._recovery_window_s=window;self._communication_deadline=None
        self._communication_stable_since=None;self._communication_checked_at=None
        self._pending_recovery_epoch=None

    def _physical_credentials_locked(self,now):
        """电机短暂处于 HOLD 时，可以保留本次交互的凭据。"""
        self._communication_locked(now)
        return bool(self._physical_feedback_enabled and self._active and not self._closed
                    and self._physical_backend_epoch==self._control_epoch
                    and self._physical_registered_epoch==self._control_epoch and not self._physical_revoked
                    and isinstance(self._physical_lease,str) and self._physical_backend_at is not None
                    and 0<=now-self._physical_backend_at)

    def _end_expired_encounter_locked(self,reason,now):
        if self._physical_registered_epoch==self._control_epoch:
            self._record_physical_revocation(reason,now);self._physical_revoked=True
        self._control_epoch+=1;self._control_action='stop'
        self._control_started_ns=round(now*1e9);self._photo_latched=False
        self._active=False;self._actions.clear();self._actions.append('stop')
        self._messages.clear();self._dialogue={};self._physical_action=None
        self._physical_transactions={}
        self._reference_restart=None
        self._reset_communication_locked()

    def _communication_locked(self,now):
        """检查真实数据流是否新鲜，并使用协商好的固定恢复截止时间。此检查不会启动电机或刷新旧数据；
        只有经过认证的原生 HOLD 恢复操作才能解除该状态。"""
        empty={'status':'healthy','recoverable':False,'remaining_ms':0}
        registered=self._physical_registered_epoch==self._control_epoch
        if (not self._recovery_enabled_locked() or not self._active or self._closed
                or registered and self._physical_revoked):return empty
        backend=self._physical_backend_at
        if (type(now) not in (int,float) or not math.isfinite(now)
                or registered and (backend is None or not 0<=now-backend) or not 0<=now-self._seen):
            self._record_physical_revocation('backend_clock_invalid',now)
            self._physical_revoked=True;self._physical_action=None
            return empty
        control_fresh=self._control_stream_fresh(now)
        camera_fresh=self._camera_stream_fresh()
        backend_fresh=not registered or now-backend<=.3
        fresh=control_fresh and camera_fresh and backend_fresh
        if self._communication_deadline is None and not fresh:
            # 使用实际中断时长，即使第一次观测来得较晚也要计入。延迟轮询不能额外延长三秒。
            losses=[]
            if not control_fresh:losses.append(self._seen+.3)
            if not backend_fresh:losses.append(backend+.3)
            if not camera_fresh:
                camera=self.remote_capture;condition=getattr(camera,'_condition',None)
                if condition is not None:
                    with condition:stamp=getattr(camera,'_last_received',None)
                else:stamp=getattr(camera,'_last_received',None)
                losses.append(stamp+.3 if type(stamp) in (int,float) and math.isfinite(stamp) else now)
            self._communication_deadline=min(losses)+3.0
            self._communication_stable_since=None
            # 中断前的到位或停止许可不能用于之后的拍摄。已经发布的 PhotoShare 图像和下载地址继续有效。
            self._physical_photo=None;self._photo_latched=False
        deadline=self._communication_deadline
        if deadline is None:return {**empty,'recoverable':True}
        if now>=deadline:
            self._end_expired_encounter_locked('communication_timeout',now)
            return empty
        prior=self._communication_checked_at
        self._communication_checked_at=now
        if not fresh:self._communication_stable_since=None
        elif (self._communication_stable_since is None or prior is None or now-prior>.3):
            self._communication_stable_since=now
        stable=self._communication_stable_since
        return {'status':'ready' if fresh and stable is not None and now-stable>=.3-1e-9 else 'waiting',
                'recoverable':True,'remaining_ms':round(max(0.,deadline-now)*1000,3)}

    def physical_backend_available(self):
        with self._lock:
            self._expire();return self._physical_backend_available_locked(time.monotonic())

    @property
    def physical_feedback_enabled(self):return self._physical_feedback_enabled

    def _physical_intro_status(self,intro_id,status):
        with self._lock:
            self._expire();now=time.monotonic();intro=self._physical_intro
            return bool(self._physical_backend_available_locked(now) and intro and
                        intro['id']==intro_id and intro['epoch']==self._control_epoch and
                        0<=now-intro['at']<=.3 and (status=='active' or intro['status']=='ready'))

    def physical_intro_active(self,intro_id):return self._physical_intro_status(intro_id,'active')
    def physical_intro_ready(self,intro_id):return self._physical_intro_status(intro_id,'ready')

    def arm_physical_photo(self,source_ns):
        with self._lock:
            self._expire();now=time.monotonic()
            if (self._physical_backend_mode!='guidance' or not self._physical_backend_available_locked(now) or type(source_ns) is not int
                    or not 0<=now-source_ns/1e9<=.3):raise ValueError('Cannot arm physical photo')
            self._physical_photo={'epoch':self._control_epoch,'source_ns':source_ns,'ready_at':None}

    def physical_photo_ready(self,source_ns):
        with self._lock:
            self._expire();now=time.monotonic();photo=self._physical_photo
            return bool(self._physical_backend_available_locked(now) and photo and
                        photo['epoch']==self._control_epoch and photo['source_ns']==source_ns and
                        photo['ready_at'] is not None and 0<=now-photo['ready_at']<=.3)

    def _clear_photo(self):
        with self._photo_lock:
            if self._photo_share:self._photo_share.clear()

    def _photo_ready_locked(self):
        # 拍摄锁定后不再依赖跟踪或相机上传的新鲜度。每次上传仍要检查已配对的所有者和会话轮次。
        return self._active and self._photo_latched

    def _expire(self):
        if self._recovery_enabled_locked() and not (self._physical_registered_epoch==self._control_epoch and self._physical_revoked):
            self._communication_locked(time.monotonic())
            return
        if self._active and time.monotonic()-self._seen>self._lease:
            self._end_expired_encounter_locked('guest_expired',time.monotonic())

    @property
    def active(self):
        with self._lock:self._expire();return self._active

    def _camera_stream_fresh(self):
        if not self.remote_capture:return False
        reader=getattr(self.remote_capture,'stream_fresh',self.remote_capture.fresh)
        return bool(reader())

    def _control_stream_fresh(self,now):
        """仅用访客所有者的实际轮询刷新在线状态，机器人后端流量不能代替它。"""
        return bool(self._active and not self._closed and 0<=now-self._seen<=.3)

    def control_snapshot(self):
        """原子读取进程内控制状态，不续期访客租约。动作监管器可以独立于视觉和模型循环进行轮询。
        相机标志取自 iPad 原始上传，不依赖预览或模型。开始、停止、暂停相机和租约到期都会切换轮次；
        关闭时清空所有就绪标志。这些是软件状态，不构成实体急停或安全认证。"""
        with self._lock:
            self._expire()
            return {'instance':self._control_instance,'epoch':self._control_epoch,
                    'control_action':self._control_action if not self._closed else 'stop',
                    'control_started_ns':self._control_started_ns,
                    'active':self._active and not self._closed,
                    'control_fresh':self._control_stream_fresh(time.monotonic()),
                    'communication':self._communication_locked(time.monotonic()),
                    'camera_connected':bool(not self._closed and self.remote_capture and self.remote_capture.connected()),
                    'camera_fresh':bool(not self._closed and self._camera_stream_fresh())}

    def take_action(self):
        with self._lock:
            self._expire()
            return self._actions.popleft() if self._actions else None

    def _control_heartbeat_locked(self,body,client,now):
        """仅检查当前所有者在本轮会话中的在线状态，不启动或恢复动作。"""
        self._expire()
        if (not isinstance(body,dict) or not self._active or self._closed
                or self._owner!=client or self._control_protocol!=1
                or body.get('instance')!=self._control_instance
                or type(body.get('epoch')) is not int or body['epoch']!=self._control_epoch):
            raise ValueError('Control heartbeat belongs to an inactive encounter')
        sequence=body.get('sequence')
        if type(sequence) is not int or not 0<sequence<2**31:
            raise ValueError('Invalid control heartbeat sequence')
        # 先检查会话是否过期，再更新活跃时间。延迟或在途请求不能使过期的实机会话重新生效。
        if self._physical_feedback_enabled:
            if self._physical_registered_epoch==self._control_epoch:
                valid=(self._physical_credentials_locked(now) if self._recovery_enabled_locked()
                       else self._physical_backend_available_locked(now))
                if not valid:
                    raise ValueError('Physical control expired')
            elif not self._recovery_enabled_locked() and not self._control_stream_fresh(now):
                raise ValueError('Physical control expired before registration')
        if sequence<=self._control_heartbeat_sequence:return {'accepted':False}
        self._control_heartbeat_sequence=sequence;self._seen=now
        return {'accepted':True,'communication':self._communication_locked(now)}

    def _set_action_locked(self,action,client,control_protocol=0, *, recovery_window_s=0.):
        if self._physical_registered_epoch==self._control_epoch:
            self._record_physical_revocation('guest_stop' if action=='stop' else 'epoch_changed',time.monotonic())
        self._control_epoch+=1
        self._reset_communication_locked(window=recovery_window_s if self._physical_feedback_enabled and action!='stop' else 0.)
        if self._recovery_window_s==3.0:self._pending_recovery_epoch=self._control_epoch
        self._physical_transactions={}
        if action!='stop':self._last_stop=None
        self._reference_restart=None
        self._control_action=action
        self._control_protocol=control_protocol;self._control_heartbeat_sequence=0
        self._control_started_ns=round(time.monotonic()*1e9)
        self._photo_latched=False
        self._active=action!='stop';self._owner=client;self._seen=time.monotonic()
        self._state={**self._state,'phase':'GUEST_READY','tracking':False,'direction':None,'ready':False}
        self._gesture=None;self._gesture_source_ns=None;self._direction=None
        self._actions.clear();self._actions.append(action)
        self._messages.clear();self._dialogue={}

    def take_control_action(self):
        """原子地取出待处理动作，同时记录它所属的访客控制轮次。"""
        with self._lock:
            self._expire()
            return {'action':self._actions.popleft() if self._actions else None,
                    'instance':self._control_instance,'epoch':self._control_epoch,
                    'control_started_ns':self._control_started_ns,
                    'active':self._active and not self._closed}

    def take_messages(self):
        with self._lock:
            self._expire()
            messages=list(self._messages);self._messages.clear();return messages

    def reference_held(self,source_ns):
        with self._lock:return self._reference_gate.held(source_ns)

    def simulator_motion_active(self,source_ns,cue):
        """返回本地桥接器当前的确认结果；未配置时返回 None。"""
        if self._sim_feedback is None:return None
        return self._sim_feedback.feedback(source_ns,cue) is not None

    def simulator_intro_available(self):
        return self._sim_feedback is not None

    def simulator_intro_active(self,intro_id):
        return bool(self._sim_feedback and self._sim_feedback.intro_active(intro_id))

    def simulator_intro_ready(self,intro_id):
        return bool(self._sim_feedback and self._sim_feedback.intro_ready(intro_id))

    def _pause_camera_locked(self):
        """在调用者持有显示锁时取消相机控制。"""
        if self._physical_registered_epoch==self._control_epoch:
            self._record_physical_revocation('epoch_changed',time.monotonic())
        self._control_epoch+=1
        self._reset_communication_locked()
        self._physical_transactions={}
        self._control_action='stop'
        self._reference_restart=None
        self._control_started_ns=round(time.monotonic()*1e9)
        self._photo_latched=False
        self._active=False;self._actions.clear();self._actions.append('stop')
        self._messages.clear();self._dialogue={}

    def pause_camera(self):
        with self._lock:self._pause_camera_locked()

    def _camera_renewal_preserves_control_locked(self,camera,client):
        """识别同一会话的所有者，不凭空刷新图像时间。"""
        now=time.monotonic()
        retained=(self._physical_credentials_locked(now) if self._recovery_enabled_locked()
                  else self._control_stream_fresh(now))
        if (not self._physical_feedback_enabled or not self._active or self._owner!=client or not retained):
            return False
        condition=getattr(camera,'_condition',None)
        if condition is None:return False
        # 统一先获取显示锁，再获取采集条件锁。
        # 这里在真正缺帧时仍要保留会话身份，和 control_owner() 的处理不同。
        with condition:
            return (not camera._closed and camera._owner==client and camera._session is not None)

    def set_dialogue(self,dialogue):
        with self._lock:
            self._dialogue=dict(dialogue) if self._active else {}

    def publish(self,frame,decision, *, gesture=None,gesture_elapsed_ms=0,source_monotonic_ns=None):
        now=time.monotonic()
        if not decision.tracking or decision.cue.value=='HOLD':gesture=None
        elapsed=gesture_elapsed_ms
        if source_monotonic_ns is not None:
            elapsed+=max(0.,(now-source_monotonic_ns/1e9)*1000)
        with self._lock:
            if self._active and decision.phase=='ARRIVED' and decision.reason=='TARGET_LATCHED':
                self._photo_latched=True
            restart=self._reference_restart
            selected_restart=(isinstance(restart,dict) and restart.get('proof') is not None
                              and restart.get('session')==self._reference_gate.session)
            if (decision.phase=='PHYSICAL_REFERENCE_HOLD' and decision.cue.value=='HOLD'
                    and not decision.tracking and not selected_restart):
                self._reference_gate.acknowledge()
            if (self._physical_feedback_enabled and self._active and not self._closed
                    and not self._physical_revoked and self._physical_backend_mode=='intro'
                    and self._physical_intro is None
                    and isinstance(restart,dict) and restart['epoch']==self._control_epoch
                    and self._physical_registered_epoch==self._control_epoch
                    and restart['session']==self._reference_gate.session
                    and decision.phase in ('INTRO','PHYSICAL_REFERENCE_HOLD')
                    and decision.cue.value=='HOLD' and not decision.tracking
                    and frame is not None and type(source_monotonic_ns) is int
                    and source_monotonic_ns/1e9>restart['began_at']
                    and 0<=now-source_monotonic_ns/1e9<=.3):
                # 开场动作前，应用以 INTRO/HOLD 发布真实新帧。
                # 缺帧时控制心跳使用 frame=None，不能将其视作观测暂停的确认。
                self._reference_gate.acknowledge()
            if gesture is not self._gesture:
                self._gesture_id+=1;self._gesture=gesture
                self._gesture_source_ns=source_monotonic_ns if gesture is not None else None
            feedback={**edge_feedback_for(gesture,elapsed),'gesture_id':self._gesture_id}
            if decision.cue.value.startswith('MOVE_'):self._direction=decision.cue.value
            if not decision.tracking or decision.phase in ('ARRIVED','SETTLING','GUEST_READY'):
                self._direction=None
            self._updated=now
            self._state={'phase':decision.phase,'tracking':decision.tracking,
                         'direction':self._direction,'progress':decision.arrival_progress,
                         'ready':decision.phase=='ARRIVED','feedback':feedback,
                         'agent':decision.interaction_mode=='claude-agent','composition':decision.composition_frame}
            geometry=frame is not None and decision.current_point is not None
            current=decision.filtered_point or decision.current_point
            target=decision.model_target if decision.interaction_mode=='claude-agent' else decision.target
            if decision.interaction_mode in ('claude-guide','gemini-guide') and decision.composition_frame is None:target=None
            self._operator={'cue':decision.cue.value,'reason':decision.reason,
                'target':target if geometry else None,
                'current':current if geometry else None,
                'photo_current':decision.current_point if geometry else None,
                'tolerances':decision.target_tolerances if geometry and target else None,
                'coordinate_frame':'mirrored_camera_image','physical_motion_verified':False,
                'profile':decision.interaction_mode,'source_monotonic_ns':source_monotonic_ns,
                'gesture_pattern':getattr(gesture,'pattern',None),
                'gesture_elapsed_ms':elapsed,'model_requests':decision.model_requests,
                'composition':decision.composition_frame if geometry else None}
        if frame is None:
            with self._lock:self._frame_feedback=feedback
            return  # 跟踪丢失时保留最后一张图，状态仍显示 HOLD。
        if now-self._last_publish<.025:return
        height,width=frame.shape[:2]
        image=(cv2.resize(frame,(768,round(height*768/width))) if width>768 else frame.copy())
        height,width=image.shape[:2]
        crop=decision.composition_frame
        if self._debug_overlay and crop and decision.tracking:
            left,top,right,bottom=crop
            cv2.rectangle(image,(round(left*width),round(top*height)),
                          (round(right*width)-1,round(bottom*height)-1),(230,240,250),2)
            x,y,_=decision.target;tx,ty,_=decision.target_tolerances
            overlay=image.copy()
            a=(round((x-tx)*width),round((y-ty)*height));b=(round((x+tx)*width),round((y+ty)*height))
            cv2.rectangle(overlay,a,b,(90,180,90),-1)
            image=cv2.addWeighted(overlay,.22,image,.78,0)
            cv2.rectangle(image,a,b,(140,230,160),2)
        if self._debug_overlay and decision.current_point and decision.tracking:
            x,y,_=decision.current_point
            cv2.circle(image,(round(x*width),round(y*height)),7,(70,210,255),-1)
        ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,64])
        if not ok:return
        with self._frame_ready:
            self._jpeg=jpeg.tobytes();self._updated=now;self._last_publish=now
            self._frame_feedback=feedback
            self._sequence+=1;self._frame_times.append(now);self._frame_ready.notify_all()

    def _handler(self):
        display=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*_):pass  # 日志中不记录配对网址或图像。
            def reply(self,code,data,kind='application/json',extra=None):
                if kind=='application/json':data=json.dumps(data).encode()
                try:
                    self.send_response(code)
                    for k,v in {'Content-Type':kind,'Content-Length':str(len(data)),
                                'Cache-Control':'no-store','Referrer-Policy':'no-referrer',
                                'X-Content-Type-Options':'nosniff',**(extra or {})}.items():self.send_header(k,v)
                    self.end_headers();self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
                    # 响应可能已发送一部分，不能再发送第二个 HTTP 响应，也不能重复更改控制状态。
                    self.close_connection=True
            def authorized(self, *, page=False):
                try:local=ipaddress.ip_address(self.client_address[0]).is_private
                except ValueError:local=False
                value=(parse_qs(urlsplit(self.path).query).get('key',[''])[0] if page
                       else self.headers.get('X-Guest-Token',''))
                return local and secrets.compare_digest(value,display.token)
            def do_GET(self):
                route=urlsplit(self.path).path
                if route in ('/operator','/operator-state') and not ipaddress.ip_address(self.client_address[0]).is_loopback:
                    return self.reply(403,{'error':'Operator view is local to this computer'})
                if route=='/operator':
                    if not self.authorized(page=True):return self.reply(403,{'error':'Paired operator link required'})
                    return self.reply(200,Path(__file__).with_name('operator.html').read_bytes(),'text/html; charset=utf-8')
                if route=='/':
                    if not self.authorized(page=True):return self.reply(403,{'error':'Open the paired iPad link from the Mac.'})
                    page='voice_check.html' if parse_qs(urlsplit(self.path).query).get('voicecheck')==['1'] else 'guest.html'
                    html=Path(__file__).with_name(page).read_bytes()
                    return self.reply(200,html,'text/html; charset=utf-8')
                if not self.authorized():return self.reply(403,{'error':'Pairing required'})
                if route=='/cue-audio':
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if set(query)!={'cue'} or len(query['cue'])!=1:
                        return self.reply(400,{'error':'One direction audio cue required'})
                    from .cue_audio import CUE_FILES,read_cue
                    cue=query['cue'][0]
                    if cue not in CUE_FILES:return self.reply(404,{'error':'Unknown direction audio cue'})
                    try:data=read_cue(cue)
                    except (OSError,ValueError,TypeError,AttributeError):
                        return self.reply(503,{'error':'Direction audio unavailable'})
                    return self.reply(200,data,'audio/wav')
                if route=='/frame.jpg':
                    try:after=int(parse_qs(urlsplit(self.path).query).get('after',['-1'])[0])
                    except ValueError:return self.reply(400,{'error':'Invalid frame sequence'})
                    with display._frame_ready:
                        if after==display._sequence:
                            display._frame_ready.wait_for(lambda:display._sequence>after or display._closed,timeout=1.)
                        data=display._jpeg;sequence=display._sequence
                        feedback=(display._current_feedback() if display._sim_feedback or display._physical_feedback_enabled else display._frame_feedback)
                    if after==sequence:return self.reply(204,b'','application/octet-stream')
                    return (self.reply(200,data,'image/jpeg',{
                                'X-Frame-Sequence':str(sequence),
                                'X-Frame-Feedback':json.dumps(feedback,separators=(',',':'))})
                            if data else self.reply(503,{'error':'Camera not ready'}))
                if route in ('/state','/operator-state'):
                    with display._lock:
                        display._expire()
                        if (route=='/state' and display._active and display._control_protocol!=1
                                and self.headers.get('X-Guest-Client')==display._owner):
                            # 记录新心跳前先检查开场控制间隔是否已过期，迟到的轮询不能恢复已过期的动作租约。
                            if display._physical_feedback_enabled and display._physical_backend_mode=='intro':
                                display._physical_backend_available_locked(time.monotonic())
                            display._seen=time.monotonic()
                        state={**display._state,'feedback':display._current_feedback(),
                               'system_run_id':display._system_run_id,
                               'control':{'instance':display._control_instance,'epoch':display._control_epoch},
                               'photo_ready':display._photo_ready_locked(),
                               'photo_latched':display._photo_ready_locked(),
                               'dialogue':dict(display._dialogue),'active':display._active,
                               'voice_backend':'local-whisper' if display.speech.ready else 'local-model-missing',
                               'camera_source':'ipad' if display.remote_capture else 'webcam',
                               'camera_diagnostics':display.remote_capture.diagnostics() if display.remote_capture else {},
                               'camera_connected':display.remote_capture.connected() if display.remote_capture else True,
                               'camera_fresh':display._camera_stream_fresh(),
                               'control_fresh':display._control_stream_fresh(time.monotonic()),
                               'communication':display._communication_locked(time.monotonic()),
                               'stale':not display._updated or time.monotonic()-display._updated>1. or bool(display.remote_capture and not display.remote_capture.fresh()),
                               'owner':display._owner==self.headers.get('X-Guest-Client'),
                               'frame_sequence':display._sequence,
                               'server_fps':round((len(display._frame_times)-1)/
                                   max(.001,display._frame_times[-1]-display._frame_times[0]),1)
                                   if len(display._frame_times)>1 else 0.,
                               'frame_bytes':len(display._jpeg) if display._jpeg else 0}
                        if display._physical_feedback_enabled:
                            state['motion']=display._current_motion()
                            status=display._runtime_status
                            state['runtime']=(dict(status) if status and status['epoch']==display._control_epoch else
                                {'status':'starting' if display._active else 'waiting_start','reason':'none'})
                            photo=display._physical_photo;now=time.monotonic()
                            confirmed_photo=bool(display._photo_ready_locked() and photo
                                and photo['epoch']==display._control_epoch and photo['ready_at'] is not None
                                and 0<=now-photo['ready_at']
                                and display._physical_backend_available_locked(now))
                            if ((state['motion'].get('active') is True and state['motion'].get('fresh') is True
                                    or confirmed_photo) and state['runtime']['status']!='fault'):
                                # 状态 POST 偶尔失败时仍应显示已经核实的实测动作和已锁定的原生停止确认。这些只影响显示，不改变控制条件。
                                state['runtime']={'epoch':display._control_epoch,'status':'running','reason':'none'}
                            # 旧后端上报故障前，会话可能已经停止并切换轮次。保留尚未解除的实际阻塞状态供查看，但不接纳旧报告或修改当前控制。
                            if (display._reference_gate.blocked and not display._active
                                    and state['runtime']['status']!='fault'):
                                state['runtime']={'epoch':display._control_epoch,
                                                  'status':'fault','reason':'reference'}
                            if state['communication']['status'] in ('waiting','ready'):
                                if state['runtime']['status']!='fault':
                                    state['runtime']={'epoch':display._control_epoch,'status':'holding','reason':'communication'}
                                state['feedback']={**edge_feedback_for(None,0),'gesture_id':0}
                                state['motion']={'active':False,'fresh':False}
                        if route=='/operator-state':
                            state['last_stop']=dict(display._last_stop) if display._last_stop is not None else None
                            state['operator']=dict(display._operator)
                            state['control']={'instance':display._control_instance,'epoch':display._control_epoch}
                            state['control_action']=display._control_action if not display._closed else 'stop'
                            state['control_started_ns']=display._control_started_ns
                            state['reference_gate']={'blocked':display._reference_gate.blocked,
                                                     'acknowledged':display._reference_gate.acknowledged,
                                                     'session':display._reference_gate.session,
                                                     'reference_check':display._reference_check_locked(time.monotonic())}
                            if display._physical_feedback_enabled:
                                photo=display._physical_photo
                                state['physical']={'backend_available':display._physical_backend_available_locked(time.monotonic()),
                                                   'mode':display._physical_backend_mode,
                                                   'revocation':dict(display._physical_revocation) if display._physical_revocation else None,
                                                   'photo_source_ns':photo['source_ns'] if photo and photo['epoch']==display._control_epoch else None}
                    return self.reply(200,state)
                self.reply(404,{'error':'Not found'})
            def do_POST(self):
                route=urlsplit(self.path).path
                camera_frame=route=='/camera/frame' and display.remote_capture is not None
                photo=route=='/photo' and display.remote_capture is not None
                audio=route=='/transcribe'
                if (not self.authorized() or self.headers.get('Origin')!='http://'+self.headers.get('Host','')
                        or self.headers.get_content_type()!=('image/jpeg' if camera_frame or photo else 'application/octet-stream' if audio else 'application/json')):
                    # 关闭连接前读完被拒绝的小型 POST 请求体，使 Windows 客户端能收到 403，而不会因残留数据遭遇连接重置。
                    self.close_connection=True
                    self.connection.settimeout(.25)
                    try:
                        rejected_length=int(self.headers.get('Content-Length','0'))
                        if 0<rejected_length<=2048:self.rfile.read(rejected_length)
                    except (ValueError,OSError):pass
                    return self.reply(403,{'error':'Paired same-origin request required'},extra={'Connection':'close'})
                body=None
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if self.headers.get('Transfer-Encoding') or not 0<length<=(524288 if camera_frame else 4*1024*1024 if photo else MAX_AUDIO_BYTES if audio else 2048):
                        self.close_connection=True;raise ValueError()
                    self.connection.settimeout(3)
                    raw=self.rfile.read(length)
                    if len(raw)!=length:raise ValueError()
                    client=self.headers.get('X-Guest-Client','')
                    if not client or len(client)>128:raise ValueError()
                    if photo:
                        with display._lock:
                            display._expire()
                            if display._owner!=client or not display._photo_ready_locked():return self.reply(409,{'error':'Photo not ready'})
                            epoch=display._control_epoch
                        try:
                            from .photo_share import PhotoShare
                            with display._photo_lock:
                                with display._lock:
                                    valid=display._owner==client and display._control_epoch==epoch and display._photo_ready_locked()
                                if not valid:return self.reply(409,{'error':'Photo session ended'})
                                if display._photo_share is None:
                                    display._photo_share=PhotoShare(host=display._photo_host).__enter__()
                                item=display._photo_share.publish(raw)
                        except (ValueError,OSError,ImportError):return self.reply(503,{'error':'Photo sharing unavailable'})
                        with display._lock:
                            display._expire()
                            valid=display._owner==client and display._control_epoch==epoch and display._photo_ready_locked()
                        if not valid:
                            token=item['_url'].rsplit('/',1)[-1]
                            with display._photo_lock:display._photo_share.discard(token)
                            return self.reply(409,{'error':'Photo session ended'})
                        item.pop('_url',None)
                        return self.reply(200,item)
                    if audio:
                        with display._lock:
                            display._expire()
                            if not display._active or display._owner!=client:return self.reply(409,{'error':'Start this display first'})
                        try:text=display.speech.transcribe(raw,self.headers.get('X-Speech-Language','en'))
                        except BlockingIOError:return self.reply(429,{'error':'Speech recognition busy'})
                        except RuntimeError:return self.reply(503,{'error':'Prepare the local voice model on the laptop'})
                        except (ValueError,ImportError):return self.reply(400,{'error':'Audio invalid or local voice dependencies missing'})
                        except Exception:return self.reply(503,{'error':'Local transcription failed'})
                        with display._lock:
                            display._expire()
                            if not display._active or display._owner!=client:return self.reply(409,{'error':'Interaction ended'})
                        return self.reply(200,{'text':text})
                    if camera_frame:
                        display.remote_capture.accept(client,self.headers.get('X-Camera-Session'),self.headers.get('X-Camera-Ticket'),raw)
                        return self.reply(200,{'ok':True})
                    body=json.loads(raw)
                    if route=='/control-heartbeat':
                        try:
                            with display._lock:
                                result=display._control_heartbeat_locked(body,client,time.monotonic())
                        except ValueError:
                            return self.reply(409,{'ok':False,'error':'Control ended; use Start for a new encounter'})
                        return self.reply(200,{'ok':True,**result})
                    if route=='/operator-action':
                        if not ipaddress.ip_address(self.client_address[0]).is_loopback or not display._physical_feedback_enabled:
                            return self.reply(403,{'error':'Physical operator control is local only'})
                        if not isinstance(body,dict) or body.get('action') not in ('start','restart','stop'):
                            raise ValueError('Invalid operator action')
                        with display._lock:
                            display._expire()
                            if (body.get('instance')!=display._control_instance or type(body.get('epoch')) is not int
                                    or body['epoch']!=display._control_epoch):
                                return self.reply(409,{'error':'Control changed; action not applied'})
                            action=body['action']
                            if action!='stop':
                                return self.reply(409,{'error':'Use Start on the paired iPad for live control'})
                            owner=(display._owner if action=='stop' else
                                   getattr(display.remote_capture,'control_owner',lambda:None)())
                            if action!='stop' and (not owner or not display._camera_stream_fresh()):
                                return self.reply(409,{'error':'A fresh paired iPad camera is required'})
                            if action=='stop':display._record_control_stop_locked(body,client,3)
                            display._set_action_locked(action,owner or client)
                        display._clear_photo()
                        return self.reply(200,{'ok':True,'active':action!='stop'})
                    if route=='/runtime-status':
                        if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                            return self.reply(403,{'error':'Runtime status is local to this computer'})
                        with display._lock:
                            if (not display._physical_feedback_enabled or not isinstance(body,dict) or
                                    body.get('instance')!=display._control_instance or type(body.get('epoch')) is not int or
                                    body['epoch']!=display._control_epoch or
                                    body.get('status') not in ('waiting_start','starting','running','holding','stopped','fault') or
                                    body.get('reason') not in ('none','camera','reference','control','speed','runtime','communication')):
                                raise ValueError('Invalid runtime status')
                            display._runtime_status={k:body[k] for k in ('epoch','status','reason')}
                        return self.reply(200,{'ok':True})
                    if route=='/physical-feedback':
                        if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                            return self.reply(403,{'error':'Physical feedback is local to this computer'})
                        with display._lock:result=display._physical_feedback_update(body)
                        return self.reply(200,{'ok':True,**(result or {})})
                    if route=='/reference-gate':
                        if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                            return self.reply(403,{'error':'Reference gate is local to this computer'})
                        with display._lock:
                            gate=display._reference_gate;now=time.monotonic();op=body['op']
                            restarting=op in ('restart_begin','restart_sample','restart_release')
                            if display._physical_feedback_enabled:
                                display._expire()
                                credential=(display._physical_credentials_locked(now) if display._recovery_enabled_locked()
                                            else display._physical_backend_available_locked(now))
                                if (not display._active or display._closed or
                                        body.get('instance')!=display._control_instance or
                                        type(body.get('epoch')) is not int or body['epoch']!=display._control_epoch or
                                        display._physical_backend_mode!=('intro' if restarting else 'guidance') or
                                        not credential or
                                        not isinstance(body.get('lease'),str) or
                                        not secrets.compare_digest(body['lease'],display._physical_lease)):
                                    raise ValueError('Physical reference operation belongs to an inactive encounter')
                            if restarting:
                                if not display._physical_backend_available_locked(now):
                                    raise ValueError('Communication hold has not resumed')
                                result=display._restart_reference_locked(body,now)
                                return self.reply(200,result)
                            if op in ('begin','release'):
                                cached=display._transaction_locked(op,body)
                                if cached is not None:return self.reply(200,cached)
                            if (op in ('begin','release') and display._recovery_enabled_locked()
                                    and not display._physical_backend_available_locked(now)):
                                raise ValueError('Communication hold has not resumed')
                            if op=='begin':
                                session=gate.begin(body['reference'],now)
                                result={'session':session};display._transaction_locked(op,body,result)
                                return self.reply(200,result)
                            if op=='sample':
                                gate.sample(body['session'],body['pose'],body['speed'],body['sampled_at'],now)
                                return self.reply(200,{'ok':True,'sampled_at':body['sampled_at'],
                                                      'return_verified':gate.return_verified(body['session'],now)})
                            if op=='release':
                                gate.release(body['session'],now)
                                result={'ok':True,'release_ns':gate.release_ns};display._transaction_locked(op,body,result)
                                return self.reply(200,result)
                            raise ValueError('Invalid operation')
                    if route.startswith('/camera/'):
                        camera=display.remote_capture
                        if camera is None:return self.reply(404,{'error':'iPad source is not enabled'})
                        if route=='/camera/start':
                            # 会话修改和控制取消必须原子完成。先锁显示，再锁采集；采集方法内部不会反向调用显示层。
                            with display._lock:
                                preserve=display._camera_renewal_preserves_control_locked(camera,client)
                                session=camera.begin(client)
                                if not preserve:display._pause_camera_locked()
                            return self.reply(200,{'session':session})
                        if route=='/camera/ticket':return self.reply(200,{'ticket':camera.ticket(client,body['session'],body.get('diagnostics'))})
                        if route=='/camera/stop':
                            with display._lock:
                                camera.end(client,body['session'])
                                display._record_control_stop_locked(body,client,2)
                                display._pause_camera_locked()
                            return self.reply(200,{'ok':True})
                        return self.reply(404,{'error':'Unknown camera route'})
                    if route=='/message':
                        text=body['text']
                        if not isinstance(text,str) or not 1<=len(text.strip())<=300:raise ValueError()
                        if any(ord(c)<32 and c not in '\n\t' for c in text):raise ValueError()
                        with display._lock:
                            display._expire()
                            if not display._active:return self.reply(409,{'error':'Start an interaction first'})
                            display._messages.append(text.strip())
                        return self.reply(200,{'ok':True})
                    action=body['action']
                    if action not in ('start','restart','stop'):raise ValueError()
                    protocol=body.get('control_protocol',0)
                    if protocol not in (0,1) or type(protocol) is not int:raise ValueError()
                    if action!='stop' and 'recovery_window_s' in body:
                        window=body['recovery_window_s']
                        if type(window) not in (int,float) or window!=3.0:raise ValueError('Invalid Start recovery negotiation')
                    if display._physical_feedback_enabled and action!='stop' and protocol!=1:
                        return self.reply(409,{'error':'Refresh the iPad page before Start'})
                    if action!='stop' and display.remote_capture and not display.remote_capture.fresh():
                        return self.reply(409,{'error':'Open the iPad camera first'})
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    if route in ('/physical-feedback','/reference-gate'):
                        with display._lock:
                            diagnostic=dict(display._physical_revocation) if display._physical_revocation else None
                        rejection={'error':'Request expired or invalid',
                            'reason':diagnostic['reason'] if diagnostic else 'request_invalid',
                            'physical_revocation':diagnostic}
                        if route=='/reference-gate':
                            message=exc.args[0] if type(exc) is ValueError and len(exc.args)==1 else None
                            reference_reason=_REFERENCE_FAILURE_REASONS.get(message) if isinstance(message,str) else None
                            if reference_reason is not None:rejection['reference_reason']=reference_reason
                            if reference_reason=='return_unverified':
                                with display._lock:rejection['reference_check']=display._reference_check_locked(time.monotonic())
                            op=body.get('op') if isinstance(body,dict) else None
                            if isinstance(op,str) and op in ('begin','sample','release','restart_begin','restart_sample','restart_release'):
                                rejection['reference_op']=op
                        return self.reply(400,rejection)
                    return self.reply(400,{'error':'Request expired or invalid'})
                if route!='/action':return self.reply(404,{'error':'Not found'})
                with display._lock:
                    if action=='stop' and display._stop_origin(body)>=5:
                        expected=body.get('control')
                        if (not isinstance(expected,dict) or expected.get('instance')!=display._control_instance
                                or type(expected.get('epoch')) is not int or expected['epoch']!=display._control_epoch
                                or display._owner!=client):
                            return self.reply(409,{'error':'Control changed; automatic Stop was not applied'})
                    if action=='stop' and 'system_run_id' in body:
                        expected=body.get('control')
                        if (not isinstance(body['system_run_id'],str) or display._closed
                                or display._system_run_id is None
                                or not secrets.compare_digest(body['system_run_id'],display._system_run_id)
                                or not isinstance(expected,dict)
                                or expected.get('instance')!=display._control_instance):
                            return self.reply(409,{'error':'System ownership changed; Stop was not applied'})
                    if action!='stop' and protocol==1:
                        expected=body.get('control')
                        if (not isinstance(expected,dict) or expected.get('instance')!=display._control_instance
                                or type(expected.get('epoch')) is not int or expected['epoch']!=display._control_epoch):
                            return self.reply(409,{'error':'Control changed; Start was not applied'})
                    if action=='stop':display._record_control_stop_locked(body,client,1)
                    if (action=='start' and protocol==1 and display._physical_feedback_enabled
                            and display._active):
                        if display._owner!=client:
                            return self.reply(409,{'error':'Another iPad owns this interaction'})
                        # 活跃会话内重复开始不改变状态。新目标与停止仍各自执行明确的操作。
                        return self.reply(200,{'ok':True,'active':True,'already_active':True,
                            'control_sequence':display._control_heartbeat_sequence,
                            'control':{'instance':display._control_instance,'epoch':display._control_epoch},
                            'communication':display._communication_locked(time.monotonic())})
                    if display._physical_feedback_enabled:
                        print('CONTROL_ACTION_ACCEPTED '+json.dumps({'action':action,
                            'previous_active':display._active,'previous_epoch':display._control_epoch,
                            'epoch':display._control_epoch+1},sort_keys=True),file=sys.stderr,flush=True)
                    display._set_action_locked(action,client,protocol,recovery_window_s=body.get('recovery_window_s',0.))
                    control={'instance':display._control_instance,'epoch':display._control_epoch}
                    communication=display._communication_locked(time.monotonic())
                display._clear_photo()
                self.reply(200,{'ok':True,'active':action!='stop','control':control,'communication':communication})
        return Handler
