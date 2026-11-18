"""按需在官方 URSim 上验证实际感知、模型、应用和原生动作流程。相机画面来自明确标注的公开样本回放；
按观测到的原生提示移动像素。测量只取自正式 PoseTracker，构图由真实 Claude 生成。
不接受实体端点、注入的人体坐标或伪造的模型回复；默认命令不启动运行接口。"""
import argparse
import copy
from contextlib import ExitStack,redirect_stderr
from datetime import datetime,timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import sys

from .ursim_system import ROOT

DEFAULT_FIXTURE=ROOT/'tmp/python-lab-review/.venv/Lib/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg'
CUES=frozenset(('MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER'))


def _point(value):
    if isinstance(value,dict):value=[value.get(k) for k in ('center_x','head_y','shoulder_width')]
    if (not isinstance(value,(list,tuple)) or len(value)!=3
            or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<x<1 for x in value)):
        return None
    return list(value)


class AppEventJournal:
    """观察实际 JSON 输出，不改动应用事件。"""
    def __init__(self,sink):
        self.sink=sink;self.lock=threading.Lock();self.pending='';self.last_observation=None
        self.max_requests=0;self.events=0;self.phases={};self.last_event=None;self.direction_events=0
    def write(self,text):
        self.sink.write(text)
        with self.lock:
            self.pending+=text
            while '\n' in self.pending:
                line,self.pending=self.pending.split('\n',1)
                if not line.startswith('{'):continue
                value=json.loads(line);self.events+=1;self.last_event=value
                if value.get('event')=='person_guidance' and value.get('cue') in CUES:self.direction_events+=1
                phase=value.get('phase',value.get('event','unknown'));self.phases[phase]=self.phases.get(phase,0)+1
                requests=value.get('model_requests')
                if type(requests) is int:self.max_requests=max(self.max_requests,requests)
                points=value.get('points') or {};point=_point(points.get('current'));target=_point(points.get('target'))
                stamp=value.get('source_monotonic_ns')
                if (value.get('tracking') is True and point is not None and target is not None
                        and type(stamp) is int and stamp>0
                        and (self.last_observation is None or stamp>self.last_observation['source_ns'])):
                    self.last_observation=dict(source_ns=stamp,point=point,target=target)
        return len(text)
    def flush(self):self.sink.flush()
    def observation(self):
        with self.lock:return copy.deepcopy(self.last_observation)
    def summary(self):
        with self.lock:return dict(events=self.events,max_model_requests=self.max_requests,direction_events=self.direction_events,
            phases=dict(self.phases),last_phase=(self.last_event or {}).get('phase'),
            last_model_status=(self.last_event or {}).get('model_status'))


class SceneReplay:
    """仅移动测试图像像素，不向应用提供检测结果或构图方案。"""
    def __init__(self,base):
        self.base=base;self.height,self.width=base.shape[:2];self.matrix=[1.,0.,0.]
        self.goal=list(self.matrix);self.lock=threading.Lock();self.reacting=False
        self.last_measurement=None;self.first_action=None
    def observe_native_action(self,motion):
        if (isinstance(motion,dict) and motion.get('active') is True and motion.get('fresh') is True
                and motion.get('started') is True and type(motion.get('progress')) in (int,float)
                and 0<motion['progress']<=1
                and motion.get('phase')=='moving' and motion.get('cue') in CUES
                and type(motion.get('action_id')) is int and motion['action_id']>0):
            with self.lock:
                self.reacting=True
                if self.first_action is None:self.first_action={k:motion[k] for k in ('action_id','cue')}
    def advance(self,observation):
        with self.lock:
            if not self.reacting or not isinstance(observation,dict):return
            point=_point(observation.get('point'));target=_point(observation.get('target'));stamp=observation.get('source_ns')
            if point is None or target is None or type(stamp) is not int:return
            if self.last_measurement is None or stamp>self.last_measurement:
                scale,tx,ty=self.matrix
                wanted=scale*target[2]/point[2]
                self.goal=[wanted,target[0]*self.width-wanted*(point[0]*self.width-tx)/scale,
                    target[1]*self.height-wanted*(point[1]*self.height-ty)/scale]
                self.last_measurement=stamp
            for i,bound in enumerate((.015,.012*self.width,.012*self.height)):
                error=self.goal[i]-self.matrix[i]
                self.matrix[i]+=max(-bound,min(bound,error*.35))
    def frame(self):
        import cv2
        import numpy as np
        with self.lock:scale,tx,ty=self.matrix
        return cv2.warpAffine(self.base,np.array([[scale,0,tx],[0,scale,ty]],np.float32),
            (self.width,self.height),borderMode=cv2.BORDER_CONSTANT,borderValue=(238,238,238))


def load_scene(path,*,full_scene=False):
    import cv2
    import numpy as np
    source=cv2.imread(str(path))
    if source is None:raise ValueError('Scene replay fixture is not a readable image')
    if full_scene:
        # 真实背景会影响人像构图，保留原图像素布局，避免用虚构背景重新拼接。
        scale=min(1.,960/max(source.shape[:2]))
        return (source if scale==1 else cv2.resize(source,
            (max(1,round(source.shape[1]*scale)),max(1,round(source.shape[0]*scale)))))
    height=430;width=round(source.shape[1]*height/source.shape[0])
    if not 1<=width<=750:raise ValueError('Portrait fixture aspect is unsuitable for scene replay')
    scene=np.full((640,960,3),238,np.uint8)
    scene[40:40+height,90:90+width]=cv2.resize(source,(width,height))
    return scene


def check_input(fixture,*,full_scene=False):
    import cv2
    from .app import DEFAULT_MODEL_PATH
    from .pose import PoseTracker
    scene=load_scene(fixture,full_scene=full_scene);observations=[]
    with PoseTracker(DEFAULT_MODEL_PATH) as tracker:
        for stamp in range(0,400,50):
            value=tracker.detect(cv2.cvtColor(scene,cv2.COLOR_BGR2RGB),stamp)
            if value is not None:observations.append(dict(center_x=value.center_x,head_y=value.head_y,
                shoulder_width=value.shoulder_width,confidence=value.confidence))
    if len(observations)<5:raise RuntimeError('Real PoseTracker did not consistently detect the replay subject')
    return dict(input_mode='public_image_scene_replay',network_requests=0,native_interfaces=0,
        image_shape=list(scene.shape),full_scene=full_scene,detections=len(observations),last_observation=observations[-1])


def photo_admission(actions,application):
    """无需移动的人像也应视为有效，但尚未完成的原生提示动作不能算通过。"""
    if actions:
        if not all(item.get('completed') is True for item in actions.values()):
            raise RuntimeError('Photo readiness lacks a completed actual native direction')
        return 'native_direction_completed'
    if (application.get('max_model_requests')==1 and application.get('direction_events')==0
            and application.get('phases',{}).get('ARRIVED',0)>0):
        return 'already_good_without_direction'
    raise RuntimeError('Photo readiness has neither verified direction completion nor actual no-move arrival')


def verify_photo_receipt(value,server_image):
    """将浏览器和二维码下载的验证结果与当前分享的实际 JPEG 对应起来。"""
    if not isinstance(server_image,bytes):raise ValueError('Actual photo server bytes required')
    digest=hashlib.sha256(server_image).hexdigest()
    if (not isinstance(value,dict) or any(value.get(key) is not True
            for key in ('success','qr_decoded','download_verified'))
            or value.get('capture_sha256')!=digest or value.get('download_sha256')!=digest):
        raise ValueError('Decoded QR download must match browser capture and actual shared image')
    return dict(success=True,qr_decoded=True,download_verified=True,sha256=digest,
                jpeg_bytes=len(server_image))


class _RecordingTransport:
    def __init__(self):self.lock=threading.Lock();self.requests=[];self.inflight=0
    def __call__(self,request,timeout):
        from .gemini_framing import _http_json
        started=time.monotonic();category='messages' if request.full_url.endswith('/messages') else 'other'
        row=dict(category=category,method=request.get_method(),started_at=started)
        with self.lock:self.inflight+=1
        try:
            result=_http_json(request,timeout);row['success']=True
            row['stop_reason']=result.get('stop_reason') if isinstance(result,dict) else None
            return result
        except Exception as exc:
            row.update(success=False,error_type=type(exc).__name__)
            if type(getattr(exc,'code',None)) is int:row['http_status']=exc.code
            raise
        finally:
            row['elapsed_s']=time.monotonic()-started
            with self.lock:
                self.requests.append(row);self.inflight-=1
    def snapshot(self):
        with self.lock:return copy.deepcopy(self.requests)
    def pending_request_count(self):
        with self.lock:return self.inflight
    def assert_retired(self):
        if self.pending_request_count():raise RuntimeError('Actual provider request is still pending')


class _CameraFeeder:
    def __init__(self,owner,scene,journal):
        self.owner,self.scene,self.journal=owner,scene,journal;self.closed=threading.Event()
        self.failure=None;self.frames=0;self.rejected=0;self.session=None;self.last_jpeg=None
        self.thread=threading.Thread(target=self._run,daemon=True,name='actual-app-scene-replay')
    def _run(self):
        import cv2
        connection=None
        try:
            self.session=self.owner.request('/camera/start',{})['session']
            port=int(self.owner.display.origin.rsplit(':',1)[1])
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=1.)
            while not self.closed.is_set():
                # 取得采集凭据后再渲染并编码；丢弃凭据时也要丢弃对应图像。
                ticket=self.owner.request('/camera/ticket',dict(session=self.session))['ticket']
                self.scene.advance(self.journal.observation());frame=self.scene.frame()
                ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                if not ok:raise RuntimeError('Replay frame encoding failed')
                headers={'X-Guest-Token':self.owner.display.token,'X-Guest-Client':self.owner.client,
                    'Origin':self.owner.display.origin,'Content-Type':'image/jpeg',
                    'X-Camera-Session':self.session,'X-Camera-Ticket':ticket}
                data=jpeg.tobytes();connection.request('POST','/camera/frame',body=data,headers=headers)
                response=connection.getresponse();response.read(4096)
                if response.status==200:self.frames+=1;self.last_jpeg=data
                elif response.status==400:self.rejected+=1  # 新一轮失效处理可能撤销一个正在使用的凭据。
                else:raise RuntimeError('Scene camera HTTP request rejected: '+str(response.status))
                self.closed.wait(.07)
        except Exception as exc:self.failure=type(exc).__name__+': '+str(exc)[:400]
        finally:
            if connection is not None:connection.close()
            conn=getattr(self.owner.local,'conn',None)
            if conn is not None:conn.close();self.owner.local.conn=None
    def close(self):
        self.closed.set()
        if self.thread.is_alive():self.thread.join(2.)
        if self.thread.is_alive():raise RuntimeError('Replay camera feeder did not retire')


class _Application:
    def __init__(self,capture,display,journal,client,timeout):
        from .app import AppConfig
        self.capture,self.display,self.journal,self.client=capture,display,journal,client
        self._existing_provider_threads=set(threading.enumerate())
        self.config=AppConfig(mode='guide-person',guidance_profile='claude-guide',single_plan=True,
            camera_source='ipad',source_mirrored=True,headless=True,guidance_stream=True,
            emit_every_frames=1,duration=timeout+80)
        self.failure=None;self.stats=None;self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        from .app import process_frames
        from .pose import PoseTracker
        try:
            with PoseTracker(self.config.model_path,self.config.min_confidence) as tracker:
                self.stats=process_frames(self.capture,tracker,self.config,output=self.journal,
                    guest_display=self.display,framing_client=self.client)
        except Exception as exc:self.failure=type(exc).__name__+': '+str(exc)[:500]
    def pending_provider_worker_count(self):
        # VisionWorker.close 不等待守护线程退出。
        # HTTP 完成后还要计入响应解析及 Future 结果交付所用的时间。
        return sum(thread not in self._existing_provider_threads and thread.name=='ai-framing'
                   and thread.is_alive() for thread in threading.enumerate())
    def close(self):
        self.config.duration=.000001
        if self.thread.is_alive():self.thread.join(3.)
        if self.thread.is_alive():raise RuntimeError('Real application frame loop did not retire')
        if self.pending_provider_worker_count():raise RuntimeError('Actual provider worker is still alive')


def validation_client(transport,key_provider):
    """使用记录的传输数据验证正式的人像配置。"""
    from .app import AppConfig,_framing_client
    client=_framing_client(AppConfig(guidance_profile='claude-guide',single_plan=True))
    client.transport=transport;client.key_provider=key_provider
    return client


def execute_validation(source,output,*,fixture=DEFAULT_FIXTURE,env_file=ROOT/'.env.local',
                       robot_model='ur10e',timeout=150.,wait_photo_complete=False,full_scene=False):
    from .app import AppConfig,guidance_controller_options
    from .canonical_delivery import runtime_source_identity
    from .claude_framing import load_api_key
    from .guest_display import GuestDisplay
    from .ipad_camera import IPadCapture
    from .motion_configuration import configure_motion_design
    from .physical_intro_program import IntroProgram
    from .physical_runtime import verify_stationary_after_close
    from .horizontal_follow import check_dashboard
    from .ursim_system import verify_docker,fresh_profile,serve
    from .ursim_transport import LoopbackGateway,SimRTDEChannel,SIM_HOST,dependencies
    from .ursim_validation import (_Owner,_NativeDiagnostics,_DiagnosticChannel,_DiagnosticPrimary,
        _DiagnosticStderr,_public_state,_cleanup_steps,_export_identity,cycle_proof)
    output=Path(output).resolve();fixture=Path(fixture).resolve();env_file=Path(env_file).resolve()
    if (output==ROOT or not output.is_relative_to(ROOT)
            or any(not path.is_relative_to(ROOT) for path in (fixture,env_file))):
        raise ValueError('Actual-app validation inputs and output must stay inside workspace')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(success=False,execution_backend='official_ursim_native',robot_model=robot_model,
        input_mode='real_application_public_scene_replay_real_claude',physical_acceptance=False,
        actual_application=True,actual_pose_tracker=True,actual_claude_provider=True,
        injected_observations=False,injected_provider_plan=False,wait_photo_complete=wait_photo_complete,full_scene=full_scene)
    diagnostics=_NativeDiagnostics();stack=ExitStack();owner=app=feeder=backend=capture=None
    transport=client=journal=None
    stopped=threading.Event();backend_errors=[];channels=[];channel_lock=threading.Lock()
    traces=[];ui=[];actions={};current_action=None;stationary=False;backend_started=False;photo=False
    inhibit=output.parent/'.canonical-validation-incomplete-cleanup.json'
    try:
        result['runtime_source_sha256']=runtime_source_identity(ROOT)['shared_runtime_source_sha256']
        if not math.isfinite(timeout) or not 0<timeout<=180:raise ValueError('Bounded validation timeout required')
        if inhibit.exists():raise RuntimeError('Previous native cleanup is unverified')
        load_api_key(env_file)  # 在启动控制器前报告错误，不输出配置中的密钥。
        result['fixture_sha256']=hashlib.sha256(Path(fixture).read_bytes()).hexdigest()
        scene=SceneReplay(load_scene(fixture,full_scene=full_scene));identity=verify_docker(robot_model=robot_model)
        gateway=stack.enter_context(LoopbackGateway());profile=fresh_profile(configure_motion_design(source),
            robot_model=robot_model,joint_speed=.7)
        result.update(controller_id=identity.controller_id,reference_pose=list(profile.cfg.neutral),
            reference_joints=list(profile.cfg.joints),joint_speed_command_rad_s=.7,
            nominal_gesture_duration_s=profile.cfg.gesture.duration_s)
        numeric={key:getattr(profile.cfg,key) for key in ('robot_model','joints','tcp','lower','upper',
            'speed','acceleration','base_limit','base_rate','max_tcp_linear_speed','max_tcp_angular_speed')}
        numeric.update(reference=profile.cfg.neutral,gesture=profile.cfg.gesture.as_dict())
        (output/'profile.json').write_text(json.dumps(dict(model=robot_model,installation=numeric,
            intro=profile.intro,paths=profile.paths,simulation_only=True),indent=2),encoding='utf-8')
        (output/'opening.urscript').write_text(IntroProgram(profile.intro,1).program(),encoding='utf-8')
        (output/'guidance.urscript').write_text(profile.cfg.program(1),encoding='utf-8');result.update(_export_identity(output))
        capture=IPadCapture();display=stack.enter_context(GuestDisplay(host='0.0.0.0',port=0,
            remote_capture=capture,physical_feedback=True))
        owner=_Owner(display);owner.client='app-native-validation-owner'
        transport=_RecordingTransport();client=validation_client(transport,lambda:load_api_key(env_file))
        sink=stack.enter_context((output/'events.jsonl').open('a',encoding='utf-8',buffering=1))
        journal=AppEventJournal(sink);feeder=_CameraFeeder(owner,scene,journal)
        app=_Application(capture,display,journal,client,timeout)
        result['controller_options']=guidance_controller_options(app.config,screen_coupled=True,physical_mode=True)
        factories=dependencies();raw_primary=factories['state_factory'];counter=[0,0]
        def channel(host,**kwargs):
            counter[0]+=1;raw=SimRTDEChannel(host,**kwargs)
            wrapped=_DiagnosticChannel(raw,diagnostics,('reader-' if kwargs.get('receive_only') else 'writer-')+str(counter[0]))
            if kwargs.get('receive_only'):
                with channel_lock:channels.append(wrapped)
            return wrapped
        def primary(host):
            counter[1]+=1;return _DiagnosticPrimary(raw_primary(host),diagnostics,'primary-'+str(counter[1]))
        factories.update(rtde_factory=channel,state_factory=primary)
        def measurement():
            with channel_lock:receivers=list(channels)
            values=[item.latest for item in receivers if not item.failure]
            values=[item for item in values if isinstance(item[0],dict)]
            return max(values,key=lambda item:item[1]) if values else (None,float('-inf'))
        def run_backend():
            try:
                with redirect_stderr(_DiagnosticStderr(sys.stderr,diagnostics)):
                    serve(profile,dict(port=display.port,key=display.token),output/'events.jsonl',
                        SimpleNamespace(is_alive=lambda:app.thread.is_alive() and not stopped.is_set()),identity,
                        should_stop=stopped.is_set,gateway=gateway,factories=factories)
            except Exception as exc:backend_errors.append(type(exc).__name__+': '+str(exc)[:1000])
        backend=threading.Thread(target=run_backend,daemon=True)
        app.thread.start();feeder.thread.start();backend.start();backend_started=True
        started=time.monotonic();deadline=started+timeout
        while not capture.fresh() and time.monotonic()<started+5:
            if feeder.failure:raise RuntimeError('Camera: '+feeder.failure)
            time.sleep(.01)
        if not capture.fresh():raise RuntimeError('Replay camera did not become fresh')
        owner.start();last_phase=None
        while time.monotonic()<deadline:
            if app.failure:raise RuntimeError('Application: '+app.failure)
            if feeder.failure:raise RuntimeError('Camera: '+feeder.failure)
            if backend_errors:raise RuntimeError('Backend: '+backend_errors[0])
            if owner.failure:raise RuntimeError('Owner: '+owner.failure)
            state=owner.request('/operator-state');public=_public_state(state);ui.append(public)
            actual_phase=journal.summary()['last_phase']
            if actual_phase in ('AI_AUTH_ERROR','AI_KEY_MISSING','AI_CONFIG_ERROR','AI_MODEL_UNAVAILABLE',
                    'AI_NETWORK_ERROR','AI_RATE_LIMIT','AI_SERVICE_BUSY','AI_API_ERROR','AI_INCOMPLETE',
                    'AI_INVALID_RESPONSE','AI_INFEASIBLE_TARGET','AI_UNCERTAIN','AI_STALE_RESULT',
                    'AI_REPLAN_REQUIRED','AI_IMAGE_INVALID'):
                raise RuntimeError('Actual composition provider did not produce an accepted plan: '+actual_phase)
            phase=state.get('phase')
            if phase!=last_phase:
                print('ACTUAL_APP_PHASE '+json.dumps(dict(phase=phase,model_requests=journal.summary()['max_model_requests'])),flush=True)
                last_phase=phase
            if not state['active']:raise RuntimeError('Encounter stopped before real application completion')
            if (state.get('runtime') or {}).get('status')=='fault':raise RuntimeError('Native runtime fault')
            motion=state.get('motion') or {};scene.observe_native_action(motion)
            feedback,stamp=measurement();now=time.monotonic()
            if isinstance(feedback,dict) and 0<=now-stamp<=.15:
                traces.append(dict(t=stamp,**feedback))
                identifier=motion.get('action_id')
                if (motion.get('active') and motion.get('fresh') and motion.get('cue') in CUES
                        and not actions.get(identifier,{}).get('completed')):
                    if current_action is not None and current_action!=identifier:
                        raise RuntimeError('Native action changed before measured reference release')
                    current_action=identifier
                    item=actions.setdefault(identifier,dict(cue=motion['cue'],samples=[],completed=False,
                        gate_session=state['reference_gate']['session'],lit_reference_hold_samples=0))
                    if item['cue']!=motion['cue']:raise RuntimeError('Native cue identity changed')
                    if (state['reference_gate']['blocked'] and
                            item['gate_session']!=state['reference_gate']['session']):
                        raise RuntimeError('Reference session changed inside the accepted native action')
                    light=state.get('feedback') or {}
                    if (state.get('tracking') is False and state.get('phase')=='PHYSICAL_REFERENCE_HOLD'
                            and light.get('gesture_id')==identifier and light.get('intensity',0)>.05):
                        item['lit_reference_hold_samples']+=1
                    origin=profile.cfg.neutral[:3];delta=[b-a for a,b in zip(origin,profile.paths[motion['cue']][-1])]
                    fraction=sum((p-o)*d for p,o,d in zip(feedback['tcp_pose'],origin,delta))/sum(d*d for d in delta)
                    sample=dict(t=stamp,robot_timestamp=feedback['timestamp'],fraction=fraction,
                        phase=motion['phase'],joint_speed=max(map(abs,feedback['joint_speed'])),
                        tcp_error_m=math.dist(feedback['tcp_pose'][:3],origin))
                    if not item['samples'] or all(sample[k]>item['samples'][-1][k] for k in ('t','robot_timestamp')):
                        item['samples'].append(sample)
                if current_action is not None and not state['reference_gate']['blocked']:
                    item=actions[current_action];samples=item['samples']
                    if samples and stamp>samples[-1]['t'] and feedback['timestamp']>samples[-1]['robot_timestamp']:
                        origin=profile.cfg.neutral[:3]
                        delta=[b-a for a,b in zip(origin,profile.paths[item['cue']][-1])]
                        fraction=sum((p-o)*d for p,o,d in zip(feedback['tcp_pose'],origin,delta))/sum(d*d for d in delta)
                        samples.append(dict(t=stamp,robot_timestamp=feedback['timestamp'],fraction=fraction,phase='reference_released',
                            joint_speed=max(map(abs,feedback['joint_speed'])),
                            tcp_error_m=math.dist(feedback['tcp_pose'][:3],profile.cfg.neutral[:3])))
                        item['proof']=cycle_proof(samples,require_controller_clock=True)
                        if item['lit_reference_hold_samples']<1:
                            raise RuntimeError('Actual motion has no measured-light evidence during perception suspension')
                        item['completed']=True;current_action=None
            if state.get('photo_ready') is True and state.get('photo_latched') is True:
                result['photo_admission']=photo_admission(actions,journal.summary())
                photo=True;break
            time.sleep(.025)
        if not photo:raise TimeoutError('Actual application did not reach photo readiness: '+json.dumps(journal.summary()))
        requests=transport.snapshot()
        if len(requests)!=1 or requests[0].get('category')!='messages' or requests[0].get('success') is not True:
            raise RuntimeError('Real workflow must use exactly one successful Claude request')
        if journal.summary()['max_model_requests']!=1:raise RuntimeError('Real application repeated its composition request')
        result.update(photo_ready=True,photo_latched=True,composition_requests=1,provider_requests=requests,
            provider_usage=dict(client.last_usage),application=journal.summary(),
            actions={str(k):dict(cue=v['cue'],reference_session_bound=True,
                lit_reference_hold_samples=v['lit_reference_hold_samples'],**v['proof']) for k,v in actions.items()})
        photo_path=output/'photo-source.jpg'
        if feeder.last_jpeg is None:raise RuntimeError('No real replay camera frame available for photo integration')
        photo_path.write_bytes(feeder.last_jpeg)
        (output/'browser-session.json').write_text(json.dumps(dict(origin=display.origin,token=display.token,
            client=owner.client,fixture_image_path=str(photo_path),fixture_image_is_mirrored=True,
            output=str(output/'photo-browser'))),encoding='utf-8')
        (output/'photo-ready.json').write_text(json.dumps(dict(photo_ready=True,provider_requests=1,
            actual_native_actions=len(actions),source='actual_application'),indent=2),encoding='utf-8')
        print('ACTUAL_APP_PHOTO_READY '+str(output/'photo-ready.json'),flush=True)
        if wait_photo_complete:
            receipt=output/'photo-complete.json';photo_deadline=time.monotonic()+60.
            while not receipt.exists() and time.monotonic()<photo_deadline:
                state=owner.request('/state')
                if state.get('active') is not True or state.get('photo_ready') is not True:
                    raise RuntimeError('Photo readiness ended before browser photo integration')
                if feeder.failure or app.failure or owner.failure:raise RuntimeError('Photo integration control or camera failed')
                server_meta=output/'photo-server.json'
                if not server_meta.exists():
                    with display._photo_lock:
                        share=display._photo_share
                        if share is not None:
                            with share.lock:item=share.item
                            if item is not None:
                                metadata=dict(address=share.address,port=share.port,
                                    sha256=hashlib.sha256(item[1]).hexdigest(),bytes=len(item[1]))
                                temporary=output/'photo-server.json.new'
                                with temporary.open('x',encoding='utf-8') as stream:json.dump(metadata,stream)
                                temporary.replace(server_meta)
                time.sleep(.05)
            if not receipt.exists():raise TimeoutError('Browser photo completion receipt was not supplied')
            value=json.loads(receipt.read_text(encoding='utf-8'))
            with display._photo_lock:
                share=display._photo_share
                if share is None:raise RuntimeError('Browser did not publish a photo to the real photo server')
                with share.lock:item=share.item
                if item is None:raise RuntimeError('Published browser photo expired before verification')
                server_image=item[1]
            proof=verify_photo_receipt(value,server_image)
            import cv2
            import numpy as np
            decoded=cv2.imdecode(np.frombuffer(server_image,np.uint8),cv2.IMREAD_COLOR)
            if decoded is None:raise RuntimeError('Actual shared photo is not a decodable image')
            proof.update(width=decoded.shape[1],height=decoded.shape[0])
            result['actual_photo_image']=proof
            result['browser_photo_receipt']={key:value[key] for key in (
                'success','qr_decoded','download_verified','capture_sha256','download_sha256')}
        owner.stop();app.close();stopped.set();backend.join(8)
        if backend.is_alive():raise RuntimeError('Native backend did not retire')
        stationary=verify_stationary_after_close(SIM_HOST,channel_factory=SimRTDEChannel,dashboard_check=check_dashboard)
        if stationary is not True:raise RuntimeError('Independent native standstill unavailable')
        result.update(success=True,stationary_after_stop=True,elapsed_s=time.monotonic()-started)
    except BaseException as exc:
        result['failure']=type(exc).__name__+': '+str(exc)[:1600];raise
    finally:
        def close_backend():
            if backend_started:
                backend.join(8)
                if backend.is_alive():raise RuntimeError('Native backend thread is still alive')
        def prove_stationary():
            nonlocal stationary
            if backend_started and not stationary:
                stationary=verify_stationary_after_close(SIM_HOST,channel_factory=SimRTDEChannel,dashboard_check=check_dashboard)
                if stationary is not True:raise RuntimeError('Cleanup standstill unverified')
        steps=[]
        if owner is not None:steps.append(('owner Stop',owner.stop))
        if app is not None:steps.append(('actual application',app.close))
        if transport is not None:steps.append(('provider requests',transport.assert_retired))
        steps.append(('cancel backend',stopped.set))
        if feeder is not None:steps.append(('camera feeder',feeder.close))
        if capture is not None:steps.append(('camera capture',capture.release))
        steps.extend([('backend',close_backend),('independent standstill',prove_stationary),('servers and gateway',stack.close)])
        errors=_cleanup_steps(steps)
        result.update(cleanup_complete=not errors,cleanup_errors=errors,backend_errors=backend_errors,
            stationary_after_cleanup=stationary,backend_thread_alive=bool(backend and backend.is_alive()),
            application_thread_alive=bool(app and app.thread.is_alive()),
            camera_frames=feeder.frames if feeder else 0,camera_rejected_frames=feeder.rejected if feeder else 0,
            native_diagnostics=diagnostics.summary())
        if app is not None:result['application_failure']=app.failure
        if feeder is not None:result['camera_failure']=feeder.failure
        if transport is not None:
            result['provider_requests']=transport.snapshot()
            result['provider_pending_requests']=transport.pending_request_count()
        if app is not None:result['provider_workers_alive']=app.pending_provider_worker_count()
        if client is not None:result['provider_usage']=dict(client.last_usage)
        if journal is not None:result['application']=journal.summary()
        if errors:
            result['success']=False
            if backend_started:inhibit.write_text(json.dumps(dict(output=str(output),errors=errors),indent=2),encoding='utf-8')
        for name,value in (('telemetry.json',traces),('ui-state.json',ui),('actions.json',actions),
                           ('native-diagnostics.json',diagnostics.trace())):
            (output/name).write_text(json.dumps(value),encoding='utf-8')
        (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main(argv=None,*,root=ROOT):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--check-input',action='store_true')
    parser.add_argument('--wait-photo-complete',action='store_true')
    parser.add_argument('--full-scene',action='store_true',help='Replay the original scene pixels without an artificial blank surround')
    parser.add_argument('--robot-model',choices=('ur5e','ur10e'),default='ur10e')
    parser.add_argument('--fixture',type=Path,default=DEFAULT_FIXTURE)
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env.local')
    parser.add_argument('--source-profile',type=Path,default=ROOT/'tmp/robot-network/physical-ipad-ur10e-trial.json')
    parser.add_argument('--timeout',type=float,default=150.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);root=Path(root).resolve()
    if not args.execute and not args.check_input:
        print('Actual-app validation CLI checked offline; no camera, API, Docker or robot interface opened.');return 0
    paths=[args.fixture,args.env_file,args.source_profile]
    paths=[path.resolve() if path.is_absolute() else (root/path).resolve() for path in paths]
    if any(not path.is_relative_to(root) for path in paths):parser.error('Inputs must stay inside workspace')
    fixture,env_file,source=paths
    if args.check_input:
        if args.execute:parser.error('Use either --check-input or --execute')
        print(json.dumps(check_input(fixture,full_scene=args.full_scene),indent=2));return 0
    output=(args.output or root/'tmp'/('actual-app-validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
    if not output.is_absolute():output=root/output
    output=output.resolve()
    if not output.is_relative_to(root) or output==root:parser.error('Output must stay inside workspace')
    result=execute_validation(json.loads(source.read_text(encoding='utf-8-sig')),output,
        fixture=fixture,env_file=env_file,robot_model=args.robot_model,timeout=args.timeout,
        wait_photo_complete=args.wait_photo_complete,full_scene=args.full_scene)
    print(json.dumps(dict(success=result['success'],output=str(output),physical_acceptance=False)));return 0
