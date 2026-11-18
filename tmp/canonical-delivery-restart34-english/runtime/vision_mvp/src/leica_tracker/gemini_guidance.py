"""异步规划构图，在本地根据新采集的相机帧控制提示动作。"""
from concurrent.futures import Future
from dataclasses import replace
import threading
import math
import json
import sys

import cv2

from .continuous import ControllerConfig
from .gemini_framing import GeminiFramingClient,FramingError,DEFAULT_MODEL
from .person_guidance import GuidanceDecision,GuidanceCue,PersonGuidanceController
from .portrait_quality import (measured_context,fit_plan,quality_tolerances,
                               projected_fits,DepthStability,PortraitGeometryError)


class VisionWorker:
    supports_portrait_context=True
    def __init__(self,client):self.client=client

    def submit(self,jpeg,point,context=None):
        future=Future()
        def run():
            try:
                options={'context':context} if context is not None and getattr(self.client,'supports_portrait_context',False) is True else {}
                future.set_result(self.client.analyze(jpeg,point,**options))
            except Exception as error:
                future.set_exception(error if isinstance(error,FramingError)
                                     else FramingError('AI_API_ERROR'))
        threading.Thread(target=run,daemon=True,name='ai-framing').start()
        return future


def _point(observation):
    return (float(observation.center_x),float(observation.head_y),float(observation.shoulder_width))


def _moved(a,b,xy=.15,scale=.35):
    return abs(a[0]-b[0])>xy or abs(a[1]-b[1])>xy or abs(a[2]-b[2])>scale*max(b[2],.03)


class GeminiGuidanceController:
    def __init__(self,config=None, *, worker=None,client=None,max_requests=20,
                 stability_ms=800,entry_ack_ms=800,interaction_mode='gemini-guide',single_plan=False,
                 fixed_plan_tolerance_scale=1.,min_subject_width=.06,portrait_quality=False,**kwargs):
        if not math.isfinite(fixed_plan_tolerance_scale) or fixed_plan_tolerance_scale < 1.:
            raise ValueError('fixed_plan_tolerance_scale must be finite and at least one')
        if not math.isfinite(min_subject_width) or not .03<=min_subject_width<=1.:
            raise ValueError('min_subject_width must be between 0.03 and 1')
        self.interaction_mode=interaction_mode
        self.single_plan=single_plan
        self.portrait_quality=portrait_quality
        self.fixed_plan_tolerance_scale=fixed_plan_tolerance_scale
        self.min_subject_width=min_subject_width
        self.config=config or ControllerConfig()
        self.worker=worker or VisionWorker(client or GeminiFramingClient(**({'portrait_quality':True} if portrait_quality else {})))
        self._kwargs=kwargs;self._max_requests=max_requests;self._requests=0
        self._stability_ms=stability_ms;self._last_request_ms=-5000
        self._entry_ack_ms=entry_ack_ms
        self._pending=None;self._scene_generation=0;self._closed=False
        self.reset()

    def reset(self):
        self._quality_depth=DepthStability();self._quality_refits=0;self._quality_context=None
        self._plan_attempted=False
        self._scene_generation+=1;self._plan=None;self._inner=None
        self._complete=False;self._last_timestamp=None;self._initial=None
        self._last_point=None;self._next_request=0;self._error=None
        self._steady_since=None;self._steady_anchor=None
        self._lost_since=None;self._loss_invalidated=False
        self._entry_seen=False;self._entry_waiting=False;self._entry_emit_pending=False
        self._entry_at=0;self._first_capture=True;self._image_aspect=16/9
        self._scale_changed_since=None
        self._scene_template=None;self._scene_reference_width=None
        self._scale_anchor=None;self._scale_candidate=None
        self._frame_aspect=None
        self._perception_suspended=False

    def close(self):
        self.reset();self._closed=True

    def restart_encounter(self):
        """访客重试时保留场景及剩余调用额度，清除上一次交互状态。"""
        if self.single_plan:
            self.reset()
            return
        self._invalidate();self._initial=None
        self._entry_seen=False;self._entry_waiting=False;self._entry_emit_pending=False
        self._lost_since=None;self._loss_invalidated=False
        self._steady_since=None;self._steady_anchor=None

    def _invalidate(self):
        # 跟踪状态属于本次交互；固定背景请求属于场景。短时跟丢或缺帧时保留请求，重置或图像格式变化时才使其失效。
        self._plan=None;self._inner=None;self._complete=False
        self._scale_changed_since=None
        self._scale_anchor=None;self._scale_candidate=None
        self._quality_depth=DepthStability();self._quality_refits=0

    def suspend_perception(self,timestamp_ms):
        """参考视角暂不可用时保留交互和场景，不能据此认定无人。可缓存此时完成的请求；旧位置和模型回复都不能单独触发动作。"""
        if type(timestamp_ms) not in (int,float) or not math.isfinite(timestamp_ms) or timestamp_ms<0:
            raise ValueError('Valid perception suspension timestamp required')
        if self._closed:return self._hold('AI_STOPPED','AI_STOPPED')
        self._perception_suspended=True
        self._receive_scene_result(timestamp_ms)
        return self._hold('PHYSICAL_REFERENCE_HOLD','AWAITING_MEASURED_REFERENCE')

    def _resume_perception(self):
        if not self._perception_suspended:return
        self._perception_suspended=False
        # 保留交互和场景身份。返回参考位后，根据最新实际观测重新初始化测量滤波与停留计时。
        self._last_timestamp=None;self._last_point=None
        self._lost_since=None;self._loss_invalidated=False
        self._steady_since=None;self._steady_anchor=None
        self._scale_changed_since=None;self._scale_anchor=None;self._scale_candidate=None
        self._quality_depth=DepthStability()
        if self._inner is not None and not self._complete:self._inner.reset()
        if self._entry_seen:
            self._entry_waiting=False;self._entry_emit_pending=False

    def _receive_scene_result(self,timestamp_ms):
        # 即使画面中无人也接收回复，但只缓存已验证的图像几何数据。所有动作仍需通过观测与进入条件检查。
        if self._pending is None or not self._pending[0].done():return
        future,generation,sent_at,snapshot=self._pending
        self._pending=None
        if generation!=self._scene_generation:return
        try:
            candidate=future.result()
            # 单次构图模式在后台分析一张参考图。此处仅限制等待模型回复的时间，每次动作仍由实时观测和实体参考位检查决定。
            if timestamp_ms-sent_at>(45000 if self.single_plan else 15000):
                raise FramingError('AI_STALE_RESULT')
            self._scene_template=candidate;self._scene_reference_width=snapshot[2]
            self._error=None
        except FramingError as error:
            if self.interaction_mode=='claude-guide' and error.code=='AI_UNCERTAIN':
                visible=error.subject_visible
                confidence=error.confidence
                category=('subject_not_visible' if visible is False else
                          'low_confidence' if visible is True and confidence is not None and confidence<.65
                          else 'invalid_visibility_or_confidence')
                print('AI_UNCERTAIN_DIAGNOSTIC '+json.dumps({
                    'reason':category,'subject_visible':visible,
                    'confidence':round(confidence,4) if confidence is not None else None,
                },separators=(',',':')),file=sys.stderr,flush=True)
            self._error=error.code;self._next_request=timestamp_ms+15000

    def _activate_template(self,point):
        self._plan=(fit_plan(self._scene_template,point,self._quality_context,
                            positional_tolerance_scale=self.fixed_plan_tolerance_scale) if self.portrait_quality else
                    self._scene_template if self.single_plan else self._scene_template.for_person(point))
        tolerances=self._fixed_tolerances() if self.single_plan or self.portrait_quality else self._plan.tolerances
        cfg=replace(self.config,target_x=self._plan.target[0],
                    target_head_y=self._plan.target[1],target_shoulder_width=self._plan.target[2],
                    horizontal_deadband=tolerances[0],
                    vertical_deadband=tolerances[1],scale_deadband=tolerances[2])
        self._inner=PersonGuidanceController(config=cfg,**{
            **self._kwargs,'natural_head_height':not (self.single_plan or self.portrait_quality),
            'repeat_cues':True,**({'depth_first':True} if self.portrait_quality else {})})
        self._scale_anchor=point[2];self._error=None

    def _fixed_tolerances(self):
        """只缩放固定方案的到位判定区域，保持在图像范围内。"""
        if self.portrait_quality:
            return quality_tolerances(self._plan,image_aspect=self._image_aspect,
                                      positional_tolerance_scale=self.fixed_plan_tolerance_scale)
        x,y,_=self._plan.target
        a,b,c=self._plan.tolerances
        scale=self.fixed_plan_tolerance_scale
        return (min(a*scale,x-.005,.995-x),min(b*scale,y-.005,.995-y),c*scale)

    def _hold(self,phase,reason,point=None):
        # 此处姿态仅用于预览，不参与动作控制；尚未规划的目标不显示。
        cue=GuidanceCue.HOLD
        benign=('AI_ENTRY_ACK','AI_THINKING','AI_WAITING_FOR_STILLNESS','AI_COOLDOWN','AI_REPLAN')
        status=phase
        if phase=='TRACKING_LOST':
            status=(self._error or ('AI_SCENE_CACHED' if self._scene_template is not None
                                   else 'AI_THINKING' if self._pending is not None else phase))
        if (point is not None and self._entry_waiting and phase in benign
                and reason not in ('SOURCE_GAP','CAMERA_FORMAT_CHANGED')):
            if self._entry_emit_pending:
                cue=GuidanceCue.INVITE;self._entry_emit_pending=False;phase='AI_ENTRY_ACK'
                reason='PERSON_ACQUIRED_LOCAL_ACK'
            else:cue=GuidanceCue.WAIT
        return GuidanceDecision(phase,cue,
            self._plan.target if self._plan else point or (.5,.25,.3),
            self._initial,point,point is not None,reason,
            filtered_point=point,interaction_mode=self.interaction_mode,
            composition_frame=self._plan.frame if self._plan else None,
            target_tolerances=(self._fixed_tolerances() if self.portrait_quality and self._plan is not None else
                               self._plan.tolerances_for(point[2],image_aspect=self._image_aspect)
                               if self._plan is not None and point is not None else (.02,.02,.03)),
            composition_style=self._plan.style if self._plan else '',
            model_status=status,model_name=getattr(getattr(self.worker,'client',None),'model',DEFAULT_MODEL),
            model_requests=self._requests)

    def update(self,observation,timestamp_ms, *, mirrored=True,frame=None):
        if timestamp_ms<0:raise ValueError('timestamp_ms must be non-negative')
        self._resume_perception()
        gap=self._last_timestamp is not None and timestamp_ms-self._last_timestamp>300
        self._last_timestamp=timestamp_ms
        format_changed=False
        if frame is not None:
            aspect=frame.shape[1]/frame.shape[0]
            format_changed=(self._frame_aspect is not None and
                            abs(aspect/self._frame_aspect-1)>.01)
            if format_changed:
                if self.single_plan:
                    self._scene_generation+=1;self._scene_template=None;self._invalidate()
                    self._plan_attempted=True;self._error='AI_REPLAN_REQUIRED'
                    self._entry_waiting=False
                else:self.reset()
            self._image_aspect=aspect;self._frame_aspect=aspect
        if self._closed:return self._hold('AI_STOPPED','AI_STOPPED')
        self._receive_scene_result(timestamp_ms)
        valid=(observation is not None and
               getattr(observation,'confidence',0)>=self._kwargs.get('min_confidence',.5)
               and observation.shoulder_width>=self.min_subject_width)
        point=_point(observation) if valid else None
        valid=valid and all(math.isfinite(v) and 0<=v<=1 for v in point)
        if not valid:
            if self.portrait_quality:self._quality_depth=DepthStability()
            if self._lost_since is None:
                self._lost_since=timestamp_ms
                self._steady_since=None;self._steady_anchor=None
                if self._inner is not None and not self._complete:self._inner.reset()
            if timestamp_ms-self._lost_since>=750 and not self._loss_invalidated:
                self._invalidate();self._last_point=None;self._loss_invalidated=True
                self._entry_seen=False;self._entry_waiting=False
            reason=('SUBJECT_TOO_SMALL' if observation is not None and observation.shoulder_width<self.min_subject_width
                    else 'PERSON_LOST')
            return self._hold('TRACKING_LOST',reason)
        if format_changed:
            return self._hold('AI_REPLAN','CAMERA_FORMAT_CHANGED',point)
        if self._lost_since is not None:
            if timestamp_ms-self._lost_since>=750 and not self._loss_invalidated:
                self._invalidate()
                self._entry_seen=False;self._entry_waiting=False
            self._lost_since=None;self._loss_invalidated=False
        if self._initial is None:self._initial=point
        if not self._entry_seen:
            self._initial=point
            self._entry_seen=True;self._entry_waiting=True;self._entry_emit_pending=True
            self._entry_at=timestamp_ms;self._first_capture=True
        if gap:
            self._invalidate();self._last_point=point
            self._steady_since=None;self._steady_anchor=None
            return self._hold('AI_REPLAN','SOURCE_GAP',point)
        self._last_point=point
        if self.portrait_quality:
            self._quality_context=measured_context(observation,image_aspect=self._image_aspect)
            depth_ready=self._quality_depth.update(point[2],timestamp_ms,
                reliable=self._quality_context['depth_reliable'])
        if self._steady_anchor is None or _moved(point,self._steady_anchor,xy=.04,scale=.15):
            self._steady_anchor=point;self._steady_since=timestamp_ms

        if self._plan is None and self._scene_template is not None:
            try:self._activate_template(point)
            except PortraitGeometryError:
                return self._hold('GUIDING','VISIBLE_BODY_DOES_NOT_FIT',point)

        if self._plan is not None:
            if not (self.single_plan or self.portrait_quality):self._plan=self._scene_template.for_person(point)
            self._inner.set_target(self._plan.target)
            self._inner.set_tolerances(self._fixed_tolerances() if self.single_plan or self.portrait_quality else self._plan.tolerances_for(point[2],image_aspect=self._image_aspect))
        if self._plan is not None and not self._complete and not (self.single_plan or self.portrait_quality):
            scale_change=abs(point[2]-self._scale_anchor)>.25*self._scale_anchor
            if scale_change:
                if self._scale_changed_since is None:
                    self._scale_changed_since=timestamp_ms;self._scale_candidate=point[2]
                    self._inner.reset()
                elif abs(point[2]/self._scale_candidate-1)>.10:
                    self._scale_changed_since=timestamp_ms;self._scale_candidate=point[2]
                if timestamp_ms-self._scale_changed_since<400:
                    return self._hold('AI_REFRAMING','LOCAL_SCALE_SETTLING',point)
                self._scale_anchor=point[2];self._scale_changed_since=None
            else:self._scale_changed_since=None

        if self._plan is not None:
            if self._entry_waiting and timestamp_ms-self._entry_at<self._entry_ack_ms:
                return self._hold('AI_ENTRY_ACK','LOCAL_ACK_IN_PROGRESS',point)
            self._entry_waiting=False
            if self.portrait_quality and not self._complete:
                if not depth_ready:
                    return self._hold('GUIDING','LOCAL_DEPTH_STABILITY',point)
                depth_outside=abs(point[2]-self._plan.target[2])>self._fixed_tolerances()[2]
                if not depth_outside and not projected_fits(self._plan,point,self._quality_context,
                        positional_tolerance_scale=self.fixed_plan_tolerance_scale):
                    if self._quality_refits>=1:
                        return self._hold('GUIDING','VISIBLE_BODY_DOES_NOT_FIT',point)
                    try:self._plan=fit_plan(self._plan,point,self._quality_context,
                                          positional_tolerance_scale=self.fixed_plan_tolerance_scale)
                    except PortraitGeometryError:
                        return self._hold('GUIDING','VISIBLE_BODY_DOES_NOT_FIT',point)
                    self._quality_refits+=1;self._inner.reset()
                    self._inner.set_target(self._plan.target)
                    self._inner.set_tolerances(self._fixed_tolerances())
                    return self._hold('GUIDING','LOCAL_FRAME_FIT',point)
            d=self._inner.update(observation,timestamp_ms,mirrored=mirrored,invitation_already_sent=True)
            if (self.portrait_quality and d.cue in (GuidanceCue.MOVE_LEFT,GuidanceCue.MOVE_RIGHT,
                    GuidanceCue.MOVE_UP,GuidanceCue.MOVE_DOWN)
                    and not projected_fits(self._plan,point,self._quality_context,
                        positional_tolerance_scale=self.fixed_plan_tolerance_scale)):
                return self._hold('GUIDING','LOCAL_DEPTH_STABILITY',point)
            self._complete=d.phase=='ARRIVED'
            return replace(d,initial_point=self._initial,interaction_mode=self.interaction_mode,
                composition_frame=self._plan.frame,composition_style=self._plan.style,
                composition_note=self._plan.explanation,model_status='AI_PLAN_LOCKED',
                model_name=getattr(getattr(self.worker,'client',None),'model',DEFAULT_MODEL),
                model_requests=self._requests,model_confidence=self._plan.confidence)

        permanent=self._error in ('AI_AUTH_ERROR','AI_KEY_MISSING','AI_CONFIG_ERROR','AI_MODEL_UNAVAILABLE')
        if self._error and (permanent or timestamp_ms<self._next_request):
            return self._hold(self._error,self._error,point)
        if self._pending is not None:
            return self._hold('AI_THINKING','AI_SELECTING_COMPOSITION',point)
        if self.single_plan and self._plan_attempted:
            return self._hold(self._error or 'AI_REPLAN_REQUIRED','EXPLICIT_REPLAN_REQUIRED',point)
        if self._requests>=self._max_requests:
            return self._hold('AI_BUDGET_REACHED','AI_BUDGET_REACHED',point)
        if not self.single_plan and not self._first_capture and timestamp_ms-self._steady_since<self._stability_ms:
            return self._hold('AI_WAITING_FOR_STILLNESS','WAIT_FOR_RELIABLE_VIEW',point)
        if timestamp_ms<max(self._next_request,self._last_request_ms+5000):
            return self._hold('AI_COOLDOWN','LOCAL_REQUEST_COOLDOWN',point)
        if self._pending is None and frame is not None:
            height,width=frame.shape[:2]
            scale=min(1.,768/max(height,width))
            image=cv2.resize(frame,(max(1,int(width*scale)),max(1,int(height*scale))))
            ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
            if not ok:return self._hold('AI_IMAGE_INVALID','AI_IMAGE_INVALID',point)
            self._requests+=1
            self._plan_attempted=True
            self._first_capture=False
            self._last_request_ms=timestamp_ms;self._error=None
            options=({'context':self._quality_context} if self.portrait_quality and
                     getattr(self.worker,'supports_portrait_context',False) is True else {})
            self._pending=(self.worker.submit(jpeg.tobytes(),point,**options),
                           self._scene_generation,timestamp_ms,point)
        return self._hold('AI_THINKING','AI_SELECTING_COMPOSITION',point)
