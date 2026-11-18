"""先粗略引导参与者站位，再模拟构图和拍照。"""
from collections import deque
from dataclasses import replace

from .continuous import ControllerConfig
from .person_guidance import PersonGuidanceController, GuidanceCue


class CoarsePortraitController:
    def __init__(self, config=None, **kwargs):
        self.config=replace(config or ControllerConfig(),target_x=.5,target_head_y=.30,
            target_shoulder_width=.30,horizontal_deadband=.15,vertical_deadband=.20,
            scale_deadband=.12)
        self._kwargs=kwargs
        self.reset()

    def reset(self):
        self._inner=PersonGuidanceController(self.config,**self._kwargs)
        self._initial=None
        self._last_timestamp=None
        self._clear_stage()
        self._complete=False

    def _clear_stage(self):
        self._history=deque()
        self._stage_start=None
        self._reference=None
        self._frame_sent=False
        self._completion_sent=False
        self._capture=False

    def _emit(self, base, **changes):
        defaults=dict(initial_point=self._initial or base.initial_point,
            interaction_mode='coarse-demo',capture_simulated=self._capture,
            framing_verified=False)
        defaults.update(changes)
        return replace(base,**defaults)

    def _inside(self, base):
        return all(point is not None and all(abs(v-t)<=tol for v,t,tol in
            zip(point,base.target,base.target_tolerances))
            for point in (base.current_point,base.filtered_point))

    def update(self, observation, timestamp_ms, *, mirrored=True):
        gap=self._last_timestamp is not None and timestamp_ms-self._last_timestamp>300
        self._last_timestamp=timestamp_ms
        base=self._inner.update(observation,timestamp_ms,mirrored=mirrored)
        if self._initial is None and base.initial_point is not None:
            self._initial=base.initial_point
        if self._complete:
            return self._emit(base,phase='DEMO_COMPLETE',
                cue=GuidanceCue.WAIT if base.tracking else GuidanceCue.HOLD,
                stage_progress=1.,reason='SIMULATED_WORKFLOW_COMPLETE')
        if gap or not base.tracking:
            self._clear_stage();self._inner.reset()
            return self._emit(base,phase='TRACKING_LOST',cue=GuidanceCue.HOLD,
                tracking=False,current_point=None,filtered_point=None,
                reason='SOURCE_GAP' if gap else base.reason)

        inside=self._inside(base)
        if self._stage_start is not None:
            drift=any(abs(a-b)>limit for a,b,limit in
                zip(base.current_point,self._reference,(.08,.10,.07)))
            if not inside or drift:
                self._clear_stage();self._inner.reset()
                return self._emit(base,phase='REPOSITION',cue=GuidanceCue.HOLD,
                    reason='PERSON_MOVED_DURING_SIMULATED_FRAMING')
        elif not inside:
            self._history.clear()
            if base.phase=='ARRIVED':
                self._inner.reset()
                base=self._inner.update(observation,timestamp_ms,mirrored=mirrored)
            return self._emit(base)

        if self._stage_start is None:
            self._history.append((timestamp_ms,base.current_point))
            while self._history and self._history[0][0]<timestamp_ms-650:
                self._history.popleft()
            still=(timestamp_ms-self._history[0][0]>=600 and
                all(max(p[i] for _,p in self._history)-min(p[i] for _,p in self._history)<=.03
                    for i in range(3)))
            if base.phase!='ARRIVED' or not still:
                if base.phase in ('SETTLING','ARRIVED'):
                    return self._emit(base,phase='COARSE_SETTLING',cue=GuidanceCue.WAIT,
                        arrival_progress=min(.99,base.arrival_progress),
                        reason='WAIT_FOR_STILLNESS' if not still else 'COARSE_DWELL')
                return self._emit(base)
            self._stage_start=timestamp_ms
            self._reference=base.filtered_point

        elapsed=timestamp_ms-self._stage_start
        if elapsed<500:
            return self._emit(base,phase='POSITION_READY',cue=GuidanceCue.WAIT,
                reason='PERSON_READY_NOT_FINAL_FRAMING',stage_progress=1.)
        if elapsed<2900:
            cue=GuidanceCue.WAIT if self._frame_sent else GuidanceCue.SIM_FRAME
            self._frame_sent=True
            return self._emit(base,phase='ARM_FRAMING',cue=cue,
                reason='SCRIPTED_SIMULATOR_FRAMING',stage_progress=(elapsed-500)/2400)
        self._capture=True
        if elapsed<3700:
            return self._emit(base,phase='SIMULATED_CAPTURE',cue=GuidanceCue.WAIT,
                reason='NO_PHOTOGRAPH_SAVED',stage_progress=(elapsed-2900)/800)
        if elapsed<9700:
            cue=GuidanceCue.WAIT if self._completion_sent else GuidanceCue.SIM_COMPLETE
            self._completion_sent=True
            return self._emit(base,phase='COMPLETING',cue=cue,
                reason='SIMULATED_CAPTURE_THEN_ENDING',stage_progress=(elapsed-3700)/6000)
        self._complete=True
        return self._emit(base,phase='DEMO_COMPLETE',cue=GuidanceCue.WAIT,
            reason='SIMULATED_WORKFLOW_COMPLETE',stage_progress=1.)
