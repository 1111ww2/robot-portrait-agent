import io
import json
import time

import numpy as np
import pytest

from leica_tracker import app
from leica_tracker.person_guidance import GuidanceCue, GuidanceDecision


def run_encounter(monkeypatch, actions, *, feedback=True, complete_at=None, acknowledge=True,
                  frames=5, frame_delay=0, ready_indices=None, physical=False, backend=True, stale_first=False,preview=False):
    if physical:monkeypatch.delenv('LEICA_SIM_FAST_GESTURES',raising=False)
    else:monkeypatch.setenv('LEICA_SIM_FAST_GESTURES', '1')
    if preview:monkeypatch.setenv('LEICA_PHYSICAL_PREVIEW','1')
    else:monkeypatch.delenv('LEICA_PHYSICAL_PREVIEW',raising=False)
    monkeypatch.setattr(app, '_framing_client', lambda config: None)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    timeline = []

    class Capture:
        def invalidate_pending(self): pass
        def read(self):
            time.sleep(frame_delay or .02)
            self.source_monotonic_ns = time.monotonic_ns()
            return True, frame
    capture=Capture()

    class Tracker:
        def detect(self, image, timestamp):
            timeline.append('detect')
            return None

    class Controller:
        def __init__(self, **kwargs): pass
        def restart_encounter(self): timeline.append('reset')
        def close(self): pass
        def update(self, observation, timestamp, **kwargs):
            timeline.append('model_update')
            return GuidanceDecision('LOOKING', GuidanceCue.WAIT, (.5, .3, .2), None, None, False, 'LOOKING')

    class Display:
        active = True
        physical_feedback_enabled = physical
        epoch=0
        floor=0
        index = -1
        phases = []
        def take_action(self):
            self.index += 1
            action = actions.get(self.index)
            if action == 'stop': self.active = False
            return action
        def take_control_action(self):
            action=self.take_action()
            if action is not None:
                self.epoch+=1
                if stale_first:self.floor=capture.source_monotonic_ns+1
            return {'action':action,'instance':'physical-test','epoch':self.epoch,'control_started_ns':self.floor}
        def simulator_intro_available(self): return feedback
        def simulator_intro_active(self, source): return feedback and acknowledge
        def physical_intro_active(self, source): return backend and acknowledge
        def physical_intro_ready(self, source): return backend and complete_at is not None and self.index >= complete_at
        def physical_backend_available(self): return backend
        def simulator_intro_ready(self, source):
            if ready_indices is not None: return self.index in ready_indices
            return complete_at is not None and self.index >= complete_at
        def reference_held(self, source): return False
        def publish(self, frame, decision, **kwargs): self.phases.append(decision.phase)

    monkeypatch.setattr(app, 'GeminiGuidanceController', Controller)
    display = Display()
    out = io.StringIO()
    app.process_frames(capture, Tracker(), app.AppConfig(mode='guide-person', camera_source='ipad',
        guidance_profile='claude-guide', single_plan=True, headless=True, max_frames=frames,
        guidance_stream=True, emit_every_frames=1), output=out, guest_display=display)
    return display, timeline, [json.loads(line) for line in out.getvalue().splitlines()]


def test_start_waits_for_measured_intro_completion_before_detecting(monkeypatch):
    display, timeline, events = run_encounter(monkeypatch, {0: 'start'}, complete_at=3)
    assert display.phases[:3] == ['INTRO'] * 3
    assert timeline.index('detect') > timeline.index('reset')
    assert timeline.count('detect') == 2
    assert [event['action'] for event in events[:3]] == ['start', 'heartbeat', 'heartbeat']
    assert all(event['event'] == 'simulator_intro' for event in events[:3])


def test_physical_start_waits_for_physical_backend_intro_confirmation(monkeypatch):
    display,timeline,events=run_encounter(monkeypatch,{0:'start'},physical=True,complete_at=3)
    assert display.phases[:3]==['INTRO']*3
    # 首次看到ready的框架比新的固定截止日期早。
    assert timeline.count('detect')==1
    assert [event['event'] for event in events[:3]]==['physical_intro']*3


def run_delayed_physical_frames(monkeypatch, *, stop_at=None,communication_frames=()):
    """实际源印章延迟控制为160毫秒；没有壁时钟睡眠。"""
    from types import SimpleNamespace
    clock=SimpleNamespace(now=10.)
    def advance(seconds):clock.now+=seconds
    fake_time=SimpleNamespace(monotonic=lambda:clock.now,
        monotonic_ns=lambda:round(clock.now*1e9),sleep=advance)
    monkeypatch.setattr(app,'time',fake_time)
    monkeypatch.delenv('LEICA_PHYSICAL_PREVIEW',raising=False)
    monkeypatch.delenv('LEICA_SIM_FAST_GESTURES',raising=False)
    monkeypatch.setattr(app,'_framing_client',lambda config:None)
    frame=np.zeros((24,32,3),dtype=np.uint8);timeline=[]
    class Capture:
        n=-1;read_timeout=.25;source_monotonic_ns=0
        def __init__(self):self.invalidations=[]
        def read(self):
            self.n+=1;advance(.08)
            if self.n in (1,2,4):return False,None
            self.source_monotonic_ns=round((clock.now-(.01 if self.n==0 else .16))*1e9)
            return True,frame
        def connected(self):return True
        def invalidate_pending(self):self.invalidations.append(round(clock.now*1e9))
    capture=Capture()
    class Tracker:
        def detect(self,image,timestamp):timeline.append(('detect',capture.n,capture.source_monotonic_ns));return None
    class Controller:
        def __init__(self,**kwargs):pass
        def restart_encounter(self):pass
        def close(self):pass
        def update(self,observation,timestamp,**kwargs):
            timeline.append(('model',capture.n,capture.source_monotonic_ns))
            return GuidanceDecision('LOOKING',GuidanceCue.WAIT,(.5,.3,.2),None,None,False,'LOOKING')
    class Display:
        active=True;physical_feedback_enabled=True;epoch=1
        def __init__(self):self.phases=[]
        def take_control_action(self):
            action='start' if capture.n==0 else 'stop' if capture.n==stop_at else None
            if action=='stop':self.active=False;self.epoch=2
            return {'action':action,'instance':'test','epoch':self.epoch,'control_started_ns':1}
        def control_snapshot(self):
            if not communication_frames:raise AttributeError('Legacy fixture has no control snapshot')
            held=capture.n in communication_frames
            return {'instance':'test','epoch':self.epoch,'active':self.active,'control_fresh':not held,
                'communication':{'status':'waiting' if held else 'healthy','recoverable':True,
                    'remaining_ms':1500 if held else 0}}
        def physical_intro_active(self,source):
            return self.active and capture.n not in communication_frames and not (communication_frames and capture.n==20)
        def physical_intro_ready(self,source):return self.active and capture.n>=(21 if communication_frames else 4)
        def physical_backend_available(self):return True
        def reference_held(self,source):return False
        def publish(self,frame,decision,**kwargs):self.phases.append((capture.n,decision.phase))
    monkeypatch.setattr(app,'GeminiGuidanceController',Controller)
    display=Display();output=io.StringIO()
    app.process_frames(capture,Tracker(),app.AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,duration=2.8 if communication_frames else 1.1,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=display)
    return capture,display,timeline,[json.loads(line) for line in output.getvalue().splitlines()]


def test_recoverable_intro_gap_longer_than_ack_timeout_keeps_one_intro_and_post_return_frame_cutoff(monkeypatch):
    capture,display,timeline,events=run_delayed_physical_frames(monkeypatch,communication_frames=tuple(range(2,20)))
    assert not any(phase=='INTRO_UNAVAILABLE' for _,phase in display.phases)
    assert any(phase=='COMMUNICATION_HOLD' for _,phase in display.phases)
    intro=[event for event in events if event['event']=='physical_intro']
    assert sum(event['action']=='start' for event in intro)==1
    assert len({event['intro_id'] for event in intro})==1
    assert len(capture.invalidations)==1
    assert timeline and all(source>capture.invalidations[0] for _,_,source in timeline)


def test_physical_intro_control_does_not_starve_delayed_fresh_images_after_ready(monkeypatch):
    capture,display,timeline,events=run_delayed_physical_frames(monkeypatch)
    assert len(capture.invalidations)==1
    cutoff=capture.invalidations[0]
    assert timeline and all(source>cutoff for _,_,source in timeline)
    assert timeline[0][1]>=7  # n=5/6张图像是在准备好之前或准备好时拍摄的。
    assert all(phase=='INTRO' for n,phase in display.phases if n<7)
    intro=[e for e in events if e['event']=='physical_intro']
    assert len(intro)>=6
    assert all(0<b['source_monotonic_ns']-a['source_monotonic_ns']<=300_000_000 for a,b in zip(intro,intro[1:]))
    # 第一个真正的图像仍然是其凭据时间戳，而不是后来的控制时钟。
    guidance=next(e for e in events if e['event']=='person_guidance')
    assert guidance['source_monotonic_ns']==timeline[0][2]


def test_stop_while_physical_intro_waits_for_post_ready_image_cancels_search(monkeypatch):
    capture,display,timeline,events=run_delayed_physical_frames(monkeypatch,stop_at=5)
    assert len(capture.invalidations)==1
    assert timeline==[]
    assert any(e.get('cue')=='HOLD' and e.get('control_epoch')==2 for e in events)


def run_action_after_frame_gap(monkeypatch, action, *, stop_at=None):
    """在途的图像比“开始”更新，但比“丢失保留”更旧。"""
    from types import SimpleNamespace
    clock=SimpleNamespace(now=10.)
    def advance(seconds):clock.now+=seconds
    monkeypatch.setattr(app,'time',SimpleNamespace(monotonic=lambda:clock.now,
        monotonic_ns=lambda:round(clock.now*1e9),sleep=advance))
    monkeypatch.delenv('LEICA_PHYSICAL_PREVIEW',raising=False)
    monkeypatch.setattr(app,'_framing_client',lambda config:None)
    frame=np.zeros((24,32,3),dtype=np.uint8);timeline=[]
    class Capture:
        n=-1;source_monotonic_ns=0
        def read(self):
            self.n+=1;advance(.1)
            if self.n==0:return False,None
            self.source_monotonic_ns=10_080_000_000 if self.n==1 else round((clock.now-.02)*1e9)
            return True,frame
        def connected(self):return True
        def invalidate_pending(self):pass
    capture=Capture()
    class Tracker:
        def detect(self,image,timestamp):timeline.append('detect');return None
    class Controller:
        def __init__(self,**kwargs):pass
        def restart_encounter(self):timeline.append('reset')
        def close(self):pass
        def update(self,observation,timestamp,**kwargs):
            timeline.append('model')
            return GuidanceDecision('LOOKING',GuidanceCue.WAIT,(.5,.3,.2),None,None,False,'LOOKING')
    class Display:
        active=True;physical_feedback_enabled=True;epoch=1
        def take_control_action(self):
            current=action if capture.n==0 else 'stop' if capture.n==stop_at else None
            if current=='stop':self.active=False;self.epoch=2
            return {'action':current,'instance':'gap-test','epoch':self.epoch,'control_started_ns':10_050_000_000}
        def physical_intro_active(self,source):return self.active
        def physical_intro_ready(self,source):return False
        def physical_backend_available(self):return True
        def reference_held(self,source):return False
        def publish(self,*args,**kwargs):pass
    monkeypatch.setattr(app,'GeminiGuidanceController',Controller)
    output=io.StringIO()
    app.process_frames(capture,Tracker(),app.AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,duration=.6,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=Display())
    return timeline,[json.loads(line) for line in output.getvalue().splitlines()]


@pytest.mark.parametrize('action',['start','restart'])
def test_pending_physical_action_survives_dropping_image_older_than_loss_hold(monkeypatch,action):
    timeline,events=run_action_after_frame_gap(monkeypatch,action)
    assert timeline.count('reset')==1
    if action=='start':
        opening=next(e for e in events if e['event']=='physical_intro')
        assert opening['action']=='start' and opening['control_epoch']==1
        assert opening['intro_id']>events[0]['source_monotonic_ns']


def test_stop_supersedes_pending_start_even_when_next_camera_image_is_old(monkeypatch):
    timeline,events=run_action_after_frame_gap(monkeypatch,'start',stop_at=1)
    assert 'reset' not in timeline
    assert all(e['event']!='physical_intro' for e in events)
    assert any(e.get('control_epoch')==2 and e.get('cue')=='HOLD' for e in events)


def test_physical_start_does_not_fake_intro_when_backend_absent(monkeypatch):
    display,timeline,events=run_encounter(monkeypatch,{0:'start'},physical=True,backend=False)
    assert display.phases==['INTRO']*5
    assert 'detect' not in timeline and 'model_update' not in timeline
    assert all(event['event']=='physical_intro' for event in events)


def test_physical_new_start_never_reuses_previous_epoch_intro(monkeypatch):
    _,_,events=run_encounter(monkeypatch,{0:'start',1:'start'},physical=True,backend=True)
    assert events[0]['control_epoch']==1 and events[1]['control_epoch']==2
    assert events[0]['intro_id']!=events[1]['intro_id']
    assert all(event['control_instance']=='physical-test' for event in events)


def test_physical_new_target_without_backend_keeps_reference_hold(monkeypatch):
    display,_,events=run_encounter(monkeypatch,{0:'restart'},physical=True,backend=False)
    assert all(event['cue']=='HOLD' for event in events)
    assert display.phases==['PHYSICAL_REFERENCE_HOLD']*5
    assert all(event['control_instance']=='physical-test' and event['control_epoch']==1 for event in events)


def test_physical_new_target_discards_frame_from_before_control_epoch(monkeypatch):
    display,_,events=run_encounter(monkeypatch,{0:'restart'},physical=True,backend=True,stale_first=True)
    assert events and all(event['source_monotonic_ns']>display.floor for event in events)
    assert all(event['control_epoch']==1 for event in events)


def test_physical_preview_waits_without_intro_or_model_calls(monkeypatch):
    display,timeline,events=run_encounter(monkeypatch,{0:'start'},physical=True,backend=False,preview=True)
    assert display.phases==['PHYSICAL_PREVIEW']*5
    assert 'detect' not in timeline and 'model_update' not in timeline
    assert all(event['event']=='person_guidance' and event['cue']=='HOLD' for event in events)


def test_repeated_start_during_intro_keeps_original_session(monkeypatch):
    display, timeline, events = run_encounter(monkeypatch, {0: 'start', 1: 'start'}, complete_at=4, frames=5)
    intro = events[:4]
    assert [event['action'] for event in intro] == ['start', 'heartbeat', 'heartbeat', 'heartbeat']
    assert len({event['intro_id'] for event in intro}) == 1
    assert timeline.count('reset') == 1
    assert display.phases[:4] == ['INTRO'] * 4
    assert timeline.count('detect') == 1


def test_start_after_completed_intro_opens_a_new_session(monkeypatch):
    _, timeline, events = run_encounter(monkeypatch, {0: 'start', 3: 'start'},
                                         frames=6, ready_indices={2, 5})
    starts = [event for event in events if event.get('event') == 'simulator_intro' and event['action'] == 'start']
    assert len(starts) == 2 and starts[0]['intro_id'] != starts[1]['intro_id']
    assert timeline.count('reset') == 2
    assert timeline.count('detect') == 2


def test_new_target_skips_intro(monkeypatch):
    display, timeline, events = run_encounter(monkeypatch, {0: 'restart'})
    assert 'detect' in timeline
    assert 'INTRO' not in display.phases
    assert all(event['event'] != 'simulator_intro' for event in events)


def test_start_without_simulator_shows_failure_and_never_detects(monkeypatch):
    display, timeline, events = run_encounter(monkeypatch, {0: 'start'}, feedback=False)
    assert display.phases == ['INTRO_UNAVAILABLE'] * 5
    assert 'detect' not in timeline and 'model_update' not in timeline
    assert all(event['cue'] == 'HOLD' for event in events)


def test_stop_during_intro_holds_and_prevents_detection(monkeypatch):
    display, timeline, events = run_encounter(monkeypatch, {0: 'start', 1: 'stop'})
    assert events[1]['cue'] == 'HOLD'
    assert 'detect' not in timeline and 'model_update' not in timeline


def test_new_target_cancels_intro_before_search(monkeypatch):
    display, timeline, events = run_encounter(monkeypatch, {0: 'start', 1: 'restart'})
    assert events[0]['event'] == 'simulator_intro'
    assert events[1]['cue'] == 'HOLD'
    assert display.phases[1] == 'GUEST_READY'
    assert timeline.count('detect') == 3
    assert all(event['event'] != 'simulator_intro' for event in events[1:])


def test_configured_but_disconnected_bridge_times_out_without_search(monkeypatch):
    display, timeline, events = run_encounter(monkeypatch, {0: 'start'}, acknowledge=False,
                                              frames=65, frame_delay=.035)
    assert display.phases[0] == 'INTRO'
    assert display.phases[-1] == 'INTRO_UNAVAILABLE'
    assert events[-1]['cue'] == 'HOLD'
    assert 'detect' not in timeline and 'model_update' not in timeline


def test_missing_camera_frames_do_not_interrupt_simulator_intro(monkeypatch):
    display, timeline, events = run_with_missing_frames(monkeypatch)
    # 心跳频率被限制在最小间隔：真正的Windows时钟分辨率可以合法地合并缺失的三个读取之一。
    first_guidance = next(i for i,e in enumerate(events) if e['event'] == 'person_guidance')
    intro_events = events[:first_guidance]
    assert 3 <= len(intro_events) <= 4
    assert [e['event'] for e in intro_events] == ['simulator_intro'] * len(intro_events)
    assert [e['action'] for e in intro_events] == ['start'] + ['heartbeat'] * (len(intro_events)-1)
    assert display.phases[:first_guidance] == ['INTRO'] * first_guidance
    assert timeline == ['detect', 'model_update']


def test_stop_during_missing_frame_holds_without_waiting_for_camera(monkeypatch):
    display, timeline, events = run_with_missing_frames(monkeypatch, stop_at=1)
    assert events[1]['cue'] == 'HOLD'
    assert display.phases[1] == 'GUEST_READY'
    assert timeline == []


def test_repeated_start_during_missing_frame_keeps_intro_session(monkeypatch):
    display, timeline, events = run_with_missing_frames(monkeypatch, repeated_start_at=1)
    intro = [event for event in events if event['event'] == 'simulator_intro']
    assert intro[0]['action'] == 'start'
    assert all(event['action'] == 'heartbeat' for event in intro[1:])
    assert len({event['intro_id'] for event in intro}) == 1
    assert all(event['event'] != 'person_guidance' for event in events[:len(intro)])
    assert timeline == ['detect', 'model_update']


def test_missing_frames_and_disconnected_bridge_timeout_to_hold(monkeypatch):
    display, timeline, events = run_with_missing_frames(monkeypatch, dropped=38, acknowledge=False)
    assert 'INTRO_UNAVAILABLE' in display.phases
    assert events[-1]['cue'] == 'HOLD'
    assert timeline == []


def run_with_missing_frames(monkeypatch, stop_at=None, dropped=3, acknowledge=True, repeated_start_at=None):
    monkeypatch.setenv('LEICA_SIM_FAST_GESTURES', '1')
    monkeypatch.setattr(app, '_framing_client', lambda config: None)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    timeline = []
    class Capture:
        n = -1
        def read(self):
            self.n += 1
            if 1 <= self.n <= dropped:
                time.sleep(.06)
                return False, None
            self.source_monotonic_ns = time.monotonic_ns()
            return True, frame
        def connected(self): return True
    class Tracker:
        def detect(self, image, timestamp):
            timeline.append('detect'); return None
    class Controller:
        def __init__(self, **kwargs): pass
        def restart_encounter(self): pass
        def close(self): pass
        def update(self, observation, timestamp, **kwargs):
            timeline.append('model_update')
            return GuidanceDecision('LOOKING', GuidanceCue.WAIT, (.5, .3, .2), None, None, False, 'LOOKING')
    class Display:
        active = True
        n = -1
        phases = []
        def take_action(self):
            self.n += 1
            if self.n == 0: return 'start'
            if self.n == stop_at:
                self.active = False
                return 'stop'
            if self.n == repeated_start_at: return 'start'
            return None
        def simulator_intro_available(self): return True
        def simulator_intro_active(self, source): return acknowledge
        def simulator_intro_ready(self, source): return acknowledge and self.n >= 3
        def reference_held(self, source): return False
        def publish(self, frame, decision, **kwargs): self.phases.append(decision.phase)
    monkeypatch.setattr(app, 'GeminiGuidanceController', Controller)
    display = Display()
    output = io.StringIO()
    app.process_frames(Capture(), Tracker(), app.AppConfig(mode='guide-person', camera_source='ipad',
        guidance_profile='claude-guide', single_plan=True, headless=True, max_frames=2,
        guidance_stream=True, emit_every_frames=1), output=output, guest_display=display)
    return display, timeline, [json.loads(line) for line in output.getvalue().splitlines()]
