"""使用假相机/控制时间运行实际帧循环；无实时服务。"""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from leica_tracker import app
from leica_tracker.person_guidance import GuidanceCue, GuidanceDecision


def encounter(monkeypatch, cancellation=None, held_frames=(),communication_frames=()):
    clock = SimpleNamespace(now=10.)
    def advance(seconds): clock.now += seconds
    monkeypatch.setattr(app, 'time', SimpleNamespace(monotonic=lambda: clock.now,
        monotonic_ns=lambda: round(clock.now * 1e9), sleep=advance))
    monkeypatch.delenv('LEICA_SIM_FAST_GESTURES', raising=False)
    monkeypatch.delenv('LEICA_PHYSICAL_PREVIEW', raising=False)
    monkeypatch.setattr(app, '_framing_client', lambda _: None)
    timeline = []

    class Capture:
        n = -1
        source_monotonic_ns = 0
        def __init__(self): self.invalidations = 0; self.last_good = clock.now
        def read(self):
            self.n += 1
            advance((.5 if communication_frames else 3.) if self.n == 2 else .08)
            timeline.append(('read', self.n))
            if self.n in (1, 2):
                if self.n == 1: self.old_stamp = round((clock.now - .01) * 1e9)
                if self.n == 2: self.disconnected_during_outage = not self.connected()
                return False, None
            self.source_monotonic_ns = self.old_stamp if self.n == 3 and not communication_frames else round((clock.now - .01) * 1e9)
            if self.n != 3: self.last_good = clock.now
            if self.n == 4: self.recovered_stamp = self.source_monotonic_ns
            return True, np.full((24, 32, 3), 2 if self.n == 3 else 1 if self.n == 0 else 3, np.uint8)
        def connected(self): return clock.now - self.last_good <= 2.5
        def invalidate_pending(self): self.invalidations += 1
    capture = Capture()

    class Tracker:
        def detect(self, frame, timestamp):
            value = int(frame[0, 0, 0]); timeline.append(('detect', capture.n, value))
            return SimpleNamespace(confidence=1., center_x=.5, head_y=.3, shoulder_width=.2)

    class Controller:
        def __init__(self, **kwargs): self.template = 'same cached scene'; self.resets = []
        def reset(self): self.template = None; self.resets.append(capture.n); timeline.append(('reset', capture.n))
        def restart_encounter(self): self.reset()
        def close(self): pass
        def suspend_perception(self,timestamp):
            timeline.append(('suspend',capture.n))
            return GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,(.5,.3,.2),
                None,None,False,'AWAITING_MEASURED_REFERENCE',interaction_mode='claude-guide',model_requests=1)
        def update(self, observation, timestamp, **kwargs):
            timeline.append(('perception', capture.n, observation is not None))
            return GuidanceDecision('GUIDING', GuidanceCue.WAIT, (.5, .3, .2), None,
                (.5, .3, .2) if observation else None, observation is not None, 'CACHED_SCENE',
                interaction_mode='claude-guide', model_requests=1)
    controller = Controller()
    monkeypatch.setattr(app, 'GeminiGuidanceController', lambda **kwargs: controller)

    class Display:
        active = True
        epoch = 1
        instance = 'owned-control'
        physical_feedback_enabled = True
        def __init__(self): self.pauses = []; self.published = []
        def take_control_action(self):
            action = None
            if capture.n == 2 and cancellation == 'stop':
                action = 'stop'; self.active = False; self.epoch = 2
            if capture.n == 2 and cancellation == 'epoch': self.epoch = 2
            if capture.n == 2 and cancellation == 'service': self.instance = 'new-service'
            return {'action': action, 'instance': self.instance, 'epoch': self.epoch,
                    'control_started_ns': 1, 'active': self.active}
        def control_snapshot(self):
            if capture.n == 2 and cancellation == 'snapshot_epoch': self.epoch = 2
            return {'instance': self.instance, 'epoch': self.epoch, 'active': self.active,
                    'control_fresh': not (cancellation == 'control' and capture.n >= 2),
                    'camera_connected': capture.connected(), 'camera_fresh': False,
                    **({'control_fresh':capture.n not in communication_frames,
                        'communication':{'status':'waiting' if capture.n in communication_frames else 'healthy',
                            'recoverable':True,'remaining_ms':2000 if capture.n in communication_frames else 0}}
                       if communication_frames else {})}
        def pause_camera(self): self.pauses.append(capture.n); self.active = False; self.epoch += 1
        def physical_backend_available(self): return not (cancellation == 'control' and capture.n >= 2)
        def reference_held(self, source): return capture.n in held_frames
        def publish(self, frame, decision, **kwargs): self.published.append((capture.n, frame is None, decision))
    display = Display()
    output = io.StringIO()
    app.process_frames(capture, Tracker(), app.AppConfig(mode='guide-person', camera_source='ipad',
        guidance_profile='claude-guide', single_plan=True, headless=True, max_frames=3 if communication_frames else 2,
        guidance_stream=True, emit_every_frames=1), output=output, guest_display=display)
    return capture, display, controller, timeline, [json.loads(line) for line in output.getvalue().splitlines()]


def test_negotiated_control_hold_keeps_ai_goal_and_skips_images_until_same_encounter_resumes(monkeypatch):
    capture,display,controller,timeline,events=encounter(monkeypatch,communication_frames=(1,2,3))
    assert display.active and display.epoch==1 and controller.template=='same cached scene'
    assert controller.resets==[] and display.pauses==[]
    assert [entry for entry in timeline if entry[0]=='perception']==[('perception',0,True),('perception',4,True)]
    assert not any(entry[0]=='detect' and entry[1]==3 for entry in timeline)
    holds=[event for event in events if event['reason']=='COMMUNICATION_RECOVERING']
    assert holds and all(event['cue']=='HOLD' and event['tracking'] is False for event in holds)


def test_long_camera_gap_retains_owned_encounter_and_only_real_new_images_resume_perception(monkeypatch):
    capture, display, controller, timeline, events = encounter(monkeypatch)
    assert display.active and display.epoch == 1 and display.pauses == []
    assert capture.disconnected_during_outage
    assert capture.invalidations == 0 and controller.template == 'same cached scene' and controller.resets == []
    assert [item for item in timeline if item[0] == 'perception'] == [('perception', 0, True), ('perception', 4, True)]
    assert [item[2] for item in timeline if item[0] == 'detect'] == [1, 3]
    missing = [event for event in events if event['reason'] == 'CAMERA_FRAME_MISSING']
    assert missing and all(event['cue'] == 'HOLD' and event['tracking'] is False for event in missing)
    assert events[-1]['source_monotonic_ns'] == capture.recovered_stamp
    assert capture.old_stamp not in [event['source_monotonic_ns'] for event in events]
    assert all(event['model_requests'] <= 1 for event in events)


def test_reference_motion_hold_calls_explicit_suspend_instead_of_person_lost(monkeypatch):
    capture,display,controller,timeline,events=encounter(monkeypatch,held_frames=(0,))
    assert ('suspend',0) in timeline
    assert [item for item in timeline if item[0]=='perception']==[('perception',4,True)]
    assert controller.template=='same cached scene' and controller.resets==[]
    assert any(event['reason']=='AWAITING_MEASURED_REFERENCE' for event in events)


@pytest.mark.parametrize('cancellation', ['stop', 'epoch', 'service', 'control', 'snapshot_epoch'])
def test_control_cancellation_during_camera_outage_cancels_perception_before_recovery(monkeypatch, cancellation):
    capture, display, controller, timeline, events = encounter(monkeypatch, cancellation)
    assert 2 in controller.resets
    assert timeline.index(('reset', 2)) < timeline.index(('read', 3))
    assert any(index == 2 and decision.cue is GuidanceCue.HOLD for index, _, decision in display.published)
    # 过时的读取循环快照无法撤销更新的启动/摄像头会话。只有GuestDisplay和控制守护程序拥有这些过渡。
    assert display.pauses == [] and capture.invalidations == 0
    assert display.active is (cancellation != 'stop')
    if cancellation == 'snapshot_epoch': assert display.epoch == 2
    if cancellation in ('stop', 'control'):
        assert ('perception', 4, True) not in timeline
    assert all(event['cue'] == 'HOLD' for event in events if event['source_monotonic_ns'] < 13_200_000_000 and event['reason'] == 'CAMERA_FRAME_MISSING')
