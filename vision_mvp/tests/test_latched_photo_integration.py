"""通过配对后的拍照状态验证实际的单次构图控制器。"""
from concurrent.futures import Future
import io
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import urllib.request

import numpy as np

import leica_tracker.app as app
from leica_tracker.gemini_framing import CompositionPlan
from leica_tracker.gemini_guidance import GeminiGuidanceController
from leica_tracker.guest_display import GuestDisplay


def test_real_controller_latch_is_seen_by_paired_photo_frontend(monkeypatch):
    """人物真实到位并保持后，必须通过和访客页面相同的检查。"""
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    target = (.5, .25, .3)
    plan = CompositionPlan(target, (.12, .04, .85, .94), .9,
                           'environmental', 'Keep space beside the subject.')

    class Worker:
        def __init__(self):
            self.calls = 0

        def submit(self, jpeg, point):
            self.calls += 1
            result = Future()
            result.set_result(plan)
            return result

    class Capture:
        def read(self):
            time.sleep(.08)
            self.source_monotonic_ns = time.monotonic_ns()
            return True, frame

        def connected(self):
            return True

        def fresh(self):
            return True

        def diagnostics(self):
            return {}

    class Tracker:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame, _timestamp):
            self.calls += 1
            return SimpleNamespace(center_x=target[0] + .075, head_y=target[1],
                                   shoulder_width=target[2], confidence=.9,
                                   landmarks=())

    worker = Worker()
    tracker = Tracker()
    capture = Capture()
    monkeypatch.setenv('LEICA_SIM_FAST_GESTURES', '1')
    monkeypatch.setattr(app, '_framing_client', lambda _config: None)
    monkeypatch.setattr(app, 'GeminiGuidanceController',
                        lambda **kwargs: GeminiGuidanceController(worker=worker, **kwargs))
    with GuestDisplay(port=0, remote_capture=capture, lease_seconds=30) as display:
        headers = {'X-Guest-Token': display.token, 'X-Guest-Client': 'photo-test',
                   'Origin': display.origin, 'Content-Type': 'application/json'}
        request = urllib.request.Request(display.origin + '/action',
                                        data=b'{"action":"start"}',
                                        headers=headers, method='POST')
        with urllib.request.urlopen(request, timeout=3):
            pass
        display.take_action()
        output = io.StringIO()
        app.process_frames(capture, tracker, app.AppConfig(
            mode='guide-person', camera_source='ipad', guidance_profile='claude-guide',
            single_plan=True, headless=True, max_frames=80, guidance_stream=True,
            emit_every_frames=1), output=output, guest_display=display)
        request = urllib.request.Request(display.origin + '/state', headers=headers)
        with urllib.request.urlopen(request, timeout=3) as response:
            state = json.load(response)

    # 访问者略微位于原始模型的x公差（.06）之外，但位于仅用于模拟的原型公差（.09）内。
    assert worker.calls == 1
    assert state['phase'] == 'ARRIVED' and state['photo_latched'] is True
    assert state['owner'] is True and state['tracking'] is False
    assert tracker.calls < 80, 'pose detection should stop after the latch'
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert any(e.get('reason') == 'TARGET_LATCHED' and e.get('cue') == 'HOLD'
               for e in events)

    # 使用GuestDisplay生成的JSON运行实际的访客页面门禁。
    html = Path(__file__).resolve().parents[1] / 'src/leica_tracker/guest.html'
    script = r'''const fs=require('fs');
const html=fs.readFileSync(process.argv[1],'utf8');
const match=html.match(/<script id="guest-photo">([\s\S]*?)<\/script>/);
if(!match)process.exit(2);
const module={exports:{}};new Function('module','exports',match[1])(module,module.exports);
const state=JSON.parse(fs.readFileSync(0,'utf8'));
const video={videoWidth:1280,videoHeight:720,readyState:4,paused:false,
 srcObject:{getVideoTracks:()=>[{readyState:'live',muted:false}]}};
if(!module.exports.canStart(state,video,false))process.exit(3);
'''
    result = subprocess.run(['node', '-e', script, str(html)], input=json.dumps(state),
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_simulated_prototype_tolerance_requires_all_dimensions_for_two_seconds():
    target = (.5, .25, .3)
    plan = CompositionPlan(target, (.12, .04, .85, .94), .9,
                           'environmental', 'A nearby portrait.')

    def inner(scale):
        controller = GeminiGuidanceController(single_plan=True, worker=object(),
                     fixed_plan_tolerance_scale=scale, arrival_dwell_ms=2000)
        controller._scene_template = plan
        controller._activate_template(target)
        return controller._inner

    def obs(x=.575, y=.25, width=.3):
        return SimpleNamespace(center_x=x, head_y=y, shoulder_width=width, confidence=.9)

    # 在原始比例尺上，略有偏移的主体不被认为是准备就绪的。
    normal = inner(1.)
    normal.update(obs(), 0, invitation_already_sent=True)
    assert normal.update(obs(), 2500, invitation_already_sent=True).phase != 'ARRIVED'

    # 在原型公差下，1.999秒不会触发照片拍摄；2秒会触发。
    prototype = inner(1.5)
    assert prototype.update(obs(), 0, invitation_already_sent=True).phase == 'SETTLING'
    assert prototype.update(obs(), 1999, invitation_already_sent=True).phase == 'SETTLING'
    assert prototype.update(obs(), 2000, invitation_already_sent=True).phase == 'ARRIVED'

    # 更大的水平、垂直或尺寸误差绝不能被视为到达。
    for outlier in (obs(x=.60), obs(y=.43), obs(width=.43)):
        c = inner(1.5)
        c.update(outlier, 0, invitation_already_sent=True)
        assert c.update(outlier, 2500, invitation_already_sent=True).phase != 'ARRIVED'

    # 跟踪丢失后须重新累计停留时间，避免按旧状态拍照。
    c = inner(1.5)
    c.update(obs(), 0, invitation_already_sent=True)
    c.update(None, 1900, invitation_already_sent=True)
    assert c.update(obs(), 2000, invitation_already_sent=True).phase == 'SETTLING'
