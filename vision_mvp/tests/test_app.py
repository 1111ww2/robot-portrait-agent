import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import leica_tracker.app as app_module
from leica_tracker.app import (
    AppConfig,
    CameraReadError,
    build_parser,
    process_frames,
    serialize_guidance_event,
    serialize_tracking_event,
    serialize_zone_event,
)
from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.robot_gestures import gesture_plan_for


def test_simulator_gesture_modes_use_hysteresis_and_reset_per_encounter():
    from leica_tracker.app import SimulatorGestureScale
    scale = SimulatorGestureScale()
    assert scale.select(.72) == 1.0
    assert scale.select(.57) == 1.0
    assert scale.select(.49) == .65
    assert scale.select(.57) == .65
    assert scale.select(.66) == 1.0
    scale.select(.35)
    scale.reset()
    assert scale.select(.57) == 1.0
    for invalid in (True, float('nan'), -.1, 1.1):
        with pytest.raises(ValueError): scale.select(invalid)


@pytest.mark.parametrize('fast,camera,profile,single,display,expected', [
    (True,'ipad','claude-guide',True,True,.04),
    (False,'ipad','claude-guide',True,True,None),
    (True,'webcam','claude-guide',True,True,None),
    (True,'ipad','gemini-guide',True,True,None),
    (True,'ipad','claude-guide',False,True,None),
    (True,'ipad','claude-guide',True,False,None),
])
def test_far_subject_threshold_is_limited_to_current_simulator_flow(monkeypatch,fast,camera,profile,single,display,expected):
    class Captured(Exception):pass
    observed={}
    def controller(**kwargs):
        observed.update(kwargs)
        raise Captured
    if fast:monkeypatch.setenv('LEICA_SIM_FAST_GESTURES','1')
    else:monkeypatch.delenv('LEICA_SIM_FAST_GESTURES',raising=False)
    monkeypatch.setattr(app_module,'_framing_client',lambda _config:None)
    monkeypatch.setattr(app_module,'GeminiGuidanceController',controller)
    config=AppConfig(mode='guide-person',camera_source=camera,guidance_profile=profile,
                     single_plan=single,headless=True)
    with pytest.raises(Captured):
        process_frames(None,None,config,guest_display=object() if display else None)
    assert observed.get('min_subject_width')==expected


def test_sim_fast_successive_direction_events_switch_scale_on_distance(monkeypatch):
    import time
    from leica_tracker.person_guidance import GuidanceDecision
    monkeypatch.setenv('LEICA_SIM_FAST_GESTURES','1')
    monkeypatch.setattr(app_module,'_framing_client',lambda _config:None)
    frame=np.zeros((72,128,3),np.uint8)
    class Capture:
        def connected(self):return True
        def read(self):
            time.sleep(.02);self.source_monotonic_ns=time.monotonic_ns();return True,frame
    class Controller:
        def __init__(self,**kwargs):self.n=0
        def close(self):pass
        def update(self,observation,timestamp_ms,**kwargs):
            samples=[.8,None,.4,None,.57,None,.7]
            value=samples[self.n];self.n+=1
            return GuidanceDecision('GUIDING' if value is not None else 'WAITING_FOR_RESPONSE',
                GuidanceCue.MOVE_LEFT if value is not None else GuidanceCue.WAIT,
                (.5,.25,.3),None,(.7,.25,.3),True,'OFF_TARGET_HORIZONTAL',
                gesture_amplitude=value if value is not None else 1.)
    class Display:
        active=True
        def take_action(self):return None
        def reference_held(self,_source):return False
        def publish(self,*args,**kwargs):pass
    monkeypatch.setattr(app_module,'GeminiGuidanceController',Controller)
    output=io.StringIO()
    process_frames(Capture(),FixedTracker(.7),AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,max_frames=7,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    plans=[e['gesture'] for e in events if e['gesture']]
    assert [p['amplitude'] for p in plans]==[1.,.65,.65,1.]
    assert all(p['profile']=='sim-fast' for p in plans)
from leica_tracker.zones import Zone


class ScriptedCapture:
    def __init__(self, reads):
        self._reads = iter(reads)

    def read(self):
        return next(self._reads, (False, None))


class FixedTracker:
    def __init__(self, x):
        self._detection = SimpleNamespace(
            center_x=x,
            head_y=0.25,
            shoulder_width=0.30,
            confidence=0.9,
            landmarks=(),
        )

    def detect(self, _frame, timestamp_ms):
        assert timestamp_ms >= 0
        return self._detection


class ScriptedTracker:
    def __init__(self, detections):
        self._detections = iter(detections)

    def detect(self, _frame, timestamp_ms):
        assert timestamp_ms >= 0
        return next(self._detections)


def test_agent_camera_loop_uses_model_gesture_without_pose_and_never_logs_dialogue(monkeypatch):
    from leica_tracker.claude_agent import ClaudeInteractionController
    from test_claude_agent import Worker
    import itertools
    clock=itertools.count(100.,.08)
    monkeypatch.setattr(app_module.time,'monotonic',lambda:next(clock))
    w=Worker();c=ClaudeInteractionController(worker=w)
    monkeypatch.setattr(app_module,'ClaudeInteractionController',lambda **kw:c)
    monkeypatch.setattr(app_module,'FacePresence',lambda:SimpleNamespace(update=lambda *a:True))
    class Display:
        active=True
        def take_action(self):return None
        def take_messages(self):return []
        def set_dialogue(self,d):self.dialogue=d
        def publish(self,*a,**kw):
            if w.calls and not w.calls[-1][1].done():w.resolve(repetitions=3,cycle_ms=2500,speech='PRIVATE TEST PHRASE')
    d=Display();output=io.StringIO();frame=np.zeros((72,128,3),np.uint8)
    process_frames(ScriptedCapture([(True,frame)]*8),ScriptedTracker([None]*8),
        AppConfig(guidance_profile='claude-agent',headless=True,max_frames=8,guidance_stream=True,emit_every_frames=1),
        output=output,guest_display=d)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    gestures=[e['gesture'] for e in events if e.get('gesture')]
    assert len(gestures)==1 and gestures[0]['repetitions']==3 and gestures[0]['duration_ms']==7500
    assert gestures[0]['profile']=='agent-coupled'
    assert all(e['interaction_mode']=='claude-agent' for e in events)
    assert 'PRIVATE TEST PHRASE' not in output.getvalue()
    assert d.dialogue['speech']=='PRIVATE TEST PHRASE'


def test_parser_defaults_match_pocket_3_setup():
    args = build_parser().parse_args([])

    assert args.camera == 0
    assert (args.width, args.height, args.fps) == (1280, 720, 30)
    assert args.mirror is True
    assert args.headless is False
    assert args.target_x == pytest.approx(0.50)
    assert args.target_head_y == pytest.approx(0.25)
    assert args.target_shoulder_width == pytest.approx(0.30)
    assert args.emit_every_frames == 3
    assert args.mode == "guide-person"


@pytest.mark.parametrize(
    "source_mirrored,preview_mirrored,raw_column,preview_column",
    [(True, True, 75, 75), (True, False, 75, 24),
     (False, True, 25, 74), (False, False, 25, 25)],
)
def test_source_parity_preserves_participant_left_cue(
    monkeypatch, source_mirrored, preview_mirrored, raw_column, preview_column
):
    """一个人必须被要求保持饲料平衡。"""
    import itertools
    from dataclasses import asdict

    clock = itertools.count(100., 1.)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: next(clock))
    frame = np.zeros((72, 100, 3), dtype=np.uint8)
    frame[:, raw_column] = 255

    class PixelTracker:
        def detect(self, image, _timestamp_ms):
            column = int(image.sum(axis=(0, 2)).argmax())
            return SimpleNamespace(center_x=column / 100, head_y=.25,
                                   shoulder_width=.30, confidence=.9, landmarks=())

    config_values = asdict(AppConfig(
        mirror=preview_mirrored, headless=True, max_frames=4,
        smoothing_alpha=1., guidance_stream=True, emit_every_frames=1))
    config_values["source_mirrored"] = source_mirrored
    config = SimpleNamespace(**config_values)
    output = io.StringIO()
    process_frames(ScriptedCapture([(True, frame.copy()) for _ in range(4)]),
                   PixelTracker(), config, output=output)
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[0]["points"]["current"]["center_x"] == preview_column / 100
    assert "MOVE_LEFT" in [event["cue"] for event in events]
    assert "MOVE_RIGHT" not in [event["cue"] for event in events]


def test_parser_keeps_follow_frame_as_an_explicit_comparison_mode():
    args = build_parser().parse_args(["--mode", "follow-frame"])

    assert args.mode == "follow-frame"


def test_horizontal_only_is_rejected_in_person_guidance_mode():
    args = build_parser().parse_args(["--horizontal-only"])

    with pytest.raises(ValueError, match="only available with --mode follow-frame"):
        app_module._config_from_args(args)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--smoothing-alpha", "0"],
        ["--target-shoulder-width", "0"],
        ["--max-step-m", "0"],
    ],
)
def test_parser_rejects_zero_for_parameters_that_must_be_positive(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_zone_event_is_machine_readable_and_preserves_loss():
    left = json.loads(serialize_zone_event(Zone.LEFT, 0.125, 1234))
    lost = json.loads(serialize_zone_event(Zone.LOST, None, 4321))

    assert left == {
        "event": "zone_change",
        "timestamp_ms": 1234,
        "x": 0.125,
        "zone": "LEFT",
    }
    assert lost == {
        "event": "zone_change",
        "timestamp_ms": 4321,
        "x": None,
        "zone": "LOST",
    }


def test_tracking_event_is_machine_readable_and_camera_relative():
    command = SimpleNamespace(
        tracking=True,
        right=0.12567,
        down=-0.25,
        toward_subject=0.75,
        reason="TRACKING",
    )
    intent = SimpleNamespace(
        hold=False,
        right_m=0.01,
        down_m=-0.02,
        toward_subject_m=0.03,
        reason="TRACKING",
    )

    event = json.loads(serialize_tracking_event(command, intent, timestamp_ms=987))

    assert event == {
        "command": {
            "down": -0.25,
            "right": 0.1257,
            "toward_subject": 0.75,
        },
        "coordinate_frame": "camera_relative_normalized",
        "event": "tracking_target",
        "motion_intent_m": {
            "down": -0.02,
            "hold": False,
            "right": 0.01,
            "toward_subject": 0.03,
        },
        "reason": "TRACKING",
        "timestamp_ms": 987,
        "tracking": True,
    }


def test_process_frames_emits_continuous_targets_and_safe_stop():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = ScriptedCapture([(True, frame.copy()) for _ in range(3)])
    output = io.StringIO()
    config = AppConfig(
        mode="follow-frame",
        headless=True,
        max_frames=3,
        emit_every_frames=1,
        smoothing_alpha=1.0,
        horizontal_deadband=0.0,
        vertical_deadband=0.0,
        scale_deadband=0.0,
    )
    detections = [
        SimpleNamespace(
            center_x=0.50,
            head_y=0.25,
            shoulder_width=0.30,
            confidence=0.9,
            landmarks=(),
        ),
        SimpleNamespace(
            center_x=0.70,
            head_y=0.35,
            shoulder_width=0.20,
            confidence=0.9,
            landmarks=(),
        ),
        None,
    ]

    stats = process_frames(capture, ScriptedTracker(detections), config, output=output)

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [event["event"] for event in events] == [
        "tracking_target",
        "tracking_target",
        "tracking_target",
    ]
    assert events[0]["command"] == {
        "down": 0.0,
        "right": 0.0,
        "toward_subject": 0.0,
    }
    assert events[1]["command"]["right"] > 0.0
    assert events[1]["command"]["down"] > 0.0
    assert events[1]["command"]["toward_subject"] > 0.0
    assert events[2]["tracking"] is False
    assert events[2]["reason"] == "PERSON_LOST"
    assert events[2]["motion_intent_m"]["hold"] is True
    assert (
        events[2]["motion_intent_m"]["right"] == events[1]["motion_intent_m"]["right"]
    )
    assert stats.frames_processed == 3
    assert stats.target_events == 3


def test_process_frames_applies_step_limit_per_emitted_target_not_camera_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = ScriptedCapture([(True, frame.copy()) for _ in range(4)])
    output = io.StringIO()
    config = AppConfig(
        mode="follow-frame",
        headless=True,
        max_frames=4,
        emit_every_frames=3,
        max_step_m=0.01,
        smoothing_alpha=1.0,
        horizontal_deadband=0.0,
    )

    process_frames(capture, FixedTracker(1.0), config, output=output)

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(events) == 2
    assert events[0]["motion_intent_m"]["right"] == pytest.approx(0.01)
    assert events[1]["motion_intent_m"]["right"] == pytest.approx(0.02)


def test_preview_consumes_the_last_emitted_bounded_motion_intent(monkeypatch):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = ScriptedCapture([(True, frame.copy()) for _ in range(2)])
    config = AppConfig(
        mode="follow-frame",
        headless=False,
        max_frames=2,
        emit_every_frames=3,
        max_step_m=0.01,
        smoothing_alpha=1.0,
        horizontal_deadband=0.0,
    )
    preview_intents = []

    def record_overlay(_frame, _detection, _command, intent, _target, _envelope, _fps):
        preview_intents.append(intent)

    monkeypatch.setattr(app_module, "_draw_overlay", record_overlay)
    monkeypatch.setattr(app_module.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(app_module.cv2, "waitKey", lambda _delay: -1)

    process_frames(capture, FixedTracker(1.0), config, output=io.StringIO())

    assert len(preview_intents) == 2
    assert preview_intents[0].right_m == pytest.approx(0.01)
    assert preview_intents[1] == preview_intents[0]


def test_process_frames_stops_after_repeated_camera_read_failures():
    capture = ScriptedCapture([(False, None)] * 3)
    config = AppConfig(headless=True, max_read_failures=3)

    with pytest.raises(CameraReadError, match="3 consecutive"):
        process_frames(capture, FixedTracker(0.5), config, output=io.StringIO())


@pytest.mark.parametrize('mode', ['guide-person', 'follow-frame'])
def test_missing_camera_frame_emits_hold_before_recovery(mode):
    frame=np.zeros((72,128,3),dtype=np.uint8)
    output=io.StringIO()
    process_frames(ScriptedCapture([(True,frame),(False,None),(True,frame)]),
        FixedTracker(.7),AppConfig(mode=mode,headless=True,max_frames=2,
        guidance_stream=True,emit_every_frames=1),output=output)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert events[1]['tracking'] is False
    assert events[1]['reason']=='CAMERA_FRAME_MISSING'
    if mode=='guide-person':
        assert events[1]['cue']=='HOLD' and events[1]['gesture'] is None
        assert events[2]['cue']=='INVITE'
    else:
        assert events[1]['motion_intent_m']['hold'] is True


def test_disconnected_camera_preview_stays_open_and_allows_quit(monkeypatch):
    images=[];keys=iter([-1,-1,ord('q')])
    monkeypatch.setattr(app_module.cv2,'imshow',lambda title,image:images.append(image.copy()))
    monkeypatch.setattr(app_module.cv2,'waitKey',lambda delay:next(keys))
    stats=process_frames(ScriptedCapture([(False,None)]*3),FixedTracker(.5),
        AppConfig(headless=False,max_read_failures=2),output=io.StringIO())
    assert stats.frames_processed==0
    assert len(images)==3 and all(image.any() for image in images)


def test_retry_key_reopens_camera_and_resets_guidance(monkeypatch):
    frame=np.zeros((72,128,3),dtype=np.uint8)
    class ReopenCapture(ScriptedCapture):
        def release(self): pass
        def open(self,index,backend):
            self._reads=iter([(True,frame)])
            return True
        def set(self,*args): return True
    keys=iter([ord('r'),ord('q')])
    monkeypatch.setattr(app_module.cv2,'imshow',lambda *args:None)
    monkeypatch.setattr(app_module.cv2,'waitKey',lambda delay:next(keys))
    output=io.StringIO()
    stats=process_frames(ReopenCapture([(False,None)]),FixedTracker(.7),
        AppConfig(headless=False,max_read_failures=1),output=output)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert stats.frames_processed==1
    assert events[0]['cue']=='HOLD' and events[1]['cue']=='INVITE'


@pytest.mark.parametrize('x,want_arrival',[(.56,True),(.7,False)])
def test_frame_guide_accepts_region_but_still_guides_outside(monkeypatch,x,want_arrival):
    import itertools
    clock=itertools.count(100.,.1)
    monkeypatch.setattr(app_module.time,'monotonic',lambda:next(clock))
    config=app_module._config_from_args(build_parser().parse_args([
        '--guidance-profile','frame-guide','--headless','--max-frames','35',
        '--guidance-stream','--emit-every-frames','1']))
    frame=np.zeros((72,128,3),dtype=np.uint8);output=io.StringIO()
    pose=SimpleNamespace(center_x=x,head_y=.31,shoulder_width=.24,confidence=.9,landmarks=())
    process_frames(ScriptedCapture([(True,frame)]*35),ScriptedTracker([pose]*35),
                   config,output=output)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert any(e['cue']=='ARRIVED' for e in events)==want_arrival
    assert any(e['cue']=='MOVE_LEFT' for e in events)==(not want_arrival)
    assert all(e['interaction_mode']=='frame-guide' and not e['capture_simulated']
               and not e['framing_verified'] for e in events)
    assert not any(e['cue'] in ('SIM_FRAME','SIM_COMPLETE') for e in events)


def test_horizontal_only_locks_vertical_and_depth_through_tracking_and_loss():
    config = app_module._config_from_args(
        build_parser().parse_args([
            "--mode", "follow-frame", "--horizontal-only", "--headless", "--max-frames", "3",
            "--emit-every-frames", "1", "--smoothing-alpha", "1",
        ])
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = ScriptedCapture([(True, frame.copy()) for _ in range(3)])
    detections = [
        SimpleNamespace(center_x=0.25, head_y=0.80, shoulder_width=0.10),
        SimpleNamespace(center_x=0.75, head_y=0.05, shoulder_width=0.60),
        None,
    ]
    output = io.StringIO()
    process_frames(capture, ScriptedTracker(detections), config, output=output)
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[0]["command"]["right"] < 0
    assert events[1]["command"]["right"] > 0
    for event in events:
        assert event["command"]["down"] == event["command"]["toward_subject"] == 0
        assert event["motion_intent_m"]["down"] == event["motion_intent_m"]["toward_subject"] == 0
    assert events[2]["motion_intent_m"]["hold"] is True
    assert events[2]["motion_intent_m"]["right"] == events[1]["motion_intent_m"]["right"]


def test_guidance_event_contains_points_semantic_gesture_and_no_image():
    decision = SimpleNamespace(
        phase="GUIDING",
        cue=GuidanceCue.MOVE_LEFT,
        tracking=True,
        reason="OFF_TARGET_HORIZONTAL",
        target=(0.5, 0.25, 0.3),
        initial_point=(0.3, 0.4, 0.2),
        current_point=(0.7, 0.4, 0.2),
    )

    event = json.loads(
        serialize_guidance_event(
            decision,
            gesture_plan_for(decision.cue),
            timestamp_ms=123,
            source_monotonic_ns=456,
        )
    )

    assert event["event"] == "person_guidance"
    assert event["coordinate_frame"] == "normalized_camera_image"
    assert event["robot_actuated"] is False
    assert event["cue"] == "MOVE_LEFT"
    assert event["gesture"]["pattern"] == "double_sweep"
    assert event["points"]["initial"] == {
        "center_x": 0.3,
        "head_y": 0.4,
        "shoulder_width": 0.2,
    }
    assert event["points"]["current"]["center_x"] == 0.7
    assert "image" not in event
    assert event["source_monotonic_ns"] == 456


def test_guidance_preview_draws_fixed_target_and_session_points():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    decision = SimpleNamespace(
        phase="GUIDING",
        cue=GuidanceCue.MOVE_RIGHT,
        reason="OFF_TARGET_HORIZONTAL",
        tracking=True,
        target=(0.5, 0.25, 0.3),
        initial_point=(0.3, 0.4, 0.2),
        current_point=(0.7, 0.4, 0.2),
    )

    app_module._draw_guidance_overlay(
        frame,
        None,
        decision,
        fps=30.0,
        gesture=gesture_plan_for(GuidanceCue.MOVE_RIGHT),
        gesture_elapsed_ms=300,
    )

    assert frame.shape == (720, 1280, 3)
    assert np.count_nonzero(frame) > 0


def test_guidance_preview_keeps_and_times_the_active_gesture(monkeypatch):
    decisions = iter(
        [
            SimpleNamespace(
                phase="GUIDING",
                cue=GuidanceCue.MOVE_RIGHT,
                tracking=True,
                reason="OFF_TARGET_HORIZONTAL",
                target=(0.5, 0.25, 0.3),
                initial_point=(0.4, 0.4, 0.3),
                current_point=(0.4, 0.4, 0.3),
            ),
            SimpleNamespace(
                phase="WAITING_FOR_RESPONSE",
                cue=GuidanceCue.WAIT,
                tracking=True,
                reason="WAITING_FOR_RESPONSE",
                target=(0.5, 0.25, 0.3),
                initial_point=(0.4, 0.4, 0.3),
                current_point=(0.4, 0.4, 0.3),
            ),
            SimpleNamespace(
                phase="TRACKING_LOST",
                cue=GuidanceCue.HOLD,
                tracking=False,
                reason="PERSON_LOST",
                target=(0.5, 0.25, 0.3),
                initial_point=(0.4, 0.4, 0.3),
                current_point=None,
            ),
        ]
    )

    class ScriptedGuidanceController:
        def __init__(self, **_kwargs):
            pass

        def update(self, _detection, _timestamp_ms, *, mirrored=True):
            return next(decisions)

    monkeypatch.setattr(
        app_module, "PersonGuidanceController", ScriptedGuidanceController
    )
    monkeypatch.setattr(app_module.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(app_module.cv2, "waitKey", lambda _delay: -1)
    previews = []

    def record_preview(
        _frame,
        _detection,
        _decision,
        _fps,
        *,
        gesture,
        gesture_elapsed_ms,
        mirrored=True,
    ):
        previews.append((gesture, gesture_elapsed_ms))

    monkeypatch.setattr(app_module, "_draw_guidance_overlay", record_preview)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = ScriptedCapture(
        [(True, frame.copy()), (True, frame.copy()), (True, frame.copy())]
    )

    process_frames(
        capture,
        FixedTracker(0.4),
        AppConfig(mode="guide-person", headless=False, max_frames=3),
        output=io.StringIO(),
    )

    assert len(previews) == 3
    assert previews[0][0].pattern == "double_sweep"
    assert previews[1][0].pattern == "double_sweep"
    assert previews[1][1] >= previews[0][1]
    assert previews[2][0] is None


def test_guide_person_loop_emits_only_semantic_state_transitions(monkeypatch):
    cues = [
        ("INVITING", GuidanceCue.INVITE, True),
        ("INVITING", GuidanceCue.INVITE, True),
        ("GUIDING", GuidanceCue.MOVE_LEFT, True),
        ("GUIDING", GuidanceCue.MOVE_LEFT, True),
        ("GUIDING", GuidanceCue.WAIT, True),
        ("TRACKING_LOST", GuidanceCue.HOLD, False),
        ("ARRIVED", GuidanceCue.ARRIVED, True),
    ]

    class ScriptedGuidanceController:
        def __init__(self, **_kwargs):
            self._decisions = iter(cues)

        def update(self, _detection, _timestamp_ms, *, mirrored=True):
            phase, cue, tracking = next(self._decisions)
            return SimpleNamespace(
                phase=phase,
                cue=cue,
                tracking=tracking,
                reason=phase,
                target=(0.5, 0.25, 0.3),
                initial_point=(0.3, 0.4, 0.2),
                current_point=(0.5, 0.25, 0.3) if tracking else None,
            )

    monkeypatch.setattr(
        app_module, "PersonGuidanceController", ScriptedGuidanceController
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    capture = ScriptedCapture([(True, frame.copy()) for _ in cues])
    detections = [SimpleNamespace(confidence=0.9) for _ in cues]
    output = io.StringIO()

    stats = process_frames(
        capture,
        ScriptedTracker(detections),
        AppConfig(mode="guide-person", headless=True, max_frames=len(cues)),
        output=output,
    )

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [event["cue"] for event in events] == [
        "INVITE",
        "MOVE_LEFT",
        "WAIT",
        "HOLD",
        "ARRIVED",
    ]
    assert [
        event["gesture"]["pattern"] if event["gesture"] else None
        for event in events
    ] == ["open_invitation", "double_sweep", None, None, "confirm_retract_hold"]
    assert stats.target_events == 5


def test_guidance_stream_heartbeats_keep_source_fresh_without_replaying_gesture():
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    output = io.StringIO()
    config = AppConfig(headless=True, max_frames=7, guidance_stream=True,
                       emit_every_frames=1)
    process_frames(ScriptedCapture([(True, frame.copy()) for _ in range(7)]),
                   FixedTracker(.4), config, output=output)
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(events) == 7
    assert len([e for e in events if e['gesture'] is not None]) == 1
    assert all(e['source_monotonic_ns'] > 0 for e in events)
    assert all(e['robot_actuated'] is False for e in events)
    args = build_parser().parse_args(['--guidance-stream'])
    assert app_module._config_from_args(args).guidance_stream is True


def test_guidance_preview_shows_control_point_without_a_second_misleading_frame():
    from leica_tracker.person_guidance import GuidanceDecision
    def render(x):
        canvas=np.zeros((720,1280,3),dtype=np.uint8)
        decision=GuidanceDecision('GUIDING',GuidanceCue.MOVE_RIGHT,
            (.5,.25,.3),(.2,.25,.12),(x,.25,.12),True,'OFF_TARGET_HORIZONTAL')
        app_module._draw_guidance_overlay(canvas,None,decision,30.)
        return canvas
    left=render(.2);right=render(.8)
    assert tuple(left[180,268]) == (0,190,255)
    assert tuple(right[180,1036]) == (0,190,255)
    assert tuple(left[450,102]) != (0,190,255)
    # 固定绿色目标的x=256边界在人移动时保持不变。
    assert tuple(left[150,256]) == (80,230,150)
    assert tuple(right[150,256]) == (80,230,150)


def test_target_head_and_shoulder_markers_are_visible_without_a_person():
    from leica_tracker.person_guidance import GuidanceDecision
    frame=np.zeros((720,1280,3),dtype=np.uint8)
    decision=GuidanceDecision('TRACKING_LOST',GuidanceCue.HOLD,
        (.5,.25,.3),None,None,False,'PERSON_LOST')
    app_module._draw_guidance_overlay(frame,None,decision,30.)
    green=(80,230,150)
    # 厚实的目标框架，放大后的头部环和肩宽=0.3 * 1280。
    assert tuple(frame[150,258]) == green
    assert tuple(frame[180,666]) == green
    assert tuple(frame[317,448]) == green
    assert tuple(frame[317,832]) == green


def test_completed_preview_has_persistent_success_border_even_after_person_leaves():
    def render(phase,tracking):
        frame=np.zeros((720,1280,3),dtype=np.uint8)
        d=SimpleNamespace(phase=phase,cue=GuidanceCue.HOLD,target=(.5,.25,.3),
            current_point=None,initial_point=None,tracking=tracking,reason=phase,
            arrival_progress=1. if phase=='ARRIVED' else 0.)
        app_module._draw_guidance_overlay(frame,None,d,30.)
        return frame
    assert tuple(render('ARRIVED',True)[10,10]) == (80,230,150)
    assert tuple(render('ARRIVED',False)[10,10]) == (80,230,150)
    assert tuple(render('TRACKING_LOST',False)[10,10]) != (80,230,150)
    assert tuple(render('WAITING_FOR_RESPONSE',True)[10,10]) != (80,230,150)


def test_arrival_confirmation_preview_shows_dwell_progress_bar():
    frame=np.zeros((720,1280,3),dtype=np.uint8)
    d=SimpleNamespace(phase='SETTLING',cue=GuidanceCue.WAIT,target=(.5,.25,.3),
        current_point=None,initial_point=None,tracking=True,reason='ARRIVAL_DWELL',
        arrival_progress=.5)
    app_module._draw_guidance_overlay(frame,None,d,30.)
    assert tuple(frame[557,400]) == (80,230,150)
    assert tuple(frame[557,800]) == (60,60,60)


def test_coarse_profile_runs_an_explicit_simulated_workflow_in_the_camera_loop(monkeypatch):
    import itertools
    clock=itertools.count(100.,.1)
    monkeypatch.setattr(app_module.time,'monotonic',lambda:next(clock))
    config=app_module._config_from_args(build_parser().parse_args(
        ['--guidance-profile','coarse-demo','--headless','--max-frames','75','--guidance-stream','--emit-every-frames','1']))
    frame=np.zeros((72,128,3),dtype=np.uint8);output=io.StringIO()
    process_frames(ScriptedCapture([(True,frame.copy()) for _ in range(75)]),
                   FixedTracker(.61),config,output=output)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert any(e['phase']=='DEMO_COMPLETE' for e in events)
    assert any(e['phase']=='SIMULATED_CAPTURE' and e['capture_simulated'] for e in events)
    assert all(e['framing_verified'] is False for e in events)
    assert sum(e['cue']=='SIM_FRAME' for e in events)==1
    assert sum(e['cue']=='SIM_COMPLETE' for e in events)==1
    with pytest.raises(ValueError,match='only available'):
        app_module._config_from_args(build_parser().parse_args(
            ['--mode','follow-frame','--guidance-profile','coarse-demo']))


def test_gemini_auth_preflight_does_not_open_camera_or_fall_back(monkeypatch,capsys):
    from leica_tracker.gemini_framing import FramingError
    class Denied:
        def __init__(self,**kwargs):pass
        def check_access(self):raise FramingError('AI_AUTH_ERROR')
    monkeypatch.setattr(app_module,'GeminiFramingClient',Denied)
    def unexpected_camera(*args):raise AssertionError('Camera opened despite denied API access')
    monkeypatch.setattr(app_module.cv2,'VideoCapture',unexpected_camera)
    assert app_module.main(['--guidance-profile','gemini-guide'])==4
    assert 'AI_AUTH_ERROR' in capsys.readouterr().err


@pytest.mark.parametrize('profile',['gemini-guide','claude-guide'])
@pytest.mark.parametrize('headless',[True,False])
@pytest.mark.parametrize('guest',[True,False])
@pytest.mark.parametrize('fast',[True,False])
def test_camera_loop_uses_gemini_crop_and_target_instead_of_fixed_coordinates(monkeypatch,headless,guest,profile,fast):
    if fast: monkeypatch.setenv('LEICA_SIM_FAST_GESTURES','1')
    else: monkeypatch.delenv('LEICA_SIM_FAST_GESTURES',raising=False)
    from concurrent.futures import Future
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from leica_tracker.gemini_framing import validate_plan
    import itertools
    clock=itertools.count(100.,.1)
    monkeypatch.setattr(app_module.time,'monotonic',lambda:next(clock))
    class Worker:
        def submit(self,jpeg,point):
            future=Future();future.set_result(validate_plan(dict(subject_visible=True,
                confidence=.9,style='half_body',frame=[.3,.02,.98,.95],
                target=[.68,.3,.2],explanation='Keep person naturally to the right.')))
            return future
    def factory(**kwargs):
        client=kwargs.pop('client')
        assert client.model.startswith('claude-' if profile=='claude-guide' else 'gemini-')
        assert kwargs['entry_ack_ms']==(3000 if guest else 800)
        assert kwargs['response_window_ms']==(3900 if fast and guest else 6700 if guest else 3000)
        assert kwargs.get('fast_direction_reversal',False)==(fast and guest)
        return GeminiGuidanceController(worker=Worker(),**kwargs)
    monkeypatch.setattr(app_module,'GeminiGuidanceController',factory)
    monkeypatch.setattr(app_module.cv2,'imshow',lambda *args:None)
    monkeypatch.setattr(app_module.cv2,'waitKey',lambda *args:-1)
    frame=np.zeros((72,128,3),dtype=np.uint8)
    observation=SimpleNamespace(center_x=.68,head_y=.3,shoulder_width=.2,confidence=.9,landmarks=())
    output=io.StringIO()
    class Display:
        active=True
        def take_action(self):return None
        def publish(self,*args,**kwargs):pass
    process_frames(ScriptedCapture([(True,frame)]*35),ScriptedTracker([observation]*35),
        AppConfig(guidance_profile=profile,headless=headless,max_frames=35,
                  guidance_stream=True,emit_every_frames=1),output=output,
                  guest_display=Display() if guest else None)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert all(e['interaction_mode']==profile for e in events)
    assert events[0]['phase']=='AI_ENTRY_ACK' and events[0]['cue']=='INVITE'
    assert events[0]['gesture']['profile']==('screen-coupled' if guest else 'semantic')
    assert sum(e['cue']=='ARRIVED' for e in events)==1
    planned=[e for e in events if e['composition_frame']]
    assert all(e['points']['target']==dict(center_x=.68,head_y=.3,shoulder_width=.2) for e in planned)
    assert all(e['composition_frame']==pytest.approx([.32235294,.03741176,.96235294,.91270588],abs=1e-7) for e in planned)
    assert all(not e['framing_verified'] and not e['capture_simulated'] for e in events)
    assert not any('data:image' in line or 'base64' in line for line in output.getvalue().splitlines())


def test_transient_camera_read_failure_keeps_the_gemini_scene_template(monkeypatch):
    from concurrent.futures import Future
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from leica_tracker.gemini_framing import validate_plan
    class ImmediateModel:
        def submit(self,jpeg,point):
            result=Future()
            result.set_result(validate_plan(dict(subject_visible=True,confidence=.9,
                style='environmental',frame=[.12,.04,.85,.94],target=[.62,.25,.3],
                explanation='Keep the window with space on the left.')))
            return result
    monkeypatch.setattr(app_module,'GeminiGuidanceController',
        lambda *,client,**options:GeminiGuidanceController(worker=ImmediateModel(),**options))
    frame=np.zeros((72,128,3),dtype=np.uint8);output=io.StringIO()
    process_frames(ScriptedCapture([(True,frame),(True,frame),(False,None),(True,frame)]),
        FixedTracker(.48),AppConfig(mode='guide-person',guidance_profile='gemini-guide',
        headless=True,max_frames=3,guidance_stream=True,emit_every_frames=1),output=output)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert events[2]['cue']=='HOLD' and events[2]['reason']=='CAMERA_FRAME_MISSING'
    assert events[-1]['composition_frame'] is not None
    assert events[-1]['model_requests']==1


def test_ipad_discards_frame_issued_before_camera_loss_hold_without_faking_freshness(monkeypatch):
    import time,itertools
    from leica_tracker.simulator_guidance import GuidanceStream
    clock=itertools.count(time.monotonic_ns()-100_000_000,10_000_000)
    monkeypatch.setattr(app_module.time,'monotonic_ns',lambda:next(clock))
    frame1=np.full((72,128,3),1,np.uint8)
    old=np.full((72,128,3),2,np.uint8)
    fresh=np.full((72,128,3),3,np.uint8)
    class Capture:
        def __init__(self):self.step=0;self.source_monotonic_ns=0
        def connected(self):return True
        def read(self):
            self.step+=1
            if self.step==1:
                self.source_monotonic_ns=time.monotonic_ns()-10_000_000;return True,frame1
            if self.step==2:
                self.issued_before_loss=time.monotonic_ns()-1;return False,None
            if self.step==3:
                self.source_monotonic_ns=self.issued_before_loss;return True,old
            self.source_monotonic_ns=time.monotonic_ns();self.fresh_stamp=self.source_monotonic_ns
            return True,fresh
    class Tracker(FixedTracker):
        def __init__(self):super().__init__(.7);self.samples=[]
        def detect(self,frame,timestamp):self.samples.append(int(frame[0,0,0]));return super().detect(frame,timestamp)
    cap=Capture();tracker=Tracker();output=io.StringIO()
    process_frames(cap,tracker,AppConfig(camera_source='ipad',mode='guide-person',
        single_plan=True,headless=True,max_frames=2,guidance_stream=True,emit_every_frames=1),output=output)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert tracker.samples==[1,3]
    assert events[1]['cue']=='HOLD' and events[1]['reason']=='CAMERA_FRAME_MISSING'
    stamps=[e['source_monotonic_ns'] for e in events]
    assert stamps==sorted(set(stamps)) and stamps[-1]==cap.fresh_stamp
    stream=GuidanceStream()
    for e in events:stream.accept(json.dumps(e),time.monotonic())


def test_reference_gate_blocks_observations_and_paid_planning_until_fresh_release(monkeypatch):
    from concurrent.futures import Future
    import itertools
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from test_single_snapshot import PLAN
    class Model:
        calls=0
        def submit(self,*args):
            self.calls+=1;f=Future();f.set_result(PLAN);return f
    model=Model()
    fake_source=itertools.count(1_000_000_000,1_000_000)
    monkeypatch.setattr(app_module.time,'monotonic_ns',lambda:next(fake_source))
    monkeypatch.setattr(app_module,'GeminiGuidanceController',
        lambda *,client,**kw:GeminiGuidanceController(worker=model,**kw))
    class Display:
        active=True;count=0
        def take_action(self):return None
        def reference_held(self,source_ns):return self.count<3
        def publish(self,*args,**kwargs):self.count+=1
    frame=np.zeros((72,128,3),np.uint8);output=io.StringIO()
    process_frames(ScriptedCapture([(True,frame)]*5),FixedTracker(.48),
        AppConfig(mode='guide-person',guidance_profile='claude-guide',single_plan=True,
            headless=True,max_frames=5,guidance_stream=True,emit_every_frames=1),
        output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert all(e['phase']=='PHYSICAL_REFERENCE_HOLD' and e['cue']=='HOLD' and not e['tracking'] for e in events[:3])
    assert all(e['model_requests']==0 for e in events[:3])
    assert model.calls==1


def test_inactive_guest_still_displays_reference_hold(monkeypatch):
    class Display:
        active=False
        def take_action(self):return None
        def reference_held(self,source_ns):return True
        def publish(self,*args,**kwargs):pass
    frame=np.zeros((72,128,3),np.uint8);output=io.StringIO()
    process_frames(ScriptedCapture([(True,frame)]),FixedTracker(.48),
        AppConfig(mode='guide-person',guidance_profile='claude-guide',single_plan=True,
            headless=True,max_frames=1,guidance_stream=True,emit_every_frames=1),
        output=output,guest_display=Display())
    event=json.loads(output.getvalue().splitlines()[0])
    assert event['phase']=='PHYSICAL_REFERENCE_HOLD' and event['cue']=='HOLD'


def test_reference_gate_keeps_locked_plan_after_camera_returns(monkeypatch):
    from concurrent.futures import Future
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from test_single_snapshot import PLAN
    import itertools
    class Model:
        calls=0
        def submit(self,*args):
            self.calls+=1;f=Future();f.set_result(PLAN);return f
    model=Model();fake_source=itertools.count(1_000_000_000,1_000_000)
    monkeypatch.setattr(app_module.time,'monotonic_ns',lambda:next(fake_source))
    monkeypatch.setattr(app_module,'GeminiGuidanceController',
        lambda *,client,**kw:GeminiGuidanceController(worker=model,**kw))
    class Display:
        active=True;count=0
        def take_action(self):return None
        def reference_held(self,source_ns):return 3<=self.count<8
        def publish(self,*args,**kwargs):self.count+=1
    frame=np.zeros((72,128,3),np.uint8);output=io.StringIO()
    process_frames(ScriptedCapture([(True,frame)]*11),FixedTracker(.48),
        AppConfig(mode='guide-person',guidance_profile='claude-guide',single_plan=True,
            headless=True,max_frames=11,guidance_stream=True,emit_every_frames=1),
        output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert events[2]['composition_frame']==list(PLAN.frame)
    assert all(e['phase']=='PHYSICAL_REFERENCE_HOLD' and e['cue']=='HOLD' for e in events[3:8])
    assert events[-1]['composition_frame']==list(PLAN.frame) and model.calls==1


@pytest.mark.parametrize('disconnect',[False,True])
def test_single_plan_ipad_gap_holds_motion_but_only_disconnect_ends_encounter(monkeypatch,disconnect):
    import itertools,time
    source_clock=itertools.count(time.monotonic_ns(),10_000_000)
    monkeypatch.setattr(app_module.time,'monotonic_ns',lambda:next(source_clock))
    from concurrent.futures import Future
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from test_single_snapshot import PLAN
    class Model:
        def submit(self,*args):
            f=Future();f.set_result(PLAN);return f
    monkeypatch.setattr(app_module,'GeminiGuidanceController',
        lambda *,client,**kw:GeminiGuidanceController(worker=Model(),**kw))
    frame=np.zeros((72,128,3),np.uint8)
    class Capture(ScriptedCapture):
        source_monotonic_ns=0
        misses=0
        def read(self):
            import time
            result=super().read()
            if not result[0]:self.misses+=1
            else:self.source_monotonic_ns=time.monotonic_ns()
            return result
        def connected(self):return not (disconnect and self.misses>=2)
        def invalidate_pending(self):pass
    class Display:
        active=True
        def take_action(self):return None
        def publish(self,*args,**kw):pass
        def pause_camera(self):self.active=False
    display=Display();output=io.StringIO()
    capture=Capture([(True,frame),(True,frame),(False,None),(False,None),(True,frame)])
    process_frames(capture,FixedTracker(.48),AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,max_frames=3,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=display)
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    missing=[e for e in events if e['reason']=='CAMERA_FRAME_MISSING']
    assert missing and all(e['cue']=='HOLD' and e['gesture'] is None for e in missing)
    assert display.active is (not disconnect)
    if not disconnect:
        assert events[-1]['composition_frame']==list(PLAN.frame)
        assert events[-1]['model_requests']==1
    else:assert events[-1]['phase']=='GUEST_READY' and events[-1]['cue']=='HOLD'


def test_guest_display_starts_held_and_stop_reaches_the_guidance_stream():
    from leica_tracker.guest_display import GuestDisplay
    from urllib.request import Request,urlopen
    import threading,queue
    frame=np.zeros((72,128,3),np.uint8)
    reads=queue.Queue();output=io.StringIO();failure=[]
    class Capture:
        def read(self):return reads.get(timeout=3)
    class Display(GuestDisplay):
        def __init__(self):super().__init__(port=0);self.published=queue.Queue()
        def publish(self,image,d,**kwargs):
            super().publish(image,d,**kwargs);self.published.put(d)
    with Display() as display:
        def run():
            try:process_frames(Capture(),FixedTracker(.7),AppConfig(headless=True,max_frames=4,
                guidance_stream=True,emit_every_frames=1),output=output,guest_display=display)
            except BaseException as error:failure.append(error)
        thread=threading.Thread(target=run);thread.start()
        def act(name):
            request=Request(display.origin+'/action',data=json.dumps({'action':name}).encode(),
                headers={'Content-Type':'application/json','Origin':display.origin,
                'X-Guest-Token':display.token,'X-Guest-Client':'ipad'},method='POST')
            with urlopen(request,timeout=2):pass
        reads.put((True,frame));assert display.published.get(timeout=2).cue is GuidanceCue.HOLD
        act('start');reads.put((True,frame));assert display.published.get(timeout=2).cue is GuidanceCue.INVITE
        act('stop');reads.put((True,frame));assert display.published.get(timeout=2).cue is GuidanceCue.HOLD
        act('restart');reads.put((True,frame));assert display.published.get(timeout=2).cue is GuidanceCue.INVITE
        thread.join(timeout=3);assert not thread.is_alive() and not failure
    cues=[json.loads(line)['cue'] for line in output.getvalue().splitlines()]
    assert cues==['HOLD','INVITE','HOLD','INVITE']


def test_entering_guest_acceptance_region_stops_the_direction_sweep_during_dwell():
    class Display:
        active=True
        def take_action(self):return None
        def publish(self,*args,**kwargs):pass
    frame=np.zeros((72,128,3),np.uint8);output=io.StringIO()
    process_frames(ScriptedCapture([(True,frame)]*3),FixedTracker(.5),
        AppConfig(headless=True,max_frames=3,guidance_stream=True,emit_every_frames=1),
        output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    settling=[e for e in events if e['phase']=='SETTLING']
    assert settling and all(e['cue']=='HOLD' and e['gesture'] is None for e in settling)


def test_simulator_arrival_recovery_waits_for_fresh_valid_target_and_restarts_once():
    from leica_tracker.app import SimulatorArrivalRecovery
    from leica_tracker.person_guidance import GuidanceDecision
    arrived=GuidanceDecision('ARRIVED',GuidanceCue.WAIT,(.5,.25,.3),None,(.5,.25,.3),True,'ARRIVED')
    recovery=SimulatorArrivalRecovery(enabled=True)
    recovery.interrupt(gesture_plan_for(GuidanceCue.ARRIVED,profile='screen-coupled'),active=True)
    assert recovery.apply(arrived,active=True,source_ns=1_000_000_000,now_ns=1_400_000_000).cue is GuidanceCue.WAIT
    moved=__import__('dataclasses').replace(arrived,current_point=(.8,.25,.3))
    assert recovery.apply(moved,active=True,source_ns=2_000_000_000,now_ns=2_050_000_000).cue is GuidanceCue.WAIT
    assert recovery.apply(arrived,active=True,source_ns=3_000_000_000,now_ns=3_050_000_000).cue is GuidanceCue.ARRIVED
    assert recovery.apply(arrived,active=True,source_ns=3_100_000_000,now_ns=3_150_000_000).cue is GuidanceCue.WAIT


def test_simulator_arrival_recovery_never_resumes_after_stop_or_without_tracking():
    from leica_tracker.app import SimulatorArrivalRecovery
    from leica_tracker.person_guidance import GuidanceDecision
    from dataclasses import replace
    arrived=GuidanceDecision('ARRIVED',GuidanceCue.WAIT,(.5,.25,.3),None,(.5,.25,.3),True,'ARRIVED')
    gesture=gesture_plan_for(GuidanceCue.ARRIVED,profile='screen-coupled')
    recovery=SimulatorArrivalRecovery(enabled=True);recovery.interrupt(gesture,active=True)
    assert recovery.apply(replace(arrived,tracking=False),active=True,source_ns=1_000_000_000,now_ns=1_050_000_000).cue is GuidanceCue.WAIT
    assert recovery.apply(arrived,active=False,source_ns=2_000_000_000,now_ns=2_050_000_000).cue is GuidanceCue.WAIT
    assert recovery.apply(arrived,active=True,source_ns=3_000_000_000,now_ns=3_050_000_000).cue is GuidanceCue.WAIT
    disabled=SimulatorArrivalRecovery(enabled=False);disabled.interrupt(gesture,active=True)
    assert disabled.apply(arrived,active=True,source_ns=4_000_000_000,now_ns=4_050_000_000).cue is GuidanceCue.WAIT


def test_simulator_latched_arrival_stays_stopped_after_camera_recovers(monkeypatch):
    from leica_tracker.person_guidance import GuidanceDecision
    from dataclasses import replace
    import time
    monkeypatch.setenv('LEICA_SIM_FAST_GESTURES','1')
    monkeypatch.setattr(app_module,'_framing_client',lambda _config:None)
    frame=np.zeros((72,128,3),np.uint8)
    class Capture:
        def __init__(self):self.reads=iter([(True,frame),(False,None),(True,frame),(True,frame)])
        def connected(self):return True
        def read(self):
            time.sleep(.02)
            self.source_monotonic_ns=time.monotonic_ns()
            return next(self.reads,(False,None))
    class Controller:
        def __init__(self,**kwargs):
            assert kwargs['arrival_dwell_ms']==2000
            self.count=0
        def close(self):pass
        def update(self,observation,timestamp_ms,**kwargs):
            d=GuidanceDecision('ARRIVED',GuidanceCue.WAIT,(.5,.25,.3),None,(.5,.25,.3),True,'ARRIVED')
            self.count+=1
            assert self.count==1, 'locked encounter must not re-run position selection'
            return replace(d,cue=GuidanceCue.ARRIVED)
    class Display:
        active=True
        def take_action(self):return None
        def reference_held(self,_source):return False
        def publish(self,*args,**kwargs):pass
    monkeypatch.setattr(app_module,'GeminiGuidanceController',Controller)
    class TrackingOnce:
        def __init__(self):self.calls=0
        def detect(self,frame,timestamp_ms):
            self.calls+=1
            assert self.calls==1, 'locked encounter must not run pose detection'
            return FixedTracker(.5)._detection
    tracker=TrackingOnce();output=io.StringIO()
    process_frames(Capture(),tracker,AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,max_frames=3,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert [e['cue'] for e in events]==['HOLD','HOLD','HOLD','HOLD']
    assert all(e['reason']=='TARGET_LATCHED' and e['gesture'] is None for e in events)
    assert tracker.calls==1
    assert events[2]['source_monotonic_ns']>events[1]['source_monotonic_ns']>events[0]['source_monotonic_ns']


@pytest.mark.parametrize('stale_frame',[False,True])
def test_simulator_latched_arrival_never_recalibrates_after_stale_source_or_gap(monkeypatch,stale_frame):
    from leica_tracker.person_guidance import GuidanceDecision
    from dataclasses import replace
    from leica_tracker.simulator_reference_guidance import ReferenceGuidanceStream
    import time
    monkeypatch.setenv('LEICA_SIM_FAST_GESTURES','1')
    monkeypatch.setattr(app_module,'_framing_client',lambda _config:None)
    frame=np.zeros((72,128,3),np.uint8)
    class Capture:
        def __init__(self):self.n=0
        def connected(self):return True
        def read(self):
            self.n+=1
            time.sleep(.40 if self.n==2 else .02)
            self.source_monotonic_ns=time.monotonic_ns()-(310_000_000 if stale_frame and self.n==2 else 0)
            return True,frame
    class Controller:
        def __init__(self,**kwargs):self.count=0
        def close(self):pass
        def update(self,observation,timestamp_ms,**kwargs):
            self.count+=1
            d=GuidanceDecision('ARRIVED',GuidanceCue.WAIT,(.5,.25,.3),None,(.5,.25,.3),True,'ARRIVED')
            return replace(d,cue=GuidanceCue.ARRIVED) if self.count==1 else d
    class Display:
        active=True
        def take_action(self):return None
        def reference_held(self,_source):return False
        def publish(self,*args,**kwargs):pass
    monkeypatch.setattr(app_module,'GeminiGuidanceController',Controller)
    output=io.StringIO()
    process_frames(Capture(),FixedTracker(.5),AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,max_frames=3,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert [e['cue'] for e in events]==['HOLD','HOLD','HOLD']
    assert all(e['reason']=='TARGET_LATCHED' and e['gesture'] is None for e in events)
    from pathlib import Path
    ref=json.loads((Path('vision_mvp/config/simulator-reference-2026-10-28.json')).read_text())
    stream=ReferenceGuidanceStream(ref['joints_rad'],ref['tcp_offset_m_rad'])
    for event in events:stream.accept(json.dumps(event),event['source_monotonic_ns']/1e9)
    assert stream.plan is None and stream.offset(events[-1]['source_monotonic_ns']/1e9) is None


def test_simulator_photo_latch_freezes_on_arrival_until_explicit_reset():
    from leica_tracker.app import SimulatorPhotoLatch
    from leica_tracker.person_guidance import GuidanceDecision
    arrived=GuidanceDecision('ARRIVED',GuidanceCue.ARRIVED,(.5,.25,.3),None,(.5,.25,.3),True,'TARGET_STABLE')
    moved=GuidanceDecision('GUIDING',GuidanceCue.MOVE_LEFT,(.5,.25,.3),None,(.1,.25,.3),True,'OFF_TARGET')
    lost=GuidanceDecision('TRACKING_LOST',GuidanceCue.HOLD,(.5,.25,.3),None,None,False,'PERSON_LOST')
    latch=SimulatorPhotoLatch(enabled=True)
    result=latch.apply(arrived,active=True)
    assert (result.phase,result.cue,result.tracking,result.reason)==('ARRIVED',GuidanceCue.HOLD,False,'TARGET_LATCHED')
    assert latch.apply(moved,active=True)==result
    assert latch.apply(lost,active=True)==result
    assert latch.apply(moved,active=False)==moved
    assert latch.apply(moved,active=True)==moved
    latch.apply(arrived,active=True);latch.reset()
    assert latch.apply(moved,active=True)==moved


def test_simulator_photo_latch_does_not_change_other_modes():
    from leica_tracker.app import SimulatorPhotoLatch
    from leica_tracker.person_guidance import GuidanceDecision
    arrived=GuidanceDecision('ARRIVED',GuidanceCue.ARRIVED,(.5,.25,.3),None,(.5,.25,.3),True,'TARGET_STABLE')
    assert SimulatorPhotoLatch(enabled=False).apply(arrived,active=True)==arrived


def test_simulator_direction_recovery_rearms_only_matching_fresh_target_once():
    from leica_tracker.app import SimulatorDirectionRecovery
    from leica_tracker.person_guidance import GuidanceDecision
    target=(.5,.25,.3)
    waiting=GuidanceDecision('WAITING_FOR_RESPONSE',GuidanceCue.WAIT,target,None,(.7,.25,.3),True,'WAITING_FOR_RESPONSE')
    plan=gesture_plan_for(GuidanceCue.MOVE_LEFT,profile='sim-fast')
    r=SimulatorDirectionRecovery(enabled=True)
    r.interrupt(plan,target,active=True)
    assert r.apply(waiting,active=True,source_ns=1_000_000_000,now_ns=1_400_000_000,mirrored=True).cue is GuidanceCue.WAIT
    recovered=r.apply(waiting,active=True,source_ns=1_100_000_000,now_ns=1_150_000_000,mirrored=True)
    assert recovered.cue is GuidanceCue.MOVE_LEFT and recovered.reason=='SIMULATOR_DIRECTION_RECOVERY'
    assert r.apply(waiting,active=True,source_ns=1_200_000_000,now_ns=1_250_000_000,mirrored=True).cue is GuidanceCue.WAIT


def test_simulator_direction_recovery_never_replays_stopped_reversed_or_new_target():
    from leica_tracker.app import SimulatorDirectionRecovery
    from leica_tracker.person_guidance import GuidanceDecision
    from dataclasses import replace
    target=(.5,.25,.3); waiting=GuidanceDecision('WAITING_FOR_RESPONSE',GuidanceCue.WAIT,target,None,(.7,.25,.3),True,'WAITING_FOR_RESPONSE')
    plan=gesture_plan_for(GuidanceCue.MOVE_LEFT,profile='sim-fast')
    for decision in (replace(waiting,current_point=(.3,.25,.3)),replace(waiting,target=(.4,.25,.3)),replace(waiting,tracking=False)):
        r=SimulatorDirectionRecovery(enabled=True);r.interrupt(plan,target,active=True)
        assert r.apply(decision,active=True,source_ns=1_000_000_000,now_ns=1_100_000_000,mirrored=True).cue is GuidanceCue.WAIT
    r=SimulatorDirectionRecovery(enabled=True);r.interrupt(plan,target,active=True)
    assert r.apply(waiting,active=False,source_ns=1_000_000_000,now_ns=1_100_000_000,mirrored=True).cue is GuidanceCue.WAIT
    assert r.apply(waiting,active=True,source_ns=1_200_000_000,now_ns=1_250_000_000,mirrored=True).cue is GuidanceCue.WAIT
    r=SimulatorDirectionRecovery(enabled=False);r.interrupt(plan,target,active=True)
    assert r.apply(waiting,active=True,source_ns=1_000_000_000,now_ns=1_100_000_000,mirrored=True).cue is GuidanceCue.WAIT


def test_direction_feedback_waits_for_ack_grace_then_requires_persistent_loss():
    from leica_tracker.app import SimulatorDirectionRecovery
    r=SimulatorDirectionRecovery(enabled=True);s=1_000_000_000
    assert not r.feedback_interrupted(False,source_ns=s,now_ns=s+250_000_000)
    assert not r.feedback_interrupted(False,source_ns=s,now_ns=s+350_000_000)
    assert not r.feedback_interrupted(False,source_ns=s,now_ns=s+500_000_000)
    assert r.feedback_interrupted(False,source_ns=s,now_ns=s+560_000_000)
    assert not r.feedback_interrupted(True,source_ns=s,now_ns=s+600_000_000)
    assert not r.feedback_interrupted(None,source_ns=s,now_ns=s+900_000_000)


def test_direction_recovery_limits_retries_until_new_original_gesture():
    from leica_tracker.app import SimulatorDirectionRecovery
    from leica_tracker.person_guidance import GuidanceDecision
    target=(.5,.25,.3);wait=GuidanceDecision('WAITING_FOR_RESPONSE',GuidanceCue.WAIT,target,None,(.7,.25,.3),True,'WAITING_FOR_RESPONSE')
    plan=gesture_plan_for(GuidanceCue.MOVE_LEFT,profile='sim-fast');r=SimulatorDirectionRecovery(enabled=True)
    for i in range(2):
        r.interrupt(plan,target,active=True)
        assert r.apply(wait,active=True,source_ns=1_000_000_000,now_ns=1_100_000_000,mirrored=True).cue is GuidanceCue.MOVE_LEFT
    r.interrupt(plan,target,active=True)
    assert r.apply(wait,active=True,source_ns=1_000_000_000,now_ns=1_100_000_000,mirrored=True).cue is GuidanceCue.WAIT
    r.new_gesture()
    r.interrupt(plan,target,active=True)
    assert r.apply(wait,active=True,source_ns=1_000_000_000,now_ns=1_100_000_000,mirrored=True).cue is GuidanceCue.MOVE_LEFT


def test_live_sim_direction_recovery_rearms_after_source_gap(monkeypatch):
    from leica_tracker.person_guidance import GuidanceDecision
    from leica_tracker.simulator_reference_guidance import ReferenceGuidanceStream
    from pathlib import Path
    import time
    monkeypatch.setenv('LEICA_SIM_FAST_GESTURES','1')
    monkeypatch.setattr(app_module,'_framing_client',lambda _config:None)
    frame=np.zeros((72,128,3),np.uint8)
    class Capture:
        def __init__(self):self.n=0
        def connected(self):return True
        def read(self):
            self.n+=1;time.sleep(.37 if self.n==2 else .02)
            self.source_monotonic_ns=time.monotonic_ns()
            return True,frame
    class Controller:
        def __init__(self,**kwargs):self.n=0
        def close(self):pass
        def update(self,observation,timestamp_ms,**kwargs):
            self.n+=1
            return GuidanceDecision('GUIDING' if self.n==1 else 'WAITING_FOR_RESPONSE',
                GuidanceCue.MOVE_LEFT if self.n==1 else GuidanceCue.WAIT,(.5,.25,.3),None,
                (.7,.25,.3),True,'OFF_TARGET_HORIZONTAL' if self.n==1 else 'WAITING_FOR_RESPONSE',
                gesture_amplitude=.4 if self.n==1 else 1.)
    class Display:
        active=True
        def take_action(self):return None
        def reference_held(self,_source):return False
        def publish(self,*args,**kwargs):pass
    monkeypatch.setattr(app_module,'GeminiGuidanceController',Controller)
    output=io.StringIO()
    process_frames(Capture(),FixedTracker(.7),AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,max_frames=3,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    assert [e['cue'] for e in events]==['MOVE_LEFT','MOVE_LEFT','WAIT']
    assert events[1]['reason']=='SIMULATOR_DIRECTION_RECOVERY'
    assert events[0]['gesture']['amplitude']==.65
    assert events[1]['gesture']['amplitude']==.65, 'WAIT recovery must retain the interrupted near mode'
    assert events[1]['source_monotonic_ns']>events[0]['source_monotonic_ns']
    ref=json.loads(Path('vision_mvp/config/simulator-reference-2026-10-28.json').read_text())
    stream=ReferenceGuidanceStream(ref['joints_rad'],ref['tcp_offset_m_rad'])
    for e in events:stream.accept(json.dumps(e),e['source_monotonic_ns']/1e9)
    assert stream.plan is not None and stream.offset(events[-1]['source_monotonic_ns']/1e9) is not None


def test_live_sim_direction_recovers_when_bridge_never_acknowledges(monkeypatch):
    from leica_tracker.person_guidance import GuidanceDecision
    import time
    monkeypatch.setenv('LEICA_SIM_FAST_GESTURES','1');monkeypatch.setattr(app_module,'_framing_client',lambda _config:None)
    frame=np.zeros((72,128,3),np.uint8)
    class Capture:
        def connected(self):return True
        def read(self):
            time.sleep(.09);self.source_monotonic_ns=time.monotonic_ns();return True,frame
    class Controller:
        def __init__(self,**kwargs):self.n=0
        def close(self):pass
        def update(self,observation,timestamp_ms,**kwargs):
            self.n+=1
            return GuidanceDecision('GUIDING' if self.n==1 else 'WAITING_FOR_RESPONSE',
                GuidanceCue.MOVE_LEFT if self.n==1 else GuidanceCue.WAIT,(.5,.25,.3),None,
                (.7,.25,.3),True,'OFF_TARGET_HORIZONTAL' if self.n==1 else 'WAITING_FOR_RESPONSE')
    class Display:
        active=True
        def take_action(self):return None
        def reference_held(self,_source):return False
        def publish(self,*args,**kwargs):pass
        def simulator_motion_active(self,source,cue):return False
    monkeypatch.setattr(app_module,'GeminiGuidanceController',Controller);output=io.StringIO()
    process_frames(Capture(),FixedTracker(.7),AppConfig(mode='guide-person',camera_source='ipad',
        guidance_profile='claude-guide',single_plan=True,headless=True,max_frames=20,
        guidance_stream=True,emit_every_frames=1),output=output,guest_display=Display())
    events=[json.loads(line) for line in output.getvalue().splitlines()]
    recoveries=[e for e in events if e['reason']=='SIMULATOR_DIRECTION_RECOVERY']
    assert len(recoveries)==2
    assert all(e['gesture']['cue']=='MOVE_LEFT' for e in recoveries)
    assert all(e['source_monotonic_ns']>events[0]['source_monotonic_ns'] for e in recoveries)


def test_direction_recovery_clears_on_arrival_and_no_longer_rearms():
    from leica_tracker.app import SimulatorDirectionRecovery
    from leica_tracker.person_guidance import GuidanceDecision
    from dataclasses import replace
    target=(.5,.25,.3);wait=GuidanceDecision('WAITING_FOR_RESPONSE',GuidanceCue.WAIT,target,None,(.7,.25,.3),True,'WAITING_FOR_RESPONSE')
    r=SimulatorDirectionRecovery(enabled=True);r.interrupt(gesture_plan_for(GuidanceCue.MOVE_LEFT,profile='sim-fast'),target,active=True)
    locked=replace(wait,phase='ARRIVED',cue=GuidanceCue.HOLD,tracking=False,reason='TARGET_LATCHED')
    assert r.apply(locked,active=True,source_ns=1_000_000_000,now_ns=1_100_000_000,mirrored=True)==locked
    assert r.apply(wait,active=True,source_ns=1_200_000_000,now_ns=1_250_000_000,mirrored=True).cue is GuidanceCue.WAIT
