import time
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
import pytest

from leica_tracker.physical_interaction_supervisor import PhysicalInteractionSupervisor, LocalReferenceGate
from leica_tracker.physical_resident_engine import Installation, ResidentEngine
from test_physical_resident_engine import reviewed, Gate, feedback
from test_guest_display import request
from leica_tracker import physical_interaction_supervisor as supervisor_module


class Immediate:
    def __init__(self, fn, *args):
        self.done = True
        try:
            self.value = fn(*args)
            self.error = None
        except Exception as exc:
            self.value, self.error = None, exc


class Ref:
    def __init__(self):
        self.calls = []
        self.ack = False
        self.session = 'session'
        self.photo_ns = 1_060_000_000
        self.identity = {'instance':'guest-a', 'epoch':1}
        self.started_ns = 1

    def begin(self, reference):
        self.calls.append(('begin', tuple(reference)))
        return self.session

    def status(self, session):
        self.calls.append(('status', session))
        return {'blocked': True, 'acknowledged': self.ack, 'session': self.session}

    def sample(self, session, pose, speed, sampled):
        self.calls.append(('sample', sampled))
        return {'ok':True,'sampled_at':sampled,'return_verified':True}

    def release(self, session):
        self.calls.append(('release', session))
        return 3_000_000_000

    def photo_source(self):
        self.calls.append(('photo_source',))
        return self.photo_ns

    def photo_ready(self, source_ns):
        self.calls.append(('photo_ready', source_ns))
        return True

    def fault(self):
        self.calls.append(('fault',))
        return True


def setup():
    cfg = Installation(reviewed())
    gate = Gate()
    gate.can_send = True
    engine = ResidentEngine(cfg, gate, 123)
    ref = Ref()
    way = {cue: [[.105, -.5, .3]] for cue in ('MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP',
           'MOVE_DOWN', 'MOVE_CLOSER', 'MOVE_FARTHER')}
    supervisor = PhysicalInteractionSupervisor(engine, ref, way, reviewed_paths=way, hold_timeout=2.,
        action_timeout=10., gate_timeout=.1, max_joint_speed=.001, worker_factory=Immediate)
    supervisor.bind_control_identity({'instance':'guest-a', 'epoch':1, 'started_ns':1})
    supervisor.set_move_readiness(lambda cue,source_ns: True)
    return cfg, gate, engine, ref, supervisor


def event(ns, cue='MOVE_LEFT', tracking=True):
    return {'event':'person_guidance', 'coordinate_frame':'normalized_camera_image',
            'robot_actuated':False, 'tracking':tracking, 'cue':cue, 'source_monotonic_ns':ns,
            'control_instance':'guest-a','control_epoch':1}


def step(supervisor, cfg, gate, t, xyz=None, joint_time=None):
    sent = gate.calls[-1] if gate.calls else None
    fb = feedback(cfg, 123, state=0 if sent else 2,
                  seq=sent[0] if sent else 0, xyz=xyz)
    joint = {'joint_speed': [0.] * 6, 'tcp_speed': [0.] * 6}
    supervisor.tick(fb, sampled_at=t, joint_feedback=joint,
                    joint_sampled_at=t if joint_time is None else joint_time, now=t)


def ready(supervisor, cfg, gate):
    step(supervisor, cfg, gate, 1.)
    step(supervisor, cfg, gate, 1.05)
    assert supervisor.engine.state == 'READY'


def completed_reference_begin(monkeypatch, *, completed_at, error=None):
    """带有注入单调时钟和本地参考栈的真实工作程序。"""
    cfg,gate,engine,ref,sup=setup();ready(sup,cfg,gate)
    now=[1.06]
    monkeypatch.setattr(supervisor_module,'time',SimpleNamespace(monotonic=lambda:now[0]))
    def begin(_reference):
        now[0]=completed_at
        if error is not None:raise error
        return 'session'
    ref.begin=begin;sup.worker_factory=supervisor_module._Worker
    assert sup.accept_event(event(1_060_000_000),now=1.06)
    assert sup._job._done.wait(1), 'local reference worker must finish before the delayed poll'
    return cfg,gate,engine,ref,sup,now


def test_fast_completed_reference_request_survives_a_late_main_poll(monkeypatch):
    cfg,gate,engine,_ref,sup,now=completed_reference_begin(monkeypatch,completed_at=1.08)
    now[0]=1.18
    step(sup,cfg,gate,1.18)
    assert sup.state=='WAIT_ACK' and engine.state=='READY'
    assert not gate.stopped and not any(packet[1] for packet in gate.calls)


def test_actually_overdue_completed_reference_request_still_faults(monkeypatch):
    cfg,gate,_engine,_ref,sup,now=completed_reference_begin(monkeypatch,completed_at=1.18)
    now[0]=1.19
    with pytest.raises(RuntimeError,match='Reference operation timed out'):
        step(sup,cfg,gate,1.19)
    assert gate.stopped and sup.state=='FAULT'


def test_fast_failed_reference_request_keeps_failure_cause_after_late_poll(monkeypatch):
    cfg,gate,_engine,_ref,sup,now=completed_reference_begin(
        monkeypatch,completed_at=1.08,error=ValueError('local reference rejected'))
    now[0]=1.18
    with pytest.raises(RuntimeError,match='Reference operation failed'):
        step(sup,cfg,gate,1.18)
    assert gate.stopped and sup.state=='FAULT'


def test_pending_reference_request_still_faults_when_original_deadline_expires(monkeypatch):
    cfg,gate,_engine,ref,sup=setup();ready(sup,cfg,gate)
    now=[1.06];release=threading.Event()
    monkeypatch.setattr(supervisor_module,'time',SimpleNamespace(monotonic=lambda:now[0]))
    def begin(_reference):
        if not release.wait(1):raise RuntimeError('test release timed out')
        return 'session'
    ref.begin=begin;sup.worker_factory=supervisor_module._Worker
    assert sup.accept_event(event(1_060_000_000),now=1.06)
    worker=sup._job
    try:
        now[0]=1.10;step(sup,cfg,gate,1.10)
        assert sup.state=='BEGIN_PENDING' and not gate.stopped
        now[0]=1.18
        with pytest.raises(RuntimeError,match='Reference operation timed out'):
            step(sup,cfg,gate,1.18)
        assert gate.stopped and sup.state=='FAULT'
    finally:
        release.set();assert worker._done.wait(1)


def test_no_motion_before_matching_hold_ack_and_post_release_frame():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    assert sup.accept_event(event(1_060_000_000), now=1.06)
    for t in (1.10, 1.15, 1.20):
        step(sup, cfg, gate, t)
    assert not any(call[1] == 1 for call in gate.calls)
    ref.session = 'other'
    ref.ack = True
    step(sup, cfg, gate, 1.25)
    with pytest.raises(RuntimeError):
        step(sup, cfg, gate, 1.30)
    assert gate.stopped
    assert not any(call[1] == 1 for call in gate.calls)


def test_direction_moves_out_returns_and_waits_for_fresh_settling_before_release():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    step(sup, cfg, gate, 1.10)
    ref.ack = True
    step(sup, cfg, gate, 1.15)
    step(sup, cfg, gate, 1.20)
    step(sup, cfg, gate, 1.25)
    assert engine.state == 'MOVING'
    t = 1.25
    for _ in range(55):
        t += .05
        target = engine.goal if engine.state == 'MOVING' else None
        xyz = target if target is not None else cfg.neutral[:3]
        step(sup, cfg, gate, t, xyz=xyz)
        if sup.state == 'OBSERVING':
            break
    assert sup.state == 'OBSERVING'
    targets = [tuple(x[3]) for x in gate.calls if x[1] == 1]
    assert (.105, -.5, .3) in targets
    assert tuple(cfg.neutral[:3]) in targets
    assert any(c[0] == 'release' for c in ref.calls)
    assert not sup.accept_event(event(3_000_000_000), now=max(t,3.))


def test_stop_and_telemetry_loss_never_command_return():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    step(sup, cfg, gate, 1.10)
    ref.ack = True
    step(sup, cfg, gate, 1.15)
    step(sup, cfg, gate, 1.20)
    step(sup, cfg, gate, 1.25)
    count = len(gate.calls)
    sup.stop()
    assert gate.stopped and len(gate.calls) == count
    with pytest.raises(RuntimeError):
        step(sup, cfg, gate, 1.30)
    assert not any(c[0] == 'release' for c in ref.calls)


def test_rejects_missing_reviewed_directions_and_stale_events():
    cfg, gate, engine, ref, sup = setup()
    with pytest.raises(ValueError):
        PhysicalInteractionSupervisor(engine, ref, {}, reviewed_paths={}, hold_timeout=1, action_timeout=1,
                                       gate_timeout=.1, max_joint_speed=.001)
    ready(sup, cfg, gate)
    assert not sup.accept_event(event(1_000_000_000), now=1.5)
    assert not any(c[0] == 'begin' for c in ref.calls)


def test_path_review_must_match_the_exact_forward_targets():
    cfg, gate, engine, ref, sup = setup()
    paths = dict(sup.paths)
    with pytest.raises(ValueError, match='review'):
        PhysicalInteractionSupervisor(engine, ref, paths, reviewed_paths={},
                                      hold_timeout=1, action_timeout=1, gate_timeout=.1,
                                      max_joint_speed=.001)


def test_hold_timeout_stops_without_a_target():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    t = 1.06
    with pytest.raises(RuntimeError, match='timed out'):
        for _ in range(50):
            t += .05
            step(sup, cfg, gate, t)
    assert gate.stopped
    assert not any(c[1] == 1 for c in gate.calls)


def test_feedback_failure_never_sends_automatic_return():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    step(sup, cfg, gate, 1.10)
    ref.ack = True
    step(sup, cfg, gate, 1.15)
    step(sup, cfg, gate, 1.20)
    step(sup, cfg, gate, 1.25)
    assert engine.state == 'MOVING'
    prior = len(gate.calls)
    fb = feedback(cfg, 123, state=1, seq=gate.calls[-1][0])
    with pytest.raises(RuntimeError):
        sup.tick(fb, sampled_at=1.25, joint_feedback={'joint_speed':[0.] * 6, 'tcp_speed':[0.] * 6},
                 joint_sampled_at=1.7, now=1.7)
    assert gate.stopped and len(gate.calls) == prior


def test_stale_independent_joint_feedback_is_rejected():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    with pytest.raises(RuntimeError, match='joint'):
        step(sup, cfg, gate, 1.10, joint_time=.5)
    assert gate.stopped


def test_wrong_epoch_event_faults_and_old_frame_never_moves():
    cfg, gate, engine, ref, sup = setup()
    ready(sup,cfg,gate)
    wrong = event(1_060_000_000); wrong['control_epoch'] = 2
    with pytest.raises(RuntimeError, match='identity'):
        sup.accept_event(wrong,now=1.06)
    assert gate.stopped and not any(c[1] == 1 for c in gate.calls)


def test_frame_issued_before_current_guest_epoch_never_moves():
    cfg, gate, engine, ref, sup = setup()
    ref.started_ns = 1_070_000_000
    sup.bind_control_identity({'instance':'guest-a','epoch':1,'started_ns':1_070_000_000})
    ready(sup,cfg,gate)
    assert not sup.accept_event(event(1_060_000_000),now=1.08)
    assert not any(c[0] == 'begin' for c in ref.calls)


def test_hold_ack_discards_an_expired_decision_without_ending_encounter():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    for i in range(1, 9):
        step(sup, cfg, gate, 1.06 + i * .05)
    ref.ack = True
    step(sup, cfg, gate, 1.51)
    step(sup, cfg, gate, 1.56)
    assert sup.recovering_expiry and not gate.stopped
    assert not any(c[1] == 1 for c in gate.calls)


def test_hold_ack_waits_for_matching_light_ack_before_motion():
    cfg, gate, engine, ref, sup = setup()
    ready_state = {'ready':False}
    sup.set_move_readiness(lambda cue,ns: ready_state['ready'] and cue=='MOVE_LEFT' and ns==1_060_000_000)
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    step(sup, cfg, gate, 1.10)
    ref.ack = True
    for t in (1.15,1.20,1.25): step(sup,cfg,gate,t)
    assert not any(c[1] == 1 for c in gate.calls)
    ready_state['ready'] = True
    step(sup,cfg,gate,1.30)
    assert engine.state == 'MOVING'


def test_light_ack_delay_cannot_move_an_expired_decision():
    cfg, gate, engine, ref, sup = setup()
    sup.set_move_readiness(lambda cue,ns: False)
    ready(sup,cfg,gate)
    sup.accept_event(event(1_060_000_000),now=1.06)
    step(sup,cfg,gate,1.10)
    ref.ack=True
    for i in range(1,10): step(sup,cfg,gate,1.10+.05*i)
    assert sup.recovering_expiry and not gate.stopped
    assert not any(c[1] == 1 for c in gate.calls)


def test_decision_must_still_be_fresh_when_first_enabled_packet_is_sent():
    cfg, gate, engine, ref, sup = setup()
    ready(sup,cfg,gate)
    sup.accept_event(event(950_000_000),now=1.06)
    step(sup,cfg,gate,1.10)
    ref.ack=True
    for t in (1.15,1.20): step(sup,cfg,gate,t)
    assert engine.state == 'MOVING' and not any(c[1] == 1 for c in gate.calls)
    step(sup,cfg,gate,1.30)
    assert sup.recovering_expiry and not gate.stopped
    assert engine.state == 'READY' and not any(c[1] == 1 for c in gate.calls)


def test_photo_waits_for_hold_and_joint_and_tcp_stillness():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    pending = event(1_060_000_000, cue='HOLD', tracking=False)
    pending.update(phase='SETTLING', reason='TARGET_PENDING_PHYSICAL_STOP')
    assert sup.accept_event(pending, now=1.06)
    for i in range(1, 12):
        step(sup, cfg, gate, 1.06 + .05 * i)
    assert not any(c[0] == 'photo_ready' for c in ref.calls)
    for i in range(12, 24):
        step(sup, cfg, gate, 1.06 + .05 * i)
        if sup.state == 'PHOTO_READY':
            break
    assert sup.state == 'PHOTO_READY'
    assert ('photo_ready', 1_060_000_000) in ref.calls
    assert not any(c[1] == 1 for c in gate.calls)


def test_photo_does_not_confirm_if_independent_tcp_is_moving():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    pending = event(1_060_000_000, cue='HOLD', tracking=False)
    pending.update(phase='SETTLING', reason='TARGET_PENDING_PHYSICAL_STOP')
    assert sup.accept_event(pending, now=1.06)
    for i in range(1, 22):
        t = 1.06 + .05 * i
        sent = gate.calls[-1]
        fb = feedback(cfg, 123, state=0, seq=sent[0])
        sup.tick(fb, sampled_at=t, joint_feedback={'joint_speed':[0.]*6,
                 'tcp_speed':[.001,0.,0.,0.,0.,0.]}, joint_sampled_at=t, now=t)
    assert sup.state == 'PHOTO_SETTLING'
    assert not any(c[0] == 'photo_ready' for c in ref.calls)


def test_photo_rejects_stop_before_steady_confirmation():
    cfg, gate, engine, ref, sup = setup()
    ready(sup, cfg, gate)
    pending = event(1_060_000_000, cue='HOLD', tracking=False)
    pending.update(phase='SETTLING', reason='TARGET_PENDING_PHYSICAL_STOP')
    assert sup.accept_event(pending, now=1.06)
    step(sup, cfg, gate, 1.11)
    sup.stop()
    with pytest.raises(RuntimeError): step(sup, cfg, gate, 1.16)
    assert not any(c[0] == 'photo_ready' for c in ref.calls)


def test_return_settling_resets_on_moving_joint_while_sample_is_pending():
    cfg, gate, engine, ref, sup = setup()
    class Delayed:
        def __init__(self, fn, *args):
            self.value = fn(*args)
            self.error = None
            self.done = False
    def worker(fn, *args):
        return Delayed(fn, *args) if fn.__name__ == 'sample' else Immediate(fn, *args)
    sup.worker_factory = worker
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    step(sup, cfg, gate, 1.10)
    ref.ack = True
    for t in (1.15, 1.20, 1.25): step(sup, cfg, gate, t)
    t = 1.25
    for _ in range(50):
        t += .05
        target = engine.goal if engine.state == 'MOVING' else None
        step(sup, cfg, gate, t, xyz=target if target is not None else cfg.neutral[:3])
        if sup.state == 'SAMPLE_PENDING': break
    assert sup.state == 'SAMPLE_PENDING'
    t += .05
    sent = gate.calls[-1]
    fb = feedback(cfg, 123, state=0, seq=sent[0])
    sup.tick(fb, sampled_at=t, joint_feedback={'joint_speed':[.01,0,0,0,0,0],
             'tcp_speed':[0.]*6}, joint_sampled_at=t, now=t)
    assert sup._settle_since is None
    assert not any(c[0] == 'release' for c in ref.calls)


def test_motion_while_release_is_pending_latches_fault():
    cfg, gate, engine, ref, sup = setup()
    class Delayed:
        def __init__(self, fn, *args):
            self.value = fn(*args)
            self.error = None
            self.done = False
    def worker(fn, *args):
        return Delayed(fn, *args) if fn.__name__ == 'release' else Immediate(fn, *args)
    sup.worker_factory = worker
    ready(sup, cfg, gate)
    sup.accept_event(event(1_060_000_000), now=1.06)
    step(sup, cfg, gate, 1.10)
    ref.ack = True
    for t in (1.15, 1.20, 1.25): step(sup, cfg, gate, t)
    t = 1.25
    for _ in range(75):
        t += .05
        target = engine.goal if engine.state == 'MOVING' else None
        step(sup, cfg, gate, t, xyz=target if target is not None else cfg.neutral[:3])
        if sup.state == 'RELEASE_PENDING': break
    assert sup.state == 'RELEASE_PENDING'
    t += .05
    sent = gate.calls[-1]
    fb = feedback(cfg, 123, state=0, seq=sent[0])
    with pytest.raises(RuntimeError, match='moved'):
        sup.tick(fb, sampled_at=t, joint_feedback={'joint_speed':[.01,0,0,0,0,0],
                 'tcp_speed':[0.]*6}, joint_sampled_at=t, now=t)
    assert gate.stopped


def test_local_reference_client_real_guest_rejects_old_epoch_after_stop():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self): return True
        def fresh(self): return True
        def stream_fresh(self): return True
        def diagnostics(self): return {}
    with GuestDisplay(port=0, remote_capture=Capture(), physical_feedback=True) as display:
        with request(display, '/action', 'POST', {'action':'start'}): pass
        snap = display.control_snapshot()
        with request(display, '/physical-feedback', 'POST', {'op':'register',
                     'instance':snap['instance'],'epoch':snap['epoch']}) as response:
            import json
            lease = json.load(response)['lease']
        ref = LocalReferenceGate(urlsplit(display.origin).port, display.token, timeout=.5,
                                 control=snap, lease=lease)
        neutral = [0.] * 6
        session = ref.begin(neutral)
        status=ref.status(session)
        assert {key:status[key] for key in ('blocked','acknowledged','session')} == {
            'blocked': True, 'acknowledged': False, 'session': session}
        assert status['reference_check']['status']=='no_sample'
        with display._lock:
            display._reference_gate.acknowledge()
        assert ref.status(session)['acknowledged']
        with request(display, '/action', 'POST', {'action':'stop'}): pass
        with pytest.raises(RuntimeError): ref.sample(session, neutral, neutral, time.monotonic())
        with display._lock: assert display._reference_gate.blocked


def test_local_lease_fault_rejects_late_release_and_keeps_gate_held():
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        def connected(self): return True
        def fresh(self): return True
        def stream_fresh(self): return True
        def diagnostics(self): return {}
    with GuestDisplay(port=0, remote_capture=Capture(), physical_feedback=True) as display:
        with request(display, '/action', 'POST', {'action':'start'}): pass
        snap = display.control_snapshot()
        with request(display, '/physical-feedback', 'POST', {'op':'register',
                     'instance':snap['instance'],'epoch':snap['epoch']}) as response:
            import json
            lease = json.load(response)['lease']
        ref = LocalReferenceGate(urlsplit(display.origin).port, display.token, timeout=.5,
                                 control=snap, lease=lease)
        session = ref.begin([0.] * 6)
        with display._lock: display._reference_gate.acknowledge()
        assert ref.fault()
        with pytest.raises(RuntimeError): ref.release(session)
        with display._lock:
            assert display._active and display._reference_gate.blocked


def test_local_fault_revokes_lease_even_if_camera_has_gone_stale():
    import json
    from leica_tracker.guest_display import GuestDisplay
    class Capture:
        live = True
        def connected(self): return self.live
        def fresh(self): return self.live
        def stream_fresh(self): return self.live
        def diagnostics(self): return {}
    capture = Capture()
    with GuestDisplay(port=0, remote_capture=capture, physical_feedback=True) as display:
        with request(display, '/action', 'POST', {'action':'start'}): pass
        snap = display.control_snapshot()
        with request(display, '/physical-feedback', 'POST', {'op':'register',
                     'instance':snap['instance'],'epoch':snap['epoch']}) as response:
            lease = json.load(response)['lease']
        ref = LocalReferenceGate(urlsplit(display.origin).port, display.token, timeout=.5,
                                 control=snap, lease=lease)
        session = ref.begin([0.] * 6)
        capture.live = False
        assert ref.fault()
        capture.live = True
        with pytest.raises(RuntimeError): ref.release(session)
        with display._lock:
            assert display._active and display._physical_revoked and display._reference_gate.blocked


@pytest.mark.parametrize('interrupt_during_motion', [False, True])
def test_actual_guest_http_fake_rtde_full_encounter_and_stop(interrupt_during_motion):
    import json
    import threading
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.person_guidance import GuidanceCue, GuidanceDecision
    from leica_tracker.resident_guest_watchdog import (GuestControlWatchdog,
        LocalGuestControlReader, ResidentMotionGate)
    class Capture:
        def connected(self): return True
        def fresh(self): return True
        def stream_fresh(self): return True
        def diagnostics(self): return {}
    class Channel:
        def __init__(self): self.calls=[]; self.closed=threading.Event()
        def send(self,*args): self.calls.append(args)
        def close(self): self.closed.set()
    with GuestDisplay(port=0, remote_capture=Capture(), physical_feedback=True) as display:
        with request(display, '/action', 'POST', {'action':'start'}): pass
        identity = display.control_snapshot()
        with request(display, '/physical-feedback', 'POST', {'op':'register',
                     'instance':identity['instance'],'epoch':identity['epoch']}) as response:
            lease = json.load(response)['lease']
        channel = Channel(); gate = ResidentMotionGate(channel)
        reader = LocalGuestControlReader(urlsplit(display.origin).port, display.token, timeout=.2)
        watchdog = GuestControlWatchdog(reader, gate, interval=.02, timeout=.3)
        watchdog.start()
        assert watchdog.wait_ready(1)
        ref = LocalReferenceGate(urlsplit(display.origin).port, display.token, timeout=.2,
                                 control=identity, lease=lease)
        cfg = Installation(reviewed()); engine = ResidentEngine(cfg, gate, 123)
        paths = {cue: [[.105, -.5, .3]] for cue in ('MOVE_LEFT','MOVE_RIGHT','MOVE_UP',
                 'MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER')}
        sup = PhysicalInteractionSupervisor(engine, ref, paths, reviewed_paths=paths,
            hold_timeout=2., action_timeout=10., gate_timeout=.25, max_joint_speed=.001)
        sup.bind_control_identity({'instance':identity['instance'],'epoch':identity['epoch'],
                                   'started_ns':identity['control_started_ns']})
        sup.set_move_readiness(lambda cue,source_ns: True)
        stopped = threading.Event()
        def heartbeat():
            sequence=0
            while not stopped.is_set():
                try:
                    sequence+=1
                    with request(display,'/control-heartbeat','POST',{
                         'instance':identity['instance'],'epoch':identity['epoch'],'sequence':sequence}): pass
                    with request(display,'/physical-feedback','POST',{'op':'heartbeat',
                         'instance':identity['instance'],'epoch':identity['epoch'],'lease':lease}): pass
                    with request(display,'/state'): pass
                except Exception: return
                stopped.wait(.06)
        thread = threading.Thread(target=heartbeat,daemon=True); thread.start()
        hold = GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,
            (.5,.5,.2),None,None,False,'PHYSICAL_REFERENCE_HOLD')
        def tick():
            now = time.monotonic()
            while sup._last_joint_sample is not None and now <= sup._last_joint_sample:
                time.sleep(.002)
                now = time.monotonic()
            sent = channel.calls[-1] if channel.calls else None
            xyz = engine.goal if engine.state == 'MOVING' else cfg.neutral[:3]
            fb = feedback(cfg,123,state=sent[1] if sent else 2,seq=sent[0] if sent else 0,xyz=xyz)
            sup.tick(fb,sampled_at=now,joint_feedback={'joint_speed':[0.]*6,
                     'tcp_speed':[0.]*6},joint_sampled_at=now,now=now)
        try:
            tick(); time.sleep(.03); tick()
            assert engine.state == 'READY'
            now = time.monotonic()
            direction = event(time.monotonic_ns())
            direction.update(control_instance=identity['instance'],control_epoch=identity['epoch'])
            assert sup.accept_event(direction,now=time.monotonic())
            deadline = time.monotonic()+5
            saw_blocked = False
            while time.monotonic()<deadline:
                with display._lock: blocked = display._reference_gate.blocked
                if blocked:
                    saw_blocked = True
                    display.publish(None,hold)
                tick()
                if interrupt_during_motion and engine.state == 'MOVING':
                    with request(display,'/action','POST',{'action':'stop'}): pass
                    assert watchdog.stopped.wait(1) and channel.closed.wait(1)
                    count = len(channel.calls)
                    with pytest.raises(RuntimeError): tick()
                    assert len(channel.calls) == count
                    return
                if saw_blocked and sup.state == 'OBSERVING': break
                time.sleep(.03)
            assert saw_blocked and sup.state == 'OBSERVING'
            assert any(c[1]==1 and tuple(c[3])==(.105,-.5,.3) for c in channel.calls)
            assert any(c[1]==1 and tuple(c[3])==tuple(cfg.neutral[:3]) for c in channel.calls)
            with display._lock: assert not display._reference_gate.blocked
            time.sleep(.03)  # 第一个符合条件的框架必须在发布后才可用。
            photo = time.monotonic_ns(); display.arm_physical_photo(photo)
            pending = event(photo,cue='HOLD',tracking=False)
            pending.update(phase='SETTLING',reason='TARGET_PENDING_PHYSICAL_STOP',
                           control_instance=identity['instance'],control_epoch=identity['epoch'])
            assert sup.accept_event(pending,now=time.monotonic())
            deadline = time.monotonic()+2
            while time.monotonic()<deadline and sup.state!='PHOTO_READY':
                tick(); time.sleep(.03)
            assert sup.state == 'PHOTO_READY' and display.physical_photo_ready(photo)
            with request(display,'/action','POST',{'action':'stop'}): pass
            assert watchdog.stopped.wait(1) and channel.closed.wait(1)
            with pytest.raises(RuntimeError): gate.send(999,1,123,[0,0,0])
        finally:
            stopped.set(); thread.join(1); watchdog.close()
