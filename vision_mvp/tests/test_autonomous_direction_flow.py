"""通过真实的iPad后端进行离线UR5e/UR10e提示接受。仅使用IPv4循环HTTP。
RTDE遥测和相机新鲜度是假的；所有型号配置文件都共享合成数字测试夹具，而不是物理校准。
这些检查不会建立物理方向标志、校准或停止。"""
import copy
import json
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from leica_tracker.app import PhysicalPhotoLatch
from leica_tracker.guest_display import GuestDisplay
from leica_tracker.person_guidance import GuidanceCue, GuidanceDecision
from leica_tracker.physical_intro_program import IntroProgram
from leica_tracker.physical_resident_engine import ResidentEngine
from leica_tracker.physical_runtime import FeedbackPublisher, PhysicalRuntime, register_backend, reviewed_profile
from leica_tracker.resident_guest_watchdog import GuestControlWatchdog, LocalGuestControlReader, ResidentMotionGate
from test_guest_display import request
from test_physical_runtime import Dashboard, Session, data


# 手动从合成测试安装中导出，而不是从生产路径构建器中导出。
# 错误的提示查找、丢失的返回、早期观察释放或没有新鲜静止状态的照片确认必须无法通过这些测试。
# 这些相同的数字同时适用于软件模型分支；它们不是物理UR10e安装或方向校准的转移到UR5e。
DIRECTIONS = (
    ("MOVE_LEFT", (.095, -.5, .3)),
    ("MOVE_RIGHT", (.105, -.5, .3)),
    ("MOVE_UP", (.1, -.5, .305)),
    ("MOVE_DOWN", (.1, -.5, .295)),
    ("MOVE_CLOSER", (.1, -.505, .3)),
    ("MOVE_FARTHER", (.1, -.495, .3)),
)
REFERENCE = (.1, -.5, .3)


class FreshCamera:
    available=True
    def connected(self): return self.available
    def fresh(self): return self.available
    def stream_fresh(self): return self.available
    def diagnostics(self): return {}


class FakeRTDE:
    """理想的静态端点工厂；仅确认实际发送的数据包。"""
    def __init__(self):
        self.calls = []
        self.receipts = {}
        self.closed = threading.Event()

    def send(self, *packet):
        assert not self.closed.is_set()
        self.calls.append(packet)
        # 这种理想的植物立即消耗命令。像真正的传输那样，独立于后来的GUI/测试循环查询，保留那个ACK时间；
        # 缓慢的测试查询不是晚期的控制器ACK。
        self.receipts[packet[0]] = time.monotonic()

    def close(self):
        self.closed.set()


@pytest.fixture(params=('ur5e', 'ur10e'))
def synthetic_model_profile(request):
    """明确地绑定每个假模型，而不需要借用物理接受。"""
    candidate = data()
    for config in (candidate, candidate['intro'], candidate['installation'], candidate['intro_preview']):
        config.update(robot_model=request.param, reviewed_robot_model=request.param)
    candidate['reviewed_installation'] = copy.deepcopy(candidate['installation'])
    # 这个螺纹环回流检查排序/所有权，而不是RPC到期。匹配两个选定的模型配置文件的250毫秒门限预算；
    # 确定性监督器测试保留单独的100毫秒到期合同。
    candidate['gate_timeout']=candidate['reviewed_runtime']['gate_timeout']=.25
    # 动态审查仍然独立地与data（）的假参数绑定；
    # 模型审查存在于其自己的reviewed_robot_model字段中。
    profile = reviewed_profile(candidate)
    assert profile.cfg.robot_model == request.param
    assert IntroProgram(profile.intro, 123).robot_model == request.param
    with pytest.raises(AttributeError):
        profile.cfg.robot_model = 'ur10e' if request.param == 'ur5e' else 'ur5e'
    return profile


@pytest.fixture
def autonomous_flow(synthetic_model_profile):
    profile = synthetic_model_profile
    cfg = profile.cfg
    with GuestDisplay(port=0, remote_capture=FreshCamera(), physical_feedback=True) as display:
        initial = display.control_snapshot()
        with request(display, '/action', 'POST', {
            'action': 'start', 'control_protocol': 1,
            'control': {key: initial[key] for key in ('instance', 'epoch')},
        }):
            pass
        identity = display.control_snapshot()
        control = {key: identity[key] for key in ('instance', 'epoch')}
        control['started_ns'] = identity['control_started_ns']
        visitor_done = threading.Event()

        def owner_heartbeat():
            sequence = 0
            while not visitor_done.is_set():
                sequence += 1
                try:
                    with request(display, '/control-heartbeat', 'POST', {
                        'instance': control['instance'], 'epoch': control['epoch'], 'sequence': sequence,
                    }):
                        pass
                except Exception:
                    return  # 停止会使遭遇和它的心跳失效。
                visitor_done.wait(.05)

        visitor = threading.Thread(target=owner_heartbeat, daemon=True)
        visitor.start()
        channel = FakeRTDE()
        gate = ResidentMotionGate(channel)
        port = urlsplit(display.origin).port
        reader = LocalGuestControlReader(port, display.token, timeout=.2)
        watchdog = GuestControlWatchdog(reader, gate, interval=.02, timeout=.3,require_camera=False)
        watchdog.start()
        publisher = None
        try:
            assert watchdog.wait_ready(1.)
            reference = register_backend(port, display.token, control, timeout=.2)
            publisher = FeedbackPublisher(reference, gate)
            engine = ResidentEngine(cfg, gate, 123)
            supervisor = profile.supervisor(engine, reference)
            session = Session(cfg)
            session.engine, session.gate = engine, gate
            dashboard = Dashboard()
            runtime = PhysicalRuntime(session, supervisor, profile, dashboard, publisher, control=control)
            hold = GuidanceDecision('PHYSICAL_REFERENCE_HOLD', GuidanceCue.HOLD,
                                    (.5, .5, .2), None, None, False, 'PHYSICAL_REFERENCE_HOLD')
            stages, lights = [], []

            def tick(event=None):
                # 视觉循环承认实际观察状态。它既不会启动另一个运动，也不会提供返回命令。
                if display.reference_held(time.monotonic_ns()):
                    display.publish(None, hold)
                sent = channel.calls[-1] if channel.calls else None
                xyz = list(sent[3]) if sent else list(REFERENCE)
                now = time.monotonic()
                sample = {'session': 123, 'state': sent[1] if sent else 2,
                          'sequence': sent[0] if sent else 0,
                          'tcp_pose': xyz + list(cfg.neutral[3:]),
                          'tcp_speed': [0.] * 6, 'joints': list(cfg.joints)}
                if sent is not None:
                    sample.update(ack_sampled_at=channel.receipts[sent[0]],worker_sampled_at=now)
                session.writer.latest = (sample, now)
                session.reader.latest = ({'joint_speed': [0.] * 6, 'tcp_speed': [0.] * 6}, now)
                session.states.latest = (session.states.latest[0], now)
                dashboard.checked_at = now
                before = len(channel.calls)
                runtime.step(event, now=now)
                stages.append(supervisor.state)
                for packet in channel.calls[before:]:
                    if packet[1] == 1:
                        with display._lock:
                            assert display._reference_gate.blocked
                            assert display._reference_gate.acknowledged
                with display._lock:
                    lights.append(dict(display._current_feedback()))

            def until(predicate, timeout=6.):
                deadline = time.monotonic() + timeout
                while not predicate() and time.monotonic() < deadline:
                    tick()
                    time.sleep(.03)
                assert predicate(), (supervisor.state, engine.state, stages)

            def event(cue, **fields):
                return {'event': 'person_guidance', 'coordinate_frame': 'normalized_camera_image',
                        'robot_actuated': False, 'tracking': True, 'cue': cue,
                        'source_monotonic_ns': time.monotonic_ns(),
                        'control_instance': control['instance'], 'control_epoch': control['epoch'], **fields}

            until(lambda: engine.state == 'READY')
            yield SimpleNamespace(display=display, channel=channel, gate=gate, watchdog=watchdog,
                                  engine=engine, supervisor=supervisor, session=session,
                                  tick=tick, until=until, event=event, stages=stages, lights=lights)
        finally:
            visitor_done.set()
            visitor.join(1.)
            if publisher is not None:
                publisher.close()
            watchdog.close()


@pytest.mark.parametrize('cue,expected_target', DIRECTIONS, ids=[cue for cue, _ in DIRECTIONS])
def test_one_cue_automatically_returns_reopens_observation_and_enables_photo(autonomous_flow, cue, expected_target):
    flow = autonomous_flow
    source = flow.event(cue)
    flow.tick(source)
    assert flow.supervisor.state == 'BEGIN_PENDING'
    flow.until(lambda: flow.engine.completed == 2 and flow.supervisor.state == 'OBSERVING')

    targets = []
    for _, enabled, _, target in flow.channel.calls:
        if enabled and (not targets or target != targets[-1]):
            targets.append(target)
    assert len(targets) == 2
    assert targets[0] == pytest.approx(expected_target)
    assert targets[1] == pytest.approx(REFERENCE)
    assert flow.channel.calls[0][1] == flow.channel.calls[-1][1] == 0
    assert flow.session.writer.latest[0]['tcp_pose'][:3] == pytest.approx(REFERENCE)
    assert flow.session.reader.latest[0]['joint_speed'] == [0.] * 6
    assert any(light.get('intensity', 0) > 0 for light in flow.lights)
    assert {'BEGIN_PENDING', 'WAIT_ACK', 'MOVING', 'SAMPLE_PENDING', 'RELEASE_PENDING', 'OBSERVING'} <= set(flow.stages)
    assert not flow.display.reference_held(time.monotonic_ns())
    assert flow.display.reference_held(source['source_monotonic_ns'])

    # 到达仍处于等待状态，直到真正的后端观察到另一个新的静止间隔。没有呼叫者发布photo_ready或释放门。
    latch = PhysicalPhotoLatch(flow.display)
    arrived = GuidanceDecision('ARRIVED', GuidanceCue.ARRIVED, (.5, .5, .2),
                               None, (.5, .5, .2), True, 'TARGET_REACHED')
    photo_source = time.monotonic_ns()
    pending = latch.apply(arrived, active=True, source_ns=photo_source)
    assert (pending.phase, pending.reason) == ('SETTLING', 'TARGET_PENDING_PHYSICAL_STOP')
    assert not flow.display.physical_photo_ready(photo_source)
    flow.tick(flow.event('HOLD', tracking=False, phase=pending.phase, reason=pending.reason))
    assert flow.supervisor.state == 'PHOTO_LOOKUP'
    flow.until(lambda: flow.supervisor.state == 'PHOTO_READY')
    assert {'PHOTO_LOOKUP', 'PHOTO_SETTLING', 'PHOTO_PENDING', 'PHOTO_READY'} <= set(flow.stages)
    final = latch.apply(arrived, active=True, source_ns=time.monotonic_ns())
    assert (final.phase, final.reason) == ('ARRIVED', 'TARGET_LATCHED')
    flow.display.publish(None, final)
    with request(flow.display, '/state') as response:
        assert json.load(response)['photo_ready'] is True
    assert flow.engine.completed == 2


@pytest.mark.parametrize('cue,expected_target', DIRECTIONS, ids=[cue for cue, _ in DIRECTIONS])
def test_ipad_stop_during_each_cue_closes_writer_without_return_or_release(autonomous_flow, cue, expected_target):
    flow = autonomous_flow
    flow.tick(flow.event(cue))
    flow.until(lambda: any(packet[1] == 1 for packet in flow.channel.calls))
    assert flow.channel.calls[-1][3] == pytest.approx(expected_target)
    flow.display.remote_capture.available=False
    with request(flow.display, '/action', 'POST', {'action': 'stop'}):
        pass
    assert flow.watchdog.stopped.wait(.5)
    assert flow.channel.closed.wait(.5)
    assert not flow.gate.can_send
    count = len(flow.channel.calls)
    with pytest.raises(RuntimeError, match='watchdog stopped'):
        flow.tick(flow.event(cue))
    assert len(flow.channel.calls) == count
    assert flow.engine.completed == 0
    assert flow.supervisor.state == 'STOPPED'
    assert flow.display.reference_held(time.monotonic_ns())
    assert all(packet[3] == pytest.approx(expected_target)
               for packet in flow.channel.calls if packet[1] == 1)


def test_camera_gap_during_accepted_finite_motion_still_returns_and_keeps_ownership(autonomous_flow):
    flow=autonomous_flow
    flow.tick(flow.event('MOVE_RIGHT'))
    flow.until(lambda:any(packet[1]==1 for packet in flow.channel.calls))
    flow.display.remote_capture.available=False
    flow.until(lambda:flow.engine.completed==2 and flow.supervisor.state=='OBSERVING')
    enabled=[packet[3] for packet in flow.channel.calls if packet[1]]
    assert enabled[0]==pytest.approx((.105,-.5,.3)) and enabled[-1]==pytest.approx(REFERENCE)
    assert flow.gate.can_send and flow.display.physical_backend_available()
    assert not flow.display._physical_revoked and not flow.display.control_snapshot()['camera_fresh']


def test_idle_camera_gap_waits_without_enabling_stale_visual_decisions_or_photos(autonomous_flow):
    flow=autonomous_flow;old=flow.event('MOVE_RIGHT')
    flow.display.remote_capture.available=False
    deadline=time.monotonic()+.4
    while time.monotonic()<deadline:
        flow.tick();time.sleep(.03)
    flow.tick(old)
    assert flow.supervisor.state=='OBSERVING' and flow.gate.can_send
    assert flow.display.physical_backend_available() and not flow.display._physical_revoked
    assert not any(packet[1] for packet in flow.channel.calls)
    with pytest.raises(ValueError):flow.display.arm_physical_photo(old['source_monotonic_ns'])
