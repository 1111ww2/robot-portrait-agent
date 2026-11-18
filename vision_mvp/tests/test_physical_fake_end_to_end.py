"""将纯粹的策略连接到假时钟/电测仪，永远不要连接到硬件。测试布线仅说明预期的顺序。
它不是应用程序适配器，不会验证输入，也不会发出可执行动作。"""
import pytest

from leica_tracker.offline_portrait_supervisor import OfflinePortraitSupervisor


REFERENCE = [-.1, -.5, .2, 0., 3., -.5]
ZERO = [0.] * 6
CONTROL = {'instance': 'fake-display', 'epoch': 1}
OWNER = {'owner': 'fake-owner', 'epoch': 1, 'active': True}
EPOCH = 1
NS = 1_000_000_000


class FakeEncounter:
    def __init__(self):
        self.time = 10.
        self.supervisor = OfflinePortraitSupervisor(hold_timeout=2., motion_timeout=2.)
        self.supervisor.start(REFERENCE, ZERO, sampled_at=self.time,
                              control=CONTROL, active_owner=OWNER, now=self.time, epoch=EPOCH)
        self.latest_frame = 0

    def tick(self, pose=REFERENCE, speed=ZERO):
        self.time = round(self.time + .1, 3)
        self.supervisor.observe_control(CONTROL, active_owner=OWNER, now=self.time)
        self.supervisor.observe_telemetry(pose, speed, sampled_at=self.time,
                                         now=self.time)

    def frame(self, *, source_ns=None, tracking=True, within=True):
        now_ns = round(self.time * NS)
        source_ns = now_ns if source_ns is None else source_ns
        self.latest_frame = source_ns
        return self.supervisor.observe_frame(source_ns, now=self.time,
                                             tracking=tracking, within=within)

    def issue(self):
        self.supervisor.acknowledge_hold(now=self.time)
        return self.supervisor.intent(now=self.time).cue

    def complete_return(self):
        self.supervisor.motion_finished(success=True, now=self.time)
        for _ in range(8):
            self.tick()
        return self.supervisor.release(now=self.time)

    def direction(self, cue):
        self.supervisor.request_direction(cue, source_ns=self.latest_frame, now=self.time)

    def stop(self):
        self.supervisor.stop()


def test_fake_opening_direction_arrival_departure_and_stop():
    e = FakeEncounter()
    assert not e.frame()
    assert e.issue() == 'OPENING'
    # 在运动完成前，初始参考观察不能释放。
    for _ in range(8):
        e.tick()
    with pytest.raises(ValueError):
        e.supervisor.release(now=e.time)
    assert not e.frame()
    first_release = e.complete_return()
    assert not e.frame(source_ns=first_release)  # 捕获必须在发布后进行
    with pytest.raises(ValueError):
        e.direction('MOVE_LEFT')
    e.tick()
    assert not e.frame(within=False)  # 新框架可能会引导，但无法拍照
    e.direction('MOVE_LEFT')
    assert e.issue() == 'MOVE_LEFT'
    assert not e.frame()  # 在移动时不能使用旧或当前的框架
    old_frame = round(e.time * NS)
    second_release = e.complete_return()
    assert not e.frame(source_ns=old_frame)
    assert not e.frame(source_ns=second_release)
    with pytest.raises(ValueError):
        e.direction('MOVE_RIGHT')

    # 每100毫秒采集一次新鲜样本；只有连续两秒钟才能触发锁定。
    for _ in range(20):
        e.tick()
        assert not e.frame()
    e.tick()
    assert e.frame()
    assert e.supervisor.photo_locked
    e.tick()
    assert e.frame(tracking=False, within=False)  # 出发不会取消照片
    with pytest.raises(ValueError):
        e.direction('MOVE_RIGHT')
    assert e.supervisor.state == 'OBSERVING'
    e.stop()
    assert not e.supervisor.photo_locked
    assert not e.frame()
    with pytest.raises(RuntimeError):
        e.direction('MOVE_RIGHT')


def test_fake_failure_cannot_release_or_take_photo():
    e = FakeEncounter()
    assert e.issue() == 'OPENING'
    e.supervisor.motion_finished(success=False, now=e.time)
    with pytest.raises(RuntimeError):
        e.supervisor.release(now=e.time)
    assert not e.frame()
    assert not e.supervisor.photo_locked


def test_direction_requires_latest_tracked_frame_and_cannot_replay():
    e = FakeEncounter()
    assert e.issue() == 'OPENING'
    e.complete_return()
    e.tick()
    assert not e.frame(tracking=False, within=False)
    with pytest.raises(ValueError): e.direction('MOVE_LEFT')

    e.tick()
    assert not e.frame(within=False)
    earlier_frame = e.latest_frame
    e.tick()
    assert not e.frame(tracking=False, within=False)
    with pytest.raises(ValueError):
        e.supervisor.request_direction('MOVE_LEFT', source_ns=earlier_frame, now=e.time)
    with pytest.raises(ValueError): e.direction('MOVE_LEFT')

    e.tick()
    assert not e.frame(within=False)
    recovered_frame = e.latest_frame
    with pytest.raises(ValueError):
        e.supervisor.request_direction('MOVE_RIGHT', source_ns=earlier_frame, now=e.time)
    e.direction('MOVE_RIGHT')
    assert e.issue() == 'MOVE_RIGHT'
    with pytest.raises(ValueError):
        e.supervisor.request_direction('MOVE_RIGHT', source_ns=recovered_frame, now=e.time)


def test_inside_target_suppresses_direction_without_resetting_photo_dwell():
    e = FakeEncounter()
    assert e.issue() == 'OPENING'
    e.complete_return()
    for _ in range(20):
        e.tick()
        assert not e.frame(tracking=True, within=True)
        with pytest.raises(ValueError):
            e.direction('MOVE_LEFT')
    e.tick()
    assert e.frame(tracking=True, within=True)
    assert e.supervisor.photo_locked


@pytest.mark.parametrize('failure', ['stop', 'disconnect', 'heartbeat', 'telemetry', 'owner'])
def test_failure_after_photo_lock_clears_policy_and_remains_closed(failure):
    e = FakeEncounter()
    assert e.issue() == 'OPENING'
    e.complete_return()
    for _ in range(21):
        e.tick()
        e.frame()
    assert e.supervisor.photo_locked
    if failure == 'stop':
        e.supervisor.stop()
    elif failure == 'disconnect':
        e.supervisor.disconnect()
    elif failure == 'heartbeat':
        with pytest.raises(RuntimeError):
            e.supervisor.watchdog(now=e.time + .4)
    elif failure == 'telemetry':
        with pytest.raises(ValueError):
            e.supervisor.observe_telemetry(REFERENCE, ZERO, sampled_at=e.time - 2, now=e.time)
    else:
        with pytest.raises(RuntimeError):
            e.supervisor.observe_control(CONTROL, active_owner={**OWNER, 'active': False}, now=e.time)
    assert not e.supervisor.photo_locked
    assert not e.frame()
    with pytest.raises(RuntimeError):
        e.direction('MOVE_LEFT')
