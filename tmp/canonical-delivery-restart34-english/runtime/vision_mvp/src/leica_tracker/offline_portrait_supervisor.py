"""仅用内存连接实验组件，不做输入输出、不连接设备或传输通道，也不生成轨迹。该模块未接入应用，不构成实体安全系统。
输入、时间戳和 HOLD 确认均由测试调用方提供，此处无法验证其真实性。
所有时间须来自同一单调时钟，底层姿态测试尚未在实体安装环境验证。"""
from .physical_interaction_coordinator import PhysicalInteractionCoordinator
from .physical_portrait_policy import PhysicalPortraitPolicy


class OfflinePortraitSupervisor:
    """对通过依赖注入提供的单个模拟会话进行协调；出错时保持禁止执行。"""

    def __init__(self, *, hold_timeout, motion_timeout):
        self._coordinator = PhysicalInteractionCoordinator(hold_timeout=hold_timeout,
                                                           motion_timeout=motion_timeout)
        self._portrait = PhysicalPortraitPolicy()
        self._epoch = None
        self._decision_frame = None

    @property
    def state(self):
        return self._coordinator.state

    @property
    def photo_locked(self):
        return self._portrait.latched

    def _clear_if_faulted(self):
        if self._coordinator.state in ('FAULT', 'STOPPED') and self._epoch is not None:
            self._portrait.stop(self._epoch)
            self._decision_frame = None

    def _call(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        finally:
            self._clear_if_faulted()

    def _inconsistent(self):
        self._coordinator.source_lost()
        self._clear_if_faulted()
        raise RuntimeError('Policy coordination failed; explicit new encounter required')

    def start(self, reference, speed, *, sampled_at, control, active_owner, now, epoch):
        if self._coordinator.state != 'NEW' or self._epoch is not None:
            raise RuntimeError('Explicit new supervisor required')
        if type(epoch) is not int or epoch <= 0:
            raise ValueError('A positive encounter epoch is required')
        if not isinstance(control, dict) or control.get('epoch') != epoch:
            raise ValueError('Encounter and control epochs must match')
        self._epoch = epoch
        self._portrait.start(epoch)
        self._call(self._coordinator.start, reference, speed, sampled_at=sampled_at,
                   control=control, active_owner=active_owner, now=now)

    def observe_control(self, control, *, active_owner, now):
        return self._call(self._coordinator.observe_control, control,
                          active_owner=active_owner, now=now)

    def observe_telemetry(self, pose, speed, *, sampled_at, now):
        return self._call(self._coordinator.observe_telemetry, pose, speed,
                          sampled_at=sampled_at, now=now)

    def watchdog(self, *, now):
        return self._call(self._coordinator.watchdog, now=now)

    def acknowledge_hold(self, *, now):
        return self._call(self._coordinator.acknowledge_hold, now=now)

    def intent(self, *, now):
        return self._call(self._coordinator.intent, now=now)

    def motion_finished(self, *, success, now):
        return self._call(self._coordinator.motion_finished, success=success, now=now)

    def release(self, *, now):
        # 只有协调器成功放行后，才能提供该次放行的时间戳。
        stamp = self._call(self._coordinator.release, now=now)
        try:
            self._portrait.release_observation(self._epoch, stamp)
        except (ValueError, RuntimeError):
            self._inconsistent()
        return stamp

    def observe_frame(self, source_ns, *, now, tracking, within):
        # 只有本次采集通过同一检查才更新策略。此后的任意帧，包括无效帧，都会使旧提示失效。
        self._decision_frame = None
        if not self._call(self._coordinator.accept_frame, source_ns, now=now):
            return False
        result = self._portrait.observe(self._epoch, source_ns=source_ns,
                                        now_ns=round(now * 1e9),
                                        tracking=tracking, within=within)
        if tracking is True and within is False and not self._portrait.latched:
            self._decision_frame = source_ns
        return result

    def request_direction(self, cue, *, source_ns, now):
        if self.state in ('NEW', 'FAULT', 'STOPPED'):
            raise RuntimeError('Encounter not active')
        if self._portrait.latched:
            raise ValueError('Photo is locked; direction is unavailable')
        if type(source_ns) is not int or source_ns != self._decision_frame:
            raise ValueError('Direction must come from the latest tracked frame')
        self._call(self._coordinator.request_direction, cue, now=now)
        self._decision_frame = None
        if not self._portrait.begin_motion(self._epoch):
            self._inconsistent()

    def disconnect(self):
        self._coordinator.source_lost()
        self._clear_if_faulted()

    def stop(self):
        self._coordinator.stop()
        self._clear_if_faulted()
