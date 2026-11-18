"""对照生成的 URScript 做离线状态模型检查，不执行实体程序。
测试核对分支顺序，并展示实机停止测试须测量的在途数据包时间窗，不能据此推断控制器上的停止耗时。"""
import re

from leica_tracker.physical_resident_engine import Installation
from leica_tracker.physical_step_preview import SharedPreviewPlan, StepProgram
from test_physical_resident_engine import reviewed
from test_physical_shared_preview import reviewed as reviewed_preview, measured


def _rendered_xyz_guards():
    script = Installation(reviewed()).program(123)
    start = script.index('    if packet_age > 0.25:', script.index('  ever_armed = False'))
    end = script.index('    seq = read_input_integer_register(42)', start)
    block = script[start:end]
    assert 'if ever_armed:' in block and 'stopl(' in block
    assert 'write_output_integer_register(42, 2)' in block
    assert block.count('break') == 2
    assert block.index('break') < block.index('if not ever_armed and startup_age >')
    assert 'global startup_age = startup_age + get_steptime()' in script
    startup = float(re.search(r'if not ever_armed and startup_age > ([0-9.]+):', block).group(1))
    heartbeat = float(re.search(r'if packet_age > ([0-9.]+):', block).group(1))
    assert 'movej(' not in script and 'movel(' not in script
    assert 'grip' not in script.lower() and 'set_tool' not in script.lower()
    return heartbeat, startup


class _ScriptOrderModel:
    """最小执行顺序模型：先判断超时，再处理输入。"""
    def __init__(self, heartbeat_timeout, startup_timeout):
        self.heartbeat_timeout = heartbeat_timeout
        self.startup_timeout = startup_timeout
        self.armed = False
        self.ever_armed = False
        self.moving = False
        self.exited = False
        self.last_packet = None
        self.trace = []

    def cycle(self, now, packet=None):
        if self.exited:
            self.trace.append('discard_after_exit')
            return
        if self.ever_armed and self.last_packet is not None and now-self.last_packet > self.heartbeat_timeout:
            if self.moving: self.trace.append('stopl')
            self.trace.append('exit')
            self.exited = True
            return
        if not self.ever_armed and now > self.startup_timeout:
            self.trace.extend(('stopl', 'exit'))
            self.exited = True
            return
        if packet == 0:
            self.armed = self.ever_armed = True
            self.moving = False
            self.last_packet = now
        elif packet == 1 and self.armed:
            self.moving = True
            self.last_packet = now
            self.trace.append('enabled')


def test_rendered_resident_exits_before_reading_late_same_session_packets():
    timeout, startup = _rendered_xyz_guards()
    robot = _ScriptOrderModel(timeout, startup)
    robot.cycle(0., 0); robot.cycle(.02, 1)
    robot.cycle(.02 + timeout + .01, 0)
    robot.cycle(.02 + timeout + .02, 1)
    assert robot.exited and robot.trace == ['enabled', 'stopl', 'exit', 'discard_after_exit']


def test_unarmed_resident_exits_on_bounded_startup_timeout_before_late_packet():
    timeout, startup = _rendered_xyz_guards()
    robot = _ScriptOrderModel(timeout, startup)
    robot.cycle(startup + .01, 0)
    robot.cycle(startup + .02, 1)
    assert robot.exited and robot.trace == ['stopl', 'exit', 'discard_after_exit']


def test_in_flight_packet_before_timeout_can_extend_stop_window():
    timeout, startup = _rendered_xyz_guards()
    robot = _ScriptOrderModel(timeout, startup)
    robot.cycle(0., 0); robot.cycle(.02, 1)
    # 在 0.03 时停止。一条先前排队的使能请求仍可能在常驻程序发现超时前抵达，被接纳并重置心跳期限。
    robot.cycle(.04, 1)
    robot.cycle(.02 + timeout + .001)
    assert not robot.exited
    robot.cycle(.04 + timeout + .01)
    assert robot.exited and robot.trace[-2:] == ['stopl', 'exit']


def test_reviewed_preview_stop_branch_has_no_return_or_tool_commands():
    draft, values = reviewed_preview()
    plan = SharedPreviewPlan(draft, values)
    script = StepProgram(plan.step_profile(0, [], measured(values), now=10.), 1).program()
    assert 'packet_age > 0.2500000000' in script
    fault_branch = script.split('    if fault != 0:', 1)[1].split('    if valid and not started:', 1)[0]
    assert 'kill motion_thread' in fault_branch and 'stopj(' in fault_branch and 'break' in fault_branch
    assert 'movej(' not in fault_branch and 'movel(' not in fault_branch
    assert 'grip' not in script.lower() and 'set_tool' not in script.lower()
