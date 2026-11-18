"""发射的暂停分支的离线执行，而不是控制器接受。"""
import math
import textwrap

import pytest

from leica_tracker.physical_intro_program import IntroProgram, IntroProgress
from test_native_gesture import installation
from test_physical_intro_program import Gate, feedback, profile, tick


def run_branch(block, values, *, scope_output=None):
    """运行这些原生协议分支生成的分配/如果子集。"""
    lines = []
    for line in textwrap.dedent(block).splitlines():
        stripped = line.strip()
        if stripped == 'end':
            continue
        line = line.replace('global ', '').replace('p[', '[').replace('return False', "raise RuntimeError('native route rejected')")
        line = line.replace('kill motion_thread', 'kill(motion_thread)')
        line = line.replace('kill brake_thread', 'kill(brake_thread)')
        line = line.replace('kill recovery_brake_thread', 'kill(recovery_brake_thread)')
        line = line.replace('run route_motion()', 'launch(route_motion)')
        line = line.replace('run pause_stop()', 'launch(pause_stop)')
        line = line.replace('run recovery_brake()', 'launch(recovery_brake)')
        if stripped == 'break':
            line = line.replace('break', "raise RuntimeError('native completed')")
        lines.append(line)
    scope = {'norm': lambda x: abs(x) if isinstance(x, (int, float)) else math.hypot(*x), **values}
    if scope_output is not None:
        scope_output.append(scope)
    exec('\n'.join(lines), {}, scope)
    return scope


def xyz_packet(script, values):
    start = script.index('    if seq == seq_again and seq > previous:')
    end = script.index('    actual = get_actual_tcp_pose()', start)
    return run_branch(script[start:end], values)


def xyz_values(**changes):
    return {**dict(seq=9, seq_again=9, previous=8, consumed=8, token=123, flag=3,
                in_box=lambda value: True, incoming=[.22, -.5, .3, 0, 0, 0],
                enabled=True, armed=True, ever_armed=True, packet_age=.04,
                gesture_active=True, gesture_mode=True, gesture_age=1.17,
                gesture_scale=.8, gesture_endpoint=[.22, -.5, .3, 0, 0, 0],
                recovery_paused=False, recovery_stop_done=True), **changes}


def test_xyz_pause_freezes_phase_and_same_endpoint_resume_does_not_replay():
    script = installation().program(123)
    held = xyz_packet(script, xyz_values())
    assert held['consumed'] == 9 and held['packet_age'] == 0
    assert held['enabled'] is False and held['gesture_active'] is False
    assert held['gesture_mode'] is True and held['gesture_age'] == 1.17
    assert held['recovery_paused'] is True
    resumed = xyz_packet(script, {**held, 'seq': 10, 'seq_again': 10, 'flag': 2})
    assert resumed['enabled'] is True and resumed['gesture_active'] is True
    assert resumed['gesture_age'] == 1.17 and resumed['recovery_paused'] is False


def test_xyz_pause_before_arm_cannot_resume_without_original_hold_handshake():
    script = installation().program(123)
    held = xyz_packet(script, xyz_values(armed=False, enabled=False, gesture_mode=False))
    assert held['enabled'] is False and held['armed'] is False
    attempted = xyz_packet(script, {**held, 'seq': 10, 'seq_again': 10, 'flag': 2})
    assert attempted['enabled'] is False


def test_xyz_paused_different_endpoint_still_disarms_and_normal_hold_clears_phase():
    script = installation().program(123)
    held = xyz_packet(script, xyz_values())
    changed = xyz_packet(script, {**held, 'seq': 10, 'seq_again': 10, 'flag': 2,
                                 'incoming': [.23, -.5, .3, 0, 0, 0]})
    assert changed['armed'] is False and changed['enabled'] is False
    cleared = xyz_packet(script, {**held, 'seq': 10, 'seq_again': 10, 'flag': 0})
    assert cleared['gesture_mode'] is False and cleared['gesture_active'] is False


def test_intro_paused_enable_ack_does_not_claim_start_or_completion():
    data = profile(); gate = Gate(); progress = IntroProgress(IntroProgram(data, 7), gate)
    tick(progress, feedback(data, 100, 0), 1.)
    tick(progress, feedback(data, 100, 1), 1.05)
    assert tick(progress, feedback(data, 103, 2), 1.10) == 'PAUSED'
    assert progress.armed_ack and not progress.started_ack
    assert progress.completion_seen is None and progress.steady_since is None
    assert gate.reason is None and gate.sent[-1][1] == 1
    assert tick(progress, feedback(data, 101, 3), 1.15) == 'ACTIVE'
    assert progress.started_ack


def test_intro_can_hold_before_handshake_without_forging_armed_ack():
    data = profile(); gate = Gate(); progress = IntroProgress(IntroProgram(data, 7), gate)
    assert tick(progress, feedback(data, 103, 0), 1.) == 'PAUSED'
    assert gate.sent[-1][1] == 0
    assert tick(progress, feedback(data, 103, 1), 1.05) == 'PAUSED'
    assert not progress.armed_ack and not progress.started_ack
    assert gate.sent[-1][1] == 0
    assert tick(progress, feedback(data, 100, 2), 1.10) == 'STARTING'
    assert progress.armed_ack and gate.sent[-1][1] == 1


@pytest.mark.parametrize('state', [-12, -16])
def test_intro_pause_does_not_absorb_real_native_fault(state):
    data = profile(); gate = Gate(); progress = IntroProgress(IntroProgram(data, 7), gate)
    tick(progress, feedback(data, 100, 0), 1.)
    tick(progress, feedback(data, 100, 1), 1.05)
    tick(progress, feedback(data, 103, 2), 1.10)
    with pytest.raises(RuntimeError, match='resident fault'):
        tick(progress, feedback(data, state, 3), 1.15)
    assert progress.state == 'FAULT'


def test_intro_pause_checkpoint_does_not_replay_finished_waypoint():
    script = IntroProgram(profile(), 7).program()
    start = script.index('  thread route_motion():\n') + len('  thread route_motion():\n')
    end = script.index('  expected_start = ', start)
    body = script[start:end].rsplit('  end\n', 1)[0]
    calls = []
    class Interrupted(Exception):
        pass
    def move(target, **unused):
        calls.append(target)
        if len(calls) == 2:
            raise Interrupted()
    values = dict(waypoint_index=0, motion_error=0, motion_done=False,
                  is_within_safety_limits=lambda target: True, movej=move)
    # 同样未完成的返回段落被恢复，而不是第一次倾斜。
    captured = []
    with pytest.raises(Interrupted):
        run_branch(body, values, scope_output=captured)
    assert captured[0]['waypoint_index'] == 1
    assert calls[0] == profile()['waypoints'][0]
    # 中断后，从原生检查点执行发射的路线。
    resumed = run_branch(body, {**captured[0], 'movej': lambda q, **kw: calls.append(q)})
    assert calls[1:] == [profile()['waypoints'][1], profile()['waypoints'][1]]
    assert resumed['motion_done'] and resumed['waypoint_index'] == 2


def intro_cycle(script, values):
    start = script.index('    if seq == seq_again and seq > previous:')
    end = script.index('    if (started and packet_age', start)
    parsed = run_branch(script[start:end], {**values, 'valid': False, 'pause_requested': False, 'fault': 0})
    start = script.index('    if pause_requested and not paused:')
    end = script.index('    sync()\n', start)
    return run_branch(script[start:end], parsed)


def intro_values(**changes):
    registers = {42: 101}; events = []
    route=object(); brake=object()
    return {**dict(seq=9, seq_again=9, previous=8, consumed=8, token=7, flag=3,
                   packet_age=.04, started=True, armed=True, paused=False,
                   motion_done=False, motion_thread=77, waypoint_index=1,
                   pause_stop_done=True, brake_thread=None, route_motion=route, pause_stop=brake,
                   write_output_integer_register=lambda k,v: registers.update({k:v}),
                   stopj=lambda a: events.append(('stop',a)), kill=lambda handle: events.append(('kill',handle)),
                   launch=lambda routine: events.append(('resume',1) if routine is route else ('brake',.2)) or 78,
                   registers=registers, events=events), **changes}


def test_intro_pause_brakes_once_then_resume_keeps_unfinished_checkpoint():
    script = IntroProgram(profile(), 7).program()
    held = intro_cycle(script, intro_values())
    assert held['paused'] and held['registers'][42] == 103
    assert held['events'] == [('kill',77),('brake',.2)]
    held = intro_cycle(script, {**held, 'seq':10, 'seq_again':10})
    assert held['events'] == [('kill',77),('brake',.2)]
    resumed = intro_cycle(script, {**held, 'seq':11, 'seq_again':11, 'flag':1, 'pause_stop_done':True})
    assert not resumed['paused'] and resumed['registers'][42] == 101
    assert resumed['waypoint_index'] == 1 and resumed['motion_thread'] == 78
    assert resumed['events'] == [('kill',77),('brake',.2),('resume',1)]


def test_intro_prestart_pause_returns_to_real_arm_state_before_enable():
    script = IntroProgram(profile(), 7).program()
    held = intro_cycle(script, intro_values(started=False, armed=False))
    assert held['registers'][42] == 103 and not held['started']
    assert held['events'] == [('brake',.2)]
    armed = intro_cycle(script, {**held,'seq':10,'seq_again':10,'flag':0,'pause_stop_done':True})
    assert armed['registers'][42] == 100 and armed['armed'] and not armed['started']
    assert armed['events'] == [('brake',.2)]


def test_intro_cannot_resume_motion_while_pause_brake_thread_is_running():
    script = IntroProgram(profile(), 7).program()
    held = intro_cycle(script, intro_values())
    assert held['pause_stop_done'] is False
    attempted = intro_cycle(script, {**held,'seq':10,'seq_again':10,'flag':1})
    assert attempted['paused'] and attempted['registers'][42] == 103
    assert attempted['events'] == [('kill',77),('brake',.2)]


def test_xyz_pause_ack_loop_does_not_block_on_deceleration_and_resume_waits():
    script = installation().program(123)
    held = xyz_packet(script, xyz_values())
    events=[]
    start=script.index('    if recovery_paused:\n')
    end=script.index('    if enabled and armed:\n',start)
    state=run_branch(script[start:end],{**held,'moving':True,'state':0,
        'recovery_brake':object(),'launch':lambda routine:events.append('brake') or 88})
    assert state['state']==4 and state['moving'] is False
    assert state['recovery_stop_done'] is False and events==['brake']
    attempted=xyz_packet(script,{**state,'seq':10,'seq_again':10,'flag':2})
    assert attempted['recovery_paused'] and not attempted['enabled']
    assert not attempted['gesture_active'] and attempted['gesture_age']==1.17


def test_intro_recovery_brake_uses_configured_acceleration_not_terminal_stop():
    data=profile(); data['joint_acceleration']=1.2; data['reviewed']['joint_acceleration']=1.2
    script=IntroProgram(data,7).program(); calls=[]
    start=script.index('  thread pause_stop():\n')+len('  thread pause_stop():\n')
    end=script.index('  end\n',start)
    stopped=run_branch(script[start:end],{'pause_stop_done':False,'stopj':calls.append})
    assert calls==[1.2] and stopped['pause_stop_done'] is True
    # 终端心跳丢失仍然是致命的，包括在刹车时也是如此。
    start=script.index('    if fault != 0:\n')
    end=script.index('    if pause_requested',start)
    events=[]
    with pytest.raises(RuntimeError,match='native completed'):
        run_branch(script[start:end],dict(fault=12,pause_stop_done=False,brake_thread=88,
            started=True,motion_done=False,paused=True,
            kill=lambda value:events.append(('kill',value)),
            write_output_integer_register=lambda key,value:events.append(('state',value)),
            stopj=lambda value:events.append(('stop',value)),textmsg=lambda *values:None))
    assert events==[('kill',88),('state',-12),('stop',.2)]
