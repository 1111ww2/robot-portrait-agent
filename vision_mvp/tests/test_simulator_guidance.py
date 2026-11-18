"""在模拟中捕获重播的线索、不安全的输入和丢失/过时的运动。"""
import json
import math
from dataclasses import asdict
from test_simulator_fixed_base import fk

import pytest

from leica_tracker import simulator_guidance as sim
from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.robot_gestures import gesture_plan_for


def event(cue, now=10., gesture=True, tracking=True):
    plan = gesture_plan_for(GuidanceCue(cue))
    return json.dumps(dict(event='person_guidance', cue=cue, tracking=tracking,
        coordinate_frame='normalized_camera_image', robot_actuated=False,
        source_monotonic_ns=round(now*1e9),
        gesture=asdict(plan) if gesture and plan else None))


HOME=[0.,-math.pi/2,math.pi/2,-math.pi/2,-math.pi/2,0.]


def test_simulation_tick_wait_is_short_sliced_and_can_be_interrupted():
    class Stop:
        stopped = False
        def is_set(self):return self.stopped
    stop=Stop()
    elapsed=[0.]
    sleeps=[]
    def now():return elapsed[0]
    def sleep(duration):
        sleeps.append(duration);elapsed[0]+=duration
        if elapsed[0]>=.012:stop.stopped=True
    sim._wait_simulation_tick(stop,.05,clock=now,sleep=sleep)
    assert elapsed[0] < .02
    assert max(sleeps) <= .005


def test_simulation_tick_wait_does_not_oversleep_deadline():
    class Stop:
        def is_set(self):return False
    elapsed=[0.]
    def now():return elapsed[0]
    def sleep(duration):elapsed[0]+=duration
    sim._wait_simulation_tick(Stop(),.012,clock=now,sleep=sleep)
    assert elapsed[0] == pytest.approx(.012)


def test_simulation_tick_wait_checks_stop_file_during_wait():
    class Stop:
        def is_set(self):return False
    elapsed=[0.]
    class StopFile:
        def exists(self):return elapsed[0]>=.01
    def now():return elapsed[0]
    def sleep(duration):elapsed[0]+=duration
    sim._wait_simulation_tick(Stop(),.05,stop_file=StopFile(),clock=now,sleep=sleep)
    assert elapsed[0] == pytest.approx(.01)


def test_simulation_tick_wait_uses_high_resolution_deadline_clock(monkeypatch):
    class Stop:
        def is_set(self):return False
    elapsed=[0.]
    monkeypatch.setattr(sim.time,'perf_counter',lambda:elapsed[0])
    monkeypatch.setattr(sim.time,'monotonic',lambda:pytest.fail('low-resolution clock used'))
    monkeypatch.setattr(sim.time,'sleep',lambda value:elapsed.__setitem__(0,elapsed[0]+value))
    sim._wait_simulation_tick(Stop(),.012)
    assert elapsed[0] == pytest.approx(.012)


def position(offset):
    return fk([a+b for a,b in zip(HOME,offset)])[:3,3]


def at_time(stream, elapsed):
    # 源保持新鲜，而 WAIT 心跳则保留了活跃的手势。
    for step in range(1,round(elapsed/.1)+1):
        stamp=10.+step*.1
        stream.accept(event('WAIT',stamp),stamp)
    return stream.offset(10.+elapsed)


def test_small_gesture_reaches_scaled_front_position_and_survives_wait():
    payload=json.loads(event('MOVE_LEFT'))
    payload['gesture']['amplitude']=.2
    stream=sim.GuidanceStream();stream.accept(json.dumps(payload),10.)
    assert position(at_time(stream,.6)) == pytest.approx([-.4679,-.1333,.2093],abs=1e-8)


def test_screen_coupled_canonical_plan_and_wait_preserve_two_smooth_sweeps():
    payload=json.loads(event('MOVE_RIGHT'))
    payload['gesture']=asdict(gesture_plan_for(GuidanceCue.MOVE_RIGHT,profile='screen-coupled'))
    stream=sim.GuidanceStream();stream.accept(json.dumps(payload),10.)
    assert position(at_time(stream,1.5))[0]==pytest.approx(-.6119,abs=1e-8)
    for step in range(16,46):
        stamp=10.+step*.1;stream.accept(event('WAIT',stamp),stamp)
    assert position(stream.offset(14.5))[0]==pytest.approx(-.6119,abs=1e-8)
    assert stream.started==10.
    payload['gesture']['profile']='arbitrary'
    with pytest.raises(ValueError):stream.accept(json.dumps(payload),10.)


def test_completion_rotates_wrist_without_directional_tcp_sweep():
    stream=sim.GuidanceStream();stream.accept(event('ARRIVED'),10.)
    offset=at_time(stream,1.1)
    actual=fk([a+b for a,b in zip(HOME,offset)])
    assert actual[:3,3] == pytest.approx([-.4919,-.1333,.2093],abs=1e-8)
    assert actual[:3,2] == pytest.approx([math.sin(.24),0.,-math.cos(.24)],abs=1e-8)
    assert offset[0] == pytest.approx(0.,abs=1e-10)


def test_completion_retracts_then_holds_unique_end_pose_across_wait():
    stream=sim.GuidanceStream();stream.accept(event('ARRIVED'),10.)
    end=at_time(stream,6.)
    assert position(end) == pytest.approx([-.3919,-.1333,.2893],abs=1e-8)
    for step in range(61,121):
        stamp=round(10.+step*.1,6)
        stream.accept(event('WAIT',stamp),stamp)
    assert end == pytest.approx(stream.offset(22.),abs=1e-10)
    assert max(map(abs,end)) < .35


@pytest.mark.parametrize('interruption',['hold','gap'])
@pytest.mark.parametrize('cue',['ARRIVED','SIM_FRAME','SIM_COMPLETE'])
def test_interrupted_completion_does_not_resume_or_return_home_on_wait(interruption,cue):
    stream=sim.GuidanceStream();stream.accept(event(cue),10.)
    at_time(stream,1.)
    if interruption == 'hold':
        stream.accept(event('HOLD',11.1,tracking=False),11.1)
        stamp=11.2
    else:
        assert stream.offset(11.4) is None
        stamp=11.5
    stream.accept(event('WAIT',stamp),stamp)
    assert stream.offset(stamp) is None
    fresh=round(stamp+.1,6)
    stream.accept(event('INVITE',fresh),fresh)
    assert stream.offset(fresh) is not None


def test_completion_path_is_continuous_and_within_existing_joint_envelope():
    stream=sim.GuidanceStream();stream.accept(event('ARRIVED'),10.)
    previous=stream.offset(10.)
    for i in range(1,131):
        stamp=10.+i*.05
        stream.accept(event('WAIT',stamp),stamp)
        offset=stream.offset(stamp)
        assert max(map(abs,offset)) <= .35
        assert max(abs(a-b) for a,b in zip(offset,previous)) < .04
        assert offset[0] == pytest.approx(0.,abs=1e-10)
        previous=offset


def test_synthetic_sequence_leaves_time_for_the_complete_six_second_ending():
    assert sim.synthetic_event(32.1,100.)['cue'] == 'ARRIVED'
    assert sim.synthetic_event(39.9,100.)['cue'] == 'ARRIVED'
    assert sim.synthetic_event(40.1,100.)['cue'] == 'HOLD'
    assert sim.synthetic_event(44.1,100.)['cue'] == 'INVITE'


def test_simulated_framing_and_completion_share_the_same_anchor_pose():
    stream=sim.GuidanceStream();stream.accept(event('SIM_FRAME'),10.)
    fine=at_time(stream,2.4)
    assert position(fine) == pytest.approx([-.4719,-.1333,.2243],abs=1e-8)
    stream.accept(event('SIM_COMPLETE',12.5),12.5)
    assert stream.offset(12.5) == pytest.approx(fine,abs=1e-10)
    previous=fine
    for step in range(1,151):
        stamp=round(12.5+step*.05,6)
        stream.accept(event('WAIT',stamp),stamp)
        offset=stream.offset(stamp)
        assert max(map(abs,offset))<=.35
        assert max(abs(a-b) for a,b in zip(previous,offset))<.04
        previous=offset
    assert position(previous) == pytest.approx([-.3919,-.1333,.2893],abs=1e-8)


@pytest.mark.parametrize('amplitude',[0.,1.1,float('nan'),float('inf'),True])
def test_stream_rejects_invalid_gesture_amplitude(amplitude):
    payload=json.loads(event('MOVE_LEFT'));payload['gesture']['amplitude']=amplitude
    with pytest.raises(ValueError):
        sim.GuidanceStream().accept(json.dumps(payload),10.)


def test_double_sweep_returns_to_rest_and_wait_does_not_cancel_or_replay():
    for elapsed, expected_x in [(.6,-.3719),(1.2,-.4919),(1.8,-.3719),(2.4,-.4919)]:
        stream=sim.GuidanceStream();stream.accept(event('MOVE_LEFT'),10.)
        offset=at_time(stream,elapsed)
        assert offset[0] == pytest.approx(0.,abs=1e-10)
        assert position(offset) == pytest.approx([expected_x,-.1333,.2093],abs=1e-8)


@pytest.mark.parametrize('cue,elapsed,xyz', [
    ('MOVE_LEFT',.6,[-.3719,-.1333,.2093]),
    ('MOVE_RIGHT',.6,[-.6119,-.1333,.2093]),
    ('MOVE_UP',.5,[-.4919,-.1333,.2893]),
    ('MOVE_DOWN',.5,[-.4919,-.1333,.1293]),
    ('MOVE_CLOSER',.5,[-.4919,-.1083,.2093]),
    ('MOVE_FARTHER',.5,[-.4919,-.1583,.2093])])
def test_front_view_cues_move_on_correct_cartesian_axis(cue,elapsed,xyz):
    stream=sim.GuidanceStream();stream.accept(event(cue),10.)
    offset=at_time(stream,elapsed)
    assert position(offset) == pytest.approx(xyz,abs=1e-8)
    actual=fk([a+b for a,b in zip(HOME,offset)])
    assert actual[:3,:3] == pytest.approx(fk(HOME)[:3,:3],abs=1e-8)
    if cue in ('MOVE_LEFT','MOVE_RIGHT'):
        assert offset[0] == pytest.approx(0.,abs=1e-10)
        assert abs(offset[1])>.1 and abs(offset[2])>.1
    elif cue in ('MOVE_CLOSER','MOVE_FARTHER'):
        assert abs(offset[0])>.01
        assert offset[5] == pytest.approx(offset[0])
    assert max(map(abs,offset)) <= .35


def test_lost_tracking_cancels_gesture_and_stale_input_holds():
    stream = sim.GuidanceStream()
    stream.accept(event('MOVE_RIGHT'),10.)
    assert stream.offset(10.31) is None
    stream.accept(event('HOLD',10.32,tracking=False),10.32)
    assert stream.offset(10.33) is None
    stream.accept(event('WAIT',10.4),10.4)
    assert stream.offset(10.41) == pytest.approx([0.]*6)


def test_delayed_camera_message_holds_without_killing_stream_or_replaying_gesture():
    stream = sim.GuidanceStream()
    stream.accept(event('MOVE_RIGHT'), 10.)
    stream.accept(event('MOVE_LEFT', 10.1), 10.5)
    assert stream.failure is None
    assert stream.offset(10.5) is None
    assert stream.plan is None
    assert stream.stamp == 10.1  # 保留原始时间；切勿重新日期旧输入。
    assert stream.camera_status['phase'] == 'SOURCE_STALE'
    stream.accept(event('WAIT', 10.6), 10.6)
    assert stream.offset(10.6) == pytest.approx([0.]*6)
    stream.accept(event('MOVE_LEFT', 10.7), 10.7)
    assert stream.started == 10.7
    assert stream.offset(10.71) is not None


def test_delayed_message_still_rejects_replay_and_noncanonical_gestures():
    stream = sim.GuidanceStream()
    stream.accept(event('MOVE_RIGHT'), 10.)
    with pytest.raises(ValueError):
        stream.accept(event('MOVE_LEFT', 9.9), 10.5)
    payload = json.loads(event('MOVE_LEFT', 10.1))
    payload['gesture']['duration_ms'] = 999999
    with pytest.raises(ValueError):
        stream.accept(json.dumps(payload), 10.5)


def test_delayed_message_cancels_completion_until_a_new_explicit_gesture():
    stream = sim.GuidanceStream()
    stream.accept(event('ARRIVED'), 10.)
    stream.accept(event('WAIT', 10.1), 10.5)
    stream.accept(event('WAIT', 10.6), 10.6)
    assert stream.offset(10.6) is None
    stream.accept(event('INVITE', 10.7), 10.7)
    assert stream.offset(10.71) is not None


@pytest.mark.parametrize('change', [dict(source_monotonic_ns=0),
    dict(source_monotonic_ns=11_000_000_000),dict(robot_actuated=True),
    dict(cue='UNKNOWN'),dict(tracking='false'),dict(gesture={'duration_ms':999999}),
    dict(coordinate_frame='robot_base')])
def test_invalid_payload_rejected(change):
    payload=json.loads(event('MOVE_LEFT'));payload.update(change)
    with pytest.raises(ValueError):
        sim.GuidanceStream().accept(json.dumps(payload),10.)


def test_monotonic_order_and_heartbeat_do_not_restart_gesture():
    stream=sim.GuidanceStream();stream.accept(event('MOVE_LEFT'),10.)
    stream.accept(event('MOVE_LEFT',10.2,gesture=False),10.2)
    assert position(stream.offset(10.25))[0] == pytest.approx(-.4419)
    with pytest.raises(ValueError):
        stream.accept(event('WAIT',10.1),10.3)


def test_joint_controller_caps_speed_and_rejects_unbounded_targets():
    neutral=[0.]*6
    rates=sim.joint_velocity(neutral,[.15,0.,0.,0.,0.,0.],neutral,[0.]*6,.05)
    assert 0 < rates[0] <= .06
    assert max(map(abs,rates)) <= .6
    with pytest.raises(ValueError):
        sim.joint_velocity(neutral,[.5,0.,0.,0.,0.,0.],neutral,[0.]*6,.05)


def test_heartbeat_after_source_gap_cancels_old_gesture():
    stream=sim.GuidanceStream()
    stream.accept(event('MOVE_LEFT'),10.)
    stream.accept(event('WAIT',10.4),10.4)
    assert stream.offset(10.45) == pytest.approx([0.]*6)


def test_home_guard_rejects_wrong_tool_or_arbitrary_neutral():
    state=dict(joints=HOME,active_tcp_offset=[0.,0.,.2786,0.,0.,0.])
    sim.validate_home(state)
    with pytest.raises(ValueError):
        sim.validate_home(dict(state,joints=[.2,*HOME[1:]]))
    with pytest.raises(ValueError):
        sim.validate_home(dict(state,active_tcp_offset=[0.]*6))


@pytest.mark.parametrize('change', [dict(instance_id='other'), dict(instance_id=None),
                                    dict(backend='real'), dict(service='other')])
def test_start_rejects_state_identity_mismatch_before_opening_socket(monkeypatch, change):
    manifest = {'target': {'kind': 'simulator', 'authority_origin': 'simulation',
                           'instance_id': 'verified'}}
    state = {'instance_id': 'verified', 'backend': 'sim', 'service': 'urx-browser-simulator',
             'joints': HOME, 'active_tcp_offset': [0., 0., .2786, 0., 0., 0.],
             'urx': {'clients': 0, 'program_running': False}}
    state.update(change)
    monkeypatch.setattr(sim, 'get_state', lambda path: manifest if 'manifest' in path else state)
    monkeypatch.setattr(sim, 'LocalJointStream', lambda: pytest.fail('Opened socket for mismatched instance'))
    with pytest.raises(RuntimeError, match='identity'):
        sim.main(['--source', 'synthetic', '--duration', '0.01'])


def test_front_translation_velocity_keeps_tool_orientation_coupled():
    stream=sim.GuidanceStream();stream.accept(event('MOVE_RIGHT'),10.)
    offset=at_time(stream,.6)
    rates=sim.joint_velocity(HOME,offset,HOME,[0.]*6,.05)
    # 肩膀+肘部+手腕必须取消才能保留向下的工具。
    assert sum(rates[1:4]) == pytest.approx(0.,abs=1e-10)
    assert rates[0] == pytest.approx(0.,abs=1e-10)
    assert max(map(abs,rates)) <= .06+1e-10


def test_sim_fast_direction_is_canonical_and_holds_on_loss():
    payload=json.loads(event('MOVE_RIGHT'))
    payload['gesture']=asdict(gesture_plan_for(GuidanceCue.MOVE_RIGHT,profile='sim-fast'))
    stream=sim.GuidanceStream();stream.accept(json.dumps(payload),10.)
    assert stream.plan.duration_ms==3300
    assert position(at_time(stream,.8))[0]==pytest.approx(-.6119,abs=.002)
    stream.accept(event('HOLD',11.,tracking=False),11.)
    assert stream.offset(11.) is None
    bad=json.loads(event('MOVE_RIGHT',11.1))
    bad['gesture']=asdict(gesture_plan_for(GuidanceCue.MOVE_RIGHT,profile='sim-fast'))
    bad['gesture']['duration_ms']=2000
    with pytest.raises(ValueError,match='canonical'):stream.accept(json.dumps(bad),11.1)
