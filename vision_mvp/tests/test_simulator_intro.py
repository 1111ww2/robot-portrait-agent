"""好奇心编舞仅限于本地模拟，以实际开始为基准。"""
from dataclasses import asdict
import json
import pytest

from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.robot_gestures import gesture_plan_for
from leica_tracker.simulator_guidance import GuidanceStream
from leica_tracker.simulator_reference_guidance import ReferenceGuidanceStream, ReferenceMapper, curiosity_offset
from leica_tracker.simulator_feedback import SimulatorIntroGate, intro_ready_from_motion
from leica_tracker.physical_spatial import nominal_fk
import numpy as np
from test_simulator_reference_guidance import Q, TCP


def event(cue, t, plan=None, tracking=True):
    return json.dumps(dict(event='person_guidance', cue=cue, tracking=tracking,
        coordinate_frame='normalized_camera_image', robot_actuated=False,
        source_monotonic_ns=round(t*1e9), gesture=asdict(plan) if plan else None))


def intro(action,t,intro_id=10_000_000_000):
    return json.dumps(dict(event='simulator_intro',action=action,intro_id=intro_id,
                           source_monotonic_ns=round(t*1e9)))


def test_intro_uses_tilt_and_invitation_then_exact_return():
    plan=gesture_plan_for(GuidanceCue.INVITE,profile='sim-intro')
    stream=ReferenceGuidanceStream(Q,TCP)
    stream.accept(intro('start',10),10)
    stream.anchor_intro(10_000_000_000, Q)
    offsets=[]
    for ms in range(0,plan.duration_ms+1,50):
        now=10+ms/1000
        if ms: stream.accept(intro('heartbeat',now),now)
        offsets.append(stream.offset(now))
    assert offsets[0]==pytest.approx([0]*6)
    assert max(abs(v) for v in offsets[20])>.08
    assert max(abs(a-b) for a,b in zip(offsets[60],offsets[80]))>.03
    assert offsets[-1]==pytest.approx([0]*6,abs=1e-10)
    assert max(abs(a-b) for x,y in zip(offsets,offsets[1:]) for a,b in zip(x,y))<.04


def test_intro_tilts_then_twice_retracts_away_from_viewer_and_returns():
    # 本地模拟器正面视图将+Y视为朝向观察者。测量工具的实际标称路径，而不仅仅是选择的关节角符号。
    origin=nominal_fk(Q,TCP)[:3,3]
    tilted_joints=np.asarray(Q).copy();tilted_joints[4]-=.28
    retract=ReferenceMapper(tilted_joints,TCP)._solve([0.,-.10,0.])
    tilted=nominal_fk(np.asarray(Q)+curiosity_offset(850,retract),TCP)[:3,3]
    first=nominal_fk(np.asarray(Q)+curiosity_offset(1515,retract),TCP)[:3,3]
    between=nominal_fk(np.asarray(Q)+curiosity_offset(2275,retract),TCP)[:3,3]
    second=nominal_fk(np.asarray(Q)+curiosity_offset(3415,retract),TCP)[:3,3]
    assert curiosity_offset(850,retract)[4]==pytest.approx(-.28)
    assert first[1]-tilted[1]<-.07
    assert second[1]-between[1]<-.07
    assert first[1]-origin[1]<-.08
    assert between[1]-tilted[1]>.01  # 明显地向观众回转
    assert second==pytest.approx(first,abs=1e-8)
    assert curiosity_offset(5600,retract)==pytest.approx([0.]*6,abs=1e-12)


def test_intro_anchors_current_pose_and_stops_on_lost_input():
    plan=gesture_plan_for(GuidanceCue.INVITE,profile='sim-intro')
    s=ReferenceGuidanceStream(Q,TCP)
    shifted=[v+.01 for v in Q]
    s.accept(intro('start',10),10)
    s.anchor_intro(10_000_000_000,shifted)
    assert list(s.mapper.joints)==pytest.approx(shifted)
    s.accept(intro('heartbeat',10.1),10.1)
    assert s.offset(10.1) is not None
    assert s.offset(10.41) is None
    s.accept(intro('heartbeat',10.3),10.42)
    assert s.offset(10.42) is None
    with pytest.raises(ValueError): GuidanceStream().accept(intro('start',11),11)


def test_intro_stop_and_transport_failure_do_not_restart_on_heartbeat():
    for interruption in ('stop','transport'):
        s=ReferenceGuidanceStream(Q,TCP)
        s.accept(intro('start',10),10);s.anchor_intro(10_000_000_000,Q)
        s.accept(intro('heartbeat',10.1),10.1)
        if interruption=='stop':s.accept(event('HOLD',10.2,tracking=False),10.2)
        else:s.interrupt_transport()
        s.accept(intro('heartbeat',10.3),10.3)
        assert s.offset(10.3) is None
        s.accept(intro('start',10.4,10_400_000_000),10.4)
        assert s.anchor_intro(10_400_000_000,Q)
        assert s.offset(10.4)==pytest.approx([0]*6)


def test_repeated_start_keeps_original_intro_id_and_anchor_until_completed():
    s=ReferenceGuidanceStream(Q,TCP)
    original=10_000_000_000
    s.accept(intro('start',10,original),10)
    assert s.anchor_intro(original,Q)
    s.accept(intro('heartbeat',10.1,original),10.1)
    tilted=[*Q];tilted[4]-=.20
    duplicate=10_200_000_000
    s.accept(intro('start',10.2,duplicate),10.2)
    assert s.plan_source_ns==original
    assert s.started==10
    assert not s.anchor_intro(duplicate,tilted)
    assert list(s.mapper.joints)==pytest.approx(Q)
    s.accept(intro('heartbeat',10.25,original),10.25)
    assert s.offset(10.25) is not None
    # 只有桥梁的测量返回和静止门标志着完成。
    s.mark_intro_completed(duplicate)
    s.accept(intro('start',10.26,duplicate),10.26)
    assert s.plan_source_ns==original
    s.mark_intro_completed(original)
    s.accept(intro('start',10.3,duplicate),10.3)
    assert s.plan_source_ns==duplicate
    assert s.anchor_intro(duplicate,tilted)
    assert list(s.mapper.joints)==pytest.approx(tilted)


def test_duplicate_start_or_wrong_id_heartbeat_does_not_keep_old_intro_alive():
    s=ReferenceGuidanceStream(Q,TCP)
    original=10_000_000_000
    s.accept(intro('start',10,original),10);assert s.anchor_intro(original,Q)
    s.accept(intro('start',10.1,10_100_000_000),10.1)
    s.accept(intro('heartbeat',10.2,10_100_000_000),10.2)
    assert s.offset(10.31) is None
    # 在实际源超时后，明确重试通常是可以接受的。
    s.accept(intro('start',10.4,10_400_000_000),10.4)
    assert s.plan_source_ns==10_400_000_000
    assert s.anchor_intro(10_400_000_000,Q)


def test_generic_guidance_cannot_smuggle_intro():
    p=gesture_plan_for(GuidanceCue.INVITE,profile='sim-intro')
    with pytest.raises(ValueError):ReferenceGuidanceStream(Q,TCP).accept(event('INVITE',10,p),10)


def test_intro_gate_requires_settled_actual_pose_and_fresh_matching_feedback():
    gate=SimulatorIntroGate()
    kw=dict(source_ns=10,cue='INVITE',pattern='curiosity_intro',hold=False,
            elapsed_ms=7000,duration_ms=6000,joints=Q,neutral=Q,velocity=[0.]*6)
    assert not gate.update(**kw,now=1.)
    assert not gate.update(**kw,now=1.2)
    assert gate.update(**kw,now=1.31)
    assert not gate.update(**{**kw,'joints':[v+.01 for v in Q]},now=1.4)
    data={'token':'x'*32,'time_ns':1000,'source_ns':10,'intro_ready':True,'hold':False,'profile':'sim-intro'}
    def ready(**change):return intro_ready_from_motion({**data,**change},token='x'*32,now_ns=1100,source_ns=10)
    assert ready()
    assert not ready(source_ns=11)
    assert not ready(hold=True)
    assert not ready(time_ns=-400_000_000)
