"""参考相对手势映射仅限于模拟和受限制。"""
from dataclasses import asdict
import json,math
from pathlib import Path

import numpy as np
import pytest

from leica_tracker.gesture_preview import normalized_offset_for
from leica_tracker.gesture_feedback import edge_feedback_for
from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.robot_gestures import gesture_plan_for


@pytest.mark.parametrize('cue,axis,sign,scale', [
    ('MOVE_LEFT',0,-1,.12), ('MOVE_RIGHT',0,1,.12),
    ('MOVE_UP',2,1,.085), ('MOVE_DOWN',2,-1,.085),
    ('MOVE_CLOSER',1,-1,.06), ('MOVE_FARTHER',1,1,.06)])
def test_fast_simulator_near_and_far_scale_same_direction(cue,axis,sign,scale):
    mapper=ReferenceMapper(Q,TCP); origin=nominal_fk(Q,TCP)[:3,3]
    for amplitude in (.65,1.):
        plan=gesture_plan_for(GuidanceCue(cue),profile='sim-fast',amplitude=amplitude)
        translation=nominal_fk(np.array(Q)+mapper.offset(plan,plan.duration_ms//4),TCP)[:3,3]-origin
        assert translation[axis]==pytest.approx(sign*scale*amplitude,abs=.001)


@pytest.mark.parametrize('cue', ['MOVE_CLOSER','MOVE_FARTHER'])
@pytest.mark.parametrize('amplitude', [.65,1.])
def test_fast_depth_motion_and_feedback_use_same_displacement(cue,amplitude):
    from leica_tracker.simulator_feedback import feedback_from_motion
    mapper=ReferenceMapper(Q,TCP);origin=nominal_fk(Q,TCP)[:3,3]
    plan=gesture_plan_for(GuidanceCue(cue),profile='sim-fast',amplitude=amplitude)
    for elapsed in (0,plan.duration_ms//8,plan.duration_ms//4,plan.duration_ms//2,plan.duration_ms):
        delta=nominal_fk(np.array(Q)+mapper.offset(plan,elapsed),TCP)[:3,3]-origin
        signal=feedback_from_motion(dict(token='x'*32,time_ns=10000000000,source_ns=123,
            cue=cue,hold=False,delta=delta.tolist(),amplitude=amplitude,profile='sim-fast'),
            token='x'*32,now_ns=10000000000,source_ns=123,cue=cue)
        expected=min(1.,max(0.,(delta[1] if cue=='MOVE_FARTHER' else -delta[1])/(.06*amplitude)))
        assert signal['edge']=='all'
        assert signal['intensity']==pytest.approx(expected,abs=1e-5)
        if elapsed in (0,plan.duration_ms//2,plan.duration_ms):
            assert signal['intensity']==pytest.approx(0.,abs=.001)
from leica_tracker.simulator_reference_guidance import ReferenceMapper,ReferenceGuidanceStream,validate_reference_home
from leica_tracker.physical_spatial import nominal_fk

REF=json.loads((Path(__file__).resolve().parents[1]/'config/simulator-reference-2026-10-28.json').read_text())
Q=REF['joints_rad'];TCP=REF['tcp_offset_m_rad']

def event(cue,t,gesture=True,tracking=True):
    p=gesture_plan_for(GuidanceCue(cue),profile='screen-coupled')
    return json.dumps(dict(event='person_guidance',cue=cue,tracking=tracking,
        coordinate_frame='normalized_camera_image',robot_actuated=False,
        source_monotonic_ns=round(t*1e9),gesture=asdict(p) if p and gesture else None))

def test_reference_validator_requires_exact_saved_pose_tcp_and_simulator_identity():
    s={'backend':'sim','service':'urx-browser-simulator','joints':Q,'active_tcp_offset':TCP}
    validate_reference_home(s,Q,TCP)
    for change in ({'backend':'real'},{'joints':[0.]*6},{'active_tcp_offset':[0.]*6}):
        with pytest.raises(ValueError):validate_reference_home({**s,**change},Q,TCP)

@pytest.mark.parametrize('cue',['MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER','INVITE','ARRIVED'])
def test_simulator_gestures_are_bounded_start_and_return_to_recorded_reference(cue):
    mapper=ReferenceMapper(Q,TCP);plan=gesture_plan_for(GuidanceCue(cue),profile='screen-coupled')
    assert mapper.offset(plan,0)==pytest.approx([0]*6,abs=1e-12)
    for t in range(0,plan.duration_ms+1,100):
        off=mapper.offset(plan,t)
        assert len(off)==6 and all(math.isfinite(x) and abs(x)<=.32 for x in off)
    assert mapper.offset(plan,plan.duration_ms)==pytest.approx([0]*6,abs=1e-10)

def test_horizontal_and_vertical_motion_are_relative_to_new_reference():
    mapper=ReferenceMapper(Q,TCP);reference=nominal_fk(Q,TCP)[:3,3]
    def translation(cue):
        p=gesture_plan_for(GuidanceCue(cue),profile='screen-coupled')
        return nominal_fk(np.array(Q)+mapper.offset(p,p.duration_ms//4),TCP)[:3,3]-reference
    assert translation('MOVE_LEFT')[0]==pytest.approx(-.12,abs=.001)
    assert translation('MOVE_RIGHT')[0]==pytest.approx(.12,abs=.001)
    assert translation('MOVE_UP')[2]==pytest.approx(.085,abs=.001)
    assert translation('MOVE_DOWN')[2]==pytest.approx(-.085,abs=.001)
    assert translation('MOVE_CLOSER')[1]==pytest.approx(.06,abs=.001)
    assert translation('MOVE_FARTHER')[1]==pytest.approx(-.06,abs=.001)

def test_full_sweep_is_continuous_and_preserves_reference_orientation():
    mapper=ReferenceMapper(Q,TCP);origin=nominal_fk(Q,TCP)
    for cue in ['MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER']:
        plan=gesture_plan_for(GuidanceCue(cue),profile='screen-coupled');previous=mapper.offset(plan,0)
        for t in range(50,plan.duration_ms+1,50):
            offset=mapper.offset(plan,t);assert max(abs(a-b) for a,b in zip(offset,previous))<.04
            actual=nominal_fk(np.array(Q)+offset,TCP);r=actual[:3,:3]@origin[:3,:3].T
            angle=math.acos(float(np.clip((np.trace(r)-1)/2,-1,1)))
            assert angle<.005
            previous=offset

def test_directional_motion_and_edge_pulse_share_the_same_gesture_phase():
    mapper=ReferenceMapper(Q,TCP);origin=nominal_fk(Q,TCP)[:3,3]
    for cue,edge,sign in [('MOVE_LEFT','left',-1),('MOVE_RIGHT','right',1)]:
        plan=gesture_plan_for(GuidanceCue(cue),profile='screen-coupled')
        for t in (round(plan.duration_ms*i/8) for i in range(9)):
            light=edge_feedback_for(plan,t)
            if t<plan.duration_ms:assert light['edge']==edge
            offset=mapper.offset(plan,t)
            x=(nominal_fk(np.array(Q)+offset,TCP)[:3,3]-origin)[0]
            assert x==pytest.approx(sign*.12*light['intensity'],abs=2e-5)

def test_arrival_confirmation_follows_green_pulses_then_stays_at_reference():
    mapper=ReferenceMapper(Q,TCP);origin=nominal_fk(Q,TCP)[:3,3]
    plan=gesture_plan_for(GuidanceCue.ARRIVED,profile='screen-coupled')
    for t in (0,400,1500,3000,4500,5999):
        light=edge_feedback_for(plan,t);offset=mapper.offset(plan,t)
        z=(nominal_fk(np.array(Q)+offset,TCP)[:3,3]-origin)[2]
        assert z==pytest.approx(.00875*light['intensity'],abs=2e-5)
    assert mapper.offset(plan,6000)==pytest.approx([0]*6,abs=1e-10)

def test_reference_stream_preserves_wait_but_stale_or_lost_input_holds():
    s=ReferenceGuidanceStream(Q,TCP);s.accept(event('MOVE_RIGHT',10.),10.)
    for i in range(1,16):
        t=10+i*.1;s.accept(event('WAIT',t,gesture=False),t)
    assert max(map(abs,s.offset(11.5)))>.02
    assert s.offset(11.81) is None
    s.accept(event('HOLD',11.9,gesture=False,tracking=False),11.9)
    assert s.offset(11.9) is None

def test_hold_then_wait_cannot_command_return_without_a_new_gesture():
    s=ReferenceGuidanceStream(Q,TCP);s.accept(event('MOVE_RIGHT',10.),10.)
    s.accept(event('WAIT',10.1,gesture=False),10.1);assert s.offset(10.1) is not None
    s.accept(event('HOLD',10.2,gesture=False,tracking=False),10.2)
    s.accept(event('WAIT',10.3,gesture=False),10.3)
    assert s.offset(10.3) is None
    s.accept(event('MOVE_LEFT',10.4),10.4);assert s.offset(10.4) is not None

def test_stale_then_late_wait_cannot_replay_an_interrupted_gesture():
    s=ReferenceGuidanceStream(Q,TCP);s.accept(event('MOVE_RIGHT',10.),10.)
    assert s.offset(10.31) is None
    # 这个来源的时间戳在收到时仍然是新的，但遵循之前的保留。
    s.accept(event('WAIT',10.2,gesture=False),10.4)
    assert s.offset(10.4) is None
    s.accept(event('MOVE_LEFT',10.5),10.5);assert s.offset(10.5) is not None


def test_reference_bridge_selects_new_mapping_only_for_idle_matching_simulator():
    from leica_tracker import simulator_guidance as sim
    state={'backend':'sim','service':'urx-browser-simulator','joints':Q,'active_tcp_offset':TCP,
           'urx':{'clients':0,'program_running':False,'control_owner':'idle'},'joint_velocity':[0.]*6}
    path=Path(__file__).resolve().parents[1]/'config/simulator-reference-2026-10-28.json'
    assert isinstance(sim.build_guidance_stream(state,path),ReferenceGuidanceStream)
    with pytest.raises(ValueError):sim.build_guidance_stream({**state,'joints':[0.]*6},path)
    with pytest.raises(RuntimeError):sim.build_guidance_stream({**state,'urx':{**state['urx'],'control_owner':'browser'}},path)


@pytest.mark.parametrize('amplitude',[.15,.35,.5,1.])
@pytest.mark.parametrize('cue,axis,expected',[(GuidanceCue.MOVE_LEFT,0,-.12),(GuidanceCue.MOVE_RIGHT,0,.12),(GuidanceCue.MOVE_UP,2,.085),(GuidanceCue.MOVE_DOWN,2,-.085),(GuidanceCue.MOVE_CLOSER,1,.06),(GuidanceCue.MOVE_FARTHER,1,-.06)])
def test_directional_demo_sweeps_full_extent_independent_of_input_amplitude(amplitude,cue,axis,expected):
    mapper=ReferenceMapper(Q,TCP);origin=nominal_fk(Q,TCP)[:3,3]
    plan=gesture_plan_for(cue,profile='screen-coupled',amplitude=amplitude)
    point=nominal_fk(np.array(Q)+mapper.offset(plan,plan.duration_ms//4),TCP)[:3,3]
    assert (point-origin)[axis]==pytest.approx(expected,abs=.001)
