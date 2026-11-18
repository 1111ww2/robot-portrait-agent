"""在没有设备接口的环境中验证重启保护逻辑和命令行入口。"""
import pytest
import time

from leica_tracker import ursim_restart_validation as v


def packet(*, native=3, position=.1, speed=0.):
    return dict(state=native, tcp_pose=[position,-.5,.3,0.,0.,0.],
                tcp_speed=[speed,0.,0.,0.,0.,0.], joint_speed=[speed,0.,0.,0.,0.,0.])


def state():
    return dict(active=True, motion=dict(active=True, fresh=True, cue='MOVE_LEFT'))


def qualifies(feedback=None, *, scenario=v.SCENARIOS[0], current=None, **changes):
    options=dict(blocked=True,reference=[.1,-.5,.3,0.,0.,0.],peak_fraction=.95)
    options.update(changes)
    return v.stop_qualifies(scenario,state() if current is None else current,
        packet() if feedback is None else feedback,10.,10.01,**options)


def test_reference_stop_requires_a_real_completed_excursion_and_native_return():
    assert qualifies()
    assert not qualifies(peak_fraction=.01)
    assert not qualifies(blocked=False)
    assert not qualifies(packet(native=1))
    assert not qualifies(packet(position=.102))
    assert not qualifies(packet(speed=.01))


def test_reference_stop_can_observe_held_ack_after_transient_native_complete_state():
    current=state();current['motion']['phase']='settling'
    assert qualifies(packet(native=0),current=current)
    assert not qualifies(packet(native=0))
    assert not qualifies(packet(native=0),current=current,peak_fraction=.1)
    assert not qualifies(packet(native=0,position=.102),current=current)


def test_displaced_stop_requires_current_motion_and_a_measured_offset():
    assert qualifies(packet(native=1,position=.12,speed=.01),scenario=v.SCENARIOS[1])
    assert not qualifies(packet(native=1,position=.1,speed=.01),scenario=v.SCENARIOS[1])
    assert not qualifies(packet(native=3,position=.12),scenario=v.SCENARIOS[1])


def test_selected_displaced_restart_uses_the_same_measured_mid_motion_stop_trigger():
    scenario='selected-displaced-stop-restart'
    assert qualifies(packet(native=1,position=.12,speed=.01),scenario=scenario)
    assert not qualifies(packet(native=3,position=.12),scenario=scenario)
    assert not qualifies(packet(native=1,position=.1,speed=.01),scenario=scenario)


def test_stale_invalid_and_unrelated_state_cannot_trigger_stop():
    options=dict(blocked=True,reference=[.1,-.5,.3,0.,0.,0.],peak_fraction=1.)
    assert not v.stop_qualifies(v.SCENARIOS[0],state(),packet(),10.,10.2,**options)
    assert not v.stop_qualifies(v.SCENARIOS[0],None,packet(),10.,10.01,**options)
    for field in ('active','fresh'):
        current=state();current['motion'][field]=False
        assert not qualifies(current=current)
    current=state();current['motion']['cue']='MOVE_RIGHT'
    assert not qualifies(current=current)
    broken=packet();broken['joint_speed']=[float('nan')]*6
    assert not qualifies(broken)


def test_cli_without_execute_opens_no_docker_or_device_and_writes_nothing(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'execute_restart_validation',lambda *a,**k:pytest.fail('Offline CLI executed'))
    target=tmp_path/'new-case'
    assert v.main(['--source-profile','missing.json','--output',str(target)],root=tmp_path)==0
    assert not target.exists()


@pytest.mark.parametrize('extra',[
    ['--timeout','181'],['--timeout','nan'],['--timeout','0'],
    ['--scenario','physical'],['--robot-model','ur11'],
])
def test_cli_rejects_invalid_execution_contract_offline(tmp_path,extra):
    with pytest.raises(SystemExit):
        v.main(['--source-profile','missing.json',*extra],root=tmp_path)


def test_cli_rejects_source_or_output_outside_workspace(tmp_path):
    with pytest.raises(SystemExit):
        v.main(['--source-profile','../outside.json'],root=tmp_path)
    with pytest.raises(SystemExit):
        v.main(['--source-profile','missing.json','--output',str(tmp_path.parent/'outside')],root=tmp_path)


def test_fixture_intro_publishes_its_genuine_camera_frame_for_restart_hold_ack(tmp_path):
    from leica_tracker.ipad_camera import IPadCapture
    from leica_tracker.person_guidance import GuidanceCue
    frames=[]
    class Display:
        started=time.monotonic_ns()
        def control_snapshot(self):
            return dict(active=True,instance='fixture',epoch=1,control_started_ns=self.started)
        def physical_intro_ready(self, identifier):
            return False
        def publish(self, frame, decision, *, source_monotonic_ns):
            frames.append((frame,decision,source_monotonic_ns))
            participant.closed.set()
    path=tmp_path/'participant.jsonl';path.touch()
    participant=v._RestartParticipant(Display(),IPadCapture(), 'fixture-owner',path)
    participant._run()
    assert participant.failure is None and len(frames)==1
    frame,decision,source_ns=frames[0]
    assert frame is not None and frame.shape==(90,160,3)
    assert decision.phase=='INTRO' and decision.cue is GuidanceCue.HOLD and not decision.tracking
    assert 0<=time.monotonic_ns()-source_ns<300_000_000


def rejection_inputs():
    return dict(failure='RuntimeError: RuntimeError: path=/control-heartbeat phase=body status=409 elapsed_ms=0.0',
        scenario='displaced-stop-restart',generation=2,second_started=10.,second_epoch=3,
        operator_state={'physical':{'mode':'intro','revocation':{'reason':'backend_fault','mode':'intro','epoch':3}}},
        current_gate={'blocked':True,'session':'original','reference':(.1,-.5,.3,0.,0.,0.)},
        original_gate={'blocked':True,'session':'original','reference':(.1,-.5,.3,0.,0.,0.)})


def test_expected_negative_owner_409_waits_for_authoritative_reference_rejection():
    assert v.expected_negative_owner_rejection(**rejection_inputs())
    row={'kind':'session_error','observed_at':11.,'message':'Local reference service unavailable '
         '{"category":"http_rejected","status":400,"reference_op":"restart_release",'
         '"reference_reason":"return_unverified"}'}
    assert v.expected_restart_terminal_rejection(row,second_started=10.)
    assert not v.expected_restart_terminal_rejection(row,second_started=12.)


@pytest.mark.parametrize('field,value',[
    ('scenario','reference-stop-restart'),('generation',1),('second_started',None),('second_epoch',4),
    ('failure','RuntimeError: path=/control-heartbeat phase=body status=500 elapsed_ms=0.0'),
    ('failure','RuntimeError: path=/action phase=body status=409 elapsed_ms=0.0'),
])
def test_unmatched_owner_error_is_not_accepted_as_negative_success(field,value):
    values=rejection_inputs();values[field]=value
    assert not v.expected_negative_owner_rejection(**values)


@pytest.mark.parametrize('tamper',['other_revocation','wrong_mode','wrong_epoch','unblocked','different_reference','different_session'])
def test_owner_rejection_requires_original_gate_and_matching_backend_fault(tamper):
    values=rejection_inputs()
    revocation=values['operator_state']['physical']['revocation']
    if tamper=='other_revocation':revocation['reason']='camera_stale'
    elif tamper=='wrong_mode':revocation['mode']='guidance'
    elif tamper=='wrong_epoch':revocation['epoch']=2
    elif tamper=='unblocked':values['current_gate']['blocked']=False
    elif tamper=='different_reference':values['current_gate']['reference']=(.2,-.5,.3,0.,0.,0.)
    else:values['current_gate']['session']='replacement'
    assert not v.expected_negative_owner_rejection(**values)


@pytest.mark.parametrize('payload',[
    '{"category":"http_rejected","status":400,"reference_op":"restart_begin","reference_reason":"return_unverified"}',
    '{"category":"http_rejected","status":400,"reference_op":"restart_release","reference_reason":"wrong_session"}',
    '{"category":"timeout","status":400,"reference_op":"restart_release","reference_reason":"return_unverified"}',
    'not structured JSON',
])
def test_unrelated_terminal_fault_cannot_satisfy_displaced_reference_negative(payload):
    row={'kind':'session_error','observed_at':11.,'message':'Local reference service unavailable '+payload}
    assert not v.expected_restart_terminal_rejection(row,second_started=10.)
