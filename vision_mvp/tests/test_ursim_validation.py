import pytest


def test_scripted_participant_points_are_direction_specific():
    from leica_tracker.ursim_validation import participant_point
    assert participant_point('MOVE_LEFT') == (.8,.3,.2)
    assert participant_point('MOVE_RIGHT') == (.2,.3,.2)
    assert participant_point('MOVE_UP') == (.5,.6,.2)
    assert participant_point('MOVE_DOWN') == (.5,.1,.2)
    assert participant_point('MOVE_CLOSER') == (.5,.3,.1)
    assert participant_point('MOVE_FARTHER') == (.5,.3,.4)
    with pytest.raises(ValueError):participant_point('UNBOUNDED')


def test_participant_points_emit_expected_cues_through_actual_guidance_controller():
    from types import SimpleNamespace
    import numpy as np
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from leica_tracker.app import AppConfig,guidance_controller_options
    from leica_tracker.ursim_validation import CUES, _FixtureProvider, participant_point
    provider=_FixtureProvider()
    config=AppConfig(camera_source='ipad',guidance_profile='claude-guide',single_plan=True)
    controller=GeminiGuidanceController(worker=provider,
        **guidance_controller_options(config,screen_coupled=True,physical_mode=True))
    frame=np.zeros((90,160,3),dtype=np.uint8);timestamp=0
    for expected in CUES:
        point=participant_point(expected);seen=[]
        for _ in range(260):
            decision=controller.update(SimpleNamespace(center_x=point[0],head_y=point[1],
                shoulder_width=point[2],confidence=.99),timestamp,mirrored=True,frame=frame)
            timestamp+=30
            if decision.cue.value in CUES:
                seen.append(decision.cue.value);break
        assert seen==[expected]
        for _ in range(150):controller.suspend_perception(timestamp);timestamp+=30
    assert provider.calls==1


def test_cycle_proof_requires_two_measured_excursions_and_actual_return():
    from leica_tracker.ursim_validation import cycle_proof
    fractions=[0,.4,.8,1,.8,.3,0,.4,.8,1,.7,.3,0]
    samples=[dict(fraction=x,joint_speed=.65,tcp_error_m=0 if i==12 else .12*x)
             for i,x in enumerate(fractions)]
    result=cycle_proof(samples)
    assert result['excursions']==2 and result['max_joint_speed_rad_s']==.65
    assert result['return_error_m']==0
    with pytest.raises(ValueError,match='excursion'):cycle_proof(samples[:7])
    with pytest.raises(ValueError,match='return'):cycle_proof(samples[:-1])
    with pytest.raises(ValueError,match='joint'):cycle_proof([dict(x,joint_speed=.8) for x in samples])
    with pytest.raises(ValueError,match='excursion'):
        cycle_proof([dict(x,fraction=x['fraction']*.7) for x in samples])


def test_cycle_proof_reports_elapsed_measurement_time():
    from leica_tracker.ursim_validation import cycle_proof
    fractions=[0,1,0,1,0]
    result=cycle_proof([dict(t=i,joint_speed=.4,fraction=x,tcp_error_m=.12*x)
                        for i,x in enumerate(fractions)])
    assert result['measured_cycle_s']==4
    assert result['peak_fractions']==[1,1]


def test_stage_proof_rejects_unbound_release_and_wrong_cue_or_changed_action():
    from leica_tracker.ursim_validation import _StageProof
    stage=_StageProof('MOVE_LEFT')
    with pytest.raises(ValueError,match='bound'):stage.release(10)
    stage.observe(True,'gate-one',dict(active=True,fresh=True,cue='MOVE_LEFT',action_id=4),10)
    with pytest.raises(ValueError,match='cue'):
        stage.observe(True,'gate-one',dict(active=True,fresh=True,cue='MOVE_RIGHT',action_id=4),10.1)
    with pytest.raises(ValueError,match='action'):
        stage.observe(True,'gate-one',dict(active=True,fresh=True,cue='MOVE_LEFT',action_id=5),10.1)
    with pytest.raises(ValueError,match='session'):
        stage.observe(True,'gate-two',dict(active=True,fresh=True,cue='MOVE_LEFT',action_id=4),10.1)
    bound=stage.release(11)
    assert bound['action_id']==4 and bound['gate_session']=='gate-one'


def test_stage_proof_ignores_late_preceding_motion_and_requires_current_gate():
    from leica_tracker.ursim_validation import _StageProof
    stage=_StageProof('MOVE_RIGHT',previous_action_id=4)
    old=dict(active=True,fresh=True,cue='MOVE_LEFT',action_id=4)
    stage.observe(True,'next-gate',old,10)
    assert stage.action_id is None
    fresh=dict(active=True,fresh=True,cue='MOVE_RIGHT',action_id=5)
    stage.observe(False,None,fresh,10)
    assert stage.action_id is None
    stage.observe(True,'next-gate',fresh,10)
    assert stage.release(11)['action_id']==5


def test_each_direction_requires_its_own_lit_reference_hold_evidence():
    from leica_tracker.ursim_validation import CUES, light_proof
    def row(cue,identifier):
        return dict(active=True,owner=True,control_fresh=True,tracking=False,
            phase='PHYSICAL_REFERENCE_HOLD',runtime={'status':'running','reason':'none'},
            motion=dict(active=True,fresh=True,cue=cue,action_id=identifier,sample_age_ms=10),
            feedback=dict(gesture_id=identifier,intensity=.5))
    bindings={cue:i+1 for i,cue in enumerate(CUES)}
    rows=[row(cue,i+1) for i,cue in enumerate(CUES)]
    assert set(light_proof(rows,bindings))==set(CUES)
    with pytest.raises(ValueError,match='MOVE_FARTHER'):light_proof(rows[:-1],bindings)
    rows[-1]['feedback']['gesture_id']=1
    with pytest.raises(ValueError,match='MOVE_FARTHER'):light_proof(rows,bindings)


def test_cleanup_attempts_every_resource_after_one_failure():
    from leica_tracker.ursim_validation import _cleanup_steps
    seen=[]
    def fail():seen.append('bad');raise RuntimeError('deliberate cleanup failure')
    errors=_cleanup_steps([('bad',fail),('good',lambda:seen.append('good'))])
    assert seen==['bad','good']
    assert errors==['bad: RuntimeError: deliberate cleanup failure']


def test_preflight_error_is_saved_without_runtime_connections(tmp_path,monkeypatch):
    import json
    from leica_tracker import ursim_validation as v
    from leica_tracker import motion_configuration
    from leica_tracker import canonical_delivery
    calls=[]
    monkeypatch.setattr(canonical_delivery,'runtime_source_identity',lambda root:(calls.append(root) or
        {'shared_runtime_source_sha256':'offline-source-hash','runtime_files':{}}),raising=False)
    monkeypatch.setattr(motion_configuration,'configure_motion_design',lambda *a,**k:(_ for _ in ()).throw(ValueError('bad design')))
    monkeypatch.setattr(v,'verify_docker',lambda *a,**k:pytest.fail('Bad design reached Docker'))
    target=tmp_path/'bad'
    with pytest.raises(ValueError,match='bad design'):v.execute_validation({},target)
    saved=json.loads((target/'result.json').read_text(encoding='utf-8'))
    assert saved['success'] is False and saved['cleanup_complete'] is True
    assert saved['failure']=='ValueError: bad design'
    assert saved['runtime_source_sha256']=='offline-source-hash' and calls==[v.ROOT]


def test_validation_without_execute_has_no_runtime_boundaries(tmp_path,monkeypatch):
    from leica_tracker import ursim_validation as v
    def forbidden(*a,**k):pytest.fail('Offline validation crossed a runtime boundary')
    monkeypatch.setattr(v,'verify_docker',forbidden)
    monkeypatch.setattr(v,'LoopbackGateway',forbidden)
    assert v.main(['--output',str(tmp_path/'result')],root=tmp_path)==0
    assert not (tmp_path/'result').exists()


def test_negative_trigger_requires_bound_native_moving_speed_and_actual_progress():
    from leica_tracker.ursim_validation import negative_trigger
    state={'motion':dict(active=True,fresh=True,phase='moving',cue='MOVE_LEFT',action_id=4,progress=.3)}
    fb=dict(timestamp=12.,joint_speed=[.1,0,0,0,0,0],tcp_speed=[.02,0,0,0,0,0])
    assert negative_trigger(state,fb,10,10.01,fraction=.3,bound_action_id=4)['action_id']==4
    assert negative_trigger(state,fb,10,10.01,fraction=.01,bound_action_id=4) is None
    assert negative_trigger(state,fb,10,10.01,fraction=.3,bound_action_id=3) is None
    assert negative_trigger(state,fb,10,10.2,fraction=.3,bound_action_id=4) is None
    assert negative_trigger(state,dict(fb,joint_speed=[0]*6,tcp_speed=[0]*6),10,10.01,fraction=.3,bound_action_id=4) is None
    state['motion']['phase']='settling'
    assert negative_trigger(state,fb,10,10.01,fraction=.3,bound_action_id=4) is None


@pytest.mark.parametrize('scenario,expected', [('stop','guest_stop'),('control-loss','control_heartbeat_expired')])
def test_negative_proof_stops_control_then_proves_stationary_before_service_cancel(scenario,expected):
    from leica_tracker.ursim_validation import prove_negative
    calls=[]
    class Owner:
        def stop(self):calls.append('Stop')
        def pause_heartbeat(self):calls.append('heartbeat only')
        def request(self,path):
            assert path=='/operator-state'
            return dict(motion={'active':False,'fresh':False},feedback={'intensity':0},
                physical={'revocation':{'reason':expected}})
    def proof():
        assert calls==['Stop' if scenario=='stop' else 'heartbeat only']
        calls.append('independent stationary');return True
    trigger=dict(action_id=4,cue='MOVE_LEFT',fraction=.3,joint_speed_rad_s=.2,tcp_linear_speed_m_s=.02)
    result=prove_negative(Owner(),scenario,trigger,verify_stationary=proof,service_alive=lambda:True,ui_trace=[])
    assert result['revocation_reason']==expected and result['stationary'] is True
    assert result['service_cancelled_before_proof'] is False
    assert calls[-1]=='independent stationary'


def test_negative_proof_rejects_service_cancellation_as_stop_evidence():
    from leica_tracker.ursim_validation import prove_negative
    class Owner:
        def stop(self):pass
        def request(self,path):return dict(motion={'active':False},feedback={'intensity':0},physical={'revocation':{'reason':'guest_stop'}})
    with pytest.raises(RuntimeError,match='service'):
        prove_negative(Owner(),'stop',dict(action_id=4),verify_stationary=lambda:pytest.fail('Cancelled service reached proof'),
            service_alive=lambda:False,ui_trace=[])


def expiry_fixture(*,premature=False,reset_deadline=False,native_state=4,resume_early=False):
    from types import SimpleNamespace
    clock=SimpleNamespace(now=10.,paused=False)
    class Owner:
        def pause_heartbeat(self):clock.paused=True
        def request(self,path):
            assert path=='/operator-state'
            if not clock.paused:
                return dict(active=True,communication=dict(status='healthy',recoverable=True,remaining_ms=0))
            if clock.now>=13.3 or premature and clock.now>=11.:
                return dict(active=False,communication=dict(status='healthy',recoverable=False,remaining_ms=0),
                    motion=dict(active=False,fresh=False),feedback={'intensity':0},
                    physical={'revocation':{'reason':'communication_timeout'}})
            if clock.now<10.3:
                return dict(active=True,communication=dict(status='healthy',recoverable=True,remaining_ms=0))
            remaining=(13.3-clock.now)*1000
            if reset_deadline and clock.now>=11.:remaining+=200
            return dict(active=True,communication=dict(status='waiting',recoverable=True,remaining_ms=remaining))
    def measurement():
        return dict(state=1 if resume_early and clock.now>=11. else native_state,session=77,timestamp=clock.now,
                    joint_speed=[0.]*6,tcp_speed=[0.]*6),clock.now
    def sleep(seconds):clock.now+=seconds
    return Owner(),clock,measurement,sleep


def test_negotiated_control_loss_proves_fixed_three_second_hold_then_real_expiry():
    from leica_tracker.ursim_validation import prove_negative
    owner,clock,measurement,sleep=expiry_fixture()
    result=prove_negative(owner,'control-loss',dict(action_id=4),verify_stationary=lambda:True,
        service_alive=lambda:True,ui_trace=[],native_measurement=measurement,
        clock=lambda:clock.now,sleep=sleep)
    assert result['revocation_reason']=='communication_timeout'
    recovery=result['communication_recovery']
    assert recovery['negotiated_window_s']==3. and recovery['waiting_observed'] is True
    assert recovery['deadline_lower_s']==pytest.approx(13.3,abs=.002)
    assert recovery['deadline_upper_s']==pytest.approx(13.3,abs=.002)
    assert recovery['native_hold']['state']==4 and recovery['native_hold']['stationary_dwell_s']>=.3
    assert result['revocation_observed_at']>=13.3 and recovery['no_premature_terminal'] is True


@pytest.mark.parametrize('change,match',[(dict(premature=True),'premature'),
    (dict(reset_deadline=True),'deadline'),(dict(native_state=1),'HOLD'),(dict(resume_early=True),'HOLD')])
def test_negotiated_loss_rejects_early_expiry_reset_deadline_or_missing_native_hold(change,match):
    from leica_tracker.ursim_validation import prove_negative
    owner,clock,measurement,sleep=expiry_fixture(**change)
    with pytest.raises(RuntimeError,match=match):
        prove_negative(owner,'control-loss',dict(action_id=4),verify_stationary=lambda:True,
            service_alive=lambda:True,ui_trace=[],native_measurement=measurement,
            clock=lambda:clock.now,sleep=sleep)


def test_negotiated_expiry_cannot_hide_a_robot_channel_failure_while_waiting():
    from leica_tracker.ursim_validation import prove_negative
    owner,clock,measurement,sleep=expiry_fixture()
    def failed():return measurement() if clock.now<11. else (None,float('-inf'))
    with pytest.raises(RuntimeError,match='Native feedback'):
        prove_negative(owner,'control-loss',dict(action_id=4),verify_stationary=lambda:True,
            service_alive=lambda:True,ui_trace=[],native_measurement=failed,
            clock=lambda:clock.now,sleep=sleep)


def test_complete_scenario_cannot_skip_opening(tmp_path,monkeypatch):
    from leica_tracker import ursim_validation as v
    monkeypatch.setattr(v,'verify_docker',lambda *a,**k:pytest.fail('Invalid scenario reached Docker'))
    with pytest.raises(ValueError,match='skip'):
        v.execute_validation({},tmp_path/'invalid',scenario='complete',skip_intro=True)


def test_cycle_cadence_is_reported_separately_from_native_completion():
    from leica_tracker.ursim_validation import cycle_proof
    fractions=[0,1,0,1,0,0]
    good=[dict(t=t,fraction=x,joint_speed=.4,tcp_error_m=.12*x,phase='settling' if i==4 else 'moving')
          for i,(t,x) in enumerate(zip([0,.8,1.6,2.4,3.3,4.0],fractions))]
    assert cycle_proof(good)['cadence_matched'] is True
    slow=[dict(s,t=s['t']*2) for s in good]
    assert cycle_proof(slow)['cadence_matched'] is False


def test_owner_request_failure_has_bounded_stage_status_and_elapsed_without_control_secrets(monkeypatch):
    from types import SimpleNamespace
    from leica_tracker import ursim_validation as v
    class Connection:
        def request(self,*a,**k):raise TimeoutError('deliberate timeout')
        def close(self):pass
    monkeypatch.setattr(v.http.client,'HTTPConnection',lambda *a,**k:Connection())
    owner=v._Owner(SimpleNamespace(token='private-token',origin='http://127.0.0.1:1234'))
    with pytest.raises(RuntimeError) as failure:owner.request('/action',{'secret':'private-control'})
    message=str(failure.value)
    assert 'TimeoutError' in message and 'phase=send' in message and 'elapsed_ms=' in message and '/action' in message
    assert 'private' not in message and len(message)<=500


def test_owner_heartbeat_preserves_bounded_error_detail():
    from types import SimpleNamespace
    from leica_tracker.ursim_validation import _Owner
    owner=_Owner(SimpleNamespace());owner.control={}
    owner.request=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('status=503 phase=headers elapsed_ms=200 '+'x'*1000))
    owner._heartbeat()
    assert 'status=503' in owner.failure and 'phase=headers' in owner.failure and len(owner.failure)<=400


def test_native_diagnostics_freeze_actual_feedback_and_first_fault():
    from leica_tracker.ursim_validation import _NativeDiagnostics
    diagnostics=_NativeDiagnostics()
    feedback=dict(timestamp=4.,state=-6,sequence=20,session=123,tcp_pose=[0]*6)
    diagnostics.feedback('writer-1',(feedback,10.),expected_session=123,pending_sequence=21)
    feedback['state']=0;feedback['tcp_pose'][0]=100
    diagnostics.session_error('Resident stopped or faulted')
    summary=diagnostics.summary()
    first=summary['first_native_fault']
    assert first['feedback']['state']==-6 and first['feedback']['tcp_pose'][0]==0
    assert first['sampled_at']==10. and first['session_match'] is True
    assert first['pending_sequence']==21 and first['sequence_match'] is False
    assert summary['session_errors'][0]['message']=='Resident stopped or faulted'


def test_diagnostic_channel_observes_existing_transport_without_replacing_samples():
    from types import SimpleNamespace
    from leica_tracker.ursim_validation import _NativeDiagnostics, _DiagnosticChannel
    feedback=dict(timestamp=4.,state=0,sequence=1,session=123)
    source=SimpleNamespace(latest=(feedback,10.),failure=None,closed=False)
    source.send=lambda *args:True;source.close=lambda:None
    diagnostics=_NativeDiagnostics();channel=_DiagnosticChannel(source,diagnostics,'writer-1')
    assert channel.latest is source.latest and channel.latest[0] is feedback
    assert channel.send(2,2,123,[.1,.2,.3]) is True
    source.latest=(dict(feedback,state=-6,sequence=2),10.1)
    assert channel.latest is source.latest
    assert diagnostics.summary()['first_native_fault']['session_match'] is True


def test_primary_notice_and_swallowed_session_stderr_are_preserved():
    import io
    from leica_tracker.ursim_validation import _NativeDiagnostics, _DiagnosticStderr
    diagnostics=_NativeDiagnostics();stream=io.StringIO();tee=_DiagnosticStderr(stream,diagnostics)
    diagnostics.primary('primary-1',({'mode':{'running':False}},10.),notice='LEICA_FAULT_XYZ_SAFETY p[0,0,0]')
    tee.write('PHYSICAL_SESSION_STOPPED: Resident stopped ');tee.write('or faulted\n');tee.flush()
    summary=diagnostics.summary()
    assert summary['first_native_fault']['notice'].startswith('LEICA_FAULT_XYZ_SAFETY')
    assert summary['session_errors'][0]['message']=='Resident stopped or faulted'
    assert stream.getvalue()=='PHYSICAL_SESSION_STOPPED: Resident stopped or faulted\n'


def test_export_identity_binds_actual_profile_and_each_program_bytes(tmp_path):
    import hashlib
    from leica_tracker.ursim_validation import _export_identity
    for name,content in [('profile.json',b'{"model":"ur10e"}'),('opening.urscript',b'opening\n'),('guidance.urscript',b'guidance\n')]:
        (tmp_path/name).write_bytes(content)
    result=_export_identity(tmp_path)
    assert result['profile_sha256']==hashlib.sha256(b'{"model":"ur10e"}').hexdigest()
    assert result['program_sha256']['opening.urscript']==hashlib.sha256(b'opening\n').hexdigest()
    (tmp_path/'guidance.urscript').write_bytes(b'changed\n')
    changed=_export_identity(tmp_path)
    assert changed['profile_sha256']==result['profile_sha256']
    assert changed['program_sha256']['guidance.urscript']!=result['program_sha256']['guidance.urscript']


def test_unexecuted_gate_reservation_does_not_complete_or_fail_participant_stage():
    from leica_tracker.ursim_validation import _StageProof
    stage=_StageProof('MOVE_DOWN',previous_action_id=3)
    assert stage.advance(True,True,'reservation',{'active':False},10) is None
    assert stage.advance(False,False,None,{'active':False},10.1) is None
    assert stage.cue=='MOVE_DOWN' and stage.action_id is None and stage.previous_action_id==3
    assert stage.unexecuted_reservations==1
    motion=dict(active=True,fresh=True,cue='MOVE_DOWN',action_id=4)
    assert stage.advance(True,True,'accepted-gate',motion,11) is None
    assert stage.advance(False,False,None,{'active':False},15)['action_id']==4
    assert stage.advance(False,False,None,{'active':False},15.1) is None


def test_gate_snapshot_reads_hold_session_and_motion_under_one_lock():
    from types import SimpleNamespace
    from leica_tracker.ursim_validation import _reference_snapshot
    held=[]
    class Lock:
        def __enter__(self):held.append(True)
        def __exit__(self,*a):held.pop()
    def require_lock(value):
        assert held
        return value
    gate=SimpleNamespace(blocked=True,session='one-gate',held=lambda ns:require_lock(ns<=10))
    display=SimpleNamespace(_lock=Lock(),_reference_gate=gate,
        _current_motion=lambda:require_lock({'active':False}),
        reference_held=lambda ns:pytest.fail('Non-atomic legacy read used'))
    assert _reference_snapshot(display,5)==dict(held=True,blocked=True,session='one-gate',motion={'active':False})


def test_return_sample_keeps_actual_timestamp_and_excludes_later_stale_motion_rows():
    from leica_tracker.ursim_validation import _samples_through_release
    samples=[dict(t=1,fraction=0),dict(t=2,fraction=1),dict(t=3,fraction=0),dict(t=4,fraction=1),dict(t=5,fraction=0)]
    release=dict(t=4.5,fraction=0)
    assert _samples_through_release(samples,release)==samples[:4]+[release]
    assert release['t']==4.5 and samples[-1]['t']==5


def test_controller_cadence_preserves_slow_simulator_wall_time():
    from leica_tracker.ursim_validation import cycle_proof
    fractions=[0,1,0,1,0,0]
    wall=[0,1,2,3,4.562,5.234]
    robot=[20+t*.8277 for t in wall]
    samples=[dict(t=t,robot_timestamp=r,fraction=x,joint_speed=.5,tcp_error_m=.00017 if i==5 else .12*x,
        phase='settling' if i==4 else 'moving') for i,(t,r,x) in enumerate(zip(wall,robot,fractions))]
    proof=cycle_proof(samples,require_controller_clock=True)
    assert proof['cadence_clock']=='controller' and proof['cadence_matched'] is True
    assert proof['measured_moving_s']==4.562 and proof['measured_cycle_s']==5.234
    assert proof['controller_moving_s']==pytest.approx(3.7759674)
    assert proof['simulator_clock_ratio']==pytest.approx(.8277)
    with pytest.raises(ValueError,match='clock'):
        cycle_proof([dict(s,robot_timestamp=20) for s in samples],require_controller_clock=True)
    with pytest.raises(ValueError,match='clock'):
        cycle_proof([{k:v for k,v in s.items() if k!='robot_timestamp'} for s in samples],require_controller_clock=True)
