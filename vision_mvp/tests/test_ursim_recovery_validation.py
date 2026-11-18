"""用注入的时钟测试恢复依据，不使用控制器接口。"""
import copy
import pytest


def test_jitter_cli_is_offline_and_validates_explicit_bounded_option(tmp_path,monkeypatch):
    from leica_tracker import ursim_validation as v
    monkeypatch.setattr(v,'execute_validation',lambda *a,**k:pytest.fail('Offline CLI executed'))
    assert v.main(['--communication-jitter-ms','1200','--output',str(tmp_path/'unused')],root=tmp_path)==0
    assert not (tmp_path/'unused').exists()
    with pytest.raises(SystemExit):v.main(['--communication-jitter-ms','2500'],root=tmp_path)


def controls():
    return dict(instance='same-instance',epoch=3,control_started_ns=1000000000,active=True)


def telemetry(state,stamp=20.,speed=.0):
    return dict(state=state,session=77,timestamp=stamp,joint_speed=[speed,0,0,0,0,0],tcp_speed=[0.]*6)


def state(status,action_id=4):
    return dict(communication=dict(status=status,recoverable=True,remaining_ms=2000),
                motion=dict(active=True,fresh=True,cue='MOVE_LEFT',action_id=action_id))


def probe(stage='MOVE_LEFT'):
    from leica_tracker.ursim_validation import _RecoveryProbe
    return _RecoveryProbe(stage,feedback=telemetry(1),control=controls(),action_id=4 if stage=='MOVE_LEFT' else None,
        intro_id=123,gap=dict(started_at=10.,resume_not_before=11.2,requested_ms=1200,last_sequence=8))


def test_probe_requires_native_pause_and_continuous_measured_stillness_before_resume():
    p=probe()
    for i in range(5):
        at=10.4+i*.1
        p.observe(state('waiting'),telemetry(4,20.+i*.1),at,at,controls())
    p.observe(state('healthy'),telemetry(1,20.5,.05),11.3,11.3,controls())
    p.complete(control=controls(),reference_release_ns=12000000000,post_return_frame_source_ns=12000000001,
               now=12.1,return_error_m=.0001)
    result=p.result()
    assert result['native_hold_state']==4 and result['same_control'] is True
    assert result['stationary_hold']['dwell_s']>=.3 and result['resumed_at']==11.3
    assert result['native_hold_s']==pytest.approx(.5) and result['completed'] is True


def test_probe_rejects_identity_change_or_resume_without_zero_barrier():
    p=probe()
    with pytest.raises(ValueError,match='identity'):
        p.observe(state('waiting'),telemetry(4),10.4,10.4,{**controls(),'epoch':4})
    p=probe();p.observe(state('waiting'),telemetry(4,speed=.05),10.4,10.4,controls())
    with pytest.raises(ValueError,match='stationary'):
        p.observe(state('healthy'),telemetry(1,20.1,.05),10.5,10.5,controls())


def test_probe_cannot_use_stale_or_repeated_robot_samples_to_build_hold_dwell():
    p=probe();p.observe(state('waiting'),telemetry(4),10.4,10.4,controls())
    p.observe(state('waiting'),telemetry(4),10.4,10.5,controls())
    assert p.result()['stationary_hold'] is None and len(p.result()['observations'])==1
    with pytest.raises(ValueError,match='fresh'):
        p.observe(state('waiting'),telemetry(4),10.4,10.6,controls())


def resumed_intro_probe():
    p=probe('intro')
    for i in range(5):
        at=10.4+i*.1;p.observe(state('waiting'),telemetry(103,20.+i*.1),at,at,controls())
    p.observe(state('healthy'),telemetry(101,20.5,.05),11.3,11.3,controls())
    return p


def intro_return_packet(native,robot,session=77):
    return dict(telemetry(native,robot),session=session,joints=[0.]*6,tcp_pose=[.1,.2,.3,0.,0.,0.])


def seal_intro_return(p):
    for i in range(8):
        at=12.+i*.1;fb=intro_return_packet(102,21.+i*.1)
        p.observe(state('healthy'),fb,at,at,controls())
        p.observe_intro_return(fb,at,at,controls(),reference_pose=[.1,.2,.3,0.,0.,0.],reference_joints=[0.]*6)


def test_intro_probe_accepts_only_proven_stationary_return_to_one_held_xyz_handoff():
    p=resumed_intro_probe();seal_intro_return(p)
    fresh=intro_return_packet(2,25.,session=88)
    p.observe(state('healthy'),fresh,15.,15.,controls())
    assert p.result()['intro_handoff']['from_session']==77 and p.result()['intro_handoff']['to_session']==88
    assert p.result()['intro_return']['stationary_dwell_s']>=.6
    assert p.result()['completed'] is False  # 放行后仍须取得一帧更新的画面。
    p.observe(state('healthy'),intro_return_packet(0,25.1,88),15.1,15.1,controls())
    p.complete(control=controls(),reference_release_ns=15000000000,post_return_frame_source_ns=15100000000,
               now=15.2,return_error_m=0.)
    assert p.result()['resumed_native_session']==77 and p.result()['completed'] is True


@pytest.mark.parametrize('tamper',['no_return','moving_new','wrong_pose','second_session','not_healthy'])
def test_intro_handoff_does_not_hide_interrupted_session_or_unmeasured_return(tamper):
    p=resumed_intro_probe()
    if tamper!='no_return':seal_intro_return(p)
    fresh=intro_return_packet(2,25.,88)
    if tamper=='moving_new':fresh['state']=1;fresh['joint_speed'][0]=.01
    elif tamper=='wrong_pose':fresh['tcp_pose'][0]+=.01
    elif tamper=='second_session':
        p.observe(state('healthy'),fresh,15.,15.,controls());fresh=intro_return_packet(0,25.1,99)
    with pytest.raises(ValueError,match='session|handoff|return'):
        p.observe(state('waiting' if tamper=='not_healthy' else 'healthy'),fresh,15.1,15.1,controls())


def participant_for_intro(capture,display):
    from leica_tracker.ursim_validation import _Participant
    p=object.__new__(_Participant)
    p.capture=capture;p.display=display;p.intro_id=123;p.intro_ready_seen=False
    p.intro_ready_cutoff_ns=0;p.intro_return_proof=None
    return p


def test_participant_intro_uses_fixed_production_cutoff_then_genuine_next_ticket():
    from types import SimpleNamespace
    from leica_tracker.ipad_camera import IPadCapture
    from test_ipad_camera import jpeg
    clock=[10.];capture=IPadCapture(clock=lambda:clock[0],read_timeout=0)
    session=capture.begin('fixture');ticket=capture.ticket('fixture',session)
    capture.accept('fixture',session,ticket,jpeg());capture.read();old_ns=capture.source_monotonic_ns
    clock[0]=10.05;pending=capture.ticket('fixture',session)
    display=SimpleNamespace(physical_intro_ready=lambda identifier:identifier==123)
    p=participant_for_intro(capture,display)
    clock[0]=10.1
    assert p._intro_frame_ready(old_ns,10_100_000_000) is False
    assert p.intro_ready_cutoff_ns==10_100_000_000 and not p.intro_ready_seen
    with pytest.raises(ValueError):capture.accept('fixture',session,pending,jpeg())
    assert capture.read()==(False,None)
    clock[0]=10.2;ticket=capture.ticket('fixture',session)
    capture.accept('fixture',session,ticket,jpeg());assert capture.read()[0]
    assert p._intro_frame_ready(capture.source_monotonic_ns,10_200_000_000) is True
    assert p.intro_return_proof==dict(reference_release_ns=10_100_000_000,post_return_frame_source_ns=10_200_000_000)
    assert p.intro_ready_cutoff_ns==10_100_000_000


def test_participant_intro_does_not_accept_ready_flag_with_old_or_stale_image():
    from types import SimpleNamespace
    calls=[];display=SimpleNamespace(physical_intro_ready=lambda identifier:True)
    p=participant_for_intro(SimpleNamespace(invalidate_pending=lambda:calls.append('invalidated')),display)
    assert p._intro_frame_ready(10_000_000_000,10_100_000_000) is False
    assert p._intro_frame_ready(10_100_000_000,10_200_000_000) is False
    assert p._intro_frame_ready(10_200_000_000,10_600_000_000) is False
    assert calls==['invalidated'] and p.intro_return_proof is None and not p.intro_ready_seen
    assert p._intro_frame_ready(10_700_000_000,10_700_000_000) is True
    assert p.intro_ready_cutoff_ns==10_100_000_000


def test_intentional_native_hold_is_reported_separately_from_active_cadence():
    from leica_tracker.ursim_validation import cycle_proof
    fractions=[0,1,0,1,0,0]
    samples=[dict(t=t,robot_timestamp=t,fraction=x,joint_speed=.4,tcp_error_m=.12*x,
                  phase='settling' if i==4 else 'moving')
             for i,(t,x) in enumerate(zip([0,.8,3.,3.8,4.7,5.4],fractions))]
    result=cycle_proof(samples,require_controller_clock=True,native_hold_s=1.4)
    assert result['controller_cycle_s']==5.4 and result['controller_active_cycle_s']==pytest.approx(4.)
    assert result['native_hold_s']==1.4 and result['cadence_matched'] is True


def valid_result():
    def stage(name,hold):
        return dict(stage=name,native_hold_state=hold,waiting_observed=True,same_control=True,
            native_session=77,action_id=4 if name=='MOVE_LEFT' else None,intro_id=123,
            resumed_native_session=77,completed_action_id=4 if name=='MOVE_LEFT' else None,
            control={k:v for k,v in controls().items() if k!='active'},
            completed_control={k:v for k,v in controls().items() if k!='active'},
            stationary_hold=dict(dwell_s=.3,sampled_at=10.8,max_joint_speed=0.,tcp_linear_speed=0.,tcp_angular_speed=0.),resumed_at=11.3,
            native_hold_s=.7,completed=True,reference_release_ns=12000000000,
            post_return_frame_source_ns=12000000001,return_error_m=.0001,
            gap=dict(requested_ms=1200,started_at=10.,resume_not_before=11.2,resumed_at=11.21,last_sequence=8))
    return dict(communication_jitter_ms=1200,scenario='complete',operator_selected_binding_count=1,
        fault_injection=dict(kind='owner_heartbeat_gap',owner_start_count=1,
                             stages=[stage('intro',103),stage('MOVE_LEFT',4)]))


def test_explicit_jitter_results_cannot_attach_without_both_recovery_proofs():
    from leica_tracker.ursim_validation import validate_communication_recovery_proof
    assert validate_communication_recovery_proof({}) is True
    assert validate_communication_recovery_proof(valid_result()) is True
    for path in ('missing_stage','new_Start','rebind','old_frame','no_pause'):
        result=copy.deepcopy(valid_result())
        if path=='missing_stage':result['fault_injection']['stages'].pop()
        elif path=='new_Start':result['fault_injection']['owner_start_count']=2
        elif path=='rebind':result['operator_selected_binding_count']=2
        elif path=='old_frame':result['fault_injection']['stages'][1]['post_return_frame_source_ns']=12000000000
        else:result['fault_injection']['stages'][0]['waiting_observed']=False
        with pytest.raises(ValueError):validate_communication_recovery_proof(result)


def test_owner_gap_suppresses_only_heartbeats_until_original_deadline(monkeypatch):
    from types import SimpleNamespace
    from leica_tracker import ursim_validation as v
    times=iter([10.,10.5,11.3,11.31])
    monkeypatch.setattr(v,'time',SimpleNamespace(monotonic=lambda:next(times)))
    owner=v._Owner(SimpleNamespace());owner.control={'same':'control'};requests=[]
    gap=owner.begin_heartbeat_gap(1200)
    def request(path,body):
        requests.append((path,body));owner.closed.set()
    owner.request=request;owner._heartbeat()
    assert requests==[('/control-heartbeat',{'same':'control','sequence':1})]
    assert gap['resume_not_before']==11.2 and gap['resumed_at']==11.31
    assert owner.start_count==0 and owner.failure is None


def test_canonical_attachment_rejects_claimed_jitter_without_native_proofs(tmp_path):
    from leica_tracker.canonical_delivery import attach_native_validation
    from test_canonical_delivery import workspace,build,sources,native_case
    import json
    root=workspace(tmp_path); output=root/'bundle'; manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest); result['communication_jitter_ms']=1200
    (case/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):attach_native_validation(output,'ur10e',case,workspace=root)
    assert not (output/'simulation').exists()


def test_canonical_accepts_only_proven_hold_exclusion_and_sanitizes_recovery_summary(tmp_path):
    from leica_tracker.canonical_delivery import attach_native_validation
    from test_canonical_delivery import workspace,build,sources,native_case
    import json
    root=workspace(tmp_path); output=root/'bundle'; manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest); result.update(valid_result())
    left=result['directions']['MOVE_LEFT']
    left.update(native_hold_s=1.5,controller_cycle_s=6.,controller_moving_s=5.4,
        measured_cycle_s=6.,measured_moving_s=5.4,controller_active_cycle_s=4.5,controller_active_moving_s=3.9)
    result['fault_injection']['stages'][1]['native_hold_s']=1.5
    (case/'result.json').write_text(json.dumps(result))
    attach_native_validation(output,'ur10e',case,workspace=root)
    text=(output/'simulation/ur10e/validation-summary.json').read_text(); summary=json.loads(text)
    assert summary['directions']['MOVE_LEFT']['controller_cycle_s']==6.
    assert summary['directions']['MOVE_LEFT']['controller_active_cycle_s']==4.5
    assert summary['bounded_communication_recovery']['stages']==['intro','MOVE_LEFT']
    assert 'same-instance' not in text and 'control_started_ns' not in text


@pytest.mark.parametrize('tamper',['no_option','wrong_hold','bad_active','different_action','other_direction'])
def test_canonical_unknown_or_inconsistent_hold_cannot_waive_cadence(tmp_path,tamper):
    from leica_tracker.canonical_delivery import attach_native_validation
    from test_canonical_delivery import workspace,build,sources,native_case
    import json
    root=workspace(tmp_path); output=root/'bundle'; manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest); result.update(valid_result())
    left=result['directions']['MOVE_LEFT']
    left.update(native_hold_s=1.5,controller_cycle_s=6.,controller_moving_s=5.4,
        measured_cycle_s=6.,measured_moving_s=5.4,controller_active_cycle_s=4.5,controller_active_moving_s=3.9)
    result['fault_injection']['stages'][1]['native_hold_s']=1.5
    if tamper=='no_option':result.pop('communication_jitter_ms')
    elif tamper=='wrong_hold':left['native_hold_s']=2.
    elif tamper=='bad_active':left['controller_active_cycle_s']=4.4
    elif tamper=='different_action':left['action_id']=99
    else:result['directions']['MOVE_RIGHT'].update(left)
    (case/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):attach_native_validation(output,'ur10e',case,workspace=root)
    assert not (output/'simulation').exists()
