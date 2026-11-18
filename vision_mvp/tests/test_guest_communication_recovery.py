"""验证协商后的通信暂停，不启动监听服务或机器人接口。"""
from types import SimpleNamespace
from email.message import Message
import io
import json

import pytest

from leica_tracker import guest_display


@pytest.fixture
def encounter(monkeypatch):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(guest_display, 'time', SimpleNamespace(monotonic=lambda:clock.now))
    server = SimpleNamespace(server_address=('127.0.0.1',12345),serve_forever=lambda:None)
    monkeypatch.setattr(guest_display, 'ThreadingHTTPServer', lambda *_:server)
    class Camera:
        _last_received=100.
        def fresh(self):return 0<=clock.now-self._last_received<=.3
        stream_fresh=fresh
        def connected(self):return clock.now-self._last_received<=2.5
        def diagnostics(self):return {}
    d=guest_display.GuestDisplay(remote_capture=Camera(),physical_feedback=True)
    d._set_action_locked('start','owner',1)
    identity={'instance':d._control_instance,'epoch':d._control_epoch}
    result=d._physical_feedback_update({'op':'register',**identity,'recovery_window_s':3.0})
    backend={**identity,'lease':result['lease']}
    return d,clock,identity,backend


def renew(d,clock,identity,backend,sequence):
    d.remote_capture._last_received=clock.now
    d._control_heartbeat_locked({**identity,'sequence':sequence},'owner',clock.now)
    d._physical_feedback_update({'op':'heartbeat',**backend})
    return d.control_snapshot()


def request(d,path,body=None):
    handler=object.__new__(d._handler());handler.path=path
    handler.client_address=('127.0.0.1',1234);handler.authorized=lambda:True
    handler.headers=Message();raw=json.dumps(body).encode()
    for key,value in {'Host':'127.0.0.1:8777','Origin':'http://127.0.0.1:8777',
            'Content-Type':'application/json','Content-Length':str(len(raw)),
            'X-Guest-Client':'owner'}.items():handler.headers[key]=value
    handler.rfile=io.BytesIO(raw);handler.connection=SimpleNamespace(settimeout=lambda _:None)
    replies=[];handler.reply=lambda code,data,*a,**kw:replies.append((code,data))
    (handler.do_GET if body is None else handler.do_POST)()
    return replies[-1]


def test_short_control_loss_holds_without_revocation_or_epoch_change_and_needs_resume(encounter):
    d,clock,identity,backend=encounter
    clock.now=100.31
    d.remote_capture._last_received=clock.now;d._physical_backend_at=clock.now
    lost=d.control_snapshot()
    assert lost['active'] and lost['epoch']==identity['epoch'] and not lost['control_fresh']
    assert lost['communication']['status']=='waiting' and lost['communication']['recoverable']
    assert not d.physical_backend_available() and not d._physical_revoked
    assert d._physical_revocation is None
    renew(d,clock,identity,backend,1)
    for sequence in range(2,6):
        clock.now+=.1
        ready=renew(d,clock,identity,backend,sequence)
    assert ready['communication']['status']=='ready'
    assert not d.physical_backend_available(),'ready is not the native resume acknowledgement'
    result=d._physical_feedback_update({'op':'resume',**backend})
    assert result['communication']['status']=='healthy'
    assert d.physical_backend_available() and d._control_epoch==identity['epoch']


def test_camera_only_gap_holds_and_late_same_owner_heartbeat_is_accepted(encounter):
    d,clock,identity,backend=encounter
    clock.now+=.31;d._seen=clock.now;d._physical_backend_at=clock.now
    lost=d.control_snapshot()
    assert lost['communication']['status']=='waiting' and not lost['camera_fresh']
    assert not d.physical_backend_available() and d._active
    assert d._control_heartbeat_locked({**identity,'sequence':1},'owner',clock.now)['accepted']
    with pytest.raises(ValueError):d._physical_feedback_update({'op':'resume',**backend})


def test_fixed_recovery_deadline_cannot_be_extended_by_flapping_or_heartbeat(encounter):
    d,clock,identity,backend=encounter
    clock.now=100.31
    first=d.control_snapshot()['communication']['remaining_ms']
    assert 2980<=first<=3000
    clock.now=102.9
    recovered=renew(d,clock,identity,backend,1)
    assert 390<=recovered['communication']['remaining_ms']<=410
    clock.now=103.31
    ended=d.control_snapshot()
    assert not ended['active'] and ended['epoch']==identity['epoch']+1
    assert ended['communication']['recoverable'] is False and d._physical_revocation is not None
    with pytest.raises(ValueError):renew(d,clock,identity,backend,2)


def test_loss_already_past_deadline_cannot_gain_new_three_seconds_on_first_read(encounter):
    d,clock,identity,backend=encounter
    clock.now=104.
    assert not d.control_snapshot()['active']
    with pytest.raises(ValueError):d._physical_feedback_update({'op':'heartbeat',**backend})


@pytest.mark.parametrize('terminal',['stop','fault'])
def test_stop_or_native_backend_fault_never_recovers_with_old_identity(encounter,terminal):
    d,clock,identity,backend=encounter
    clock.now+=.31;d.control_snapshot()
    if terminal=='stop':d._set_action_locked('stop','owner',1)
    else:d._physical_feedback_update({'op':'fault',**backend})
    with pytest.raises(ValueError):renew(d,clock,identity,backend,1)
    with pytest.raises(ValueError):d._physical_feedback_update({'op':'resume',**backend})
    assert not d.physical_backend_available()


def test_stability_restarts_on_another_real_camera_gap(encounter):
    d,clock,identity,backend=encounter
    clock.now+=.31;d.control_snapshot()
    renew(d,clock,identity,backend,1)
    clock.now+=.2;renew(d,clock,identity,backend,2)
    clock.now+=.31;d._seen=clock.now;d._physical_backend_at=clock.now
    assert d.control_snapshot()['communication']['status']=='waiting'
    renew(d,clock,identity,backend,3)
    clock.now+=.1
    assert renew(d,clock,identity,backend,4)['communication']['status']=='waiting'


def test_duplicate_wrong_owner_or_changed_identity_cannot_renew_hold(encounter):
    d,clock,identity,backend=encounter
    clock.now+=.31;d.control_snapshot()
    before=d._seen
    for bad,owner in ((identity,'stranger'),({**identity,'epoch':99},'owner')):
        with pytest.raises(ValueError):d._control_heartbeat_locked({**bad,'sequence':1},owner,clock.now)
    assert d._seen==before
    assert d._control_heartbeat_locked({**identity,'sequence':2},'owner',clock.now)['accepted']
    clock.now+=.1
    assert not d._control_heartbeat_locked({**identity,'sequence':1},'owner',clock.now)['accepted']
    assert d._seen<clock.now


@pytest.mark.parametrize('window',[True,0,2.9,3.1,'3'])
def test_invalid_recovery_negotiation_does_not_register_backend(encounter,window):
    d,clock,identity,backend=encounter
    d._set_action_locked('start','owner',1)
    identity={**identity,'epoch':d._control_epoch}
    with pytest.raises(ValueError):d._physical_feedback_update({'op':'register',**identity,'recovery_window_s':window})
    assert d._physical_registered_epoch!=d._control_epoch


def test_legacy_backend_still_revokes_instead_of_auto_recovery(encounter):
    d,clock,identity,backend=encounter
    d._set_action_locked('start','owner',1)
    identity={**identity,'epoch':d._control_epoch}
    d._physical_feedback_update({'op':'register',**identity})
    clock.now+=.31
    assert not d.physical_backend_available() and d._physical_revoked
    assert not d.control_snapshot()['communication']['recoverable']


def test_public_and_operator_states_expose_real_hold_not_fault(encounter):
    d,clock,identity,backend=encounter
    clock.now+=.31
    for path in ('/state','/operator-state'):
        code,state=request(d,path)
        assert code==200 and state['communication']['status']=='waiting'
        assert state['active'] and not state['control_fresh'] and not state['camera_fresh']
        assert state['runtime']=={'epoch':identity['epoch'],'status':'holding','reason':'communication'}


def test_communication_wait_never_conceals_an_actual_runtime_fault(encounter):
    d,clock,identity,backend=encounter
    d._runtime_status={'epoch':identity['epoch'],'status':'fault','reason':'speed'}
    clock.now+=.31
    assert request(d,'/operator-state')[1]['runtime']['status']=='fault'


def test_reference_begin_reply_is_deduplicated_during_hold_with_same_payload(encounter):
    d,clock,identity,backend=encounter
    body={'op':'begin',**backend,'reference':[0.]*6,'request_id':'a'*32}
    code,first=request(d,'/reference-gate',body)
    assert code==200 and first['session']
    clock.now+=.31
    assert request(d,'/reference-gate',body)==(200,first)
    assert request(d,'/reference-gate',{**body,'reference':[1.,0,0,0,0,0]})[0]==400
    assert request(d,'/reference-gate',{**body,'request_id':'b'*32})[0]==400
    assert request(d,'/reference-gate',{**body,'lease':'wrong'})[0]==400


def test_reference_release_reply_survives_resume_but_not_stop(encounter):
    d,clock,identity,backend=encounter
    session=d._reference_gate.begin([0.]*6,clock.now);d._reference_gate.acknowledge()
    d._reference_gate.steady_since=clock.now-.7;d._reference_gate.last_sample=clock.now
    body={'op':'release',**backend,'session':session,'request_id':'c'*32}
    code,first=request(d,'/reference-gate',body)
    assert code==200 and first['release_ns']>0 and not d._reference_gate.blocked
    clock.now+=.31;d.control_snapshot()
    assert request(d,'/reference-gate',body)==(200,first)
    renew(d,clock,identity,backend,1)
    for sequence in range(2,6):clock.now+=.1;renew(d,clock,identity,backend,sequence)
    d._physical_feedback_update({'op':'resume',**backend})
    assert request(d,'/reference-gate',body)==(200,first)
    d._set_action_locked('stop','owner',1)
    assert request(d,'/reference-gate',body)[0]==400


def test_registration_reply_can_be_recovered_without_second_lease(encounter):
    d,clock,identity,backend=encounter
    d._set_action_locked('start','owner',1);identity={**identity,'epoch':d._control_epoch}
    body={'op':'register',**identity,'recovery_window_s':3.0,'request_id':'d'*32}
    first=d._physical_feedback_update(body)
    clock.now+=.31
    assert d._physical_feedback_update(body)==first
    with pytest.raises(ValueError):d._physical_feedback_update({**body,'mode':'intro'})


@pytest.mark.parametrize('request_id',['',True,'a'*31,'G'*32,'a'*33])
def test_bad_reference_request_identity_never_mutates_gate(encounter,request_id):
    d,clock,identity,backend=encounter
    assert request(d,'/reference-gate',{'op':'begin',**backend,'reference':[0.]*6,'request_id':request_id})[0]==400
    assert not d._reference_gate.blocked


def test_hold_removes_old_photo_permission_and_resume_needs_fresh_arrival(encounter):
    d,clock,identity,backend=encounter
    d.arm_physical_photo(round(clock.now*1e9));d._physical_photo['ready_at']=clock.now
    d._photo_latched=True
    completed_share=object();d._photo_share=completed_share
    clock.now+=.31;d.control_snapshot()
    assert d._physical_photo is None and not d._photo_latched
    assert d._photo_share is completed_share,'already published photo bytes are retained'
    renew(d,clock,identity,backend,1)
    for sequence in range(2,6):clock.now+=.1;renew(d,clock,identity,backend,sequence)
    d._physical_feedback_update({'op':'resume',**backend})
    assert not d._photo_ready_locked() and d._physical_photo is None


def test_recoverable_hold_retains_same_action_but_only_new_measured_sample_restores_light(encounter):
    d,clock,identity,backend=encounter
    action={'action_id':7,'source_ns':round(clock.now*1e9),'cue':'MOVE_RIGHT'}
    d._physical_feedback_update({'op':'begin',**backend,**action,'reference_position':[0.,0.,0.],
        'direction':[1.,0.,0.],'planned_displacement':.1})
    d._physical_feedback_update({'op':'sample',**backend,**action,'captured_at':clock.now,
        'current_position':[.05,0.,0.],'phase':'moving'})
    assert d._current_feedback()['gesture_id']==7
    clock.now+=.31
    assert d._current_feedback()['intensity']==0
    assert d._physical_action['action_id']==7 and not d._current_motion()['active']
    renew(d,clock,identity,backend,1)
    for sequence in range(2,6):clock.now+=.1;renew(d,clock,identity,backend,sequence)
    d._physical_feedback_update({'op':'resume',**backend})
    assert d._current_feedback()['intensity']==0,'old pose cannot relight the resumed native phase'
    d._physical_feedback_update({'op':'sample',**backend,**action,'captured_at':clock.now,
        'current_position':[.06,0.,0.],'phase':'moving'})
    assert d._current_feedback()['gesture_id']==7 and d._current_feedback()['intensity']>0
    assert d._current_motion()['action_id']==7
    d._physical_feedback_update({'op':'fault',**backend})
    assert d._physical_action is None and d._current_feedback()['intensity']==0


@pytest.mark.parametrize('stage',['waiting','ready'])
def test_inflight_feedback_rejection_during_hold_preserves_action_without_accepting_sample_or_begin(encounter,stage):
    d,clock,identity,backend=encounter
    action={'action_id':7,'source_ns':round(clock.now*1e9),'cue':'MOVE_RIGHT'}
    begin={'op':'begin',**backend,**action,'reference_position':[0.,0.,0.],
        'direction':[1.,0.,0.],'planned_displacement':.1}
    d._physical_feedback_update(begin)
    d._physical_feedback_update({'op':'sample',**backend,**action,'captured_at':clock.now,
        'current_position':[.05,0.,0.],'phase':'moving'})
    accepted=d._physical_action;captured=accepted['captured_at'];feedback=dict(accepted['feedback'])
    clock.now+=.31;d.control_snapshot()
    sequence=1
    if stage=='ready':
        renew(d,clock,identity,backend,sequence)
        for sequence in range(2,6):clock.now+=.1;renew(d,clock,identity,backend,sequence)
    assert d.control_snapshot()['communication']['status']==stage
    late={'op':'sample',**backend,**action,'captured_at':clock.now-.02,
        'current_position':[.07,0.,0.],'phase':'moving'}
    for packet in (late,{**late,'action_id':6},{**begin,'action_id':8}):
        with pytest.raises(ValueError):d._physical_feedback_update(packet)
        assert d._physical_action is accepted
        assert accepted['captured_at']==captured and accepted['feedback']==feedback
        assert d._current_feedback()['intensity']==0 and not d._current_motion()['active']
        assert d._physical_last_id==7,'a rejected begin cannot replace the accepted native action'
    if stage=='waiting':
        renew(d,clock,identity,backend,sequence)
        for sequence in range(2,6):clock.now+=.1;renew(d,clock,identity,backend,sequence)
    d._physical_feedback_update({'op':'resume',**backend})
    assert d._current_feedback()['intensity']==0
    with pytest.raises(ValueError):d._physical_feedback_update(begin)
    d._physical_feedback_update({**late,'captured_at':clock.now,'current_position':[.06,0.,0.]})
    assert d._current_motion()['action_id']==7 and d._current_feedback()['gesture_id']==7
    d._set_action_locked('stop','owner',1)
    assert d._current_feedback()['intensity']==0 and d._physical_action is None


def pending_start(encounter, *, recovery=True):
    d,clock,identity,backend=encounter
    assert request(d,'/action',{'action':'stop'})[0]==200
    identity={'instance':d._control_instance,'epoch':d._control_epoch}
    body={'action':'start','control_protocol':1,'control':identity}
    if recovery:body['recovery_window_s']=3.0
    code,result=request(d,'/action',body)
    assert code==200
    return d,clock,result['control'],result


def test_opted_in_start_survives_initial_400ms_gap_without_backend_or_replaying_start(encounter):
    d,clock,identity,result=pending_start(encounter)
    assert result['communication']=={'status':'healthy','recoverable':True,'remaining_ms':0}
    assert d._physical_registered_epoch!=identity['epoch'] and not d.physical_backend_available()
    clock.now+=.4;d.remote_capture._last_received=clock.now
    state=d.control_snapshot()
    assert state['active'] and state['epoch']==identity['epoch'] and not state['control_fresh']
    assert state['communication']['status']=='waiting'
    assert d._control_heartbeat_locked({**identity,'sequence':1},'owner',clock.now)['accepted']
    assert d._control_epoch==identity['epoch'] and not d.physical_backend_available()
    assert not d._physical_credentials_locked(clock.now),'pending Start is never a robot credential'


def test_backend_must_negotiate_pending_start_and_inherit_its_fixed_deadline(encounter):
    d,clock,identity,result=pending_start(encounter)
    clock.now+=.4;d.remote_capture._last_received=clock.now;d.control_snapshot()
    deadline=d._communication_deadline
    d._control_heartbeat_locked({**identity,'sequence':1},'owner',clock.now)
    with pytest.raises(ValueError):d._physical_feedback_update({'op':'register',**identity})
    result=d._physical_feedback_update({'op':'register',**identity,'recovery_window_s':3.0})
    assert result['lease'] and d._communication_deadline==deadline
    assert d.control_snapshot()['communication']['status']=='waiting' and not d.physical_backend_available()


@pytest.mark.parametrize('stop',[False,True])
def test_pending_start_stop_or_expired_first_gap_cannot_be_revived(encounter,stop):
    d,clock,identity,result=pending_start(encounter)
    if stop:d._set_action_locked('stop','owner',1)
    else:clock.now+=3.4
    with pytest.raises(ValueError):d._control_heartbeat_locked({**identity,'sequence':1},'owner',clock.now)
    assert not d.control_snapshot()['active'] and not d.physical_backend_available()


def test_legacy_start_initial_gap_remains_terminal(encounter):
    d,clock,identity,result=pending_start(encounter,recovery=False)
    clock.now+=.4;d.remote_capture._last_received=clock.now
    with pytest.raises(ValueError):d._control_heartbeat_locked({**identity,'sequence':1},'owner',clock.now)
    assert not d.control_snapshot()['communication']['recoverable']


@pytest.mark.parametrize('window',[None,True,2.9,'3'])
def test_bad_start_recovery_opt_in_never_activates_new_epoch(encounter,window):
    d,clock,identity,backend=encounter
    assert request(d,'/action',{'action':'start','control_protocol':1,'control':identity,
        'recovery_window_s':window})[0]==400
    assert d._control_epoch==identity['epoch']
