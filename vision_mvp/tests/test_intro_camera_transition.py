import pytest
from leica_tracker import physical_runtime as runtime
from leica_tracker.network_recovery import CommunicationRecovery
from leica_tracker.resident_guest_watchdog import GuestSnapshotRequestError


CONTROL={'instance':'test','epoch':1,'started_ns':1}


def snapshot(**changes):
    value={**CONTROL,'active':True,'control_fresh':True,'camera_connected':True,'camera_fresh':True}
    value.update(changes);return value


def test_return_waits_for_real_camera_before_opening_guidance():
    now=[0.];checks=[]
    states=iter([snapshot(camera_fresh=False),snapshot(camera_fresh=False),snapshot()])
    def sleep(dt):now[0]+=dt
    runtime.wait_for_intro_camera(lambda:next(states),CONTROL,lambda:checks.append(now[0]),clock=lambda:now[0],sleep=sleep)
    assert len(checks)==3 and now[0]>0


@pytest.mark.parametrize('change',[{'active':False},{'control_fresh':False},{'epoch':2},{'instance':'new'}])
def test_return_wait_cannot_continue_after_control_loss(change):
    with pytest.raises(RuntimeError,match='control'):
        runtime.wait_for_intro_camera(lambda:snapshot(**change),CONTROL,lambda:None)


def test_return_camera_wait_is_bounded_and_does_not_enable_any_motion():
    now=[0.]
    def sleep(dt):now[0]+=dt
    with pytest.raises(RuntimeError,match='camera'):
        runtime.wait_for_intro_camera(lambda:snapshot(camera_fresh=False),CONTROL,lambda:None,
            timeout=.1,clock=lambda:now[0],sleep=sleep)


def test_intro_backend_registration_is_explicit():
    requests=[]
    class Gate:
        def __init__(self,*args,**kwargs):pass
        def _request(self,method,path,body):requests.append(body);return {'ok':True,'lease':'lease'}
    runtime.register_backend(8767,'local-key',CONTROL,timeout=.1,factory=Gate,mode='intro')
    assert requests==[{'op':'register','mode':'intro'}]


def test_default_camera_wait_can_exceed_ten_seconds_while_control_remains_fresh():
    now=[0.];checks=[]
    def sleep(delta):now[0]+=delta
    def reader():return snapshot(camera_fresh=now[0]>=10.2)
    runtime.wait_for_intro_camera(reader,CONTROL,lambda:checks.append(now[0]),
                                 clock=lambda:now[0],sleep=sleep)
    assert now[0]>=10.2 and checks


def recovery_wait_fixture(read_state):
    """使用实际的相遇策略；只有回环读取器/时钟是假的。"""
    now=[10.];reads=[];checks=[]
    recovery=CommunicationRecovery(CONTROL,clock=lambda:now[0])
    def reader():
        reads.append(now[0])
        return read_state(now[0])
    reader.communication_recovery=recovery
    def check_source():checks.append((now[0],recovery.status))
    def sleep(delta):
        now[0]+=delta
        assert now[0]<13.2,'A handoff wait must not extend the fixed recovery window'
    return now,recovery,reader,reads,checks,check_source,sleep


def test_handoff_owner_gap_waits_for_same_encounter_healthy_dwell_without_resuming_motion():
    now,recovery,reader,reads,checks,check,sleep=recovery_wait_fixture(
        lambda at:snapshot(control_fresh=at>=10.2))
    runtime.wait_for_intro_camera(reader,CONTROL,check,clock=lambda:now[0],sleep=sleep)
    assert len(reads)>1 and now[0]>=10.5
    assert any(status=='waiting' for _,status in checks)
    assert recovery.status=='ready' and recovery.paused
    assert recovery.deadline==13.


@pytest.mark.parametrize('category',['timeout','connection'])
def test_handoff_recovers_only_known_guest_transport_gaps(category):
    attempts=[]
    def state(at):
        attempts.append(at)
        if at<10.2:raise GuestSnapshotRequestError(category,'request',100.)
        return snapshot()
    now,recovery,reader,reads,checks,check,sleep=recovery_wait_fixture(state)
    runtime.wait_for_intro_camera(reader,CONTROL,check,clock=lambda:now[0],sleep=sleep)
    assert len(attempts)>1 and now[0]>=10.5
    assert recovery.status=='ready' and recovery.paused and recovery.deadline==13.


@pytest.mark.parametrize('change',[
    {'active':False},{'control_action':'stop'},{'epoch':2},
    {'instance':'replacement'},{'started_ns':2},{'camera_fresh':1},
])
def test_handoff_recovery_never_continues_after_stop_identity_or_schema_change(change):
    now,recovery,reader,reads,checks,check,sleep=recovery_wait_fixture(
        lambda at:snapshot(camera_fresh=False) if at<10.1 else snapshot(**change))
    with pytest.raises(RuntimeError):
        runtime.wait_for_intro_camera(reader,CONTROL,check,clock=lambda:now[0],sleep=sleep)
    assert recovery.failure is not None


def test_handoff_repeated_owner_loss_cannot_extend_original_three_second_deadline():
    now,recovery,reader,reads,checks,check,sleep=recovery_wait_fixture(
        lambda at:snapshot(control_fresh=False,camera_fresh=at>=11.))
    with pytest.raises(RuntimeError,match='expired'):
        runtime.wait_for_intro_camera(reader,CONTROL,check,clock=lambda:now[0],sleep=sleep)
    assert 13.<=now[0]<13.03 and recovery.deadline==13.


def test_handoff_recovery_keeps_reference_source_checks_active():
    now,recovery,reader,reads,checks,check,sleep=recovery_wait_fixture(
        lambda at:snapshot(control_fresh=False))
    def stopped_source():
        if now[0]>=10.1:raise RuntimeError('operator stopped handoff')
    with pytest.raises(RuntimeError,match='operator stopped handoff'):
        runtime.wait_for_intro_camera(reader,CONTROL,stopped_source,clock=lambda:now[0],sleep=sleep)
    assert 10.1<=now[0]<10.13


def test_handoff_communication_metadata_without_shared_policy_cannot_authorize_owner_gap():
    value=snapshot(control_fresh=False,communication={
        'status':'waiting','recoverable':True,'remaining_ms':2900})
    with pytest.raises(RuntimeError,match='control'):
        runtime.wait_for_intro_camera(lambda:value,CONTROL,lambda:None)


@pytest.mark.parametrize('error',[
    GuestSnapshotRequestError('http_rejected','headers',1.,403),
    GuestSnapshotRequestError('protocol','body',1.),
    GuestSnapshotRequestError('response','body',1.),
    TimeoutError('unclassified timeout'),
])
def test_handoff_recovery_rejects_terminal_or_unclassified_transport_failures(error):
    def state(at):raise error
    now,recovery,reader,reads,checks,check,sleep=recovery_wait_fixture(state)
    with pytest.raises((RuntimeError,TimeoutError)):
        runtime.wait_for_intro_camera(reader,CONTROL,check,clock=lambda:now[0],sleep=sleep)
    assert recovery.failure is not None
