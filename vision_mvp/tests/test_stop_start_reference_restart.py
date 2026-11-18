"""新的明确的开始可能会证明旧的参考，永远不要盲目地澄清它。"""
from types import SimpleNamespace
import pytest

from leica_tracker import physical_runtime as runtime
from leica_tracker.reference_gate import ReferenceGate, RestartReferenceProof
from test_physical_lifecycle import control, Source
from test_physical_interaction_supervisor import Immediate


def restart_fixture(monkeypatch, *, displaced=False, allow=True, selected=False):
    now=[10.]
    calls=[]
    reference_pose=[.1,-.5,.3,0.,0.,0.]
    gate=ReferenceGate()
    session_id=gate.begin(reference_pose,1.)
    gate.acknowledge()
    gate.steady_since=1.;gate.last_sample=2.
    class Reference:
        restart_reference=tuple(reference_pose)
        proof=gate
        def status(self, _):
            return dict(blocked=gate.blocked,session=gate.session,acknowledged=gate.acknowledged)
        def restart_begin(self, session, *, reanchor=False):
            assert session==session_id
            calls.append('restart_begin')
            gate.steady_since=gate.last_sample=None
            if reanchor:self.proof=RestartReferenceProof(gate);calls.append('selected_reanchor')
            return True
        def restart_sample(self,session,pose,speed,at):
            self.proof.sample(session,pose,speed,at,now[0])
            calls.append('sample')
            return dict(ok=True,sampled_at=at,return_verified=self.proof.return_verified(session,now[0]))
        def restart_release(self,session,pose=None,speed=None,sampled_at=None):
            if pose is not None:self.proof.sample(session,pose,speed,sampled_at,now[0])
            if not self.proof.return_verified(session,now[0]):
                raise RuntimeError('Restart requires the previous reference')
            committed=self.proof.reference
            self.proof.release(session,now[0]);calls.append('release')
            if selected:self.restart_reference=tuple(committed)
            return gate.release_ns
        def fault(self):calls.append('fault')
    monkeypatch.setattr(runtime,'register_backend',lambda *a,**k:Reference())
    monkeypatch.setattr(runtime,'FeedbackPublisher',lambda *a,**k:SimpleNamespace(close=lambda:calls.append('publisher_closed')))
    monkeypatch.setattr(runtime,'LeaseHeartbeat',lambda *a:SimpleNamespace(failure=None,close=lambda:calls.append('lease_closed')))
    monkeypatch.setattr(runtime,'time',SimpleNamespace(monotonic=lambda:now[0],sleep=lambda _:None))
    cfg=SimpleNamespace(robot_model='ur5e')
    profile=SimpleNamespace(host='offline',cfg=cfg,gate_timeout=.1)
    if selected:profile.reference_restart_policy='capture_stationary_on_explicit_start'
    bound=SimpleNamespace(host='offline',cfg=cfg,gate_timeout=.1)
    def bind(**options):
        calls.append('capture')
        assert 'sample_observer' in options
        for i in range(8):
            now[0]=10.1+i*.1
            pose=reference_pose[:]
            if displaced:pose[0]+=.01
            options['sample_observer']({},dict(tcp_pose=pose,tcp_speed=[0.]*6,joint_speed=[0.]*6),now[0])
        now[0]=10.9
        options['proof_observer']({},dict(tcp_pose=pose,tcp_speed=[0.]*6,joint_speed=[0.]*6),now[0])
        # 真正的仅接收SDK拆解可能会消耗几乎整个样本年龄预算。发布必须在开始之前已提交。
        now[0]+=.14
        calls.append('capture_closed')
        return bound
    profile.bind_before_motion=bind
    motion_gate=SimpleNamespace(can_send=True,stop=lambda _:None)
    def step(event):raise RuntimeError('fake encounter finished')
    sup=SimpleNamespace(close_workers=lambda:calls.append('reference_workers_closed'))
    session=SimpleNamespace(gate=motion_gate,stop=lambda:calls.append('native_stopped'),close=lambda:calls.append('native_closed'))
    active_runtime=SimpleNamespace(step=step,supervisor=sup)
    def intro(*args,**kwargs):
        assert not gate.blocked
        calls.append('native_open')
        return session,active_runtime,SimpleNamespace(close=lambda:calls.append('dashboard_closed'))
    def trigger(*args,**kwargs):
        calls.append('fresh_trigger' if 'release' in calls else 'trigger')
        if 'release' in calls:assert kwargs['observed_at']>=gate.release_ns/1e9
        return dict(event='physical_intro',intro_id=1,source_monotonic_ns=round(now[0]*1e9))
    monkeypatch.setattr(runtime,'wait_for_trigger',trigger)
    lifecycle={}
    def run():
        runtime.run_session(profile,8767,'offline',Source(),lambda:control(),before=control(),
            dependencies={'intro_runner':intro,'startup_reference_worker':Immediate},
            allow_reference_rearm=allow,lifecycle=lifecycle)
    return run,calls,gate,lifecycle


def test_explicit_restart_proves_original_reference_before_any_new_native_open(monkeypatch):
    run,calls,gate,lifecycle=restart_fixture(monkeypatch)
    with pytest.raises(RuntimeError,match='fake encounter finished'):run()
    assert calls.index('restart_begin')<calls.index('capture')<calls.index('release')
    assert calls.index('release')<calls.index('capture_closed')
    assert calls.index('release')<calls.index('fresh_trigger')<calls.index('native_open')
    assert calls.index('reference_workers_closed')<calls.index('publisher_closed')
    assert not gate.blocked and lifecycle['reference_owned']


def test_stop_away_from_reference_never_redefines_it_or_automatically_returns(monkeypatch):
    run,calls,gate,_=restart_fixture(monkeypatch,displaced=True)
    old_reference=gate.reference
    with pytest.raises(RuntimeError,match='previous reference'):run()
    assert gate.blocked and gate.reference==old_reference
    assert 'release' not in calls and 'native_open' not in calls


def test_selected_design_explicit_new_start_binds_stationary_current_pose_after_stop(monkeypatch):
    run,calls,gate,lifecycle=restart_fixture(monkeypatch,displaced=True,selected=True)
    with pytest.raises(RuntimeError,match='fake encounter finished'):run()
    assert 'selected_reanchor' in calls
    assert calls.index('release')<calls.index('capture_closed')<calls.index('native_open')
    assert not gate.blocked and lifecycle['reference_owned']


def test_blocked_gate_without_clean_prior_lineage_still_refuses_start(monkeypatch):
    run,calls,gate,lifecycle=restart_fixture(monkeypatch,allow=False)
    with pytest.raises(RuntimeError,match='no automatic reset'):run()
    assert gate.blocked and 'restart_begin' not in calls and not lifecycle


def test_same_service_allows_rearm_only_after_clean_close_and_measured_stop(monkeypatch):
    snapshots=[control(1),control(2,False,'stop'),control(3),control(4)]
    snapshots[-1]['instance']='new-display'
    calls=[]
    def run(*args,**options):
        calls.append((options['before']['epoch'],options['allow_reference_rearm']))
        options['lifecycle']['reference_owned']=True
        raise RuntimeError('user stopped')
    monkeypatch.setattr(runtime,'run_session',run)
    runtime.serve_sessions(SimpleNamespace(host='offline'),8767,'local',Source(),
        read_guest=lambda:snapshots.pop(0),stopped_check=lambda _:True,report=lambda *a:None,
        is_alive=lambda:bool(snapshots),sleep=lambda _:None)
    assert calls==[(1,False),(3,True),(4,False)]


def test_cleanup_failure_prevents_future_rearm_attempts(monkeypatch):
    snapshots=[control(1),control(2),control(3)]
    calls=[]
    def run(*args,**options):
        calls.append(options['before']['epoch'])
        options['lifecycle']['reference_owned']=True
        raise runtime.SessionCleanupError('native writer still alive')
    monkeypatch.setattr(runtime,'run_session',run)
    runtime.serve_sessions(SimpleNamespace(host='offline'),8767,'local',Source(),
        read_guest=lambda:snapshots.pop(0),stopped_check=lambda _:True,report=lambda *a:None,
        is_alive=lambda:bool(snapshots),sleep=lambda _:None)
    assert calls==[1]
