"""实测状态视图不提供运动接口，模拟样本也不能触发自动播放。"""
from contextlib import nullcontext
import copy
import http.client
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from leica_tracker import ursim_viewer as viewer
from leica_tracker.ursim_system import DockerIdentity
from test_physical_system import workspace


@pytest.mark.parametrize('model,lengths,offsets',[
    ('ur10e',(.6127,.57155),(.1807,.17415,.11985,.11655)),
    ('ur5e',(.425,.3922),(.1625,.1333,.0997,.0996))])
def test_nontrivial_dh_pose_matches_vertical_arm_geometry_and_orientation(model,lengths,offsets):
    # 基座 +90°、肩关节 −90° 时，两段长连杆竖直向上。
    # 在制造商定义的基座坐标系中，各腕部偏移依次沿 +X、−Y、+X。
    result=viewer.forward_kinematics([math.pi/2,-math.pi/2,0.,0.,0.,0.],model)
    d1,d4,d5,d6=offsets
    assert result['origins'][-1]==pytest.approx([d4+d6,-d5,d1+sum(lengths)])
    rotation=[row[:3] for row in result['frames'][-1][:3]]
    assert [v for row in rotation for v in row]==pytest.approx([0,0,1,0,1,0,-1,0,0],abs=1e-12)
    assert len(result['origins'])==7 and result['origins'][0]==[0.,0.,0.]


@pytest.mark.parametrize('q',[[0.]*5,[0.,0.,False,0.,0.,0.],[float('nan')]*6])
def test_invalid_angles_never_make_a_display_pose(q):
    with pytest.raises(ValueError):viewer.forward_kinematics(q)


def source(clock):
    feedback={'timestamp':1.,'joints':[.3,-1.2,-1.4,-.8,1.4,.2],
              'tcp_pose':[.4,-.2,.6,0.,0.,0.]}
    receiver=SimpleNamespace(latest=(feedback,clock[0]),failure=None)
    primary=SimpleNamespace(latest=({'mode':{'mode':7,'running':False}},clock[0]),failure=None)
    return receiver,primary


def test_state_uses_actual_measurements_and_does_not_refresh_repeated_robot_time():
    clock=[10.];receiver,primary=source(clock)
    state=viewer.MeasuredState(receiver,primary,clock=lambda:clock[0]);first=state.snapshot()
    assert first['fresh'] and first['q']==receiver.latest[0]['joints']
    assert first['tcp_pose']==receiver.latest[0]['tcp_pose'] and first['sampled_at']==10.
    clock[0]=10.2;receiver.latest=(receiver.latest[0],10.2)
    assert state.snapshot()['sampled_at']==10.,'polling or repeated robot time cannot redate old q'
    clock[0]=10.8;stale=state.snapshot()
    assert not stale['fresh'] and stale['sample_age_ms']==800.
    assert stale['q']==first['q'] and stale['origins']==first['origins']


def test_waiting_and_readonly_reconnect_keep_last_pose_until_new_samples():
    clock=[10.];receiver,primary=source(clock)
    state=viewer.MeasuredState(None,None,clock=lambda:clock[0])
    assert state.snapshot()['q'] is None
    state.attach(receiver,primary);first=state.snapshot()
    state.attach(None,None)
    assert state.snapshot()['q']==first['q'] and not state.snapshot()['fresh']
    clock[0]+=.1;new_receiver,new_primary=source(clock)
    new_receiver.latest[0]['timestamp']=2.;new_receiver.latest[0]['joints'][0]=.5
    state.attach(new_receiver,new_primary)
    assert state.snapshot()['fresh'] and state.snapshot()['q'][0]==.5


def test_reconnect_accepts_restarted_controller_clock_only_after_new_fresh_measurement():
    clock=[10.];receiver,primary=source(clock)
    receiver.latest[0]['timestamp']=60000.
    state=viewer.MeasuredState(receiver,primary,clock=lambda:clock[0]);first=state.snapshot()
    state.attach(None,None)
    clock[0]=10.1;new_receiver,new_primary=source(clock)
    new_receiver.latest=(None,0.)
    state.attach(new_receiver,new_primary)
    waiting=state.snapshot()
    assert waiting['q']==first['q'] and waiting['sampled_at']==10. and not waiting['fresh']
    clock[0]=10.2;feedback=copy.deepcopy(receiver.latest[0]);feedback['timestamp']=1.;feedback['joints'][0]=.7
    new_receiver.latest=(feedback,10.2);new_primary.latest=(new_primary.latest[0],10.2)
    recovered=state.snapshot()
    assert recovered['fresh'] and recovered['robot_timestamp']==1.
    assert recovered['q'][0]==.7 and recovered['sampled_at']==10.2


def test_reattaching_same_readers_cannot_reset_a_backward_controller_clock():
    clock=[10.];receiver,primary=source(clock)
    state=viewer.MeasuredState(receiver,primary,clock=lambda:clock[0]);first=state.snapshot()
    clock[0]=10.2;feedback=copy.deepcopy(receiver.latest[0]);feedback['timestamp']=.5;feedback['joints'][0]=.7
    receiver.latest=(feedback,10.2);primary.latest=(primary.latest[0],10.2)
    state.attach(receiver,primary)
    invalid=state.snapshot()
    assert not invalid['fresh'] and invalid['q']==first['q'] and invalid['sampled_at']==10.
    state.attach(None,None);state.attach(receiver,primary)
    assert not state.snapshot()['fresh'] and state.snapshot()['sampled_at']==10.


@pytest.mark.parametrize('failed_source',['receiver','primary','stale_primary','missing_primary'])
def test_reconnect_does_not_replace_last_pose_from_failed_or_unqualified_readers(failed_source):
    clock=[10.];receiver,primary=source(clock)
    state=viewer.MeasuredState(receiver,primary,clock=lambda:clock[0]);first=state.snapshot()
    clock[0]=10.2;new_receiver,new_primary=source(clock)
    new_receiver.latest[0]['timestamp']=2.;new_receiver.latest[0]['joints'][0]=.7
    if failed_source=='receiver':new_receiver.failure='receive worker failed'
    elif failed_source=='primary':new_primary.failure='primary stream failed'
    elif failed_source=='stale_primary':new_primary.latest=(new_primary.latest[0],9.)
    else:new_primary.latest=(None,0.)
    state.attach(new_receiver,new_primary)
    invalid=state.snapshot()
    assert not invalid['fresh'] and invalid['q']==first['q'] and invalid['sampled_at']==10.


def test_readonly_owner_verifies_identity_and_opens_only_exact_receive_factories(monkeypatch):
    identity=DockerIdentity('c','g','n');events=[]
    state=viewer.MeasuredState(None,None)
    monkeypatch.setattr(viewer,'verify_docker',lambda _:identity)
    def primary(host):
        events.append(('primary',host));return SimpleNamespace(close=lambda:events.append('close-primary'),failure=None)
    def rtde(host,**kwargs):
        events.append(('rtde',host,kwargs));return SimpleNamespace(close=lambda:events.append('close-rtde'),failure=None)
    monkeypatch.setattr(viewer,'SimStateReceiver',primary);monkeypatch.setattr(viewer,'SimRTDEChannel',rtde)
    owner=viewer.ReadOnlyTelemetry(state,identity);owner.connect()
    assert events[:2]==[('primary','127.0.0.2'),('rtde','127.0.0.2',{'spatial':True,'receive_only':True})]
    assert owner.healthy()
    owner.receiver.failure='source disconnected';assert not owner.healthy()
    owner.connect();assert events[2:4]==['close-rtde','close-primary']
    owner.close()


def test_readonly_reconnect_refuses_container_identity_change_before_opening(monkeypatch):
    monkeypatch.setattr(viewer,'verify_docker',lambda _:DockerIdentity('different','g','n'))
    monkeypatch.setattr(viewer,'SimStateReceiver',lambda _:pytest.fail('changed identity reached telemetry'))
    owner=viewer.ReadOnlyTelemetry(viewer.MeasuredState(None,None),DockerIdentity('c','g','n'))
    with pytest.raises(RuntimeError,match='identity'):owner.connect()


def test_get_only_routes_expose_state_and_local_assets_without_control_endpoints(workspace):
    calls=[]
    measured=SimpleNamespace(snapshot=lambda:calls.append(True) or {'fresh':False,'q':None})
    with viewer.ViewerServer(measured,root=workspace,port=0) as server:
        connection=http.client.HTTPConnection('127.0.0.1',server.port,timeout=2)
        try:
            connection.request('GET','/state');response=connection.getresponse()
            assert response.status==200 and json.loads(response.read())['q'] is None
            for method in ('POST','PUT','DELETE','PATCH','OPTIONS','HEAD'):
                connection.request(method,'/state',body=b'{}');response=connection.getresponse();response.read()
                assert response.status==405 and response.getheader('Allow')=='GET'
            assert calls==[True]
            for path in ('/action','/stop','/../PROJECT_MEMORY.md','/state?command=start'):
                connection.request('GET',path);response=connection.getresponse();response.read();assert response.status==404
        finally:connection.close()


def test_local_asset_whitelist_is_bundled_and_needs_no_teacher_cache(workspace):
    assert not (workspace/'tmp/python-lab-review').exists()
    assets=viewer.local_assets()
    assert set(assets)=={'/assets/three.module.js','/assets/three.core.js','/assets/OrbitControls.js','/assets/THREE-LICENSE.txt'}
    base=Path(viewer.__file__).with_name('viewer_assets').resolve()
    assert all(path.parent==base and path.is_file() for path in assets.values())
    assert 'MIT License' in (base/'LICENSE').read_text(encoding='utf-8')
    assert '0.178.0' in (base/'README.md').read_text(encoding='utf-8')
    page=Path(viewer.__file__).with_name('ursim_viewer.html').read_text(encoding='utf-8')
    assert '/assets/THREE-LICENSE.txt' in page and 'autoplay' not in page.lower()
    assert "fetch('/state'" in page and "method:'POST'" not in page


def test_default_entry_does_not_inspect_docker_or_open_telemetry(monkeypatch,workspace):
    def forbidden(*a,**k):pytest.fail('offline viewer crossed runtime boundary')
    for name in ('verify_docker','SimStateReceiver','SimRTDEChannel','ViewerServer'):
        monkeypatch.setattr(viewer,name,forbidden)
    assert viewer.main([],root=workspace)==0
