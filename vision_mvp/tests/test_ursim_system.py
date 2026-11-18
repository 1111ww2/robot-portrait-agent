"""URSim 集成边界仅使用模拟的环境检查结果与遥测数据。"""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leica_tracker import ursim_system as system
from test_physical_runtime import data
from test_physical_system import workspace
from test_ursim_profile import snapshots


def inspection():
    networks = {system.NETWORK: {'NetworkID':'n'*64,'IPAddress':'172.28.0.2'}}
    config = {'Labels': {'org.leica.purpose': 'ursim-diagnostics'}}
    controller = {'Id': 'c'*64, 'Image':'sha256:'+'i'*64, 'Name': '/'+system.CONTROLLER, 'State': {'Running': True},
        'Config': {**copy.deepcopy(config), 'Image': system.IMAGE, 'Env': ['ROBOT_MODEL=UR10']},
        'NetworkSettings': {'Networks': networks, 'Ports': {}}, 'HostConfig': {'PortBindings': None}}
    ports = {f'{key}/tcp': [{'HostIp': '127.0.0.1', 'HostPort': str(value)}]
             for key, value in system.FORWARD_PORTS.items()}
    gateway = {'Id': 'g'*64, 'Image':controller['Image'], 'Name': '/'+system.GATEWAY, 'State': {'Running': True},
        'Config':{**config,'Image':system.IMAGE,'Entrypoint':['python3','-u','/gateway.py'],'Cmd':None},
        'NetworkSettings': {'Networks': {system.NETWORK:{'NetworkID':'n'*64,'IPAddress':'172.28.0.3'},
            'leica-ursim-diagnostics_ingress':{}}, 'Ports': ports}, 'HostConfig': {'PortBindings': copy.deepcopy(ports)}}
    return [controller, gateway], [{'Id': 'n'*64, 'Name': system.NETWORK, 'Internal': True}]


def fake_inspect(containers, networks, route=None):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {'check': True, 'capture_output': True, 'text': True, 'timeout': 10}
        if command[1]=='exec':
            result=route or {'sha256':'11c5ce2a4f51c083a483ddb0ac0b4774b104f30e07d1a951bd8bbb662e537062',
                            'controller_ip':'172.28.0.2'}
        else: result = networks if command[1:3] == ['network', 'inspect'] else containers
        return SimpleNamespace(stdout=json.dumps(result))
    return run, calls


def test_exact_docker_fence_records_identity_without_mutating_docker():
    run, calls = fake_inspect(*inspection())
    result = system.verify_docker('fixture-docker', run=run)
    assert result == system.DockerIdentity('c'*64, 'g'*64, 'n'*64)
    assert calls[:2] == [['fixture-docker', 'inspect', system.CONTROLLER, system.GATEWAY],
                        ['fixture-docker', 'network', 'inspect', system.NETWORK]]
    assert len(calls)==3 and calls[2][:5]==['fixture-docker','exec','g'*64,'python3','-c']
    assert '/gateway.py' in calls[2][5] and 'gethostbyname' in calls[2][5]


def test_explicit_ur5e_fence_matches_model_without_changing_transport():
    containers, networks = inspection()
    containers[0]['Config']['Env'] = ['ROBOT_MODEL=UR5']
    run, calls = fake_inspect(containers, networks)
    result = system.verify_docker('fixture-docker', run=run, robot_model='ur5e')
    assert result == system.DockerIdentity('c'*64, 'g'*64, 'n'*64, 'ur5e')
    assert all(command[1] in ('inspect', 'network', 'exec') for command in calls)


def test_explicit_ur5e_fence_rejects_ur10e_and_unknown_model():
    run, _ = fake_inspect(*inspection())
    with pytest.raises(ValueError, match='model'):
        system.verify_docker('fixture-docker', run=run, robot_model='ur5e')
    with pytest.raises(ValueError, match='model'):
        system.verify_docker('fixture-docker', run=run, robot_model='ur11')


def test_canonical_artifact_records_direction_cadence_and_resident_program(tmp_path):
    from leica_tracker.motion_configuration import configure_motion_design
    source=configure_motion_design(data())
    profile=system.build_profile(source,*snapshots(),joint_speed=.7)
    source_path=tmp_path/'source.json';source_path.write_text(json.dumps(source))
    target=tmp_path/'canonical.json'
    system.write_artifact(profile,system.DockerIdentity('c','g','n'),target,source_path)
    artifact=json.loads(target.read_text())
    assert artifact['gesture']=={'duration_s':3.3,'repetitions':2,'joint_speed_limit':.7}
    assert artifact['opening_joint_speed']==.7
    resident=target.with_suffix('.resident.urscript').read_text()
    assert 'gesture_age' in resident and 'flag == 2' in resident


@pytest.mark.parametrize('case', ['image','purpose','running','model','network','internal',
                                 'controller_port','gateway_address','gateway_port','missing_port'])
def test_docker_fence_rejects_each_identity_or_exposure_change(case):
    containers, networks = inspection(); controller, gateway = containers
    if case == 'image': controller['Config']['Image'] = 'universalrobots/ursim_e-series:latest'
    elif case == 'purpose': gateway['Config']['Labels'] = {}
    elif case == 'running': controller['State']['Running'] = False
    elif case == 'model': controller['Config']['Env'] = ['ROBOT_MODEL=UR5']
    elif case == 'network': controller['NetworkSettings']['Networks'] = {'bridge': {}}
    elif case == 'internal': networks[0]['Internal'] = False
    elif case == 'controller_port': controller['NetworkSettings']['Ports']['30002/tcp'] = [{'HostIp':'127.0.0.1','HostPort':'30002'}]
    elif case == 'gateway_address': gateway['NetworkSettings']['Ports']['30002/tcp'][0]['HostIp'] = '0.0.0.0'
    elif case == 'gateway_port': gateway['HostConfig']['PortBindings']['30002/tcp'][0]['HostPort'] = '30002'
    elif case == 'missing_port': gateway['NetworkSettings']['Ports'].pop('30011/tcp')
    run, _ = fake_inspect(containers, networks)
    with pytest.raises(ValueError): system.verify_docker('fixture-docker', run=run)


@pytest.mark.parametrize('case',['gateway_image','image_id','entrypoint','cmd','gateway_network',
                                'controller_network_id','gateway_network_id','route_hash','route_dns'])
def test_gateway_route_cannot_be_replaced_by_a_lookalike_forwarder(case):
    containers,networks=inspection();controller,gateway=containers
    route={'sha256':'11c5ce2a4f51c083a483ddb0ac0b4774b104f30e07d1a951bd8bbb662e537062',
           'controller_ip':'172.28.0.2'}
    if case=='gateway_image':gateway['Config']['Image']='python:latest'
    elif case=='image_id':gateway['Image']='sha256:different'
    elif case=='entrypoint':gateway['Config']['Entrypoint']=['python3','-u','/physical-forwarder.py']
    elif case=='cmd':gateway['Config']['Cmd']=['override']
    elif case=='gateway_network':gateway['NetworkSettings']['Networks']['bridge']={}
    elif case=='controller_network_id':controller['NetworkSettings']['Networks'][system.NETWORK]['NetworkID']='other'
    elif case=='gateway_network_id':gateway['NetworkSettings']['Networks'][system.NETWORK]['NetworkID']='other'
    elif case=='route_hash':route['sha256']='0'*64
    elif case=='route_dns':route['controller_ip']='192.168.1.20'
    run,_=fake_inspect(containers,networks,route)
    with pytest.raises(ValueError):system.verify_docker('fixture-docker',run=run)


def local_files(root):
    home = root / 'tmp/ursim-ipad-home'; home.mkdir(parents=True)
    (home / 'ipad-session.json').write_text(json.dumps({'port':8777,'key':'k'*32}))
    (home / 'camera.jsonl').write_text('')
    (root / 'tmp/robot-network/physical-ipad-ur10e-trial.json').write_text(json.dumps(data()))


def test_default_cli_has_no_docker_pid_network_or_source_thread_boundary(workspace, monkeypatch):
    local_files(workspace)
    def forbidden(*args, **kwargs): pytest.fail('offline validation crossed a runtime boundary')
    for name in ('verify_docker','CameraProcess','LoopbackGateway','LocalGuestControlReader','TailEventStream'):
        monkeypatch.setattr(system, name, forbidden)
    assert system.main(['--camera-pid','123'], root=workspace) == 0


def test_execute_rejects_already_active_ui_without_changing_it(workspace, monkeypatch):
    local_files(workspace); closed = []
    monkeypatch.setattr(system, 'verify_docker', lambda _: system.DockerIdentity('c','g','n'))
    monkeypatch.setattr(system, 'CameraProcess', lambda _: SimpleNamespace(close=lambda:closed.append(True)))
    monkeypatch.setattr(system, 'LocalGuestControlReader', lambda *a, **k:lambda:{'active':True})
    monkeypatch.setattr(system, 'LoopbackGateway', lambda:pytest.fail('active UI reached gateway'))
    with pytest.raises(RuntimeError, match='fresh explicit Start'):
        system.main(['--camera-pid','123','--execute'],root=workspace)
    assert closed == [True]


def test_preflight_requires_fresh_advancing_samples_and_closes_receivers():
    clock = [10.]; closed = []
    primary, feedback = snapshots()
    class Receiver:
        failure = None
        def __init__(self, primary): self.primary = primary
        @property
        def latest(self):
            value = copy.deepcopy(primary if self.primary else feedback)
            if self.primary: value['mode']['timestamp'] = clock[0]*1e6
            else: value['timestamp'] = clock[0]
            return value, clock[0]
        def close(self): closed.append(self.primary)
    factories = {'dashboard_check':lambda host,**kw:None,
        'state_factory':lambda host:Receiver(True), 'rtde_factory':lambda host,**kw:Receiver(False)}
    result = system.fresh_profile(data(), factories=factories, clock=lambda:clock[0],
                                  sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    assert result.simulation_only and result.host == system.SIM_HOST
    assert closed == [False,True] and clock[0] >= 10.1


@pytest.mark.parametrize('case', ['stale','frozen','moving'])
def test_preflight_rejects_stale_frozen_or_moving_reference(case):
    clock = [10.]; primary, feedback = snapshots(); closed = []
    class Receiver:
        failure = None
        def __init__(self, primary): self.primary = primary
        @property
        def latest(self):
            value = copy.deepcopy(primary if self.primary else feedback)
            producer = 1. if case == 'frozen' else clock[0]
            if self.primary:value['mode']['timestamp']=producer
            else:
                value['timestamp']=producer
                if case=='moving':value['tcp_speed'][0]=.1
            return value, clock[0]-(1. if case=='stale' else 0.)
        def close(self):closed.append(True)
    factories={'dashboard_check':lambda *a,**k:None,'state_factory':lambda _:Receiver(True),
               'rtde_factory':lambda *a,**k:Receiver(False)}
    with pytest.raises(RuntimeError, match='Fresh advancing stationary'):
        system.fresh_profile(data(),factories=factories,timeout=.2,clock=lambda:clock[0],
                             sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    assert len(closed)==2


def test_artifact_is_numeric_simulation_only_and_contains_no_physical_review(workspace):
    source=workspace/'design.json';source.write_text(json.dumps(data()))
    profile=system.build_profile(data(),*snapshots()); artifact=workspace/'simulation.json'
    system.write_artifact(profile,system.DockerIdentity('c','g','n'),artifact,source)
    result=json.loads(artifact.read_text())
    assert result['status']=='SIMULATION_ONLY' and result['physical_acceptance'] is False
    assert result['host']==system.SIM_HOST and result['active_tcp']==[0.]*6
    assert 'reviewed' not in artifact.read_text() and 'PHYSICAL_INTERACTION_REVIEWED' not in artifact.read_text()
    assert 'SIMULATION_ONLY' in artifact.with_suffix('.urscript').read_text()


def test_simulation_serve_revalidates_identity_and_injects_native_stops(workspace,monkeypatch):
    identity=system.DockerIdentity('c','g','n'); profile=system.build_profile(data(),*snapshots())
    calls=[]; stopping=[False]
    monkeypatch.setattr(system,'verify_docker',lambda _:calls.append('verify') or identity)
    monkeypatch.setattr(system,'TailEventStream',lambda *a,**k:SimpleNamespace(close=lambda:calls.append('tail-close')))
    monkeypatch.setattr(system.native,'LatestEventSource',lambda tail:SimpleNamespace(thread=SimpleNamespace(
        join=lambda **kw:calls.append('join'),is_alive=lambda:False)))
    def run(*args,**kwargs):
        assert kwargs['should_stop']() is False
        stopping[0]=True
        assert kwargs['should_stop']() is True
        assert kwargs['dependencies']['xyz_opener'].func is system.open_resident
        calls.append('session')
    monkeypatch.setattr(system.native,'run_session',run)
    def serve(*args,**kwargs):
        kwargs['session_runner'](profile,8777,'secret',None,None,before={})
        return 0
    monkeypatch.setattr(system.native,'serve_sessions',serve)
    assert system.serve(profile,{'port':8777,'key':'secret'},workspace/'camera.jsonl',
        SimpleNamespace(is_alive=lambda:True),identity,factories={'dashboard_factory':object},
        should_stop=lambda:stopping[0])==0
    assert calls==['verify','session','tail-close','join']


def test_simulation_serve_refuses_a_replaced_container_before_native_session(workspace,monkeypatch):
    profile=system.build_profile(data(),*snapshots())
    monkeypatch.setattr(system,'verify_docker',lambda _:system.DockerIdentity('changed','g','n'))
    monkeypatch.setattr(system,'TailEventStream',lambda *a,**k:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(system.native,'LatestEventSource',lambda tail:SimpleNamespace(thread=SimpleNamespace(join=lambda **kw:None,is_alive=lambda:False)))
    monkeypatch.setattr(system.native,'run_session',lambda *a,**k:pytest.fail('changed identity reached native session'))
    monkeypatch.setattr(system.native,'serve_sessions',lambda *a,**kw:kw['session_runner'](profile,8777,'secret',None,None,before={}))
    with pytest.raises(RuntimeError,match='identity changed'):
        system.serve(profile,{'port':8777,'key':'secret'},workspace/'camera.jsonl',SimpleNamespace(is_alive=lambda:True),
                     system.DockerIdentity('c','g','n'),factories={'dashboard_factory':object})


def test_source_thread_start_failure_closes_its_owned_tail(workspace,monkeypatch):
    profile=system.build_profile(data(),*snapshots()); closed=[]
    monkeypatch.setattr(system,'TailEventStream',lambda *a,**k:SimpleNamespace(close=lambda:closed.append(True)))
    def fail_start(tail):raise RuntimeError('thread start failed')
    monkeypatch.setattr(system.native,'LatestEventSource',fail_start)
    with pytest.raises(RuntimeError,match='thread start failed'):
        system.serve(profile,{'port':8777,'key':'secret'},workspace/'camera.jsonl',
                     SimpleNamespace(is_alive=lambda:True),system.DockerIdentity('c','g','n'),
                     factories={'dashboard_factory':object})
    assert closed==[True]


def test_preflight_only_cli_never_attaches_event_source_or_session(workspace,monkeypatch):
    local_files(workspace); identity=system.DockerIdentity('c','g','n'); closed=[]
    monkeypatch.setattr(system,'verify_docker',lambda _:identity)
    monkeypatch.setattr(system,'CameraProcess',lambda _:SimpleNamespace(is_alive=lambda:True,close=lambda:closed.append(True)))
    monkeypatch.setattr(system,'LocalGuestControlReader',lambda *a,**k:lambda:{'active':False})
    monkeypatch.setattr(system,'signal_stop',lambda:nullcontext(lambda:False))
    monkeypatch.setattr(system,'LoopbackGateway',lambda:nullcontext(SimpleNamespace(failure=None)))
    monkeypatch.setattr(system,'dependencies',lambda:{})
    monkeypatch.setattr(system,'fresh_profile',lambda *a,**k:system.build_profile(data(),*snapshots()))
    monkeypatch.setattr(system,'serve',lambda *a,**k:pytest.fail('preflight attached backend'))
    assert system.main(['--execute','--preflight-only','--camera-pid','123','--output','result.json'],root=workspace)==0
    assert json.loads((workspace/'result.json').read_text())['status']=='SIMULATION_ONLY'
    assert closed==[True]
