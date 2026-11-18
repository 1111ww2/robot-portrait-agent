"""仅把已有 iPad 相机连接到明确限定的官方 URSim。
默认只读取本地文件，--execute 允许检查指定 Docker 环境并使用回环地址进行仿真通信；
仍需在 iPad 上明确点击开始。"""
import argparse
import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from . import physical_runtime as native
from .horizontal_follow import check_dashboard
from .physical_intro_bootstrap import open_intro
from .physical_intro_program import IntroProgram
from .physical_resident_bootstrap import open_resident
from .physical_system import ROOT, signal_stop, validated_pair, workspace_path
from .resident_guest_watchdog import LocalGuestControlReader
from .ursim_profile import SimulationProfile, build_profile
from .ursim_source import TailEventStream
from .ursim_transport import SIM_HOST, FORWARD_PORTS, LoopbackGateway, SimRTDEChannel, dependencies

CONTROLLER = 'leica-ursim-diagnostics-controller-1'
GATEWAY = 'leica-ursim-diagnostics-gateway-1'
NETWORK = 'leica-ursim-diagnostics_isolated'
INGRESS = 'leica-ursim-diagnostics_ingress'
IMAGE = 'universalrobots/ursim_e-series:5.13'
GATEWAY_SHA256 = '11c5ce2a4f51c083a483ddb0ac0b4774b104f30e07d1a951bd8bbb662e537062'


@dataclass(frozen=True)
class DockerIdentity:
    controller_id: str
    gateway_id: str
    network_id: str
    robot_model: str = 'ur10e'


def _docker_path():
    local = os.environ.get('LOCALAPPDATA')
    candidate = Path(local) / 'Programs/DockerDesktop/resources/bin/docker.exe' if local else None
    found = str(candidate) if candidate is not None and candidate.is_file() else shutil.which('docker')
    if not found: raise ValueError('Docker Desktop executable is unavailable')
    return found


def verify_docker(docker=None, run=subprocess.run, *, robot_model='ur10e'):
    """读取确切的容器/网络身份；永远不要启动或更改Docker。"""
    models_by_name={'ur5e':'UR5','ur10e':'UR10'}
    if robot_model not in models_by_name:raise ValueError('Unsupported isolated simulator model')
    docker = str(docker or _docker_path())
    def query(*args):
        result = run([docker, *args], check=True, capture_output=True, text=True, timeout=10)
        return json.loads(result.stdout)
    def inspect(*args):
        value = query(*args)
        if not isinstance(value, list): raise ValueError('Invalid Docker inspection')
        return value
    containers = inspect('inspect', CONTROLLER, GATEWAY)
    networks = inspect('network', 'inspect', NETWORK)
    if len(containers) != 2 or len(networks) != 1: raise ValueError('URSim Docker identities are missing')
    controller, gateway = containers
    def validate_container(value, name):
        if (not isinstance(value, dict) or value.get('Name') != '/' + name
                or not isinstance(value.get('Id'), str) or not value['Id']
                or value.get('State', {}).get('Running') is not True
                or value.get('Config', {}).get('Labels', {}).get('org.leica.purpose') != 'ursim-diagnostics'):
            raise ValueError('URSim container identity, purpose or running state differs')
    validate_container(controller, CONTROLLER); validate_container(gateway, GATEWAY)
    if controller['Config'].get('Image') != IMAGE:
        raise ValueError('The fixed official URSim 5.13 image is required')
    gateway_config = gateway['Config']
    if (gateway_config.get('Image') != IMAGE or not isinstance(controller.get('Image'), str)
            or not controller['Image'].startswith('sha256:') or gateway.get('Image') != controller['Image']
            or gateway_config.get('Entrypoint') != ['python3', '-u', '/gateway.py']
            or gateway_config.get('Cmd') is not None):
        raise ValueError('URSim gateway must use the verified controller image and fixed route entrypoint')
    models = [item for item in controller['Config'].get('Env', []) if item.startswith('ROBOT_MODEL=')]
    if models != ['ROBOT_MODEL='+models_by_name[robot_model]]:
        raise ValueError('The isolated simulator model does not match the explicitly selected model')
    controller_networks = controller.get('NetworkSettings', {}).get('Networks', {})
    if set(controller_networks) != {NETWORK}:
        raise ValueError('URSim controller must have only its isolated network')
    gateway_networks = gateway.get('NetworkSettings', {}).get('Networks', {})
    if set(gateway_networks) != {NETWORK, INGRESS}:
        raise ValueError('URSim gateway must have only the fixed isolated and ingress networks')
    network = networks[0]
    if (network.get('Name') != NETWORK or network.get('Internal') is not True
            or not isinstance(network.get('Id'), str) or not network['Id']):
        raise ValueError('URSim network must remain internal')
    if (controller_networks[NETWORK].get('NetworkID') != network['Id']
            or gateway_networks[NETWORK].get('NetworkID') != network['Id']):
        raise ValueError('URSim containers do not share the inspected isolated network identity')
    for section in ('NetworkSettings', 'HostConfig'):
        field = 'Ports' if section == 'NetworkSettings' else 'PortBindings'
        if any((controller.get(section, {}).get(field, {}) or {}).values()):
            raise ValueError('URSim controller ports must not be published')
    for section, field in (('NetworkSettings', 'Ports'), ('HostConfig', 'PortBindings')):
        ports = gateway.get(section, {}).get(field, {}) or {}
        for native_port, published in FORWARD_PORTS.items():
            if ports.get(f'{native_port}/tcp') != [{'HostIp': '127.0.0.1', 'HostPort': str(published)}]:
                raise ValueError('URSim native gateway ports must use the exact loopback mapping')
        for bindings in ports.values():
            if bindings and any(binding.get('HostIp') != '127.0.0.1' for binding in bindings):
                raise ValueError('URSim gateway published ports must remain loopback only')
    # 通过其不可变的容器ID检查已运行的网关。这只读取其固定的路由并解析其内部控制器名称。
    route = query('exec', gateway['Id'], 'python3', '-c',
        "import hashlib,json,socket;from pathlib import Path;"
        "print(json.dumps({'sha256':hashlib.sha256(Path('/gateway.py').read_bytes()).hexdigest(),"
        "'controller_ip':socket.gethostbyname('controller')}))")
    controller_ip = controller_networks[NETWORK].get('IPAddress')
    if (not isinstance(route, dict) or route.get('sha256') != GATEWAY_SHA256
            or not isinstance(controller_ip, str) or not controller_ip
            or route.get('controller_ip') != controller_ip):
        raise ValueError('URSim gateway route or controller DNS differs from the verified isolated controller')
    return DockerIdentity(controller['Id'], gateway['Id'], network['Id'],robot_model)


class CameraProcess:
    """持有Windows进程句柄，以便PID重复使用不能改变其身份。"""
    def __init__(self, pid):
        if type(pid) is not int or pid <= 0: raise ValueError('A captured camera PID is required')
        self.pid = pid; self._handle = None
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            self._ctypes = ctypes
            self._kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self._kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            self._kernel.OpenProcess.restype = wintypes.HANDLE
            self._kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            self._kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            self._handle = self._kernel.OpenProcess(0x1000, False, pid)
            if not self._handle: raise ValueError('Captured camera process is unavailable')
        if not self.is_alive():
            self.close(); raise ValueError('Captured camera process has exited')
    def is_alive(self):
        if os.name == 'nt':
            if not self._handle: return False
            from ctypes import wintypes
            result = wintypes.DWORD()
            return bool(self._kernel.GetExitCodeProcess(self._handle, self._ctypes.byref(result)) and result.value == 259)
        try: os.kill(self.pid, 0); return True
        except (OSError, ProcessLookupError): return False
    def close(self):
        if self._handle:
            self._kernel.CloseHandle(self._handle); self._handle = None


def fresh_profile(source, *, factories=None, timeout=8., clock=time.monotonic, sleep=time.sleep,
                  should_stop=lambda: False,robot_model='ur10e',joint_speed=.6):
    """需要多个前进、新鲜、静止的仅接收快照。"""
    d = dependencies() if factories is None else factories
    states = receiver = None; deadline = clock() + timeout
    previous = None; count = 0; since = None; error = None
    try:
        d['dashboard_check'](SIM_HOST, idle=True, robot_model=robot_model)
        states = d['state_factory'](SIM_HOST)
        receiver = d['rtde_factory'](SIM_HOST, spatial=True, receive_only=True)
        while clock() < deadline:
            if should_stop(): raise RuntimeError('URSim preflight cancelled')
            primary, ps = states.latest; feedback, rs = receiver.latest; now = clock()
            if states.failure or receiver.failure: raise RuntimeError('URSim preflight telemetry failed')
            if (isinstance(primary, dict) and isinstance(feedback, dict)
                    and all(type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= now-stamp <= .15 for stamp in (ps, rs))):
                producer = (primary.get('mode', {}).get('timestamp'), feedback.get('timestamp'))
                if all(type(v) in (int, float) and math.isfinite(v) for v in producer):
                    if previous is None or all(a > b for a, b in zip((ps, rs, *producer), previous)):
                        try:
                            profile = build_profile(source, copy.deepcopy(primary), copy.deepcopy(feedback),
                                                    robot_model=robot_model,joint_speed=joint_speed)
                            count += 1; since = now if since is None else since
                            if count >= 3 and now-since >= .1: return profile
                        except ValueError as exc:
                            error = exc; count = 0; since = None
                        previous = (ps, rs, *producer)
            sleep(.01)
        raise RuntimeError('Fresh advancing stationary URSim reference unavailable' + (': ' + str(error) if error else ''))
    finally:
        errors = []
        for item in (receiver, states):
            if item is not None:
                try: item.close()
                except Exception as exc: errors.append(exc)
        if errors: raise RuntimeError('URSim preflight receiver cleanup incomplete') from errors[0]


def write_artifact(profile, identity, path, source_path):
    if type(profile) is not SimulationProfile or profile.host != SIM_HOST:
        raise ValueError('An isolated simulation-only profile is required')
    value = {'status': 'SIMULATION_ONLY', 'physical_acceptance': False,
             'controller_id': identity.controller_id, 'gateway_id': identity.gateway_id,
             'network_id': identity.network_id, 'robot_model': identity.robot_model, 'host': SIM_HOST,
             'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
             'reference_pose': list(profile.cfg.neutral), 'reference_joints': list(profile.cfg.joints),
             'active_tcp': list(profile.cfg.tcp), 'opening_waypoints': profile.intro['waypoints'],
             'opening_joint_speed': profile.intro['joint_speed'],
             'opening_joint_acceleration': profile.intro['joint_acceleration'], 'paths': profile.paths}
    value['gesture']=profile.cfg.gesture.as_dict() if profile.cfg.gesture else None
    value['guidance_linear_tracking_ceiling_m_s']=profile.cfg.speed
    value['guidance_acceleration_m_s2']=profile.cfg.acceleration
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    script = path.with_suffix('.urscript')
    with script.open('x', encoding='utf-8') as stream:
        stream.write('# SIMULATION_ONLY inspection session 1; not an executable physical profile.\n')
        stream.write(IntroProgram(profile.intro, 1).program())
    with path.with_suffix('.resident.urscript').open('x',encoding='utf-8') as stream:
        stream.write('# SIMULATION_ONLY inspection; same native resident generator.\n'+profile.cfg.program(1))


def serve(profile, pair, log_path, camera, identity, *, docker=None, factories=None,
          should_stop=lambda: False, gateway=None):
    if type(profile) is not SimulationProfile or profile.host != SIM_HOST:
        raise ValueError('The dedicated URSim runtime accepts only SimulationProfile at 127.0.0.2')
    d = dependencies() if factories is None else factories
    def alive():
        return camera.is_alive() and not should_stop() and not (gateway is not None and gateway.failure)
    def same_docker():
        current=(verify_docker(docker) if identity.robot_model=='ur10e' else
                 verify_docker(docker,robot_model=identity.robot_model))
        if current != identity: raise RuntimeError('URSim Docker identity changed; no new session')
    if profile.operator_design is not None:
        required = ('dashboard_check','state_factory','rtde_factory')
        if any(not callable(d.get(name)) for name in required):
            raise ValueError('Selected URSim capture requires all fenced simulator factories')
        capture_dependencies = {name:d[name] for name in required}
        for name in ('clock','sleep'):
            if name in d: capture_dependencies[name]=d[name]
        def capture_selected(target, *, reader, control, should_stop, sample_observer=None,
                             proof_observer=None):
            from .operator_design_capture import capture_selected_reference
            same_docker()
            if (target.host != SIM_HOST or target.cfg.robot_model != identity.robot_model):
                raise ValueError('Selected URSim capture target identity mismatch')
            if not alive() or should_stop():
                raise RuntimeError('Selected URSim reference capture cancelled')
            def verify_before_final_snapshot():
                if not alive() or should_stop():
                    raise RuntimeError('Selected URSim reference capture cancelled')
                same_docker()
                if not alive() or should_stop():
                    raise RuntimeError('Selected URSim reference capture cancelled')
            options = {} if sample_observer is None else {'sample_observer': sample_observer}
            if proof_observer is not None:options['proof_observer'] = proof_observer
            result = capture_selected_reference(target,reader=reader,control=control,
                should_stop=lambda: not alive() or should_stop(),
                dependencies={**capture_dependencies,'before_final_snapshot':verify_before_final_snapshot},
                **options)
            if not alive() or should_stop():
                raise RuntimeError('Selected URSim reference capture cancelled')
            return result
        profile = replace(profile,capture_provider=capture_selected)
    runtime_dependencies = {
        'intro_runner': partial(native.run_intro_then_open_xyz,
            intro_opener=partial(open_intro, dependencies=d), xyz_opener=partial(open_resident, dependencies=d),
            dashboard_factory=d['dashboard_factory']),
        'xyz_opener': partial(open_resident, dependencies=d), 'dashboard_factory': d['dashboard_factory']}
    def runner(*args, **kwargs):
        same_docker()
        return native.run_session(*args, **kwargs, should_stop=lambda: not alive(), dependencies=runtime_dependencies)
    runner.supports_reference_restart = True
    def stopped(host):
        if host != SIM_HOST: raise ValueError('URSim stop observation requires isolated loopback')
        same_docker()
        return native.verify_stationary_after_close(host, channel_factory=SimRTDEChannel, dashboard_check=check_dashboard)
    tail = TailEventStream(log_path, is_alive=camera.is_alive)
    source = None
    try:
        source = native.LatestEventSource(tail)
        print('URSIM_BACKEND_READY: waiting for a fresh explicit iPad Start.', flush=True)
        return native.serve_sessions(profile, pair['port'], pair['key'], source,
            session_runner=runner, stopped_check=stopped, is_alive=alive)
    finally:
        tail.close()
        if source is not None:
            source.thread.join(timeout=1.)
            if source.thread.is_alive(): raise RuntimeError('URSim camera source cleanup incomplete')


def main(argv=None, *, root=ROOT):
    root = Path(root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--camera-pid', type=int, required=True)
    parser.add_argument('--source-profile', type=Path, default=Path('tmp/robot-network/physical-ipad-ur10e-trial.json'))
    parser.add_argument('--guest-session', type=Path, default=Path('tmp/ursim-ipad-home/ipad-session.json'))
    parser.add_argument('--camera-log', type=Path, default=Path('tmp/ursim-ipad-home/camera.jsonl'))
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--docker')
    parser.add_argument('--robot-model',choices=('ur5e','ur10e'),default='ur10e')
    parser.add_argument('--joint-speed',type=float,default=.7,help='Global native joint limit, rad/s; not a Cartesian speed')
    args = parser.parse_args(argv)
    if args.camera_pid <= 0: parser.error('A captured camera PID is required')
    if args.preflight_only and not args.execute: parser.error('--preflight-only requires explicit --execute for simulator reads')
    source_path = workspace_path(root, args.source_profile)
    pair_path = workspace_path(root, args.guest_session)
    log_path = workspace_path(root, args.camera_log)
    stop_file = workspace_path(root, args.stop_file) if args.stop_file else None
    if stop_file is not None and stop_file.exists(): parser.error('Stop file already exists')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    artifact = workspace_path(root, args.output or Path('tmp/ursim-ipad-home') / f'simulation-{stamp}.json')
    if any(path.exists() for path in (artifact,artifact.with_suffix('.urscript'),artifact.with_suffix('.resident.urscript'))):
        parser.error('Preserve existing simulation artifacts; choose a fresh output')
    source = json.loads(source_path.read_text(encoding='utf-8-sig'))
    if not isinstance(source, dict): parser.error('Local motion design must be an object')
    if not math.isfinite(args.joint_speed) or not 0<args.joint_speed<=.7:
        parser.error('Joint speed must be positive and at most 0.7 rad/s')
    from .motion_configuration import configure_motion_design
    source=configure_motion_design(source,joint_speed=args.joint_speed)
    pair = validated_pair(pair_path)
    if pair['port'] != 8777: parser.error('The isolated home iPad guest must use port 8777')
    if not log_path.is_file(): parser.error('Existing owned camera JSONL is required')
    if not args.execute:
        print('URSim local files validated offline; no Docker, simulator, camera or backend started.')
        return 0
    identity=(verify_docker(args.docker) if args.robot_model=='ur10e' else
              verify_docker(args.docker,robot_model=args.robot_model))
    camera = CameraProcess(args.camera_pid)
    try:
        reader = LocalGuestControlReader(pair['port'], pair['key'], timeout=.2)
        try:current = reader()
        finally:
            close_reader=getattr(reader,'close_current_thread',None)
            if close_reader is not None:close_reader()
        if current.get('active') is True:
            raise RuntimeError('Existing iPad encounter is active; use iPad Stop, then a fresh explicit Start after backend readiness')
        with signal_stop() as cancelled:
            def stop(): return cancelled() or not camera.is_alive() or bool(stop_file and stop_file.exists())
            with LoopbackGateway() as gateway:
                d = dependencies()
                profile = fresh_profile(source, factories=d, should_stop=lambda: stop() or bool(gateway.failure),
                                        robot_model=args.robot_model,joint_speed=args.joint_speed)
                after=(verify_docker(args.docker) if args.robot_model=='ur10e' else
                       verify_docker(args.docker,robot_model=args.robot_model))
                if after != identity: raise RuntimeError('URSim identity changed during preflight')
                write_artifact(profile, identity, artifact, source_path)
                print(f'URSim simulation-only artifact: {artifact.relative_to(root)}', flush=True)
                if args.preflight_only: return 0
                return serve(profile, pair, log_path, camera, identity, docker=args.docker,
                             factories=d, should_stop=stop, gateway=gateway)
    finally: camera.close()
