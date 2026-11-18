"""为选定/审核的本地会话和iPad HTTPS设置一个前台所有者。默认的CLI路径仅执行文件验证。
执行保留给下一次明确授权的现场使用；它从不自己启动会话。"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time

from .physical_robot_model import SUPPORTED_MODELS, model_for_profile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = Path('tmp/robot-network/physical-ipad-ur10e-trial.json')
LIVE = Path('tmp/guidance-live')


def workspace_path(root, path):
    path = Path(path)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('System files must remain inside the project workspace')
    return resolved


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validated_pair(path):
    value = read_json(path)
    if (not isinstance(value, dict) or type(value.get('port')) is not int
            or not 1 <= value['port'] <= 65535 or not isinstance(value.get('key'), str)
            or re.fullmatch(r'[A-Za-z0-9_-]{24,128}', value['key']) is None):
        raise ValueError('Existing local iPad pairing is invalid')
    return {'port': value['port'], 'key': value['key']}


@dataclass(frozen=True)
class SystemPlan:
    root: Path
    profile_path: Path
    guest_session: Path
    address: str
    https_port: int
    setup_port: int
    pairing: dict = field(repr=False)
    robot_model: str = 'ur10e'


def prepare_plan(*, root=ROOT, profile=None, guest_session=None, address=None,
                 https_port=None, setup_port=None, profile_validator=None, robot_model='ur10e'):
    """验证本地输入；切勿探测端口、写入文件或加载TLS。"""
    root = Path(root).resolve()
    profile_path = workspace_path(root, profile or DEFAULT_PROFILE)
    if robot_model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported expected robot model')
    profile_data = read_json(profile_path)
    actual_model = model_for_profile(profile_data)
    if actual_model != robot_model:
        raise ValueError(f'Profile robot model {actual_model} does not match expected model {robot_model}')
    if profile_validator is None:
        from .physical_runtime import reviewed_profile
        profile_validator = reviewed_profile
    profile_validator(profile_data)
    live_pair = workspace_path(root, LIVE / 'guest-link.json')
    cached_pair = workspace_path(root, LIVE / 'https/pair.json')
    pair_path = workspace_path(root, guest_session) if guest_session is not None else (
        live_pair if live_pair.is_file() else cached_pair)
    pairing = validated_pair(pair_path)
    if guest_session is None and live_pair.is_file() and cached_pair.is_file():
        if pairing != validated_pair(cached_pair):
            raise ValueError('Live and cached iPad pairing records differ; inspect them locally')
    status_path = workspace_path(root, LIVE / 'https-public-status.json')
    status = read_json(status_path) if status_path.is_file() else {}
    if not isinstance(status, dict):
        raise ValueError('Cached HTTPS address record is invalid')
    address = address or status.get('address')
    try:
        ip = ipaddress.ip_address(address)
    except (TypeError, ValueError):
        raise ValueError('A private LAN IPv4 is required; use --address onsite') from None
    if (ip.version != 4 or not ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_unspecified or ip.is_reserved or ip.is_multicast):
        raise ValueError('A private LAN IPv4 is required; use --address onsite')
    https_port = https_port if https_port is not None else status.get('port', 8768)
    setup_port = setup_port if setup_port is not None else status.get('setup_port', 8769)
    ports = (pairing['port'], https_port, setup_port)
    if any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
        raise ValueError('System ports must be integers from 1 to 65535')
    if len(set(ports)) != 3:
        raise ValueError('Guest, HTTPS and certificate ports must be distinct')
    runner = workspace_path(root, 'vision_mvp/scripts/run-physical-live.py')
    if not runner.is_file():
        raise ValueError('Existing physical runtime script is missing from the workspace')
    # 也解决输出根，这样重定向的本地文件夹就不会逃避范围。
    workspace_path(root, LIVE / 'https')
    workspace_path(root, 'tmp/physical-system')
    return SystemPlan(root, profile_path, pair_path, str(ip), https_port, setup_port, pairing, actual_model)


def probe_guest(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.2):
            return True
    except OSError:
        return False


def read_guest(plan):
    connection = http.client.HTTPConnection('127.0.0.1', plan.pairing['port'], timeout=.3)
    try:
        connection.request('GET', '/state', headers={
            'X-Guest-Token': plan.pairing['key'], 'X-Guest-Client': 'physical-system',
            'Connection': 'close'})
        response = connection.getresponse()
        data = response.read(256 * 1024 + 1)
        if response.status != 200 or len(data) > 256 * 1024:
            raise RuntimeError('Owned iPad service is not ready')
        state = json.loads(data)
        if not isinstance(state, dict) or not isinstance(state.get('control'), dict):
            raise RuntimeError('Owned iPad service returned invalid state')
        return state
    finally:
        connection.close()


def request_stop(plan, expected_instance, system_run_id):
    """经过认证的本地停止；HTTP接受不是机器人停止证明。"""
    if not expected_instance or not system_run_id:
        return False
    port = plan.pairing['port']
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=.5)
    try:
        body = json.dumps({'action': 'stop', 'control': {'instance': expected_instance},
                           'system_run_id': system_run_id}).encode()
        connection.request('POST', '/action', body=body, headers={
            'X-Guest-Token': plan.pairing['key'], 'X-Guest-Client': 'physical-system',
            'Origin': f'http://127.0.0.1:{port}', 'Content-Type': 'application/json',
            'Connection': 'close'})
        response = connection.getresponse()
        body = response.read(4096)
        return response.status == 200 and json.loads(body).get('active') is False
    finally:
        connection.close()


def prepare_pairing_files(plan, files):
    """刷新此地址和此确切配对的本地设置链接。"""
    import qrcode
    from .guest_https import _private_write
    output = workspace_path(plan.root, LIVE)
    pair_cache = workspace_path(plan.root, LIVE / 'https/pair.json')
    _private_write(pair_cache, json.dumps(plan.pairing).encode())
    setup_url = f'http://{plan.address}:{plan.setup_port}/leica-root.mobileconfig'
    app_url = f"https://{plan.address}:{plan.https_port}/?key={plan.pairing['key']}"
    qrcode.make(setup_url).save(output / 'https-certificate-qr.png')
    qrcode.make(app_url).save(output / 'https-ipad-qr.png')
    fingerprint = hashlib.sha256(files['der'].read_bytes()).hexdigest().upper()
    page = ('<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer">'
            '<title>iPad physical interaction</title><style>body{font:20px system-ui;max-width:850px;'
            'margin:40px auto;padding:20px;line-height:1.5}img{width:280px}code{overflow-wrap:anywhere}</style>'
            '<h1>iPad physical interaction</h1><p>Keep the foreground system terminal open. '
            'Refresh Safari, prepare the camera, then use iPad Start. Stop remains available on iPad.</p>'
            '<h2>Open the paired page</h2><img src="https-ipad-qr.png"><p><a href="'
            + html.escape(app_url, quote=True) + '">Open paired iPad page</a></p>'
            '<h2>Certificate setup if this iPad has not trusted this computer</h2>'
            '<img src="https-certificate-qr.png"><p><a href="' + html.escape(setup_url, quote=True)
            + '">Install local certificate profile</a></p><p>After installation, enable Leica Local '
            'iPad Root in Settings → General → About → Certificate Trust Settings.</p>'
            '<p>Root SHA-256: <code>' + fingerprint + '</code></p>')
    _private_write(output / 'https-ipad-pair.html', page.encode('utf-8'))
    _private_write(output / 'https-public-status.json', json.dumps({
        'address': plan.address, 'port': plan.https_port, 'setup_port': plan.setup_port,
        'root_sha256': fingerprint}).encode())


@dataclass
class SystemDependencies:
    popen: object = subprocess.Popen
    probe_guest: object = probe_guest
    read_guest: object = read_guest
    request_stop: object = request_stop
    prepare_pairing: object = prepare_pairing_files
    clock: object = time.monotonic
    sleep: object = time.sleep
    certificates: object = None
    https: object = None

    def __post_init__(self):
        if self.certificates is None or self.https is None:
            from .guest_https import create_certificates, GuestHTTPS
            self.certificates = self.certificates or create_certificates
            self.https = self.https or GuestHTTPS


def source_fingerprint(plan):
    paths = sorted((plan.root / 'vision_mvp/src/leica_tracker').rglob('*.py'))
    paths += sorted((plan.root / 'vision_mvp/src/leica_tracker').rglob('*.html'))
    paths += sorted((plan.root / 'vision_mvp/scripts').glob('*physical*'))
    paths += [plan.root / f'vision_mvp/scripts/run-{plan.robot_model}-ipad-system.py',
              plan.root / f'vision_mvp/scripts/start-{plan.robot_model}-ipad-system.ps1', plan.profile_path]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(plan.root)).replace('\\', '/').encode())
            digest.update(b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def child_command(plan, stop_file):
    return [sys.executable, '-u', str(plan.root / 'vision_mvp/scripts/run-physical-live.py'),
            '--profile', str(plan.profile_path), '--guest-session', str(plan.guest_session),
            '--robot-model', plan.robot_model,
            '--execute', '--require-free-port', '--stop-file', str(stop_file)]


def wait_for_guest(child, plan, deps, should_stop, timeout, system_run_id):
    deadline = deps.clock() + timeout
    while not should_stop():
        if child.poll() is not None:
            raise RuntimeError('Owned physical service exited during startup')
        try:
            snapshot = deps.read_guest(plan)
            instance = snapshot['control']['instance']
            if isinstance(instance, str) and instance and snapshot.get('system_run_id') == system_run_id:
                return instance
        except (OSError, http.client.HTTPException, ValueError, KeyError, RuntimeError):
            pass
        if deps.clock() >= deadline:
            raise RuntimeError('Owned physical service did not become ready; no retry')
        deps.sleep(.1)
    return None


def run_system(plan, *, deps=None, should_stop=lambda: False, startup_timeout=20., cleanup_timeout=15.):
    """仅执行一次。回收子进程时保持 HTTPS 运行，只结束本任务负责的子进程。"""
    deps = deps or SystemDependencies()
    if deps.probe_guest(plan.pairing['port']):
        raise RuntimeError('Guest port is occupied; explicitly stop the old onsite service first')
    logs_root = workspace_path(plan.root, 'tmp/physical-system')
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-'),
                                    dir=logs_root))
    stop_file = run_dir / 'stop.request'
    metadata_path = run_dir / 'run.json'
    metadata = {'profile': str(plan.profile_path.relative_to(plan.root)), 'robot_model': plan.robot_model,
                'address': plan.address,
                'https_port': plan.https_port, 'setup_port': plan.setup_port,
                'guest_port': plan.pairing['port'], 'source_sha256': source_fingerprint(plan),
                'system_run_id': secrets.token_hex(16),
                'child_retired': False, 'local_stop_accepted': False,
                'robot_stop': 'not_verified_by_parent', 'status': 'preparing'}
    def record():
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    record()
    child = None
    instance = None
    outcome = 2
    try:
        files = deps.certificates(workspace_path(plan.root, LIVE / 'https'), plan.address)
        try:
            proxy = deps.https(files, upstream_port=plan.pairing['port'], host=plan.address, port=plan.https_port,
                               setup_port=plan.setup_port, addresses=[plan.address])
        except OSError as exc:
            raise RuntimeError('HTTPS bind failed or port occupied; old services are never reused or stopped') from exc
        with proxy:
            deps.prepare_pairing(plan, files)
            with (run_dir / 'stdout.log').open('w', encoding='utf-8') as stdout, (
                    run_dir / 'stderr.log').open('w', encoding='utf-8') as stderr:
                child_env = dict(os.environ, LEICA_SYSTEM_RUN_ID=metadata['system_run_id'])
                kwargs = {'cwd': plan.root / 'vision_mvp', 'stdout': stdout, 'stderr': stderr,
                          'env': child_env}
                if os.name == 'nt':
                    kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                else:
                    kwargs['start_new_session'] = True
                child = deps.popen(child_command(plan, stop_file), **kwargs)
                try:
                    metadata['child_pid'] = child.pid
                    metadata['status'] = 'starting'
                    record()
                    instance = wait_for_guest(child, plan, deps, should_stop, startup_timeout,
                                              metadata['system_run_id'])
                    if instance is not None:
                        metadata['status'] = 'waiting_for_ipad_start'
                        record()
                        print('SYSTEM_READY: prepare the camera and use iPad Start; no encounter started by launcher.', flush=True)
                        print('Local pairing page: tmp/guidance-live/https-ipad-pair.html (token not logged).', flush=True)
                    while child.poll() is None and not should_stop():
                        deps.sleep(.2)
                    if child.poll() is None:
                        outcome = 0
                    else:
                        print('Owned physical service exited; inspect this run logs. No retry.', file=sys.stderr)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr, flush=True)
                    outcome = 2
                finally:
                    # 保持代理和日志手柄打开，直到有限合作退休。
                    try:
                        metadata['local_stop_accepted'] = deps.request_stop(plan, instance, metadata['system_run_id']) is True
                    except Exception:
                        metadata['local_stop_accepted'] = False
                    try:
                        stop_file.write_text('Stop requested by this foreground system owner.\n', encoding='utf-8')
                    except OSError:
                        outcome = 2
                        print('Cooperative stop file could not be written; waiting for owned child without killing it.', file=sys.stderr)
                    try:
                        child_result = child.wait(timeout=cleanup_timeout)
                        metadata['child_retired'] = True
                        metadata['child_exit_code'] = child_result
                        if child_result != 0:
                            outcome = 2
                            print('Owned child reported a nonzero exit; inspect this run logs.', file=sys.stderr)
                    except (subprocess.TimeoutExpired, OSError):
                        outcome = 2
                        print('UNKNOWN STOP STATE: owned child did not retire; no process tree was killed. '
                              'Use the onsite stop procedure and inspect this run logs.', file=sys.stderr, flush=True)
                    if not metadata['local_stop_accepted']:
                        print('Local Stop acceptance unavailable; robot stop is not verified by this launcher.', file=sys.stderr)
    except Exception:
        metadata['status'] = 'failed'
        record()
        raise
    finally:
        if metadata['status'] != 'failed':
            metadata['status'] = 'closed' if outcome == 0 else 'attention_required'
        record()
    print(f'System logs: {run_dir.relative_to(plan.root)}; robot stop not verified by parent.', flush=True)
    return outcome


@contextmanager
def signal_stop():
    """使用Windows和POSIX上提供的处理器；退出后恢复。"""
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    previous = {}
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, 'SIGBREAK'):
        signals.append(signal.SIGBREAK)
    try:
        for item in signals:
            previous[item] = signal.signal(item, stop)
        yield lambda: stopping
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


def main(argv=None, *, root=ROOT, deps=None, profile_validator=None,
         default_model='ur10e', default_profile=DEFAULT_PROFILE):
    parser = argparse.ArgumentParser(description=__doc__)
    if default_model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported entry robot model')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--robot-model', choices=SUPPORTED_MODELS, default=default_model,
                        help='Expected profile identity; dedicated entry identity cannot be changed')
    parser.add_argument('--guest-session', type=Path)
    parser.add_argument('--address', help='Current laptop private LAN IPv4; otherwise reuse cached address')
    parser.add_argument('--port', type=int, help='HTTPS port; otherwise reuse cached port or 8768')
    parser.add_argument('--setup-port', type=int, help='Certificate setup port; otherwise cached port or 8769')
    parser.add_argument('--execute', action='store_true', help='Explicitly start full system onsite; iPad Start still required')
    args = parser.parse_args(argv)
    if args.robot_model != default_model:
        parser.error(f'This {default_model} entry requires the same robot model; use the dedicated {args.robot_model} entry')
    plan = prepare_plan(root=root, profile=args.profile or default_profile, guest_session=args.guest_session,
                        address=args.address, https_port=args.port, setup_port=args.setup_port,
                        profile_validator=profile_validator, robot_model=default_model)
    if not args.execute:
        label = plan.robot_model[:2].upper() + plan.robot_model[2:]
        print(f'{label} reviewed profile, local pairing and files validated offline; no services, sockets, certificates or robot started.')
        print(f'Configured LAN address {plan.address} is not verified on this network; use --address onsite if it changed.')
        return 0
    with signal_stop() as should_stop:
        return run_system(plan, deps=deps, should_stop=should_stop)
