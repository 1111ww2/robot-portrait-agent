"""启动 iPad/Claude 应用和选定的原生动作程序。默认仅验证配置；需要 --execute 才会启动服务。
夹爪、工具电源、网络和机器人安装参数保持现状。
区分操作员选定的方案与历史实机审核记录，交互须由已配对 iPad 明确发起。"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
VISION = ROOT / 'vision_mvp'
PAIR = ROOT / 'tmp' / 'guidance-live' / 'guest-link.json'


def validate_profile(path, *, robot_model='ur10e'):
    from leica_tracker.physical_robot_model import SUPPORTED_MODELS, model_for_profile
    from leica_tracker.physical_runtime import reviewed_profile
    if robot_model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported expected robot model')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    actual_model = model_for_profile(data)
    if actual_model != robot_model:
        raise ValueError(f'Profile robot model {actual_model} does not match expected model {robot_model}')
    return reviewed_profile(data)


def validate_pairing(value):
    if not isinstance(value, dict):
        raise ValueError('Existing local iPad pairing is required')
    port, key = value.get('port'), value.get('key')
    if (type(port) is not int or not 1 <= port <= 65535 or not isinstance(key, str)
            or not 24 <= len(key) <= 128
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in key)):
        raise ValueError('Existing local iPad pairing is invalid')
    return port, key


def camera_command(python, port, env_file):
    return [python, '-u', '-m', 'leica_tracker', '--camera-source', 'ipad',
            '--mode', 'guide-person', '--guidance-profile', 'claude-guide',
            '--claude-model', 'claude-opus-5-5', '--claude-effort', 'medium',
            '--claude-env-file', str(env_file), '--single-plan', '--guidance-stream',
            '--emit-every-frames', '1', '--headless', '--guest-display',
            '--guest-host', '0.0.0.0', '--guest-port', str(port)]


def camera_environment(existing, key):
    env = dict(existing)
    env.pop('LEICA_SIM_FEEDBACK', None)
    env.pop('LEICA_SIM_FAST_GESTURES', None)
    env['LEICA_PHYSICAL_FEEDBACK'] = '1'
    env['LEICA_GUEST_SESSION_TOKEN'] = key
    return env


def _port_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.2):
            return True
    except OSError:
        return False


def ensure_guest_port_available(port):
    if not _port_open(port):
        return
    if sys.platform != 'win32':
        raise RuntimeError('Guest port is occupied; stop the existing iPad app before switching modes')
    subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(VISION / 'scripts' / 'stop-managed-simulator.ps1'), '-Port', str(port)],
                   cwd=ROOT, check=True, timeout=45)
    deadline = time.monotonic() + 5.
    while _port_open(port):
        if time.monotonic() >= deadline:
            raise RuntimeError('Guest port remains occupied after stopping the managed simulator')
        time.sleep(.1)


def _wait_for_guest(process, port, key, timeout=15.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Physical iPad app exited during startup')
        request = urllib.request.Request(f'http://127.0.0.1:{port}/state',
                                         headers={'X-Guest-Token': key})
        try:
            with urllib.request.urlopen(request, timeout=.3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.1)
    raise RuntimeError('Physical iPad app did not start')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--robot-model', choices=('ur5e', 'ur10e'),
                        help='Expected model identity; defaults to legacy ur10e')
    parser.add_argument('--guest-session', type=Path, default=PAIR)
    parser.add_argument('--guest-wait', type=float, default=600.,
                        help='legacy option; the persistent launcher now waits for Start indefinitely')
    parser.add_argument('--keep-display-on-fault', action='store_true',
                        help='legacy option; the display now always survives individual encounter faults')
    parser.add_argument('--require-free-port', action='store_true',
                        help='refuse an occupied guest port without stopping another service')
    parser.add_argument('--stop-file', type=Path,
                        help='workspace-local cooperative service retirement request')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    args.profile = args.profile.resolve()
    args.guest_session = args.guest_session.resolve()
    if not args.profile.is_relative_to(ROOT) or not args.guest_session.is_relative_to(ROOT):
        raise ValueError('Profile and pairing must be inside the project workspace')
    if args.stop_file is not None:
        args.stop_file = args.stop_file.resolve()
        if not args.stop_file.is_relative_to(ROOT):
            raise ValueError('Stop file must be inside the project workspace')
        if args.stop_file.exists():
            raise ValueError('Stop file already exists; use a fresh run directory')
    profile = (validate_profile(args.profile) if args.robot_model is None
               else validate_profile(args.profile, robot_model=args.robot_model))
    if not args.execute:
        print('Profile validated offline; no app or robot started.')
        return 0
    port, key = validate_pairing(json.loads(args.guest_session.read_text(encoding='utf-8-sig')))
    if args.require_free_port:
        if _port_open(port):
            raise RuntimeError('Guest port is occupied; no existing service was stopped')
    else:
        ensure_guest_port_available(port)
    from leica_tracker import physical_runtime
    camera = subprocess.Popen(camera_command(sys.executable, port, ROOT / '.env.local'),
                              cwd=VISION, stdout=subprocess.PIPE, text=True,
                              encoding='utf-8', bufsize=1,
                              env=camera_environment(os.environ, key))
    source = physical_runtime.LatestEventSource(camera.stdout)
    try:
        _wait_for_guest(camera, port, key)
        print('PHYSICAL_SERVICE_READY: waiting for explicit iPad Start.',file=sys.stderr,flush=True)
        return physical_runtime.serve_sessions(profile,port,key,source,
                                               is_alive=lambda:camera.poll() is None and not (
                                                   args.stop_file and args.stop_file.exists()))
    finally:
        if camera.poll() is None:
            camera.terminate()
            try:
                camera.wait(timeout=3)
            except subprocess.TimeoutExpired:
                camera.kill()
                camera.wait(timeout=3)
        if camera.stdout:
            camera.stdout.close()
        source.thread.join(timeout=1.)


if __name__ == '__main__':
    raise SystemExit(main())
