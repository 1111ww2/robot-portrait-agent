"""在视觉环境中启动摄像头与本地教学仿真器的引导流程。
在预览中按 Q、Esc 或在终端按 Ctrl+C 可同时停止相机和动作桥接。这里不接受实体机器人地址。"""
from pathlib import Path
import argparse
import json
import math
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
VISION = ROOT / 'vision_mvp'
LOGS = ROOT / 'tmp/guidance-live'


def manifest():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8766/api/workbench/v1/manifest',timeout=1) as response:
            data=json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError('Port 8766 is occupied by a different service') from error
    target=data.get('target',{})
    if target.get('kind')!='simulator' or target.get('authority_origin')!='simulation':
        raise RuntimeError('Local teacher simulator identity required')
    return data


def bridge_command(reference):
    command=[sys.executable,'-u','-m','leica_tracker.simulator_guidance','--source','stdin']
    if reference is not None:command.extend(['--reference',str(reference)])
    return command


def child_environment(*, simulation_feedback=False):
    """反馈仅用于本次仿真，不向局域网开放。"""
    env=os.environ.copy()
    env.pop('LEICA_SIM_FEEDBACK',None)
    env.pop('LEICA_SIM_FAST_GESTURES',None)
    if simulation_feedback:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        env['LEICA_SIM_FEEDBACK']=f'{port}:{secrets.token_hex(16)}'
        env['LEICA_SIM_FAST_GESTURES']='1'
    return env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--camera-source',choices=('webcam','ipad'),default='webcam')
    parser.add_argument('--profile',choices=('precise','coarse-demo','frame-guide','gemini-guide','claude-guide','claude-agent'),default='precise')
    parser.add_argument('--gemini-model',default='gemini-3.1-flash-lite')
    parser.add_argument('--single-plan',action='store_true')
    parser.add_argument('--sim-reference',type=Path,help='saved local simulator joint/TCP reference')
    parser.add_argument('--claude-model',default='claude-opus-5-5')
    parser.add_argument('--claude-effort',choices=('medium','high'),default='medium')
    parser.add_argument('--claude-env-file',type=Path,default=ROOT/'.env.local')
    parser.add_argument('--gemini-env-file',type=Path,default=ROOT/'.env.local')
    parser.add_argument('--source-mirrored',action=argparse.BooleanOptionalAction,
                        default=False,help='camera input is already mirrored')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--guest-display',action='store_true')
    parser.add_argument('--duration',type=float,default=0.)
    args=parser.parse_args()
    if args.sim_reference is not None:
        args.sim_reference=args.sim_reference.resolve()
        if not args.sim_reference.is_relative_to(ROOT) or not args.sim_reference.is_file():
            parser.error('--sim-reference must name a file inside this workspace')
    if args.single_plan and args.profile!='claude-guide':parser.error('--single-plan requires --profile claude-guide')
    if args.camera_source=='ipad' and (not args.guest_display or not args.headless):
        parser.error('iPad source requires --guest-display --headless')
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('duration must be finite and nonnegative')
    if args.camera_source!='ipad' and args.profile in ('gemini-guide','claude-guide','claude-agent'):
        from leica_tracker.app import AppConfig, _framing_client
        from leica_tracker.gemini_framing import FramingError
        try:
            _framing_client(AppConfig(guidance_profile=args.profile,
                gemini_model=args.gemini_model,gemini_env_file=args.gemini_env_file,
                claude_model=args.claude_model,claude_effort=args.claude_effort,
                claude_env_file=args.claude_env_file)).check_access()
        except (FramingError,ValueError) as error:
            print(f'{args.profile} preflight failed: {error}; camera and controller not started.',file=sys.stderr)
            return 4
    LOGS.mkdir(parents=True,exist_ok=True)
    processes=[]
    service_stop=None
    service=None
    handles=[]
    stopping=False

    def stop(*_):
        nonlocal stopping
        stopping=True

    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)

    def log(name):
        handle=(LOGS/name).open('w',encoding='utf-8',buffering=1)
        handles.append(handle)
        return handle

    try:
        try:
            manifest()
        except urllib.error.URLError:
            # 通过停止文件让包装程序在 Windows 上也能回收子进程；
            # Popen.terminate() 会绕过 Python 的信号和 finally 处理。
            import os
            service_stop=LOGS/f'simulator-{os.getpid()}.stop'
            service_stop.unlink(missing_ok=True)
            service=subprocess.Popen([sys.executable,str(VISION/'scripts/run-teacher-simulator.py'),
                '--http-port','8766','--stop-file',str(service_stop)],cwd=ROOT,stdout=log('simulator.log'),stderr=subprocess.STDOUT)
            processes.append(service)
            deadline=time.monotonic()+30
            while True:
                if service.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Simulator did not start; see tmp/guidance-live/simulator.log')
                try:
                    manifest()
                    break
                except urllib.error.URLError:
                    time.sleep(.2)
        camera_command=[sys.executable,'-u','-m','leica_tracker','--camera',str(args.camera),
                        '--camera-source',args.camera_source,
                        '--mode','guide-person','--guidance-profile',args.profile,
                        '--guidance-stream','--emit-every-frames','1']
        if args.profile=='gemini-guide':
            camera_command.extend(['--gemini-model',args.gemini_model,
                                   '--gemini-env-file',str(args.gemini_env_file.resolve())])
        if args.profile in ('claude-guide','claude-agent'):
            camera_command.extend(['--claude-model',args.claude_model,
                '--claude-effort',args.claude_effort,'--claude-env-file',str(args.claude_env_file.resolve())])
        if args.source_mirrored:
            camera_command.append('--source-mirrored')
        if args.headless:
            camera_command.append('--headless')
        if args.single_plan:camera_command.append('--single-plan')
        if args.guest_display:
            camera_command.extend(['--guest-display','--guest-host','0.0.0.0'])
        if args.duration:
            camera_command.extend(['--duration',str(args.duration)])
        child_env=child_environment(simulation_feedback=args.sim_reference is not None and args.guest_display)
        camera=subprocess.Popen(camera_command,cwd=VISION,stdout=subprocess.PIPE,
                                stderr=log('camera.log'),env=child_env)
        processes.append(camera)
        bridge=subprocess.Popen(bridge_command(args.sim_reference),cwd=VISION,stdin=camera.stdout,
            stdout=log('bridge.jsonl'),stderr=log('bridge.log'),env=child_env)
        processes.append(bridge)
        camera.stdout.close()
        print('Webcam guidance -> simulated UR5e: http://127.0.0.1:8766/',flush=True)
        print(f'Guidance profile: {args.profile}; physical framing unverified; '
              'coarse-demo capture is simulated and saves no photo.',flush=True)
        print(f'Camera source mirrored: {args.source_mirrored}; preview mirrored: True; '
              'left/right cues use participant directions.',flush=True)
        print('Q/Esc: stop preview and bridge. R: reset session. Logs: tmp/guidance-live/',flush=True)
        while not stopping and camera.poll() is None and bridge.poll() is None:
            time.sleep(.1)
        if camera.poll() is not None and camera.returncode:
            raise RuntimeError('Webcam failed; see tmp/guidance-live/camera.log')
        if bridge.poll() is not None and bridge.returncode:
            raise RuntimeError('Simulator bridge failed; see tmp/guidance-live/bridge.log')
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                if process is service and service_stop is not None:
                    service_stop.touch()
                    try:
                        process.wait(timeout=4)
                        continue
                    except subprocess.TimeoutExpired:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        for handle in handles:
            handle.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
