"""在前台运行本地教学仿真器，按 Ctrl+C 关闭。"""
from pathlib import Path
import argparse
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'tmp/python-lab-review'


def run_service(command, cwd, *, stop_file=None):
    """转发终止请求并回收仿真进程，避免遗留孤立进程。"""
    process = subprocess.Popen(command, cwd=cwd)
    stopping = False
    def terminate(*_):
        nonlocal stopping
        stopping = True
        if process.poll() is None:
            process.terminate()
    handlers = {sig: signal.signal(sig, terminate)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while not stopping and not (stop_file and stop_file.exists()):
            try:
                return process.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                continue
        return 130
    finally:
        terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--stop-file',type=Path)
    args = parser.parse_args()
    if not 1024 <= args.http_port <= 65535:
        parser.error('HTTP port must be between 1024 and 65535')
    python = RUNTIME / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        raise SystemExit('Run scripts/setup-teacher-simulator.py first.')
    command = [str(python), '-u', '-m', 'urx_browser_simulator', '--host', '127.0.0.1',
               '--http-port', str(args.http_port), '--urx-port', '30002', '--web-root',
               str(RUNTIME / 'source/web/dist-workbench-v3'), '--web-entry', 'workbench-v3.html',
               '--scene-file', str(RUNTIME / 'scene.json')]
    try:
        return run_service(command, cwd=RUNTIME,stop_file=args.stop_file)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
