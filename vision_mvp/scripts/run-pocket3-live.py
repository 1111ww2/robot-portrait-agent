import json
from pathlib import Path
import subprocess
import time
from leica_tracker.simulator_follow import get_json, verify_identity

root = Path(__file__).resolve().parents[2]
runtime = root / 'tmp/python-lab-review'
stop = runtime / 'pocket3-live.stop'
vision_python = root / 'vision_mvp/.venv/Scripts/python.exe'
sim_python = runtime / '.venv/Scripts/python.exe'
vision = adapter = None
try:
    verify_identity(get_json('/api/workbench/v1/manifest'))
    if get_json('/api/sim/v1/motion')['urx']['clients']:
        raise RuntimeError('Another simulator controller is already connected')
    if stop.exists():
        raise RuntimeError('pocket3-live.stop exists; review it before restarting')
    subprocess.run([str(sim_python), '-c',
        "from urx_browser_simulator.student_client import SimulatorClient; SimulatorClient('http://127.0.0.1:8765').reset()"],
        cwd=root, check=True)
    with (runtime/'pocket3-camera.stderr').open('w') as camera_errors, \
            (runtime/'pocket3-live.jsonl').open('w') as output, \
            (runtime/'pocket3-live.stderr').open('w') as adapter_errors:
        vision = subprocess.Popen([str(vision_python), '-u', '-m', 'leica_tracker',
            '--camera', '1', '--emit-every-frames', '1',
            '--max-right-m', '0.10', '--max-step-m', '0.002'], cwd=root,
            stdout=subprocess.PIPE, stderr=camera_errors)
        adapter = subprocess.Popen([str(sim_python), '-u', '-m', 'leica_tracker.simulator_fixed_base',
            '--duration', '0', '--speed-percent', '100', '--depth', '--stop-file', str(stop)], cwd=root,
            stdin=vision.stdout, stdout=output, stderr=adapter_errors)
        vision.stdout.close()
        (runtime/'pocket3-live-processes.json').write_text(json.dumps(
            {'vision_pid': vision.pid, 'adapter_pid': adapter.pid, 'source': 'Pocket3', 'simulated_robot': True}))
        while adapter.poll() is None:
            time.sleep(.2)
finally:
    if adapter is not None and adapter.poll() is None:
        adapter.terminate()
        adapter.wait(timeout=3)
    if vision is not None and vision.poll() is None:
        subprocess.run(['taskkill', '/PID', str(vision.pid), '/T', '/F'], capture_output=True, timeout=5)
