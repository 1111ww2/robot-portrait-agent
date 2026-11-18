"""重建笔记本上的本地仿真环境，不启动相机或实体机器人。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
VISION=ROOT/'vision_mvp'
MODEL_SHA='59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Only verify prerequisite tools and bundled model')
    parser.add_argument('--skip-simulator',action='store_true',help='Vision/tests only; does not establish simulator readiness')
    args=parser.parse_args()
    uv=shutil.which('uv');node=shutil.which('node')
    npm=shutil.which('npm.cmd' if sys.platform=='win32' else 'npm')
    if not all((uv,node,npm)):
        raise SystemExit('Missing uv, Node.js or npm. Read LAPTOP_START_HERE.md; reopen terminal after installation.')
    version=subprocess.check_output([node,'--version'],text=True).strip()
    if int(version.removeprefix('v').split('.')[0])<20:
        raise SystemExit('Node.js 20+ required for the pinned simulator build.')
    model=VISION/'models/pose_landmarker_lite.task'
    if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest()!=MODEL_SHA:
        raise SystemExit('Bundled pose model missing/checksum mismatch. Re-fetch this branch; do not substitute models.')
    if args.check_only:
        print('Prerequisites/model OK; Python environments and runtime have NOT been validated.')
        return 0
    def run(command,cwd=ROOT):
        print('RUN: '+' '.join(map(str,command)),flush=True)
        subprocess.run(list(map(str,command)),cwd=cwd,check=True)
    run([uv,'sync','--frozen','--python','3.12','--extra','handoff'],VISION)
    python=VISION/'.venv'/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    run([python,VISION/'scripts/prepare-local-voice.py'])
    run([python,'-m','pytest','-q'],VISION)
    run([python,'-m','compileall','-q','src','scripts'],VISION)
    run([python,'-m','leica_tracker','--help'],VISION)
    run([node,'--test',VISION/'tests/guest_player.test.cjs'])
    if not args.skip_simulator:
        run([python,VISION/'scripts/setup-teacher-simulator.py'])
    report={'platform':sys.platform,'python':str(python),'node':version,
            'software_checks':'passed','simulator_installed':not args.skip_simulator,
            'runtime_simulator_checked':False,'camera_checked':False,
            'gemini_checked':False,'ipad_checked':False,'physical_robot_checked':False}
    target=ROOT/'tmp/laptop-bootstrap.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print('Software installed/checked. Next: LAPTOP_START_HERE.md runtime checks. No device acceptance claimed.')
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except subprocess.CalledProcessError as error:raise SystemExit(f'Stopped: command exit {error.returncode}; fix that error before continuing.')
