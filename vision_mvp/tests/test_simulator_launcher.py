"""在启动程序终止后，使用真正的子进程来检测孤儿进程。"""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_live_launcher_passes_explicit_reference_only_when_requested():
    import runpy
    module=runpy.run_path(str(Path(__file__).parents[1]/'scripts/run-guidance-live.py'))
    plain=module['bridge_command'](None)
    selected=module['bridge_command'](Path('example-reference.json'))
    assert '--reference' not in plain
    assert selected[-2:]==['--reference','example-reference.json']


@pytest.mark.parametrize("ignore_sigterm", [False, True])
@pytest.mark.skipif(sys.platform=='win32',reason='POSIX SIGTERM handler semantics; stop-file test covers Windows shutdown')
def test_service_wrapper_terminates_and_reaps_its_child(tmp_path, ignore_sigterm):
    module=Path(__file__).parents[1]/'scripts/run-teacher-simulator.py'
    pid_file=tmp_path/'child.json'
    child_code=("import os,time,pathlib,signal;"
                + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_sigterm else "")
                + f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(20)")
    wrapper_code=("import runpy,sys;mod=runpy.run_path(sys.argv[1]);"
                  "sys.exit(mod['run_service']([sys.executable,'-c',sys.argv[2]],sys.argv[3]))")
    wrapper=subprocess.Popen([sys.executable,'-c',wrapper_code,str(module),child_code,str(tmp_path)],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    child_pid=None
    try:
        deadline=time.monotonic()+4
        while not pid_file.exists() and time.monotonic()<deadline and wrapper.poll() is None:
            time.sleep(.02)
        assert pid_file.exists(), wrapper.communicate(timeout=1)[1].decode()
        child_pid=int(pid_file.read_text())
        wrapper.terminate()
        wrapper.communicate(timeout=5)
        try:
            os.kill(child_pid,0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('Child survived service-wrapper termination')
    finally:
        if wrapper.poll() is None:
            wrapper.kill();wrapper.wait(timeout=3)
        if child_pid:
            try:os.kill(child_pid,9)
            except ProcessLookupError:pass


def test_stop_file_reaps_child_and_closes_inherited_output_pipe(tmp_path):
    module=Path(__file__).parents[1]/'scripts/run-teacher-simulator.py'
    ready=tmp_path/'ready';stop=tmp_path/'stop'
    child_code=f"import pathlib,time;pathlib.Path({str(ready)!r}).write_text('ready');time.sleep(20)"
    wrapper_code=("import runpy,sys;from pathlib import Path;mod=runpy.run_path(sys.argv[1]);"
                  "sys.exit(mod['run_service']([sys.executable,'-c',sys.argv[2]],sys.argv[3],stop_file=Path(sys.argv[4])))")
    p=subprocess.Popen([sys.executable,'-c',wrapper_code,str(module),child_code,str(tmp_path),str(stop)],
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+4
        while not ready.exists() and time.monotonic()<deadline and p.poll() is None:time.sleep(.02)
        assert ready.exists()
        stop.touch()
        # 保留这些继承的控制器的孤儿会防止通信EOF。
        _,err=p.communicate(timeout=4)
        assert p.returncode==130,err.decode()
    finally:
        stop.touch()
        if p.poll() is None:p.kill();p.wait(timeout=3)


def test_simulation_feedback_is_per_run_and_absent_from_other_modes(monkeypatch):
    import runpy
    from leica_tracker.simulator_feedback import parse_config
    module=runpy.run_path(str(Path(__file__).parents[1]/'scripts/run-guidance-live.py'))
    monkeypatch.setenv('LEICA_SIM_FEEDBACK','unrelated')
    plain=module['child_environment']()
    one=module['child_environment'](simulation_feedback=True)
    two=module['child_environment'](simulation_feedback=True)
    assert 'LEICA_SIM_FEEDBACK' not in plain
    assert 'LEICA_SIM_FAST_GESTURES' not in plain
    assert one['LEICA_SIM_FAST_GESTURES']=='1'
    first_port,first_token=parse_config(one['LEICA_SIM_FEEDBACK'])
    second_port,second_token=parse_config(two['LEICA_SIM_FEEDBACK'])
    assert first_port>=1024 and second_port>=1024 and first_token!=second_token
