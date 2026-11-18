"""完整的物理/iPad条目在离线状态下仍然不活跃，仅拥有其运行状态。"""
import importlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    source = Path(__file__).resolve().parents[1] / 'src/leica_tracker/physical_system.py'
    assert source.is_file(), 'Full-stack physical system entry is not implemented'
    return importlib.import_module('leica_tracker.physical_system')


@pytest.fixture
def workspace():
    fixtures = Path(__file__).resolve().parents[2] / 'tmp/test-physical-system'
    fixtures.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='case-', dir=fixtures) as directory:
        root = Path(directory)
        make_workspace(root)
        yield root


def make_workspace(root):
    (root / 'vision_mvp/scripts').mkdir(parents=True)
    (root / 'vision_mvp/scripts/run-physical-live.py').write_text('# fixture\n')
    (root / 'vision_mvp/src/leica_tracker').mkdir(parents=True)
    (root / 'vision_mvp/src/leica_tracker/app.py').write_text('# fixture\n')
    live = root / 'tmp/guidance-live'
    (live / 'https').mkdir(parents=True)
    (live / 'https/pair.json').write_text(json.dumps({'port': 8767, 'key': 'k' * 32}))
    (live / 'https-public-status.json').write_text(json.dumps({'address': '192.168.1.24', 'port': 8768, 'setup_port': 8769}))
    profile = root / 'tmp/robot-network/physical-ipad-ur10e-trial.json'
    profile.parent.mkdir(parents=True)
    profile.write_text('{"review":"fixture"}')


def plan_for(root, **kwargs):
    return module().prepare_plan(root=root, profile_validator=lambda value: value, **kwargs)


def test_offline_default_validates_existing_files_without_any_runtime_boundary(workspace, capsys):
    system = module()
    before = {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('Offline validation touched runtime dependency: ' + name)
    assert system.main([], root=workspace, deps=Forbidden(), profile_validator=lambda value: value) == 0
    after = {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    assert before == after
    output = capsys.readouterr().out
    assert 'offline' in output.lower() and 'k' * 32 not in output
    assert 'not verified' in output.lower()


@pytest.mark.parametrize('address', ['127.0.0.1', '0.0.0.0', '169.254.1.1', '8.8.8.8', '::1'])
def test_offline_rejects_non_lan_address(workspace, address):
    with pytest.raises(ValueError, match='private LAN IPv4'):
        plan_for(workspace, address=address)


def test_offline_review_failure_is_not_bypassed(workspace):
    def reject(_):
        raise ValueError('Review does not match this speed')
    with pytest.raises(ValueError, match='Review'):
        module().prepare_plan(root=workspace, profile_validator=reject)


def test_paths_cannot_escape_workspace(workspace):
    outside = workspace.parent / 'outside.json'
    with pytest.raises(ValueError, match='workspace'):
        plan_for(workspace, profile=outside)


def test_duplicate_ports_and_mismatched_pairing_are_rejected(workspace):
    with pytest.raises(ValueError, match='distinct'):
        plan_for(workspace, https_port=8767)
    (workspace / 'tmp/guidance-live/guest-link.json').write_text(json.dumps({'port': 8767, 'key': 'x' * 32}))
    with pytest.raises(ValueError, match='pairing.*differ'):
        plan_for(workspace)


class Child:
    pid = 4321
    def __init__(self, *, timeout=False, exited=None):
        self.timeout = timeout
        self.exited = exited
        self.waited = False
    def poll(self):
        return self.exited
    def wait(self, timeout):
        self.waited = True
        if self.timeout:
            raise subprocess.TimeoutExpired('own-child', timeout)
        self.exited = 0
        return 0
    def terminate(self):
        pytest.fail('Physical child must retire cooperatively')
    kill = terminate


def runtime_dependencies(system, *, child=None, occupied=False, bind_failure=False):
    events = []
    identity = {}
    child = child or Child()
    class HTTPS:
        def __enter__(self):
            events.append('https-enter')
            return self
        def __exit__(self, *args):
            events.append('https-close')
    def certificates(directory, address):
        events.append('certificates')
        return {'fixture': directory}
    def https(files, **kwargs):
        events.append(('https-bind', kwargs))
        if bind_failure:
            raise OSError('address already in use')
        return HTTPS()
    def start(command, **kwargs):
        events.append(('child', command, kwargs))
        identity['run_id'] = kwargs['env']['LEICA_SYSTEM_RUN_ID']
        return child
    def read(plan):
        return {'control': {'instance': 'own-camera-instance', 'epoch': 0}, 'active': False,
                'system_run_id': identity['run_id']}
    def stop(plan, instance, run_id):
        events.append(('stop', instance, run_id))
        return instance is not None
    deps = SimpleNamespace(probe_guest=lambda _: occupied, certificates=certificates, https=https,
                           popen=start, read_guest=read, request_stop=stop, clock=lambda: 0.,
                           sleep=lambda _: None, prepare_pairing=lambda *args: events.append('pair-files'))
    return deps, events, child


def test_execute_owns_fresh_https_and_one_child_without_starting_encounter(workspace):
    system = module()
    plan = plan_for(workspace)
    deps, events, child = runtime_dependencies(system)
    stopped = iter([False, True])
    assert system.run_system(plan, deps=deps, should_stop=lambda: next(stopped)) == 0
    command_event = next(event for event in events if isinstance(event, tuple) and event[0] == 'child')
    command = command_event[1]
    assert '--execute' in command and '--require-free-port' in command
    assert '--stop-file' in command and '--guest-session' in command
    assert not any('simulator' in str(part) for part in command)
    assert 'k' * 32 not in ' '.join(command)
    stop_path = Path(command[command.index('--stop-file') + 1])
    assert stop_path.is_relative_to(workspace) and stop_path.exists()
    metadata = json.loads((stop_path.parent / 'run.json').read_text())
    stop_event = ('stop', 'own-camera-instance', metadata['system_run_id'])
    assert child.waited and stop_event in events
    assert events.index('https-close') > events.index(stop_event)
    assert metadata['child_retired'] is True
    assert metadata['robot_stop'] == 'not_verified_by_parent'
    assert 'k' * 32 not in json.dumps(metadata)
    assert len(metadata['source_sha256']) == 64
    assert command_event[2]['env']['LEICA_SYSTEM_RUN_ID'] == metadata['system_run_id']
    bind_event = next(event for event in events if isinstance(event, tuple) and event[0] == 'https-bind')
    assert bind_event[1]['host'] == '192.168.1.24'


def test_occupied_guest_port_does_not_create_certificates_or_processes(workspace):
    system = module()
    deps, events, _ = runtime_dependencies(system, occupied=True)
    with pytest.raises(RuntimeError, match='occupied'):
        system.run_system(plan_for(workspace), deps=deps)
    assert events == []


def test_occupied_https_is_not_reused_and_no_child_is_created(workspace):
    system = module()
    deps, events, _ = runtime_dependencies(system, bind_failure=True)
    with pytest.raises(RuntimeError, match='HTTPS.*occupied|HTTPS.*bind'):
        system.run_system(plan_for(workspace), deps=deps)
    assert not any(isinstance(event, tuple) and event[0] == 'child' for event in events)


def test_unretired_child_reports_unknown_stop_without_killing(workspace, capsys):
    system = module()
    deps, events, child = runtime_dependencies(system, child=Child(timeout=True))
    stopped = iter([False, True])
    assert system.run_system(plan_for(workspace), deps=deps, should_stop=lambda: next(stopped)) == 2
    assert child.waited and any(event[:2] == ('stop', 'own-camera-instance') for event in events)
    assert 'unknown' in capsys.readouterr().err.lower()


def test_child_exit_triggers_cooperative_cleanup_and_failure(workspace):
    system = module()
    deps, events, _ = runtime_dependencies(system, child=Child(exited=2))
    assert system.run_system(plan_for(workspace), deps=deps) == 2
    assert any(path.exists() for path in (workspace / 'tmp/physical-system').glob('*/stop.request'))


def test_run_directories_do_not_overwrite_previous_logs(workspace):
    system = module()
    for _ in range(2):
        deps, _, _ = runtime_dependencies(system)
        stopped = iter([False, True])
        assert system.run_system(plan_for(workspace), deps=deps, should_stop=lambda: next(stopped)) == 0
    runs = list((workspace / 'tmp/physical-system').iterdir())
    assert len(runs) == 2 and all((run / 'run.json').is_file() for run in runs)


def test_startup_timeout_still_requests_stop_and_retires_own_child(workspace):
    system = module()
    deps, events, child = runtime_dependencies(system)
    clock = iter([0., 1.])
    deps.clock = lambda: next(clock)
    def unavailable(_):
        raise OSError('not ready')
    deps.read_guest = unavailable
    assert system.run_system(plan_for(workspace), deps=deps, startup_timeout=.5) == 2
    assert child.waited and any(event[:2] == ('stop', None) for event in events)


def test_stop_http_request_is_authenticated_local_and_does_not_claim_robot_stop(workspace, monkeypatch):
    system = module()
    plan = plan_for(workspace)
    requests = []
    replies = iter([{'active': False}])
    class Connection:
        def __init__(self, host, port, timeout):
            assert host == '127.0.0.1' and port == 8767
        def request(self, method, path, body=None, headers=None):
            requests.append((method, path, body, headers))
        def getresponse(self):
            return SimpleNamespace(status=200, read=lambda *args: json.dumps(next(replies)).encode())
        def close(self):
            pass
    monkeypatch.setattr(system.http.client, 'HTTPConnection', Connection)
    assert system.request_stop(plan, 'mine', 'owned-run') is True
    assert [request[:2] for request in requests] == [('POST', '/action')]
    assert json.loads(requests[0][2]) == {'action': 'stop', 'control': {'instance': 'mine'}, 'system_run_id': 'owned-run'}
    assert requests[0][3]['X-Guest-Token'] == 'k' * 32
    assert requests[0][3]['Origin'] == 'http://127.0.0.1:8767'


@pytest.mark.parametrize('instance,run_id', [(None, 'run'), ('instance', None), (None, None)])
def test_stop_does_not_target_an_unidentified_camera(workspace, monkeypatch, instance, run_id):
    system = module()
    monkeypatch.setattr(system.http.client, 'HTTPConnection', lambda *args, **kwargs: pytest.fail('must not send Stop without owned identity'))
    assert system.request_stop(plan_for(workspace), instance, run_id) is False


def test_ready_response_from_another_run_is_never_accepted(workspace, capsys):
    system = module()
    deps, events, child = runtime_dependencies(system)
    clock = iter([0., 1.])
    deps.clock = lambda: next(clock)
    deps.read_guest = lambda _: {'control': {'instance': 'foreign'}, 'system_run_id': 'other-run'}
    assert system.run_system(plan_for(workspace), deps=deps, startup_timeout=.5) == 2
    assert 'SYSTEM_READY' not in capsys.readouterr().out
    assert not any(event[:2] == ('stop', 'foreign') for event in events)
    assert child.waited


def test_stop_file_write_failure_still_waits_for_own_child(workspace, monkeypatch, capsys):
    system = module()
    deps, _, child = runtime_dependencies(system)
    original = Path.write_text
    def write(path, *args, **kwargs):
        if path.name == 'stop.request':
            raise OSError('simulated write failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', write)
    stopped = iter([False, True])
    assert system.run_system(plan_for(workspace), deps=deps, should_stop=lambda: next(stopped)) == 2
    assert child.waited
    assert 'stop file' in capsys.readouterr().err.lower()


def test_nonzero_cooperative_child_exit_is_attention_required(workspace):
    system = module()
    child = Child()
    def wait(timeout):
        child.exited = 2
        return 2
    child.wait = wait
    deps, _, _ = runtime_dependencies(system, child=child)
    stopped = iter([False, True])
    assert system.run_system(plan_for(workspace), deps=deps, should_stop=lambda: next(stopped)) == 2


def test_execute_pairing_artifacts_use_current_address_and_same_secret_only_locally(workspace, monkeypatch, capsys):
    system = module()
    plan = plan_for(workspace, address='192.168.1.25')
    der = workspace / 'fixture.der'
    der.write_bytes(b'public certificate fixture')
    encoded = []
    def make(url):
        encoded.append(url)
        return SimpleNamespace(save=lambda path: path.write_bytes(b'fixture png'))
    monkeypatch.setitem(system.sys.modules, 'qrcode', SimpleNamespace(make=make))
    system.prepare_pairing_files(plan, {'der': der})
    page = (workspace / 'tmp/guidance-live/https-ipad-pair.html').read_text(encoding='utf-8')
    assert 'https://192.168.1.25:8768/?key=' + 'k' * 32 in page
    assert 'http://192.168.1.25:8769/leica-root.mobileconfig' in encoded
    assert json.loads((workspace / 'tmp/guidance-live/https/pair.json').read_text()) == {'port': 8767, 'key': 'k' * 32}
    assert json.loads((workspace / 'tmp/guidance-live/https-public-status.json').read_text())['address'] == '192.168.1.25'
    assert 'k' * 32 not in capsys.readouterr().out


def test_signal_stop_handles_windows_and_restores_handlers(monkeypatch):
    system = module()
    installed = {}
    calls = []
    def register(item, handler):
        calls.append((item, handler))
        installed[item] = handler
        return 'previous-handler'
    monkeypatch.setattr(system.signal, 'signal', register)
    with system.signal_stop() as requested:
        assert requested() is False
        installed[system.signal.SIGINT]()
        assert requested() is True
    assert all(handler == 'previous-handler' for _, handler in calls[len(calls) // 2:])
