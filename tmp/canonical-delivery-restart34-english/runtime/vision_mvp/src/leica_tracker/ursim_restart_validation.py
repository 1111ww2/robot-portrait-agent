"""在隔离的官方 Docker URSim 上验证停止后重新开始。
仅使用 127.0.0.2 和已验证的仿真网关，以带标记的新帧夹具驱动实际访客接口和共享的原生常驻程序。
全程不连接实体控制器或 AI 服务。导入模块或省略 --execute 时不启动服务。"""
import argparse
from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

from .ursim_system import ROOT, fresh_profile, serve, verify_docker
from .ursim_transport import LoopbackGateway, SimRTDEChannel, SIM_HOST, dependencies
from .ursim_validation import (_Owner, _Participant, _NativeDiagnostics,
    _DiagnosticPrimary, _DiagnosticChannel, _DiagnosticStderr, _cleanup_steps,
    _public_state, cycle_proof, record_execution_profile)


SCENARIOS = ('reference-stop-restart', 'displaced-stop-restart', 'selected-displaced-stop-restart')


def expected_negative_owner_rejection(*, failure, scenario, generation, second_started,
                                      second_epoch, operator_state, current_gate, original_gate):
    """仅对被拒绝的第二个启动的预期心跳后果进行分类。这并不意味着成功处理了负面情况。
    呼叫者仍然必须等待实际后端的引用拒绝，并证明没有新的本地写入。"""
    if (scenario != 'displaced-stop-restart' or generation != 2
            or type(second_started) not in (int, float) or not math.isfinite(second_started)
            or second_started < 0 or type(second_epoch) is not int
            or not isinstance(failure, str)
            or re.search(r'(?:^|\s)path=/control-heartbeat phase=body status=409(?:\s|$)', failure) is None
            or not isinstance(operator_state, dict)
            or not isinstance(current_gate, dict) or not isinstance(original_gate, dict)):
        return False
    physical = operator_state.get('physical')
    revocation = physical.get('revocation') if isinstance(physical, dict) else None
    if (not isinstance(revocation, dict) or physical.get('mode') != 'intro'
            or revocation.get('mode') != 'intro' or revocation.get('reason') != 'backend_fault'
            or type(revocation.get('epoch')) is not int or revocation['epoch'] != second_epoch):
        return False
    references = [gate.get('reference') for gate in (current_gate, original_gate)]
    if any(not isinstance(value, (tuple, list)) or len(value) != 6
           or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
           for value in references):
        return False
    return (current_gate.get('blocked') is True and original_gate.get('blocked') is True
            and isinstance(original_gate.get('session'), str) and bool(original_gate['session'])
            and current_gate.get('session') == original_gate['session']
            and tuple(references[0]) == tuple(references[1]))


def expected_restart_terminal_rejection(row, *, second_started):
    """识别实际的第二次启动后服务器无返回证明的拒绝。"""
    if (not isinstance(row, dict) or row.get('kind') != 'session_error'
            or type(row.get('observed_at')) not in (int, float)
            or row['observed_at'] < second_started or not isinstance(row.get('message'), str)):
        return False
    message = row['message']
    try:
        details = json.loads(message[message.index('{'):])
    except (ValueError, TypeError):
        return False
    return (isinstance(details, dict) and details.get('category') == 'http_rejected'
            and details.get('status') == 400 and details.get('reference_op') == 'restart_release'
            and details.get('reference_reason') == 'return_unverified')


def stop_qualifies(scenario, state, feedback, stamp, now, *, blocked, reference, peak_fraction):
    """在任何故意停止触发之前都需要测量运动。"""
    if scenario not in SCENARIOS:
        raise ValueError('Unsupported restart validation scenario')
    motion = state.get('motion', {}) if isinstance(state, dict) else {}
    if (not isinstance(state, dict) or blocked is not True or state.get('active') is not True
            or motion.get('active') is not True or motion.get('fresh') is not True
            or motion.get('cue') != 'MOVE_LEFT'
            or type(stamp) not in (int, float) or not math.isfinite(stamp)
            or type(now) not in (int, float) or not math.isfinite(now)
            or not 0 <= now - stamp <= .15 or not isinstance(feedback, dict)):
        return False
    vectors = [feedback.get(k) for k in ('tcp_pose', 'tcp_speed', 'joint_speed')]
    if any(not isinstance(v, (list, tuple)) or len(v) != 6
           or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
           for v in vectors):
        return False
    error = math.dist(feedback['tcp_pose'][:3], reference[:3])
    qd, speed = feedback['joint_speed'], feedback['tcp_speed']
    if scenario in ('displaced-stop-restart','selected-displaced-stop-restart'):
        return (feedback.get('state') == 1 and peak_fraction >= .1
                and error > .002 and max(map(abs, qd)) > .003)
    completed_native = (feedback.get('state') == 3 or
                        feedback.get('state') == 0 and motion.get('phase') == 'settling')
    return (completed_native and peak_fraction >= .9 and error <= .0005
            and max(map(abs, qd)) < .003 and math.hypot(*speed[:3]) < .0005
            and math.hypot(*speed[3:]) < .002)


class _RestartParticipant:
    """新鲜的JPEG票和一个明确的方向；没有人工智能推理。"""
    _intro_frame_ready = _Participant._intro_frame_ready

    def __init__(self, display, capture, client, path):
        import cv2
        import numpy as np
        self.display, self.capture, self.client, self.path = display, capture, client, path
        self.closed = threading.Event()
        self.failure = None
        self.intro_id = None
        self.intro_ready_seen = False
        self.intro_ready_cutoff_ns = 0
        self.intro_return_proof = None
        self.epoch = None
        self.intros = {}
        self.jpeg = cv2.imencode('.jpg', np.zeros((90, 160, 3), dtype=np.uint8))[1].tobytes()
        self.session = capture.begin(client)
        self.thread = threading.Thread(target=self._run, name='restart-fixture-camera', daemon=True)

    def _run(self):
        from .person_guidance import GuidanceCue, GuidanceDecision
        try:
            with self.path.open('a', encoding='utf-8', buffering=1) as output:
                while not self.closed.is_set():
                    ticket = self.capture.ticket(self.client, self.session)
                    self.capture.accept(self.client, self.session, ticket, self.jpeg)
                    ok, frame = self.capture.read()
                    if not ok:
                        continue
                    ns = self.capture.source_monotonic_ns
                    control = self.display.control_snapshot()
                    if not control['active'] or ns <= control['control_started_ns']:
                        self.closed.wait(.03)
                        continue
                    if self.epoch != control['epoch']:
                        self.epoch = control['epoch']
                        self.intro_id = ns
                        self.intro_ready_seen = False
                        self.intro_ready_cutoff_ns = 0
                        self.intro_return_proof = None
                    if not self._intro_frame_ready(ns, time.monotonic_ns()):
                        event = dict(event='physical_intro', action='heartbeat', intro_id=self.intro_id,
                            source_monotonic_ns=ns)
                        decision = GuidanceDecision('INTRO', GuidanceCue.HOLD, None, None, None,
                                                    False, 'VALIDATION_RESTART_INTRO')
                    else:
                        self.intros[self.epoch] = dict(self.intro_return_proof)
                        blocked = self.display.reference_held(ns)
                        cue = GuidanceCue.HOLD if blocked else GuidanceCue.MOVE_LEFT
                        phase = 'PHYSICAL_REFERENCE_HOLD' if blocked else 'GUIDING'
                        decision = GuidanceDecision(phase, cue, None, None, None,
                                                    not blocked, 'VALIDATION_RESTART_DIRECTION')
                        event = dict(event='person_guidance', coordinate_frame='normalized_camera_image',
                            robot_actuated=False, tracking=not blocked, cue=cue.value, phase=phase,
                            source_monotonic_ns=ns)
                    event.update(control_instance=control['instance'], control_epoch=control['epoch'])
                    self.display.publish(frame, decision, source_monotonic_ns=ns)
                    output.write(json.dumps(event, separators=(',', ':')) + '\n')
                    self.closed.wait(.03)
        except Exception as exc:
            self.failure = type(exc).__name__ + ': ' + str(exc)

    def close(self):
        self.closed.set()
        if self.thread.ident is not None:
            self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError('Restart participant did not retire')


def execute_restart_validation(source, output, *, robot_model='ur5e',
                               scenario='reference-stop-restart', timeout=150.):
    from .canonical_delivery import runtime_source_identity
    from .guest_display import GuestDisplay
    from .horizontal_follow import check_dashboard
    from .ipad_camera import IPadCapture
    from .motion_configuration import configure_motion_design
    from .physical_runtime import verify_stationary_after_close
    if scenario not in SCENARIOS:
        raise ValueError('Unsupported restart validation scenario')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 180:
        raise ValueError('Restart validation timeout must be finite and at most180seconds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(success=False, scenario=scenario, robot_model=robot_model,
        execution_backend='official_ursim_native', physical_acceptance=False,
        input_mode='labelled_fresh_camera_and_one_direction_fixture',
        runtime_source_sha256=runtime_source_identity(ROOT)['shared_runtime_source_sha256'],
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        opening_speed_rad_s=.7, expected_explicit_start_count=2)
    stack = ExitStack()
    stopped = threading.Event()
    owners = []
    participant = backend = None
    errors = []
    diagnostics = _NativeDiagnostics()
    trace, ui_trace, bindings = [], [], []
    active_profile = [None]
    inhibit = output.parent / '.canonical-validation-incomplete-cleanup.json'
    try:
        if inhibit.exists():
            raise RuntimeError('Previous isolated native cleanup is unverified')
        identity = verify_docker(robot_model=robot_model)
        gateway = stack.enter_context(LoopbackGateway())
        profile = fresh_profile(configure_motion_design(source), robot_model=robot_model, joint_speed=.7)
        if profile.operator_design is None:
            raise ValueError('Restart validation requires the selected-design workflow')
        if scenario=='displaced-stop-restart':
            # 明确执行旧的原始参考合同。下面的选定位移场景执行当前姿势捕获。
            profile=replace(profile,preserve_restart_reference=True)
        active_profile[0] = profile
        result['controller_id'] = identity.controller_id
        def record_bound(actual):
            directory = output / ('binding-%02d' % (len(bindings) + 1))
            directory.mkdir()
            record = {}
            record_execution_profile(actual, directory, record, robot_model=robot_model)
            bindings.append(record)
            active_profile[0] = actual
        profile = replace(profile, binding_observer=record_bound)
        capture = IPadCapture()
        display = stack.enter_context(GuestDisplay(host='127.0.0.1', port=0,
            remote_capture=capture, physical_feedback=True))
        owner = _Owner(display)
        owners.append(owner)
        path = output / 'participant.jsonl'
        path.touch()
        participant = _RestartParticipant(display, capture, owner.client, path)
        factories = dependencies()
        primary_factory = factories['state_factory']
        channels, channels_lock = [], threading.Lock()
        counts = {'reader': 0, 'writer': 0, 'primary': 0}
        def primary_factory_diagnostic(host, **options):
            counts['primary'] += 1
            return _DiagnosticPrimary(primary_factory(host, **options), diagnostics,
                                      'primary-' + str(counts['primary']))
        def channel_factory(host, **options):
            kind = 'reader' if options.get('receive_only') else 'writer'
            counts[kind] += 1
            channel = _DiagnosticChannel(SimRTDEChannel(host, **options), diagnostics,
                                         kind + '-' + str(counts[kind]))
            if options.get('receive_only'):
                with channels_lock:
                    channels.append(channel)
            return channel
        factories.update(state_factory=primary_factory_diagnostic, rtde_factory=channel_factory)
        def measurement():
            with channels_lock:
                candidates = list(channels)
            samples = [channel.latest for channel in candidates if not channel.failure]
            samples = [sample for sample in samples if isinstance(sample[0], dict)]
            return max(samples, key=lambda sample: sample[1]) if samples else (None, float('-inf'))
        def run_backend():
            try:
                with redirect_stderr(_DiagnosticStderr(sys.stderr, diagnostics)):
                    serve(profile, {'port': int(display.origin.rsplit(':', 1)[1]), 'key': display.token},
                        path, SimpleNamespace(is_alive=lambda: not stopped.is_set()), identity,
                        should_stop=stopped.is_set, gateway=gateway, factories=factories)
            except Exception as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
        backend = threading.Thread(target=run_backend, name='restart-validation-backend', daemon=True)
        participant.thread.start()
        backend.start()
        started = time.monotonic()
        deadline = started + timeout
        while not capture.fresh() and time.monotonic() < min(deadline, started + 3):
            time.sleep(.01)
        if not capture.fresh():
            raise RuntimeError('Restart fixture camera did not become fresh')
        owner.start()
        first_epoch = owner.control['epoch']
        result['first_start_epoch'] = first_epoch
        generation, peak, cycles = 1, 0., [[], []]
        first_stop = None
        second_started = None
        prior_stop_errors = 0
        old_gate = None
        last_stamps = {}
        while time.monotonic() < deadline:
            if participant.failure:
                raise RuntimeError('Participant: ' + participant.failure)
            if errors:
                raise RuntimeError('Backend: ' + errors[0])
            if not backend.is_alive():
                raise RuntimeError('Persistent restart backend exited')
            state = owner.request('/state')
            ui_trace.append(_public_state(state))
            now = time.monotonic()
            with display._lock:
                gate = display._reference_gate
                gate_snapshot = dict(blocked=gate.blocked, reference=gate.reference,
                                     session=gate.session, release_ns=gate.release_ns)
            if owner.failure:
                operator_state = owner.request('/operator-state')
                if not expected_negative_owner_rejection(failure=owner.failure,
                        scenario=scenario,generation=generation,second_started=second_started,
                        second_epoch=result.get('second_start_epoch'),operator_state=operator_state,
                        current_gate=gate_snapshot,original_gate=old_gate):
                    raise RuntimeError('Owner: ' + owner.failure)
                recorded = result.setdefault('expected_owner_heartbeat_rejections', [])
                if not recorded:
                    recorded.append(dict(observed_at=now,failure=owner.failure,
                        revocation=operator_state['physical']['revocation']))
            feedback, stamp = measurement()
            if feedback is not None and 0 <= now - stamp <= .15:
                row = dict(t=now, sampled_at=stamp, generation=generation,
                    robot_timestamp=feedback['timestamp'], native_state=feedback.get('state'),
                    native_session=feedback.get('session'), native_sequence=feedback.get('sequence'),
                    tcp_pose=feedback['tcp_pose'], joints=feedback['joints'],
                    joint_speed=feedback['joint_speed'], tcp_speed=feedback['tcp_speed'])
                trace.append(row)
                actual = active_profile[0]
                origin = actual.cfg.neutral[:3]
                endpoint = actual.paths['MOVE_LEFT'][-1]
                delta = [b-a for a, b in zip(origin, endpoint)]
                length2 = sum(d*d for d in delta)
                fraction = sum((p-o)*d for p, o, d in zip(feedback['tcp_pose'], origin, delta)) / length2
                motion = state.get('motion') or {}
                if ((first_stop is None or generation == 2) and motion.get('active')
                        and motion.get('fresh') and motion.get('cue') == 'MOVE_LEFT'):
                    peak = max(peak, fraction)
                    last = last_stamps.get(generation)
                    if last is None or stamp > last[0] and feedback['timestamp'] > last[1]:
                        cycles[generation-1].append(dict(t=stamp,robot_timestamp=feedback['timestamp'],
                            fraction=fraction,joint_speed=max(map(abs, feedback['joint_speed'])),
                            tcp_error_m=math.dist(feedback['tcp_pose'][:3],origin),
                            phase='settling' if feedback.get('state') == 3 else motion.get('phase', 'moving')))
                        last_stamps[generation] = (stamp, feedback['timestamp'])
                qualified = stop_qualifies(scenario if generation == 1 else 'reference-stop-restart',
                    state, feedback, stamp, now, blocked=gate_snapshot['blocked'],
                    reference=actual.cfg.neutral, peak_fraction=peak)
                if first_stop is None and qualified:
                    old_gate = gate_snapshot
                    first_stop = dict(at=now, pose=list(feedback['tcp_pose']),
                        reference=list(gate_snapshot['reference']), peak_fraction=peak,
                        return_error_m=math.dist(feedback['tcp_pose'][:3],actual.cfg.neutral[:3]),
                        blocked=True, gate_session_sha256=hashlib.sha256(gate_snapshot['session'].encode()).hexdigest())
                    if scenario == 'reference-stop-restart':
                        first_stop['cycle'] = cycle_proof(cycles[0], require_controller_clock=True)
                    prior_stop_errors = len(diagnostics.summary()['session_errors'])
                    owner.stop()
                    result['first_stationary_stop'] = verify_stationary_after_close(SIM_HOST,
                        channel_factory=SimRTDEChannel, dashboard_check=check_dashboard)
                    print('RESTART_VALIDATION_STOP ' + json.dumps(dict(scenario=scenario,
                        generation=1, return_error_m=first_stop['return_error_m'])), flush=True)
            if first_stop is not None and generation == 1:
                if len(diagnostics.summary()['session_errors']) <= prior_stop_errors:
                    time.sleep(.01)
                    continue
                if not backend.is_alive():
                    raise RuntimeError('Backend did not survive the first Stop')
                owner = _Owner(display)
                owners.append(owner)
                owner.start()
                generation, peak = 2, 0.
                second_started = time.monotonic()
                result['second_start_epoch'] = owner.control['epoch']
                if result['second_start_epoch'] <= first_epoch:
                    raise RuntimeError('New explicit Start did not advance control epoch')
                print('RESTART_VALIDATION_NEW_START', flush=True)
                continue
            if generation == 2:
                if scenario == 'displaced-stop-restart':
                    if state.get('runtime', {}).get('status') == 'fault':
                        if state['runtime'].get('reason') != 'reference':
                            raise RuntimeError('Displaced Start failed for an unrelated reason')
                        terminal = [row for row in diagnostics.summary()['session_errors']
                                    if row['observed_at'] >= second_started]
                        if not terminal:
                            # 在后端完成独立停止证明之前，UI可能会显示未解决的门槛。保持原始验证截止日期不变。
                            time.sleep(.02)
                            continue
                        if not any(expected_restart_terminal_rejection(row,second_started=second_started)
                                   for row in terminal):
                            raise RuntimeError('Displaced Start lacks its expected reference return rejection')
                        with display._lock:
                            gate = display._reference_gate
                            if not gate.blocked or gate.session != old_gate['session'] or gate.reference != old_gate['reference']:
                                raise RuntimeError('Displaced Start cleared or replaced the old reference')
                        commands = [r for r in diagnostics.trace()
                                    if r['kind'] == 'command' and r['observed_at'] >= second_started]
                        if commands:
                            raise RuntimeError('Displaced restart sent a native command before reference verification')
                        result.update(displaced_restart_rejected=True, original_reference_retained=True,
                            no_native_command_after_second_start=True,
                            reference_return_rejection=next(row for row in terminal
                                if expected_restart_terminal_rejection(row,second_started=second_started)))
                        break
                else:
                    if state.get('runtime', {}).get('status') == 'fault' or not state.get('active'):
                        raise RuntimeError('New Start faulted: ' + str(state.get('runtime')))
                    if feedback is not None and 0 <= now-stamp <= .15 and qualified:
                        if owner.control['epoch'] not in participant.intros:
                            raise RuntimeError('Second opening did not reach its genuine post-return camera frame')
                        result.update(second_opening_complete=True,
                            second_direction_cycle=cycle_proof(cycles[1], require_controller_clock=True),
                            second_intro_camera_proof=participant.intros[owner.control['epoch']],
                            original_gate_released_before_second_direction=gate_snapshot['session'] != old_gate['session'])
                        if scenario=='selected-displaced-stop-restart':
                            shift=math.dist(active_profile[0].cfg.neutral[:3],old_gate['reference'][:3])
                            if shift<=.002 or len(bindings)!=2:
                                raise RuntimeError('Selected restart did not bind the measured displaced current pose')
                            result.update(selected_current_reference_verified=True,reference_shift_m=shift)
                        if not result['original_gate_released_before_second_direction']:
                            raise RuntimeError('Second direction reused the old reference gate')
                        break
            time.sleep(.02)
        else:
            raise TimeoutError('Native Stop/new-Start validation timed out')
        result.update(success=True, first_stop=first_stop, explicit_start_count=sum(o.start_count for o in owners),
            selected_binding_count=len(bindings), bound_profiles=bindings, elapsed_s=time.monotonic()-started)
        if result['explicit_start_count'] != 2:
            raise RuntimeError('Validation must use exactly two user-equivalent Starts')
    except BaseException as exc:
        result.update(success=False, failure=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        def close_backend():
            if backend is not None:
                backend.join(8)
                if backend.is_alive():
                    raise RuntimeError('Restart backend did not retire')
        def stopped_proof():
            if backend is not None:
                result['stationary_after_cleanup'] = verify_stationary_after_close(SIM_HOST,
                    channel_factory=SimRTDEChannel, dashboard_check=check_dashboard)
        steps = [('owner Stop', owner.stop) for owner in owners]
        steps.append(('cancel backend', stopped.set))
        if participant is not None:
            steps.append(('participant', participant.close))
        steps.extend([('backend', close_backend), ('native stationary proof', stopped_proof),
                      ('servers and gateway', stack.close)])
        cleanup_errors = _cleanup_steps(steps)
        result.update(cleanup_errors=cleanup_errors, cleanup_complete=not cleanup_errors,
            backend_errors=errors, backend_thread_alive=bool(backend is not None and backend.is_alive()),
            participant_failure=participant.failure if participant else None,
            owner_failures=[dict(owner=i+1,failure=owner.failure) for i,owner in enumerate(owners)
                            if owner.failure], **diagnostics.summary())
        if errors:
            result['success'] = False
            result.setdefault('failure', 'Backend: ' + errors[0])
        if cleanup_errors:
            result['success'] = False
            inhibit.write_text(json.dumps(dict(output=str(output), errors=cleanup_errors), indent=2), encoding='utf-8')
        for filename, data in (('native-diagnostics.json',diagnostics.trace()),('telemetry.json',trace),
                               ('ui-state.json',ui_trace),('result.json',result)):
            (output/filename).write_text(json.dumps(data, indent=2), encoding='utf-8')
    return result


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--robot-model', choices=('ur5e', 'ur10e'), default='ur5e')
    parser.add_argument('--scenario', choices=SCENARIOS, default=SCENARIOS[0])
    parser.add_argument('--source-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=150.)
    args = parser.parse_args(argv)
    root = Path(root).resolve()
    source_path = (root / args.source_profile).resolve()
    output = (args.output or root/'tmp'/('restart-validation-'+args.robot_model+'-'+
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    if not source_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        parser.error('Source and output must stay inside this workspace')
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 180:
        parser.error('Timeout must be positive and at most180seconds')
    if not args.execute:
        print('Restart validation checked offline; no Docker, process or robot interface opened.')
        return 0
    result = execute_restart_validation(json.loads(source_path.read_text(encoding='utf-8-sig')),
        output, robot_model=args.robot_model, scenario=args.scenario, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
