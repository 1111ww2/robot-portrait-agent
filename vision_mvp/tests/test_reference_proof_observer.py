"""保持读取通道打开并执行参考位验证回调，之后重新收集最新数据做验证。"""
from dataclasses import replace

import pytest

from leica_tracker.operator_design import selected_profile
from leica_tracker.operator_design_capture import capture_selected_reference
from leica_tracker.ursim_profile import build_profile
from test_operator_design import design
from test_operator_design_capture import setup
from test_ursim_profile import snapshots


def test_proof_hook_runs_once_with_open_readers_then_rebuilds_full_dwell():
    profile, options, clock, objects = setup()
    proofs = []
    samples = []
    def sample(primary, native, sampled_at):
        samples.append(sampled_at)
    def proved(primary, native, sampled_at):
        assert all(not item.closed for item in objects['channels'])
        assert clock.now >= 10.6
        assert sampled_at == min(primary['mode']['timestamp'] / 1e6, native['timestamp'])
        assert samples and sampled_at > samples[-1]
        proofs.append(clock.now)
        primary['pose'][0] = native['tcp_pose'][0] = 99.
        return 'ignored'
    primary, native = capture_selected_reference(profile, sample_observer=sample,
        proof_observer=proved, **options)
    assert len(proofs) == 1 and clock.now - proofs[0] >= .6
    assert primary['pose'][0] == native['tcp_pose'][0] == .4
    assert all(item.closed for item in objects['channels'])


def test_blocking_proof_hook_reacquires_fresh_evidence_within_original_deadline():
    profile, options, clock, objects = setup()
    completed = []
    def proved(*_):
        clock.sleep(.3)
        completed.append(clock.now)
    primary, native = capture_selected_reference(profile, proof_observer=proved, **options)
    assert len(completed) == 1 and clock.now - completed[0] >= .6
    assert clock.now < 15.
    assert clock.now - primary['mode']['timestamp'] / 1e6 <= .15
    assert clock.now - native['timestamp'] <= .15
    assert all(item.closed for item in objects['channels'])


def test_proof_hook_preserves_docker_check_and_both_subsequent_dwell_periods():
    profile, options, clock, objects = setup()
    events = []
    def docker_check():
        clock.sleep(.25)
        events.append(('docker', clock.now))
    def proved(*_):
        events.append(('proof', clock.now))
    options['dependencies']['before_final_snapshot'] = docker_check
    capture_selected_reference(profile, proof_observer=proved, **options)
    assert [item[0] for item in events] == ['docker', 'proof']
    assert events[1][1] - events[0][1] >= .6
    assert clock.now - events[1][1] >= .6
    assert all(item.closed for item in objects['channels'])


@pytest.mark.parametrize('change', ['stop', 'epoch', 'exception', 'tcp'])
def test_proof_hook_cancellation_or_changes_prevent_admission_and_close_readers(change):
    changed = [False]
    def guest(state, _):
        if change == 'epoch' and changed[0]:
            state['epoch'] += 1
    def native(kind, frame, stamp, now):
        if change == 'tcp' and changed[0] and kind == 'primary':
            frame['tcp_offset'][2] += .001
    profile, options, _, objects = setup(change=native, guest_change=guest)
    def proved(*_):
        changed[0] = True
        if change == 'stop':
            objects['stop'] = True
        if change == 'exception':
            raise RuntimeError('proof rejected')
    with pytest.raises(RuntimeError):
        capture_selected_reference(profile, proof_observer=proved, **options)
    assert all(item.closed for item in objects['channels'])


def test_proof_callback_cannot_extend_fixed_five_second_deadline():
    profile, options, clock, objects = setup()
    began = []
    def proved(*_):
        began.append(clock.now)
        clock.sleep(4.5)
    with pytest.raises(RuntimeError, match='timed out|expired'):
        capture_selected_reference(profile, proof_observer=proved, **options)
    assert clock.now == pytest.approx(began[0] + 4.5)
    assert all(item.closed for item in objects['channels'])


def test_invalid_proof_hook_is_rejected_before_opening_readers():
    profile, options, _, objects = setup()
    with pytest.raises(ValueError, match='observer'):
        capture_selected_reference(profile, proof_observer=True, **options)
    assert not objects['channels'] and not objects['dashboard']


def test_remote_proof_pending_keeps_sampling_within_the_same_capture():
    profile, options, clock, objects = setup()
    proofs = []
    samples = []
    def proved(*_):
        proofs.append(clock.now)
        return len(proofs) > 1
    capture_selected_reference(profile, sample_observer=lambda *a: samples.append(a[2]),
        proof_observer=proved, **options)
    assert len(proofs) == 2
    assert any(proofs[0] < stamp < proofs[1] for stamp in samples)
    assert clock.now - proofs[-1] >= .6 and clock.now < 15.
    assert all(item.closed for item in objects['channels'])


def test_remote_proof_that_never_becomes_ready_times_out_without_admission():
    profile, options, clock, objects = setup()
    with pytest.raises(RuntimeError, match='timed out'):
        capture_selected_reference(profile, proof_observer=lambda *_: False, **options)
    assert clock.now <= 15.05
    assert all(item.closed for item in objects['channels'])


@pytest.mark.parametrize('simulation', [False, True])
def test_selected_profile_forwards_optional_proof_hook_without_ignoring_it(simulation):
    calls = []
    def provider(profile, **kwargs):
        calls.append(kwargs)
        return snapshots()
    if simulation:
        profile = replace(build_profile(design('ur5e'), *snapshots(), robot_model='ur5e',
            joint_speed=.7), capture_provider=provider)
    else:
        profile = selected_profile(design('ur5e'), capture_provider=provider)
    observer = lambda *_: None
    profile.bind_before_motion(reader=lambda: None, control={}, should_stop=lambda: False,
        proof_observer=observer)
    assert calls[0]['proof_observer'] is observer and 'sample_observer' not in calls[0]
