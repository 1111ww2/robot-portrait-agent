import pytest

from leica_tracker.physical_portrait_policy import PhysicalPortraitPolicy

NS = 1_000_000_000
STEP = 200_000_000


def frame(policy, timestamp, *, epoch=1, now=None, tracking=True, within=True):
    return policy.observe(epoch, source_ns=timestamp, now_ns=timestamp if now is None else now,
                          tracking=tracking, within=within)


def dwell(policy, start, *, epoch=1):
    for i in range(10):
        assert not frame(policy, start + STEP * i, epoch=epoch)
    return frame(policy, start + 10 * STEP, epoch=epoch)


def test_only_fresh_frames_after_release_can_complete_two_second_dwell():
    p = PhysicalPortraitPolicy()
    p.start(1)
    assert not frame(p, NS)
    p.release_observation(1, NS)
    assert not frame(p, NS)
    assert not frame(p, NS + 1, now=NS + 400_000_000)
    assert dwell(p, NS + STEP)
    assert p.latched
    assert not p.begin_motion(1)


def test_gap_loss_or_departure_resets_unlocked_dwell():
    p = PhysicalPortraitPolicy()
    p.start(1)
    p.release_observation(1, NS)
    assert not frame(p, NS + STEP)
    assert not frame(p, NS + 2 * STEP)
    assert not frame(p, NS + 4 * STEP)  # 400毫秒间隔
    assert not frame(p, NS + 5 * STEP, within=False)
    assert not frame(p, NS + 6 * STEP, tracking=False)
    assert dwell(p, NS + 7 * STEP)


def test_duplicate_old_future_and_stale_images_do_not_advance_dwell():
    p = PhysicalPortraitPolicy()
    p.start(1)
    p.release_observation(1, NS)
    t = NS + STEP
    assert not frame(p, t)
    assert not frame(p, t, now=t + STEP)
    assert not frame(p, t + STEP, now=t)  # 未来形象
    assert not frame(p, t + STEP, now=t + STEP + 300_000_001)
    assert not frame(p, t - 1)
    assert dwell(p, NS + 3 * STEP)


def test_motion_blocks_frames_and_new_gate_release_is_required():
    p = PhysicalPortraitPolicy()
    p.start(1)
    p.release_observation(1, NS)
    assert not frame(p, NS + STEP)
    assert p.begin_motion(1)
    assert not frame(p, 5 * NS)
    with pytest.raises(ValueError):
        p.release_observation(1, NS)
    p.release_observation(1, 5 * NS)
    assert not frame(p, 5 * NS)
    assert dwell(p, 5 * NS + STEP)


def test_photo_lock_survives_subject_leaving_but_stop_and_disconnect_clear():
    p = PhysicalPortraitPolicy()
    p.start(1)
    p.release_observation(1, NS)
    assert dwell(p, NS + STEP)
    assert frame(p, 5 * NS, tracking=False, within=False)
    p.expire(1, 7 * NS)
    assert p.latched
    assert p.stop(1)
    assert not p.latched and not p.active
    p.start(2)
    assert not p.stop(1)
    assert p.active
    assert p.disconnect(2)
    assert not p.active and not p.latched


def test_expiry_or_invalid_timestamp_cannot_carry_forward_dwell():
    p = PhysicalPortraitPolicy()
    p.start(1)
    p.release_observation(1, NS)
    assert not frame(p, NS + STEP)
    p.expire(1, NS + STEP + 300_000_001)
    assert not frame(p, NS + 2 * STEP, now=NS + 2 * STEP)
    assert not frame(p, True)
    assert dwell(p, 2 * NS)


def test_session_and_gate_must_be_explicit():
    p = PhysicalPortraitPolicy()
    with pytest.raises(ValueError):
        p.start(True)
    p.start(1)
    with pytest.raises(ValueError):
        p.start(2)
    with pytest.raises(ValueError):
        p.release_observation(1, 0)
    assert not frame(p, NS, epoch=2)


def test_previously_seen_frames_cannot_be_replayed_after_tracking_loss():
    p = PhysicalPortraitPolicy()
    p.start(1)
    p.release_observation(1, NS)
    for i in range(5):
        assert not frame(p, NS + (i + 1) * STEP)
    lost = NS + 6 * STEP
    assert not frame(p, lost, tracking=False)
    # 旧帧在丢失后无法重新启动计时器。
    for i in range(5):
        assert not frame(p, NS + (i + 1) * STEP, now=lost)
    assert dwell(p, NS + 7 * STEP)
