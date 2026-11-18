import importlib
from types import SimpleNamespace

import pytest


def motion_module():
    return importlib.import_module("leica_tracker.motion")


def command(*, tracking=True, right=0.0, down=0.0, toward_subject=0.0):
    return SimpleNamespace(
        tracking=tracking,
        right=right,
        down=down,
        toward_subject=toward_subject,
        reason="TRACKING" if tracking else "PERSON_LOST",
    )


def test_motion_mapper_limits_each_update_to_a_small_camera_relative_step():
    motion = motion_module()
    mapper = motion.MotionIntentMapper(
        motion.MotionEnvelope(
            max_right_m=0.20,
            max_down_m=0.12,
            max_toward_subject_m=0.18,
            max_step_m=0.01,
        )
    )

    intent = mapper.update(command(right=1.0, down=-1.0, toward_subject=1.0))

    assert intent.hold is False
    assert intent.right_m == pytest.approx(0.01)
    assert intent.down_m == pytest.approx(-0.01)
    assert intent.toward_subject_m == pytest.approx(0.01)


def test_motion_mapper_never_exceeds_the_configured_envelope():
    motion = motion_module()
    envelope = motion.MotionEnvelope(max_step_m=1.0)
    mapper = motion.MotionIntentMapper(envelope)

    intent = mapper.update(command(right=5.0, down=-5.0, toward_subject=5.0))

    assert intent.right_m == pytest.approx(envelope.max_right_m)
    assert intent.down_m == pytest.approx(-envelope.max_down_m)
    assert intent.toward_subject_m == pytest.approx(envelope.max_toward_subject_m)


def test_motion_mapper_holds_current_position_immediately_on_person_loss():
    motion = motion_module()
    mapper = motion.MotionIntentMapper(motion.MotionEnvelope(max_step_m=1.0))
    moving = mapper.update(command(right=0.5, down=0.25, toward_subject=-0.5))

    lost = mapper.update(command(tracking=False))

    assert lost.hold is True
    assert lost.reason == "PERSON_LOST"
    assert lost.right_m == pytest.approx(moving.right_m)
    assert lost.down_m == pytest.approx(moving.down_m)
    assert lost.toward_subject_m == pytest.approx(moving.toward_subject_m)
