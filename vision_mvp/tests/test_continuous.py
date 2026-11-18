import importlib
from types import SimpleNamespace

import pytest


def controller_module():
    return importlib.import_module("leica_tracker.continuous")


def observation(*, center_x=0.5, head_y=0.25, shoulder_width=0.30):
    return SimpleNamespace(
        center_x=center_x,
        head_y=head_y,
        shoulder_width=shoulder_width,
        confidence=0.9,
    )


def test_controller_outputs_continuous_camera_relative_corrections():
    continuous = controller_module()
    controller = continuous.ContinuousController(
        continuous.ControllerConfig(
            target_x=0.50,
            target_head_y=0.25,
            target_shoulder_width=0.30,
            smoothing_alpha=1.0,
            horizontal_deadband=0.0,
            vertical_deadband=0.0,
            scale_deadband=0.0,
        )
    )

    command = controller.update(
        observation(center_x=0.70, head_y=0.40, shoulder_width=0.18)
    )

    assert command.tracking is True
    assert command.right == pytest.approx(0.8)
    assert command.down == pytest.approx(0.75)
    assert command.toward_subject == pytest.approx(0.8)


def test_controller_applies_deadbands_and_clamps_large_errors():
    continuous = controller_module()
    controller = continuous.ContinuousController(
        continuous.ControllerConfig(smoothing_alpha=1.0)
    )

    centered = controller.update(
        observation(center_x=0.505, head_y=0.255, shoulder_width=0.305)
    )
    extreme = controller.update(
        observation(center_x=1.0, head_y=1.0, shoulder_width=0.01)
    )

    assert (centered.right, centered.down, centered.toward_subject) == (0.0, 0.0, 0.0)
    assert (extreme.right, extreme.down, extreme.toward_subject) == (1.0, 1.0, 1.0)


def test_controller_stops_immediately_when_person_is_lost():
    continuous = controller_module()
    controller = continuous.ContinuousController(
        continuous.ControllerConfig(smoothing_alpha=1.0)
    )
    controller.update(observation(center_x=0.75))

    command = controller.update(None)

    assert command.tracking is False
    assert command.reason == "PERSON_LOST"
    assert (command.right, command.down, command.toward_subject) == (0.0, 0.0, 0.0)
