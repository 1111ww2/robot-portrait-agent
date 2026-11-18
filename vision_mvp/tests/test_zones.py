import pytest

from leica_tracker.zones import Zone, ZoneStabilizer, classify_zone


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, Zone.LEFT),
        (0.32, Zone.LEFT),
        (1 / 3, Zone.CENTER),
        (0.50, Zone.CENTER),
        (2 / 3, Zone.RIGHT),
        (0.90, Zone.RIGHT),
        (1.0, Zone.RIGHT),
    ],
)
def test_classify_zone_uses_equal_horizontal_thirds(x, expected):
    assert classify_zone(x) is expected


@pytest.mark.parametrize("x", [-0.01, 1.01])
def test_classify_zone_rejects_coordinates_outside_normalised_frame(x):
    with pytest.raises(ValueError, match="between 0 and 1"):
        classify_zone(x)


def test_classify_zone_keeps_previous_zone_inside_hysteresis_margin():
    assert classify_zone(0.36, previous=Zone.LEFT, margin=0.05) is Zone.LEFT
    assert classify_zone(0.40, previous=Zone.LEFT, margin=0.05) is Zone.CENTER
    assert classify_zone(0.31, previous=Zone.CENTER, margin=0.05) is Zone.CENTER
    assert classify_zone(0.70, previous=Zone.CENTER, margin=0.05) is Zone.CENTER
    assert classify_zone(0.62, previous=Zone.RIGHT, margin=0.05) is Zone.RIGHT
    assert classify_zone(0.60, previous=Zone.RIGHT, margin=0.05) is Zone.CENTER


def test_stabilizer_emits_transition_only_after_required_consistent_frames():
    stabilizer = ZoneStabilizer(required_frames=4, lost_frames=3)

    assert [stabilizer.update(Zone.LEFT) for _ in range(3)] == [None, None, None]
    assert stabilizer.update(Zone.LEFT) is Zone.LEFT
    assert stabilizer.stable_zone is Zone.LEFT
    assert stabilizer.update(Zone.LEFT) is None


def test_stabilizer_resets_pending_transition_when_candidate_changes():
    stabilizer = ZoneStabilizer(required_frames=3, lost_frames=3)

    assert stabilizer.update(Zone.LEFT) is None
    assert stabilizer.update(Zone.LEFT) is None
    assert stabilizer.update(Zone.CENTER) is None
    assert stabilizer.update(Zone.LEFT) is None
    assert stabilizer.update(Zone.LEFT) is None
    assert stabilizer.update(Zone.LEFT) is Zone.LEFT


def test_stabilizer_tolerates_short_detection_loss_then_emits_lost():
    stabilizer = ZoneStabilizer(required_frames=2, lost_frames=3)
    stabilizer.update(Zone.RIGHT)
    assert stabilizer.update(Zone.RIGHT) is Zone.RIGHT

    assert stabilizer.update(None) is None
    assert stabilizer.update(None) is None
    assert stabilizer.stable_zone is Zone.RIGHT
    assert stabilizer.update(None) is Zone.LOST
    assert stabilizer.stable_zone is Zone.LOST
    assert stabilizer.update(None) is None


@pytest.mark.parametrize(
    ("required_frames", "lost_frames"),
    [(0, 2), (2, 0)],
)
def test_stabilizer_rejects_non_positive_thresholds(required_frames, lost_frames):
    with pytest.raises(ValueError, match="positive"):
        ZoneStabilizer(required_frames=required_frames, lost_frames=lost_frames)
