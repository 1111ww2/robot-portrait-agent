import pytest

from leica_tracker.physical_feedback import physical_edge_feedback


def sample(**changes):
    values = dict(cue='MOVE_LEFT', action_id=3, expected_action_id=3,
                  started_at=9.9, captured_at=10.0, now=10.05, max_age=.2, active=True,
                  reference_position=[0., 0., 0.], current_position=[-.03, 0., 0.],
                  direction=[-1., 0., 0.], planned_displacement=.06)
    values.update(changes)
    return values


def test_feedback_uses_projected_measured_displacement():
    assert physical_edge_feedback(**sample()) == {'edge': 'left', 'tone': 'red', 'intensity': .5}
    assert physical_edge_feedback(**sample(current_position=[.02, 0., 0.]))['intensity'] == 0.
    assert physical_edge_feedback(**sample(current_position=[-.12, 0., 0.]))['intensity'] == 1.


@pytest.mark.parametrize('cue,edge', [('MOVE_RIGHT', 'right'), ('MOVE_UP', 'top'),
    ('MOVE_DOWN', 'bottom'), ('MOVE_CLOSER', 'all'), ('MOVE_FARTHER', 'all')])
def test_feedback_preserves_cue_edge(cue, edge):
    assert physical_edge_feedback(**sample(cue=cue))['edge'] == edge


@pytest.mark.parametrize('changes', [dict(active=False), dict(action_id=2),
    dict(captured_at=9.0), dict(started_at=10.01), dict(captured_at=10.1), dict(captured_at=float('nan')),
    dict(current_position=[float('inf'), 0., 0.]), dict(reference_position=[0, 0]),
    dict(direction=[0, 0, 0]), dict(direction=[2, 0, 0]), dict(planned_displacement=0),
    dict(planned_displacement=float('nan')), dict(max_age=0), dict(cue='HOLD')])
def test_invalid_or_stale_feedback_is_not_rendered(changes):
    assert physical_edge_feedback(**sample(**changes)) is None
