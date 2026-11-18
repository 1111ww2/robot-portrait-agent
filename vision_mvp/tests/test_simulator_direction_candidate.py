"""仿真来源的比较结果不能授权实体机器人移动。"""
import json
from pathlib import Path

import pytest

from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.physical_spatial import SpatialConfig
from leica_tracker.robot_gestures import gesture_plan_for
from leica_tracker.simulator_reference_guidance import ReferenceMapper


CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'simulator_direction_candidate.unreviewed.json'


def test_unreviewed_candidate_matches_sim_fast_reference_direction():
    data = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    assert data['kind'] == 'unreviewed_simulator_direction_candidate'
    assert data['physical_mapping_verified'] is False
    assert data['hardware_execution_enabled'] is False
    assert data['coordinate_frame'] == 'simulator_base_axes'
    mapper = ReferenceMapper.__new__(ReferenceMapper)
    mapper._solve = lambda translation: translation
    assert set(data['directions']) == {
        'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN', 'MOVE_CLOSER', 'MOVE_FARTHER'
    }
    for cue, expected in data['directions'].items():
        plan = gesture_plan_for(GuidanceCue(cue), profile='sim-fast')
        displacement = mapper.offset(plan, plan.duration_ms / 4)
        index = 'XYZ'.index(expected[1])
        sign = 1 if expected[0] == '+' else -1
        assert displacement[index] * sign > 0
        assert all(value == 0 for i, value in enumerate(displacement) if i != index)


def test_candidate_is_not_a_physical_execution_profile():
    data = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    assert 'status' not in data
    assert 'review' not in data
    assert not any(key in data for key in ('speed_m_s', 'acceleration_m_s2', 'neutral_tcp_m_rad'))
    with pytest.raises((KeyError, ValueError)):
        SpatialConfig.from_dict(data)
