"""离线动力学矛盾，而不是物理调试证据。"""
import copy
import json
import math

import pytest

from leica_tracker.physical_speed_audit import audit_intro_speed


def intro(points=None):
    return {
        'reference_joints': [0.] * 6,
        'waypoints': points or [[0., 0., 0., 0., .075, 0.], [0.] * 6],
        'max_tcp_angular_speed': .12,
        'max_tcp_linear_speed': .065,
    }


def test_single_joint_short_segment_finds_angular_conflict_without_fk():
    # A=.075rad在1.2rad/s²时，在.5s内产生一个三角形的.3rad/s峰值。
    report = audit_intro_speed(intro(), .6, 1.2)
    assert report['target_joint_speed_rad_s'] == .6
    assert report['target_joint_acceleration_rad_s2'] == 1.2
    assert report['max_tcp_angular_speed_rad_s'] == .12
    assert report['tcp_angular_monitor_enabled'] is True
    assert report['tcp_angular_guard_origin'] == 'project_software_monitor_not_machine_limit'
    assert report['max_tcp_linear_speed_m_s'] == .065
    assert report['tcp_linear_monitor_enabled'] is True
    assert report['tcp_linear_guard_origin'] == 'project_software_monitor_not_machine_limit'
    conflict = report['known_angular_conflicts'][0]
    assert conflict['segment_index'] == 1
    assert conflict['isolated_joint_index'] == 4
    assert conflict['isolated_joint_distance_rad'] == pytest.approx(.075)
    assert conflict['ideal_tcp_angular_peak_rad_s'] == pytest.approx(.3)
    assert conflict['ideal_duration_s'] == pytest.approx(.5)
    assert report['segments'][1]['ideal_tcp_angular_peak_rad_s'] == pytest.approx(.3)


def test_explicit_null_angular_monitor_keeps_peaks_without_claiming_guard_assessment():
    data = intro()
    data['max_tcp_angular_speed'] = None
    original = copy.deepcopy(data)

    report = audit_intro_speed(data, .6, 1.2)

    assert report['tcp_angular_monitor_enabled'] is False
    assert report['tcp_angular_guard_origin'] == 'project_software_monitor_not_machine_limit'
    assert report['max_tcp_angular_speed_rad_s'] is None
    assert report['max_tcp_linear_speed_m_s'] == .065
    assert report['known_angular_conflicts'] == []
    assert report['segments'][0]['ideal_tcp_angular_peak_rad_s'] == pytest.approx(.3)
    assert report['segments'][0]['ideal_duration_s'] == pytest.approx(.5)
    assert all(segment['angular_guard_conflict'] is None for segment in report['segments'])
    assert report['tcp_linear_speed_assessed'] is False
    assert report['requires_onsite_review'] is True
    assert data == original
    json.dumps(report, allow_nan=False)


def test_missing_angular_monitor_does_not_silently_disable_it():
    data = intro()
    del data['max_tcp_angular_speed']
    with pytest.raises(ValueError):
        audit_intro_speed(data, .6, 1.2)


@pytest.mark.parametrize('angular_limit', [.12, None])
def test_explicit_null_linear_monitor_preserves_timing_and_angular_audit(angular_limit):
    data = intro()
    data['max_tcp_linear_speed'] = None
    data['max_tcp_angular_speed'] = angular_limit
    original = copy.deepcopy(data)

    report = audit_intro_speed(data, .6, 1.2)

    assert report['tcp_linear_monitor_enabled'] is False
    assert report['tcp_linear_guard_origin'] == 'project_software_monitor_not_machine_limit'
    assert report['max_tcp_linear_speed_m_s'] is None
    assert report['tcp_angular_monitor_enabled'] is (angular_limit is not None)
    assert len(report['known_angular_conflicts']) == (2 if angular_limit is not None else 0)
    assert report['segments'][0]['ideal_tcp_angular_peak_rad_s'] == pytest.approx(.3)
    assert report['segments'][0]['ideal_duration_s'] == pytest.approx(.5)
    assert report['joint_peaks_are_measured'] is False
    assert report['tcp_linear_speed_assessed'] is False
    assert report['requires_onsite_review'] is True
    assert data == original
    json.dumps(report, allow_nan=False)


def test_missing_linear_monitor_does_not_silently_disable_it():
    data = intro()
    del data['max_tcp_linear_speed']
    with pytest.raises(ValueError):
        audit_intro_speed(data, .6, 1.2)


def test_long_single_joint_segment_includes_cruise_time():
    report = audit_intro_speed(intro([[.6, 0., 0., 0., 0., 0.], [0.] * 6]), .6, 1.2)
    segment = report['segments'][0]
    assert segment['ideal_joint_peak_rad_s'] == pytest.approx(.6)
    assert segment['ideal_duration_s'] == pytest.approx(1.5)
    assert segment['motion_profile'] == 'trapezoidal'


def test_triangle_boundary_is_continuous_and_guard_equality_is_not_conflict():
    data = intro([[.3, 0., 0., 0., 0., 0.], [0.] * 6])
    data['max_tcp_angular_speed'] = .6
    report = audit_intro_speed(data, .6, 1.2)
    assert report['segments'][0]['ideal_duration_s'] == pytest.approx(1.)
    assert report['segments'][0]['ideal_joint_peak_rad_s'] == pytest.approx(.6)
    assert report['known_angular_conflicts'] == []
    assert report['requires_onsite_review'] is True


def test_multijoint_motion_does_not_receive_single_joint_angular_identity():
    report = audit_intro_speed(intro([[.6, .3, 0., 0., 0., 0.], [0.] * 6]), .6, 1.2)
    assert report['known_angular_conflicts'] == []
    assert report['segments'][0]['ideal_tcp_angular_peak_rad_s'] is None
    assert report['segments'][0]['isolated_joint_index'] is None
    assert report['segments'][0]['angular_guard_conflict'] is None
    assert report['segments'][0]['ideal_joint_peak_rad_s'] == pytest.approx(.6)
    assert report['segments'][0]['ideal_duration_s'] == pytest.approx(1.5)
    assert report['multi_joint_tcp_angular_assessed'] is False
    assert report['tcp_linear_speed_assessed'] is False
    assert report['requires_onsite_review'] is True
    assert report['joint_peaks_are_measured'] is False


def test_stationary_segment_does_not_divide_by_zero_or_claim_return_proof():
    report = audit_intro_speed(intro([[0.] * 6, [0.] * 6]), .6, 1.2)
    assert report['segments'][0]['ideal_joint_peak_rad_s'] == 0.
    assert report['segments'][0]['ideal_duration_s'] == 0.
    assert report['segments'][0]['motion_profile'] == 'stationary'
    assert report['known_angular_conflicts'] == []
    assert report['requires_onsite_review'] is True


def test_any_nonzero_second_joint_prevents_exact_single_joint_claim():
    report = audit_intro_speed(intro([[1e-14, 0., 0., 0., .075, 0.], [0.] * 6]), .6, 1.2)
    assert report['segments'][0]['isolated_joint_index'] is None
    assert report['known_angular_conflicts'] == []


def test_negative_joint_changes_are_audited_and_input_is_not_mutated():
    data = intro([[0., 0., 0., 0., -.075, 0.], [0.] * 6])
    original = copy.deepcopy(data)
    report = audit_intro_speed(data, .6, 1.2)
    assert report['segments'][0]['isolated_joint_distance_rad'] == pytest.approx(.075)
    assert len(report['known_angular_conflicts']) == 2
    assert data == original
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('bad', [0., -1., math.inf, -math.inf, math.nan, True, False, '0.6'])
@pytest.mark.parametrize('which', ['speed', 'acceleration', 'angular_guard', 'linear_guard'])
def test_rejects_missing_or_invalid_units_bearing_scalar_values(bad, which):
    data = intro()
    speed, acceleration = .6, 1.2
    if which == 'speed': speed = bad
    elif which == 'acceleration': acceleration = bad
    elif which == 'angular_guard': data['max_tcp_angular_speed'] = bad
    else: data['max_tcp_linear_speed'] = bad
    with pytest.raises(ValueError):
        audit_intro_speed(data, speed, acceleration)


@pytest.mark.parametrize('which', ['speed', 'acceleration'])
def test_null_is_rejected_for_required_motion_parameters(which):
    data = intro()
    speed, acceleration = .6, 1.2
    if which == 'speed': speed = None
    else: acceleration = None
    with pytest.raises(ValueError):
        audit_intro_speed(data, speed, acceleration)


@pytest.mark.parametrize('bad', [None, {}, [], [0.] * 5, [0.] * 7,
                                  [0., 0., 0., 0., math.nan, 0.],
                                  [0., 0., 0., 0., math.inf, 0.],
                                  [0., 0., 0., 0., True, 0.]])
@pytest.mark.parametrize('which', ['reference', 'waypoint'])
def test_rejects_invalid_six_joint_vectors(bad, which):
    data = intro()
    if which == 'reference': data['reference_joints'] = bad
    else: data['waypoints'][0] = bad
    with pytest.raises(ValueError):
        audit_intro_speed(data, .6, 1.2)


@pytest.mark.parametrize('bad', [None, [], {}, [[0.] * 6], [[0.] * 6] * 21])
def test_rejects_missing_or_unsupported_intro_waypoint_list(bad):
    data = intro()
    data['waypoints'] = bad
    with pytest.raises(ValueError):
        audit_intro_speed(data, .6, 1.2)


def test_rejects_overflow_in_derived_joint_displacement():
    data = intro([[1e308, 0., 0., 0., 0., 0.], [0.] * 6])
    data['reference_joints'][0] = -1e308
    with pytest.raises(ValueError):
        audit_intro_speed(data, .6, 1.2)


@pytest.mark.parametrize('bad', [None, [], 'intro'])
def test_requires_intro_mapping(bad):
    with pytest.raises(ValueError):
        audit_intro_speed(bad, .6, 1.2)


def test_unrepresentable_positive_integer_is_rejected_as_invalid_input():
    with pytest.raises(ValueError):
        audit_intro_speed(intro(), 10 ** 1000, 1.2)


def test_unrepresentable_integer_joint_is_rejected_as_invalid_input():
    data = intro()
    data['reference_joints'][0] = 10 ** 1000
    with pytest.raises(ValueError):
        audit_intro_speed(data, .6, 1.2)


def test_nonfinite_derived_duration_is_rejected_before_json_export():
    with pytest.raises(ValueError):
        audit_intro_speed(intro(), 1e-320, 1.2)
