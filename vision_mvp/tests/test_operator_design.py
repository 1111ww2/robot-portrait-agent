"""操作员选择的数字设计；所有测试都是纯净的，没有套接字。"""
import copy
import math

import pytest

from test_physical_runtime import data
from test_ursim_profile import snapshots
from leica_tracker.operator_design import (
    INSTALLATION_SELECTION_FIELDS, INTRO_SELECTION_FIELDS, SelectedDesignProfile,
    prepare_selected_design, selected_profile, validate_operator_selection,
)


START = [-1.3132341543780726, -1.3506406110576172, -2.0171611309051514,
         -1.3065217298320313, 1.5492992401123047, .19203215837478638]


def design(model='ur5e'):
    return prepare_selected_design(data(), model, start_joints=START)


@pytest.mark.parametrize('model', ['ur5e', 'ur10e'])
def test_design_reuses_relative_motion_and_shared_angles_without_fabricated_reviews(model):
    source = data(); before = copy.deepcopy(source)
    result = prepare_selected_design(source, model, start_joints=START)
    assert source == before
    assert result['approval_mode'] == 'operator_selected_design'
    assert result['robot_model'] == result['reviewed_robot_model'] == model
    assert result['operator_design']['starting_joints_rad'] == START
    assert result['operator_design']['reference_policy'] == 'capture_stationary_on_explicit_start'
    assert result['operator_design']['tcp_policy'] == 'use_current_controller_tcp'
    assert result['operator_design']['physical_acceptance'] is False
    assert result['installation']['joints'] == START
    assert result['intro']['reference_joints'] == START
    assert result['intro']['waypoints'][-1] == START
    assert len(result['intro']['waypoints']) == 14 and len(result['paths']) == 6
    for original, point in zip(source['intro']['waypoints'], result['intro']['waypoints']):
        assert [q-home for q, home in zip(point, START)] == pytest.approx(
            [q-home for q, home in zip(original, source['intro']['reference_joints'])])
    for cue, path in result['paths'].items():
        assert [q-home for q, home in zip(path[-1], result['installation']['reference'][:3])] == pytest.approx(
            [q-home for q, home in zip(source['paths'][cue][-1], source['installation']['reference'][:3])])
    assert result['intro']['joint_speed'] == .7 and result['intro']['joint_acceleration'] == 1.2
    assert result['installation']['gesture'] == dict(duration_s=3.3, repetitions=2, joint_speed_limit=.7)
    assert 'intro_preview' not in result and 'reviewed_installation' not in result
    assert 'review' not in result['installation'] and 'reviewed' not in result['intro']


def test_same_joint_angles_have_model_specific_nominal_cartesian_reference():
    five, ten = design('ur5e'), design('ur10e')
    assert five['installation']['joints'] == ten['installation']['joints']
    assert math.dist(five['installation']['reference'][:3], ten['installation']['reference'][:3]) > .1
    assert five['intro']['reference_pose'] == five['installation']['reference']
    assert ten['intro']['reference_pose'] == ten['installation']['reference']


def test_canonical_configuration_roundtrip_preserves_exact_numeric_selection():
    from leica_tracker.motion_configuration import configure_motion_design
    value = configure_motion_design(design('ur5e'), joint_speed=.7, duration_s=3.3, repetitions=2)
    profile = selected_profile(value)
    assert profile.cfg.robot_model == 'ur5e'


def test_changed_default_joint_metadata_cannot_disagree_with_nominal_design():
    value=design();value['operator_design']['starting_joints_rad'][0]+=.1
    with pytest.raises(ValueError,match='starting'):
        selected_profile(value)


@pytest.mark.parametrize('key, value', [('gate_timeout', .3), ('max_joint_speed', .1), ('hold_timeout', 999.)])
def test_runtime_values_are_part_of_the_selected_design_binding(key, value):
    changed=design();changed[key]=value
    with pytest.raises(ValueError,match='selection'):
        selected_profile(changed)


@pytest.mark.parametrize('model', ['UR5e', 'ur5', 'ur11', None])
def test_unsupported_model_is_not_silently_relabelled(model):
    with pytest.raises(ValueError):
        design(model)


@pytest.mark.parametrize('bad', [[0.]*5, [0.]*5+[float('nan')], [0.]*5+[True]])
def test_invalid_shared_start_is_rejected_before_building_any_profile(bad):
    with pytest.raises(ValueError):
        prepare_selected_design(data(), 'ur5e', start_joints=bad)


@pytest.mark.parametrize('part, fields', [
    ('installation', INSTALLATION_SELECTION_FIELDS), ('intro', INTRO_SELECTION_FIELDS),
])
def test_numeric_selection_is_exact_kind_bound_and_does_not_mean_physical_acceptance(part, fields):
    record = design()[part]
    assert validate_operator_selection(record, kind=part, fields=fields) is True
    assert record['operator_selection']['physical_acceptance'] is False
    for key in ['payload_sha256', 'kind', 'physical_acceptance']:
        changed = copy.deepcopy(record)
        changed['operator_selection'][key] = True if key == 'physical_acceptance' else 'changed'
        with pytest.raises(ValueError):
            validate_operator_selection(changed, kind=part, fields=fields)
    with pytest.raises(ValueError):
        validate_operator_selection(record, kind='intro' if part == 'installation' else 'installation', fields=fields)
    with pytest.raises(ValueError):
        validate_operator_selection(record, kind=part, fields=fields[:-1])


@pytest.mark.parametrize('part, key', [('installation', 'speed'), ('intro', 'joint_speed')])
def test_changing_selected_numerics_invalidates_the_selection(part, key):
    record = design(); record[part][key] *= 1.01
    with pytest.raises(ValueError, match='selection'):
        selected_profile(record)


def test_legacy_missing_selection_is_not_approved_by_the_new_helper():
    assert validate_operator_selection({}, kind='intro', fields=INTRO_SELECTION_FIELDS) is False
    with pytest.raises(ValueError):
        selected_profile(data())


def test_offline_loader_has_no_preview_requirement_and_uses_the_real_compilers():
    from leica_tracker.physical_intro_program import IntroProgram
    value = selected_profile(design())
    assert isinstance(value, SelectedDesignProfile)
    assert value.provenance == 'OPERATOR_SELECTED_DESIGN' and value.physical_acceptance is False
    assert value.reference_bound is False
    assert value.cfg.robot_model == 'ur5e' and value.cfg.joints == tuple(START)
    assert 'review' not in value.intro and 'intro_preview' not in value.intro
    for program in [IntroProgram(value.intro, 123).program(), value.cfg.program(123)]:
        assert program.startswith('def ')
        assert 'set_tcp(' not in program and 'set_payload(' not in program
        assert 'set_tool_voltage(' not in program


def test_model_identity_mismatch_is_not_permitted_by_selected_mode():
    value = design(); value['reviewed_robot_model'] = 'ur10e'
    with pytest.raises(ValueError):
        selected_profile(value)


def test_manual_operator_pose_and_current_tcp_reanchor_only_the_bound_session():
    spec = design(); original = copy.deepcopy(spec)
    profile = selected_profile(spec)
    primary, feedback = snapshots()
    primary['tcp_offset'] = [.01, -.02, .3, 0., 0., .04]
    feedback['joints'][0] += .12
    bound = profile.bind_reference(primary, feedback)
    assert spec == original and profile.cfg.joints == tuple(START)
    assert bound.reference_bound is True and bound.physical_acceptance is False
    assert bound.cfg.joints == tuple(feedback['joints'])
    assert bound.cfg.neutral == tuple(feedback['tcp_pose'])
    assert bound.cfg.tcp == tuple(primary['tcp_offset'])
    assert bound.intro['waypoints'][-1] == feedback['joints']
    assert bound.selection_metadata['starting_joints_rad'] == START
    assert bound.selection_metadata['captured_joints_rad'] == feedback['joints']
    assert bound.selection_metadata['nominal_reference_only'] is False
    assert 'intro_preview' not in bound.intro
    assert bound.intro['joint_speed'] == .7


@pytest.mark.parametrize('failure', ['running', 'mode', 'protective', 'joint_motion',
                                     'linear_motion', 'angular_motion', 'disagreement', 'nan'])
def test_binding_rejects_unusable_native_snapshots(failure):
    primary, feedback = snapshots()
    if failure == 'running': primary['mode']['running'] = True
    if failure == 'mode': primary['mode']['mode'] = 5
    if failure == 'protective': primary['mode']['protective'] = True
    if failure == 'joint_motion': feedback['joint_speed'][1] = .004
    if failure == 'linear_motion': feedback['tcp_speed'][0] = .001
    if failure == 'angular_motion': feedback['tcp_speed'][3] = .003
    if failure == 'disagreement': primary['pose'][0] += .001
    if failure == 'nan': feedback['tcp_pose'][0] = float('nan')
    with pytest.raises(ValueError):
        selected_profile(design()).bind_reference(primary, feedback)


def test_bind_hook_only_uses_injected_readonly_capture_on_explicit_call():
    calls=[]; primary, feedback=snapshots()
    reader=object();control={'epoch':3};stop=lambda:False
    def capture(profile, **arguments):
        calls.append((profile, arguments));return primary,feedback
    profile=selected_profile(design(), capture_provider=capture)
    assert calls == []
    bound=profile.bind_before_motion(reader=reader,control=control,should_stop=stop)
    assert len(calls)==1 and calls[0][0] is profile
    assert calls[0][1]==dict(reader=reader,control=control,should_stop=stop)
    assert bound.reference_bound is True
    with pytest.raises(RuntimeError,match='bound'):
        bound.bind_before_motion(reader=reader,control=control,should_stop=stop)


def test_stop_before_capture_never_calls_provider():
    calls=[]
    profile=selected_profile(design(),capture_provider=lambda *_a,**_k:calls.append(True))
    with pytest.raises(RuntimeError,match='stop'):
        profile.bind_before_motion(reader=None,control=None,should_stop=lambda:True)
    assert calls == []


def test_operator_selected_design_has_no_inherited_arbitrary_base_angle_cap():
    value=design()
    assert value['installation']['base_limit'] is None
    profile=selected_profile(value)
    assert profile.cfg.base_limit is None
    # 实际失败的UR5e运行在与选定的起始点相距-0.3rad时停止；
    # 在逆转之前，一个过期的项目位置限制器固定了phase_time。
    sample=dict(tcp_pose=list(profile.cfg.neutral),tcp_speed=[0.]*6,
                joints=list(profile.cfg.joints))
    sample['joints'][0]-=.35
    profile.cfg.check_feedback(sample)
    program=profile.cfg.program(123)
    assert 'norm(q[0]-' not in program
    assert '/(base_rate*' not in program and '/(-base_rate*' not in program
    assert 'base_lookahead' not in program
    # 该修复仅移除继承的位置范围，而不是速度、控制器IK/安全、等距坐标系边界、租约或完成检查。
    assert 'base_scale = smaller(1.0,0.7000000000/' in program
    assert 'norm(qd[0]) > 0.8400000000' in program
    assert '0.7000000000/larger(0.0000001,norm(rate))' in program
    assert 'if norm(qd[i]) > 0.7370000000:' in program
    assert 'get_inverse_kin_has_solution' in program and 'is_within_safety_limits(probe)' in program
    assert 'if not in_box(actual)' in program and 'packet_age > 0.25' in program
    assert 'gesture_age + get_steptime()*gesture_scale' in program


def test_legacy_reviewed_installation_cannot_silently_drop_base_angle_cap():
    from leica_tracker.physical_resident_engine import Installation
    from test_physical_resident_engine import reviewed
    source=reviewed();source['base_limit']=None
    with pytest.raises(ValueError,match='base_limit'):
        Installation(source)


def test_bounded_legacy_design_still_rejects_outside_base_and_keeps_both_governors():
    from leica_tracker.physical_resident_engine import Installation
    from test_physical_resident_engine import reviewed
    cfg=Installation(reviewed())
    sample=dict(tcp_pose=list(cfg.neutral),tcp_speed=[0.]*6,joints=list(cfg.joints))
    sample['joints'][0]+=.1
    with pytest.raises(ValueError,match='Base joint'):
        cfg.check_feedback(sample)
    program=cfg.program(123)
    assert 'norm(q[0]-' in program
    assert '/(base_rate*0.35)' in program and '/(-base_rate*0.35)' in program
