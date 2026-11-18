"""离线选择实体机型，测试中的管理面板通道均为替身。"""
import copy
from types import SimpleNamespace

import pytest

from leica_tracker import horizontal_follow
from leica_tracker.physical_intro_bootstrap import open_intro
from leica_tracker.physical_intro_program import IntroProgram
from leica_tracker.physical_resident_engine import Installation
from leica_tracker.physical_runtime import reviewed_profile
from leica_tracker.physical_step_preview import (SharedPreviewPlan, StepProgram, StepSequence,
                                                 draft_step_template, shared_plan_template)
from leica_tracker.physical_review_merge import merge_preview
from test_horizontal_follow import _DashboardSocket
from test_continuous_preview import continuous_data
from test_physical_intro_bootstrap import setup as intro_setup
from test_physical_intro_program import profile as intro_profile
from test_physical_resident_engine import reviewed as installation_profile
from test_physical_runtime import data as runtime_profile
from test_physical_step_preview import example as step_example
from test_physical_shared_preview import reviewed as shared_reviewed, measured
from test_physical_review_merge import draft_and_report


def explicit(profile, model='ur5e'):
    profile['robot_model'] = profile['reviewed_robot_model'] = model
    return profile


def test_model_helper_preserves_legacy_default_and_requires_exact_explicit_review():
    from leica_tracker.physical_robot_model import SUPPORTED_MODELS, model_for_profile
    assert set(SUPPORTED_MODELS) == {'ur5e', 'ur10e'}
    assert model_for_profile({}) == 'ur10e'
    assert model_for_profile({}, default='ur5e') == 'ur5e'
    for model in SUPPORTED_MODELS:
        assert model_for_profile(explicit({}, model)) == model
        with pytest.raises(ValueError, match='model'):
            model_for_profile({'robot_model': model})
        with pytest.raises(ValueError, match='model'):
            model_for_profile({'robot_model': model, 'reviewed_robot_model': 'ur3e'})
    for model in ('ur5', 'UR5e', 'ur3e', None, False, ['ur5e']):
        with pytest.raises(ValueError, match='model'):
            model_for_profile(explicit({}, model))
    with pytest.raises(ValueError, match='model'):
        model_for_profile({}, default='ur5')


@pytest.mark.parametrize('factory,make_profile', [
    (lambda p: IntroProgram(p, 7), intro_profile),
    (Installation, installation_profile),
    (lambda p: StepProgram(p, 7), lambda: step_example()[1]),
])
def test_programs_retain_immutable_reviewed_model(factory, make_profile):
    candidate = explicit(make_profile())
    compiled = factory(candidate)
    assert compiled.robot_model == 'ur5e'
    candidate['robot_model'] = candidate['reviewed_robot_model'] = 'ur10e'
    assert compiled.robot_model == 'ur5e'
    with pytest.raises(AttributeError):
        compiled.robot_model = 'ur10e'
    candidate = explicit(make_profile())
    candidate['reviewed_robot_model'] = 'ur10e'
    with pytest.raises(ValueError, match='model'):
        factory(candidate)


@pytest.mark.parametrize('model,native', [('ur5e', 'UR5'), ('ur5e', 'UR5e'),
                                         ('ur10e', 'UR10'), ('ur10e', 'UR10e')])
def test_dashboard_checks_selected_model_and_eseries_software(monkeypatch, model, native):
    fake = _DashboardSocket([b'Connected\n', b'Safetystatus: NORMAL\n', b'true\n',
                             (native+'\n').encode(), b'URSoftware 5.13.0.113898\n'])
    monkeypatch.setattr(horizontal_follow.socket, 'create_connection', lambda *_a, **_k: fake)
    horizontal_follow.check_dashboard('192.0.2.1', robot_model=model)
    assert fake.sent[-2:] == ['get robot model', 'PolyscopeVersion']


@pytest.mark.parametrize('model,native', [('ur5e', 'UR5'), ('ur10e', 'UR10')])
def test_cb3_software_is_rejected_for_both_models(monkeypatch, model, native):
    fake = _DashboardSocket([b'Connected\n', b'Safetystatus: NORMAL\n', b'true\n',
                             (native+'\n').encode(), b'URSoftware 3.15.0\n'])
    monkeypatch.setattr(horizontal_follow.socket, 'create_connection', lambda *_a, **_k: fake)
    with pytest.raises(RuntimeError, match='e-Series'):
        horizontal_follow.check_dashboard('192.0.2.1', robot_model=model)


@pytest.mark.parametrize('native', ['UR10', 'UR10e', 'UR3e'])
def test_ur5e_dashboard_rejects_other_models(monkeypatch, native):
    fake = _DashboardSocket([b'Connected\n', b'Safetystatus: NORMAL\n', b'true\n',
                             (native+'\n').encode()])
    monkeypatch.setattr(horizontal_follow.socket, 'create_connection', lambda *_a, **_k: fake)
    with pytest.raises(RuntimeError, match='model mismatch'):
        horizontal_follow.check_dashboard('192.0.2.1', robot_model='ur5e')


def test_ur5e_intro_uses_selected_model_on_first_and_final_dashboard_checks():
    candidate, deps, objects = intro_setup()
    session = open_intro('192.0.2.1', explicit(candidate), lambda: None, dependencies=deps)
    try:
        assert len(objects['dashboard_checks']) == 2
        assert all(kwargs['robot_model'] == 'ur5e' for _, kwargs in objects['dashboard_checks'])
        assert objects['writer'].sent == []
    finally:
        session.close()


def test_mismatched_intro_model_is_rejected_before_any_dashboard_or_channel():
    candidate, deps, objects = intro_setup()
    explicit(candidate)
    candidate['reviewed_robot_model'] = 'ur10e'
    with pytest.raises(ValueError, match='model'):
        open_intro('192.0.2.1', candidate, lambda: None, dependencies=deps)
    assert objects == {}


def reviewed_variant():
    candidate = explicit(runtime_profile())
    explicit(candidate['installation'])
    candidate['reviewed_installation'] = copy.deepcopy(candidate['installation'])
    explicit(candidate['intro'])
    explicit(candidate['intro_preview'])
    return candidate


def test_runtime_binds_top_intro_installation_and_preview_to_ur5e():
    result = reviewed_profile(reviewed_variant())
    assert result.cfg.robot_model == IntroProgram(result.intro, 1).robot_model == 'ur5e'


@pytest.mark.parametrize('section', ['installation', 'intro', 'intro_preview'])
def test_ur5e_runtime_rejects_legacy_ur10e_nested_review(section):
    candidate = reviewed_variant()
    candidate[section].pop('robot_model')
    candidate[section].pop('reviewed_robot_model')
    if section == 'installation':
        candidate['reviewed_installation'] = copy.deepcopy(candidate['installation'])
    with pytest.raises(ValueError, match='model'):
        reviewed_profile(candidate)


def test_legacy_ur10e_runtime_rejects_explicit_ur5e_nested_review():
    candidate = runtime_profile()
    explicit(candidate['intro'])
    with pytest.raises(ValueError, match='model'):
        reviewed_profile(candidate)


def continuous_variant():
    candidate = continuous_data()
    for profile in (candidate, candidate['installation'], candidate['intro'],
                    candidate['intro_preview'], candidate['intro_preview']['execution_profile'],
                    candidate['intro_preview']['execution_profile']['intro']):
        explicit(profile)
    candidate['reviewed_installation'] = copy.deepcopy(candidate['installation'])
    preview = candidate['intro_preview']
    preview['reviewed'] = {key: copy.deepcopy(preview[key]) for key in preview['reviewed']}
    return candidate


def test_ur5e_continuous_preview_binds_executed_model_and_acceptance():
    assert reviewed_profile(continuous_variant()).cfg.robot_model == 'ur5e'


@pytest.mark.parametrize('nested_intro', [False, True])
def test_ur5e_continuous_preview_rejects_legacy_execution_model(nested_intro):
    candidate = continuous_variant()
    execution = candidate['intro_preview']['execution_profile']
    profile = execution['intro'] if nested_intro else execution
    profile.pop('robot_model')
    profile.pop('reviewed_robot_model')
    candidate['intro_preview']['reviewed']['execution_profile'] = copy.deepcopy(execution)
    with pytest.raises(ValueError, match='model'):
        reviewed_profile(candidate)


def shared_variant():
    draft, plan = shared_reviewed()
    for profile in (draft, draft['intro'], plan):
        explicit(profile)
    return draft, plan


def test_ur5e_shared_plan_binds_identity_and_generates_matching_reviewed_steps():
    draft, profile = shared_variant()
    plan = SharedPreviewPlan(draft, profile)
    step = plan.step_profile(0, [], measured(profile), now=10.)
    assert plan.robot_model == step['robot_model'] == step['reviewed_robot_model'] == 'ur5e'
    assert StepProgram(step, 7).robot_model == 'ur5e'
    with pytest.raises(AttributeError):
        plan.robot_model = 'ur10e'


@pytest.mark.parametrize('section', ['draft', 'intro', 'plan'])
def test_shared_plan_rejects_mixed_models(section):
    draft, plan = shared_variant()
    profile = {'draft': draft, 'intro': draft['intro'], 'plan': plan}[section]
    profile.pop('robot_model')
    profile.pop('reviewed_robot_model')
    with pytest.raises(ValueError, match='model'):
        SharedPreviewPlan(draft, plan)


def test_ur5e_preview_templates_preserve_selection_without_inventing_approval():
    draft, _ = shared_variant()
    for template in (draft_step_template(draft, 0), shared_plan_template(draft)):
        assert template['robot_model'] == 'ur5e'
        assert template['reviewed_robot_model'] is None
        assert template['status'] == 'DRAFT'
        assert template['reviewed'] == {} and not any(template['review'].values())


@pytest.mark.parametrize('ur5e_draft', [False, True])
def test_step_sequence_rejects_other_model_before_opening(ur5e_draft):
    draft, step = step_example()
    calls = []
    if ur5e_draft:
        explicit(draft)
        explicit(draft['intro'])
    else:
        explicit(step)
    sequence = StepSequence(draft, draft['robot_ip'], lambda: None,
                            opener=lambda *_args: calls.append('open'))
    with pytest.raises(ValueError, match='model'):
        sequence.next(step)
    assert calls == [] and sequence.state == 'WAITING'


def test_ur5e_acceptance_records_only_the_model_used_for_all_fourteen_fake_steps(monkeypatch):
    from leica_tracker import physical_step_preview
    now = [10.]
    monkeypatch.setattr(physical_step_preview, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    draft, profile = shared_variant()
    plan = SharedPreviewPlan(draft, profile)
    opened_models = []
    class Session:
        def __init__(self, step):
            program = StepProgram(step, 1)
            opened_models.append(program.robot_model)
            self.progress = SimpleNamespace(program=program)
            self.reader = SimpleNamespace(failure=None, latest=({
                'joints': step['target_joints'], 'tcp_pose': step['reference_pose'],
                'tcp_speed': [0.]*6, 'joint_speed': [0.]*6}, now[0]))
        def tick(self): return 'COMPLETE'
        def stop(self): pass
        def close(self): pass
    sequence = StepSequence(draft, draft['robot_ip'], lambda: None,
                            opener=lambda _host, step, _guest: Session(step))
    for index in range(14):
        now[0] += .01
        stamp = now[0]
        current = measured(profile, index=index, stamp=stamp)
        step = plan.step_profile(index, sequence.observations, current, now=stamp)
        sequence.next(step)
        sequence.tick()
    acceptance = sequence.acceptance('accept')
    assert opened_models == ['ur5e']*14
    assert acceptance['robot_model'] == acceptance['reviewed_robot_model'] == 'ur5e'
    assert 'robot_model' not in acceptance['reviewed']


def test_ur5e_preview_merge_accepts_matching_identity_and_keeps_draft_unapproved():
    draft, report = draft_and_report()
    for profile in (draft, draft['intro'], report):
        explicit(profile)
    result = merge_preview(draft, report)
    assert result['intro_preview']['robot_model'] == 'ur5e'
    assert result['status'] == 'DRAFT'
    assert not any(result['intro']['review'].values())


def test_ur5e_preview_merge_accepts_matching_continuous_execution_without_approving_draft():
    draft = continuous_variant()
    report = copy.deepcopy(draft['intro_preview'])
    draft['status'] = 'DRAFT'
    draft['intro']['review'] = {key: False for key in draft['intro']['review']}
    draft['intro_preview'] = {'status': 'NOT_ACCEPTED'}
    result = merge_preview(draft, report)
    assert result['intro_preview'] == report and result['status'] == 'DRAFT'
    assert not any(result['intro']['review'].values())


@pytest.mark.parametrize('section', ['draft', 'intro', 'report'])
def test_ur5e_preview_merge_rejects_mixed_models(section):
    draft, report = draft_and_report()
    for profile in (draft, draft['intro'], report):
        explicit(profile)
    profile = {'draft': draft, 'intro': draft['intro'], 'report': report}[section]
    profile.pop('robot_model')
    profile.pop('reviewed_robot_model')
    with pytest.raises(ValueError, match='model'):
        merge_preview(draft, report)
