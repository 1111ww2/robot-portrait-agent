"""审核绑定验证失败时禁止继续。测试不使用硬件、网络或机器人指令。"""
import pytest

from leica_tracker.physical_resident_engine import Installation
from leica_tracker.physical_intro_program import IntroProgram
from leica_tracker.physical_step_preview import SharedPreviewPlan, StepProgram
from test_physical_resident_engine import reviewed as xyz_profile
from test_physical_intro_program import profile as intro_profile
from test_physical_shared_preview import reviewed as shared_profile, measured


def test_xyz_render_cannot_use_speed_changed_after_review():
    cfg=Installation(xyz_profile())
    try:
        cfg.speed=.15
        cfg.acceleration=.5
    except (AttributeError,TypeError):
        return
    with pytest.raises((ValueError,RuntimeError)):
        cfg.program(123)


def test_intro_render_cannot_use_dynamics_changed_after_review():
    intro=IntroProgram(intro_profile(),7)
    try:
        intro.joint_speed=.7
    except (AttributeError,TypeError):
        return
    with pytest.raises((ValueError,RuntimeError)):
        intro.program()


def test_reviewed_step_cannot_mutate_target_or_stop_parameters():
    draft,raw=shared_profile(); plan=SharedPreviewPlan(draft,raw)
    step=StepProgram(plan.step_profile(0,[],measured(raw),now=10.),1)
    with pytest.raises(AttributeError): step.final_joints=(0.,)*6
    with pytest.raises(AttributeError): step._review_frozen=False
    with pytest.raises(TypeError): step.stop_settings['watch_timeout']=.2
    assert step.stop_settings['watch_timeout']==raw['stop']['watch_timeout']


@pytest.mark.parametrize('section,key,value', [
    ('dynamics','joint_speed',.09),('stop','watch_timeout',.2),('review','stop',False)])
def test_shared_plan_cannot_generate_changed_values_from_old_review(section,key,value):
    draft,raw=shared_profile(); plan=SharedPreviewPlan(draft,raw)
    try:
        getattr(plan,section)[key]=value
    except (AttributeError,TypeError):
        return
    if not plan.matches(draft,raw):
        return
    with pytest.raises((ValueError,RuntimeError)):
        plan.step_profile(0,[],measured(raw),now=10.)


def test_shared_plan_cannot_replace_robot_identity_or_route():
    draft,raw=shared_profile(); plan=SharedPreviewPlan(draft,raw)
    with pytest.raises(AttributeError): plan.host='192.0.2.2'
    with pytest.raises(AttributeError): plan.points=tuple(reversed(plan.points))
    with pytest.raises(AttributeError): plan._review_frozen=False
    assert plan.matches(draft,raw)
