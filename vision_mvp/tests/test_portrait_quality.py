"""场景引导的构图可行性使用当前的图像几何学，不使用机器人I/O。"""
from types import SimpleNamespace
import pytest

from leica_tracker.gemini_framing import validate_plan


def plan(x=.5, y=.3, width=.24, frame=(0., 0., 1., 1.)):
    return validate_plan(dict(subject_visible=True, confidence=.9, style='environmental',
        frame=list(frame), target=[x, y, width], explanation='Balance the visible background.'))


def observation(x=.5, y=.3, width=.24, confidence=.9, landmarks=None):
    value = SimpleNamespace(center_x=x, head_y=y, shoulder_width=width, confidence=confidence)
    if landmarks is not None:
        value.landmarks = landmarks
    return value


def test_edge_goal_reserves_band_and_live_check_detects_current_oversize():
    from leica_tracker.portrait_quality import fit_plan, quality_tolerances, projected_fits
    p = fit_plan(plan(.9, .3, .18), (.85, .3, .20))
    assert p.target[0] < .9
    assert not projected_fits(p, (.85, .3, .20))
    assert projected_fits(p, (.85, .3, p.target[2]))
    assert quality_tolerances(p)[2] == pytest.approx(.018)
    assert 'margin' in p.explanation.lower()


def test_already_good_position_is_not_forced_to_move():
    from leica_tracker.portrait_quality import fit_plan
    p = plan()
    assert fit_plan(p, p.target).target == p.target


@pytest.mark.parametrize('x', [.08, .15, .85, .92])
def test_projection_is_symmetric_bounded_and_idempotent(x):
    from leica_tracker.portrait_quality import fit_plan, projected_fits
    p = fit_plan(plan(x, .3, .12), (x, .3, .16))
    assert projected_fits(p, (x, .3, p.target[2]))
    assert fit_plan(p, (x, .3, .16)).target == pytest.approx(p.target)
    assert 0 <= p.frame[0] < p.frame[2] <= 1


def test_measured_context_names_sparse_landmark_bounds_and_excludes_hands():
    from leica_tracker.portrait_quality import measured_context
    points = [SimpleNamespace(x=.5, y=.3, visibility=0., presence=0.) for _ in range(33)]
    for i, x, y in ((0,.5,.25),(7,.46,.27),(8,.54,.27),(11,.4,.4),(12,.6,.4),
                    (23,.43,.7),(24,.57,.7),(15,.99,.4),(16,.01,.4)):
        points[i] = SimpleNamespace(x=x, y=y, visibility=.95, presence=.95)
    ctx = measured_context(observation(width=.2, landmarks=points))
    assert ctx['head_shoulders_landmark_bounds'] == pytest.approx([.4,.25,.6,.4])
    assert ctx['core_landmark_bounds'] == pytest.approx([.4,.25,.6,.7])
    assert ctx['body_landmark_bounds'] == pytest.approx([.4,.25,.6,.7])
    assert ctx['lower_body_landmarks_complete'] is False
    assert not {'head_bounds','core_bounds','body_bounds','full_body_visible'}.intersection(ctx)
    assert ctx['depth_reliable'] is True
    assert ctx['coordinate_frame'] == 'full_mirrored_image'


def test_turned_body_does_not_make_shoulder_width_a_distance_measurement():
    from leica_tracker.portrait_quality import measured_context
    points = [SimpleNamespace(x=.5,y=.3,visibility=0.,presence=0.) for _ in range(33)]
    for i,x,y in ((0,.5,.25),(7,.49,.26),(8,.51,.26),(11,.48,.4),(12,.52,.4),
                  (23,.48,.8),(24,.52,.8)):
        points[i]=SimpleNamespace(x=x,y=y,visibility=.95,presence=.95)
    ctx=measured_context(observation(width=.04,landmarks=points))
    assert ctx['depth_reliable'] is False


def test_quality_scale_acceptance_is_relative_and_does_not_chase_current_width():
    from leica_tracker.portrait_quality import quality_tolerances
    p=plan(width=.2)
    assert quality_tolerances(p)[2] == pytest.approx(.02)


def test_production_position_band_preserves_width_based_policy_without_edge_clipping():
    from leica_tracker.portrait_quality import quality_tolerances
    p=plan(.5,.3,.3)
    assert quality_tolerances(p,positional_tolerance_scale=1.5) == pytest.approx(
        (.09,.16,.03))
    edge=plan(.98,.3,.04)
    assert quality_tolerances(edge,positional_tolerance_scale=1.5) == pytest.approx(
        (.03,.0375,.008))


def test_scaled_position_band_is_fully_reserved_by_projection_and_live_check():
    from leica_tracker.portrait_quality import fit_plan,quality_tolerances,projected_fits,MARGIN
    original=plan(.9,.3,.18)
    corrected=fit_plan(original,(.85,.3,.2),positional_tolerance_scale=1.5)
    xband,yband,scale_band=quality_tolerances(corrected,positional_tolerance_scale=1.5)
    x,y,width=corrected.target
    assert x-xband-(width+scale_band)/2 >= MARGIN-1e-9
    assert x+xband+(width+scale_band)/2 <= 1-MARGIN+1e-9
    assert y-yband >= MARGIN-1e-9 and y+yband <= 1-MARGIN+1e-9
    assert projected_fits(corrected,(.85,.3,width),positional_tolerance_scale=1.5)
    assert not projected_fits(corrected,(.85,.3,.2),positional_tolerance_scale=1.5)
    assert fit_plan(corrected,(.85,.3,.2),positional_tolerance_scale=1.5).target == pytest.approx(corrected.target)


def test_scaled_position_projection_reserves_measured_full_body_acceptance_corners():
    from leica_tracker.portrait_quality import fit_plan,quality_tolerances,MARGIN
    point=(.5,.3,.2)
    context=dict(image_aspect=16/9,depth_reliable=True,lower_body_landmarks_complete=True,
                 body_landmark_bounds=[.4,.25,.6,.9])
    corrected=fit_plan(plan(.9,.75,.12),point,context,positional_tolerance_scale=1.5)
    xband,yband,scale_band=quality_tolerances(corrected,positional_tolerance_scale=1.5)
    x,y,width=corrected.target;accepted=width+scale_band
    assert x-xband-.5*accepted >= MARGIN-1e-9
    assert x+xband+.5*accepted <= 1-MARGIN+1e-9
    assert y-yband-.25*accepted >= MARGIN-1e-9
    assert y+yband+3*accepted <= 1-MARGIN+1e-9


def test_local_projection_reason_survives_a_long_model_explanation():
    from dataclasses import replace
    from leica_tracker.portrait_quality import fit_plan
    p=replace(plan(.9,.3,.18),explanation='A'*240)
    corrected=fit_plan(p,(.85,.3,.2))
    assert len(corrected.explanation)<=240 and 'measured body margins' in corrected.explanation
