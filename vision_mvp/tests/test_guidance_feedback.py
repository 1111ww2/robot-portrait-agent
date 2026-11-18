from types import SimpleNamespace
import pytest
from leica_tracker import guidance_feedback as feedback
from leica_tracker.person_guidance import GuidanceCue


def decision(point=(.5,.25,.3),filtered=None,phase='WAITING_FOR_RESPONSE',tracking=True):
    return SimpleNamespace(target=(.5,.25,.3),current_point=point,
        filtered_point=filtered if filtered is not None else point,
        target_tolerances=(.02,.02,.03),error_spans=(.25,.2,.15),
        phase=phase,tracking=tracking,reason=phase,cue=GuidanceCue.WAIT)


def test_height_can_be_wrong_even_when_center_and_size_are_correct():
    checks=feedback.position_checks(decision((.5,.1,.3)))
    assert [r['state'] for r in checks] == ['OK','ADJUST','OK']
    assert checks[1]['action'] == 'MOVE DOWN'


def test_raw_jitter_shows_checking_instead_of_premature_success():
    checks=feedback.position_checks(decision((.5,.3,.3),(.5,.25,.3)))
    assert checks[1]['state'] == 'CHECKING'
    assert checks[1]['action'] == 'HOLD STILL'


@pytest.mark.parametrize('mirrored,action',[(True,'MOVE LEFT'),(False,'MOVE RIGHT')])
def test_screen_direction_respects_preview_mirror(mirrored,action):
    checks=feedback.position_checks(decision((.7,.25,.3)),mirrored=mirrored)
    assert checks[0]['action'] == action


def test_custom_tolerance_matches_the_actual_controller_configuration():
    d=decision((.53,.25,.3));d.target_tolerances=(.04,.02,.03)
    assert feedback.position_checks(d)[0]['state'] == 'OK'


def test_waiting_message_explicitly_says_not_in_position_and_gives_needed_action():
    title,detail,_=feedback.feedback_text(decision((.5,.1,.3),phase='PAUSED_FOR_PARTICIPANT'))
    assert title == 'NOT IN POSITION'
    assert 'MOVE DOWN' in detail


def test_lost_tracking_does_not_offer_positioning_instructions():
    d=decision(None,tracking=False,phase='TRACKING_LOST')
    assert all(r['state']=='UNAVAILABLE' for r in feedback.position_checks(d))


def test_local_request_cap_is_distinguished_from_provider_rate_limit():
    d=decision(phase='AI_BUDGET_REACHED');d.interaction_mode='gemini-guide'
    local_title,local_detail,_=feedback.feedback_text(d)
    d.phase='AI_RATE_LIMIT'
    provider_title,provider_detail,_=feedback.feedback_text(d)
    assert 'LOCAL' in local_title and 'app' in local_detail
    assert 'GEMINI' in provider_title and 'Google' in provider_detail
