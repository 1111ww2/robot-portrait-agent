"""正式流程和原生验证采用同一套构图控制策略。"""
from types import SimpleNamespace

import pytest

from leica_tracker import app


def test_native_policy_preserves_actual_application_defaults():
    config=app.AppConfig(camera_source='ipad',guidance_profile='claude-guide',single_plan=True)
    options=app.guidance_controller_options(config,screen_coupled=True,physical_mode=True)
    assert options==dict(min_confidence=.5,response_window_ms=6700,median_window=5,
        single_plan=True,entry_ack_ms=3000,interaction_mode='claude-guide',
        arrival_dwell_ms=2000,fixed_plan_tolerance_scale=1.5,min_subject_width=.04,
        portrait_quality=True)


@pytest.mark.parametrize('coupled,native,fast,response',[(False,False,False,3000),
    (True,False,False,6700),(True,True,False,6700),(True,False,True,3900)])
def test_policy_keeps_existing_response_windows(coupled,native,fast,response):
    config=app.AppConfig(camera_source='ipad',guidance_profile='claude-guide',single_plan=True)
    options=app.guidance_controller_options(config,screen_coupled=coupled,physical_mode=native,sim_fast=fast)
    assert options['response_window_ms']==response
    assert options['entry_ack_ms']==(3000 if coupled else 800)
    assert ('min_subject_width' in options)==(native or fast)


def test_actual_frame_loop_consumes_shared_policy_and_provided_real_client(monkeypatch):
    observed={};client=object()
    class Captured(Exception):pass
    def capture(**kwargs):observed.update(kwargs);raise Captured
    monkeypatch.setattr(app,'GeminiGuidanceController',capture)
    monkeypatch.setattr(app,'_framing_client',lambda _:pytest.fail('provided client was ignored'))
    monkeypatch.setattr(app,'guidance_controller_options',lambda *a,**k:dict(
        min_confidence=.6,response_window_ms=5432,median_window=3,single_plan=True,
        entry_ack_ms=4321,interaction_mode='claude-guide'))
    config=app.AppConfig(camera_source='ipad',guidance_profile='claude-guide',single_plan=True)
    with pytest.raises(Captured):
        app.process_frames(None,None,config,guest_display=SimpleNamespace(physical_feedback_enabled=True),
                           framing_client=client)
    assert observed['client'] is client
    assert observed['response_window_ms']==5432 and observed['entry_ack_ms']==4321


def test_missing_optional_client_still_uses_normal_provider_factory(monkeypatch):
    provider=object();observed={}
    class Captured(Exception):pass
    def capture(**kwargs):observed.update(kwargs);raise Captured
    monkeypatch.setattr(app,'GeminiGuidanceController',capture)
    monkeypatch.setattr(app,'_framing_client',lambda _:provider)
    config=app.AppConfig(camera_source='ipad',guidance_profile='claude-guide',single_plan=True)
    with pytest.raises(Captured):
        app.process_frames(None,None,config,guest_display=SimpleNamespace(physical_feedback_enabled=True))
    assert observed['client'] is provider


def test_post_motion_resume_releases_response_wait_with_production_policy():
    from concurrent.futures import Future
    from leica_tracker.gemini_guidance import GeminiGuidanceController
    from leica_tracker.gemini_framing import CompositionPlan
    import numpy as np
    class Provider:
        def submit(self,*_):
            future=Future();future.set_result(CompositionPlan((.5,.3,.2),(.1,.05,.9,.95),.99,
                'environmental','policy fixture'));return future
    config=app.AppConfig(camera_source='ipad',guidance_profile='claude-guide',single_plan=True)
    options=app.guidance_controller_options(config,screen_coupled=True,physical_mode=True)
    controller=GeminiGuidanceController(worker=Provider(),**options)
    observation=SimpleNamespace(center_x=.8,head_y=.3,shoulder_width=.2,confidence=.99)
    frame=np.zeros((90,160,3),dtype=np.uint8);first=None
    for stamp in range(0,8000,50):
        decision=controller.update(observation,stamp,mirrored=True,frame=frame)
        if decision.cue.value=='MOVE_LEFT':first=stamp;break
    assert first is not None
    controller.suspend_perception(first+100)
    decisions=[controller.update(observation,stamp,mirrored=True,frame=frame)
               for stamp in (first+200,first+250,first+300)]
    assert all(d.cue.value=='HOLD' and d.reason=='LOCAL_DEPTH_STABILITY'
               for d in decisions[:2])
    assert decisions[-1].cue.value=='MOVE_LEFT'
