"""生产单计划输入选择加入，而无需改变机器人动力学。"""
import pytest


@pytest.mark.parametrize('single',[False,True])
def test_actual_single_plan_enables_portrait_quality_for_controller(single):
    from leica_tracker.app import AppConfig,guidance_controller_options
    config=AppConfig(guidance_profile='claude-guide',camera_source='ipad',single_plan=single)
    options=guidance_controller_options(config,physical_mode=True)
    assert options.get('portrait_quality',False) is single
    assert options['median_window']==5


@pytest.mark.parametrize('single',[False,True])
def test_actual_claude_client_receives_the_same_quality_opt_in(monkeypatch,single):
    from leica_tracker import app
    seen={}
    def factory(**options):seen.update(options);return object()
    monkeypatch.setattr(app,'ClaudeFramingClient',factory)
    config=app.AppConfig(guidance_profile='claude-guide',single_plan=single)
    app._framing_client(config)
    assert seen.get('portrait_quality',False) is single
    assert seen['model']=='claude-opus-5-5'


@pytest.mark.parametrize('single',[False,True])
def test_optional_gemini_entry_keeps_quality_prompt_and_controller_consistent(monkeypatch,single):
    from leica_tracker import app
    seen={}
    monkeypatch.setattr(app,'GeminiFramingClient',lambda **kw:seen.update(kw))
    app._framing_client(app.AppConfig(guidance_profile='gemini-guide',single_plan=single))
    assert seen.get('portrait_quality',False) is single
