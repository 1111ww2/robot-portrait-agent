"""诊断程序只包含受限制的模型基本单元，从不包含图像或回复文本。"""
import json
from concurrent.futures import Future

import pytest

from leica_tracker.claude_framing import ClaudeFramingClient
from leica_tracker.gemini_framing import FramingError
from leica_tracker.gemini_guidance import GeminiGuidanceController


def uncertain_error(visible,confidence):
    value=dict(subject_visible=visible,confidence=confidence,style='environmental',
               frame=[.1,.1,.9,.9],target=[.5,.3,.2],explanation='private model words')
    data={'stop_reason':'end_turn','content':[{'type':'text','text':json.dumps(value)}]}
    client=ClaudeFramingClient(single_plan=True,key_provider=lambda:'private-api-key',
                              transport=lambda *_:data)
    with pytest.raises(FramingError) as caught:
        client.analyze(b'private-image-content',(.48,.3,.2))
    return caught.value


@pytest.mark.parametrize(('visible','confidence','reason'),[
    (False,.88,'subject_not_visible'),(True,.41,'low_confidence'),
    ('private-invalid',float('nan'),'invalid_visibility_or_confidence'),
])
def test_claude_uncertainty_logs_only_bounded_primitives(visible,confidence,reason,capsys):
    error=uncertain_error(visible,confidence)
    assert str(error)=='AI_UNCERTAIN'
    future=Future();future.set_exception(error)
    controller=GeminiGuidanceController(interaction_mode='claude-guide')
    controller._pending=(future,controller._scene_generation,100,(.48,.3,.2))
    controller._receive_scene_result(200)
    output=capsys.readouterr().err
    assert 'private' not in output
    prefix='AI_UNCERTAIN_DIAGNOSTIC '
    assert output.startswith(prefix)
    payload=json.loads(output[len(prefix):])
    assert set(payload)=={'reason','subject_visible','confidence'}
    assert payload['reason']==reason
    assert payload['subject_visible']==(visible if type(visible) is bool else None)
    assert payload['confidence']==(confidence if type(confidence) in (int,float) and 0<=confidence<=1 else None)
    assert controller._error=='AI_UNCERTAIN'


def test_discarded_request_and_other_modes_do_not_emit_diagnostics(capsys):
    for mode,stale in [('claude-guide',True),('gemini-guide',False)]:
        future=Future();future.set_exception(uncertain_error(False,.1))
        controller=GeminiGuidanceController(interaction_mode=mode)
        generation=controller._scene_generation-(1 if stale else 0)
        controller._pending=(future,generation,100,(.48,.3,.2))
        controller._receive_scene_result(200)
    assert capsys.readouterr().err==''
