from concurrent.futures import Future
import json

import pytest

from leica_tracker.claude_framing import ClaudeFramingClient
from leica_tracker.gemini_framing import FramingError
from leica_tracker.gemini_guidance import GeminiGuidanceController


def test_single_plan_network_budget_is_separate_from_live_modes():
    assert ClaudeFramingClient(single_plan=True).timeout == 30.
    assert ClaudeFramingClient().timeout == 12.
    assert ClaudeFramingClient(single_plan=True, timeout=7.).timeout == 7.


@pytest.mark.parametrize('single,elapsed,accepted', [(True,30000,True),(True,45001,False),(False,15001,False)])
def test_background_plan_reply_window_keeps_a_bound(single,elapsed,accepted):
    controller=GeminiGuidanceController(single_plan=single)
    future=Future();candidate=object();future.set_result(candidate)
    controller._pending=(future,controller._scene_generation,0,(.5,.3,.2))
    controller._receive_scene_result(elapsed)
    assert (controller._scene_template is candidate) == accepted
    if not accepted: assert controller._error == 'AI_STALE_RESULT'


def test_network_diagnostic_does_not_expose_exception_or_key(capsys):
    def fail(*args):raise TimeoutError('secret-key secret-image')
    client=ClaudeFramingClient(key_provider=lambda:'secret-key',transport=fail)
    with pytest.raises(FramingError,match='AI_NETWORK_ERROR'):client.check_access()
    output=capsys.readouterr().err
    assert 'secret' not in output
    diagnostic=json.loads(output.split('AI_REQUEST_DIAGNOSTIC ',1)[1])
    assert diagnostic['kind']=='TimeoutError' and diagnostic['elapsed_s']>=0
