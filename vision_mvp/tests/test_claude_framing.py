import io
import json
import urllib.error

import pytest

from leica_tracker.gemini_framing import FramingError


def proposal():
    return dict(subject_visible=True, confidence=.92, style='environmental',
                frame=[.12,.04,.85,.94], target=[.62,.30,.24],
                explanation='Leave space beside the window.')


def response(value=None, stop='end_turn'):
    return dict(stop_reason=stop, content=[{'type':'thinking','thinking':'ignored'},
        {'type':'text','text':json.dumps(proposal() if value is None else value)}],
        usage={'input_tokens':100,'output_tokens':50})


def test_claude_uses_opus_medium_image_and_structured_output():
    from leica_tracker.claude_framing import ClaudeFramingClient
    calls=[]
    def transport(req,timeout):
        calls.append(req);return response()
    c=ClaudeFramingClient(key_provider=lambda:'test-secret',transport=transport)
    assert c.analyze(b'jpeg',(.48,.3,.24)).target==(.62,.3,.24)
    req=calls[0];body=json.loads(req.data)
    assert req.full_url=='https://api.anthropic.com/v1/messages'
    assert dict((k.lower(),v) for k,v in req.header_items())['x-api-key']=='test-secret'
    assert body['model']=='claude-opus-5-5'
    assert body['output_config']['effort']=='medium'
    assert body['output_config']['format']['type']=='json_schema'
    assert body['max_tokens']==4096
    assert body['messages'][0]['content'][0]['source']['data']=='anBlZw=='
    assert 'tool_choice' not in body and 'store' not in body
    assert 'minimum' not in json.dumps(body['output_config']['format'])
    assert c.last_usage=={'input_tokens':100,'output_tokens':50}


@pytest.mark.parametrize('stop',['max_tokens','refusal','tool_use','pause_turn'])
def test_incomplete_or_refused_reply_cannot_create_motion(stop):
    from leica_tracker.claude_framing import ClaudeFramingClient
    c=ClaudeFramingClient(key_provider=lambda:'test',transport=lambda *a:response(stop=stop))
    with pytest.raises(FramingError):c.analyze(b'jpeg',(.48,.3,.24))


@pytest.mark.parametrize('change',[
    {'target':[float('nan'),.3,.24]}, {'target':[.9,.3,.24]},
    {'frame':[.9,.1,.1,.9]}, {'confidence':.2}, {'subject_visible':False},
    {'joint_angles':[0,0,0,0,0,0]}, {'target':[.48,.3,.24]},
])
def test_claude_preserves_existing_geometry_and_encounter_contract(change):
    from leica_tracker.claude_framing import ClaudeFramingClient
    p=proposal();p.update(change)
    c=ClaudeFramingClient(key_provider=lambda:'test',transport=lambda *a:response(p))
    with pytest.raises(FramingError):c.analyze(b'jpeg',(.48,.3,.24))


@pytest.mark.parametrize('status,code',[(401,'AI_AUTH_ERROR'),(403,'AI_AUTH_ERROR'),
    (429,'AI_RATE_LIMIT'),(529,'AI_SERVICE_BUSY'),(404,'AI_MODEL_UNAVAILABLE')])
def test_provider_errors_are_redacted(status,code):
    from leica_tracker.claude_framing import ClaudeFramingClient
    def transport(req,timeout):
        raise urllib.error.HTTPError(req.full_url,status,'secret',{},io.BytesIO(b'test-secret'))
    c=ClaudeFramingClient(key_provider=lambda:'test-secret',transport=transport)
    with pytest.raises(FramingError) as caught:c.check_access()
    assert str(caught.value)==code


def test_key_selection_and_high_effort(tmp_path,monkeypatch):
    from leica_tracker.claude_framing import ClaudeFramingClient,load_api_key
    p=tmp_path/'.env.local';p.write_text('GEMINI_API_KEY=other\nANTHROPIC_API_KEY=local-test\n')
    monkeypatch.setenv('ANTHROPIC_API_KEY','process-test')
    assert load_api_key(p)=='local-test'
    assert ClaudeFramingClient(effort='high').effort=='high'
    with pytest.raises(ValueError):ClaudeFramingClient(effort='max')


def test_claude_cli_auth_failure_precedes_camera_open(monkeypatch,capsys):
    from leica_tracker import app
    class Denied:
        def __init__(self,**kwargs):
            assert kwargs['effort']=='high'
        def check_access(self):raise FramingError('AI_AUTH_ERROR')
    monkeypatch.setattr(app,'ClaudeFramingClient',Denied)
    monkeypatch.setattr(app.cv2,'VideoCapture',lambda *a:pytest.fail('opened camera'))
    assert app.main(['--guidance-profile','claude-guide','--claude-effort','high'])==4
    assert 'Claude' in capsys.readouterr().err


@pytest.mark.parametrize('x',[.48,.55,.62])
def test_single_snapshot_accepts_feasible_goal_without_old_minimum_shift_rule(x):
    from leica_tracker.claude_framing import ClaudeFramingClient
    p=proposal();p['target'][0]=x
    c=ClaudeFramingClient(single_plan=True,key_provider=lambda:'test',transport=lambda *a:response(p))
    assert c.analyze(b'jpeg',(.48,.3,.24)).target[0]==x


def test_single_snapshot_prompt_distinguishes_visibility_from_artistic_quality():
    from leica_tracker.claude_framing import ClaudeFramingClient
    bodies=[]
    def transport(req,timeout):
        bodies.append(json.loads(req.data));return response()
    client=ClaudeFramingClient(single_plan=True,key_provider=lambda:'test',transport=transport)
    client.analyze(b'jpeg',(.48,.3,.24))
    prompt=' '.join(bodies[0]['system'].lower().split())
    assert 'subject_visible indicates whether the person is visible' in prompt
    assert 'not an aesthetic judgment' in prompt
    assert 'confidence represents visibility and geometric feasibility' in prompt
    assert 'genuinely cannot see the person' in prompt
    assert len(bodies)==1


@pytest.mark.parametrize('change',[{'confidence':.64}, {'subject_visible':False}])
def test_single_snapshot_still_honors_model_uncertainty_without_retry(change):
    from leica_tracker.claude_framing import ClaudeFramingClient
    p=proposal();p.update(change);calls=[]
    def transport(*args):calls.append(1);return response(p)
    client=ClaudeFramingClient(single_plan=True,key_provider=lambda:'test',transport=transport)
    with pytest.raises(FramingError,match='AI_UNCERTAIN'):
        client.analyze(b'jpeg',(.48,.3,.24))
    assert len(calls)==1


@pytest.mark.parametrize('change',[{'target':[.85,.3,.24]}, {'confidence':.2}, {'target':[float('nan'),.3,.24]}])
def test_single_snapshot_still_rejects_invalid_or_unreliable_geometry(change):
    from leica_tracker.claude_framing import ClaudeFramingClient
    p=proposal();p.update(change)
    c=ClaudeFramingClient(single_plan=True,key_provider=lambda:'test',transport=lambda *a:response(p))
    with pytest.raises(FramingError):c.analyze(b'jpeg',(.48,.3,.24))
