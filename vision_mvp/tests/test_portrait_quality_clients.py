"""质量规划使用现有的提供商，无需发出任何出站请求。"""
import json
from types import SimpleNamespace
import pytest

from leica_tracker.claude_framing import ClaudeFramingClient
from leica_tracker.gemini_framing import FramingError
from leica_tracker.gemini_guidance import VisionWorker


def result(target=(.5,.3,.2)):
    p=dict(subject_visible=True,confidence=.9,style='environmental',frame=[0,0,1,1],
           target=list(target),explanation='Balance the visible background.')
    return dict(stop_reason='end_turn',content=[dict(type='text',text=json.dumps(p))])


def test_quality_prompt_chooses_composition_instead_of_demonstration():
    calls=[]
    def transport(req,timeout):calls.append(json.loads(req.data));return result()
    c=ClaudeFramingClient(single_plan=True,portrait_quality=True,
        key_provider=lambda:'test',transport=transport)
    assert c.analyze(b'jpeg',(.5,.3,.2)).target == (.5,.3,.2)
    prompt=calls[0]['system']
    assert 'current position may already be good' in prompt
    assert 'metric distance' not in prompt or 'Do not' in prompt
    assert 'quick interactive prototype' not in prompt
    assert '.12 to .18' not in prompt
    assert '.9 to 1.1' not in prompt
    assert 'sparse visible landmark envelopes' in prompt
    assert 'not the visible silhouette' in prompt
    assert 'top of the head' in prompt
    assert calls[0]['model']=='claude-opus-5-5'


def test_quality_allows_feasible_semantic_goal_without_old_demo_offset_bounds():
    options=dict(single_plan=True,key_provider=lambda:'test',transport=lambda *a:result((.7,.3,.2)))
    quality=ClaudeFramingClient(portrait_quality=True,**options)
    assert quality.analyze(b'jpeg',(.3,.3,.1)).target == (.7,.3,.2)
    with pytest.raises(FramingError,match='AI_INFEASIBLE_TARGET'):
        ClaudeFramingClient(**options).analyze(b'jpeg',(.3,.3,.1))


def test_quality_measured_context_uses_full_image_without_new_provider_fields():
    calls=[]
    context=dict(coordinate_frame='full_mirrored_image',current_pose=[.5,.3,.2],
                 depth_reliable=True,core_landmark_bounds=[.4,.2,.6,.7])
    def transport(req,timeout):calls.append(json.loads(req.data));return result()
    c=ClaudeFramingClient(single_plan=True,portrait_quality=True,
        key_provider=lambda:'test',transport=transport)
    c.analyze(b'jpeg',(.5,.3,.2),context=context)
    text=json.loads(calls[0]['messages'][0]['content'][1]['text'])
    assert text['measured_context']==context
    assert text['current_pose']==[.5,.3,.2]


def test_worker_preserves_two_argument_clients_when_context_not_supported():
    calls=[]
    class Client:
        def analyze(self,jpeg,point):calls.append((jpeg,point));return 'plan'
    future=VisionWorker(Client()).submit(b'jpeg',(.5,.3,.2),context={'safe':True})
    assert future.result(timeout=1)=='plan'
    assert calls==[(b'jpeg',(.5,.3,.2))]


def test_worker_delivers_context_only_to_explicitly_capable_client():
    calls=[]
    class Client:
        supports_portrait_context=True
        def analyze(self,jpeg,point,context=None):calls.append(context);return 'plan'
    future=VisionWorker(Client()).submit(b'jpeg',(.5,.3,.2),context={'safe':True})
    assert future.result(timeout=1)=='plan' and calls==[{'safe':True}]
