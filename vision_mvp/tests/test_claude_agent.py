from concurrent.futures import Future
from dataclasses import asdict
from types import SimpleNamespace
import json
import math

import numpy as np
import pytest


def turn(**changes):
    data=dict(intent='guide',speech='A little to your right, keeping the background.',cue='MOVE_RIGHT',
              amplitude=.4,repetitions=1,cycle_ms=3000,pause_ms=3000,
              wait_for_reply=False,subject_visible=True,confidence=.9,
              frame=[.1,.1,.9,.95],style='half_body')
    data.update(changes);return data


def pose(x=.5):return SimpleNamespace(center_x=x,head_y=.3,shoulder_width=.2,confidence=.9)


class Worker:
    def __init__(self):self.calls=[]
    def submit(self,jpeg,context):
        f=Future();self.calls.append((context,f));return f
    def resolve(self,**changes):
        from leica_tracker.agent_actions import validate_turn
        self.calls[-1][1].set_result(validate_turn(turn(**changes)))


def setup():
    from leica_tracker.claude_agent import ClaudeInteractionController
    w=Worker();return ClaudeInteractionController(worker=w),w


def tick(c,t,p=True,observation=True):
    return c.update(pose() if observation else None,t,
        frame=np.zeros((72,128,3),np.uint8),present=p)


def test_model_parameters_change_canonical_motion_and_screen_phase():
    from leica_tracker.agent_actions import validate_turn
    from leica_tracker.gesture_preview import normalized_offset_for
    from leica_tracker.gesture_feedback import edge_feedback_for
    p=validate_turn(turn(repetitions=3,cycle_ms=2000)).gesture
    assert p.duration_ms==6000 and p.repetitions==3
    for t in (0,2000,4000,6000):assert normalized_offset_for(p,t)==pytest.approx((0,0,0),abs=1e-10)
    for t in (1000,3000,5000):
        assert normalized_offset_for(p,t)==pytest.approx((.4,0,0))
        assert edge_feedback_for(p,t)['edge']=='right'


@pytest.mark.parametrize('change',[
    {'amplitude':float('nan')},{'amplitude':True},{'amplitude':1.},
    {'repetitions':4},{'repetitions':True},{'cycle_ms':100},
    {'pause_ms':0},{'frame':[.9,0,.2,1]},{'joint_angles':[1,2]},
    {'cue':'RUN_CODE'},{'speech':'x'*241},{'confidence':float('inf')},
])
def test_untrusted_turn_rejected(change):
    from leica_tracker.agent_actions import validate_turn
    from leica_tracker.gemini_framing import FramingError
    with pytest.raises(FramingError):validate_turn(turn(**change))


def test_stream_accepts_bounded_agent_plan_but_rejects_forged_speed_or_pattern():
    from leica_tracker.agent_actions import validate_turn
    from leica_tracker.simulator_guidance import GuidanceStream
    p=validate_turn(turn()).gesture
    e=dict(event='person_guidance',cue=p.cue,tracking=True,coordinate_frame='normalized_camera_image',
           robot_actuated=False,source_monotonic_ns=10_000_000_000,gesture=asdict(p))
    stream=GuidanceStream();stream.accept(json.dumps(e),10.)
    assert stream.plan==p
    for field,value in [('duration_ms',100),('pattern','arbitrary')]:
        bad={**e,'gesture':{**e['gesture'],field:value}}
        with pytest.raises(ValueError):GuidanceStream().accept(json.dumps(bad),10.)
    stream.accept(json.dumps({**e,'cue':'WAIT','gesture':None,'source_monotonic_ns':10_100_000_000}),10.1)
    assert stream.started==10.
    assert stream.offset(10.5) is None


def test_close_up_without_shoulders_can_converse_and_move_with_face_presence():
    c,w=setup();assert tick(c,0,observation=False).cue.value=='HOLD'
    assert len(w.calls)==1 and w.calls[0][0]['pose'] is None
    w.resolve(cue='INVITE',intent='invite')
    d=tick(c,100,observation=False)
    assert d.cue.value=='INVITE' and c.dialogue['speech']
    assert c.gesture is not None and d.tracking
    assert tick(c,200,observation=False).cue.value=='WAIT'


def test_uncertain_scene_can_ask_but_cannot_move():
    c,w=setup();tick(c,0,observation=False)
    w.resolve(intent='ask',cue='INVITE',subject_visible=False,confidence=.2)
    d=tick(c,100,observation=False)
    assert c.dialogue['speech'] and d.cue.value=='HOLD' and c.gesture is None


def test_new_message_cancels_motion_and_discards_pending_response():
    c,w=setup();tick(c,0)
    c.say('Do not ask me to move',100)
    w.resolve();assert tick(c,200).cue.value=='HOLD'
    for t in range(300,5100,100):tick(c,t)
    assert len(w.calls)==2
    assert w.calls[-1][0]['history'][-1]['text']=='Do not ask me to move'
    w.resolve(intent='ask',cue='WAIT',wait_for_reply=True)
    tick(c,5100)
    for t in range(5200,30000,100):assert tick(c,t).cue.value=='HOLD'
    assert len(w.calls)==2


@pytest.mark.parametrize('interrupt',['stop','reset','loss','gap'])
def test_interrupt_never_replays_previous_movement(interrupt):
    c,w=setup();tick(c,0);w.resolve();assert tick(c,100).cue.value=='MOVE_RIGHT'
    if interrupt=='stop':c.stop();d=tick(c,200)
    elif interrupt=='reset':c.reset();d=tick(c,200)
    elif interrupt=='loss':d=tick(c,200,p=False,observation=False)
    else:d=tick(c,1000)
    assert d.cue.value=='HOLD' and c.gesture is None


def test_finished_gesture_observes_again_without_replaying_and_stays_bounded():
    c,w=setup();tick(c,0);w.resolve();tick(c,100)
    for t in range(200,6200,100):tick(c,t)
    assert len(w.calls)==2 and w.calls[-1][0]['previous_outcome']=='completed'
    w.resolve(intent='wait',cue='WAIT',wait_for_reply=True)
    tick(c,6200)
    for t in range(6300,20000,100):assert tick(c,t).cue.value=='HOLD'
    assert len(w.calls)==2


def test_delayed_result_is_discarded_and_stopping_invalidates_speech():
    c,w=setup();tick(c,0)
    for t in range(100,26000,100):tick(c,t)
    w.resolve();d=tick(c,26000)
    assert d.cue.value=='HOLD' and not c.dialogue.get('speech')
    c.stop();assert not c.dialogue.get('speech')


def test_claude_agent_payload_has_history_and_no_forced_lateral_target():
    from leica_tracker.claude_agent import ClaudeInteractionClient
    calls=[]
    def transport(req,timeout):
        calls.append(json.loads(req.data))
        return dict(stop_reason='end_turn',content=[{'type':'text','text':json.dumps(turn(cue='WAIT',intent='ask'))}])
    c=ClaudeInteractionClient(key_provider=lambda:'test',transport=transport)
    result=c.decide(b'jpeg',{'history':[{'role':'user','text':'Half-length portrait'}],'pose':None})
    assert result.cue=='WAIT'
    assert calls[0]['model']=='claude-opus-5-5'
    assert calls[0]['output_config']['effort']=='medium'
    assert 'Half-length portrait' in calls[0]['messages'][0]['content'][1]['text']
    assert '.12 to .18' not in calls[0]['system']


def test_model_goal_is_explicit_image_geometry_and_never_implied_from_current_pose():
    from leica_tracker.agent_actions import validate_turn
    assert validate_turn(turn()).target is None
    result=validate_turn(turn(target=[.62,.3,.2]))
    assert result.target==(.62,.3,.2)
    c,w=setup();tick(c,0);w.resolve(target=[.62,.3,.2]);d=tick(c,100)
    assert d.model_target==(.62,.3,.2) and d.target==(.62,.3,.2)
    c.stop();assert tick(c,200).model_target is None


@pytest.mark.parametrize('target',[[1.1,.3,.2],[.5,.3,0],[.5,.3],[.5,float('nan'),.2]])
def test_invalid_model_goals_are_rejected(target):
    from leica_tracker.agent_actions import validate_turn
    from leica_tracker.gemini_framing import FramingError
    with pytest.raises(FramingError):validate_turn(turn(target=target))


def test_presence_reentry_invalidates_a_decision_from_before_departure():
    c,w=setup();tick(c,0)
    tick(c,100,p=False,observation=False)
    tick(c,200)
    w.resolve();d=tick(c,300)
    assert d.cue.value=='HOLD' and c.gesture is None
    assert not c.dialogue['speech']


def test_completed_interaction_is_latched_across_presence_flicker():
    c,w=setup();tick(c,0);w.resolve(intent='finish',cue='WAIT',pause_ms=1000)
    for t in range(100,1300,100):tick(c,t)
    assert c.dialogue['finished']
    tick(c,1300,p=False,observation=False)
    for t in range(1400,10000,100):assert tick(c,t).cue.value=='HOLD'
    assert len(w.calls)==1 and c.dialogue['finished']
    c.say('I would like another composition',10000);tick(c,10100)
    assert len(w.calls)==2 and not c.dialogue['finished']


def test_agent_rejects_unmirrored_preview_before_camera_open():
    from leica_tracker.app import _config_from_args,build_parser
    with pytest.raises(ValueError,match='mirrored'):
        _config_from_args(build_parser().parse_args(['--guidance-profile','claude-agent','--no-mirror']))


def test_temporal_request_preserves_order_and_limits_frame_count():
    from leica_tracker.claude_agent import ClaudeInteractionClient
    calls=[]
    def transport(req,timeout):
        calls.append(json.loads(req.data))
        return dict(stop_reason='end_turn',content=[{'type':'text','text':json.dumps(turn(cue='WAIT'))}])
    client=ClaudeInteractionClient(key_provider=lambda:'test',transport=transport)
    client.decide((b'earlier',b'latest'),{'frame_offsets_ms':[-400,0]})
    import base64
    images=[part for part in calls[0]['messages'][0]['content'] if part['type']=='image']
    assert [base64.b64decode(i['source']['data']) for i in images]==[b'earlier',b'latest']
    from leica_tracker.gemini_framing import FramingError
    with pytest.raises(FramingError):client.decide((b'x',)*5,{})


def test_passive_wait_requires_visible_change_and_respects_cap():
    c,w=setup();tick(c,0);w.resolve(intent='ask',cue='WAIT',wait_for_reply=True)
    for t in range(100,8800,100):tick(c,t)
    assert len(w.calls)==1  # 静止的访客不得在定时器上进行通话。
    changed=np.zeros((72,128,3),np.uint8);changed[10:60,20:50]=255
    c.update(pose(),8800,frame=changed,present=True)
    assert len(w.calls)==2
    ctx=w.calls[-1][0]
    assert ctx['previous_outcome']=='observe_nonverbal'
    assert ctx['frame_offsets_ms']==[-1200,-800,-400,0]
    c._max_requests=2;w.resolve(intent='wait',cue='WAIT',wait_for_reply=True)
    for t in range(8900,20000,100):tick(c,t)
    assert len(w.calls)==2


def test_absence_does_not_spend_calls_or_retain_temporal_frames():
    c,w=setup()
    for t in range(0,2000,100):tick(c,t,p=False,observation=False)
    assert not w.calls
    tick(c,2000);w.resolve(cue='WAIT',intent='ask',wait_for_reply=True);tick(c,2100)
    for t in range(2200,2600,100):tick(c,t)
    tick(c,2600,p=False,observation=False)
    for t in range(2700,7300,100):tick(c,t,p=False,observation=False)
    assert len(w.calls)==1
    tick(c,7300)
    assert w.calls[-1][0]['frame_offsets_ms']==[0]


def test_finished_encounter_rearms_only_after_sustained_departure():
    c,w=setup();tick(c,0);w.resolve(intent='finish',cue='WAIT',pause_ms=1000)
    for t in range(100,1400,100):tick(c,t)
    assert c.dialogue['finished']
    for t in range(1400,5100,100):tick(c,t,p=False,observation=False)
    assert len(w.calls)==1 and not c.dialogue['present']
    tick(c,5100)
    assert len(w.calls)==2 and not c.dialogue['finished']
    assert w.calls[-1][0]['history']==[]


@pytest.mark.parametrize('cue,edge',[
    ('MOVE_LEFT','left'),('MOVE_RIGHT','right'),('MOVE_UP','top'),('MOVE_DOWN','bottom'),
    ('MOVE_CLOSER','all'),('MOVE_FARTHER','all'),('INVITE','all'),('ACKNOWLEDGE','all'),('WAIT',None)])
def test_every_model_action_reaches_shared_motion_and_edge_timeline(cue,edge):
    from leica_tracker.agent_actions import validate_turn
    from leica_tracker.simulator_guidance import GuidanceStream
    from leica_tracker.gesture_feedback import edge_feedback_for
    p=validate_turn(turn(cue=cue)).gesture
    if p is None:
        assert edge_feedback_for(p,1500)['edge'] is None
        return
    stream=GuidanceStream()
    for i in range(16):
        now=10+i*.1
        stream.accept(json.dumps(dict(event='person_guidance',cue=cue if i==0 else 'WAIT',
            tracking=True,coordinate_frame='normalized_camera_image',robot_actuated=False,
            source_monotonic_ns=round(now*1e9),gesture=asdict(p) if i==0 else None)),now)
    assert stream.offset(11.5) is not None
    signal=edge_feedback_for(p,1500)
    assert signal['edge']==edge and signal['intensity']>0
    stream.accept(json.dumps(dict(event='person_guidance',cue='HOLD',tracking=False,
        coordinate_frame='normalized_camera_image',robot_actuated=False,source_monotonic_ns=11_600_000_000)),11.6)
    assert stream.offset(11.6) is None


def test_uniform_exposure_shift_does_not_trigger_passive_model_calls():
    c,w=setup();tick(c,0);w.resolve(intent='ask',cue='WAIT',wait_for_reply=True)
    for t in range(100,15000,100):
        c.update(pose(),t,frame=np.full((72,128,3),30 if t>4000 else 0,np.uint8),present=True)
    assert len(w.calls)==1


def test_explicit_words_get_a_reply_without_visual_presence_and_cannot_move():
    c,w=setup();c.say('Can you hear me?',0)
    tick(c,0,p=False,observation=False)
    assert len(w.calls)==1
    assert w.calls[0][0]['history'][-1]['text']=='Can you hear me?'
    w.resolve(speech='Yes, I can hear you.',cue='MOVE_RIGHT',subject_visible=True)
    d=tick(c,100,p=False,observation=False)
    assert c.dialogue['speech']=='Yes, I can hear you.'
    assert d.cue.value=='HOLD' and c.gesture is None


def test_visual_loss_does_not_discard_an_explicit_voice_reply():
    c,w=setup();c.say('Can you hear me?',0);tick(c,0)
    tick(c,100,p=False,observation=False)
    w.resolve(speech='Yes, I can hear you.',cue='WAIT',wait_for_reply=True)
    tick(c,200,p=False,observation=False)
    assert c.dialogue['speech']=='Yes, I can hear you.'


def test_visual_requests_reserve_five_slots_for_explicit_words():
    c,w=setup();c._requests=15
    tick(c,0);assert len(w.calls)==0
    c.say('Can you hear me?',100);tick(c,100)
    assert len(w.calls)==1 and c.dialogue['requests']==16
