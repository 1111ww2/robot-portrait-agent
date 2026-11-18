from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
from leica_tracker.gemini_guidance import GeminiGuidanceController
from leica_tracker.gemini_framing import CompositionPlan, FramingError

FRAME=np.zeros((72,128,3),np.uint8)
PLAN=CompositionPlan((.62,.30,.24),(.12,.04,.85,.94),.9,'environmental','Keep space beside the subject.')

class Worker:
    def __init__(self):self.calls=[]
    def submit(self,jpeg,point):
        f=Future();self.calls.append((jpeg,point,f));return f

def setup():
    w=Worker();c=GeminiGuidanceController(worker=w,single_plan=True,entry_ack_ms=1,stability_ms=0)
    return c,w

def tick(c,t,x=.5,y=.3,width=.2,frame=FRAME):
    return c.update(SimpleNamespace(center_x=x,head_y=y,shoulder_width=width,confidence=.9),t,frame=frame)

def test_single_snapshot_holds_all_three_target_components_while_person_moves():
    c,w=setup();tick(c,0);w.calls[0][2].set_result(PLAN)
    for t in range(100,1600,100):d=tick(c,t,y=.4,width=.3)
    assert d.target==PLAN.target and d.composition_frame==PLAN.frame
    assert len(w.calls)==1 and c._inner.natural_head_height is False

def test_single_snapshot_error_never_spends_another_request_without_explicit_restart():
    c,w=setup();tick(c,0);w.calls[0][2].set_exception(FramingError('AI_UNCERTAIN'))
    for t in range(100,40000,100):tick(c,t)
    assert len(w.calls)==1
    c.restart_encounter();tick(c,40100)
    assert len(w.calls)==2 and c._requests==2

def test_format_change_needs_explicit_replan_and_cannot_reuse_old_snapshot():
    c,w=setup();tick(c,0);w.calls[0][2].set_result(PLAN);tick(c,100)
    rotated=np.zeros((128,72,3),np.uint8)
    for t in range(200,20000,100):d=tick(c,t,frame=rotated)
    assert d.cue.value=='HOLD' and d.composition_frame is None
    assert len(w.calls)==1 and d.model_status=='AI_REPLAN_REQUIRED'

def test_single_snapshot_tracking_loss_and_return_do_not_request_again():
    c,w=setup();tick(c,0);w.calls[0][2].set_result(PLAN);tick(c,100)
    for t in range(200,1200,100):c.update(None,t,frame=FRAME)
    for t in range(1200,8000,100):d=tick(c,t)
    assert len(w.calls)==1 and d.target==PLAN.target


def test_cli_wires_single_snapshot_and_model_prompt_without_forced_route():
    from leica_tracker.app import build_parser,_config_from_args,_framing_client
    from leica_tracker.claude_framing import SINGLE_SNAPSHOT_PROMPT
    config=_config_from_args(build_parser().parse_args(['--guidance-profile','claude-guide','--single-plan']))
    assert config.single_plan and _framing_client(config).single_plan
    assert 'not a fixed direction' in SINGLE_SNAPSHOT_PROMPT
    assert 'prototype' in SINGLE_SNAPSHOT_PROMPT


def test_reference_snapshot_uses_first_valid_person_frame_without_stability_wait():
    w=Worker();c=GeminiGuidanceController(worker=w,single_plan=True,stability_ms=800)
    c.update(None,0,frame=FRAME);assert not w.calls
    tick(c,100);assert len(w.calls)==1
    for t in range(200,800,100):tick(c,t,x=.4+t/5000)
    assert len(w.calls)==1


def test_sim_fast_reversal_option_flows_through_single_snapshot_controller():
    from leica_tracker.continuous import ControllerConfig
    from leica_tracker.person_guidance import GuidanceCue
    w=Worker()
    c=GeminiGuidanceController(config=ControllerConfig(smoothing_alpha=1.),worker=w,single_plan=True,
        entry_ack_ms=1,response_window_ms=4200,fast_direction_reversal=True)
    tick(c,0,x=.8,y=.3,width=.24);w.calls[0][2].set_result(PLAN)
    assert tick(c,100,x=.8,y=.3,width=.24).cue is GuidanceCue.MOVE_LEFT
    assert tick(c,200,x=.4,y=.3,width=.24).cue is GuidanceCue.WAIT
    assert tick(c,300,x=.4,y=.3,width=.24).cue is GuidanceCue.MOVE_RIGHT
    assert len(w.calls)==1
