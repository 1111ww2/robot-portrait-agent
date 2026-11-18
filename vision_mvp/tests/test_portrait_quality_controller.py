"""带有确定性模型回复的实际语义控制器行为。"""
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
import pytest

from leica_tracker.gemini_framing import validate_plan
from leica_tracker.gemini_guidance import GeminiGuidanceController
from leica_tracker.person_guidance import PersonGuidanceController, GuidanceCue
from leica_tracker.continuous import ControllerConfig


class Worker:
    def __init__(self, target):
        self.target=target;self.calls=[]
    def submit(self,jpeg,point):
        self.calls.append(point);f=Future()
        f.set_result(validate_plan(dict(subject_visible=True,confidence=.95,style='environmental',
            frame=[0.,0.,1.,1.],target=list(self.target),explanation='Use the visible background.')))
        return f


def drive(point,target):
    w=Worker(target)
    c=GeminiGuidanceController(worker=w,single_plan=True,portrait_quality=True,
                              entry_ack_ms=1,arrival_dwell_ms=300)
    obs=SimpleNamespace(center_x=point[0],head_y=point[1],shoulder_width=point[2],confidence=.95)
    result=[c.update(obs,t,frame=np.zeros((72,128,3),dtype=np.uint8)) for t in range(0,1500,100)]
    return c,w,result


@pytest.mark.parametrize('point,target,wanted', [
    ((.5,.3,.35),(.5,.3,.2),'MOVE_FARTHER'),
    ((.5,.3,.10),(.5,.3,.2),'MOVE_CLOSER'),
])
def test_quality_mode_guides_real_near_far_error(point,target,wanted):
    c,w,result=drive(point,target)
    assert any(d.cue.value==wanted for d in result)
    assert c._plan.target[2] == pytest.approx(target[2])
    assert len(w.calls)==1


def test_quality_edge_case_never_tells_person_further_out():
    _,_,result=drive((.85,.3,.20),(.90,.3,.18))
    assert not any(d.cue is GuidanceCue.MOVE_RIGHT for d in result)


def test_quality_already_good_can_arrive_without_a_direction():
    _,w,result=drive((.5,.3,.2),(.5,.3,.2))
    assert any(d.phase=='ARRIVED' for d in result)
    assert not any(d.cue.value.startswith('MOVE_') for d in result)
    assert len(w.calls)==1


def test_depth_first_is_opt_in_and_preserves_direction_signs():
    p=SimpleNamespace(center_x=.1,head_y=.3,shoulder_width=.4,confidence=.95)
    cfg=ControllerConfig(target_x=.6,target_head_y=.3,target_shoulder_width=.2)
    default=PersonGuidanceController(cfg)
    depth=PersonGuidanceController(cfg,depth_first=True)
    default.update(p,0);depth.update(p,0)
    assert default.update(p,1200).cue is GuidanceCue.MOVE_RIGHT
    assert depth.update(p,1200).cue is GuidanceCue.MOVE_FARTHER


def test_quality_left_edge_case_never_tells_person_further_out():
    _,_,result=drive((.15,.3,.20),(.10,.3,.18))
    assert not any(d.cue is GuidanceCue.MOVE_LEFT for d in result)


def test_reference_resume_requires_new_stable_geometry_without_another_ai_call():
    c,w,_=drive((.5,.3,.35),(.5,.3,.2))
    c.suspend_perception(1500)
    obs=SimpleNamespace(center_x=.5,head_y=.3,shoulder_width=.35,confidence=.95)
    decisions=[c.update(obs,t,frame=np.zeros((72,128,3),dtype=np.uint8)) for t in (1600,1700,1800)]
    assert all(not d.cue.value.startswith('MOVE_') for d in decisions[:2])
    assert decisions[-1].cue is GuidanceCue.MOVE_FARTHER
    assert len(w.calls)==1


def test_brief_detection_loss_requires_three_new_samples_without_replanning():
    c,w,_=drive((.5,.3,.35),(.5,.3,.2))
    lost=c.update(None,1500,frame=np.zeros((72,128,3),dtype=np.uint8))
    assert lost.cue is GuidanceCue.HOLD and lost.phase=='TRACKING_LOST'
    obs=SimpleNamespace(center_x=.5,head_y=.3,shoulder_width=.35,confidence=.95)
    resumed=[c.update(obs,t,frame=np.zeros((72,128,3),dtype=np.uint8))
             for t in (1600,1700,1800)]
    assert all(d.cue is GuidanceCue.HOLD and d.reason=='LOCAL_DEPTH_STABILITY'
               for d in resumed[:2])
    assert resumed[-1].cue is GuidanceCue.MOVE_FARTHER
    assert len(w.calls)==1


def test_closed_quality_controller_never_emits_a_direction():
    c,w,_=drive((.5,.3,.35),(.5,.3,.2));c.close()
    obs=SimpleNamespace(center_x=.5,head_y=.3,shoulder_width=.35,confidence=.95)
    d=c.update(obs,1600,frame=np.zeros((72,128,3),dtype=np.uint8))
    assert d.cue is GuidanceCue.HOLD and d.phase=='AI_STOPPED' and len(w.calls)==1


def test_body_turn_cannot_be_misinterpreted_as_being_too_far():
    w=Worker((.5,.3,.2));c=GeminiGuidanceController(worker=w,single_plan=True,
        portrait_quality=True,entry_ack_ms=1)
    points=[SimpleNamespace(x=.5,y=.3,visibility=0.,presence=0.) for _ in range(33)]
    for i,x,y in ((0,.5,.25),(7,.49,.26),(8,.51,.26),(11,.45,.3),(12,.55,.3),
                  (23,.48,.9),(24,.52,.9)):
        points[i]=SimpleNamespace(x=x,y=y,visibility=.95,presence=.95)
    obs=SimpleNamespace(center_x=.5,head_y=.3,shoulder_width=.1,confidence=.95,landmarks=points)
    result=[c.update(obs,t,frame=np.zeros((72,128,3),dtype=np.uint8)) for t in range(0,2000,100)]
    assert not any(d.cue.value.startswith('MOVE_') for d in result)
    assert any(d.reason=='LOCAL_DEPTH_STABILITY' for d in result)
    assert len(w.calls)==1
    assert c._plan.target[2]==pytest.approx(.2)


@pytest.mark.parametrize('x,target,cue',[(.25,.65,'MOVE_RIGHT'),(.75,.35,'MOVE_LEFT')])
def test_quality_lateral_mapping_still_points_toward_the_fitted_goal(x,target,cue):
    _,_,result=drive((x,.3,.2),(target,.3,.2))
    assert any(d.cue.value==cue for d in result)
