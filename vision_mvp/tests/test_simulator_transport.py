"""反馈中断可以停止运动，而不会中断 iPad 摄像头进程。"""
import http.client
import itertools
import json
from dataclasses import asdict

import pytest

from leica_tracker import simulator_guidance as sim
from leica_tracker.person_guidance import GuidanceCue
from leica_tracker.robot_gestures import gesture_plan_for


def event(cue,stamp):
    plan=gesture_plan_for(GuidanceCue(cue))
    return json.dumps(dict(event='person_guidance',cue=cue,tracking=True,
        coordinate_frame='normalized_camera_image',robot_actuated=False,
        source_monotonic_ns=round(stamp*1e9),gesture=asdict(plan) if plan else None))


@pytest.mark.parametrize('cue',['MOVE_RIGHT','ARRIVED'])
def test_transport_interruption_latches_hold_through_wait_until_new_explicit_cue(cue):
    stream=sim.GuidanceStream();stream.accept(event(cue,10.),10.)
    assert stream.offset(10.05) is not None
    stream.interrupt_transport()
    for i in range(1,10):
        stamp=10.+i*.1;stream.accept(event('WAIT',stamp),stamp)
        assert stream.offset(stamp) is None
    stream.accept(event('MOVE_LEFT',11.),11.)
    assert stream.offset(11.05) is not None


def backend(monkeypatch,tmp_path,error=None,changed_identity=False):
    stop=tmp_path/'stop';sent=[];closed=[];calls=[]
    state=dict(joints=list(sim.HOME_JOINTS),tcp_pose=[-.4919,-.1333,.2093,0.,0.,0.],
        active_tcp_offset=[0.,0.,.2786,0.,0.,0.],urx=dict(clients=0,program_running=False),
        instance_id='test-simulator',backend='sim',service='urx-browser-simulator')
    def get(path):
        if path.endswith('manifest'):
            return dict(target=dict(kind='simulator',authority_origin='simulation',instance_id='test-simulator'))
        if path.endswith('/motion'):
            calls.append(path)
            if error and len(calls)<=2:raise error('simulated feedback outage')
            stop.touch()
            return dict(state,instance_id='unexpected-instance' if changed_identity else 'test-simulator')
        return state
    class Robot:
        def send(self,rates):sent.append(rates)
        def close(self):closed.append(True)
    clock=itertools.count(100.,1/32)
    monkeypatch.setattr(sim.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(sim,'get_state',get)
    monkeypatch.setattr(sim,'LocalJointStream',Robot)
    return stop,sent,closed,calls


@pytest.mark.parametrize('error',[TimeoutError,ConnectionResetError,http.client.RemoteDisconnected])
def test_feedback_failure_keeps_bridge_alive_and_never_replays_interrupted_motion(monkeypatch,tmp_path,error):
    stop,sent,closed,calls=backend(monkeypatch,tmp_path,error=error)
    assert sim.main(['--source','synthetic','--stop-file',str(stop)])==0
    assert len(calls)==3 and len(sent)==3
    assert all(rates==[0.]*6 for rates in sent)
    assert closed==[True]


def test_simulator_identity_change_still_fails_closed(monkeypatch,tmp_path):
    stop,sent,closed,_=backend(monkeypatch,tmp_path,changed_identity=True)
    with pytest.raises(RuntimeError,match='identity changed'):
        sim.main(['--source','synthetic','--stop-file',str(stop)])
    assert not sent and closed==[True]
