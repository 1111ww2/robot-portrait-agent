import time
import json
import pytest
from leica_tracker.simulator_feedback import feedback_from_motion


def sample(**changes):
    d={'token':'x'*32,'time_ns':10000000000,'source_ns':123456789,'cue':'MOVE_LEFT',
       'hold':False,'delta':[-.06,0.,0.], 'amplitude':1.}
    d.update(changes)
    return d


def test_feedback_follows_actual_position_and_matching_event():
    data=sample()
    assert feedback_from_motion(data,token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT')['intensity']==.5
    assert feedback_from_motion(data,token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT')['edge']=='left'
    assert feedback_from_motion(data,token='x'*32,now_ns=10100000000,source_ns=123456790,cue='MOVE_LEFT') is None
    assert feedback_from_motion(data,token='x'*32,now_ns=10350000000,source_ns=123456789,cue='MOVE_LEFT') is None
    assert feedback_from_motion(sample(hold=True),token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT') is None
    assert feedback_from_motion(sample(token='wrong'),token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT') is None


def test_wrong_direction_does_not_illuminate_opposite_edge():
    assert feedback_from_motion(sample(delta=[.06,0.,0.]),token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT')['intensity']==0


def test_sim_fast_near_feedback_uses_same_scale_as_motion():
    signal=feedback_from_motion(sample(profile='sim-fast',amplitude=.65,delta=[-.12*.65*.5,0.,0.]),
        token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT')
    assert signal['motion_mode']=='near'
    assert signal['intensity']==pytest.approx(.5,abs=1e-5)
    far=feedback_from_motion(sample(profile='sim-fast',amplitude=1.,delta=[-.12*.5,0.,0.]),
        token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT')
    assert far['motion_mode']=='far'
    assert far['intensity']==pytest.approx(.5,abs=1e-5)


@pytest.mark.parametrize('cue,sign', [('MOVE_CLOSER',-1),('MOVE_FARTHER',1)])
@pytest.mark.parametrize('amplitude', [.65,1.])
def test_sim_fast_depth_light_tracks_reversed_measured_motion(cue,sign,amplitude):
    def feedback(delta):
        return feedback_from_motion(sample(cue=cue,profile='sim-fast',amplitude=amplitude,delta=[0.,delta,0.]),
            token='x'*32,now_ns=10100000000,source_ns=123456789,cue=cue)
    assert feedback(sign*.06*amplitude*.5)['intensity']==pytest.approx(.5,abs=1e-5)
    assert feedback(-sign*.06*amplitude*.5)['intensity']==0.
    assert feedback(0.)['intensity']==0.


@pytest.mark.parametrize("amplitude", [0.15, 0.35, 0.5, 1.0])
def test_feedback_brightness_follows_full_demo_displacement(amplitude):
    data=sample(amplitude=amplitude,delta=[-.12*.5,0.,0.])
    result=feedback_from_motion(data,token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT')
    assert result['intensity']==pytest.approx(.5,abs=1e-5)
    data['delta'][0]=-.12
    assert feedback_from_motion(data,token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT')['intensity']==1.

@pytest.mark.parametrize("amplitude", [None, True, 0., .149, 1.01, float('nan'), float('inf')])
def test_reject_invalid_amplitude(amplitude):
    assert feedback_from_motion(sample(amplitude=amplitude),token='x'*32,now_ns=10100000000,source_ns=123456789,cue='MOVE_LEFT') is None


def test_sender_serializes_plan_amplitude_for_receiver():
    from leica_tracker.simulator_feedback import MotionFeedbackSender
    class SocketStub:
        def sendto(self,payload,address):
            self.data=json.loads(payload);self.address=address
    sender=object.__new__(MotionFeedbackSender)
    sender.port=23456;sender.token='x'*32;sender.socket=SocketStub()
    sender.send(source_ns=123456789,cue='MOVE_LEFT',hold=False,delta=[-.06,0.,0.],amplitude=.35)
    data=sender.socket.data
    assert data['amplitude']==.35
    assert sender.socket.address==('127.0.0.1',23456)
    result=feedback_from_motion(data,token='x'*32,now_ns=data['time_ns'],source_ns=123456789,cue='MOVE_LEFT')
    assert result['intensity']==pytest.approx(.5)


@pytest.mark.parametrize("cue,edge,delta", [
    ("MOVE_LEFT", "left", [-.06, 0., 0.]),
    ("MOVE_RIGHT", "right", [.06, 0., 0.]),
    ("MOVE_UP", "top", [0., 0., .0425]),
    ("MOVE_DOWN", "bottom", [0., 0., -.0425]),
    ("MOVE_CLOSER", "all", [0., .03, 0.]),
    ("MOVE_FARTHER", "all", [0., -.03, 0.]),
])
def test_directional_feedback_uses_half_of_new_simulator_extent(cue,edge,delta):
    signal=feedback_from_motion(sample(cue=cue,delta=delta),token='x'*32,
        now_ns=10100000000,source_ns=123456789,cue=cue)
    assert signal['edge']==edge
    assert signal['intensity']==pytest.approx(.5)


def test_photo_gate_requires_completed_matching_arrival_and_stable_reference():
    from leica_tracker.simulator_feedback import SimulatorPhotoGate
    gate=SimulatorPhotoGate()
    def update(now,**changes):
        data=dict(source_ns=123,cue='ARRIVED',pattern='confirm_retract_hold',hold=False,
                  elapsed_ms=9000,duration_ms=8800,joints=[0.]*6,neutral=[0.]*6,
                  velocity=[0.]*6,now=now)
        data.update(changes)
        return gate.update(**data)
    assert not update(1.,elapsed_ms=8799)
    assert not update(2.,joints=[.01,0,0,0,0,0])
    assert not update(3.,velocity=[.01,0,0,0,0,0])
    assert not update(4.)
    assert not update(4.299)
    assert update(4.301)
    assert update(4.351)
    assert not update(4.4,hold=True)
    assert not update(4.45)
    assert not update(4.46,cue='MOVE_LEFT')
    assert not update(5.)
    assert not update(5.4,source_ns=124)
    assert not update(5.55,source_ns=124)
    assert update(5.71,source_ns=124)
    assert not update(5.72,pattern='double_sweep')


def test_photo_gate_rejects_bad_or_missing_measurements():
    from leica_tracker.simulator_feedback import SimulatorPhotoGate
    gate=SimulatorPhotoGate()
    args=dict(source_ns=1,cue='ARRIVED',pattern='confirm_retract_hold',hold=False,
              elapsed_ms=9000,duration_ms=8800,joints=[0.]*6,neutral=[0.]*6,
              velocity=[0.]*6,now=1.)
    assert not gate.update(**{**args,'velocity':None})
    assert not gate.update(**{**args,'joints':[float('nan')]*6})
    assert not gate.update(**{**args,'source_ns':None})


def test_photo_gate_allows_stable_reference_after_last_confirmation_pulse():
    from leica_tracker.simulator_feedback import SimulatorPhotoGate
    gate=SimulatorPhotoGate()
    args=dict(source_ns=1,cue='ARRIVED',pattern='confirm_retract_hold',hold=False,
              duration_ms=6000,joints=[0.]*6,neutral=[0.]*6,velocity=[0.]*6)
    assert not gate.update(**args,elapsed_ms=5999,now=1.)
    assert not gate.update(**args,elapsed_ms=6000,now=1.1)
    assert not gate.update(**args,elapsed_ms=6200,now=1.3)
    assert gate.update(**args,elapsed_ms=6350,now=1.45)


def test_photo_receiver_requires_matching_fresh_authenticated_arrival():
    from leica_tracker.simulator_feedback import photo_ready_from_motion
    data=sample(cue='ARRIVED',photo_ready=True)
    def ready(**changes):
        return photo_ready_from_motion({**data,**changes},token='x'*32,
                 now_ns=10100000000,source_ns=123456789)
    assert ready()
    assert not ready(hold=True)
    assert not ready(source_ns=123456790)
    assert not ready(cue='MOVE_LEFT')
    assert not ready(photo_ready=False)
    assert not ready(time_ns=9700000000)
    assert not ready(token='y'*32)


def test_sender_receiver_photo_ready_requires_live_arrival():
    import socket
    from leica_tracker.simulator_feedback import MotionFeedbackReceiver,MotionFeedbackSender
    probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    probe.bind(('127.0.0.1',0));port=probe.getsockname()[1];probe.close()
    config=f'{port}:'+'a'*32
    receiver=MotionFeedbackReceiver(config);receiver.start();sender=MotionFeedbackSender(config)
    try:
        sender.send(source_ns=123,cue='ARRIVED',hold=False,delta=[0.,0.,0.],photo_ready=True)
        deadline=time.monotonic()+1
        while time.monotonic()<deadline and not receiver.photo_ready(123):time.sleep(.005)
        assert receiver.photo_ready(123)
        assert not receiver.photo_ready(124)
        sender.send(source_ns=123,cue='ARRIVED',hold=True,delta=[0.,0.,0.],photo_ready=False)
        deadline=time.monotonic()+1
        while time.monotonic()<deadline and receiver.photo_ready(123):time.sleep(.005)
        assert not receiver.photo_ready(123)
    finally:
        sender.close();receiver.close()
