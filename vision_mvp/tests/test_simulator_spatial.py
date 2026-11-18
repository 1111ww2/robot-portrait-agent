import json
import math
import numpy as np
import pytest
from leica_tracker import simulator_spatial as spatial
from leica_tracker.simulator_fixed_base import HOME_X
from leica_tracker.simulator_planar import HOME_Z
from test_simulator_fixed_base import fk


def test_spatial_path_preserves_orientation_and_small_base_envelope():
    home=fk(spatial.pose_at_xyz(HOME_X,spatial.HOME_Y,HOME_Z))
    for right in np.linspace(-1,1,7):
        for down in [-1,0,1]:
            for depth in [-1,0,1]:
                p=spatial.input_target_3d(right,down,depth)
                q=spatial.pose_at_xyz(*p)
                assert abs(q[0])<=math.radians(3)+1e-8
                assert q[5]==pytest.approx(q[0])
                actual=fk(q)
                assert actual[:3,3]==pytest.approx(p,abs=1e-8)
                assert actual[:3,:3]==pytest.approx(home[:3,:3],abs=1e-8)
    assert spatial.input_target_3d(0,0,1)[1]>spatial.HOME_Y
    assert spatial.input_target_3d(0,0,-1)[1]<spatial.HOME_Y


def sample_line(stamp,width,tracking=True):
    return json.dumps(dict(event='tracking_target',source_monotonic_ns=stamp,tracking=tracking,
        command=dict(right=.1,down=.2,toward_subject=.9),motion_intent_m=dict(hold=not tracking),
        image_observation=dict(shoulder_width=width,confidence=.9)))


def test_depth_calibrates_from_width_then_near_recedes_and_far_approaches():
    s=spatial.SpatialSamples()
    for n in range(50):s.accept(sample_line(10_000_000_000+n*50_000_000,.3),10+n*.05)
    assert s.baseline==pytest.approx(.3)
    assert s.latest[0][2]==pytest.approx(0)
    for n in range(50,80):s.accept(sample_line(10_000_000_000+n*50_000_000,.4),10+n*.05)
    assert s.latest[0][2]<-.8
    for n in range(80,120):s.accept(sample_line(10_000_000_000+n*50_000_000,.2),10+n*.05)
    assert s.latest[0][2]>.8
    s.accept(sample_line(16_000_000_000,.2,False),16)
    assert s.latest[0] is None
    assert s.baseline==pytest.approx(.3)


def test_3d_velocity_has_shared_budget_base_rate_cap_and_loss_hold():
    p=[HOME_X,spatial.HOME_Y,HOME_Z];q=spatial.pose_at_xyz(*p)
    rates,v=spatial.joint_command_3d(q,p,((1,-1,1),10,3,0),10,[0,0,0],.05,.18)
    assert np.linalg.norm(v)<=.03+1e-8
    assert abs(rates[0])<=math.radians(3)+1e-8
    assert max(map(abs,rates))<=1.1
    assert spatial.joint_command_3d(q,p,(None,10,0,1),10,v,.05,.18)==([0.]*6,[0.,0.,0.])


def test_camera_can_emit_numeric_width_for_calibration():
    from leica_tracker.app import serialize_tracking_event
    from leica_tracker.continuous import CameraRelativeCommand
    from leica_tracker.motion import MotionIntent
    from types import SimpleNamespace
    command=CameraRelativeCommand(True,0,0,0,'TRACKING')
    intent=SimpleNamespace(right_m=0,down_m=0,toward_subject_m=0,hold=False)
    result=json.loads(serialize_tracking_event(command,intent,1,
        observation=SimpleNamespace(shoulder_width=.32,confidence=.9)))
    assert result['image_observation']==dict(shoulder_width=.32,confidence=.9)


def test_stream_initializes_depth_calibration_before_reading():
    from unittest.mock import patch
    with patch('threading.Thread.start'):
        stream=spatial.SpatialInput()
    assert stream.baseline is None and stream.failure is None
    stream.accept(sample_line(10_000_000_000,.3),10.)
    assert stream.latest[0]==(.1,.2,0.)
