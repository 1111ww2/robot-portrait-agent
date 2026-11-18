import json
import math
import numpy as np
import pytest
from leica_tracker import simulator_planar as planar
from leica_tracker.simulator_fixed_base import HOME_X, LEFT_X, RIGHT_X, input_target
from test_simulator_fixed_base import fk


def test_workspace_projection_and_fk_preserve_depth_orientation():
    home=fk(planar.pose_at_xz(HOME_X,planar.HOME_Z))
    for right in np.linspace(-1,1,13):
        for down in np.linspace(-1,1,13):
            x,z=planar.input_target_2d(right,down)
            q=planar.pose_at_xz(x,z);actual=fk(q)
            assert LEFT_X<=x<=RIGHT_X and planar.MIN_Z<=z<=planar.MAX_Z
            assert math.hypot(x+.0997,z+.2157)<=planar.MAX_RADIUS+1e-10
            assert actual[:3,3]==pytest.approx([x,-.1333,z],abs=1e-9)
            assert actual[:3,:3]==pytest.approx(home[:3,:3],abs=1e-9)
    assert planar.input_target_2d(.5,0)[0]==input_target(.5)
    assert planar.input_target_2d(0,-.5)[1]-planar.HOME_Z==pytest.approx(.5*(HOME_X-LEFT_X))
    assert planar.input_target_2d(0,1)[1]==planar.MIN_Z


def test_diagonal_motion_has_one_speed_and_acceleration_budget():
    position=[HOME_X,planar.HOME_Z];q=planar.pose_at_xz(*position)
    rates,v=planar.joint_command_2d(q,position,((1,-1),10,3,0),10,[0,0],.05,.18)
    assert v[0]<0<v[1]
    assert math.hypot(*v)<=.03+1e-9
    assert rates[0]==0 and max(map(abs,rates))<=1.1
    for _ in range(20):
        rates,v=planar.joint_command_2d(q,position,((1,-1),10,3,0),10,v,.05,.18)
        assert math.hypot(*v)<=.18+1e-9


def test_loss_holds_both_axes_and_endpoints_cannot_be_extrapolated():
    x,z=planar.input_target_2d(1,-1);q=planar.pose_at_xz(x,z)
    rates,v=planar.joint_command_2d(q,[x,z],((1,-1),10,3,0),10,[-.1,.1],.05,.18)
    assert rates==pytest.approx([0]*6,abs=1e-9)
    assert v==pytest.approx([0,0],abs=1e-9)
    assert planar.joint_command_2d(q,[x,z],(None,10,0,1),10,[-.1,.1],.05,.18)==([0.]*6,[0.,0.])


def event(stamp,**changes):
    e=dict(event='tracking_target',source_monotonic_ns=stamp,tracking=True,
           command=dict(right=.2,down=-.3,toward_subject=.8),motion_intent_m=dict(hold=False))
    e.update(changes);return json.dumps(e)


def test_planar_samples_accept_vertical_but_do_not_command_depth():
    samples=planar.PlanarSamples()
    for n in range(3):samples.accept(event(10_000_000_000+n),10.)
    assert samples.latest[0]==(.2,-.3) and samples.latest[2]==3
    samples.accept(event(10_000_000_003,tracking=False),10.)
    assert samples.latest[0] is None
    with pytest.raises(ValueError):samples.accept(event(10_000_000_003),10.)
    samples.accept(event(10_000_000_004),11.)
    assert samples.latest[0] is None
