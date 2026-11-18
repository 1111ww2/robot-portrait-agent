import math
import numpy as np
import pytest
from leica_tracker import simulator_fixed_base as fixed


def fk(q):
    t = np.eye(4)
    for angle, (a, alpha, d) in zip(q, [(0, math.pi/2, .1625), (-.425, 0, 0),
             (-.3922, 0, 0), (0, math.pi/2, .1333), (0, -math.pi/2, .0997), (0, 0, .0996)]):
        c,s,ca,sa=math.cos(angle),math.sin(angle),math.cos(alpha),math.sin(alpha)
        t=t@np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d],[0,0,0,1]])
    t[:3,3] += t[:3,:3]@np.array([0,0,.2786])
    return t


def test_entire_path_is_horizontal_with_fixed_base_and_tool_orientation():
    home=fk(fixed.pose_at_x(fixed.HOME_X))
    for x in np.linspace(fixed.LEFT_X, fixed.RIGHT_X, 101):
        q=fixed.pose_at_x(x)
        actual=fk(q)
        assert q[0] == 0
        assert actual[:3,3] == pytest.approx([x,-.1333,.2093], abs=1e-9)
        assert actual[:3,:3] == pytest.approx(home[:3,:3], abs=1e-9)


def test_asymmetric_endpoints_and_home_are_clamped():
    assert fixed.input_target(-1)==fixed.RIGHT_X
    assert fixed.input_target(1)==fixed.LEFT_X
    assert fixed.input_target(0)==fixed.HOME_X
    assert fixed.input_target(5)==fixed.LEFT_X
    assert fixed.input_target(-5)==fixed.RIGHT_X


def test_commands_accelerate_without_base_motion_and_stop_at_boundary():
    q=fixed.pose_at_x(fixed.HOME_X)
    rates,v=fixed.joint_command(q,fixed.HOME_X,(1,10,3,0),10,0,.05,.18)
    assert rates[0]==0 and -.03-1e-9<=v<0 and max(map(abs,rates))<=1.1
    q=fixed.pose_at_x(fixed.RIGHT_X)
    rates,v=fixed.joint_command(q,fixed.RIGHT_X,(-1,10,3,0),10,.18,.05,.18)
    assert rates==pytest.approx([0]*6,abs=1e-9)
    assert v==0


@pytest.mark.parametrize('sample', [(1,9,3,0),(None,10,0,1),(1,11,3,0),(1,10,2,0)])
def test_loss_stale_future_unconfirmed_hold(sample):
    q=fixed.pose_at_x(fixed.HOME_X)
    assert fixed.joint_command(q,fixed.HOME_X,sample,10,.18,.05,.18)==([0.]*6,0.)


def test_small_base_correction_allowed_but_large_rotation_rejected():
    q=fixed.pose_at_x(fixed.HOME_X); q[0]=math.radians(1)
    rates,v=fixed.joint_command(q,fixed.HOME_X,(0,10,3,0),10,0,.05,.18)
    assert -math.radians(3)<=rates[0]<0
    q[0]=math.radians(4)
    with pytest.raises(ValueError):
        fixed.joint_command(q,fixed.HOME_X,(1,10,3,0),10,0,.05,.18)
    with pytest.raises(ValueError):
        fixed.input_target(math.nan)
