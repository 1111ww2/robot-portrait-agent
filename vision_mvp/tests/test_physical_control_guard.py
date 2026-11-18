import pytest
from leica_tracker.physical_control_guard import ControlGuard

def test_stop_epoch_change_latches_until_explicit_new_guard():
    g=ControlGuard();g.observe({'instance':'test-instance','epoch':0},10);g.check(10.1)
    with pytest.raises(RuntimeError):g.observe({'instance':'test-instance','epoch':1},10.2)
    with pytest.raises(RuntimeError):g.observe({'instance':'test-instance','epoch':0},10.3)

def test_service_restart_and_stale_control_cannot_resume_motion():
    g=ControlGuard();g.observe({'instance':'first','epoch':0},10)
    with pytest.raises(RuntimeError):g.observe({'instance':'second','epoch':0},10.1)
    g=ControlGuard();g.observe({'instance':'first','epoch':0},10)
    with pytest.raises(RuntimeError):g.check(10.36)
    with pytest.raises(RuntimeError):g.observe({'instance':'first','epoch':0},10.4)

@pytest.mark.parametrize('value',[{}, {'instance':'x','epoch':True},{'instance':'x','epoch':-1},{'instance':'','epoch':0}])
def test_invalid_control_cannot_arm(value):
    g=ControlGuard()
    with pytest.raises(RuntimeError):g.observe(value,10)

def test_no_control_observation_means_no_motion():
    with pytest.raises(RuntimeError):ControlGuard().check(10)
