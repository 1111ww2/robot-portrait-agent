import pytest
from leica_tracker.physical_runtime import _fault_reason


@pytest.mark.parametrize('error,reason',[
    ('Intro control stopped: Local reference service unavailable control_heartbeat_expired','control'),
    ('Intro resident fault -16: TCP linear speed exceeds reviewed limit','speed'),
    ('Intro resident fault -12: controller input heartbeat expired','control'),
    ('Robot is not at the reviewed stationary intro reference','reference'),
])
def test_first_cause_is_not_masked_by_reference_transport_name(error,reason):
    assert _fault_reason(RuntimeError(error))==reason
