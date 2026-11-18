"""在导入任何SDK之前，共享空间入口必须选择批量运输。"""
import sys
from types import SimpleNamespace
import pytest
from leica_tracker import ur_rtde_transport as transport


@pytest.mark.parametrize('receive_only',[False,True])
def test_spatial_entry_uses_the_same_raw_worker_without_importing_sdk(monkeypatch,receive_only):
    calls=[]
    def worker(host,**options):
        calls.append((host,options));return 17
    monkeypatch.setitem(sys.modules,'leica_tracker.atomic_rtde_transport',SimpleNamespace(worker=worker))
    monkeypatch.setitem(sys.modules,'rtde_receive',None)
    monkeypatch.setitem(sys.modules,'rtde_io',None)
    assert transport.worker('fixture.invalid',spatial=True,receive_only=receive_only)==17
    assert len(calls)==1 and calls[0][0]=='fixture.invalid'
    assert calls[0][1]==dict(receive_only=receive_only,ack_clock_factory=transport.AckReceiptClock,
                            validate_update=transport.validate_spatial_target)


def test_scalar_entry_retains_explicit_legacy_sdk_path(monkeypatch):
    calls=[]
    assert hasattr(transport,'_sdk_worker')
    monkeypatch.setattr(transport,'_sdk_worker',lambda *a,**k:calls.append((a,k)) or 19)
    assert transport.worker('fixture.invalid',spatial=False,receive_only=False)==19
    assert calls==[(('fixture.invalid',),dict(spatial=False,receive_only=False))]
