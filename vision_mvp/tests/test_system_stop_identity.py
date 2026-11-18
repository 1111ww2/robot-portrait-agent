"""仅使用回环地址上的模拟显示服务检查监管器的控制权判断。"""
import os
import urllib.error

import pytest

from leica_tracker.guest_display import GuestDisplay
from test_control_heartbeat import Capture, call, start


def test_state_exposes_this_process_startup_identity(monkeypatch):
    monkeypatch.setenv('LEICA_SYSTEM_RUN_ID','a'*32)
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        assert call(d,'/state')['system_run_id']=='a'*32


@pytest.mark.parametrize('nonce,instance', [('b'*32,None),('a'*32,'old-process'),(None,None)])
def test_supervisor_stop_cannot_stop_another_process(monkeypatch,nonce,instance):
    monkeypatch.setenv('LEICA_SYSTEM_RUN_ID','a'*32)
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        identity=start(d)
        with pytest.raises(urllib.error.HTTPError) as error:
            call(d,'/action',{'action':'stop','system_run_id':nonce,
                 'control':{'instance':instance or identity['instance']}},client='supervisor')
        assert error.value.code==409
        assert d.active


def test_owned_supervisor_stop_and_normal_ipad_stop_remain_available(monkeypatch):
    monkeypatch.setenv('LEICA_SYSTEM_RUN_ID','a'*32)
    with GuestDisplay(port=0,remote_capture=Capture(),physical_feedback=True) as d:
        identity=start(d)
        assert not call(d,'/action',{'action':'stop','system_run_id':'a'*32,
            'control':{'instance':identity['instance']}},client='supervisor')['active']
        start(d)
        assert not call(d,'/action',{'action':'stop'})['active']


def test_malformed_startup_identity_fails_before_server_creation(monkeypatch):
    monkeypatch.setenv('LEICA_SYSTEM_RUN_ID','not-valid')
    with pytest.raises(ValueError,match='system'):
        GuestDisplay(port=0)
