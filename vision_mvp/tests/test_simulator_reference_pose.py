import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/set-simulator-reference.py'

def module():
    spec=importlib.util.spec_from_file_location('set_simulator_reference',SCRIPT)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def fixture():
    return {'scope':'historical_simulator_reference_only',
            'joints_rad':[-1.2561581771,-1.3703361017,-1.9575747252,-1.0283110899,1.4664084911,.2953072190],
            'tcp_offset_m_rad':[0,0,.2786,0,0,0],
            'measured_tcp_m_rad':[-.00226555,-.55313305,.18877248,.00216296,3.07961448,-.57692677]}

class FakeSimulator:
    def __init__(self,*,busy=False,backend='sim',tcp=None,owner='idle',speed=None):
        self.backend=backend;self.busy=busy;self.target=None;self.tcp=tcp or [0,0,.2786,0,0,0]
        self.owner=owner;self.speed=speed or [0]*6
    def health(self):return {'service':'urx-browser-simulator','backend':self.backend,'instance_id':'instance'}
    def state(self):
        return {'service':'urx-browser-simulator','backend':self.backend,'instance_id':'instance',
                'urx':{'clients':int(self.busy),'program_running':self.busy,'control_owner':self.owner},'active_tcp_offset':self.tcp,
                'joints':self.target or [0]*6,'joint_velocity':self.speed,'tcp_pose':fixture()['measured_tcp_m_rad']}
    def move_joints(self,target,**kwargs):self.target=list(target);assert kwargs['wait'] is True

def test_reproduces_six_joints_on_verified_idle_simulator_only():
    c=FakeSimulator();result=module().set_reference(c,fixture())
    assert c.target==fixture()['joints_rad'] and result['simulated'] is True
    assert result['max_joint_error_rad']==0 and result['tcp_position_difference_mm']==0

@pytest.mark.parametrize('client',[FakeSimulator(busy=True),FakeSimulator(backend='physical'),FakeSimulator(tcp=[0]*6),
                                  FakeSimulator(owner='browser'),FakeSimulator(speed=[.05,0,0,0,0,0])])
def test_refuses_busy_wrong_identity_or_wrong_tcp_without_moving(client):
    with pytest.raises((ValueError,RuntimeError)):module().set_reference(client,fixture())
    assert client.target is None

@pytest.mark.parametrize('field,bad',[('joints_rad',[0]*5),('joints_rad',[float('nan')]*6),('scope','physical')])
def test_rejects_invalid_or_non_simulator_reference(field,bad):
    value=fixture();value[field]=bad;c=FakeSimulator()
    with pytest.raises(ValueError):module().set_reference(c,value)
    assert c.target is None
