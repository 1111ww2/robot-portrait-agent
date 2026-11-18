"""交付准备仅编译文件；禁止构造套接字。"""
import copy
import json
import socket

import pytest
from test_physical_runtime import data


def test_delivery_exports_complete_programs_and_unreviewed_target(tmp_path,monkeypatch):
    from leica_tracker.physical_delivery import prepare_delivery
    monkeypatch.setattr(socket,'socket',lambda *a,**k:pytest.fail('offline preparation opened socket'))
    original=data();before=copy.deepcopy(original)
    result=prepare_delivery(original,tmp_path/'delivery',workspace=tmp_path,
        target_joint_speed=.6,target_joint_acceleration=1.2)
    assert original==before
    assert result['target']['joint_speed_rad_s']==.6
    assert result['target']['joint_acceleration_rad_s2']==1.2
    assert result['physical_acceptance']=='pending'
    assert result['target']['executable'] is False
    root=tmp_path/'delivery'
    assert (root/'opening-reviewed.urscript').read_text().count('movej(')==14
    assert 'read_input_integer_register' in (root/'guidance-reviewed.urscript').read_text()
    draft=json.loads((root/'target-speed-DRAFT.json').read_text())
    assert draft['intro']['joint_speed']==.6
    assert draft['intro']['joint_acceleration']==1.2
    assert draft['intro']['max_tcp_linear_speed']==original['intro']['max_tcp_linear_speed']
    assert draft['intro']['max_tcp_angular_speed']==original['intro']['max_tcp_angular_speed']
    from leica_tracker.physical_runtime import reviewed_profile
    from leica_tracker.physical_intro_program import IntroProgram
    with pytest.raises(ValueError):reviewed_profile(draft)
    with pytest.raises(ValueError):IntroProgram(draft['intro'],1)
    assert 'Start on the iPad' in (root/'README.md').read_text()
    assert '-Profile' in (root/'start-reviewed-ipad.ps1').read_text()


def test_invalid_source_and_existing_output_do_not_write(tmp_path):
    from leica_tracker.physical_delivery import prepare_delivery
    bad=data();bad['intro']['review']['path']=False
    root=tmp_path/'bad'
    with pytest.raises(ValueError):prepare_delivery(bad,root,workspace=tmp_path)
    assert not root.exists()
    root.mkdir();(root/'keep').write_text('preserve')
    with pytest.raises(FileExistsError):prepare_delivery(data(),root,workspace=tmp_path)
    assert (root/'keep').read_text()=='preserve'


def test_preparation_refuses_output_outside_workspace_before_write(tmp_path):
    from leica_tracker.physical_delivery import prepare_delivery
    with pytest.raises(ValueError,match='workspace'):
        prepare_delivery(data(),tmp_path/'outside',workspace=tmp_path/'workspace')


def test_explicit_target_monitor_removal_preserves_baseline_and_is_not_hardware_approval(tmp_path):
    from leica_tracker.physical_delivery import prepare_delivery
    original=data();before=copy.deepcopy(original)
    root=tmp_path/'delivery'
    result=prepare_delivery(original,root,workspace=tmp_path,disable_target_angular_monitor=True)
    assert original==before
    baseline=json.loads((root/'baseline-profile.json').read_text())
    assert baseline['intro']['max_tcp_angular_speed']==original['intro']['max_tcp_angular_speed']
    target=json.loads((root/'target-speed-DRAFT.json').read_text())
    assert target['intro']['max_tcp_angular_speed'] is None
    assert target['intro']['max_tcp_linear_speed']==original['intro']['max_tcp_linear_speed']
    assert target['intro']['reviewed']=={}
    assert result['target']['project_tcp_angular_monitor_enabled'] is False
    assert result['target']['executable'] is False
    audit=json.loads((root/'target-speed-audit.json').read_text())
    assert audit['tcp_angular_monitor_enabled'] is False
    assert audit['known_angular_conflicts']==[]
    assert audit['requires_onsite_review'] is True


def test_target_can_omit_extra_motion_monitors_without_removing_control_interlocks(tmp_path):
    from leica_tracker.physical_delivery import prepare_delivery
    original=data();before=copy.deepcopy(original);root=tmp_path/'delivery'
    result=prepare_delivery(original,root,workspace=tmp_path,disable_extra_motion_monitors=True)
    assert original==before
    target=json.loads((root/'target-speed-DRAFT.json').read_text())
    for key in ('max_tcp_linear_speed','max_tcp_angular_speed'):
        assert target['intro'][key] is None
        assert target['installation'][key] is None
    assert not target['installation']['review']['stop']
    assert target['reviewed_installation']=={}
    assert target['hold_timeout']==original['hold_timeout']
    assert target['installation']['lower']==original['installation']['lower']
    assert target['installation']['upper']==original['installation']['upper']
    assert result['target']['project_tcp_linear_monitor_enabled'] is False
    assert result['target']['executable'] is False


@pytest.mark.parametrize('speed,acceleration',[(True,1.2),(.6,float('inf')),(0,1.2),(.6,-1)])
def test_invalid_target_dynamics_do_not_create_output(tmp_path,speed,acceleration):
    from leica_tracker.physical_delivery import prepare_delivery
    with pytest.raises(ValueError):
        prepare_delivery(data(),tmp_path/'bad',workspace=tmp_path,
            target_joint_speed=speed,target_joint_acceleration=acceleration)
    assert not (tmp_path/'bad').exists()
