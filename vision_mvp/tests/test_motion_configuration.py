"""选择固定版动作时只计算配置，不伪造实机审核结果。"""
import copy
import math

import pytest

from test_physical_runtime import data


def configure(*args, **kwargs):
    from leica_tracker.motion_configuration import configure_motion_design
    return configure_motion_design(*args, **kwargs)


def test_canonical_design_changes_both_motion_stages_and_preserves_geometry():
    original=data(); before=copy.deepcopy(original)
    result=configure(original)
    assert original==before
    assert result['intro']['joint_speed']==.7
    assert result['intro']['joint_acceleration']==1.2
    assert result['installation']['gesture']==dict(duration_s=3.3,repetitions=2,joint_speed_limit=.7)
    assert result['installation']['base_rate']==.7
    displacement=max(math.dist(point,original['installation']['reference'][:3])
                     for path in original['paths'].values() for point in path)
    assert result['installation']['speed']==pytest.approx(1.15*displacement*math.pi*2/3.3)
    assert result['installation']['speed']!=.7
    for key in ('reference','joints','tcp','payload_kg','center_of_gravity_m','lower','upper','base_limit'):
        assert result['installation'].get(key)==original['installation'].get(key)
    assert result['paths']==original['paths']


def test_changed_dynamics_invalidate_acceptance_without_dropping_stop_protocol():
    original=data(); result=configure(original)
    assert result['reviewed_installation']=={}
    assert result['intro']['reviewed']=={}
    assert not any(result['installation']['review'].values())
    assert not any(result['intro']['review'].values())
    assert result.get('intro_preview') is None
    assert result['status']=='CANONICAL_DESIGN_PHYSICAL_ACCEPTANCE_PENDING'
    assert result['motion_design']['physical_acceptance'] is False
    assert result['hold_timeout']==original['hold_timeout']
    assert result['max_joint_speed']==original['max_joint_speed']
    for section in ('intro','installation'):
        assert result[section]['max_tcp_linear_speed'] is None
        assert result[section]['max_tcp_angular_speed'] is None


def test_ur5_template_gets_cadence_without_inventing_an_installation():
    import json
    from pathlib import Path
    template=json.loads((Path(__file__).parents[1]/'config/physical-ur5e.template.json').read_text())
    result=configure(template)
    assert result['robot_model']=='ur5e'
    assert result['installation']['reference'] is None
    assert result['installation']['tcp'] is None
    assert result['paths']==template['paths']
    assert result['intro']['waypoints']==[]
    assert result['installation']['speed']==pytest.approx(1.15*.12*math.pi*2/3.3)


@pytest.mark.parametrize('speed',[0,-1,.71,float('nan'),True])
def test_invalid_global_joint_speed_rejected(speed):
    with pytest.raises(ValueError): configure(data(),joint_speed=speed)


def test_malformed_path_is_not_replaced_by_a_default_amplitude():
    profile=data();profile['paths']['MOVE_LEFT']=[[float('nan'),0,0]]
    with pytest.raises(ValueError): configure(profile)


def test_motion_cap_does_not_become_reference_joint_stillness_threshold():
    from types import SimpleNamespace
    from leica_tracker.physical_interaction_supervisor import PhysicalInteractionSupervisor
    source=data();source['max_joint_speed']=.1
    result=configure(source)
    assert result['max_joint_speed']==.003
    observer=SimpleNamespace(engine=SimpleNamespace(state='READY'),
        max_joint_speed=result['max_joint_speed'],_last_enabled_sequence=1)
    feedback=dict(state=0,sequence=2,tcp_speed=[0.]*6)
    assert not PhysicalInteractionSupervisor._held_and_still(observer,feedback,[0.]*6,[.69,0,0,0,0,0])
    assert PhysicalInteractionSupervisor._held_and_still(observer,feedback,[0.]*6,[0.]*6)
