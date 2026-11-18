"""打包后须原样保留显式选定的数值方案。"""
import copy

import pytest

from leica_tracker.motion_configuration import configure_motion_design
from leica_tracker.operator_design import prepare_selected_design, selected_profile
from test_physical_runtime import data


def selected():
    source=data()
    return prepare_selected_design(source,'ur5e',start_joints=source['intro']['reference_joints'])


def test_canonical_configuration_keeps_selected_design_and_hashes_byte_equivalent():
    original=selected();before=copy.deepcopy(original)
    result=configure_motion_design(original)
    assert original==before and result==before and result is not original
    assert selected_profile(result).cfg.robot_model=='ur5e'


def test_packaging_never_resigns_a_tampered_selection():
    original=selected();original['installation']['speed']*=1.0000000001
    with pytest.raises(ValueError):configure_motion_design(original)


@pytest.mark.parametrize('kwargs',[dict(joint_speed=.6),dict(duration_s=3.4),dict(repetitions=1)])
def test_changed_selected_dynamics_require_a_new_explicit_design(kwargs):
    with pytest.raises(ValueError,match='selected design'):
        configure_motion_design(selected(),**kwargs)


def test_selected_bundle_marks_live_reference_capture_without_physical_acceptance(tmp_path):
    from test_canonical_delivery import workspace
    from leica_tracker.canonical_delivery import prepare_canonical_delivery
    root=workspace(tmp_path)
    source=data();q=source['intro']['reference_joints']
    sources={model:prepare_selected_design(source,model,start_joints=q) for model in ('ur5e','ur10e')}
    manifest=prepare_canonical_delivery(sources,root/'bundle',workspace=root)
    for record in manifest['models'].values():
        assert record['physical_ready'] is False
        assert record['physical_installation_configured'] is False
        assert record['runtime_reference_capture']=='on_explicit_start'
        assert record['approval_mode']=='operator_selected_design'
