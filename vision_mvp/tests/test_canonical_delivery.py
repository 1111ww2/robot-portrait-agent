"""完整的共享代码包永远不会将设计转化为物理接受。"""
import copy
import json
import hashlib
from pathlib import Path

import pytest

from test_physical_runtime import data


def sources():
    ur10=data(); ur10.update(robot_model='ur10e',reviewed_robot_model='ur10e')
    ur5=json.loads((Path(__file__).parents[1]/'config/physical-ur5e.template.json').read_text())
    return {'ur10e':ur10,'ur5e':ur5}


def workspace(path):
    package=path/'vision_mvp/src/leica_tracker'; package.mkdir(parents=True)
    (package/'__init__.py').write_text('# complete source fixture\n')
    (package/'.env').write_text('secret-excluded')
    (package/'private.json').write_text('{"token":"secret-excluded"}')
    (package/'discard.pyc').write_bytes(b'secret-excluded')
    (package/'leica-logo.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    scripts=path/'vision_mvp/scripts'; scripts.mkdir()
    (scripts/'check-motion-light-trace.cjs').write_text('// production replay validator\n')
    for model in ('ur10e','ur5e'):
        (scripts/f'start-{model}-ipad-system.ps1').write_text('# shared entry\n')
        (scripts/f'run-{model}-ipad-system.py').write_text('# shared entry\n')
    (path/'vision_mvp/pyproject.toml').write_text('[project]\nname="fixture"\nversion="1"\n')
    return path


def build(*args,**kwargs):
    from leica_tracker.canonical_delivery import prepare_canonical_delivery
    return prepare_canonical_delivery(*args,**kwargs)


def test_build_contains_both_canonical_configs_shared_code_and_model_entries(tmp_path):
    root=workspace(tmp_path); original=sources(); before=copy.deepcopy(original)
    output=root/'bundle'; manifest=build(original,output,workspace=root)
    assert original==before
    assert manifest['format_version']==1 and manifest['joint_speed_rad_s']==.7
    assert manifest['physical_acceptance']=='pending'
    assert manifest['shared_runtime_source_sha256']
    for model in ('ur5e','ur10e'):
        profile=json.loads((output/f'profiles/{model}.json').read_text())
        assert profile['intro']['joint_speed']==.7
        assert profile['installation']['gesture']['joint_speed_limit']==.7
        assert profile['motion_design']['robot_model']==model
        assert profile['intro']['reviewed']=={}
        assert (output/f'start-{model}.ps1').is_file()
    assert (output/'runtime/vision_mvp/src/leica_tracker/__init__.py').is_file()
    assert not (output/'runtime/vision_mvp/src/leica_tracker/.env').exists()
    assert not (output/'runtime/vision_mvp/src/leica_tracker/discard.pyc').exists()
    assert not (output/'runtime/vision_mvp/src/leica_tracker/private.json').exists()
    assert (output/'shared-runtime.zip').is_file()
    assert (output/'runtime/vision_mvp/scripts/check-motion-light-trace.cjs').is_file()
    # 下载的照片页面读取此运行时资产，而不仅仅是guest.html。
    assert (output/'runtime/vision_mvp/src/leica_tracker/leica-logo.svg').is_file()
    assert '0.7 rad/s' in (output/'README.md').read_text()


def test_runtime_identity_check_rejects_stale_source_before_device_access(tmp_path):
    from leica_tracker.canonical_delivery import verify_runtime_source
    root=workspace(tmp_path); output=root/'bundle'; build(sources(),output,workspace=root)
    assert verify_runtime_source(output,root) is True
    (root/'vision_mvp/src/leica_tracker/__init__.py').write_text('# stale or changed runtime\n')
    with pytest.raises(ValueError,match='source'):
        verify_runtime_source(output,root)


def test_runtime_identity_rejects_new_unrecorded_executable_source(tmp_path):
    from leica_tracker.canonical_delivery import verify_runtime_source
    root=workspace(tmp_path);output=root/'bundle';build(sources(),output,workspace=root)
    (root/'vision_mvp/src/leica_tracker/new_motion_adapter.py').write_text('# unvalidated new executable source\n')
    with pytest.raises(ValueError,match='source'):
        verify_runtime_source(output,root)


def test_runtime_source_identity_matches_exact_packaged_source(tmp_path):
    from leica_tracker.canonical_delivery import runtime_source_identity
    root=workspace(tmp_path);output=root/'bundle'
    before=runtime_source_identity(root);manifest=build(sources(),output,workspace=root)
    assert before['runtime_files']==manifest['runtime_files']
    assert before['shared_runtime_source_sha256']==manifest['shared_runtime_source_sha256']
    (root/'vision_mvp/src/leica_tracker/__init__.py').write_text('# changed source\n')
    assert runtime_source_identity(root)['shared_runtime_source_sha256']!=before['shared_runtime_source_sha256']


def test_bad_model_binding_or_existing_output_is_rejected_without_writing(tmp_path):
    root=workspace(tmp_path); invalid=sources(); invalid['ur5e']['robot_model']='ur10e'
    with pytest.raises(ValueError):build(invalid,root/'bad',workspace=root)
    assert not (root/'bad').exists()
    output=root/'keep'; output.mkdir();(output/'evidence').write_text('keep')
    with pytest.raises(FileExistsError):build(sources(),output,workspace=root)
    assert (output/'evidence').read_text()=='keep'


def test_build_outside_workspace_is_rejected(tmp_path):
    root=workspace(tmp_path/'workspace')
    with pytest.raises(ValueError,match='workspace'):
        build(sources(),tmp_path/'outside',workspace=root)


def test_simulation_exports_require_the_actual_simulation_profile_type(tmp_path):
    from types import SimpleNamespace
    from leica_tracker.canonical_delivery import attach_simulation_exports
    root=workspace(tmp_path);output=root/'bundle';build(sources(),output,workspace=root)
    with pytest.raises(ValueError):
        attach_simulation_exports(output,'ur5e',SimpleNamespace(host='127.0.0.2'),workspace=root)
    assert not (output/'simulation').exists()


@pytest.mark.parametrize('model',['ur5e','ur10e'])
def test_real_simulation_profile_exports_same_canonical_programs_for_each_model(tmp_path,model):
    from leica_tracker.canonical_delivery import attach_simulation_exports
    from leica_tracker.motion_configuration import configure_motion_design
    from leica_tracker.ursim_profile import build_profile
    from test_ursim_profile import snapshots
    root=workspace(tmp_path);output=root/'bundle';build(sources(),output,workspace=root)
    source=configure_motion_design(data())
    profile=build_profile(source,*snapshots(),robot_model=model,joint_speed=.7)
    info=attach_simulation_exports(output,model,profile,workspace=root)
    assert info['simulation_only'] is True and info['physical_acceptance'] is False
    assert info['robot_model']==model and info['joint_speed_rad_s']==.7
    opening=(output/f'simulation/{model}/opening.urscript').read_text()
    guidance=(output/f'simulation/{model}/guidance.urscript').read_text()
    assert 'SIMULATION_ONLY' in opening and 'v=0.7000000000' in opening
    assert 'gesture_age' in guidance and 'flag == 2' in guidance
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['simulation_exports'][model]['physical_acceptance'] is False
    assert manifest['models'][model]['physical_ready'] is False
    with pytest.raises(FileExistsError):attach_simulation_exports(output,model,profile,workspace=root)


def test_export_rejects_opening_or_gesture_dynamics_drift_from_bundle(tmp_path):
    from leica_tracker.canonical_delivery import attach_simulation_exports
    from leica_tracker.motion_configuration import configure_motion_design
    from leica_tracker.ursim_profile import build_profile
    from test_ursim_profile import snapshots
    root=workspace(tmp_path);output=root/'bundle';build(sources(),output,workspace=root)
    source=configure_motion_design(data())
    opening_drift=build_profile(source,*snapshots(),joint_speed=.6)
    with pytest.raises(ValueError,match='dynamics'):
        attach_simulation_exports(output,'ur10e',opening_drift,workspace=root)
    source['installation']['gesture']['joint_speed_limit']=.6
    gesture_drift=build_profile(source,*snapshots(),joint_speed=.7)
    with pytest.raises(ValueError,match='dynamics'):
        attach_simulation_exports(output,'ur10e',gesture_drift,workspace=root)
    assert not (output/'simulation').exists()


def native_case(root,manifest,model='ur10e'):
    from leica_tracker.motion_configuration import configure_motion_design
    from leica_tracker.ursim_profile import build_profile
    from leica_tracker.physical_intro_program import IntroProgram
    from test_ursim_profile import snapshots
    profile=build_profile(configure_motion_design(data()),*snapshots(),robot_model=model,joint_speed=.7)
    case=root/f'native-{model}';case.mkdir()
    installation={key:getattr(profile.cfg,key) for key in ('robot_model','joints','tcp','lower','upper',
        'speed','acceleration','base_limit','base_rate','max_tcp_linear_speed','max_tcp_angular_speed')}
    installation.update(reference=profile.cfg.neutral,gesture=profile.cfg.gesture.as_dict())
    record=dict(model=model,installation=installation,intro=profile.intro,paths=profile.paths,simulation_only=True)
    (case/'profile.json').write_text(json.dumps(record))
    (case/'opening.urscript').write_text(IntroProgram(profile.intro,1).program())
    (case/'guidance.urscript').write_text(profile.cfg.program(1))
    result=dict(success=True,ready=True,robot_model=model,execution_backend='official_ursim_native',
        input_mode='deterministic_camera_and_composition_fixture',physical_acceptance=False,
        control_script_shared=True,joint_speed_command_rad_s=.7,nominal_gesture_duration_s=3.3,
        scenario='complete',opening_skipped=False,native_workflow_success=True,cadence_matched=True,
        reference_pose=list(profile.cfg.neutral),reference_joints=list(profile.cfg.joints),
        runtime_source_sha256=manifest['shared_runtime_source_sha256'],
        cleanup_complete=True,backend_thread_alive=False,cleanup_errors=[],composition_requests=1,
        photo_ready=True,stationary_after_stop=True,
        production_ui_replay=dict(success=True,directions=sorted(profile.paths),observed_photo_rows=1),
        lit_reference_hold_samples={cue:10 for cue in profile.paths},
        directions={cue:dict(excursions=2,peak_fractions=[1.,1.],max_joint_speed_rad_s=.6,
            max_displacement_fraction=1.,return_error_m=.0001,cadence_matched=True,
            measured_cycle_s=4.5,measured_moving_s=3.9,controller_cycle_s=4.5,controller_moving_s=3.9,
            cadence_clock='controller',simulator_clock_ratio=1.,action_id=i+1,reference_session_bound=True)
            for i,cue in enumerate(sorted(profile.paths))})
    result['program_sha256']={name:hashlib.sha256((case/name).read_bytes()).hexdigest()
                             for name in ('opening.urscript','guidance.urscript')}
    result['profile_sha256']=hashlib.sha256((case/'profile.json').read_bytes()).hexdigest()
    (case/'result.json').write_text(json.dumps(result))
    for name in ('telemetry.json','ui-state.json','cycles.json'):(case/name).write_text('[]')
    (case/'participant.jsonl').write_text('{"token":"must-not-export","control_instance":"private"}\n')
    return case,result


@pytest.mark.parametrize('model',['ur5e','ur10e'])
def test_accepted_native_case_attaches_actual_programs_and_bounded_evidence(tmp_path,model):
    from leica_tracker.canonical_delivery import attach_native_validation
    root=workspace(tmp_path);output=root/'bundle';manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest,model)
    info=attach_native_validation(output,model,case,workspace=root)
    destination=output/'simulation'/model
    assert info['ready'] is True and info['physical_acceptance'] is False
    assert (destination/'opening.urscript').read_text().endswith((case/'opening.urscript').read_text())
    assert (destination/'guidance.urscript').read_text().endswith((case/'guidance.urscript').read_text())
    assert 'SIMULATION_ONLY' in (destination/'guidance.urscript').read_text()
    summary=json.loads((destination/'validation-summary.json').read_text())
    assert summary['production_ui_replay']['success'] is True
    assert summary['runtime_source_sha256']==manifest['shared_runtime_source_sha256']
    assert 'must-not-export' not in ''.join(p.read_text() for p in destination.iterdir())
    assert not (destination/'participant.jsonl').exists()
    updated=json.loads((output/'manifest.json').read_text())
    assert updated['models'][model]['simulation_ready'] is True
    assert updated['models'][model]['physical_ready'] is False
    assert updated['simulation_exports'][model]['validation']['ready'] is True
    with pytest.raises(FileExistsError):attach_native_validation(output,model,case,workspace=root)


@pytest.mark.parametrize('change',[
    {'runtime_source_sha256':'0'*64},{'robot_model':'ur5e'},{'cadence_matched':False},
    {'native_workflow_success':False},{'physical_acceptance':True},{'scenario':'stop'},
    {'cleanup_complete':False},{'backend_thread_alive':True},{'composition_requests':2},
    {'production_ui_replay':{'success':False}},
])
def test_native_attachment_rejects_unaccepted_or_mismatched_case_before_writing(tmp_path,change):
    from leica_tracker.canonical_delivery import attach_native_validation
    root=workspace(tmp_path);output=root/'bundle';manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest);result.update(change)
    (case/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):attach_native_validation(output,'ur10e',case,workspace=root)
    assert not (output/'simulation').exists()


def test_native_attachment_rejects_profile_or_program_drift(tmp_path):
    from leica_tracker.canonical_delivery import attach_native_validation
    root=workspace(tmp_path);output=root/'bundle';manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest)
    profile=json.loads((case/'profile.json').read_text());profile['installation']['base_rate']=.6
    (case/'profile.json').write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='dynamics'):attach_native_validation(output,'ur10e',case,workspace=root)
    assert not (output/'simulation').exists()


def test_native_attachment_accepts_controller_cadence_and_preserves_actual_wall_time(tmp_path):
    from leica_tracker.canonical_delivery import attach_native_validation
    root=workspace(tmp_path);output=root/'bundle';manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest)
    for item in result['directions'].values():
        item.update(measured_moving_s=4.8,measured_cycle_s=6.,controller_moving_s=3.9,
            controller_cycle_s=4.9,simulator_clock_ratio=4.9/6.)
    (case/'result.json').write_text(json.dumps(result))
    attach_native_validation(output,'ur10e',case,workspace=root)
    summary=json.loads((output/'simulation/ur10e/validation-summary.json').read_text())
    metric=summary['directions']['MOVE_LEFT']
    assert metric['cadence_clock']=='controller'
    assert metric['measured_moving_s']==4.8 and metric['measured_cycle_s']==6.
    assert metric['controller_moving_s']==3.9 and metric['controller_cycle_s']==4.9
    assert metric['simulator_clock_ratio']==4.9/6.


@pytest.mark.parametrize('change',[
    {'controller_moving_s':None},{'controller_cycle_s':float('nan')},
    {'controller_moving_s':4.2},{'controller_cycle_s':5.4},
    {'cadence_clock':'wall'},{'simulator_clock_ratio':0},
])
def test_native_attachment_rejects_missing_or_invalid_controller_clock(tmp_path,change):
    from leica_tracker.canonical_delivery import attach_native_validation
    root=workspace(tmp_path);output=root/'bundle';manifest=build(sources(),output,workspace=root)
    case,result=native_case(root,manifest);result['directions']['MOVE_LEFT'].update(change)
    (case/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):attach_native_validation(output,'ur10e',case,workspace=root)
    assert not (output/'simulation').exists()


def test_generated_model_entries_and_profiles_are_bound_before_source_acceptance(tmp_path):
    from leica_tracker.canonical_delivery import verify_runtime_source
    root=workspace(tmp_path);output=root/'bundle';manifest=build(sources(),output,workspace=root)
    expected={f'start-{model}.ps1' for model in ('ur5e','ur10e')}|{
        f'profiles/{model}.json' for model in ('ur5e','ur10e')}
    assert set(manifest['generated_files'])==expected
    assert all(manifest['generated_files'][name]==hashlib.sha256((output/name).read_bytes()).hexdigest()
               for name in expected)
    assert verify_runtime_source(output,root) is True
    (output/'start-ur5e.ps1').write_text('# changed generated entry\n')
    with pytest.raises(ValueError,match='generated'):verify_runtime_source(output,root)


@pytest.mark.parametrize('failure',['missing_binding','extra_binding','profile_drift','entry_path'])
def test_generated_file_verifier_rejects_incomplete_or_changed_builder_outputs(tmp_path,failure):
    from leica_tracker.canonical_delivery import verify_runtime_source
    root=workspace(tmp_path);output=root/'bundle';build(sources(),output,workspace=root)
    path=output/'manifest.json';manifest=json.loads(path.read_text())
    if failure=='missing_binding':manifest.pop('generated_files',None)
    elif failure=='extra_binding':manifest.setdefault('generated_files',{})['../outside.ps1']='0'*64
    elif failure=='profile_drift':(output/'profiles/ur10e.json').write_text('{}')
    else:manifest['models']['ur5e']['entry']='start-ur10e.ps1'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='generated'):verify_runtime_source(output,root)


def accepted_delivery(root):
    from leica_tracker.canonical_delivery import attach_native_validation
    output=root/'bundle';manifest=build(sources(),output,workspace=root)
    for model in ('ur5e','ur10e'):
        case,_=native_case(root,manifest,model)
        attach_native_validation(output,model,case,workspace=root)
    return output


def test_publish_only_after_both_exact_source_native_attachments_and_replace_atomically(tmp_path,monkeypatch):
    from leica_tracker import canonical_delivery as module
    root=workspace(tmp_path);output=accepted_delivery(root)
    pointer=root/'tmp/canonical-delivery-current.json';pointer.parent.mkdir();pointer.write_text('old evidence')
    replaced=[];original_replace=module.os.replace
    def replace(source,destination):
        assert Path(source).parent==pointer.parent and Path(source)!=pointer
        assert pointer.read_text()=='old evidence'
        replaced.append((Path(source),Path(destination)))
        return original_replace(source,destination)
    monkeypatch.setattr(module.os,'replace',replace)
    published=module.publish_canonical_delivery(output,workspace=root)
    assert len(replaced)==1 and replaced[0][1]==pointer
    assert json.loads(pointer.read_text())==published
    assert published['bundle_path']=='bundle'
    assert published['joint_speed_rad_s']==.7 and published['models']==['ur5e','ur10e']
    assert published['physical_acceptance']=='pending'
    assert published['shared_runtime_source_sha256']==json.loads((output/'manifest.json').read_text())['shared_runtime_source_sha256']
    assert not replaced[0][0].exists()


@pytest.mark.parametrize('failure',[
    'one_model','runtime_bundle','runtime_workspace','generated_entry','generated_profile',
    'native_result','native_telemetry','attached_summary','attached_reference','attached_program','design',
])
def test_publisher_preserves_existing_pointer_on_any_unaccepted_or_changed_evidence(tmp_path,failure):
    from leica_tracker import canonical_delivery as module
    root=workspace(tmp_path);output=accepted_delivery(root)
    pointer=root/'tmp/canonical-delivery-current.json';pointer.parent.mkdir();pointer.write_text('old evidence')
    manifest_path=output/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if failure=='one_model':manifest['models']['ur5e']['simulation_ready']=False
    elif failure=='runtime_bundle':(output/'runtime/vision_mvp/src/leica_tracker/__init__.py').write_text('# changed')
    elif failure=='runtime_workspace':(root/'vision_mvp/src/leica_tracker/__init__.py').write_text('# changed')
    elif failure=='generated_entry':(output/'start-ur5e.ps1').write_text('# changed')
    elif failure=='generated_profile':(output/'profiles/ur10e.json').write_text('{}')
    elif failure=='native_result':(root/'native-ur5e/result.json').write_text('{}')
    elif failure=='native_telemetry':(root/'native-ur5e/telemetry.json').write_text('[{"changed":true}]')
    elif failure=='attached_summary':(output/'simulation/ur5e/validation-summary.json').write_text('{}')
    elif failure=='attached_reference':(output/'simulation/ur5e/reference.json').write_text('{}')
    elif failure=='attached_program':(output/'simulation/ur5e/guidance.urscript').write_text('# changed')
    else:manifest['joint_speed_rad_s']=.6
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):module.publish_canonical_delivery(output,workspace=root)
    assert pointer.read_text()=='old evidence'
    assert sorted(path.name for path in pointer.parent.iterdir())==[pointer.name]
