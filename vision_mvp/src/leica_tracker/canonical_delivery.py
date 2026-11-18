"""按机型离线打包共享运行环境和动作方案。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

from .motion_configuration import configure_motion_design
from .physical_robot_model import model_for_profile
from .cue_audio import CUE_FILES

ROOT=Path(__file__).resolve().parents[3]
MODELS=('ur5e','ur10e')
_SOURCE_SUFFIXES={'.py','.html','.js','.cjs','.css','.svg','.md','.ps1','.cmd','.toml','.lock'}
_CUE_AUDIO_FILES=frozenset(CUE_FILES.values())|{'manifest.json','PIPER-SOURCE.txt','PIPER-MODEL-CARD.txt'}


def _json(value):
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def _inside(root,path):
    path=Path(path).resolve()
    if path==root or not path.is_relative_to(root):
        raise ValueError('Canonical delivery must stay inside the workspace')
    return path


def _runtime_sources(workspace):
    vision=workspace/'vision_mvp'
    package=vision/'src/leica_tracker'
    if not package.is_dir():raise ValueError('Shared runtime source package is missing')
    files=[]
    for base in (package,vision/'scripts'):
        if base.is_dir():
            for path in sorted(base.rglob('*')):
                if (path.is_file() and not any(part.startswith('.') or part=='__pycache__' for part in path.relative_to(base).parts)
                        and (path.suffix in _SOURCE_SUFFIXES or path.name=='LICENSE'
                             or path.parent==package/'cue_audio' and path.name in _CUE_AUDIO_FILES)):
                    if not path.resolve().is_relative_to(workspace):
                        raise ValueError('Runtime source link points outside workspace')
                    files.append(path)
    files.extend(path for path in (vision/'pyproject.toml',vision/'uv.lock',vision/'README.md') if path.is_file())
    return sorted(set(files))


def _source_identity(hashes):
    hashes=dict(sorted(hashes.items()))
    return {'runtime_files':hashes,
            'shared_runtime_source_sha256':hashlib.sha256(_json(hashes).encode()).hexdigest()}


def runtime_source_identity(workspace=ROOT):
    """对离线构建器实际打包的每个源码文件计算哈希。"""
    workspace=Path(workspace).resolve()
    return _source_identity({'runtime/'+source.relative_to(workspace).as_posix():
        hashlib.sha256(source.read_bytes()).hexdigest() for source in _runtime_sources(workspace)})


def verify_runtime_source(bundle,runtime_root):
    """调用实体机器人入口前，拒绝与清单不符或过期的源码。"""
    bundle=Path(bundle).resolve();runtime_root=Path(runtime_root).resolve()
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8-sig'))
    _verify_generated_files(bundle,manifest)
    hashes=manifest.get('runtime_files')
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Runtime source identity missing')
    for name,expected in hashes.items():
        if (not isinstance(name,str) or not name.startswith('runtime/vision_mvp/')
                or not isinstance(expected,str) or len(expected)!=64):
            raise ValueError('Invalid runtime source manifest')
        path=(runtime_root/name.removeprefix('runtime/')).resolve()
        if not path.is_relative_to(runtime_root) or not path.is_file():
            raise ValueError('Runtime source is incomplete or outside the selected project')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Runtime source differs from the validated canonical build: '+name)
    expected_identity=_source_identity(hashes)['shared_runtime_source_sha256']
    actual_identity=runtime_source_identity(runtime_root)
    if (manifest.get('shared_runtime_source_sha256')!=expected_identity
            or actual_identity['runtime_files']!=hashes
            or actual_identity['shared_runtime_source_sha256']!=expected_identity):
        raise ValueError('Runtime source set differs from the validated canonical build')
    return True


def _verify_generated_files(bundle,manifest):
    """核对 runtime/ 外生成的四个启动或配置文件。"""
    expected={f'start-{model}.ps1' for model in MODELS}|{f'profiles/{model}.json' for model in MODELS}
    hashes=manifest.get('generated_files');models=manifest.get('models')
    if not isinstance(hashes,dict) or set(hashes)!=expected or not isinstance(models,dict) or set(models)!=set(MODELS):
        raise ValueError('Canonical generated file identity is incomplete')
    for model in MODELS:
        item=models[model]
        if (not isinstance(item,dict) or item.get('entry')!=f'start-{model}.ps1'
                or item.get('profile')!=f'profiles/{model}.json'):
            raise ValueError('Canonical generated model entry/profile is not model-bound')
    for name,expected_hash in hashes.items():
        path=(bundle/name).resolve()
        if (not path.is_relative_to(bundle) or not path.is_file()
                or not isinstance(expected_hash,str) or len(expected_hash)!=64
                or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash):
            raise ValueError('Canonical generated file differs from its frozen build: '+name)


def prepare_canonical_delivery(sources,output,*,workspace=ROOT,joint_speed=.7,duration_s=3.3,repetitions=2):
    """离线构建两个机型的方案及完整共享源码。保留各机型自己的安装坐标；
    动力学设置改变后原验收失效，打包时不能借用另一机型的姿态。"""
    workspace=Path(workspace).resolve();output=_inside(workspace,output)
    if output.exists():raise FileExistsError('Preserve the previous canonical delivery; choose a new output')
    if not isinstance(sources,dict) or set(sources)!=set(MODELS):
        raise ValueError('Separate UR5e and UR10e source designs are required')
    profiles={}
    for model in MODELS:
        if model_for_profile(sources[model])!=model:raise ValueError('Source design robot model mismatch')
        profiles[model]=configure_motion_design(sources[model],joint_speed=joint_speed,
                                               duration_s=duration_s,repetitions=repetitions)
    files={}
    for source in _runtime_sources(workspace):
        relative=source.relative_to(workspace)
        files['runtime/'+relative.as_posix()]=source.read_bytes()
    for model,profile in profiles.items():
        files[f'profiles/{model}.json']=_json(profile).encode('utf-8')
        entry=f'''# Model-bound canonical runtime entry. Uses this existing project's environment.
param([string]$RuntimeRoot = '{str(workspace).replace("'","''")}',
      [string]$Profile = (Join-Path $PSScriptRoot 'profiles/{model}.json'),
      [string]$GuestSession = '')
$ErrorActionPreference='Stop'
$python=Join-Path $RuntimeRoot 'vision_mvp/.venv/Scripts/python.exe'
$verifier=Join-Path $PSScriptRoot 'runtime/vision_mvp/scripts/verify-canonical-runtime.py'
$previousPythonPath=$env:PYTHONPATH
$env:PYTHONPATH=Join-Path $RuntimeRoot 'vision_mvp/src'
try {{
    & $python $verifier --bundle $PSScriptRoot --runtime-root $RuntimeRoot
    if ($LASTEXITCODE -ne 0) {{ throw 'Use the exact validated runtime source before connecting a robot.' }}
    $entry=Join-Path $RuntimeRoot 'vision_mvp/scripts/start-{model}-ipad-system.ps1'
    if ($GuestSession) {{ & $entry -Execute -Profile $Profile -GuestSession $GuestSession }}
    else {{ & $entry -Execute -Profile $Profile }}
    if ($LASTEXITCODE -ne 0) {{ throw 'The model-bound runtime entry failed; inspect its preflight output.' }}
}} finally {{ $env:PYTHONPATH=$previousPythonPath }}
'''
        files[f'start-{model}.ps1']=entry.encode('utf-8')
    identity=_source_identity({name:hashlib.sha256(value).hexdigest() for name,value in files.items() if name.startswith('runtime/')})
    runtime_hashes=identity['runtime_files'];source_hash=identity['shared_runtime_source_sha256']
    manifest={
        'format_version':1,'workflow':'ONE_SHARED_NATIVE_RUNTIME','preparation':'OFFLINE_NO_DEVICE_ACCESS',
        'joint_speed_rad_s':float(joint_speed),'joint_acceleration_rad_s2':1.2,
        'nominal_gesture_duration_s':float(duration_s),
        'gesture_repetitions':repetitions,'physical_acceptance':'pending',
        'shared_runtime_source_sha256':source_hash,
        'models':{model:{'profile':f'profiles/{model}.json','entry':f'start-{model}.ps1',
                         'physical_installation_configured':(profiles[model]['installation']['reference'] is not None
                             and profiles[model].get('approval_mode')!='operator_selected_design'),
                         'physical_ready':False,
                         **({'runtime_reference_capture':'on_explicit_start',
                             'approval_mode':'operator_selected_design'}
                            if profiles[model].get('approval_mode')=='operator_selected_design' else {})}
                    for model in MODELS},
        'runtime_files':runtime_hashes,'simulation_exports':{},
        'generated_files':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()
                           if not name.startswith('runtime/')},
    }
    files['README.md']=f'''# Shared canonical UR5e and UR10e runtime

Both model entries use the same packaged Python, URScript generators, native
finite gesture protocol, perception pause/resume and measured-motion UI code.
Global joint limit: {joint_speed:g} rad/s. Opening acceleration: 1.2 rad/s².
Nominal direction cadence: {duration_s:g} seconds, {repetitions} smooth sweeps.
Cartesian speed/acceleration are derived from displacement and cadence; they
are not set to 0.7 m/s. Native IK joint-rate limiting may extend actual time.

`runtime/` and `shared-runtime.zip` contain the complete common source and
model entry scripts. They contain no Python environment, API keys, pairing
secrets or camera media. Reuse the existing project's Python environment,
Claude configuration, HTTPS certificates and paired iPad page.

`profiles/ur5e.json` and `profiles/ur10e.json` select the same canonical motion
settings. Existing model-specific geometry/TCP/payload/path values are kept;
unknown UR5e installation values remain unknown. Changed dynamics invalidate
historical installation/dynamics acceptance. No old 0.1 baseline is selected
by these entries and no simulated pose is substituted for a physical setup.

The generated `start-ur5e.ps1` / `start-ur10e.ps1` launch the normal shared
runtime with that model's profile. Live preflight requires the matching model
and fresh actual state. Inputs explicitly distinguish a historical physical
review from an operator-selected design; the latter is not a claim that a
physical run has already been accepted.
When the installation-specific profile is ready, pass it through `-Profile`;
`-RuntimeRoot` selects the existing repository if this bundle has been moved.
Startup pins Python imports to that selected source tree, so an older editable
environment cannot silently load a different checkout.

Official URSim program exports and measured results, when attached, appear
under `simulation/<model>/`. Their headers state SIMULATION_ONLY. They are
inspection exports of the same generator with RTDE session1, not standalone
physical iPad/AI programs. Normal runtime generates a fresh session itself.

Shared source identity: `{source_hash}`.
'''.encode('utf-8')
    if any(profile.get('approval_mode')=='operator_selected_design' for profile in profiles.values()):
        files['README.md']+=b'''
## Operator-selected direct entry

Selected profiles record the user's motion design separately from a completed
physical trial. They require no historical 14-point preview or small trial.
Both models retain their own geometry and default to the selected joint angles.
On explicit iPad Start the shared runtime reads independent fresh stationary
controller state and binds the design to the current operator-positioned pose
and current active TCP. Manual adjustment is therefore captured on the next
Start. Merely connecting does not move or home the arm. Stop, model/state checks,
command/feedback deadlines and native controller protections remain effective.
No prior physical acceptance, payload number or controller setting is invented.
'''
    output.mkdir(parents=True,exist_ok=False)
    for name,value in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    with zipfile.ZipFile(output/'shared-runtime.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in files.items():
            if name.startswith('runtime/'):archive.writestr(name.removeprefix('runtime/'),value)
    (output/'manifest.json').write_text(_json(manifest),encoding='utf-8')
    return manifest


def attach_simulation_exports(output,robot_model,profile,*,workspace=ROOT):
    """附上来自实际仿真配置的程序，不生成虚构的审核记录。"""
    from .physical_intro_program import IntroProgram
    from .ursim_profile import SimulationProfile
    from .ursim_transport import SIM_HOST
    workspace=Path(workspace).resolve();output=_inside(workspace,output)
    if (robot_model not in MODELS or type(profile) is not SimulationProfile or profile.host!=SIM_HOST
            or profile.cfg.robot_model!=robot_model or profile.cfg.gesture is None):
        raise ValueError('An explicitly matched canonical SimulationProfile is required')
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if robot_model not in manifest['models']:raise ValueError('Model not present in this bundle')
    cadence=profile.cfg.gesture.as_dict()
    if (profile.intro['joint_speed']!=manifest['joint_speed_rad_s']
            or profile.intro['joint_acceleration']!=manifest['joint_acceleration_rad_s2']
            or profile.cfg.base_rate!=manifest['joint_speed_rad_s']
            or cadence['joint_speed_limit']!=manifest['joint_speed_rad_s']
            or cadence['duration_s']!=manifest['nominal_gesture_duration_s']
            or cadence['repetitions']!=manifest['gesture_repetitions']):
        raise ValueError('Simulation dynamics do not match the canonical bundle')
    destination=output/'simulation'/robot_model
    if destination.exists():raise FileExistsError('Preserve existing simulation exports')
    header='# SIMULATION_ONLY / physical acceptance pending / inspection RTDE session1\n'
    exports={'opening.urscript':header+IntroProgram(profile.intro,1).program(),
             'guidance.urscript':header+profile.cfg.program(1)}
    info={'simulation_only':True,'physical_acceptance':False,'robot_model':robot_model,
          'reference_pose':list(profile.cfg.neutral),'reference_joints':list(profile.cfg.joints),
          'active_tcp':list(profile.cfg.tcp),'joint_speed_rad_s':profile.intro['joint_speed'],
          'gesture':profile.cfg.gesture.as_dict(),
          'program_sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in exports.items()}}
    destination.mkdir(parents=True,exist_ok=False)
    for name,value in exports.items():(destination/name).write_text(value,encoding='utf-8')
    (destination/'reference.json').write_text(_json(info),encoding='utf-8')
    manifest['simulation_exports'][robot_model]={'path':f'simulation/{robot_model}',**info}
    (output/'manifest.json').write_text(_json(manifest),encoding='utf-8')
    return info


def _case_bytes(workspace,case,name,*,limit=8*1024*1024):
    path=_inside(workspace,case/name)
    if not path.is_relative_to(case) or not path.is_file() or path.stat().st_size>limit:
        raise ValueError('Native validation artifact is missing, too large or outside its case')
    return path.read_bytes()


def _finite_vector(value,size):
    return (isinstance(value,(list,tuple)) and len(value)==size
            and all(type(v) in (int,float) and math.isfinite(v) for v in value))


def _native_result(result,model,manifest):
    recovery=None
    if isinstance(result,dict) and result.get('communication_jitter_ms'):
        from .ursim_validation import validate_communication_recovery_proof
        validate_communication_recovery_proof(result)
        recovery=result['fault_injection']['stages'][1]
    expected_true=('success','ready','native_workflow_success','cadence_matched','cleanup_complete',
                   'control_script_shared','photo_ready','stationary_after_stop')
    if (not isinstance(result,dict) or any(result.get(key) is not True for key in expected_true)
            or result.get('backend_thread_alive') is not False or result.get('physical_acceptance') is not False
            or result.get('cleanup_errors')!=[] or result.get('scenario')!='complete'
            or result.get('opening_skipped') is not False or result.get('robot_model')!=model
            or result.get('execution_backend')!='official_ursim_native'
            or result.get('input_mode')!='deterministic_camera_and_composition_fixture'
            or type(result.get('composition_requests')) is not int or result['composition_requests']!=1
            or result.get('joint_speed_command_rad_s')!=manifest['joint_speed_rad_s']
            or result.get('nominal_gesture_duration_s')!=manifest['nominal_gesture_duration_s']):
        raise ValueError('Native case is not a complete accepted official-controller workflow')
    if result.get('runtime_source_sha256')!=manifest['shared_runtime_source_sha256']:
        raise ValueError('Native case source identity differs from the frozen bundle')
    cues={'MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER'}
    replay=result.get('production_ui_replay');directions=result.get('directions');lights=result.get('lit_reference_hold_samples')
    if (not isinstance(replay,dict) or replay.get('success') is not True
            or not isinstance(replay.get('directions'),list) or set(replay['directions'])!=cues
            or type(replay.get('observed_photo_rows')) is not int or replay['observed_photo_rows']<1
            or not isinstance(directions,dict) or set(directions)!=cues
            or not isinstance(lights,dict) or set(lights)!=cues
            or any(type(n) is not int or n<1 for n in lights.values())):
        raise ValueError('Native case lacks complete measured direction and production light evidence')
    actions=[]
    for cue,item in directions.items():
        if not isinstance(item,dict):raise ValueError('Native direction metrics must be records')
        peaks=item.get('peak_fractions') if isinstance(item,dict) else None
        numeric=('max_joint_speed_rad_s','return_error_m','measured_cycle_s','measured_moving_s',
                 'controller_cycle_s','controller_moving_s','simulator_clock_ratio')
        moving=item.get('controller_moving_s') if isinstance(item,dict) else None
        cycle=item.get('controller_cycle_s') if isinstance(item,dict) else None
        hold_keys=('native_hold_s','controller_active_cycle_s','controller_active_moving_s')
        claimed_hold=isinstance(item,dict) and any(key in item for key in hold_keys)
        if recovery is not None and cue=='MOVE_LEFT':
            if (not all(type(item.get(key)) in (int,float) and math.isfinite(item[key]) for key in hold_keys)
                    or item.get('action_id')!=recovery['action_id']
                    or not math.isclose(item['native_hold_s'],recovery['native_hold_s'],rel_tol=1e-9,abs_tol=1e-9)
                    or not all(type(item.get(key)) in (int,float) and math.isfinite(item[key])
                               for key in ('controller_cycle_s','controller_moving_s'))
                    or not 0<item['native_hold_s']<item['controller_moving_s']<=item['controller_cycle_s']
                    or not math.isclose(item['controller_active_cycle_s'],item['controller_cycle_s']-item['native_hold_s'],rel_tol=1e-9,abs_tol=1e-9)
                    or not math.isclose(item['controller_active_moving_s'],item['controller_moving_s']-item['native_hold_s'],rel_tol=1e-9,abs_tol=1e-9)):
                raise ValueError('Native cadence exclusion lacks consistent measured recovery HOLD')
            moving=item['controller_active_moving_s'];cycle=item['controller_active_cycle_s']
        elif claimed_hold:raise ValueError('Unproven native HOLD cannot waive cadence')
        if (not isinstance(item,dict) or item.get('excursions')!=2 or not _finite_vector(peaks,2)
                or any(v<.9 for v in peaks) or item.get('cadence_matched') is not True
                or item.get('reference_session_bound') is not True
                or type(item.get('action_id')) is not int or item['action_id']<=0
                or any(type(item.get(k)) not in (int,float) or not math.isfinite(item[k]) for k in numeric)
                or not 0<=item['max_joint_speed_rad_s']<=manifest['joint_speed_rad_s']*1.05
                or not 0<=item['return_error_m']<=.0005
                or item.get('cadence_clock')!='controller'
                or not 0<moving<=4.1 or not 0<cycle<=5.3
                or not 0<item['measured_moving_s']<=item['measured_cycle_s']
                or item['controller_moving_s']>item['controller_cycle_s']
                or item['simulator_clock_ratio']<=0
                or not math.isclose(item['simulator_clock_ratio'],
                    item['controller_cycle_s']/item['measured_cycle_s'],rel_tol=1e-9,abs_tol=1e-9)):
            raise ValueError('Native direction metrics do not prove the accepted cadence and return')
        actions.append(item['action_id'])
    if len(set(actions))!=6:raise ValueError('Native direction action identities are not distinct')


def _native_profile(record,result,model,manifest):
    if not isinstance(record,dict) or record.get('model')!=model or record.get('simulation_only') is not True:
        raise ValueError('Native validation profile is not model-bound simulation data')
    installation=record.get('installation');intro=record.get('intro')
    if not isinstance(installation,dict) or not isinstance(intro,dict):raise ValueError('Native validation profile is incomplete')
    gesture=installation.get('gesture')
    if (installation.get('robot_model')!=model or intro.get('robot_model')!=model
            or intro.get('simulation_only') is not True or intro.get('provenance')!='SIMULATION_ONLY'
            or installation.get('base_rate')!=manifest['joint_speed_rad_s']
            or intro.get('joint_speed')!=manifest['joint_speed_rad_s']
            or intro.get('joint_acceleration')!=manifest['joint_acceleration_rad_s2']
            or not isinstance(gesture,dict) or gesture!=dict(duration_s=manifest['nominal_gesture_duration_s'],
                repetitions=manifest['gesture_repetitions'],joint_speed_limit=manifest['joint_speed_rad_s'])):
        raise ValueError('Native validation dynamics do not match the frozen bundle')
    reference=installation.get('reference');joints=installation.get('joints');tcp=installation.get('tcp')
    if (not all(_finite_vector(v,6) for v in (reference,joints,tcp))
            or list(reference)!=result.get('reference_pose') or list(joints)!=result.get('reference_joints')
            or intro.get('reference_pose')!=list(reference) or intro.get('reference_joints')!=list(joints)
            or intro.get('tcp')!=list(tcp)):
        raise ValueError('Native measured reference does not match the executed profile')
    # 只导出数值仿真数据，避免仿真审核字典被误当成实体机器人验收记录。
    return dict(simulation_only=True,physical_acceptance=False,robot_model=model,
        reference_pose=list(reference),reference_joints=list(joints),active_tcp=list(tcp),
        joint_speed_rad_s=intro['joint_speed'],joint_acceleration_rad_s2=intro['joint_acceleration'],gesture=gesture)


def _native_summary(result,artifact_index):
    summary_keys=('success','ready','robot_model','execution_backend','input_mode','physical_acceptance',
        'control_script_shared','joint_speed_command_rad_s','nominal_gesture_duration_s','scenario',
        'opening_skipped','native_workflow_success','cadence_matched','composition_requests','photo_ready',
        'stationary_after_stop','elapsed_s','cleanup_complete','backend_thread_alive','runtime_source_sha256',
        'production_ui_replay','lit_reference_hold_samples')
    summary={key:result[key] for key in summary_keys if key in result}
    metric_keys=('excursions','peak_fractions','max_joint_speed_rad_s','max_displacement_fraction',
        'return_error_m','cadence_matched','cadence_clock','measured_cycle_s','measured_moving_s',
        'controller_cycle_s','controller_moving_s','simulator_clock_ratio','action_id','reference_session_bound',
        'native_hold_s','controller_active_cycle_s','controller_active_moving_s')
    summary['directions']={cue:{key:item[key] for key in metric_keys if key in item}
        for cue,item in result['directions'].items()}
    summary['raw_evidence']=artifact_index
    if result.get('communication_jitter_ms'):
        from .ursim_validation import validate_communication_recovery_proof
        validate_communication_recovery_proof(result)
        summary['bounded_communication_recovery']=dict(requested_gap_ms=result['communication_jitter_ms'],
            stages=[item['stage'] for item in result['fault_injection']['stages']],
            owner_start_count=result['fault_injection']['owner_start_count'],
            reference_binding_count=result['operator_selected_binding_count'],
            stationary_hold_and_return_verified=True,post_return_frame_verified=True)
    return summary


def _native_attachment_content(workspace,case,robot_model,manifest):
    """从不可变的实测样本字节重新构建允许纳入的附件。"""
    result_bytes=_case_bytes(workspace,case,'result.json');result=json.loads(result_bytes)
    _native_result(result,robot_model,manifest)
    profile_bytes=_case_bytes(workspace,case,'profile.json');profile=json.loads(profile_bytes)
    info=_native_profile(profile,result,robot_model,manifest)
    if hashlib.sha256(profile_bytes).hexdigest()!=result.get('profile_sha256'):
        raise ValueError('Native profile differs from its pre-execution record')
    programs={name:_case_bytes(workspace,case,name,limit=1024*1024) for name in ('opening.urscript','guidance.urscript')}
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in programs.items()}
    if hashes!=result.get('program_sha256'):raise ValueError('Native programs differ from their pre-execution record')
    header=b'# SIMULATION_ONLY / measured official URSim validation / inspection RTDE session1\n'
    exports={name:header+value for name,value in programs.items()}
    indexed={'result.json':result_bytes,'profile.json':profile_bytes,**programs}
    for name in ('telemetry.json','ui-state.json','cycles.json'):
        indexed[name]=_case_bytes(workspace,case,name,limit=64*1024*1024)
    artifact_index={name:{'workspace_path':(case/name).relative_to(workspace).as_posix(),
        'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)} for name,value in indexed.items()}
    info.update(ready=True,validation_source='official_ursim_native',input_mode=result['input_mode'],
        cadence_clock='controller',wall_time_reported=True,
        runtime_source_sha256=result['runtime_source_sha256'],
        program_sha256={name:hashlib.sha256(value).hexdigest() for name,value in exports.items()},
        original_program_sha256=hashes)
    return exports,info,_native_summary(result,artifact_index)


def attach_native_validation(output,robot_model,case_dir,*,workspace=ROOT):
    """附上已通过验证的实测导出文件，区分测试夹具与实机结果。
    大型原始轨迹保留在工作区并用路径与哈希索引，只打包允许的测量摘要和程序，不复制参与者日志、相机素材、所有者身份或配对信息，也不据此生成 SimulationProfile 或实机验收。"""
    workspace=Path(workspace).resolve();output=_inside(workspace,output);case=_inside(workspace,case_dir)
    if robot_model not in MODELS or case.is_relative_to(output):raise ValueError('Invalid native validation case location or model')
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if robot_model not in manifest.get('models',{}):raise ValueError('Model not present in this bundle')
    if _source_identity(manifest.get('runtime_files',{}))['shared_runtime_source_sha256']!=manifest.get('shared_runtime_source_sha256'):
        raise ValueError('Frozen bundle source identity is inconsistent')
    verify_runtime_source(output,output/'runtime')
    destination=output/'simulation'/robot_model
    if destination.exists():raise FileExistsError('Preserve existing simulation exports')
    exports,info,summary=_native_attachment_content(workspace,case,robot_model,manifest)
    # 以上检查均须在首次写入前完成。
    destination.mkdir(parents=True,exist_ok=False)
    for name,value in exports.items():(destination/name).write_bytes(value)
    (destination/'reference.json').write_text(_json(info),encoding='utf-8')
    (destination/'validation-summary.json').write_text(_json(summary),encoding='utf-8')
    manifest.setdefault('simulation_exports',{})[robot_model]={'path':f'simulation/{robot_model}',**info,
        'validation':{'ready':True,'summary':f'simulation/{robot_model}/validation-summary.json',
                      'workspace_case':case.relative_to(workspace).as_posix()}}
    manifest['models'][robot_model]['simulation_ready']=True
    manifest['models'][robot_model]['physical_ready']=False
    (output/'manifest.json').write_text(_json(manifest),encoding='utf-8')
    return info


def publish_canonical_delivery(output,*,workspace=ROOT):
    """离线校验通过后发布与源码严格匹配的原生构建。发布只选定版本，现场安装仍待验收。
    保留旧包和样例，校验失败时不切换当前版本指针。"""
    workspace=Path(workspace).resolve();output=_inside(workspace,output)
    manifest_path=output/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if (manifest.get('workflow')!='ONE_SHARED_NATIVE_RUNTIME' or manifest.get('joint_speed_rad_s')!=.7
            or manifest.get('joint_acceleration_rad_s2')!=1.2
            or manifest.get('nominal_gesture_duration_s')!=3.3 or manifest.get('gesture_repetitions')!=2
            or manifest.get('physical_acceptance')!='pending'):
        raise ValueError('Publication requires the canonical 0.7 rad/s design with physical acceptance pending')
    verify_runtime_source(output,output/'runtime')
    verify_runtime_source(output,workspace)
    for model in MODELS:
        model_record=manifest['models'][model]
        if model_record.get('simulation_ready') is not True or model_record.get('physical_ready') is not False:
            raise ValueError('Both models require accepted native validation before publication')
        profile=json.loads((output/model_record['profile']).read_text(encoding='utf-8-sig'))
        intro=profile.get('intro',{});installation=profile.get('installation',{});design=profile.get('motion_design',{})
        if (model_for_profile(profile)!=model or intro.get('joint_speed')!=.7 or intro.get('joint_acceleration')!=1.2
                or installation.get('base_rate')!=.7
                or installation.get('gesture')!=dict(duration_s=3.3,repetitions=2,joint_speed_limit=.7)
                or design.get('robot_model')!=model or design.get('physical_acceptance') is not False
                or design.get('joint_speed_rad_s')!=.7 or design.get('duration_s')!=3.3 or design.get('repetitions')!=2):
            raise ValueError('Generated model profile does not match the canonical design')
        attached=manifest.get('simulation_exports',{}).get(model,{})
        validation=attached.get('validation',{})
        expected_path=f'simulation/{model}';summary_path=f'{expected_path}/validation-summary.json'
        case_relative=validation.get('workspace_case')
        if (attached.get('path')!=expected_path or validation.get('ready') is not True
                or validation.get('summary')!=summary_path or not isinstance(case_relative,str)
                or Path(case_relative).is_absolute()):
            raise ValueError('Model is missing its accepted native attachment')
        case=_inside(workspace,workspace/case_relative)
        if case.is_relative_to(output):raise ValueError('Native raw evidence must remain outside the delivery')
        exports,info,summary=_native_attachment_content(workspace,case,model,manifest)
        expected_attachment={'path':expected_path,**info,'validation':{
            'ready':True,'summary':summary_path,'workspace_case':case.relative_to(workspace).as_posix()}}
        if attached!=expected_attachment:
            raise ValueError('Native attachment record differs from the accepted raw evidence')
        if (json.loads((output/expected_path/'reference.json').read_text(encoding='utf-8-sig'))!=info
                or json.loads((output/summary_path).read_text(encoding='utf-8-sig'))!=summary):
            raise ValueError('Native attached reference or summary differs from the accepted raw evidence')
        for name,value in exports.items():
            if (output/expected_path/name).read_bytes()!=value:
                raise ValueError('Native attached program differs from the accepted raw evidence')
    pointer=_inside(workspace,workspace/'tmp/canonical-delivery-current.json')
    pointer.parent.mkdir(parents=True,exist_ok=True)
    published=dict(format_version=1,bundle_path=output.relative_to(workspace).as_posix(),
        shared_runtime_source_sha256=manifest['shared_runtime_source_sha256'],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        models=list(MODELS),joint_speed_rad_s=.7,physical_acceptance='pending')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=pointer.parent,
                prefix='.canonical-delivery-current-',suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name);handle.write(_json(published));handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,pointer)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return published


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ur10e-profile',type=Path,default=ROOT/'tmp/robot-network/physical-ipad-ur10e-trial.json')
    parser.add_argument('--ur5e-profile',type=Path,default=ROOT/'vision_mvp/config/physical-ur5e.template.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--joint-speed',type=float,default=.7)
    args=parser.parse_args(argv)
    try:
        sources={model:json.loads(_inside(ROOT,getattr(args,model+'_profile')).read_text(encoding='utf-8-sig')) for model in MODELS}
        output=args.output or ROOT/'tmp/canonical-delivery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        result=prepare_canonical_delivery(sources,output,joint_speed=args.joint_speed)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    print(_json({'output':str(Path(output).resolve()),'joint_speed_rad_s':result['joint_speed_rad_s'],
                 'models':list(result['models']),'shared_runtime_source_sha256':result['shared_runtime_source_sha256'],
                 'physical_acceptance':result['physical_acceptance']}))
    return 0
