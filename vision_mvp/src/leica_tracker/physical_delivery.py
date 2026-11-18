"""按机型离线准备实体控制文件，不连接或启动设备。"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .physical_intro_program import IntroProgram
from .physical_robot_model import SUPPORTED_MODELS, model_for_profile
from .physical_runtime import reviewed_profile
from .physical_speed_audit import audit_intro_speed


ROOT=Path(__file__).resolve().parents[3]


def _json(value):
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def _ps_quote(value):
    return "'"+str(value).replace("'","''")+"'"


def prepare_delivery(data,output,*,workspace=ROOT,target_joint_speed=.6,target_joint_acceleration=1.2,
                     disable_target_angular_monitor=False,disable_extra_motion_monitors=False,
                     robot_model=None):
    """编译已审核的基线配置；请求调整动力学后，新参数仍待审核。
    生成的 URScript 供检查，实际启动沿用现有后端，由后端创建会话并检查实时状态。
    不为提速要求伪造审核标志或测试依据。"""
    workspace=Path(workspace).resolve();output=Path(output).resolve()
    if not output.is_relative_to(workspace) or output==workspace:
        raise ValueError('Delivery output must be a new directory inside the workspace')
    if output.exists():raise FileExistsError('Delivery output already exists; preserve the previous bundle')
    baseline=copy.deepcopy(data)
    actual_model=model_for_profile(baseline)
    if robot_model is not None:
        if robot_model not in SUPPORTED_MODELS:raise ValueError('Unsupported expected robot model')
        if actual_model!=robot_model:
            raise ValueError(f'Profile robot model {actual_model} does not match expected model {robot_model}')
    model_label=actual_model[:2].upper()+actual_model[2:]
    reviewed=reviewed_profile(baseline)
    if type(disable_target_angular_monitor) is not bool or type(disable_extra_motion_monitors) is not bool:
        raise ValueError('Explicit boolean target angular monitor selection required')
    prior_audit=audit_intro_speed(baseline['intro'],target_joint_speed,target_joint_acceleration)
    opening=IntroProgram(reviewed.intro,1).program()
    guidance=reviewed.cfg.program(1)
    draft=copy.deepcopy(baseline)
    draft['status']='DRAFT_TARGET_DYNAMICS_NOT_EXECUTABLE'
    draft['intro']['joint_speed']=float(target_joint_speed)
    draft['intro']['joint_acceleration']=float(target_joint_acceleration)
    if disable_target_angular_monitor:draft['intro']['max_tcp_angular_speed']=None
    if disable_extra_motion_monitors:
        for key in ('max_tcp_linear_speed','max_tcp_angular_speed'):
            draft['intro'][key]=None;draft['installation'][key]=None
        draft['installation']['review']={key:False for key in draft['installation']['review']}
        draft['reviewed_installation']={}
    draft['intro']['reviewed']={}
    draft['intro']['review']={key:False for key in draft['intro']['review']}
    # 历史验收只属于原基线，不能通过另一入口将其当作新动力学配置的证据。
    draft.pop('intro_preview',None)
    draft['target_note']='User-requested simulator dynamics goal; new physical dynamics/path/Stop acceptance is pending.'
    audit=audit_intro_speed(draft['intro'],target_joint_speed,target_joint_acceleration)
    audit['previous_project_monitor_conflicts']=prior_audit['known_angular_conflicts']
    manifest={
        'format_version':1,'robot_model':actual_model,'preparation':'OFFLINE_NO_DEVICE_ACCESS',
        'physical_acceptance':'pending','profile_sha256':hashlib.sha256(_json(baseline).encode()).hexdigest(),
        'baseline':{'review_record_valid':True,'joint_speed_rad_s':reviewed.intro['joint_speed'],
                    'joint_acceleration_rad_s2':reviewed.intro['joint_acceleration'],
                    'max_tcp_linear_speed_m_s':reviewed.intro['max_tcp_linear_speed'],
                    'max_tcp_angular_speed_rad_s':reviewed.intro['max_tcp_angular_speed'],
                    'guidance_linear_speed_m_s':baseline['installation']['speed'],
                    'guidance_acceleration_m_s2':baseline['installation']['acceleration']},
        'target':{'scope':'opening_joint_dynamics_only_not_xyz_guidance',
                  'project_tcp_angular_monitor_enabled':draft['intro']['max_tcp_angular_speed'] is not None,
                  'project_tcp_linear_monitor_enabled':draft['intro']['max_tcp_linear_speed'] is not None,
                  'guidance_extra_speed_monitors_disabled':disable_extra_motion_monitors,
                  'joint_speed_rad_s':float(target_joint_speed),
                  'joint_acceleration_rad_s2':float(target_joint_acceleration),
                  'executable':False,'known_angular_conflict_segments':[x['segment_index'] for x in audit['known_angular_conflicts']],
                  'review_required':True},
        'robot_program_exports':'inspection_only_session_1_not_a_standalone_ipad_controller',
        'files':['baseline-profile.json','opening-reviewed.urscript','guidance-reviewed.urscript',
                 'target-speed-DRAFT.json','target-speed-audit.json','start-reviewed-ipad.ps1','README.md'],
    }
    startup=("# Onsite only. Runs the reviewed baseline, NOT the desired-speed draft.\n"
        "param([string]$GuestSession = "+_ps_quote(workspace/'tmp/robot-network/ipad-trial-pairing.json')+")\n"
        "$ErrorActionPreference='Stop'\n"
        "& "+_ps_quote(workspace/f'vision_mvp/scripts/start-{actual_model}-ipad-system.ps1')+
        " -Execute -Profile (Join-Path $PSScriptRoot 'baseline-profile.json') -GuestSession $GuestSession\n")
    readme=f"""# Complete {model_label} + iPad control delivery

This bundle was prepared offline. No robot state was measured and no process was started.
Physical acceptance of the updated software is pending.
The profile and startup entry are bound to {model_label}; they do not transfer installation review to another model.

## Contents and use

- `baseline-profile.json`: the exact existing reviewed record, preserved without changing dynamics.
- `opening-reviewed.urscript`: complete opening robot program (14 `movej` points).
- `guidance-reviewed.urscript`: complete XYZ resident robot program.
- The URScript exports use inspection session 1. Do not upload them manually: they require
  the RTDE host protocol. They are not a standalone iPad/AI app. The existing Python service
  regenerates fresh-session programs, manages Start/Stop/leases, verifies actual reference,
  and transitions to the camera/AI workflow.
- `target-speed-DRAFT.json`: requested {target_joint_speed:g} rad/s and {target_joint_acceleration:g} rad/s²;
  deliberately has no dynamics review and is rejected by all normal physical execution gates.
- `target-speed-audit.json`: ideal stop-to-stop timing and isolated-joint angular conflicts only.
  It does not certify multi-joint TCP speed, real scaling, collision clearance or physical Stop.

The project-added angular monitor is {'disabled explicitly (null)' if draft['intro']['max_tcp_angular_speed'] is None else 'retained'} in the target draft only.
It is an application monitor, not the robot's native speed limit. Native safety settings,
joint/trajectory tracking and control-loss stops are not disabled.
The extra intro linear and guidance TCP speed monitors are {'explicitly disabled in this unreviewed draft' if disable_extra_motion_monitors else 'retained'}.
Reference/target validity, feedback freshness, Stop and reviewed path geometry are retained.
Previous-project-monitor conflicts are historical comparisons, not machine-capability limits.

## Onsite entry

Use the same repository and its existing Python environment and pairing.
After the onsite operator verifies actual state, deliberately stop any older controller process
and start the complete updated system using `start-reviewed-ipad.ps1`.
The full-stack entry owns fresh HTTPS and physical/camera services and refuses occupied ports,
so it cannot silently reuse an old proxy or controller. It does not kill unrelated processes.
Refresh the original paired iPad page, prepare its camera, then **Start on the iPad**.
Do not invoke the startup script from home. It does not automatically home the arm.
No Codex turn, terminal Next command or per-direction operator prompt is part of the normal
encounter: the backend performs its own live checks, opening, guidance and photo coordination.

Baseline opening: {reviewed.intro['joint_speed']:g} rad/s, {reviewed.intro['joint_acceleration']:g} rad/s².
Baseline directional guidance: {baseline['installation']['speed']:g} m/s.
These are different controllers and units; changing opening speed does not speed up XYZ guidance.
Desired simulator dynamics are a separate goal and are not silently applied to the baseline.
Real loss of control, stale telemetry or overspeed still stops the arm.

No API credential files, pairing files or camera media are copied into this bundle.
The local installation profile is installation-specific; use the separate issue report/email
for sharing rather than treating this profile as a portable calibration.
"""
    files={
        'baseline-profile.json':_json(baseline),'target-speed-DRAFT.json':_json(draft),
        'opening-reviewed.urscript':'# Inspection export only; runtime generates a fresh RTDE session.\n'+opening,
        'guidance-reviewed.urscript':'# Inspection export only; runtime generates a fresh RTDE session.\n'+guidance,
        'target-speed-audit.json':_json(audit),'manifest.json':_json(manifest),
        'start-reviewed-ipad.ps1':startup,'README.md':readme,
    }
    # 所有校验和编译均在创建文件之前完成。
    output.mkdir(parents=True,exist_ok=False)
    for name,text in files.items():(output/name).write_text(text,encoding='utf-8')
    return manifest


def main(argv=None, *, default_model='ur10e', default_profile=None, default_output=None):
    parser=argparse.ArgumentParser(description=__doc__)
    if default_model not in SUPPORTED_MODELS:raise ValueError('Unsupported entry robot model')
    profile_default=default_profile or ROOT/'tmp/robot-network/physical-ipad-ur10e-trial.json'
    parser.add_argument('--profile',type=Path,default=profile_default)
    parser.add_argument('--robot-model',choices=SUPPORTED_MODELS,default=default_model,
                        help='Expected profile identity; dedicated entry identity cannot be changed')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--target-joint-speed',type=float,default=.6)
    parser.add_argument('--target-joint-acceleration',type=float,default=1.2)
    parser.add_argument('--disable-target-angular-monitor',action='store_true',
                        help='Explicitly omit only the project angular monitor from the unreviewed target draft')
    parser.add_argument('--disable-extra-motion-monitors',action='store_true',
                        help='Omit the extra intro/guidance TCP speed monitors in the unreviewed draft; retain control interlocks')
    args=parser.parse_args(argv)
    if args.robot_model!=default_model:
        parser.error(f'This {default_model} entry requires the same robot model; use the dedicated {args.robot_model} entry')
    profile=args.profile.resolve()
    if not profile.is_relative_to(ROOT):parser.error('Profile must be inside workspace')
    output=args.output or Path(default_output or ROOT/f'tmp/{default_model}-delivery')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        manifest=prepare_delivery(json.loads(profile.read_text(encoding='utf-8-sig')),output,
            target_joint_speed=args.target_joint_speed,target_joint_acceleration=args.target_joint_acceleration,
            disable_target_angular_monitor=args.disable_target_angular_monitor,
            disable_extra_motion_monitors=args.disable_extra_motion_monitors,robot_model=default_model)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    print(_json({'output':str(output.resolve()),'robot_model':manifest['robot_model'],'preparation':manifest['preparation'],
                 'baseline':manifest['baseline'],'target':manifest['target']}))
    return 0
