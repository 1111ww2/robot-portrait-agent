"""在离线环境中选择通用动作方案。关节限速的单位是弧度/秒；
笛卡尔跟踪上限由动作位移与时间规律推导，不能直接取同一数值。"""
import copy
import math

from .native_gesture import canonical_dynamics
from .physical_robot_model import model_for_profile

DEFAULT_MAX_DISPLACEMENT_M = .12


def _finite_vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value)==size
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def _maximum_displacement(source):
    installation=source['installation']; paths=source.get('paths')
    reference=installation.get('reference')
    if reference is None and (paths is None or isinstance(paths,dict)
                              and all(path is None for path in paths.values())):
        # 模板只接收动力学参数，不生成参考位或路径。
        return DEFAULT_MAX_DISPLACEMENT_M, 'unconfigured_template_default'
    if not _finite_vector(reference,6) or not isinstance(paths,dict) or not paths:
        raise ValueError('Finite reference and direction paths required')
    distances=[]
    for path in paths.values():
        if not isinstance(path,(list,tuple)) or not path:
            raise ValueError('Finite nonempty direction paths required')
        for point in path:
            if not _finite_vector(point,3):raise ValueError('Invalid canonical direction path')
            distances.append(math.dist(point,reference[:3]))
    maximum=max(distances)
    if not math.isfinite(maximum) or maximum<=0:
        raise ValueError('Positive canonical displacement required')
    return maximum, 'provided_paths'


def configure_motion_design(source, *, joint_speed=.7, duration_s=3.3, repetitions=2):
    """复制方案并选择标准动力学参数，原有验收记录随之失效。
    保留该机型的安装几何、TCP、负载、边界和方向路径，缺失的配置仍然缺失。
    结果供仿真或安装审核使用，不表示实机已按新参数运行。"""
    if (not isinstance(source,dict) or not isinstance(source.get('installation'),dict)
            or not isinstance(source.get('intro'),dict)):
        raise ValueError('Motion design requires installation and opening sections')
    if type(joint_speed) not in (int,float) or not math.isfinite(joint_speed) or not 0<joint_speed<=.7:
        raise ValueError('Joint speed must be positive and at most0.7rad/s')
    if source.get('approval_mode')=='operator_selected_design':
        from .operator_design import selected_profile
        selected=selected_profile(source)
        gesture=selected.cfg.gesture.as_dict()
        if (joint_speed!=selected.intro['joint_speed'] or
                duration_s!=gesture['duration_s'] or repetitions!=gesture['repetitions']):
            raise ValueError('Changing a selected design requires a new explicit design selection')
        # 保留选定的原始字节和字段绑定。根据浮点路径距离重算动力学可能改变已选数值。
        return copy.deepcopy(source)
    model=model_for_profile(source)
    displacement,basis=_maximum_displacement(source)
    dynamics=canonical_dynamics(displacement,duration_s=duration_s,repetitions=repetitions,
                                joint_speed_limit=joint_speed)
    result=copy.deepcopy(source)
    result['installation'].update(dynamics)
    result['intro'].update(joint_speed=float(joint_speed),joint_acceleration=1.2)
    # 此处移除的仅是项目层面的指定 TCP 监测项。机器人自身保护、控制权、数据包与遥测检查以及关节速率上限仍然有效。
    for name in ('intro','installation'):
        result[name]['max_tcp_linear_speed']=None
        result[name]['max_tcp_angular_speed']=None
        result[name]['review']={key:False for key in result[name].get('review',{})}
    result['intro']['reviewed']={}
    result['reviewed_installation']={}
    result.pop('intro_preview',None)
    # 该旧字段是参考位或拍照时判断静止的容差。
    # 运动指令的限值由 installation.gesture/base_rate 决定。
    stationary=source.get('max_joint_speed',.003)
    if type(stationary) not in (int,float) or not math.isfinite(stationary) or stationary<=0:
        raise ValueError('Positive reference joint stillness threshold required')
    result['max_joint_speed']=min(float(stationary),.003)
    # 参考位准入和观测 RPC 不属于控制心跳。
    # 保持静止时允许其用完有限的本地调度预算，原生控制截止时间和源帧时效条件保持不变。
    result['gate_timeout']=.25
    result['status']='CANONICAL_DESIGN_PHYSICAL_ACCEPTANCE_PENDING'
    result['motion_design']={
        'format_version':1,'robot_model':model,'joint_speed_rad_s':float(joint_speed),
        'joint_acceleration_rad_s2':1.2,'max_displacement_m':displacement,
        'max_displacement_basis':basis,'duration_s':float(duration_s),
        'repetitions':repetitions,'physical_acceptance':False,
        'reference_rpc_timeout_s':.25,
        'stationary_joint_speed_rad_s':result['max_joint_speed'],
        'cadence_note':'Nominal controller phase time; IK joint-rate limiting can extend actual duration.',
    }
    return result
