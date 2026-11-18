"""根据新的仿真参考测量，在内存中构建 URSim 方案。本模块不连接设备或写文件。
调用 build_profile 前，调用方须验证隔离的官方仿真器身份、时间戳持续前进且样本有效；
单凭遥测字典无法证明时效和控制器身份。
只从 source 取动作设计增量和 XYZ 动力学配置，忽略实机状态、安装验收与开场预览记录。
重新绑定操作员显式选定的方案时保留原数值授权，不改写为旧式安装审核。
生成的 review 和 reviewed 仅用于共享数值验证器，明确标为 SIMULATION_ONLY，不构成操作员验收或实体调试记录。
此处不导出实机配置；上层打开任何套接字前须检查确切的仿真配置类型和专用回环地址。"""
import copy
from dataclasses import dataclass, field, replace
import json
import math
from types import MappingProxyType
from typing import ClassVar, Mapping

from .physical_intro_program import IntroProgram
from .physical_resident_engine import Installation, ResidentEngine
from .physical_robot_model import SUPPORTED_MODELS
from .physical_runtime import ReviewedProfile
from .reference_gate import rotation_vector_distance


SIMULATOR_HOST = '127.0.0.2'
_CUES = frozenset(('MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN',
                   'MOVE_CLOSER', 'MOVE_FARTHER'))


@dataclass(frozen=True)
class SimulationProfile(ReviewedProfile):
    """原生运行字段携带不可变的“仅仿真”身份标记。"""
    simulation_only: ClassVar[bool] = True
    provenance: ClassVar[str] = 'SIMULATION_ONLY'
    simulation_metadata: Mapping = field(default_factory=lambda: MappingProxyType({}))
    operator_design: str | None = field(default=None, repr=False)
    capture_provider: object = field(default=None, repr=False, compare=False)
    binding_observer: object = field(default=None, repr=False, compare=False)
    preserve_restart_reference: bool = False

    @property
    def reference_restart_policy(self):
        if type(self.preserve_restart_reference) is not bool:
            raise ValueError('Explicit simulator restart reference mode required')
        return ('capture_stationary_on_explicit_start'
                if self.operator_design is not None and not self.preserve_restart_reference else None)

    def bind_before_motion(self, *, reader, control, should_stop, sample_observer=None,
                           proof_observer=None):
        """仅在明确设置了围栏的模拟器捕获中使用所选的粘合剂。"""
        options = {}
        if sample_observer is not None:
            if not callable(sample_observer):
                raise ValueError('Simulator sample observer must be callable')
            options['sample_observer'] = sample_observer
        if proof_observer is not None:
            if not callable(proof_observer):
                raise ValueError('Simulator proof observer must be callable')
            options['proof_observer'] = proof_observer
        if self.operator_design is None:
            if options:
                raise RuntimeError('Legacy simulator profile does not support restart reference observation')
            return self
        if self.host != SIMULATOR_HOST:
            raise ValueError('Selected simulation binding requires the isolated host')
        if not callable(self.capture_provider):
            raise RuntimeError('Selected simulator requires a fenced reference capture provider')
        from .operator_design import selected_profile
        selected = selected_profile(json.loads(self.operator_design))
        if selected.cfg.robot_model != self.cfg.robot_model:
            raise ValueError('Selected simulator reference model changed')
        if should_stop(): raise RuntimeError('Selected simulator reference capture stopped')
        # 仅修改内存中只读采集端点的目标。原选定文件及绑定 IP 地址的摘要既不修改，也不重新签署。
        capture_target = replace(selected, host=SIMULATOR_HOST)
        primary, feedback = self.capture_provider(capture_target, reader=reader,
            control=control, should_stop=should_stop, **options)
        if should_stop(): raise RuntimeError('Selected simulator reference capture stopped')
        bound = selected.bind_reference(primary, feedback)
        if bound.cfg.robot_model != self.cfg.robot_model:
            raise ValueError('Selected simulator bound model changed')
        intro = copy.deepcopy(bound.intro)
        intro.update(provenance='SIMULATION_ONLY', simulation_only=True)
        metadata = dict(self.simulation_metadata, operator_reference_bound=True)
        actual = SimulationProfile(SIMULATOR_HOST, bound.cfg, copy.deepcopy(bound.paths),
            copy.deepcopy(bound.reviewed_paths), bound.hold_timeout, bound.action_timeout,
            bound.gate_timeout, bound.max_joint_speed, intro,
            simulation_metadata=MappingProxyType(metadata))
        if self.binding_observer is not None:
            if not callable(self.binding_observer):
                raise ValueError('Selected simulator binding observer must be callable')
            self.binding_observer(actual)
        return actual


def _vector(value, size, name):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f'Invalid simulator {name}')
    return [float(v) for v in value]


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid simulator {name}')
    return float(value)


def _stationary_reference(primary, feedback):
    if not isinstance(primary, dict) or not isinstance(feedback, dict):
        raise ValueError('Fresh simulator Primary and RTDE snapshots required')
    mode = primary.get('mode')
    if (not isinstance(mode, dict) or mode.get('mode') != 7
            or any(mode.get(k) is not True for k in ('connected', 'enabled', 'power'))
            or any(mode.get(k) is not False for k in ('emergency', 'protective', 'paused', 'running'))):
        raise ValueError('Simulator must be ready and idle before recording its reference')
    pose = _vector(feedback.get('tcp_pose'), 6, 'TCP pose')
    joints = _vector(feedback.get('joints'), 6, 'joints')
    tcp = _vector(primary.get('tcp_offset'), 6, 'active TCP')
    primary_pose = _vector(primary.get('pose'), 6, 'Primary TCP pose')
    speed = _vector(feedback.get('tcp_speed'), 6, 'TCP speed')
    joint_speed = _vector(feedback.get('joint_speed'), 6, 'joint speed')
    if (math.hypot(*speed[:3]) >= .0005 or math.hypot(*speed[3:]) >= .002
            or max(abs(v) for v in joint_speed) >= .003):
        raise ValueError('Simulator reference must be stationary')
    if (math.dist(primary_pose[:3], pose[:3]) > .0005
            or rotation_vector_distance(primary_pose[3:], pose[3:]) > .005):
        raise ValueError('Simulator Primary and RTDE reference poses disagree')
    return pose, joints, tcp


def _straight_paths(paths, origin):
    """复用的边缘灯光在每个动作中只表示一条向外的轴。"""
    for path in paths.values():
        delta = [v - o for v, o in zip(path[-1], origin)]
        length = math.hypot(*delta)
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Simulator directional path has no outward displacement')
        axis = [v / length for v in delta]
        previous = 0.
        for point in path:
            offset = [v - o for v, o in zip(point, origin)]
            projection = sum(a * b for a, b in zip(offset, axis))
            residual = math.hypot(*(v - projection * a for v, a in zip(offset, axis)))
            if projection <= previous or residual > 1e-6:
                raise ValueError('Simulator edge lights require a straight outward path')
            previous = projection


def build_profile(source, primary, feedback, *, host=SIMULATOR_HOST,
                  joint_speed=.6, joint_acceleration=1.2, robot_model='ur10e') -> SimulationProfile:
    """根据新仿真状态重新锚定十四点设计和六条 XYZ 路径。调用者检查样本时效与时钟推进；
    本工厂函数另要求快照一致且静止，仿真控制器已使能、通电并空闲。它不修改 TCP、负载、工具电源或输入输出。
    来源中的实机审核标志与已接受的预览数据均不参与计算。"""
    if host != SIMULATOR_HOST:
        raise ValueError('URSim profile host must be exactly 127.0.0.2')
    if robot_model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported simulator robot model')
    if not isinstance(source, dict):
        raise ValueError('Simulator motion design required')
    operator_design = None
    if 'approval_mode' in source:
        from .operator_design import selected_profile
        selected = selected_profile(source)
        if selected.cfg.robot_model != robot_model:
            raise ValueError('Selected source model does not match the simulator model')
        if joint_speed != .7 or joint_acceleration != 1.2:
            raise ValueError('Selected simulator dynamics must match the selected design')
        operator_design = json.dumps(source, ensure_ascii=True, allow_nan=False)
    pose, joints, tcp = _stationary_reference(primary, feedback)
    speed = _positive(joint_speed, 'intro joint speed')
    acceleration = _positive(joint_acceleration, 'intro joint acceleration')
    if operator_design is not None:
        # 完整保留选定验证器的策略，包括显式选定且值为空的基座通道。
        # 用旧式 review=True 克隆配置既不能授权该策略，也会误报方案来源。
        bound = selected.bind_reference(primary, feedback)
        intro = copy.deepcopy(bound.intro)
        intro.update(provenance='SIMULATION_ONLY', simulation_only=True)
        design = source['installation']; cog = design.get('center_of_gravity_m')
        metadata = MappingProxyType({
            'provenance':'SIMULATION_ONLY', 'physical_acceptance_reused':False,
            'payload_kg':copy.deepcopy(design.get('payload_kg')),
            'center_of_gravity_m':tuple(_vector(cog,3,'payload metadata CoG')) if cog is not None else None,
        })
        profile = SimulationProfile(host,bound.cfg,copy.deepcopy(bound.paths),copy.deepcopy(bound.reviewed_paths),
            bound.hold_timeout,bound.action_timeout,bound.gate_timeout,bound.max_joint_speed,intro,
            simulation_metadata=metadata,operator_design=operator_design)
        profile.supervisor(ResidentEngine(profile.cfg,None,1),None)
        _straight_paths(profile.paths,pose[:3])
        return profile
    try:
        design = source['installation']
        opening_design = source['intro']
        if not isinstance(design, dict) or not isinstance(opening_design, dict):
            raise ValueError('Simulator installation and opening design required')
        old_reference = _vector(design['reference'], 6, 'design reference')
        old_joints = _vector(opening_design['reference_joints'], 6, 'design reference joints')
        original_waypoints = opening_design['waypoints']
        if not isinstance(original_waypoints, list) or len(original_waypoints) != 14:
            raise ValueError('Simulator opening design requires exactly 14 points')
        original_waypoints = [_vector(q, 6, 'design waypoint') for q in original_waypoints]
        if max(abs(a - b) for a, b in zip(original_waypoints[-1], old_joints)) > 1e-8:
            raise ValueError('Simulator opening design must end at its reference')
        waypoints = [[new + value - old for new, value, old in zip(joints, q, old_joints)]
                     for q in original_waypoints]
        waypoints[-1] = list(joints)  # 保留最新实测端点的准确数值。
        installation = {
            'robot_model': robot_model, 'reviewed_robot_model': robot_model,
            'reference': pose, 'joints': joints, 'tcp': tcp,
            'lower': [new + value - old for new, value, old in
                      zip(pose[:3], _vector(design['lower'], 3, 'design lower bound'), old_reference[:3])],
            'upper': [new + value - old for new, value, old in
                      zip(pose[:3], _vector(design['upper'], 3, 'design upper bound'), old_reference[:3])],
            'review': {k: True for k in ('reference', 'tcp', 'payload', 'path', 'dynamics', 'direction', 'stop')},
        }
        for key in ('speed', 'acceleration', 'base_limit', 'base_rate'):
            installation[key] = design[key]
        if 'gesture' in design:installation['gesture']=copy.deepcopy(design['gesture'])
        for key in ('max_tcp_linear_speed', 'max_tcp_angular_speed'):
            if key in design:
                installation[key] = design[key]
        cfg = Installation(installation)
        intro = {
            'robot_model': robot_model, 'reviewed_robot_model': robot_model,
            'provenance': 'SIMULATION_ONLY', 'simulation_only': True,
            'reference_joints': list(joints), 'reference_pose': list(pose), 'tcp': list(tcp),
            'waypoints': waypoints, 'joint_speed': speed, 'joint_acceleration': acceleration,
            'max_tcp_linear_speed': None, 'max_tcp_angular_speed': None,
            'timeout': _positive(opening_design['timeout'], 'intro timeout'),
            'review': {k: True for k in ('robot', 'mount', 'tcp', 'payload', 'path', 'dynamics', 'stop', 'gesture')},
        }
        reviewed_keys = ('reference_joints', 'reference_pose', 'tcp', 'waypoints', 'joint_speed',
                         'joint_acceleration', 'max_tcp_angular_speed', 'max_tcp_linear_speed', 'timeout')
        intro['reviewed'] = {key: copy.deepcopy(intro[key]) for key in reviewed_keys}
        IntroProgram(intro, 1)
        design_paths = source['paths']
        if not isinstance(design_paths, dict) or set(design_paths) != _CUES:
            raise ValueError('Simulator design requires all six directional paths')
        paths = {}
        for cue, path in design_paths.items():
            if not isinstance(path, list) or not 1 <= len(path) <= 20:
                raise ValueError('Invalid simulator directional path')
            paths[cue] = [[new + value - old for new, value, old in
                           zip(pose[:3], _vector(point, 3, 'design path point'), old_reference[:3])]
                          for point in path]
        runtime = {key: _positive(source[key], key) for key in
                   ('hold_timeout', 'action_timeout', 'gate_timeout', 'max_joint_speed')}
        cog = design.get('center_of_gravity_m')
        metadata = MappingProxyType({
            'provenance': 'SIMULATION_ONLY', 'physical_acceptance_reused': False,
            'payload_kg': copy.deepcopy(design.get('payload_kg')),
            'center_of_gravity_m': tuple(_vector(cog, 3, 'payload metadata CoG')) if cog is not None else None,
        })
        profile = SimulationProfile(host, cfg, paths, copy.deepcopy(paths), intro=intro,
                                    simulation_metadata=metadata, operator_design=operator_design, **runtime)
        # 构造过程只验证数值。上层运行程序明确开始交互后，才启动工作线程和门控。
        profile.supervisor(ResidentEngine(cfg, None, 1), None)
        _straight_paths(paths, pose[:3])
        return profile
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Incomplete or unrepresentable simulator motion design') from exc
