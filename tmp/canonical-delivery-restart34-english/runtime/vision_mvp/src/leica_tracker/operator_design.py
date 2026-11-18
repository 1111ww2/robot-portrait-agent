"""明确区分操作员选定的运动方案与实机运行记录。本模块只处理文件和数值，选定方案不代表已完成调试验收或实测。
沿用现有动作编译器及数值限值，不伪造审核标志或十四点预览。
真实执行时仍检查停止状态、租约、当前状态、TCP、机型与控制器。
可选采集回调只在运行时收到新的明确开始指令后执行，不会在读文件时调用。"""
import copy
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import math
from typing import ClassVar

import numpy as np

from .motion_configuration import configure_motion_design
from .physical_robot_model import SUPPORTED_MODELS, model_for_profile
from .physical_runtime import ReviewedProfile
from .reference_gate import rotation_vector_distance


MODE = 'operator_selected_design'
PROVENANCE = 'OPERATOR_SELECTED_DESIGN'
INSTALLATION_SELECTION_FIELDS = (
    'robot_model', 'reviewed_robot_model', 'reference', 'joints', 'tcp', 'lower',
    'upper', 'speed', 'acceleration', 'base_limit', 'base_rate', 'gesture',
    'max_tcp_linear_speed', 'max_tcp_angular_speed',
)
INTRO_SELECTION_FIELDS = (
    'robot_model', 'reviewed_robot_model', 'reference_joints', 'reference_pose',
    'tcp', 'waypoints', 'joint_speed', 'joint_acceleration',
    'max_tcp_angular_speed', 'max_tcp_linear_speed', 'timeout',
)
_FIELDS = {'installation': INSTALLATION_SELECTION_FIELDS, 'intro': INTRO_SELECTION_FIELDS}
_CUES = frozenset(('MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN', 'MOVE_CLOSER', 'MOVE_FARTHER'))
_RUNTIME = ('hold_timeout', 'action_timeout', 'gate_timeout', 'max_joint_speed')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _vector(value, size, name):
    if (not isinstance(value, (list, tuple)) or len(value) != size or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('Invalid operator design ' + name)
    return [float(x) for x in value]


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid operator design ' + name)
    return float(value)


def _selection_payload(data, *, kind, fields):
    if kind not in _FIELDS or set(fields) != set(_FIELDS[kind]) or len(fields) != len(_FIELDS[kind]):
        raise ValueError('Invalid operator selection field contract')
    try:
        return {'mode': MODE, 'kind': kind, 'fields': {key: data[key] for key in _FIELDS[kind]}}
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete operator selection fields') from exc


def operator_selection_record(data, *, kind, fields):
    """绑定选定的数值输入，不记录实机观测。"""
    try:
        digest = hashlib.sha256(_json(_selection_payload(data, kind=kind, fields=fields)).encode()).hexdigest()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid operator selection payload') from exc
    return {'mode': MODE, 'version': 1, 'kind': kind, 'physical_acceptance': False,
            'payload_sha256': digest}


def validate_operator_selection(data, *, kind, fields):
    """对旧格式记录返回 False；记录格式错误或选定值被改动时验证失败。字段约束与数值构造器一致。
    此检查用于发现意外修改，不能验证文件是否代表经认证的人为意图，也不能证明曾在硬件上试验。"""
    if not isinstance(data, dict):
        return False
    if 'operator_selection' not in data:
        return False
    actual = data['operator_selection']
    expected = operator_selection_record(data, kind=kind, fields=fields)
    if (not isinstance(actual, dict) or set(actual) != set(expected) or
            type(actual.get('version')) is not int or actual.get('physical_acceptance') is not False or
            actual != expected):
        raise ValueError('Operator selection does not match the exact numeric design')
    return True


def _tag(data, kind):
    data['operator_selection'] = operator_selection_record(data, kind=kind, fields=_FIELDS[kind])
    return data


def _design_digest(data):
    """除编译器输入外，同时绑定路径、运行时和参考位策略。"""
    bank = {key: value for key, value in data['operator_design'].items() if key != 'selection_sha256'}
    payload = {'approval_mode': data['approval_mode'], 'robot_model': data['robot_model'],
               'reviewed_robot_model': data['reviewed_robot_model'], 'robot_ip': data['robot_ip'],
               'operator_design': bank, 'paths': data['paths'],
               'installation_selection': data['installation']['operator_selection'],
               'intro_selection': data['intro']['operator_selection'],
               **{key: data[key] for key in _RUNTIME}}
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def nominal_reference_pose(joints, tcp, robot_model):
    """各机型的标称姿态仅供离线使用，实际启动须读取实测姿态。"""
    from .physical_spatial import rotation_vector
    from .ursim_viewer import forward_kinematics
    q = _vector(joints, 6, 'starting joints'); tcp = _vector(tcp, 6, 'nominal TCP')
    matrix = np.asarray(forward_kinematics(q, robot_model)['frames'][-1], dtype=float)
    tool = np.eye(4); tool[:3, :3] = rotation_vector(tcp[3:]); tool[:3, 3] = tcp[:3]
    matrix = matrix @ tool
    r = matrix[:3, :3]
    cosine = max(-1., min(1., float((np.trace(r)-1.)/2.)))
    angle = math.acos(cosine)
    if angle < 1e-10:
        vector = np.zeros(3)
    elif math.pi-angle < 1e-6:
        # 对称分量沿旋转轴的特征值为 1。当反对称分量非零时，用它确定符号。
        values, axes = np.linalg.eigh((r+r.T)/2.)
        axis = axes[:, int(np.argmax(values))]
        skew = np.asarray([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
        if float(np.dot(axis, skew)) < 0: axis = -axis
        vector = axis*angle
    else:
        vector = np.asarray([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])*(angle/(2.*math.sin(angle)))
    return [*matrix[:3, 3].tolist(), *vector.tolist()]


def _straight_paths(paths, origin):
    if not isinstance(paths, dict) or set(paths) != _CUES:
        raise ValueError('Operator design requires all six directional paths')
    for path in paths.values():
        if not isinstance(path, list) or not 1 <= len(path) <= 20:
            raise ValueError('Invalid operator directional path')
        points = [_vector(p, 3, 'path point') for p in path]
        delta = [p-o for p, o in zip(points[-1], origin)]; length = math.hypot(*delta)
        if length <= 0: raise ValueError('Directional path must move outward')
        axis = [x/length for x in delta]; previous = 0.
        for point in points:
            offset = [p-o for p, o in zip(point, origin)]
            projection = sum(a*b for a, b in zip(offset, axis))
            residual = math.hypot(*(p-projection*a for p, a in zip(offset, axis)))
            if projection <= previous or residual > 1e-6:
                raise ValueError('Directional light requires a straight outward path')
            previous = projection


def _bank(source):
    try:
        installation, intro = source['installation'], source['intro']
        origin = _vector(installation['reference'], 6, 'source reference')
        joints = _vector(intro['reference_joints'], 6, 'source joints')
        points = intro['waypoints']
        if not isinstance(points, list) or len(points) != 14:
            raise ValueError('Operator opening design requires exactly14 points')
        points = [_vector(q, 6, 'source opening point') for q in points]
        if max(abs(a-b) for a, b in zip(points[-1], joints)) > 1e-8:
            raise ValueError('Operator opening must end at its selected reference')
        _straight_paths(source['paths'], origin[:3])
        return {
            'intro_joint_deltas': [[q-home for q, home in zip(point, joints)] for point in points],
            'directional_deltas': {cue: [[p-home for p, home in zip(point, origin[:3])] for point in path]
                                   for cue, path in source['paths'].items()},
            'lower_delta': [p-home for p, home in zip(_vector(installation['lower'], 3, 'lower bound'), origin[:3])],
            'upper_delta': [p-home for p, home in zip(_vector(installation['upper'], 3, 'upper bound'), origin[:3])],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete operator source design') from exc


def _reanchor(data, pose, joints, tcp):
    result = copy.deepcopy(data); bank = result['operator_design']
    installation, intro = result['installation'], result['intro']
    installation.update(reference=list(pose), joints=list(joints), tcp=list(tcp),
        lower=[p+d for p, d in zip(pose[:3], bank['lower_delta'])],
        upper=[p+d for p, d in zip(pose[:3], bank['upper_delta'])])
    intro.update(reference_pose=list(pose), reference_joints=list(joints), tcp=list(tcp),
        waypoints=[[q+d for q, d in zip(joints, point)] for point in bank['intro_joint_deltas']])
    intro['waypoints'][-1] = list(joints)
    result['paths'] = {cue: [[p+d for p, d in zip(pose[:3], point)] for point in path]
                       for cue, path in bank['directional_deltas'].items()}
    _tag(installation, 'installation'); _tag(intro, 'intro')
    bank['selection_sha256'] = _design_digest(result)
    return result


def prepare_selected_design(source, robot_model, *, start_joints):
    """离线使用同一组选定角度准备两种原生动作方案。"""
    if robot_model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported operator design robot model')
    start = _vector(start_joints, 6, 'starting joints')
    source = configure_motion_design(source, joint_speed=.7, duration_s=3.3, repetitions=2)
    bank = _bank(source)
    installation_source, intro_source = source['installation'], source['intro']
    tcp = _vector(installation_source['tcp'], 6, 'nominal TCP')
    pose = nominal_reference_pose(start, tcp, robot_model)
    installation = {key: copy.deepcopy(installation_source[key]) for key in (
        'speed', 'acceleration', 'base_limit', 'base_rate', 'gesture',
        'max_tcp_linear_speed', 'max_tcp_angular_speed')}
    # 针对旧参考位设置的基座通道不属于选定的六方向方案。在其他允许的起始姿态下，它可能在换向前卡住有限轨迹相位。
    # 保留原关节速率上限。
    installation['base_limit'] = None
    installation.update(robot_model=robot_model, reviewed_robot_model=robot_model)
    intro = {'robot_model': robot_model, 'reviewed_robot_model': robot_model,
             'provenance': PROVENANCE, 'physical_acceptance': False,
             'joint_speed': .7, 'joint_acceleration': 1.2,
             'max_tcp_linear_speed': None, 'max_tcp_angular_speed': None,
             'timeout': _positive(intro_source['timeout'], 'opening timeout')}
    value = {
        'status': PROVENANCE, 'approval_mode': MODE,
        'robot_model': robot_model, 'reviewed_robot_model': robot_model,
        'robot_ip': source['robot_ip'], 'installation': installation, 'intro': intro,
        'motion_design': copy.deepcopy(source['motion_design']),
        'operator_design': {
            'version': 1, 'provenance': PROVENANCE, 'physical_acceptance': False,
            'starting_joints_rad': start,
            'reference_policy': 'capture_stationary_on_explicit_start',
            'tcp_policy': 'use_current_controller_tcp',
            'nominal_reference_only': True, **bank,
        },
        **{key: _positive(source[key], key) for key in _RUNTIME},
    }
    value['motion_design']['robot_model'] = robot_model
    result = _reanchor(value, pose, start, tcp)
    _validate_data(result)
    return result


def _validate_data(data, *, reference_bound=False):
    if not isinstance(data, dict) or data.get('approval_mode') != MODE:
        raise ValueError('Explicit operator-selected design mode required')
    model = model_for_profile(data)
    try:
        host = str(ipaddress.ip_address(data['robot_ip']))
        if ipaddress.ip_address(host).is_unspecified or ipaddress.ip_address(host).is_multicast:
            raise ValueError('Concrete operator controller address required')
        if model_for_profile(data['installation']) != model or model_for_profile(data['intro']) != model:
            raise ValueError('Operator design model mismatch')
        for kind, fields in _FIELDS.items():
            if not validate_operator_selection(data[kind], kind=kind, fields=fields):
                raise ValueError('Operator selection missing from ' + kind)
        bank = data['operator_design']
        if (not isinstance(bank, dict) or type(bank.get('version')) is not int or bank['version'] != 1
                or bank.get('provenance') != PROVENANCE or bank.get('physical_acceptance') is not False
                or bank.get('reference_policy') != 'capture_stationary_on_explicit_start'
                or bank.get('tcp_policy') != 'use_current_controller_tcp'):
            raise ValueError('Invalid operator reference selection policy')
        selected_start = _vector(bank['starting_joints_rad'], 6, 'default starting joints')
        if bank.get('nominal_reference_only') is not (not reference_bound):
            raise ValueError('Operator starting reference provenance mismatch')
        if not reference_bound and data['installation']['joints'] != selected_start:
            raise ValueError('Operator starting joint metadata differs from the nominal design')
        if bank.get('selection_sha256') != _design_digest(data):
            raise ValueError('Operator selection paths, policy or runtime changed')
        expected = _bank(data)
        for key, value in expected.items():
            original = bank.get(key)
            # 重新锚定时可能发生正常的浮点相消，元数据仍须在舍入误差范围内表示同一组动作。
            def close(a, b):
                if isinstance(a, dict):
                    return isinstance(b, dict) and set(a) == set(b) and all(close(v, b[k]) for k, v in a.items())
                if isinstance(a, list):
                    return isinstance(b, list) and len(a) == len(b) and all(close(v, w) for v, w in zip(a, b))
                return type(b) in (int, float) and math.isfinite(b) and abs(a-b) <= 1e-12
            if not close(value, original):
                raise ValueError('Operator selection relative motion bank changed')
        if (data['intro']['joint_speed'] != .7 or data['intro']['joint_acceleration'] != 1.2
                or data['installation']['base_rate'] != .7
                or data['installation']['gesture'] != dict(duration_s=3.3, repetitions=2, joint_speed_limit=.7)):
            raise ValueError('Operator design requires the selected0.7/1.2/3.3 motion')
        for key in _RUNTIME: _positive(data[key], key)
        return model, host
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Incomplete operator-selected design') from exc


def _stationary_reference(primary, feedback):
    if not isinstance(primary, dict) or not isinstance(feedback, dict):
        raise ValueError('Independent native reference snapshots required')
    mode = primary.get('mode')
    if (not isinstance(mode, dict) or mode.get('mode') != 7 or
            any(mode.get(k) is not True for k in ('connected', 'enabled', 'power')) or
            any(mode.get(k) is not False for k in ('emergency', 'protective', 'paused', 'running'))):
        raise ValueError('Native controller must be ready and idle')
    pose = _vector(feedback.get('tcp_pose'), 6, 'measured TCP pose')
    joints = _vector(feedback.get('joints'), 6, 'measured joints')
    tcp = _vector(primary.get('tcp_offset'), 6, 'current controller TCP')
    primary_pose = _vector(primary.get('pose'), 6, 'independent TCP pose')
    speed = _vector(feedback.get('tcp_speed'), 6, 'TCP speed')
    joint_speed = _vector(feedback.get('joint_speed'), 6, 'joint speed')
    if (math.hypot(*speed[:3]) >= .0005 or math.hypot(*speed[3:]) >= .002
            or max(map(abs, joint_speed)) >= .003):
        raise ValueError('Native reference must be stationary')
    if (math.dist(primary_pose[:3], pose[:3]) > .0005 or
            rotation_vector_distance(primary_pose[3:], pose[3:]) > .005):
        raise ValueError('Independent native reference poses disagree')
    return pose, joints, tcp


@dataclass(frozen=True)
class SelectedDesignProfile(ReviewedProfile):
    """记录选定的数值，不据此断言已在实体机器人上运行。"""
    provenance: ClassVar[str] = PROVENANCE
    physical_acceptance: ClassVar[bool] = False
    reference_restart_policy: ClassVar[str] = 'capture_stationary_on_explicit_start'
    reference_bound: bool = False
    _design_json: str = field(default='', repr=False)
    _capture_provider: object = field(default=None, repr=False, compare=False)

    @property
    def selection_metadata(self):
        result = copy.deepcopy(json.loads(self._design_json)['operator_design'])
        result['captured_joints_rad'] = list(self.cfg.joints) if self.reference_bound else None
        result['captured_tcp_m_rad'] = list(self.cfg.tcp) if self.reference_bound else None
        result['reference_bound'] = self.reference_bound
        return result

    def bind_reference(self, primary, feedback):
        """仅绑定数值；调用者须单独验证样本时效和身份。"""
        if self.reference_bound:
            raise RuntimeError('Operator profile reference is already bound')
        pose, joints, tcp = _stationary_reference(primary, feedback)
        value = _reanchor(json.loads(self._design_json), pose, joints, tcp)
        value['operator_design']['nominal_reference_only'] = False
        value['operator_design']['selection_sha256'] = _design_digest(value)
        return _profile(value, reference_bound=True, capture_provider=self._capture_provider)

    def bind_before_motion(self, *, reader, control, should_stop, sample_observer=None,
                           proof_observer=None):
        if self.reference_bound:
            raise RuntimeError('Operator profile reference is already bound')
        if should_stop():
            raise RuntimeError('Operator reference capture stopped before opening')
        options = {}
        if sample_observer is not None:
            if not callable(sample_observer):
                raise ValueError('Sample observer must be callable')
            options['sample_observer'] = sample_observer
        if proof_observer is not None:
            if not callable(proof_observer):
                raise ValueError('Proof observer must be callable')
            options['proof_observer'] = proof_observer
        provider = self._capture_provider
        if provider is None:
            from .operator_design_capture import capture_selected_reference
            provider = capture_selected_reference
        primary, feedback = provider(self, reader=reader, control=control,
                                     should_stop=should_stop, **options)
        if should_stop():
            raise RuntimeError('Operator reference capture stopped before binding')
        return self.bind_reference(primary, feedback)


def _profile(data, *, reference_bound=False, capture_provider=None):
    from .physical_intro_program import IntroProgram
    from .physical_resident_engine import Installation, ResidentEngine
    _, host = _validate_data(data, reference_bound=reference_bound)
    value = copy.deepcopy(data)
    cfg = Installation(value['installation'])
    opening = IntroProgram(value['intro'], 1)
    if (opening.reference_joints != cfg.joints or opening.reference_pose != cfg.neutral
            or opening.tcp != cfg.tcp):
        raise ValueError('Operator opening and XYZ references differ')
    paths = value['paths']
    _straight_paths(paths, cfg.neutral[:3])
    result = SelectedDesignProfile(host, cfg, paths, copy.deepcopy(paths),
        intro=value['intro'], reference_bound=reference_bound,
        _design_json=_json(value), _capture_provider=capture_provider,
        **{key: value[key] for key in _RUNTIME})
    # 精确继承的路径绑定表示数值方案选择，不代表已保存的实机观测。所有路径数值约束仍然有效。
    result.supervisor(ResidentEngine(cfg, None, 1), None)
    return result


def selected_profile(data, *, capture_provider=None):
    """加载标称离线方案，明确开始前不调用任何服务提供方。"""
    return _profile(data, capture_provider=capture_provider)
