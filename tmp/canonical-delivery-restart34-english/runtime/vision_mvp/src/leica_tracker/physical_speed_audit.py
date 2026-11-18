"""对固定从始至终的关节路标进行纯线下时间审计。不会创建机器人模型、网络连接、控制器程序或审查。
使用的唯一TCP计算是仅一个移动旋转关节的角速度恒等式：其角速度大小等于TCP的角速度大小。
多关节角速度和每个TCP线性速度仍未评估。可选的TCP保护器是项目软件监控器，而不是机器限制。"""
import math


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive(value, name):
    if not _finite_number(value) or value <= 0:
        raise ValueError('Explicit finite positive ' + name + ' required')
    return float(value)


def _joints(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 6 or
            any(not _finite_number(q) for q in value)):
        raise ValueError('Six finite joint angles required for ' + name)
    return tuple(float(q) for q in value)


def audit_intro_speed(intro, joint_speed, joint_acceleration):
    """报告理想的movej时间和可证明的单关节角度冲突。
    每个段落在静止状态下开始和结束，与当前的“movej（...，r=0）”路线相同。
    关节峰值描述了在所需速度和加速度下的理想前导关节包络，而不是测量运动。
    控制器速度缩放、实际跟踪和安装几何形状未被推断。在这个有限的计算中没有观察到的冲突无法确定物理适用性。
    明确的“max_tcp_angular_speed=None”将角度保护比较留而不评估，同时保留了理想峰值；
    省略该设置是不有效的。明确的“max_tcp_linear_speed=None”将线性监视器记录为禁用；
    无论使用哪个设置，线性速度都未被评估，省略它是不有效的。"""
    if not isinstance(intro, dict):
        raise ValueError('Explicit intro mapping required')
    speed = _positive(joint_speed, 'joint speed in rad/s')
    acceleration = _positive(joint_acceleration, 'joint acceleration in rad/s²')
    if 'max_tcp_angular_speed' not in intro:
        raise ValueError('Explicit TCP angular guard setting required')
    angular_limit = intro['max_tcp_angular_speed']
    if angular_limit is not None:
        angular_limit = _positive(angular_limit, 'TCP angular guard in rad/s')
    if 'max_tcp_linear_speed' not in intro:
        raise ValueError('Explicit TCP linear guard setting required')
    linear_limit = intro['max_tcp_linear_speed']
    if linear_limit is not None:
        linear_limit = _positive(linear_limit, 'TCP linear guard in m/s')
    start = _joints(intro.get('reference_joints'), 'reference')
    raw_points = intro.get('waypoints')
    if not isinstance(raw_points, list) or not 2 <= len(raw_points) <= 20:
        raise ValueError('An intro list of two to twenty waypoints is required')
    points = [_joints(point, 'waypoint') for point in raw_points]
    segments, conflicts = [], []
    for index, (before, after) in enumerate(zip([start] + points[:-1], points), 1):
        distances = [abs(a - b) for a, b in zip(after, before)]
        if not all(math.isfinite(distance) for distance in distances):
            raise ValueError('Derived joint displacement is not finite')
        moving = [joint for joint, distance in enumerate(distances) if distance > 0.]
        distance = max(distances)
        if not moving:
            peak, duration, shape = 0., 0., 'stationary'
        else:
            # sqrt(a)*sqrt(d)避免溢出中间a*d。
            triangular_peak = math.sqrt(acceleration) * math.sqrt(distance)
            peak = min(speed, triangular_peak)
            if triangular_peak <= speed:
                duration, shape = 2. * (distance / peak), 'triangular'
            else:
                duration, shape = distance / speed + speed / acceleration, 'trapezoidal'
            if not math.isfinite(peak) or not math.isfinite(duration):
                raise ValueError('Ideal joint timing cannot be represented finitely')
        isolated = moving[0] if len(moving) == 1 else None
        angular_peak = peak if isolated is not None else None
        conflict = (angular_peak > angular_limit
                    if angular_peak is not None and angular_limit is not None else None)
        segment = {
            'segment_index': index,
            'moving_joint_indices': moving,
            'max_joint_displacement_rad': distance,
            'motion_profile': shape,
            'ideal_joint_peak_rad_s': peak,
            'ideal_duration_s': duration,
            'isolated_joint_index': isolated,
            'isolated_joint_distance_rad': distance if isolated is not None else None,
            'ideal_tcp_angular_peak_rad_s': angular_peak,
            'angular_guard_conflict': conflict,
        }
        segments.append(segment)
        if conflict:
            conflicts.append(dict(segment))
    return {
        'scope': 'OFFLINE_IDEAL_JOINT_TIMING_NOT_PHYSICAL_APPROVAL',
        'target_joint_speed_rad_s': speed,
        'target_joint_acceleration_rad_s2': acceleration,
        'max_tcp_angular_speed_rad_s': angular_limit,
        'tcp_angular_monitor_enabled': angular_limit is not None,
        'tcp_angular_guard_origin': 'project_software_monitor_not_machine_limit',
        'max_tcp_linear_speed_m_s': linear_limit,
        'tcp_linear_monitor_enabled': linear_limit is not None,
        'tcp_linear_guard_origin': 'project_software_monitor_not_machine_limit',
        'segments': segments,
        'known_angular_conflicts': conflicts,
        'joint_peaks_are_measured': False,
        'joint_peak_scope': 'ideal_rest_to_rest_leading_joint_envelope',
        'controller_speed_scaling_assessed': False,
        'multi_joint_tcp_angular_assessed': False,
        'tcp_linear_speed_assessed': False,
        'requires_onsite_review': True,
        'limitations': [
            'Single-joint TCP angular identity requires exactly one changing revolute joint.',
            'Actual controller scaling or tracking can reduce or alter these ideal peaks and times.',
            'Multi-joint TCP angular speed and all TCP linear speeds require installation geometry or measurements.',
            'No conflict found does not validate a path, dynamics, payload, mount, stop distance or physical execution.',
        ],
    }
