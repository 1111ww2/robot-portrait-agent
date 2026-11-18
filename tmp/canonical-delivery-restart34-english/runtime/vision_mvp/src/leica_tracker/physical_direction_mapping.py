"""将经过标定的视觉方向转换为基座坐标系中的单位向量。这里只计算方向，不访问设备、生成轨迹或执行安全检查。
R_TCP_camera 把相机光学坐标转换到当前 TCP 坐标系。
在经过方向校正且未镜像的图像中，相机 +X 向右，+Y 向下，+Z 从镜头指向场景。
标定须包含实际安装和图像方向，机器人的 TCP 偏移不能提供这个矩阵。
participant 按面对相机的人物自身定义左右，其左侧始终对应相机 +X；
preview 按显示画面定义左右，须显式应用镜像标志。
FARTHER 表示人物沿 +Z 远离，CLOSER 表示沿 -Z 靠近。
每个提示都须提供安装审核确定的符号，将人物移动映射为机械臂动作。仅有方向不能确定路径，也不能证明安装安全。"""

import math
from numbers import Real

import numpy as np


def _numeric_array(value, shape, name):
    try:
        result = np.asarray(value, dtype=object)
        if (result.shape != shape or any(isinstance(item, (bool, np.bool_))
                or not isinstance(item, Real) for item in result.flat)):
            raise ValueError
        result = result.astype(float)
        if not np.isfinite(result).all():
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid {name}') from exc


def _tcp_to_base(rotation_vector):
    vector = _numeric_array(rotation_vector, (3,), 'TCP rotation vector')
    angle = float(np.linalg.norm(vector))
    if not math.isfinite(angle):
        raise ValueError('Invalid TCP rotation vector')
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def direction_in_base(cue, *, camera_to_tcp, tcp_rotation_vector, preview_mirrored, cue_frame,
                      gesture_signs):
    """根据最新 TCP 姿态，返回基座坐标中的机械臂单位方向。
    所有关键字参数均必填：cue_frame 取 participant 或 preview；
    gesture_signs 为六个提示分别指定整数 +1 或 -1，表示与人物移动同向或反向。
    调用者负责核验安装、姿态时效、路径和实际停止行为。"""
    if type(preview_mirrored) is not bool or cue_frame not in ('participant', 'preview'):
        raise ValueError('Explicit mirror and cue frame required')
    matrix = _numeric_array(camera_to_tcp, (3, 3), 'camera-to-TCP rotation')
    if (not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0)
            or not math.isclose(float(np.linalg.det(matrix)), 1., abs_tol=1e-6, rel_tol=0)):
        raise ValueError('Camera-to-TCP matrix must be a proper rotation')
    tcp_to_base = _tcp_to_base(tcp_rotation_vector)
    # 面对相机的人物，其左侧位于未镜像图像的右侧。
    left_sign = 1 if cue_frame == 'participant' or preview_mirrored else -1
    vectors = {'MOVE_LEFT': (left_sign, 0, 0), 'MOVE_RIGHT': (-left_sign, 0, 0),
               'MOVE_UP': (0, -1, 0), 'MOVE_DOWN': (0, 1, 0),
               'MOVE_CLOSER': (0, 0, -1), 'MOVE_FARTHER': (0, 0, 1)}
    if type(cue) is not str or cue not in vectors:
        raise ValueError('An explicit directional cue is required')
    if (not isinstance(gesture_signs, dict) or set(gesture_signs) != set(vectors)
            or any(type(value) is not int or value not in (-1, 1)
                   for value in gesture_signs.values())):
        raise ValueError('Explicit +1/-1 arm-gesture signs for all six cues are required')
    vector = tcp_to_base @ matrix @ (gesture_signs[cue] * np.asarray(vectors[cue], dtype=float))
    # 验证矩阵时允许实测标定值存在合理的舍入误差。
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length == 0:
        raise ValueError('Invalid mapped direction')
    return tuple(float(x / length) for x in vector)
