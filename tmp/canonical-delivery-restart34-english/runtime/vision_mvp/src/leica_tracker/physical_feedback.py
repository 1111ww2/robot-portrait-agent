"""将测量到的物理位移转换为方向光强度。此计算既不会命令也不会停止机器人。
呼叫者必须将样本与经过认证的动作和实时、经过验证的运输关联起来。
呼叫者提供测量到的起始、经过审查的方向和计划的旅行；该功能拒绝过期的、无效的、不活动的或不匹配的动作样本。"""

import math


_EDGES = {'MOVE_LEFT': 'left', 'MOVE_RIGHT': 'right', 'MOVE_UP': 'top',
          'MOVE_DOWN': 'bottom', 'MOVE_CLOSER': 'all', 'MOVE_FARTHER': 'all'}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value):
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(map(_number, value))


def physical_edge_feedback(*, cue, action_id, expected_action_id, started_at, captured_at, now,
                           max_age, active, reference_position, current_position,
                           direction, planned_displacement):
    """如果必须不显示电测数据，请返回“显示客人反馈”或“无”。"""
    if (not isinstance(cue, str) or cue not in _EDGES or active is not True or
            type(action_id) is not int or action_id <= 0 or
            type(expected_action_id) is not int or action_id != expected_action_id or
            not all(map(_number, (started_at, captured_at, now, max_age, planned_displacement))) or
            max_age <= 0 or planned_displacement <= 0 or
            not started_at <= captured_at <= now or now - captured_at > max_age or
            not all(map(_vector, (reference_position, current_position, direction)))):
        return None
    magnitude = math.hypot(*direction)
    if not math.isclose(magnitude, 1., rel_tol=1e-6, abs_tol=1e-6):
        return None
    delta = [current - reference for current, reference in zip(current_position, reference_position)]
    projection = sum(component * axis for component, axis in zip(delta, direction))
    if not math.isfinite(projection):
        return None
    return {'edge': _EDGES[cue], 'tone': 'red',
            'intensity': round(min(1., max(0., projection / planned_displacement)), 5)}
