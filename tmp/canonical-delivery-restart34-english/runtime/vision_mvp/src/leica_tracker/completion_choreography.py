"""设置仿真确认动作和末端姿态保持时间。"""
import math

SETTLE_MS = 400
CONFIRM_END_MS = 3200
COMPLETE_MS = 6000
SCREEN_CONFIRM_END_MS = 6000
SCREEN_COMPLETE_MS = 8800


def simulated_frame_pose_at(elapsed_ms):
    progress=max(0.,min(1.,elapsed_ms/2400))
    amount=progress**3*(10-15*progress+6*progress**2)
    return .02*amount,.015*amount,0.


def completion_pose_at(elapsed_ms, *, framed=False,screen_coupled=False):
    """返回沿 X 轴向内、沿 Z 轴向上和腕部俯仰的分量，并保持最终姿态。"""
    if not math.isfinite(elapsed_ms):
        raise ValueError('Invalid completion time')
    start_x,start_z=(.02,.015) if framed else (0.,0.)
    confirm_end=SCREEN_CONFIRM_END_MS if screen_coupled else CONFIRM_END_MS
    complete=SCREEN_COMPLETE_MS if screen_coupled else COMPLETE_MS
    if elapsed_ms <= SETTLE_MS:
        return start_x,start_z,0.
    if elapsed_ms < confirm_end:
        progress=(elapsed_ms-SETTLE_MS)/(confirm_end-SETTLE_MS)
        return start_x,start_z,(.20 if framed else .24)*math.sin(2*math.pi*progress)**2
    progress=max(0.,min(1.,(elapsed_ms-confirm_end)/(complete-confirm_end)))
    # 回撤轨迹的两个端点速度和加速度均为零。
    amount=progress**3*(10-15*progress+6*progress**2)
    return start_x+(.10-start_x)*amount,start_z+(.08-start_z)*amount,.10*amount
