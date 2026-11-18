"""屏幕边缘提示与机械臂共用同一动作阶段。"""
import math
from .gesture_preview import normalized_offset_for
from .completion_choreography import completion_pose_at,CONFIRM_END_MS,SCREEN_CONFIRM_END_MS


def edge_feedback_for(plan,elapsed_ms):
    empty={'edge':None,'intensity':0.,'tone':'green'}
    if plan is None or not math.isfinite(elapsed_ms) or elapsed_ms<0:return empty
    if plan.pattern in ('confirm_retract_hold','sim_frame_confirm_retract_hold'):
        framed=plan.pattern=='sim_frame_confirm_retract_hold'
        paired=plan.profile=='screen-coupled'
        confirm_end=SCREEN_CONFIRM_END_MS if paired else CONFIRM_END_MS
        intensity=(1. if elapsed_ms>=confirm_end else
                   completion_pose_at(elapsed_ms,framed=framed,screen_coupled=paired)[2]/(.20 if framed else .24))
        return {'edge':'all','intensity':round(intensity,5),'tone':'green'}
    if elapsed_ms>=plan.duration_ms:return empty
    offset=normalized_offset_for(plan,elapsed_ms)
    if plan.profile=='agent-coupled' and plan.pattern in ('agent_invite','agent_ack'):
        index=0 if plan.pattern=='agent_invite' else 1
        factor=.65 if index==0 else .3
        return {'edge':'all','intensity':round(abs(offset[index])/(factor*plan.amplitude),5),'tone':'blue' if index==0 else 'green'}
    if plan.pattern=='open_invitation':
        return {'edge':'all','intensity':round(abs(offset[0])/.65,5),'tone':'blue'}
    if plan.pattern not in ('double_sweep','double_sweep_smooth','agent_sweep'):return empty
    axis={'horizontal':0,'vertical':1,'depth':2}[plan.axis]
    edge=({'horizontal':('left','right'),'vertical':('top','bottom'),
           'depth':('all','all')}[plan.axis][int(plan.direction>0)])
    return {'edge':edge,'intensity':round(abs(offset[axis])/plan.amplitude,5),'tone':'green'}
