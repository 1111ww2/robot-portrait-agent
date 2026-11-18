"""在原始 UR5e 归位平面路径上增加 X/Z 方向跟随，仅供仿真。"""
import json
import math
from .horizontal_follow import VisionSamples
from .simulator_follow import InputStream
from .simulator_fixed_base import HOME_X, LEFT_X, RIGHT_X, HORIZON, input_target

HOME_Z=.2093
MIN_Z=.05
MAX_Z=HOME_Z+(HOME_X-LEFT_X)
MAX_RADIUS=.79  # 肩、肘到腕部中心的半径应小于完全伸展时的距离。


def project(x,z):
    if not math.isfinite(x) or not math.isfinite(z):
        raise ValueError('Nonfinite simulator target')
    x=max(LEFT_X,min(RIGHT_X,x))
    ceiling=min(MAX_Z,math.sqrt(MAX_RADIUS**2-(x+.0997)**2)-.2157)
    return [x,max(MIN_Z,min(ceiling,z))]


def input_target_2d(right,down):
    if not math.isfinite(down):
        raise ValueError('Nonfinite vertical input')
    down=max(-1.,min(1.,down))
    gain=(HOME_X-LEFT_X) if down<0 else (RIGHT_X-HOME_X)
    return project(input_target(right),HOME_Z-down*gain)


def pose_at_xz(x,z):
    if (not all(math.isfinite(v) for v in [x,z]) or not LEFT_X-.003<=x<=RIGHT_X+.003
            or not MIN_Z-.003<=z<=MAX_Z+.003 or math.hypot(x+.0997,z+.2157)>MAX_RADIUS+.003):
        raise ValueError('Outside planar simulator workspace')
    a,b=.425,.3922
    px,pz=-(x+.0997),-(z+.2157)
    elbow=math.acos((px*px+pz*pz-a*a-b*b)/(2*a*b))
    shoulder=math.atan2(pz,px)-math.atan2(b*math.sin(elbow),a+b*math.cos(elbow))
    return [0.,shoulder,elbow,-math.pi/2-shoulder-elbow,-math.pi/2,0.]


def limited_vector(vector,limit):
    scale=min(1.,limit/max(1e-12,math.hypot(*vector)))
    return [v*scale for v in vector]


def joint_command_2d(joints,position,sample,now,previous_v,dt,max_speed):
    if (len(joints)!=6 or len(position)!=2 or len(previous_v)!=2
            or not all(math.isfinite(q) for q in [*joints,*position,*previous_v,now,dt,max_speed])
            or abs(joints[0])>math.radians(3) or abs(joints[4]+math.pi/2)>.001
            or abs(joints[5])>.001 or not 0<dt<=.2 or not 0<max_speed<=.18):
        raise ValueError('Invalid planar simulator state')
    pose_at_xz(*position)
    pair,stamp,good,_=sample
    if pair is None or good<3 or not 0<=now-stamp<=.3:
        return [0.]*6,[0.,0.]
    target=input_target_2d(*pair)
    error=[a-b for a,b in zip(target,position)]
    desired=limited_vector([4*e if abs(e)>.0002 else 0. for e in error],max_speed)
    change=limited_vector([a-b for a,b in zip(desired,previous_v)],.6*dt)
    velocity=limited_vector([a+b for a,b in zip(previous_v,change)],max_speed)
    nxt=[p+v*HORIZON for p,v in zip(position,velocity)]
    for i in range(2):
        if (target[i]-position[i])*(target[i]-nxt[i])<=0:nxt[i]=target[i]
    nxt=project(*nxt)
    # 将目标投影到允许的工作空间内，边界修正的位移不超过跟踪本身的位移预算。
    displacement=limited_vector([a-b for a,b in zip(nxt,position)],max_speed*HORIZON)
    nxt=[p+d for p,d in zip(position,displacement)]
    q=pose_at_xz(*nxt)
    rates=[(a-b)/HORIZON for a,b in zip(q,joints)]
    rates[0]=max(-math.radians(3),min(math.radians(3),rates[0]))
    rates[4]=rates[5]=0.
    scale=min(1.,1.1/max(1e-12,max(map(abs,rates))))
    return [v*scale for v in rates],[d/HORIZON*scale for d in displacement]


class PlanarSamples(VisionSamples):
    def accept(self,line,now):
        event=json.loads(line)
        stamp=event['source_monotonic_ns']
        if isinstance(stamp,bool) or not isinstance(stamp,int) or stamp<=self.previous:
            raise ValueError('Camera source timestamp stopped advancing')
        self.previous=stamp
        source_time=stamp/1e9
        if source_time>now+.001:raise ValueError('Camera source clock is invalid')
        pair=None
        try:
            command=event['command']
            values=[command['right'],command['down']]
            if (event['event']=='tracking_target' and event['tracking'] is True
                    and event['motion_intent_m']['hold'] is False and now-source_time<=.3
                    and all(not isinstance(v,bool) and isinstance(v,(float,int))
                            and math.isfinite(v) and -1<=v<=1 for v in values)):
                pair=tuple(float(v) for v in values)
        except (KeyError,TypeError):
            pass
        good=self.latest[2]+1 if pair is not None else 0
        if pair is None:self.loss_generation+=1
        self.latest=(pair,source_time,good,self.loss_generation)


class PlanarInput(InputStream):
    accept=PlanarSamples.accept
