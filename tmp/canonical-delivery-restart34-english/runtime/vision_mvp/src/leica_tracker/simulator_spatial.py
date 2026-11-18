"""仅在仿真中进行限幅 XYZ 跟随，用相对肩宽估计深度。"""
import json
import math
import statistics
from collections import deque
from .simulator_follow import InputStream
from .simulator_fixed_base import HOME_X,LEFT_X,RIGHT_X,HORIZON
from .simulator_planar import HOME_Z,MIN_Z,MAX_Z,MAX_RADIUS,input_target_2d,limited_vector,PlanarSamples

HOME_Y=-.1333
BASE_LIMIT=math.radians(3)


def project3(x,y,z):
    if not all(math.isfinite(v) for v in [x,y,z]):raise ValueError('Invalid spatial target')
    x=max(LEFT_X,min(RIGHT_X,x))
    edges=[(x*math.sin(t)-.1333)/math.cos(t) for t in [-BASE_LIMIT,BASE_LIMIT]]
    y=max(max(HOME_Y-.05,min(edges)),min(min(HOME_Y+.05,max(edges)),y))
    u=-math.sqrt(x*x+y*y-.1333**2)
    ceiling=min(MAX_Z,math.sqrt(MAX_RADIUS**2-(u+.0997)**2)-.2157)
    return [x,y,max(MIN_Z,min(ceiling,z))]


def input_target_3d(right,down,toward):
    if not math.isfinite(toward):raise ValueError('Invalid depth input')
    x,z=input_target_2d(right,down)
    # 在已核对的视角中，屏幕右方对应 −X，上方对应 +Z，朝观看者对应 +Y。
    return project3(x,HOME_Y+.05*max(-1,min(1,toward)),z)


def pose_at_xyz(x,y,z):
    if (not all(math.isfinite(v) for v in [x,y,z]) or not LEFT_X-.003<=x<=RIGHT_X+.003
            or abs(y-HOME_Y)>.053 or not MIN_Z-.003<=z<=MAX_Z+.003):
        raise ValueError('Outside spatial simulator corridor')
    u=-math.sqrt(x*x+y*y-.1333**2)
    base=(math.atan2(y,x)-math.atan2(-.1333,u)+math.pi)%(2*math.pi)-math.pi
    if abs(base)>BASE_LIMIT+.003 or math.hypot(u+.0997,z+.2157)>MAX_RADIUS+.003:
        raise ValueError('Spatial simulator joint envelope exceeded')
    a,b=.425,.3922;px,pz=-(u+.0997),-(z+.2157)
    elbow=math.acos((px*px+pz*pz-a*a-b*b)/(2*a*b))
    shoulder=math.atan2(pz,px)-math.atan2(b*math.sin(elbow),a+b*math.cos(elbow))
    return [base,shoulder,elbow,-math.pi/2-shoulder-elbow,-math.pi/2,base]


def joint_command_3d(joints,position,sample,now,previous_v,dt,max_speed):
    if (len(joints)!=6 or len(position)!=3 or len(previous_v)!=3
            or not all(math.isfinite(v) for v in [*joints,*position,*previous_v,now,dt,max_speed])
            or abs(joints[0])>BASE_LIMIT+.003 or abs(joints[4]+math.pi/2)>.001
            or abs(joints[5]-joints[0])>.002 or not 0<dt<=.2 or not 0<max_speed<=.18):
        raise ValueError('Invalid spatial simulator state')
    pose_at_xyz(*position)
    triple,stamp,good,_=sample
    if triple is None or good<3 or not 0<=now-stamp<=.3:return [0.]*6,[0.,0.,0.]
    target=input_target_3d(*triple)
    desired=limited_vector([4*(a-b) if abs(a-b)>.0002 else 0. for a,b in zip(target,position)],max_speed)
    change=limited_vector([a-b for a,b in zip(desired,previous_v)],.6*dt)
    velocity=limited_vector([a+b for a,b in zip(previous_v,change)],max_speed)
    nxt=[p+v*HORIZON for p,v in zip(position,velocity)]
    for i in range(3):
        if (target[i]-position[i])*(target[i]-nxt[i])<=0:nxt[i]=target[i]
    nxt=project3(*nxt)
    displacement=limited_vector([a-b for a,b in zip(nxt,position)],max_speed*HORIZON)
    nxt=[p+d for p,d in zip(position,displacement)]
    q=pose_at_xyz(*nxt)
    rates=[(a-b)/HORIZON for a,b in zip(q,joints)]
    rates[4]=0.
    scale=min(1.,1.1/max(1e-12,max(map(abs,rates))),BASE_LIMIT/max(1e-12,abs(rates[0])))
    return [v*scale for v in rates],[d/HORIZON*scale for d in displacement]


class SpatialSamples(PlanarSamples):
    def __init__(self):
        PlanarSamples.__init__(self)
        self.baseline=None
        self.widths=deque(maxlen=40)
        self.smoothed=None

    def accept(self,line,now):
        PlanarSamples.accept(self,line,now)
        pair,stamp,good,loss=self.latest
        if pair is None:
            if self.baseline is None:self.widths.clear()
            self.smoothed=None
            return
        observation=json.loads(line).get('image_observation',{})
        width=observation.get('shoulder_width')
        confidence=observation.get('confidence',0)
        if (isinstance(width,bool) or not isinstance(width,(float,int)) or not math.isfinite(width)
                or not .01<width<1 or not isinstance(confidence,(float,int)) or not .6<=confidence<=1):
            self.latest=(None,stamp,0,loss+1)
            return
        if self.baseline is None:
            self.widths.append((stamp,width))
            if len(self.widths)==40 and stamp-self.widths[0][0]>=1.:
                values=[v for _,v in self.widths]
                median=statistics.median(values)
                if max(values)-min(values)<median*.08:self.baseline=median
        self.smoothed=width if self.smoothed is None else .15*width+.85*self.smoothed
        relative=0. if self.baseline is None else (self.baseline-self.smoothed)/self.baseline
        depth=0. if abs(relative)<.05 else max(-1.,min(1.,relative/.30))
        self.latest=((*pair,depth),stamp,good,loss)


class SpatialInput(InputStream):
    def __init__(self):
        SpatialSamples.__init__(self)
        InputStream.__init__(self)
    accept=SpatialSamples.accept
