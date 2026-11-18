"""按仿真器中验证过的输入映射执行实体 XYZ 跟随。先完成审核过的连接初始化，再开始相机跟随。
必须提供已审核的实机配置及 --execute 参数，不自动重连。仿真运动学只供离线评估，不用于实机逆解。"""
import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

import numpy as np

from .simulator_spatial import SpatialSamples, input_target_3d
from .horizontal_follow import VisionReceiver, StateReceiver, DashboardReceiver, CommandChannel, check_dashboard
from .resident_follow import close_session, check_state
from .ur_rtde_transport import RTDEChannel

SIM_HOME = np.array([-.4919, -.1333, .2093])
HOME_JOINTS = [0., -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.]


def rotation_vector(v):
    v=np.array(v,dtype=float); angle=np.linalg.norm(v)
    if angle<1e-12:return np.eye(3)
    x,y,z=v/angle
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)


def orientation_error(a,b):
    r=rotation_vector(a).T@rotation_vector(b)
    return math.acos(float(np.clip((np.trace(r)-1)/2,-1,1)))


def nominal_fk(q,tcp):
    """UR5e标称DH：仅评估，不校准物理FK。"""
    t=np.eye(4)
    for angle,(a,alpha,d) in zip(q,[(0,math.pi/2,.1625),(-.425,0,0),
            (-.3922,0,0),(0,math.pi/2,.1333),(0,-math.pi/2,.0997),(0,0,.0996)]):
        c,s,ca,sa=math.cos(angle),math.sin(angle),math.cos(alpha),math.sin(alpha)
        t=t@np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d],[0,0,0,1]])
    tool=np.eye(4);tool[:3,3]=tcp[:3];tool[:3,:3]=rotation_vector(tcp[3:])
    return t@tool


@dataclass
class SpatialConfig:
    data: dict

    @classmethod
    def from_dict(cls,data):
        cfg=cls(data)
        for key in ('neutral_tcp_m_rad','neutral_joints_rad','tcp_offset_m_rad'):
            values=data[key]
            if len(values)!=6 or any(type(v) not in (float,int) or not math.isfinite(v) for v in values):
                raise ValueError('Invalid six-axis calibration')
        r=cfg.rotation
        if (r.shape!=(3,3) or not np.isfinite(r).all()
                or not np.allclose(r.T@r,np.eye(3),atol=1e-8) or abs(np.linalg.det(r)-1)>1e-8):
            raise ValueError('Mapping must be a proper rotation')
        for key,maximum in [('range_scale',1.),('speed_m_s',.18),('acceleration_m_s2',.6),
                            ('base_angle_deg',3.),('base_rate_deg_s',3.)]:
            value=data[key]
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=maximum:
                raise ValueError(f'Invalid {key}')
        if type(data.get('camera')) is not int or data['camera']<0:raise ValueError('Invalid camera')
        if not isinstance(data.get('robot_ip'),str) or not data['robot_ip']:raise ValueError('Missing robot host')
        return cfg

    @property
    def neutral(self):return self.data['neutral_tcp_m_rad']
    @property
    def joints(self):return self.data['neutral_joints_rad']
    @property
    def tcp(self):return self.data['tcp_offset_m_rad']
    @property
    def rotation(self):return np.array(self.data['rotation'],dtype=float)
    @property
    def speed(self):return self.data['speed_m_s']
    @property
    def acceleration(self):return self.data['acceleration_m_s2']
    @property
    def base_limit(self):return math.radians(self.data['base_angle_deg'])
    @property
    def base_rate(self):return math.radians(self.data['base_rate_deg_s'])

    def require_execution_review(self):
        fields=('home','path','direction','tool_payload','dynamics')
        if (self.data.get('status')!='PHYSICAL_XYZ_REVIEWED'
                or any(self.data.get('review',{}).get(k) is not True for k in fields)):
            raise ValueError('Physical home/path/direction/tool/dynamics review is incomplete')

    def target(self,command):
        if len(command)!=3 or not all(type(v) in (float,int) and math.isfinite(v) and -1<=v<=1 for v in command):
            raise ValueError('Invalid XYZ input')
        offset=np.array(input_target_3d(*command))-SIM_HOME
        return (np.array(self.neutral[:3])+self.rotation@offset*self.data['range_scale']).tolist()

    def bounds(self):
        # 保守的外框。实际目标映射保留了模拟器的弯曲可达性/深度投影；框是一个独立的外护栏。
        points=[np.array(self.neutral[:3])+self.rotation@np.array(p)*self.data['range_scale']
                for p in itertools.product([-.2481,.2919],[-.05,.05],[-.1593,.2481])]
        return np.min(points,axis=0),np.max(points,axis=0)

    def check_start(self,pose,joints,tcp):
        if (math.dist(pose[:3],self.neutral[:3])>.003
                or orientation_error(pose[3:],self.neutral[3:])>.01
                or max(abs(a-b) for a,b in zip(joints,self.joints))>.01):
            raise ValueError('Robot is not at the reviewed home; no automatic home move')
        if math.dist(tcp,self.tcp)>.0001:raise ValueError('TCP changed')

    def check_feedback(self,fb):
        pose,q,speed=fb['tcp_pose'],fb['joints'],fb['tcp_speed']
        if any(len(v)!=6 or not all(math.isfinite(x) for x in v) for v in (pose,q,speed)):
            raise ValueError('Invalid physical feedback')
        low,high=self.bounds()
        if np.any(np.array(pose[:3])<low-.001) or np.any(np.array(pose[:3])>high+.001):
            raise ValueError('Robot outside physical envelope')
        if orientation_error(pose[3:],self.neutral[3:])>.02:raise ValueError('Tool orientation changed')
        if abs(q[0]-self.joints[0])>self.base_limit+.002:raise ValueError('Base angle exceeded')
        if math.hypot(*speed[:3])>self.speed*1.15+.002 or math.hypot(*speed[3:])>.02:
            raise ValueError('Measured TCP speed exceeded')


class SpatialPolicy:
    """会话/ACK/丢失锁定器，携带已承诺的XYZ目标，而不是X。"""
    def __init__(self,cfg,session):
        self.cfg,self.session=cfg,session
        self.sequence=0;self.pending=None;self.need_hold=True;self.seen_loss=-1
        self.last_send=-1.;self.handshake_complete=False

    def next(self,now,pose,sample,feedback):
        if feedback['session']!=self.session:raise RuntimeError('Resident session changed')
        if feedback['state']<0:raise RuntimeError(f"Resident invariant fault {feedback['state']}")
        if feedback['state']==2 and self.handshake_complete:raise RuntimeError('Resident timeout; explicit restart required')
        if self.pending:
            sequence,sent,enabled=self.pending
            if now-sent>.2:raise RuntimeError('Resident acknowledgement timeout')
            if feedback['sequence']==sequence:
                self.pending=None
                if enabled==0 and feedback['state']==0:
                    self.need_hold=False;self.handshake_complete=True
        triple,stamp,good,loss=sample
        valid=triple is not None and good>=3 and 0<=now-stamp<=.3
        if loss!=self.seen_loss or not valid or feedback['state']==2:self.need_hold=True
        self.seen_loss=loss
        if self.pending or now-self.last_send<.02:return None
        enabled=int(valid and not self.need_hold)
        target=self.cfg.target(triple) if enabled else list(pose[:3])
        self.sequence+=1;self.pending=(self.sequence,now,enabled);self.last_send=now
        return self.sequence,enabled,self.session,target


def cartesian_velocity(cfg,position,target):
    """机器人的比例和制动信封的离线对应物。IK/关节限制是额外的，由实际控制器评估。
    350毫秒的备用时间包括250毫秒的守护程序加上排程余量。"""
    if len(position)!=3 or len(target)!=3 or not all(math.isfinite(v) for v in [*position,*target]):
        raise ValueError('Invalid target/position')
    v=np.array([4*(t-p) if abs(t-p)>.0002 else 0. for p,t in zip(position,target)])
    v*=min(1.,cfg.speed/max(1e-12,float(np.linalg.norm(v))))
    low,high=cfg.bounds();scale=1.
    for i in range(3):
        distance=high[i]-position[i] if v[i]>0 else position[i]-low[i]
        a=cfg.acceleration
        margin=min(.003,(high[i]-low[i])*.05)
        allowed=math.sqrt((a*.35)**2+2*a*max(0.,distance-margin))-a*.35
        scale=min(scale,allowed/max(1e-12,abs(v[i]))) if abs(v[i])>1e-12 else scale
    return (v*scale).tolist()


class SpatialVision(SpatialSamples,VisionReceiver):
    def __init__(self,camera):
        SpatialSamples.__init__(self)
        self.failure=None
        self.process=subprocess.Popen([sys.executable,'-u','-m','leica_tracker',
            '--camera',str(camera),'--emit-every-frames','1','--max-right-m','.10','--max-step-m','.002'],
            stdout=subprocess.PIPE,text=True,encoding='utf-8',bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()


def resident_program(cfg,session):
    if type(session) is not int or not 0<session<2**31:raise ValueError('Invalid session')
    def vector(v):return '['+','.join(f'{x:.10f}' for x in v)+']'
    low,high=cfg.bounds()
    base_angle_fault = (f'norm(q[0]-{cfg.joints[0]:.10f}) > {cfg.base_limit+.002:.10f} or '
                        if cfg.base_limit is not None else '')
    base_position_governor = (f'''      if base_rate > 0:
        base_scale = min(base_scale, max(0.0,({cfg.joints[0]+cfg.base_limit:.10f}-q[0])/(base_rate*0.35)))
      elif base_rate < 0:
        base_scale = min(base_scale, max(0.0,(q[0]-{cfg.joints[0]-cfg.base_limit:.10f})/(-base_rate*0.35)))
      end
''' if cfg.base_limit is not None else '')
    script = f'''def leica_resident_xyz():
  def in_box(p):
    return p[0] >= {low[0]:.10f} and p[0] <= {high[0]:.10f} and p[1] >= {low[1]:.10f} and p[1] <= {high[1]:.10f} and p[2] >= {low[2]:.10f} and p[2] <= {high[2]:.10f}
  end
  def smaller(a, b):
    if a < b:
      return a
    end
    return b
  end
  def larger(a, b):
    if a > b:
      return a
    end
    return b
  end
  global packet_age = 1.0
  thread heartbeat_clock():
    while True:
      global packet_age = packet_age + get_steptime()
      sync()
    end
  end
  clock_thread = run heartbeat_clock()
  neutral = p{vector(cfg.neutral)}
  expected_tcp = p{vector(cfg.tcp)}
  lower = {vector(low)}
  upper = {vector(high)}
  boundary_margin = {vector([min(.003,(b-a)*.05) for a,b in zip(low,high)])}
  previous = 0
  consumed = 0
  armed = False
  enabled = False
  moving = False
  target = get_actual_tcp_pose()
  write_output_integer_register(42, 2)
  write_output_integer_register(43, 0)
  write_output_integer_register(40, {session})
  while True:
    if packet_age > 0.25:
      armed = False
      enabled = False
    end
    seq = read_input_integer_register(42)
    flag = read_input_integer_register(43)
    token = read_input_integer_register(44)
    incoming = p[read_input_float_register(42),read_input_float_register(43),read_input_float_register(44),neutral[3],neutral[4],neutral[5]]
    seq_again = read_input_integer_register(42)
    if seq == seq_again and seq > previous:
      if token == {session} and (flag == 0 or flag == 1) and in_box(incoming):
        previous = seq
        consumed = seq
        global packet_age = 0.0
        if flag == 0:
          armed = True
          enabled = False
        elif armed:
          enabled = True
          target = incoming
        end
      else:
        armed = False
        enabled = False
      end
    end
    actual = get_actual_tcp_pose()
    actual_speed = get_actual_tcp_speed()
    q = get_actual_joint_positions()
    qd = get_actual_joint_speeds()
    offset = get_tcp_offset()
    relative = pose_trans(pose_inv(neutral), actual)
    fault = 0
    if not in_box(actual):
      fault = 1
    elif norm([relative[3],relative[4],relative[5]]) > 0.02:
      fault = 3
    elif norm([offset[0]-expected_tcp[0],offset[1]-expected_tcp[1],offset[2]-expected_tcp[2],offset[3]-expected_tcp[3],offset[4]-expected_tcp[4],offset[5]-expected_tcp[5]]) > 0.0001:
      fault = 4
    elif norm([actual_speed[0],actual_speed[1],actual_speed[2]]) > {cfg.speed*1.15+.002:.10f}:
      fault = 5
    elif norm([actual_speed[3],actual_speed[4],actual_speed[5]]) > 0.02:
      fault = 7
    elif {base_angle_fault}norm(qd[0]) > {cfg.base_rate*1.2:.10f}:
      fault = 8
    end
    i = 0
    while i < 6:
      if norm(qd[i]) > 1.15:
        fault = 9
      end
      i = i + 1
    end
    if fault != 0:
      textmsg("LEICA_FAULT_XYZ_", fault)
      stopl({max(.1,cfg.acceleration):.10f})
      write_output_integer_register(42, -fault)
      break
    end
    state = 0
    if not armed:
      state = 2
    end
    if enabled and armed:
      v = [4*(target[0]-actual[0]),4*(target[1]-actual[1]),4*(target[2]-actual[2])]
      i = 0
      while i < 3:
        if norm(target[i]-actual[i]) < 0.0002:
          v[i] = 0
        end
        i = i + 1
      end
      scale = min(1.0, {cfg.speed:.10f}/max(0.0000001,norm(v)))
      v = [v[0]*scale,v[1]*scale,v[2]*scale]
      scale = 1.0
      i = 0
      while i < 3:
        distance = actual[i]-lower[i]
        if v[i] > 0:
          distance = upper[i]-actual[i]
        end
        allowed = sqrt({(cfg.acceleration*.35)**2:.12f}+{2*cfg.acceleration:.10f}*max(0.0,distance-boundary_margin[i]))-{cfg.acceleration*.35:.10f}
        if norm(v[i]) > 0.0000001:
          scale = min(scale,allowed/norm(v[i]))
        end
        i = i + 1
      end
      v = [v[0]*scale,v[1]*scale,v[2]*scale]
      probe = p[actual[0]+v[0]*0.1,actual[1]+v[1]*0.1,actual[2]+v[2]*0.1,actual[3],actual[4],actual[5]]
      probe_in_box = in_box(probe)
      solution_valid = False
      if probe_in_box:
        known_pose = get_forward_kin(q, offset)
        known_error = pose_trans(pose_inv(probe), known_pose)
        if norm([known_error[0],known_error[1],known_error[2]]) <= 1e-10 and norm([known_error[3],known_error[4],known_error[5]]) <= 1e-10:
          q_next = q
          solution_valid = True
        elif get_inverse_kin_has_solution(probe, q, tcp=offset):
          q_next = get_inverse_kin(probe, q, tcp=offset)
          solution_valid = True
        end
      end
      if not probe_in_box or not solution_valid:
        textmsg("LEICA_FAULT_XYZ_REACH", probe)
        stopl({max(.1,cfg.acceleration):.10f})
        write_output_integer_register(42, -6)
        break
      end
      scale = 1.0
      i = 0
      while i < 6:
        rate = (q_next[i]-q[i])/0.1
        scale = min(scale,1.1/max(0.0000001,norm(rate)))
        i = i + 1
      end
      base_rate = (q_next[0]-q[0])/0.1
      base_scale = min(1.0,{cfg.base_rate:.10f}/max(0.0000001,norm(base_rate)))
{base_position_governor}      scale = min(scale,base_scale)
      v = [v[0]*scale,v[1]*scale,v[2]*scale]
      probe = p[actual[0]+v[0]*0.1,actual[1]+v[1]*0.1,actual[2]+v[2]*0.1,actual[3],actual[4],actual[5]]
      if not is_within_safety_limits(probe):
        textmsg("LEICA_FAULT_XYZ_SAFETY", probe)
        stopl({max(.1,cfg.acceleration):.10f})
        write_output_integer_register(42, -6)
        break
      end
      speedl([v[0],v[1],v[2],0,0,0], a={cfg.acceleration:.10f}, t=0.02)
      moving = True
      if norm(v) > 0.00001:
        state = 1
      end
    else:
      if moving:
        stopl({max(.1,cfg.acceleration):.10f})
        moving = False
      end
      sleep(0.02)
    end
    write_output_integer_register(42, state)
    write_output_integer_register(43, consumed)
  end
  kill clock_thread
end
'''
    return script.replace('min(', 'smaller(').replace('max(', 'larger(')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--speed-percent',type=float,default=100.,
                        help='Follow speed as 0<percent<=100 of the reviewed profile ceiling; home speed is separate')
    parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--stop-file',type=Path)
    args=parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration<0:parser.error('Invalid duration')
    if not math.isfinite(args.speed_percent) or not 0<args.speed_percent<=100:
        parser.error('Speed percent must be greater than zero and at most 100')
    cfg=SpatialConfig.from_dict(json.loads(args.profile.read_text(encoding='utf-8-sig')))
    cfg.data['speed_m_s']*=args.speed_percent/100.
    if not args.execute:
        print(json.dumps(dict(mode='offline_profile_preview',robot_connected=False,
            neutral=cfg.neutral,speed_m_s=cfg.speed,acceleration_m_s2=cfg.acceleration,
            target_home=cfg.target((0,0,0)),bounds=[v.tolist() for v in cfg.bounds()],
            initialization_on_connect=cfg.data.get('initialization',{}).get('on_connect',False),
            execution_reviewed=cfg.data.get('status')=='PHYSICAL_XYZ_REVIEWED')))
        return 0
    cfg.require_execution_review()
    if args.stop_file is None:parser.error('--stop-file is required for physical execution')
    if args.stop_file.exists():raise ValueError('Stop file exists; review before explicit restart')
    host=cfg.data['robot_ip'];check_dashboard(host,idle=True)
    states=dashboard=vision=rtde=channel=None;uploaded=False
    started=time.monotonic();sends=0;session=secrets.randbelow(2**30)+1
    try:
        states=StateReceiver(host);dashboard=DashboardReceiver(host)
        rtde=RTDEChannel(host,spatial=True)
        deadline=time.monotonic()+3
        while states.latest[0] is None:
            if states.failure or time.monotonic()>deadline:raise RuntimeError(states.failure or 'No state')
            time.sleep(.02)
        current,stamp=states.latest;fb,fs=rtde.latest;now=time.monotonic()
        check_state(current,stamp,cfg.data,now)
        if rtde.failure or now-fs>.15:raise RuntimeError(rtde.failure or 'Stale RTDE')
        from .physical_home import initialize_home
        measured_home=initialize_home(cfg,states,dashboard,rtde,args.stop_file)
        cfg.data['neutral_tcp_m_rad']=list(measured_home)
        vision=SpatialVision(cfg.data['camera'])
        deadline=time.monotonic()+90
        while vision.latest[1]==0:
            if vision.failure or time.monotonic()>deadline:raise RuntimeError(vision.failure or 'Camera startup timeout')
            if args.stop_file.exists():return 0
            time.sleep(.05)
        check_dashboard(host,idle=True)
        current,stamp=states.latest;fb,fs=rtde.latest;now=time.monotonic()
        check_state(current,stamp,cfg.data,now)
        if rtde.failure or now-fs>.15:raise RuntimeError(rtde.failure or 'Stale RTDE')
        cfg.check_start(fb['tcp_pose'],fb['joints'],current['tcp_offset'])
        channel=CommandChannel(host);uploaded=True
        channel.send(resident_program(cfg,session))
        deadline=time.monotonic()+3
        while True:
            fb,fs=rtde.latest
            if rtde.failure or states.failure or time.monotonic()>deadline:
                raise RuntimeError(rtde.failure or states.failure or 'Resident handshake timeout')
            if fb['session']==session and states.latest[0]['mode']['running']:break
            time.sleep(.02)
        policy=SpatialPolicy(cfg,session);started=time.monotonic();last_status=0.
        while not args.duration or time.monotonic()-started<args.duration:
            if args.stop_file.exists():break
            current,stamp=states.latest;sample=vision.latest;fb,fs=rtde.latest;now=time.monotonic()
            for resource in (states,dashboard,vision,rtde):
                if resource.failure:raise RuntimeError(resource.failure)
            if now-dashboard.checked_at>2 or not 0<=now-fs<=.15 or not current['mode']['running']:
                raise RuntimeError('Physical feedback stale or resident ended')
            check_state(current,stamp,cfg.data,now);cfg.check_feedback(fb)
            update=policy.next(now,fb['tcp_pose'],sample,fb)
            if update:rtde.send(*update);sends+=1
            if now-last_status>=.5:
                print(json.dumps(dict(event='physical_xyz_sync',simulated=False,session=session,
                    tracking=sample[0] is not None and sample[2]>=3 and 0<=now-sample[1]<=.3,
                    depth_calibrated=vision.baseline is not None,resident_state=fb['state'],
                    resident_running=current['mode']['running'],rtde_updates=sends,
                    tcp_pose=fb['tcp_pose'],joints=fb['joints'],tcp_speed=fb['tcp_speed'])),flush=True)
                last_status=now
            time.sleep(.01)
    finally:
        close_session(uploaded,host,channel,rtde,vision,states,dashboard,
                      stop_acceleration=max(.1,cfg.acceleration))
        print(json.dumps(dict(event='physical_xyz_finished',rtde_updates=sends,
            elapsed_s=time.monotonic()-started,stop_requested=uploaded)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
