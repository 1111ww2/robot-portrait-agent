"""仅用于教学仿真器：从原始归位姿态跟随目标，基座提供有限辅助。
解析路径仅适用 UR5e 的原位逆解分支和向下 278.6 毫米的 RG6 TCP，不能用作通用逆解器或实机控制器。"""
import argparse
import json
import math
from pathlib import Path
import socket
import threading
import time

from .simulator_follow import InputStream, get_json, verify_identity

HOME_X = -.4919
LEFT_X = -.74
RIGHT_X = -.20
HORIZON = .10


class SimJointStream:
    """直接发送有限时长的指令并读空教学仿真器的二进制反馈。URX 在反馈循环内发送，存在状态到指令的延迟；
    本地专用流用于避免该延迟，权威状态仍通过仿真器 HTTP API 独立核对。"""
    def __init__(self):
        verify_identity(get_json('/api/workbench/v1/manifest'))
        self.socket=socket.create_connection(('127.0.0.1',30002),timeout=.5)
        self.socket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.closed=threading.Event()
        self.reader=threading.Thread(target=self._drain,daemon=True)
        self.reader.start()

    def _drain(self):
        while not self.closed.is_set():
            try:
                if not self.socket.recv(65536):
                    break
            except socket.timeout:
                continue
            except OSError:
                break
        self.closed.set()

    def speedj(self,rates,**kwargs):
        if self.closed.is_set():
            raise RuntimeError('Simulator stream disconnected')
        values=','.join(f'{q:.9f}' for q in rates)
        self.socket.sendall(f'speedj([{values}], 0.8, {HORIZON})\n'.encode('ascii'))

    def stopj(self,acceleration):
        self.socket.sendall(b'stopj(0.8)\n')

    def close(self):
        self.closed.set()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        self.reader.join(timeout=1.)


def pose_at_x(x):
    if not math.isfinite(x) or not LEFT_X-.002 <= x <= RIGHT_X+.002:
        raise ValueError('Outside fixed-base simulator corridor')
    # 腕部固定向下后，只需求解肩、肘两连杆的平面问题。
    a, b = .425, .3922
    px, pz = -(x+.0997), -.425
    elbow = math.acos((px*px+pz*pz-a*a-b*b)/(2*a*b))
    shoulder = math.atan2(pz,px)-math.atan2(b*math.sin(elbow),a+b*math.cos(elbow))
    return [0., shoulder, elbow, -math.pi/2-shoulder-elbow, -math.pi/2, 0.]


def input_target(right):
    if not math.isfinite(right):
        raise ValueError('Invalid tracking input')
    # 已在实际 Chrome 相机视图中核对，相机画面向右对应基座 −X。
    right = -max(-1.,min(1.,right))
    return HOME_X + right*((RIGHT_X-HOME_X) if right>=0 else (HOME_X-LEFT_X))


def joint_command(joints, x, sample, now, previous_v, dt, max_speed):
    if (len(joints)!=6 or not all(math.isfinite(q) for q in joints)
            or abs(joints[0])>math.radians(3) or abs(joints[4]+math.pi/2)>.001
            or abs(joints[5])>.001 or not LEFT_X-.002<=x<=RIGHT_X+.002):
        raise ValueError('Fixed-base simulator pose changed')
    if not all(math.isfinite(v) for v in [now,previous_v,dt,max_speed]) or not 0<dt<=.2 or not 0<max_speed<=.18:
        raise ValueError('Invalid simulator timing or speed')
    right, stamp, good, _ = sample
    if right is None or good<3 or not 0<=now-stamp<=.3:
        return [0.]*6, 0.
    target = input_target(right)
    error=target-x
    desired=max(-max_speed,min(max_speed,4*error)) if abs(error)>.0002 else 0.
    v=max(previous_v-.6*dt,min(previous_v+.6*dt,desired))
    # 有限指令制动时不能越过任一端点或当前目标。先约束位置，再求解固定基座的路径。
    next_x=max(LEFT_X,min(RIGHT_X,x+v*HORIZON))
    if (target-x)*(target-next_x)<=0:
        next_x=target
    q=pose_at_x(next_x)
    rates=[(a-b)/HORIZON for a,b in zip(q,joints)]
    # 以原位作为首选基座角度，允许小幅朝原位修正；不能通过持续增大基座旋转来延伸通道。
    rates[0]=max(-math.radians(3),min(math.radians(3),rates[0]))
    rates[4]=rates[5]=0.
    scale=min(1.,1.1/max(1e-12,max(map(abs,rates))))
    return [rate*scale for rate in rates], (next_x-x)/HORIZON*scale


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed-percent',type=float,default=100.)
    parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--vertical',action='store_true',help='Enable bounded simulator X/Z following')
    parser.add_argument('--depth',action='store_true',help='Enable calibrated reverse depth following; implies vertical')
    args=parser.parse_args(argv)
    args.vertical=args.vertical or args.depth
    if not math.isfinite(args.speed_percent) or not 0<args.speed_percent<=100:
        parser.error('Speed percent must be in (0,100]')
    if not math.isfinite(args.duration) or args.duration<0:
        parser.error('Duration must be nonnegative')
    instance=verify_identity(get_json('/api/workbench/v1/manifest'))
    state=get_json('/api/sim/v1/state')
    if (state['urx']['clients'] or state['urx']['program_running']
            or max(abs(a-b) for a,b in zip(state['joints'],pose_at_x(HOME_X)))>.002
            or state['active_tcp_offset']!=[0.,0.,.2786,0.,0.,0.]):
        raise RuntimeError('Fixed-base following requires idle original home and original RG6 TCP')
    robot=None
    if args.vertical:
        from .simulator_planar import PlanarInput, joint_command_2d, HOME_Z, MIN_Z, MAX_Z
    if args.depth:
        from .simulator_spatial import SpatialInput, joint_command_3d
    samples=SpatialInput() if args.depth else (PlanarInput() if args.vertical else InputStream())
    previous_v=[0.,0.,0.] if args.depth else ([0.,0.] if args.vertical else 0.)
    controller='limited_base_spatial' if args.depth else ('limited_base_planar' if args.vertical else 'limited_base')
    started=last=time.monotonic()
    count=0
    try:
        robot=SimJointStream()
        print(json.dumps(dict(event='simulator_connected',simulated=True,controller=controller,
            neutral_tcp=state['tcp_pose'],left_x=LEFT_X,right_x=RIGHT_X,
            base_assistance_limit_deg=3.,base_assistance_rate_deg_s=3.,
            speed_percent=args.speed_percent,max_speed_m_s=.18*args.speed_percent/100)),flush=True)
        while not args.duration or time.monotonic()-started<args.duration:
            if args.stop_file and args.stop_file.exists():
                break
            tick=time.monotonic()
            if samples.failure:
                raise RuntimeError(samples.failure)
            state=get_json('/api/sim/v1/motion')
            if (state.get('instance_id')!=instance or state.get('backend')!='sim'
                    or state.get('service')!='urx-browser-simulator' or state['urx']['last_error']):
                raise RuntimeError('Simulator identity or controller error')
            pose=state['tcp_pose']
            if abs(pose[1]+.1333)>(.055 if args.depth else .04) or (not args.vertical and abs(pose[2]-.2093)>.002):
                raise RuntimeError('Simulator left configured path')
            sample=samples.latest
            command_fn=joint_command_3d if args.depth else (joint_command_2d if args.vertical else joint_command)
            position=pose[:3] if args.depth else ([pose[0],pose[2]] if args.vertical else pose[0])
            rates,previous_v=command_fn(state['joints'],position,sample,time.monotonic(),
                previous_v,max(.001,min(.1,tick-last)),.18*args.speed_percent/100)
            robot.speedj(rates,acc=.8,min_time=HORIZON)
            last=tick
            count+=1
            right=sample[0][0] if args.vertical and sample[0] is not None else sample[0]
            down=sample[0][1] if args.vertical and sample[0] is not None else None
            toward=sample[0][2] if args.depth and sample[0] is not None else None
            print(json.dumps(dict(event='simulator_sync',simulated=True,controller=controller,
                sequence=count,right=right,down=down,toward_subject=toward,
                depth_calibrated=getattr(samples,'baseline',None) is not None,
                depth_baseline=getattr(samples,'baseline',None),
                tracking=sample[0] is not None and sample[2]>=3 and 0<=time.monotonic()-sample[1]<=.3,
                tcp_pose=pose,joints=state['joints'],joint_velocity_command=rates,
                x_offset_cm=(pose[0]-HOME_X)*100,y_offset_cm=(pose[1]+.1333)*100,z_offset_cm=(pose[2]-.2093)*100,
                source_age_ms=round((time.monotonic()-sample[1])*1000,2))),flush=True)
            time.sleep(max(0.,.05-(time.monotonic()-tick)))
    finally:
        if robot is not None:
            try:
                robot.stopj(.8)
            finally:
                robot.close()
        print(json.dumps(dict(event='simulator_finished',simulated=True,updates=count)),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
