"""两种 e-Series 机型共用有限提示动作的节奏。相位时间由控制器推进；
若逆解要求降低关节速率，同步放慢相位，使动画与实际动作一致。此处不连接设备，不做安装验收或实体路径审核。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GestureCadence:
    duration_s: float = 3.3
    repetitions: int = 2
    joint_speed_limit: float = .7

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value,dict) or set(value)!={'duration_s','repetitions','joint_speed_limit'}
                or type(value['repetitions']) is not int or not 1<=value['repetitions']<=4
                or any(type(value[k]) not in (float,int) or not math.isfinite(value[k]) or value[k]<=0
                       for k in ('duration_s','joint_speed_limit'))):
            raise ValueError('Invalid canonical gesture configuration')
        return cls(float(value['duration_s']),value['repetitions'],float(value['joint_speed_limit']))

    def as_dict(self):
        return dict(duration_s=self.duration_s,repetitions=self.repetitions,joint_speed_limit=self.joint_speed_limit)

    def sample(self, elapsed):
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
            raise ValueError('Invalid canonical gesture time')
        t=min(elapsed,self.duration_s)
        angle=2*math.pi*self.repetitions*t/self.duration_s
        return (.5-.5*math.cos(angle),math.pi*self.repetitions/self.duration_s*math.sin(angle))


def canonical_dynamics(max_displacement, *, duration_s=3.3,repetitions=2,joint_speed_limit=.7):
    """笛卡尔跟踪上限根据周期计算，不能直接套用以弧度/秒表示的速率。"""
    cadence=GestureCadence.from_dict(dict(duration_s=duration_s,repetitions=repetitions,joint_speed_limit=joint_speed_limit))
    if type(max_displacement) not in (int,float) or not math.isfinite(max_displacement) or max_displacement<=0:
        raise ValueError('Positive finite gesture displacement required')
    return dict(gesture=cadence.as_dict(),
                speed=1.15*max_displacement*math.pi*repetitions/duration_s,
                acceleration=1.15*2*max_displacement*(math.pi*repetitions/duration_s)**2,
                base_rate=joint_speed_limit)


def finite_gesture_program(program,cfg,session):
    """扩展常驻程序的显式 flag2，保留 flag0 和 flag1 的原有含义。"""
    cadence=cfg.gesture
    if cadence is None:return program
    def replace(old,new):
        nonlocal program
        if program.count(old)!=1:raise RuntimeError('Native gesture resident structure changed')
        program=program.replace(old,new)
    duration=f'{cadence.duration_s:.10f}';cycles=f'{2*cadence.repetitions:.10f}'
    cap=f'{cadence.joint_speed_limit:.10f}'
    replace('  global packet_age = 1.0\n',
        '  global gesture_age = 0.0\n  global gesture_scale = 1.0\n  global gesture_active = False\n'
        '  global recovery_stop_done = True\n'
        '  global packet_age = 1.0\n')
    replace('  clock_thread = run heartbeat_clock()\n',
        '  thread recovery_brake():\n'
        f'    stopl({max(.1,cfg.acceleration):.10f})\n'
        '    global recovery_stop_done = True\n  end\n'
        '  clock_thread = run heartbeat_clock()\n')
    replace('  target = get_actual_tcp_pose()\n',
        '  target = get_actual_tcp_pose()\n  gesture_mode = False\n  gesture_endpoint = neutral\n  recovery_paused = False\n')
    replace('      global packet_age = packet_age + get_steptime()\n',
        '      global packet_age = packet_age + get_steptime()\n'
        '      if gesture_active:\n        global gesture_age = gesture_age + get_steptime()*gesture_scale\n      end\n')
    replace(f'token == {session} and (flag == 0 or flag == 1)',f'token == {session} and (flag == 0 or flag == 1 or flag == 2 or flag == 3)')
    replace('          enabled = False\n          ever_armed = True\n',
        '          enabled = False\n          ever_armed = True\n'
        '          gesture_mode = False\n          global gesture_active = False\n'
        '          recovery_paused = not recovery_stop_done\n')
    replace('        if flag == 0:\n',
        '        if flag == 3:\n'
        '          enabled = False\n          ever_armed = True\n'
        '          recovery_paused = True\n          global gesture_active = False\n'
        '        elif flag == 0:\n')
    replace('        elif armed:\n','        elif armed and recovery_stop_done:\n')
    replace('          enabled = True\n          target = incoming\n',
        '          enabled = True\n          recovery_paused = False\n'
        '          if flag == 2:\n'
        '            if not gesture_mode:\n'
        '              gesture_mode = True\n              gesture_endpoint = incoming\n'
        '              global gesture_age = 0.0\n              global gesture_scale = 1.0\n'
        '              global gesture_active = True\n'
        '            elif norm([incoming[0]-gesture_endpoint[0],incoming[1]-gesture_endpoint[1],incoming[2]-gesture_endpoint[2]]) > 0.000001:\n'
        '              armed = False\n              enabled = False\n'
        '            else:\n'
        f'              global gesture_active = gesture_age < {duration}\n'
        '            end\n'
        '          else:\n            gesture_mode = False\n            global gesture_active = False\n            target = incoming\n          end\n')
    replace('    if enabled and armed:\n',
        '    if recovery_paused:\n      state = 4\n'
        '      if moving:\n'
        '        global recovery_stop_done = False\n'
        '        recovery_brake_thread = run recovery_brake()\n'
        '        moving = False\n      end\n'
        '    end\n    if enabled and armed:\n')
    # 不可恢复的故障必须终止负责运动的线程并结束会话，即使故障发生时流程正处于暂停状态。
    replace('      if ever_armed:\n',
        '      if ever_armed:\n'
        '        if not recovery_stop_done:\n          kill recovery_brake_thread\n'
        f'          stopl({max(.1,cfg.acceleration):.10f})\n        end\n')
    replace('    if fault != 0:\n',
        '    if fault != 0:\n'
        '      if not recovery_stop_done:\n        kill recovery_brake_thread\n      end\n')
    old='      v = [4*(target[0]-actual[0]),4*(target[1]-actual[1]),4*(target[2]-actual[2])]\n'
    new=('      feedforward = [0,0,0]\n'
         '      if gesture_mode:\n'
         f'        phase_time = smaller(gesture_age,{duration})\n'
         f'        fraction = 0.5-0.5*cos({cycles}*3.141592653589793*phase_time/{duration})\n'
         f'        derivative = {cadence.repetitions:.10f}*3.141592653589793/{duration}*sin({cycles}*3.141592653589793*phase_time/{duration})\n'
         f'        if gesture_age >= {duration}:\n          fraction = 0.0\n          derivative = 0.0\n          global gesture_active = False\n        end\n'
         '        target = p[neutral[0]+(gesture_endpoint[0]-neutral[0])*fraction,neutral[1]+(gesture_endpoint[1]-neutral[1])*fraction,neutral[2]+(gesture_endpoint[2]-neutral[2])*fraction,neutral[3],neutral[4],neutral[5]]\n'
         '        feedforward = [(gesture_endpoint[0]-neutral[0])*derivative,(gesture_endpoint[1]-neutral[1])*derivative,(gesture_endpoint[2]-neutral[2])*derivative]\n'
         '      end\n'
         '      tracking_gain = 4.0\n'
         '      if gesture_mode:\n        tracking_gain = 8.0\n      end\n'
         '      v = [feedforward[0]+tracking_gain*(target[0]-actual[0]),feedforward[1]+tracking_gain*(target[1]-actual[1]),feedforward[2]+tracking_gain*(target[2]-actual[2])]\n'
         '      gesture_correcting = False\n'
         '      if gesture_mode:\n'
         '        i = 0\n        while i < 3:\n'
         '          if (v[i] < 0 and actual[i]-lower[i] <= boundary_margin[i] and target[i] > actual[i]) or (v[i] > 0 and upper[i]-actual[i] <= boundary_margin[i] and target[i] < actual[i]):\n'
         '            gesture_correcting = True\n          end\n'
         '          i = i+1\n        end\n'
         '        if gesture_correcting:\n'
         '          v = [tracking_gain*(target[0]-actual[0]),tracking_gain*(target[1]-actual[1]),tracking_gain*(target[2]-actual[2])]\n'
         '        end\n      end\n')
    replace(old,new)
    # 有限轨迹已由控制器规划好速度降为零再反向的过程。
    # 保留实际制动距离边界、逆解和关节限速，不再为连续运动额外预留 350 毫秒，否则会拖慢正常的向外摆动。
    braking=(f'        allowed = sqrt({(cfg.acceleration*.35)**2:.12f}+'
             f'{2*cfg.acceleration:.10f}*larger(0.0,distance-boundary_margin[i]))-'
             f'{cfg.acceleration*.35:.10f}\n')
    replace(braking,
        '        if gesture_mode:\n'
        f'          allowed = sqrt({2*cfg.acceleration:.10f}*larger(0.0,distance-boundary_margin[i]))\n'
        '        else:\n'+braking.replace('        allowed','          allowed')+
        '        end\n')
    # 当有限轨迹的位置误差接近零时，说明跟踪正常，不能据此清除非零前馈速度。原位置死区仅用于目标跟随和末端稳定。
    replace('        if norm(target[i]-actual[i]) < 0.0002:\n',
            '        if norm(target[i]-actual[i]) < 0.0002 and (not gesture_mode or not gesture_active):\n')
    # 基座位置限制按实际 100 毫秒的逆解探测窗口计算。350 毫秒的连续运动预留会拖慢已知的换向过程；
    # 每次探测仍须遵守原有基座范围和关节速率上限。
    if cfg.base_limit is not None:
        replace('      base_rate = (q_next[0]-q[0])/0.1\n',
                '      base_lookahead = 0.35\n'
                '      if gesture_mode:\n        base_lookahead = 0.1\n      end\n'
                '      base_rate = (q_next[0]-q[0])/0.1\n')
        replace('(base_rate*0.35)','(base_rate*base_lookahead)')
        replace('(-base_rate*0.35)','(-base_rate*base_lookahead)')
    velocity='      v = [v[0]*scale,v[1]*scale,v[2]*scale]\n'
    if program.count(velocity)!=3:raise RuntimeError('Native gesture speed scaling changed')
    first=program.index(velocity)
    # 即使固定的轨迹目标在内侧，受限前馈也可能指向外侧。
    # 保持轨迹相位不变，只用误差项校正位置，仍执行速度、笛卡尔空间、逆解和关节约束。
    # 向外可动距离为零不能同时卡住相位和位置。
    phase_scale=('      if gesture_mode:\n'
                 '        if gesture_correcting:\n          global gesture_scale = 0.0\n'
                 '        else:\n          global gesture_scale = scale\n        end\n      end\n')
    program=program[:first]+program[first:].replace(velocity,velocity+phase_scale,1)
    # 后续每项限幅都以相同的比例缩放控制器相位。
    offset=first+len(velocity)+len(phase_scale)
    program=program[:offset]+program[offset:].replace(velocity,velocity+'      if gesture_mode:\n        global gesture_scale = gesture_scale*scale\n      end\n')
    replace('scale = smaller(scale,1.1/larger(0.0000001,norm(rate)))',f'scale = smaller(scale,{cap}/larger(0.0000001,norm(rate)))')
    replace('if norm(qd[i]) > 1.15:',f'if norm(qd[i]) > {cadence.joint_speed_limit*1.05+.002:.10f}:')
    replace('      if norm(v) > 0.00001:\n        state = 1\n      end\n',
        '      if norm(v) > 0.00001:\n        state = 1\n      end\n'
        f'      if gesture_mode and gesture_age >= {duration} and norm([actual[0]-neutral[0],actual[1]-neutral[1],actual[2]-neutral[2]]) <= 0.0005 and norm([actual_speed[0],actual_speed[1],actual_speed[2]]) < 0.0005:\n'
        '        state = 3\n      end\n')
    return program
