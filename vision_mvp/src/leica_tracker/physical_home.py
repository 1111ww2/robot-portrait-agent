"""通过经过审查的联合路线进行明确开始初始化，永远不会自动重新连接。目标安全限值检查不是碰撞/路径验证。
需要确切的已知开始、经过审查的中间路标和新的遥测数据。"""
import copy
import json
import math
import secrets
import time

from .horizontal_follow import CommandChannel, check_dashboard, send_script
from .resident_follow import check_state


def joint_vector(q):
    return (isinstance(q,(list,tuple)) and len(q)==6
            and all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=2*math.pi for v in q))


def at_rest_at_home(q,tcp_speed,home):
    return (joint_vector(q) and joint_vector(home) and len(tcp_speed)==6
            and all(math.isfinite(v) for v in tcp_speed)
            and max(abs(a-b) for a,b in zip(q,home))<=.003
            and math.hypot(*tcp_speed[:3])<.001
            and math.hypot(*tcp_speed[3:])<.005)


def prepare_route(profile,current_q):
    home=profile['neutral_joints_rad']
    if not joint_vector(home) or not joint_vector(current_q):raise ValueError('Invalid home/start joints')
    if max(abs(a-b) for a,b in zip(current_q,home))<=.003:return None
    route=copy.deepcopy(profile.get('initialization',{}))
    if route.get('on_connect') is not True:raise ValueError('Robot is not at home; initialization disabled')
    if route.get('path_reviewed') is not True:raise ValueError('Initialization transition path is not reviewed')
    start=route.get('start_joints_rad')
    if not joint_vector(start) or max(abs(a-b) for a,b in zip(start,current_q))>.01:
        raise ValueError('Current joints differ from the reviewed route start')
    points=route.get('waypoints_rad')
    if (not isinstance(points,list) or not 1<=len(points)<=20 or not all(joint_vector(q) for q in points)
            or max(abs(a-b) for a,b in zip(points[-1],home))>1e-8):
        raise ValueError('Reviewed route must end at the configured home')
    for key,maximum in [('joint_speed_rad_s',.1),('joint_acceleration_rad_s2',.1),('timeout_s',600.)]:
        value=route.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=maximum:
            raise ValueError(f'Invalid initialization {key}')
    minimum=sum(max(abs(a-b) for a,b in zip(q0,q1))/route['joint_speed_rad_s']
                for q0,q1 in zip([start]+points[:-1],points))
    if route['timeout_s']<minimum+3:raise ValueError('Initialization timeout is shorter than route travel time')
    return route


def home_program(route,tcp,session, *, recovery_pause=False):
    if type(session) is not int or not 0<session<2**31:raise ValueError('Invalid initialization session')
    if len(tcp)!=6 or not all(math.isfinite(v) for v in tcp):raise ValueError('Invalid TCP')
    def vector(q):return '['+','.join(f'{v:.10f}' for v in q)+']'
    moves=[]
    for index,q in enumerate(route['waypoints_rad']):
        segment=f'''    if not is_within_safety_limits({vector(q)}):
      global motion_error = 3
      return False
    end
    movej({vector(q)}, a={route['joint_acceleration_rad_s2']:.10f}, v={route['joint_speed_rad_s']:.10f}, r=0)'''
        if recovery_pause:
            segment=(f'    if waypoint_index <= {index}:\n'+
                     '\n'.join('  '+line for line in segment.splitlines())+
                     f'\n      global waypoint_index = {index+1}\n    end')
        moves.append(segment)
    checkpoint='  global waypoint_index = 0\n' if recovery_pause else ''
    return f'''def leica_initialize_home():
{checkpoint}  global packet_age = 0.0
  global motion_done = False
  global motion_error = 0
  thread timer():
    while True:
      global packet_age = packet_age + get_steptime()
      sync()
    end
  end
  thread route_motion():
{chr(10).join(moves)}
    global motion_done = True
  end
  expected_start = {vector(route['start_joints_rad'])}
  expected_tcp = p{vector(tcp)}
  current = get_actual_joint_positions()
  i = 0
  while i < 6:
    if norm(current[i]-expected_start[i]) > 0.01:
      textmsg("LEICA_FAULT_HOME_START", current)
      write_output_integer_register(40, {session})
      write_output_integer_register(42, -10)
      return False
    end
    i = i + 1
  end
  clock_thread = run timer()
  started = False
  previous = 0
  consumed = 0
  write_output_integer_register(42, 100)
  write_output_integer_register(43, 0)
  write_output_integer_register(40, {session})
  while True:
    seq = read_input_integer_register(42)
    flag = read_input_integer_register(43)
    token = read_input_integer_register(44)
    seq_again = read_input_integer_register(42)
    valid = False
    fault = 0
    if seq == seq_again and seq > previous:
      if token == {session} and flag == 1:
        previous = seq
        consumed = seq
        global packet_age = 0.0
        valid = True
      elif token == {session}:
        fault = 11
      end
    end
    if (started and packet_age > 0.25) or (not started and packet_age > 2.0):
      fault = 12
    end
    actual_tcp = get_tcp_offset()
    if norm([actual_tcp[0]-expected_tcp[0],actual_tcp[1]-expected_tcp[1],actual_tcp[2]-expected_tcp[2],actual_tcp[3]-expected_tcp[3],actual_tcp[4]-expected_tcp[4],actual_tcp[5]-expected_tcp[5]]) > 0.0001:
      fault = 13
    end
    qd = get_actual_joint_speeds()
    i = 0
    while i < 6:
      if norm(qd[i]) > {route['joint_speed_rad_s']*1.3+.002:.10f}:
        fault = 14
      end
      i = i + 1
    end
    if motion_error != 0:
      fault = 15
    end
    if fault != 0:
      if started and not motion_done:
        kill motion_thread
      end
      write_output_integer_register(42, -fault)
      stopj(0.2)
      textmsg("LEICA_FAULT_HOME_", fault)
      break
    end
    if valid and not started:
      motion_thread = run route_motion()
      started = True
      write_output_integer_register(42, 101)
    end
    write_output_integer_register(43, consumed)
    if motion_done:
      write_output_integer_register(42, 102)
      break
    end
    sync()
  end
  kill clock_thread
end
'''


def initialize_home(cfg,states,dashboard,rtde,stop_file,clock=time.monotonic,sleep=time.sleep):
    """在明确启动期间最多运行一次，在构建相机之前。拥有其关节停止清理。调用者在初始化成功之前不得标记以下程序上传。"""
    def snapshot():
        current,stamp=states.latest;fb,feedback_time=rtde.latest;now=clock()
        for resource in (states,dashboard,rtde):
            if resource.failure:raise RuntimeError(resource.failure)
        if not 0<=now-feedback_time<=.15 or now-dashboard.checked_at>2:
            raise RuntimeError('Initialization telemetry stale')
        check_state(current,stamp,cfg.data,now)
        if stop_file.exists():raise RuntimeError('Initialization cancelled by stop file')
        return current,fb

    current,fb=snapshot()
    route=prepare_route(cfg.data,fb['joints'])
    if route is None:
        if not at_rest_at_home(fb['joints'],fb['tcp_speed'],cfg.joints):
            raise ValueError('Robot must be stationary at home')
        cfg.check_start(fb['tcp_pose'],fb['joints'],current['tcp_offset'])
        print(json.dumps(dict(event='physical_home_ready',moved=False,tcp_pose=fb['tcp_pose'])),flush=True)
        return fb['tcp_pose']
    if math.hypot(*fb['tcp_speed'][:3])>=.001 or math.hypot(*fb['tcp_speed'][3:])>=.005:
        raise ValueError('Initialization requires a stationary robot')
    host=cfg.data['robot_ip'];check_dashboard(host,idle=True)
    current,fb=snapshot();prepare_route(cfg.data,fb['joints'])
    session=secrets.randbelow(2**30)+1
    channel=None;uploaded=False;finished=False
    start=clock();sequence=0;pending=None;last_send=-1.;settled_since=None;last_status=-1.
    try:
        channel=CommandChannel(host);uploaded=True
        channel.send(home_program(route,cfg.tcp,session))
        while clock()-start<route['timeout_s']:
            current,fb=snapshot();now=clock()
            if fb['session']!=session:
                if now-start>3:raise RuntimeError('Initialization handshake timeout')
                sleep(.01);continue
            if fb['state']<0:raise RuntimeError(f"Initialization fault {fb['state']}")
            if fb['state']==102:
                if at_rest_at_home(fb['joints'],fb['tcp_speed'],cfg.joints) and not current['mode']['running']:
                    settled_since=now if settled_since is None else settled_since
                    if now-settled_since>=.15:
                        cfg.check_start(fb['tcp_pose'],fb['joints'],current['tcp_offset'])
                        finished=True
                        print(json.dumps(dict(event='physical_home_ready',moved=True,tcp_pose=fb['tcp_pose'])),flush=True)
                        return fb['tcp_pose']
                else:settled_since=None
                sleep(.01);continue
            if fb['state'] not in (100,101):raise RuntimeError('Unexpected initialization state')
            if pending:
                if fb['sequence']==pending[0]:pending=None
                elif now-pending[1]>.2:raise RuntimeError('Initialization acknowledgement timeout')
            if pending is None and now-last_send>=.05:
                sequence+=1;rtde.send(sequence,1,session,list(fb['tcp_pose'][:3]))
                pending=(sequence,now);last_send=now
            if now-last_status>=1:
                print(json.dumps(dict(event='physical_home_initializing',state=fb['state'],
                    joints=fb['joints'],elapsed_s=now-start)),flush=True)
                last_status=now
            sleep(.01)
        raise TimeoutError('Initialization did not settle before timeout')
    finally:
        try:
            if uploaded and not finished:
                try:channel.send('stopj(0.2)')
                except (OSError,RuntimeError):
                    channel.close();send_script(host,'stopj(0.2)')
        finally:
            if channel is not None:channel.close()
