"""使用机器人侧的心跳守护程序审查了一次性打开路线。该模块从不连接到硬件或上传程序。
所有路标、当前姿势和动力学都必须来自对当前物理安装的审查。
呼叫者必须使用独立可停止的RTDE门和单独的仅接收遥测通道；ACK不是安全等级的停止。
不使用任何工具电源、夹具、故障后自动返回或隐式路径。"""
import math

from .physical_home import home_program
from .physical_robot_model import model_for_profile
from .reference_gate import rotation_vector_distance


def _vector(value, name):
    if (not isinstance(value,(tuple,list)) or len(value)!=6 or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in value)):
        raise ValueError('Invalid '+name)
    return tuple(float(v) for v in value)


class IntroProgram:
    """编译一个经过审查的开头；只有明确的空值才能禁用TCP监视器。"""
    __slots__=('robot_model','reference_joints','reference_pose','tcp','waypoints','joint_speed',
               'joint_acceleration','max_tcp_angular_speed','max_tcp_linear_speed',
               'timeout','session','_review_frozen')

    def __setattr__(self,name,value):
        if getattr(self,'_review_frozen',False):
            raise AttributeError('Reviewed physical program is immutable')
        object.__setattr__(self,name,value)

    def __delattr__(self,name):
        if getattr(self,'_review_frozen',False):
            raise AttributeError('Reviewed physical program is immutable')
        object.__delattr__(self,name)

    def __copy__(self):
        return self

    def __deepcopy__(self,memo):
        return self

    def __init__(self,profile,session):
        if not isinstance(profile,dict) or type(session) is not int or not 0<session<2**31:
            raise ValueError('Explicit profile and session required')
        self.robot_model=model_for_profile(profile)
        reviewed_keys=('reference_joints','reference_pose','tcp','waypoints','joint_speed',
                       'joint_acceleration','max_tcp_angular_speed','max_tcp_linear_speed','timeout')
        if 'operator_selection' in profile:
            from .operator_design import validate_operator_selection, INTRO_SELECTION_FIELDS
            if not validate_operator_selection(profile,kind='intro',fields=INTRO_SELECTION_FIELDS):
                raise ValueError('Invalid operator-selected opening')
        else:
            required=('robot','mount','tcp','payload','path','dynamics','stop','gesture')
            review=profile.get('review')
            if not isinstance(review,dict) or any(review.get(k) is not True for k in required):
                raise ValueError('Opening route review incomplete')
            reviewed=profile.get('reviewed')
            if (not isinstance(reviewed,dict) or set(reviewed)!=set(reviewed_keys) or
                    any(reviewed[key]!=profile.get(key) for key in reviewed_keys)):
                raise ValueError('Opening review does not match current route and dynamics')
        self.reference_joints=_vector(profile.get('reference_joints'),'reference joints')
        self.reference_pose=_vector(profile.get('reference_pose'),'reference pose')
        self.tcp=_vector(profile.get('tcp'),'TCP')
        waypoints=profile.get('waypoints')
        if not isinstance(waypoints,list) or not 2<=len(waypoints)<=20:
            raise ValueError('Reviewed outgoing and return waypoints required')
        self.waypoints=tuple(_vector(p,'waypoint') for p in waypoints)
        if (max(abs(a-b) for a,b in zip(self.waypoints[-1],self.reference_joints))>1e-8 or
                all(max(abs(a-b) for a,b in zip(q,self.reference_joints))<=1e-8 for q in self.waypoints)):
            raise ValueError('Reviewed opening must move and end at reference')
        for key in ('joint_speed','joint_acceleration','max_tcp_angular_speed','max_tcp_linear_speed','timeout'):
            value=profile.get(key)
            if key in ('max_tcp_angular_speed','max_tcp_linear_speed') and key in profile and value is None:
                # 缺失的字段绝不能在无声的情况下禁用此监控器。
                setattr(self,key,None)
                continue
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Explicit finite positive '+key+' required')
            setattr(self,key,float(value))
        minimum=sum(max(abs(a-b) for a,b in zip(a0,a1))/self.joint_speed for a0,a1 in
                    zip((self.reference_joints,)+self.waypoints[:-1],self.waypoints))
        if self.timeout<minimum+3:
            raise ValueError('Opening timeout shorter than reviewed route')
        self.session=session
        self._review_frozen=True

    def program(self):
        route={'start_joints_rad':self.reference_joints,'waypoints_rad':self.waypoints,
               'joint_speed_rad_s':self.joint_speed,'joint_acceleration_rad_s2':self.joint_acceleration}
        # 明确的单步预览子类保留了其一次性程序和单独审查的停止设置。
        # 恢复属于这个完整的打开路线，而不是那些单独的手动预览段落。
        recovery_pause=type(self) is IntroProgram
        result=home_program(route,self.tcp,self.session,recovery_pause=recovery_pause)
        marker='    qd = get_actual_joint_speeds()\n    i = 0\n'
        if result.count(marker)!=1:raise RuntimeError('Resident route format changed; review required')
        guard='    actual_speed = get_actual_tcp_speed()\n'
        if self.max_tcp_linear_speed is not None:
            guard+=(f'    if norm([actual_speed[0],actual_speed[1],actual_speed[2]]) > {self.max_tcp_linear_speed:.10f}:\n'
                    f'      fault = 16\n    end\n')
        if self.max_tcp_angular_speed is not None:
            guard+=(f'    if norm([actual_speed[3],actual_speed[4],actual_speed[5]]) > {self.max_tcp_angular_speed:.10f}:\n'
                    f'      fault = 17\n    end\n')
        result=result.replace(marker,guard+marker)
        declaration='  started = False\n  previous = 0\n'
        old=(f'      if token == {self.session} and flag == 1:\n'
             '        previous = seq\n        consumed = seq\n'
             '        global packet_age = 0.0\n        valid = True\n')
        new=(f'      if token == {self.session} and (flag == 0 or flag == 1 or flag == 3):\n'
             '        previous = seq\n        consumed = seq\n'
             '        global packet_age = 0.0\n'
             '        if flag == 3:\n          pause_requested = True\n'
             '        elif flag == 0 and not started:\n          armed = True\n'
             '          if pause_stop_done:\n            paused = False\n'
             '            write_output_integer_register(42, 100)\n          end\n'
             '        elif flag == 1 and armed:\n          valid = True\n'
             '        else:\n          fault = 11\n        end\n')
        if not recovery_pause:
            new=(f'      if token == {self.session} and (flag == 0 or flag == 1):\n'
                 '        previous = seq\n        consumed = seq\n'
                 '        global packet_age = 0.0\n'
                 '        if flag == 0 and not started:\n          armed = True\n'
                 '        elif flag == 1 and armed:\n          valid = True\n'
                 '        else:\n          fault = 11\n        end\n')
        if result.count(declaration)!=1 or result.count(old)!=1:
            raise RuntimeError('Resident handshake format changed; review required')
        if not recovery_pause:
            return result.replace(declaration,'  started = False\n  armed = False\n  previous = 0\n').replace(old,new)
        result=result.replace(declaration,'  started = False\n  armed = False\n  paused = False\n  previous = 0\n').replace(old,new)
        result=result.replace('  clock_thread = run timer()\n',
            '  global pause_stop_done = True\n'
            '  thread pause_stop():\n'
            f'    stopj({self.joint_acceleration:.10f})\n'
            '    global pause_stop_done = True\n  end\n'
            '  clock_thread = run timer()\n')
        result=result.replace('    valid = False\n    fault = 0\n',
                              '    valid = False\n    pause_requested = False\n    fault = 0\n')
        result=result.replace('    if fault != 0:\n      if started and not motion_done:\n',
                              '    if fault != 0:\n'
                              '      if not pause_stop_done:\n        kill brake_thread\n      end\n'
                              '      if started and not motion_done and not paused:\n')
        start='    if valid and not started:\n'
        pause=('    if pause_requested and not paused:\n'
               '      if started and not motion_done:\n        kill motion_thread\n      end\n'
               '      paused = True\n      global pause_stop_done = False\n'
               '      brake_thread = run pause_stop()\n'
               '    end\n'
               '    if valid and paused:\n'
               '      if pause_stop_done:\n'
               '        paused = False\n'
               '        if started and not motion_done:\n'
               '          motion_thread = run route_motion()\n'
               '          write_output_integer_register(42, 101)\n'
               '        end\n'
               '      else:\n        valid = False\n      end\n'
               '    end\n')
        if result.count(start)!=1:raise RuntimeError('Resident start structure changed')
        result=result.replace(start,pause+start)
        completion='    if motion_done:\n      write_output_integer_register(42, 102)\n      break\n    end\n'
        paused_completion=('    if paused:\n      write_output_integer_register(42, 103)\n'
                           '    elif motion_done:\n      write_output_integer_register(42, 102)\n      break\n    end\n')
        if result.count(completion)!=1:raise RuntimeError('Resident completion structure changed')
        return result.replace(completion,paused_completion)


class IntroProgress:
    """心跳和路由后测量返回率；调用者会序列化通话。"""
    def __init__(self,program,gate):
        if not isinstance(program,IntroProgram) or not callable(getattr(gate,'send',None)) or not callable(getattr(gate,'stop',None)):
            raise ValueError('Reviewed program and stoppable resident gate required')
        self.program,self.gate=program,gate
        self.state='STARTING';self.sequence=0;self.pending=None;self.started=None
        self.last_writer=None;self.last_reader=None;self.steady_since=None
        self.armed_ack=False;self.started_ack=False;self.completion_seen=None
        self.primary_stopped_at=None
        self.ack_diagnostic=None

    def _fault(self,reason):
        self.state='FAULT';self.gate.stop(reason);raise RuntimeError(reason)

    def stop(self):
        self.state='STOPPED';self.gate.stop('Physical intro stopped')

    def tick(self,writer, *, writer_sampled_at,reader,reader_sampled_at,now,primary_running=True):
        if self.state in ('FAULT','STOPPED','COMPLETE'):
            raise RuntimeError('Intro is no longer active')
        try:
            if (type(now) not in (int,float) or not math.isfinite(now) or
                    any(type(t) not in (int,float) or not math.isfinite(t) or not 0<=now-t<=.15
                        for t in (writer_sampled_at,reader_sampled_at)) or
                    self.last_writer is not None and writer_sampled_at<=self.last_writer or
                    self.last_reader is not None and reader_sampled_at<=self.last_reader):
                raise ValueError('Stale or repeated intro telemetry')
            if self.started is None:self.started=now
            if not isinstance(writer,dict) or writer.get('session')!=self.program.session or not isinstance(reader,dict):
                raise ValueError('Intro resident session mismatch')
            state=writer.get('state');ack=writer.get('sequence')
            if type(state) is int and state < 0:
                reason={-12:'controller input heartbeat expired',
                        -16:'TCP linear speed exceeds reviewed limit',
                        -17:'TCP angular speed exceeds reviewed limit'}.get(state,'robot-side guard')
                raise ValueError(f'Intro resident fault {state}: {reason}')
            if now-self.started>self.program.timeout:raise ValueError('Intro route timeout')
            if type(state) is not int or type(ack) is not int or state not in (100,101,102,103):
                raise ValueError('Intro resident fault or invalid state')
            q=_vector(reader.get('joints'),'joint telemetry')
            qd=_vector(reader.get('joint_speed'),'joint speed')
            pose=_vector(reader.get('tcp_pose'),'TCP pose')
            speed=_vector(reader.get('tcp_speed'),'TCP speed')
            if ((self.program.max_tcp_linear_speed is not None and
                     math.hypot(*speed[:3])>self.program.max_tcp_linear_speed) or
                    (self.program.max_tcp_angular_speed is not None and
                     math.hypot(*speed[3:])>self.program.max_tcp_angular_speed) or
                    max(abs(v) for v in qd)>self.program.joint_speed*1.3+.002):
                raise ValueError('Intro speed guard')
            near_start=(max(abs(a-b) for a,b in zip(q,self.program.reference_joints))<=.003 and
                  math.dist(pose[:3],self.program.reference_pose[:3])<=.0005 and
                  rotation_vector_distance(pose[3:],self.program.reference_pose[3:])<=.005)
            final_joints=getattr(self.program,'final_joints',self.program.reference_joints)
            final_pose=getattr(self.program,'final_pose',self.program.reference_pose)
            final_tolerance=getattr(self.program,'joint_tolerance',.003)
            near_final=(max(abs(a-b) for a,b in zip(q,final_joints))<=final_tolerance and
                        (final_pose is None or (math.dist(pose[:3],final_pose[:3])<=.0005 and
                         rotation_vector_distance(pose[3:],final_pose[3:])<=.005)))
            still=(math.hypot(*speed[:3])<.0005 and math.hypot(*speed[3:])<.002 and
                   max(abs(v) for v in qd)<.003)
            if self.last_reader is None and (state not in (100,103) or not near_start or not still):
                raise ValueError('Intro is not at its reviewed stationary start')
            if self.last_reader is not None and reader_sampled_at-self.last_reader>.15:self.steady_since=None
            self.last_writer=writer_sampled_at;self.last_reader=reader_sampled_at
            if self.pending is not None:
                seq,sent,enabled=self.pending
                if ack==seq:
                    self.pending=None
                    if state==103:
                        # 门可能会重写任何恢复HOLD的请求。其ACK既不能证明原始臂也不能证明运动开始。
                        pass
                    elif enabled:
                        if state not in (101,102):raise ValueError('Intro enabled before running state')
                        self.started_ack=True
                    else:
                        if state!=100:raise ValueError('Intro hold acknowledgement has unexpected state')
                        self.armed_ack=True
                elif state!=102 and now-sent>.2:
                    self.ack_diagnostic={'expected_sequence':seq,'observed_sequence':ack,
                        'elapsed_s':now-sent,'resident_state':state,
                        'writer_age_s':now-writer_sampled_at,'reader_age_s':now-reader_sampled_at}
                    raise ValueError(f'Intro resident acknowledgement timeout: expected={seq}, observed={ack}, elapsed={now-sent:.3f}s')
            if type(primary_running) is not bool:
                raise ValueError('Intro Primary running state unavailable')
            if self.completion_seen is not None and state!=102:
                raise ValueError('Intro completion state regressed')
            if self.primary_stopped_at is not None and primary_running:
                raise ValueError('Intro resident restarted during completion observation')
            if not primary_running:
                if state!=102 and (not self.started_ack or not near_final or not still):
                    raise ValueError('Intro resident no longer running before stationary endpoint')
                if self.primary_stopped_at is None:self.primary_stopped_at=now
            if (self.primary_stopped_at is not None and self.completion_seen is None
                    and now-self.primary_stopped_at>.2):
                raise ValueError('Intro completion observation timed out')
            if state!=102 and self.primary_stopped_at is not None:
                # 不要在这里关闭共享写入者/接收者：这将隐藏延迟完成注册表。
                # 这种单向状态会抑制所有进一步的发送，而独立的访客监视器和不变的会话、速度、新鲜度和ACK检查仍然保持活跃。
                # 等待102不会延长现有的.2秒ACK预算。
                self.state='AWAITING_COMPLETION';return self.state
            if state==102:
                if not self.started_ack:
                    raise ValueError('Intro completed without a start acknowledgement')
                # 状态和位置通过独立的RTDE通道到达。不要从完成前的一个样本中判断最终姿势。
                if self.completion_seen is None:
                    self.completion_seen=now;self.steady_since=None;self.state='SETTLING';return self.state
                if now-self.completion_seen>2.:
                    raise ValueError('Intro did not return to a stationary reference')
                if reader_sampled_at<=self.completion_seen:
                    self.state='SETTLING';return self.state
                if not near_final or not still:
                    # movej完成不是一个有衡量的停滞保证。不要发送更多的启用；
                    # 保留速度/新鲜度检查，同时允许伺服器/独立流短暂稳定。
                    self.steady_since=None
                    self.state='SETTLING';return self.state
                if self.steady_since is None:self.steady_since=reader_sampled_at
                elif reader_sampled_at-self.steady_since>=.6:
                    self.state='COMPLETE';return self.state
                self.state='SETTLING';return self.state
            if state==103:
                self.steady_since=None
                self.state='PAUSED'
            else:self.state='ACTIVE' if self.started_ack else 'STARTING'
            if self.pending is None:
                enabled=int(self.armed_ack)
                self.sequence+=1;self.gate.send(self.sequence,enabled,self.program.session,0.)
                self.pending=(self.sequence,now,enabled)
            return self.state
        except (ValueError,RuntimeError,TypeError,KeyError) as exc:
            self._fault(str(exc))
