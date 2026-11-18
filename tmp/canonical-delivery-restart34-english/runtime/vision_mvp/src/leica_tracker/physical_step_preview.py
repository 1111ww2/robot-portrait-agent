"""对已审查的候选关节位置进行明确的一路预览。每个Next都会上传一个已审查的片段，并等待测量静止。
没有步骤自动开始，Stop永远不会命令返回。模拟器中的候选人将在每个实际安装片段被审查之前保持草稿状态。
此代码尚未在物理硬件上投入使用。"""
import copy
import json
import ipaddress
import math
import threading,time

from .physical_intro_bootstrap import open_intro
from .physical_intro_program import IntroProgram, _vector
from .physical_robot_model import model_for_profile
from .physical_resident_bootstrap import _dependencies, _state_snapshot
from .reference_gate import rotation_vector_distance
from types import MappingProxyType, SimpleNamespace


def _stop_settings(value):
    keys=('resident_timeout','stopj_acceleration','watch_interval','watch_timeout')
    if (not isinstance(value,dict) or set(value)!=set(keys) or
            any(type(value[k]) not in (int,float) or not math.isfinite(value[k]) or value[k]<=0 for k in keys) or
            value['resident_timeout']>.3 or value['watch_timeout']>.3 or
            value['watch_interval']>=value['watch_timeout']):
        raise ValueError('Explicit reviewed stop settings invalid')
    return {k:float(value[k]) for k in keys}


def _candidate_model(draft):
    model=model_for_profile(draft)
    if model_for_profile(draft['intro'])!=model:
        raise ValueError('Candidate draft and intro robot models differ')
    return model


class StepProgram(IntroProgram):
    """一个明确审查的联合空间段；没有隐式返回。"""
    __slots__=('robot_ip','final_joints','final_pose','joint_tolerance','stop_settings')
    def __init__(self,profile,session):
        if not isinstance(profile,dict) or type(session) is not int or not 0<session<2**31:
            raise ValueError('Explicit step profile and session required')
        if profile.get('status')!='STEP_REVIEWED':raise ValueError('Step is an unreviewed draft')
        self.robot_model=model_for_profile(profile)
        required=('robot','mount','tcp','payload','path','dynamics','stop','gesture')
        review=profile.get('review')
        if not isinstance(review,dict) or any(review.get(k) is not True for k in required):
            raise ValueError('Step route review incomplete')
        keys=('robot_ip','reference_joints','reference_pose','tcp','target_joints','joint_tolerance','joint_speed',
              'joint_acceleration','max_tcp_angular_speed','max_tcp_linear_speed','timeout')
        if 'stop' in profile:keys=(*keys,'stop')
        approved=profile.get('reviewed')
        if (not isinstance(approved,dict) or set(approved)!=set(keys) or
                any(approved[k]!=profile.get(k) for k in keys)):
            raise ValueError('Step review does not match route and dynamics')
        try:self.robot_ip=str(ipaddress.ip_address(profile.get('robot_ip')))
        except ValueError:raise ValueError('Explicit reviewed robot IP required') from None
        self.reference_joints=_vector(profile.get('reference_joints'),'start joints')
        self.reference_pose=_vector(profile.get('reference_pose'),'start pose')
        self.tcp=_vector(profile.get('tcp'),'TCP')
        self.final_joints=_vector(profile.get('target_joints'),'target joints')
        self.final_pose=None
        separation=max(abs(a-b) for a,b in zip(self.final_joints,self.reference_joints))
        if separation<=1e-8:
            raise ValueError('Step target equals start')
        tolerance=profile.get('joint_tolerance')
        if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or not 0<tolerance<separation or tolerance>.003:
            raise ValueError('Explicit reviewed endpoint tolerance must be smaller than step separation')
        self.joint_tolerance=float(tolerance)
        self.waypoints=(self.final_joints,)
        for key in ('joint_speed','joint_acceleration','max_tcp_angular_speed','max_tcp_linear_speed','timeout'):
            value=profile.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Explicit finite positive '+key+' required')
            setattr(self,key,float(value))
        minimum=max(abs(a-b) for a,b in zip(self.final_joints,self.reference_joints))/self.joint_speed
        if self.timeout<minimum+3:raise ValueError('Step timeout shorter than reviewed segment')
        self.session=session
        self.stop_settings=MappingProxyType(_stop_settings(profile['stop'])) if 'stop' in profile else None
        self._review_frozen=True

    def program(self):
        result=super().program()
        if self.stop_settings is None:return result
        substitutions=(('packet_age > 0.25',f"packet_age > {self.stop_settings['resident_timeout']:.10f}"),
                       ('stopj(0.2)',f"stopj({self.stop_settings['stopj_acceleration']:.10f})"))
        for old,new in substitutions:
            if result.count(old)!=1:raise RuntimeError('Step resident stop settings changed; review required')
            result=result.replace(old,new)
        return result


def open_step(host,profile,read_guest,**kwargs):
    """上传一个保留的步骤；执行需要随后的明确勾选。"""
    program=StepProgram(profile,1)
    try:host=str(ipaddress.ip_address(host))
    except ValueError:raise ValueError('Explicit robot IP required') from None
    if host!=program.robot_ip:raise ValueError('Robot IP differs from reviewed step')
    if program.stop_settings:
        for key in ('watch_interval','watch_timeout'):
            value=program.stop_settings[key]
            if key in kwargs and kwargs[key]!=value:raise ValueError('Watchdog does not match reviewed stop settings')
            kwargs[key]=value
    return open_intro(host,profile,read_guest,_program_type=StepProgram,**kwargs)


class SharedPreviewPlan:
    """一套经过明确审查的14个段落和通用参数。这是一份普通的本地审查记录，而不是签署的安全证书。
    它无法证明任何外部审查实际上已经进行了。
    每个生成的步骤都会根据新的测量机器人状态重新验证，实际步骤引导在上传程序之前重复该检查。"""
    _fields=('status','robot_ip','reference_joints','reference_pose','tcp','waypoints',
             'dynamics','stop','review','operator_approval')
    _reviews=('robot','mount','tcp','payload','path','dynamics','stop','gesture')
    _dynamics=('joint_tolerance','joint_speed','joint_acceleration',
               'max_tcp_angular_speed','max_tcp_linear_speed','timeout')

    __slots__=('robot_model','host','initial','initial_pose','tcp','points','dynamics','stop','review',
               '_explicit_model','_draft_json','_plan_json','_review_frozen')

    def __setattr__(self,name,value):
        if getattr(self,'_review_frozen',False):
            raise AttributeError('Reviewed preview plan is immutable')
        object.__setattr__(self,name,value)

    def __delattr__(self,name):
        if getattr(self,'_review_frozen',False):
            raise AttributeError('Reviewed preview plan is immutable')
        object.__delattr__(self,name)

    @staticmethod
    def _canonical(value):
        # 该计划来自JSON；规范序列化会冻结嵌套的审查证据，并在非JSON/非有限值上无法关闭。
        return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

    def __init__(self,draft,plan):
        if (not isinstance(draft,dict) or draft.get('status')!='DRAFT' or
                not isinstance(draft.get('intro'),dict) or not isinstance(plan,dict)):
            raise ValueError('Explicit candidate draft and reviewed plan required')
        self.robot_model=_candidate_model(draft)
        if model_for_profile(plan)!=self.robot_model:
            raise ValueError('Reviewed preview plan and candidate robot models differ')
        self._explicit_model='robot_model' in draft or 'robot_model' in plan
        if (plan.get('status')!='PREVIEW_PLAN_REVIEWED' or plan.get('operator_approval')!='accept_all_14' or
                not isinstance(plan.get('review'),dict) or
                any(plan['review'].get(k) is not True for k in self._reviews)):
            raise ValueError('Full 14-segment operator review required')
        approved=plan.get('reviewed')
        if (not isinstance(approved,dict) or set(approved)!=set(self._fields) or
                any(approved[k]!=plan.get(k) for k in self._fields)):
            raise ValueError('Reviewed plan no longer matches its inputs')
        intro=draft['intro']
        if (plan.get('robot_ip')!=draft.get('robot_ip') or
                plan.get('reference_joints')!=intro.get('reference_joints') or
                plan.get('reference_pose')!=intro.get('reference_pose') or
                plan.get('tcp')!=intro.get('tcp') or plan.get('waypoints')!=intro.get('waypoints')):
            raise ValueError('Reviewed plan differs from the selected candidate draft')
        try:self.host=str(ipaddress.ip_address(plan['robot_ip']))
        except (ValueError,TypeError):raise ValueError('Explicit reviewed robot IP required') from None
        self.initial=_vector(plan.get('reference_joints'),'reviewed start joints')
        self.initial_pose=_vector(plan.get('reference_pose'),'reviewed start pose')
        self.tcp=_vector(plan.get('tcp'),'reviewed TCP')
        if not isinstance(plan.get('waypoints'),list) or len(plan['waypoints'])!=14:
            raise ValueError('Exactly 14 reviewed points required')
        self.points=tuple(_vector(q,'reviewed candidate') for q in plan['waypoints'])
        dynamics=copy.deepcopy(plan.get('dynamics'))
        if not isinstance(dynamics,dict) or set(dynamics)!=set(self._dynamics):
            raise ValueError('Explicit common dynamics required')
        self.dynamics=MappingProxyType(dynamics)
        self.stop=MappingProxyType(_stop_settings(plan.get('stop')))
        self.review=MappingProxyType(copy.deepcopy(plan['review']))
        self._draft_json=self._canonical(draft);self._plan_json=self._canonical(plan)
        # 在打开任何设备之前，请验证每个片段及其生成的程序。
        for index in range(14):
            candidate=self._profile(index,self.initial_pose)
            StepProgram(candidate,1).program()
        self._review_frozen=True

    def matches(self,draft,plan):
        """检测任何磁盘上的更改；旧的评论永远不会遵循新的值。"""
        try:return self._canonical(draft)==self._draft_json and self._canonical(plan)==self._plan_json
        except (TypeError,ValueError):return False

    def _profile(self,index,reference_pose):
        if type(index) is not int or not 0<=index<14:raise ValueError('Invalid preview index')
        start=self.initial if index==0 else self.points[index-1]
        profile={'status':'STEP_REVIEWED','robot_ip':self.host,'reference_joints':list(start),
                 'reference_pose':list(reference_pose),'tcp':list(self.tcp),
                 'target_joints':list(self.points[index]),**dict(self.dynamics),
                 'stop':dict(self.stop),'review':dict(self.review)}
        keys=('robot_ip','reference_joints','reference_pose','tcp','target_joints','joint_tolerance',
              'joint_speed','joint_acceleration','max_tcp_angular_speed','max_tcp_linear_speed','timeout','stop')
        profile['reviewed']={k:copy.deepcopy(profile[k]) for k in keys}
        if self._explicit_model:
            profile.update(robot_model=self.robot_model,reviewed_robot_model=self.robot_model)
        return profile

    def preflight_profile(self):
        """只有在启动访客之前才验证；这不会读取机器人。"""
        return self._profile(0,self.initial_pose)

    def step_profile(self,index,observations,current, *, now):
        """在两个新的只读源之后，才生成一个经过审查的步骤。"""
        if type(index) is not int or not 0<=index<14 or not isinstance(observations,list) or len(observations)!=index:
            raise ValueError('Completed observations do not match the requested step')
        if not isinstance(current,dict) or type(now) not in (int,float) or not math.isfinite(now):
            raise ValueError('Fresh current robot observation required')
        start=self.initial if index==0 else self.points[index-1]
        expected_pose=self.initial_pose
        if index:
            previous_time=0.
            for position,observation in enumerate(observations,1):
                if not isinstance(observation,dict) or observation.get('index')!=position:
                    raise ValueError('Previous observation identity mismatch')
                observed_joints=_vector(observation.get('joints'),'previous observed joints')
                expected_pose=_vector(observation.get('tcp_pose'),'previous observed TCP pose')
                tolerance=observation.get('joint_tolerance');stamp=observation.get('sampled_at')
                if (type(tolerance) not in (int,float) or not math.isfinite(tolerance) or
                        tolerance!=self.dynamics['joint_tolerance'] or
                        max(abs(a-b) for a,b in zip(observed_joints,self.points[position-1]))>tolerance or
                        type(stamp) not in (int,float) or not math.isfinite(stamp) or
                        not previous_time<stamp<=now):
                    raise ValueError('Previous observed point differs from the reviewed plan')
                previous_time=stamp
        for field in ('sampled_at','tcp_sampled_at'):
            stamp=current.get(field)
            if type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<=now-stamp<=.15:
                raise ValueError('Current robot observation is stale')
        joints=_vector(current.get('joints'),'current joints')
        pose=_vector(current.get('tcp_pose'),'current TCP pose')
        tcp=_vector(current.get('tcp_offset'),'current TCP offset')
        speed=_vector(current.get('tcp_speed'),'current TCP speed')
        joint_speed=_vector(current.get('joint_speed'),'current joint speed')
        if (max(abs(a-b) for a,b in zip(joints,start))>.003 or
                math.dist(pose[:3],expected_pose[:3])>.0005 or
                rotation_vector_distance(pose[3:],expected_pose[3:])>.005 or
                math.dist(tcp,self.tcp)>.0001 or
                math.hypot(*speed[:3])>=.0005 or math.hypot(*speed[3:])>=.002 or
                max(abs(v) for v in joint_speed)>=.003 or
                max(abs(a-b) for a,b in zip(joints,self.points[index]))<=self.dynamics['joint_tolerance']):
            raise ValueError('Current robot state differs from the reviewed stationary step start')
        profile=self._profile(index,expected_pose)
        StepProgram(profile,1)
        return profile


def shared_plan_template(draft):
    """创建一个单个未经过批准的计划模板；永远不提供限制或批准。"""
    if not isinstance(draft,dict) or draft.get('status')!='DRAFT' or not isinstance(draft.get('intro'),dict):
        raise ValueError('Explicit candidate draft required')
    robot_model=_candidate_model(draft)
    intro=draft['intro'];points=intro.get('waypoints')
    if not isinstance(points,list) or len(points)!=14:raise ValueError('Exactly 14 candidates required')
    template={'status':'DRAFT','robot_ip':draft.get('robot_ip'),
            'reference_joints':list(_vector(intro.get('reference_joints'),'candidate start joints')),
            'reference_pose':list(_vector(intro.get('reference_pose'),'candidate start pose')),
            'tcp':list(_vector(intro.get('tcp'),'candidate TCP')),
            'waypoints':[list(_vector(point,'candidate joints')) for point in points],
            'dynamics':{key:None for key in SharedPreviewPlan._dynamics},
            'stop':{key:None for key in ('resident_timeout','stopj_acceleration','watch_interval','watch_timeout')},
            'review':{key:False for key in SharedPreviewPlan._reviews},
            'operator_approval':None,'reviewed':{}}
    if 'robot_model' in draft:
        template.update(robot_model=robot_model,reviewed_robot_model=None)
    return template


def prepare_shared_step(host,plan,index,observations, *, dependencies=None,timeout=2.):
    """从全新的仅接收原始数据和 RTDE 数据中创建一个经过审核的步骤。"""
    if not isinstance(plan,SharedPreviewPlan) or host!=plan.host:
        raise ValueError('Robot differs from the reviewed plan')
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('Positive snapshot timeout required')
    d=_dependencies() if dependencies is None else dependencies
    states=reader=None
    try:
        states=d['state_factory'](host)
        reader=d['rtde_factory'](host,spatial=True,receive_only=True)
        deadline=d['clock']()+timeout
        while True:
            primary,ps=states.latest;state_failure=states.failure
            rtde,rs=reader.latest;reader_failure=reader.failure;now=d['clock']()
            if state_failure or reader_failure:raise RuntimeError('Read-only robot telemetry failed')
            if (isinstance(primary,dict) and isinstance(rtde,dict) and
                    type(ps) in (int,float) and math.isfinite(ps) and 0<=now-ps<=.15 and
                    type(rs) in (int,float) and math.isfinite(rs) and 0<=now-rs<=.15):
                _state_snapshot(SimpleNamespace(latest=(primary,ps),failure=None),plan,now)
                current={'sampled_at':rs,'tcp_sampled_at':ps,'joints':rtde.get('joints'),
                         'tcp_pose':rtde.get('tcp_pose'),'joint_speed':rtde.get('joint_speed'),
                         'tcp_speed':rtde.get('tcp_speed'),'tcp_offset':primary.get('tcp_offset')}
                return plan.step_profile(index,observations,current,now=now)
            if now>=deadline:raise RuntimeError('Read-only robot telemetry unavailable or stale')
            d['sleep'](.02)
    finally:
        for source in (reader,states):
            if source is not None:
                try:source.close()
                except Exception:pass


class StepSequence:
    """手动下一步，只有一步；失败和停止锁定，无需重试。"""
    def __init__(self,draft,host,read_guest, *, opener=open_step):
        if not isinstance(draft,dict) or draft.get('status')!='DRAFT' or not isinstance(draft.get('intro'),dict):
            raise ValueError('Explicit unreviewed candidate draft required')
        self.robot_model=_candidate_model(draft)
        self._explicit_model='robot_model' in draft
        points=draft['intro'].get('waypoints')
        if not isinstance(points,list) or len(points)!=14:raise ValueError('Exactly 14 candidate points required')
        self.points=tuple(_vector(point,'candidate joints') for point in points)
        self.draft=copy.deepcopy(draft)
        self.initial=_vector(draft['intro'].get('reference_joints'),'candidate reference joints')
        self.tcp=_vector(draft['intro'].get('tcp'),'candidate TCP')
        try:self.host=str(ipaddress.ip_address(host))
        except ValueError:raise ValueError('Explicit robot IP required') from None
        if draft.get('robot_ip')!=self.host or not callable(read_guest) or not callable(opener):
            raise ValueError('Draft robot or guest control mismatch')
        self.read_guest,self.opener=read_guest,opener
        self.index=0;self.state='WAITING';self.active=None;self.reason=None
        self.observations=[]
        self._lock=threading.Lock();self._opening=False;self._cleanup_lock=threading.Lock();self._stopped_session=None
        self._cleaned=set()
        self._cancel=threading.Event()

    def snapshot(self):
        with self._lock:
            return {'state':self.state,'opening':self._opening,'index':self.index,
                    'total':len(self.points),'target_joints':list(self.points[self.index]) if self.index<len(self.points) else None,
                    'reason':self.reason}

    def _close_once(self,session):
        with self._cleanup_lock:
            if session in self._cleaned:return
            session.close();self._cleaned.add(session)

    def _read_guest(self):
        # 在外部读取前后都会检查取消情况。
        # 已经在在途的启动程序可能会上传保留的程序，但监控程序无法在该锁定后武装闸门或发送启用信号。
        if self._cancel.is_set():return {'active':False,'camera_fresh':False,'camera_connected':False}
        snapshot=self.read_guest()
        if self._cancel.is_set():
            if not isinstance(snapshot,dict):return {'active':False,'camera_fresh':False,'camera_connected':False}
            return {**snapshot,'active':False,'camera_fresh':False,'camera_connected':False}
        return snapshot

    def _check_next_locked(self,checked):
        if self.state!='WAITING' or self._opening or self.index>=len(self.points):
            raise RuntimeError('Preview is not waiting for Next')
        if checked.robot_model!=self.robot_model:
            raise ValueError('Reviewed segment robot model differs from the selected candidate')
        expected_start=self.initial if self.index==0 else self.points[self.index-1]
        if (checked.robot_ip!=self.host or checked.reference_joints!=expected_start or
                checked.final_joints!=self.points[self.index] or checked.tcp!=self.tcp):
            raise ValueError('Reviewed segment differs from the selected candidate')
        if self.index>0:
            if (len(self.observations)!=self.index or self.observations[-1].get('index')!=self.index or
                    checked.reference_pose!=tuple(self.observations[-1].get('tcp_pose',()))):
                raise ValueError('Reviewed start pose differs from the previous observed point')

    def validate_next(self,profile):
        """在不打开流程、服务或机器人的情况下检查步骤。"""
        checked=StepProgram(copy.deepcopy(profile),1)
        with self._lock:self._check_next_locked(checked)

    def next(self,profile):
        """明确的“下一个”将开始一个经过审查的候选人，永远不会开始下一个。"""
        checked=StepProgram(copy.deepcopy(profile),1)
        with self._lock:
            self._check_next_locked(checked)
            self._opening=True
        try:
            session=self.opener(self.host,copy.deepcopy(profile),self._read_guest)
        except Exception:
            with self._lock:
                self._opening=False
                if self.state!='STOPPED':self.state='FAULT';self.reason='Step startup failed'
            raise
        with self._lock:
            self._opening=False
            if self.state!='WAITING':cancelled=True
            else:self.active=session;self.state='MOVING';cancelled=False
        if cancelled:
            session.stop();self._close_once(session);raise RuntimeError('Preview stopped during startup')

    def tick(self):
        with self._lock:
            if self.state!='MOVING' or self.active is None:raise RuntimeError('No step in progress')
            active=self.active
        try:
            result=active.tick()
            if result!='COMPLETE':return result
            source=getattr(active,'reader',None)
            if source is None or source.failure:raise RuntimeError('Independent preview telemetry unavailable')
            data,stamp=source.latest;now=time.monotonic()
            if (not isinstance(data,dict) or type(stamp) not in (int,float) or not math.isfinite(stamp)
                    or not 0<=now-stamp<=.15):raise RuntimeError('Preview telemetry stale at point')
            joints=_vector(data.get('joints'),'observed joints')
            speed=_vector(data.get('tcp_speed'),'observed TCP speed')
            joint_speed=_vector(data.get('joint_speed'),'observed joint speed')
            pose=_vector(data.get('tcp_pose'),'observed TCP pose')
            program=getattr(getattr(active,'progress',None),'program',None)
            tolerance=getattr(program,'joint_tolerance',None)
            if (type(tolerance) not in (int,float) or not math.isfinite(tolerance) or tolerance<=0 or
                    max(abs(a-b) for a,b in zip(joints,self.points[self.index]))>tolerance or
                    math.hypot(*speed[:3])>=.0005 or math.hypot(*speed[3:])>=.002 or
                    max(abs(v) for v in joint_speed)>=.003):
                raise RuntimeError('Preview point is not observed stationary')
            observation={'index':self.index+1,'joints':list(joints),'tcp_pose':list(pose),
                         'sampled_at':stamp,'joint_tolerance':tolerance}
            self._close_once(active)
        except Exception:
            self.stop('Step fault');raise
        with self._lock:
            if self.state!='MOVING' or self.active is not active:
                raise RuntimeError('Preview stopped during step completion')
            self.observations.append(observation);self.active=None;self.index+=1
            self.state='DONE' if self.index==len(self.points) else 'WAITING'
            return self.state

    def stop(self,reason='Operator stopped'):
        self._cancel.set()
        with self._lock:
            active=self.active;self.active=None;self.state='STOPPED';self.reason=reason
            if active is not None:self._stopped_session=active
            gate=getattr(active,'gate',None) if active is not None else None
            if gate is not None:gate.stop(reason)
        if active is not None:
            active.stop()
            # 让独立读者随时准备进行单独的测量静止检查；呼叫者在检查后会明确地关闭。

    def stopped_feedback(self):
        """返回调用者的独立最后一次电测数据；不是停止证明。"""
        with self._lock:session=self._stopped_session
        if session is None:return None
        source=getattr(session,'reader',None)
        if source is None or source.failure:return None
        return source.latest

    def close(self):
        """在调用者观察后关闭停止的独立读屏。"""
        with self._lock:state=self.state
        if state not in ('STOPPED','DONE'):self.stop('Preview closed')
        with self._lock:session=self._stopped_session
        if session is not None:self._close_once(session)

    def acceptance(self,decision):
        """记录操作员明确的预览后决定，而不是安全审查。"""
        with self._lock:
            if self.state!='DONE' or self.index!=len(self.points) or decision!='accept':
                raise ValueError('All 14 points and an explicit accept decision are required')
            record={'status':'OPERATOR_ACCEPTED','robot_ip':self.host,
                    'reference_joints':list(self.initial),'tcp':list(self.tcp),
                    'waypoints':[list(p) for p in self.points],
                    'observed_indices':list(range(1,15)),'operator_decision':'accept',
                    'observations':copy.deepcopy(self.observations)}
            result={**record,'reviewed':copy.deepcopy(record)}
            if self._explicit_model:
                result.update(robot_model=self.robot_model,reviewed_robot_model=self.robot_model)
            return result


def draft_step_template(draft,index, *, observations=None):
    """生成一个未经批准的模板；它不能像生成的那样执行。"""
    if (not isinstance(draft,dict) or draft.get('status')!='DRAFT' or
            not isinstance(draft.get('intro'),dict) or type(index) is not int or not 0<=index<14):
        raise ValueError('Explicit draft and candidate index required')
    robot_model=_candidate_model(draft)
    intro=draft['intro'];points=intro.get('waypoints')
    if not isinstance(points,list) or len(points)!=14:raise ValueError('Exactly 14 candidates required')
    target=list(_vector(points[index],'candidate joints'))
    start=list(_vector(intro.get('reference_joints') if index==0 else points[index-1],'candidate start joints'))
    reference_pose=copy.deepcopy(intro.get('reference_pose')) if index==0 else None
    if index>0 and observations is not None:
        if not isinstance(observations,list) or len(observations)!=index:
            raise ValueError('Prior observed points do not match requested step')
        prior=observations[-1]
        if not isinstance(prior,dict) or prior.get('index')!=index:
            raise ValueError('Prior observation identity mismatch')
        actual=_vector(prior.get('joints'),'previous observed joints')
        if max(abs(a-b) for a,b in zip(actual,start))>.003:
            raise ValueError('Prior observed joints differ from previous candidate')
        reference_pose=list(_vector(prior.get('tcp_pose'),'previous observed TCP pose'))
    template={'status':'DRAFT','robot_ip':draft.get('robot_ip'),'reference_joints':start,
            'reference_pose':reference_pose,
            'tcp':list(_vector(intro.get('tcp'),'candidate TCP')),'target_joints':target,
            'joint_tolerance':None,
            'joint_speed':None,'joint_acceleration':None,'max_tcp_angular_speed':None,
            'max_tcp_linear_speed':None,'timeout':None,'reviewed':{},
            'review':{k:False for k in ('robot','mount','tcp','payload','path','dynamics','stop','gesture')}}
    if 'robot_model' in draft:
        template.update(robot_model=robot_model,reviewed_robot_model=None)
    return template
