"""一个有限控制器时钟手势，保留了传统目标协议。"""
import math
import re
import pytest

from leica_tracker.physical_resident_engine import Installation, ResidentEngine
from test_physical_resident_engine import reviewed, Gate, feedback


def installation(model='ur10e'):
    data=reviewed()
    data.update(robot_model=model,reviewed_robot_model=model,
                lower=[-.3,-.9,-.1],upper=[.5,-.1,.7],speed=.26,acceleration=1.,
                base_limit=1.,base_rate=.7,max_tcp_linear_speed=None,max_tcp_angular_speed=None,
                gesture={'duration_s':3.3,'repetitions':2,'joint_speed_limit':.7})
    return Installation(data)


@pytest.mark.parametrize('model',['ur5e','ur10e'])
def test_controller_program_contains_finite_timed_gesture_and_retained_stop(model):
    cfg=installation(model);program=cfg.program(123)
    assert 'flag == 2' in program
    assert 'global gesture_age = 0.0' in program
    assert 'gesture_age + get_steptime()*gesture_scale' in program
    assert 'cos(4.0000000000*3.141592653589793' in program
    assert '0.7000000000/larger(0.0000001,norm(rate))' in program
    assert 'state = 3' in program
    assert program.index('neutral = p[')<program.index('gesture_endpoint = neutral')
    assert 'packet_age > 0.25' in program
    assert 'is_within_safety_limits(probe)' in program
    assert 'set_tool' not in program and 'set_payload' not in program


def test_legacy_installation_does_not_silently_enable_finite_gestures():
    cfg=Installation(reviewed());engine=ResidentEngine(cfg,Gate(),123)
    assert cfg.gesture is None
    assert 'gesture_age' not in cfg.program(123)
    engine.state='READY'
    with pytest.raises(ValueError,match='gesture'):
        engine.gesture([.105,-.5,.3],now=1.)


@pytest.mark.parametrize('model',['ur5e','ur10e'])
def test_finite_gesture_sends_endpoint_then_waits_native_done_and_holds(model):
    cfg=installation(model);gate=Gate();engine=ResidentEngine(cfg,gate,123)
    engine.tick(feedback(cfg,123),sampled_at=1.,now=1.)
    engine.tick(feedback(cfg,123,state=0,seq=1),sampled_at=1.03,now=1.03)
    endpoint=[cfg.neutral[0]+.12,*cfg.neutral[1:3]]
    engine.gesture(endpoint,now=1.04)
    engine.tick(feedback(cfg,123,state=0,seq=2),sampled_at=1.06,now=1.06)
    assert gate.calls[-1][1]==2 and gate.calls[-1][3]==endpoint
    # 第一次摆动后途经参考位不能触发整个动作完成。
    engine.tick(feedback(cfg,123,state=1,seq=3),sampled_at=1.09,now=1.09)
    assert engine.state=='MOVING'
    engine.tick(feedback(cfg,123,state=3,seq=4),sampled_at=1.12,now=1.12)
    assert engine.state=='HOLDING' and gate.calls[-1][1]==0
    engine.tick(feedback(cfg,123,state=0,seq=5),sampled_at=1.15,now=1.15)
    assert engine.state=='READY' and engine.completed==1


def test_canonical_dynamics_uses_period_and_displacement_not_joint_units():
    from leica_tracker.native_gesture import canonical_dynamics
    values=canonical_dynamics(.12)
    assert values['gesture']==dict(duration_s=3.3,repetitions=2,joint_speed_limit=.7)
    assert values['speed']>2*math.pi*.12/3.3
    assert values['acceleration']>8*math.pi**2*.12/3.3**2
    assert values['speed']<.7


def test_finite_braking_envelope_preserves_planned_reversal_cadence():
    """根据实际全振幅时间定律施加发射的约束。
    即使不饱和笛卡尔或关节动力学，传统的350毫秒开端行程储备也会减缓已知的反转。"""
    from leica_tracker.native_gesture import canonical_dynamics, GestureCadence
    cfg=installation();program=cfg.program(123)
    match=re.search(r'if gesture_mode:\n          allowed = sqrt\(([0-9.]+)\*larger\(0.0,distance-boundary_margin\[i\]\)\)\n        else:\n          allowed = sqrt',program)
    assert match, 'Finite trajectories need stopping distance, rather than open-ended travel reserve'
    twice_acceleration=float(match[1])
    assert twice_acceleration==pytest.approx(2*cfg.acceleration)
    cadence=GestureCadence()
    for displacement in (.12,.085,.06):
        dynamics=canonical_dynamics(.12)
        for i in range(3301):
            fraction,derivative=cadence.sample(i*.001)
            velocity=displacement*derivative
            position=displacement*fraction
            # 六个实际方向方案均在端点保留 6 毫米余量。
            distance=(displacement+.006-position if velocity>=0 else position+.006)
            allowed=math.sqrt(twice_acceleration*max(0,distance-.003))
            assert allowed+1e-9>=abs(velocity)
            # 该设计仍满足原有 100 毫秒逆解和边界探测条件。
            assert -.006<=position+.1*velocity<=displacement+.006
        fraction=.95;velocity=displacement*2*math.pi/3.3*2*math.sqrt(fraction*(1-fraction))
        distance=displacement*(1-fraction)+.006
        legacy=math.sqrt((dynamics['acceleration']*.35)**2+2*dynamics['acceleration']*(distance-.003))-dynamics['acceleration']*.35
        assert legacy<velocity


def test_finite_feedforward_survives_small_tracking_error_until_final_settle():
    program=installation().program(123)
    assert ('if norm(target[i]-actual[i]) < 0.0002 and (not gesture_mode or not gesture_active):\n'
            '          v[i] = 0') in program
    # 原有目标跟随模式沿用既有死区和制动边界。
    legacy=Installation(reviewed()).program(123)
    assert 'if norm(target[i]-actual[i]) < 0.0002:\n' in legacy
    assert 'distance-boundary_margin[i]))-' in legacy


def test_finite_base_governor_uses_its_existing_ik_probe_horizon():
    from leica_tracker.native_gesture import GestureCadence
    program=installation().program(123)
    assert ('base_lookahead = 0.35\n      if gesture_mode:\n'
            '        base_lookahead = 0.1\n      end') in program
    assert 'base_rate*base_lookahead' in program and '-base_rate*base_lookahead' in program
    cadence=GestureCadence()
    # UR5e 的 native01 记录中，基座偏移达 0.28085 弧度，仍在安装规定的 0.3 弧度范围内。
    # 有限轨迹按标称方案换向，无须额外预留 350 毫秒连续行程；100 毫秒逆解探测也未越界。
    legacy_clipped=False
    for i in range(3301):
        fraction,derivative=cadence.sample(i*.001)
        displacement=.281*fraction;rate=.281*derivative
        assert abs(rate)<.7
        assert -.3<=displacement+.1*rate<=.3
        legacy_clipped |= abs(displacement+.35*rate)>.3
    assert legacy_clipped
    # 仅更改有限动作所用的 flag2；旧安装仍保留原预留量。
    legacy=Installation(reviewed()).program(123)
    assert 'base_rate*0.35' in legacy


def test_finite_tracking_closes_terminal_error_without_a_long_exponential_tail():
    """一个受限的20毫秒速度模型捕捉到了观察到的终端增益缺陷。这是一个控制法回归，而不是官方/原生时间接受。
    物理控制器仍然决定实际已过时间和速度。"""
    from leica_tracker.native_gesture import GestureCadence
    program=installation().program(123)
    match=re.search(r'tracking_gain = 4.0\n      if gesture_mode:\n        tracking_gain = ([0-9.]+)',program)
    assert match, 'Finite tracking needs its own bandwidth, leaving legacy gain unchanged'
    gain=float(match[1]);assert 0<gain*.02<1
    cadence=GestureCadence();position=velocity=0.;dt=.02;completed=None;peak=0.
    for step in range(250):
        now=step*dt;fraction,derivative=cadence.sample(now)
        target=.12*fraction;command=.12*derivative+gain*(target-position)
        if now>=cadence.duration_s and abs(target-position)<.0002:command=0.
        next_velocity=velocity+max(-dt,min(dt,command-velocity))
        position+=(velocity+next_velocity)*dt*.5;velocity=next_velocity
        peak=max(peak,abs(position))
        if now>=cadence.duration_s and abs(position)<.0005 and abs(velocity)<.0005:
            completed=now;break
    assert completed is not None and completed<=3.7
    assert peak<.123 # 小于此前增益为 4 时的跟踪过冲。
    assert abs(position)<.0005 and abs(velocity)<.0005


def test_only_spatial_protocol_accepts_explicit_finite_gesture_flag():
    from leica_tracker.ur_rtde_transport import validate_spatial_target, validate_target
    validate_spatial_target((1,2,123,[.1,-.5,.3]))
    with pytest.raises(ValueError):validate_target((1,2,123,.1))
    with pytest.raises(ValueError):validate_spatial_target((1,4,123,[.1,-.5,.3]))


@pytest.mark.parametrize('changes',[{'duration_s':0},{'repetitions':0},{'joint_speed_limit':float('nan')}])
def test_invalid_gesture_configuration_is_rejected(changes):
    data=reviewed();data['gesture']={'duration_s':3.3,'repetitions':2,'joint_speed_limit':.7,**changes}
    with pytest.raises(ValueError,match='gesture'):Installation(data)


@pytest.mark.parametrize('model',['ur5e','ur10e'])
@pytest.mark.parametrize('cue,delta',[
    ('MOVE_LEFT',[-.12,0,0]),('MOVE_RIGHT',[.12,0,0]),
    ('MOVE_UP',[0,0,.085]),('MOVE_DOWN',[0,0,-.085]),
    ('MOVE_CLOSER',[0,-.06,0]),('MOVE_FARTHER',[0,.06,0])])
def test_shared_supervisor_admits_one_finite_gesture_for_each_model_and_direction(model,cue,delta):
    from leica_tracker.physical_interaction_supervisor import PhysicalInteractionSupervisor
    from test_physical_interaction_supervisor import Ref, Immediate
    cfg=installation(model);gate=Gate();gate.can_send=True
    engine=ResidentEngine(cfg,gate,123)
    engine.tick(feedback(cfg,123),sampled_at=1.,now=1.)
    engine.tick(feedback(cfg,123,state=0,seq=1),sampled_at=1.03,now=1.03)
    vectors=dict(MOVE_LEFT=[-.12,0,0],MOVE_RIGHT=[.12,0,0],MOVE_UP=[0,0,.085],
                 MOVE_DOWN=[0,0,-.085],MOVE_CLOSER=[0,-.06,0],MOVE_FARTHER=[0,.06,0])
    paths={name:[[a+b for a,b in zip(cfg.neutral[:3],offset)]] for name,offset in vectors.items()}
    ref=Ref();ref.ack=True
    sup=PhysicalInteractionSupervisor(engine,ref,paths,reviewed_paths=paths,
        hold_timeout=1.,action_timeout=10.,gate_timeout=.15,max_joint_speed=.7,worker_factory=Immediate)
    sup.bind_control_identity(dict(instance='guest-a',epoch=1,started_ns=1))
    sup.set_move_readiness(lambda *_:True)
    event=dict(event='person_guidance',control_instance='guest-a',control_epoch=1,
        source_monotonic_ns=1_040_000_000,coordinate_frame='normalized_camera_image',
        robot_actuated=False,tracking=True,cue=cue)
    assert sup.accept_event(event,now=1.04)
    for index in range(3):
        now=1.06+index*.03
        sample=feedback(cfg,123,state=0,seq=gate.calls[-1][0])
        sup.tick(sample,sampled_at=now,joint_feedback={'joint_speed':[0.]*6,'tcp_speed':[0.]*6},
                 joint_sampled_at=now,now=now)
    assert sup.state=='MOVING'
    assert gate.calls[-1][1]==2
    assert gate.calls[-1][3]==pytest.approx(paths[cue][-1])
    assert sup._path==[] # 原生程序，而不是Python，拥有两个外向/返回循环。
