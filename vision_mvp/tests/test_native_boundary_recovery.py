"""用有界的一维线性关节模型离线验证生成的原生控制律。停滞位置来自 UR5e 的 source22 故障记录；
相位网格用于构造允许的停滞相位，并非遥测。测试验证能否退出死锁并保留约束，不作为实机验收。"""
import math
import pytest

from leica_tracker.native_gesture import GestureCadence
from leica_tracker.operator_design import INSTALLATION_SELECTION_FIELDS,operator_selection_record
from leica_tracker.physical_resident_engine import Installation
from test_physical_resident_engine import reviewed
from test_native_recovery_pause import run_branch


def configuration():
    data=reviewed();reference=-.030914328832445354
    data.update(robot_model='ur5e',reviewed_robot_model='ur5e',reference=[reference,0.,0.,0.,0.,0.],
        lower=[reference-.126,-.1,-.1],upper=[reference+.126,.1,.1],
        speed=.2627513855729645,acceleration=1.000554936529169,
        base_limit=None,base_rate=.7,max_tcp_linear_speed=None,max_tcp_angular_speed=None,
        gesture=GestureCadence().as_dict())
    data['operator_selection']=operator_selection_record(data,kind='installation',fields=INSTALLATION_SELECTION_FIELDS)
    return Installation(data)


def emitted_blocks(cfg):
    script=cfg.program(123)
    start=script.index('      feedforward = [0,0,0]\n')
    end=script.index('      probe = p[',start)
    cartesian=script[start:end]
    start=script.index('      scale = 1.0\n      i = 0\n      while i < 6:',end)
    end=script.index('      probe = p[',start)
    return cartesian,script[start:end]


def stage(cfg,blocks,position,age,sign):
    actual=[cfg.neutral[0]+sign*position,0.,0.,0.,0.,0.]
    endpoint=[cfg.neutral[0]+sign*.12,0.,0.,0.,0.,0.]
    values=dict(actual=actual,neutral=cfg.neutral,gesture_endpoint=endpoint,gesture_age=age,
        gesture_scale=1.,gesture_mode=True,gesture_active=age<3.3,lower=cfg.lower,upper=cfg.upper,
        boundary_margin=[.003]*3,smaller=min,larger=max,sqrt=math.sqrt,sin=math.sin,cos=math.cos)
    values=run_branch(blocks[0],values)
    # 通过显式的测试对象和有限正值的雅可比系数，检查生成的六关节及基座速率限制；不代入实机逆解。
    sensitivity=(3.,.8,-.5,.3,.2,2.7)
    values['q']=cfg.joints
    values['q_next']=[q+c*values['v'][0]*.1 for q,c in zip(cfg.joints,sensitivity)]
    return run_branch(blocks[1],values)


@pytest.mark.parametrize('sign',[-1,1])
def test_actual_plateau_has_inward_corrective_motion_with_phase_frozen(sign):
    cfg=configuration();blocks=emitted_blocks(cfg)
    age=3.3*math.acos(1-2*.93)/(4*math.pi)
    # 使用 UR5e 实测的停滞位置：端点过冲 3.3120715 毫米，仍在边界内。
    position=.12331207152261755
    result=stage(cfg,blocks,position,age,sign)
    assert sign*result['v'][0]<0, 'A zero governor must retain a safe inward corrective direction'
    assert result['gesture_scale']==0 and result['gesture_age']==age
    assert abs(result['v'][0])<=cfg.speed and abs(3*result['v'][0])<=.7+1e-10


@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('fraction',[.1,.3,.5,.7,.9,.99])
def test_unconstrained_nominal_tracking_keeps_original_phase_rate_and_feedforward(sign,fraction):
    cfg=configuration();blocks=emitted_blocks(cfg)
    age=3.3*math.acos(1-2*fraction)/(4*math.pi)
    expected=sign*.12*2*math.pi/3.3*math.sqrt(1-(1-2*fraction)**2)
    result=stage(cfg,blocks,.12*fraction,age,sign)
    assert result['gesture_correcting'] is False
    assert result['gesture_scale']==pytest.approx(1.)
    assert result['v'][0]==pytest.approx(expected,abs=1e-10)


@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('fraction',[.88,.90,.93,.96,.98,.99])
@pytest.mark.parametrize('position',[.12*1.026,.12331207152261755,.12*1.03])
def test_admissible_stalled_phase_grid_finishes_two_sweeps_inside_unchanged_caps(sign,fraction,position):
    cfg=configuration();blocks=emitted_blocks(cfg);dt=.002
    age=3.3*math.acos(1-2*fraction)/(4*math.pi)
    derivative=2*math.pi/3.3*math.sqrt(1-(1-2*fraction)**2)
    assert .12*derivative+8*(.12*fraction-position)>0  # 向外运动的标称轨迹规律。
    start_age=age;velocity=command=0.;scale=1.;active=True;excursions=0;high=False
    max_rate=0.;corrected=False
    for n in range(4000):
        if n%10==0:
            result=stage(cfg,blocks,position,age,sign)
            command=sign*result['v'][0];scale=result['gesture_scale'];active=result['gesture_active']
            if result.get('gesture_correcting'):
                corrected=True
                assert scale==0 and command<0
        old=velocity
        velocity+=max(-cfg.acceleration*dt,min(cfg.acceleration*dt,command-velocity))
        position+=(old+velocity)*dt/2
        if active:age+=dt*scale
        max_rate=max(max_rate,abs(3*velocity))
        assert -.126<=position<=.126
        assert abs(velocity)<=cfg.speed+1e-10 and max_rate<=.7+1e-9
        if position/.12>=.9 and not high:high=True
        if high and position/.12<=.35:excursions+=1;high=False
        if age>=3.3 and abs(position)<.0005 and abs(velocity)<.0005:break
    assert corrected and excursions==2
    assert n<3999, 'Finite phase/return deadlocked despite an admissible correction'
    assert start_age+(n+1)*dt<4.1
