"""在离线情况下执行发射的证人分支；没有机器人/运输接口。"""
import math
import re

import numpy as np
import pytest

from leica_tracker.physical_resident_engine import Installation
from test_physical_resident_engine import reviewed
from test_native_gesture import installation


class ReachFault(Exception):pass


def execute_branch(program,*,position_error=0.,rotation_error=0.,in_box=True,solver=True,tcp=(0,0,0,0,0,0)):
    """仅使用测试数学翻译实际发射的非运动分支。独立的同质变换练习两个残差范数。
    求解器回调记录发射的程序是否只重用已证明的当前q。"""
    start=program.index('      probe_in_box = in_box(probe)\n')
    end=program.index('      scale = 1.0\n',start)
    source=[];depth=0
    for line in program[start:end].splitlines():
        line=line.strip()
        if line=='end':depth-=1;continue
        if line.startswith(('elif ','else:')):depth-=1
        line=re.sub(r'\bTrue\b','True',line);line=re.sub(r'\bFalse\b','False',line)
        source.append('    '*depth+('raise ReachFault()' if line=='break' else line))
        if line.startswith(('if ','elif ','else:')):depth+=1
    calls=[];q=[.1,-1.,-2.,-.8,1.6,-.03];offset=list(tcp);probe=np.eye(4)
    known=np.eye(4);known[0,3]=position_error
    angle=rotation_error;known[:3,:3]=[[math.cos(angle),-math.sin(angle),0],
        [math.sin(angle),math.cos(angle),0],[0,0,1]]
    def forward(actual_q,active_tcp):
        calls.append(('fk',actual_q,active_tcp));assert actual_q is q and active_tcp is offset
        result=known.copy();result[2,3]+=active_tcp[2]
        return result
    def pose_trans(a,b):
        value=a@b;r=value[:3,:3]
        skew=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])*.5
        sine=float(np.linalg.norm(skew));angle=math.atan2(sine,(np.trace(r)-1)*.5)
        rotation=skew*(angle/sine) if sine else skew
        return [*value[:3,3],*rotation]
    def has_solution(p,near,**kwargs):
        calls.append(('has',near,kwargs));assert p is probe and near is q and kwargs=={'tcp':offset}
        return solver
    def inverse(p,near,**kwargs):
        calls.append(('ik',near,kwargs));assert p is probe and near is q and kwargs=={'tcp':offset}
        return [v+.01 for v in near]
    scope=dict(probe=probe,q=q,offset=offset,in_box=lambda p:in_box,
        get_forward_kin=forward,pose_inv=np.linalg.inv,pose_trans=pose_trans,norm=np.linalg.norm,
        get_inverse_kin_has_solution=has_solution,get_inverse_kin=inverse,
        textmsg=lambda *a:calls.append(('text',)),stopl=lambda *a:calls.append(('stop',)),
        write_output_integer_register=lambda *a:calls.append(('register',a)),ReachFault=ReachFault)
    fault=False
    try:exec('\n'.join(source),scope)
    except ReachFault:fault=True
    return scope,calls,fault


@pytest.mark.parametrize('model',['ur5e','ur10e'])
@pytest.mark.parametrize('finite',[False,True])
def test_proved_current_q_avoids_inverse_and_keeps_all_downstream_guards(model,finite):
    data=reviewed();data.update(robot_model=model,reviewed_robot_model=model)
    program=installation(model).program(123) if finite else Installation(data).program(123)
    scope,calls,fault=execute_branch(program,position_error=.5e-10,rotation_error=.5e-10)
    assert not fault and scope['q_next'] is scope['q'] and [row[0] for row in calls]==['fk']
    assert 'packet_age > 0.25' in program and 'get_actual_joint_positions()' in program
    assert 'offset = get_tcp_offset()' in program and 'get_forward_kin(q, offset)' in program
    assert 'rate = (q_next[i]-q[i])/0.1' in program
    assert 'base_rate = (q_next[0]-q[0])/0.1' in program
    assert 'if not is_within_safety_limits(probe):' in program
    assert program.index('if not is_within_safety_limits(probe):')<program.index('speedl(')
    assert 'maxPositionError' not in program and 'maxOrientationError' not in program


@pytest.mark.parametrize('changes',[
    {'position_error':1.001e-10},{'rotation_error':1.001e-10},
    {'tcp':(0,0,.001,0,0,0)},{'position_error':float('nan')},
])
def test_any_unproved_position_orientation_or_active_tcp_uses_default_solver(changes):
    scope,calls,fault=execute_branch(installation().program(123),**changes)
    assert not fault and [row[0] for row in calls]==['fk','has','ik']
    assert scope['q_next']!=scope['q']


@pytest.mark.parametrize('changes',[{'in_box':False},{'position_error':1e-5,'solver':False}])
def test_out_of_box_or_unproved_unsolved_probe_stops_and_faults(changes):
    scope,calls,fault=execute_branch(installation().program(123),**changes)
    assert fault and ('stop',) in calls and ('register',(42,-6)) in calls and 'q_next' not in scope
    if not changes.get('in_box',True):assert all(row[0] not in ('fk','has','ik') for row in calls)
