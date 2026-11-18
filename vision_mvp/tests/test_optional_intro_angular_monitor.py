"""在没有任何物理通道的情况下测试了明确的可选引出监视。"""
import math
import socket

import pytest

from leica_tracker.physical_intro_program import IntroProgram, IntroProgress
from leica_tracker.physical_runtime import reviewed_profile
from leica_tracker.physical_step_preview import StepProgram
from test_continuous_preview import continuous_data, rebind
from test_physical_intro_program import Gate, feedback, profile, tick
from test_physical_step_preview import example


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Optional-monitor tests must never open a socket')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def optional_profile():
    candidate=profile()
    candidate['max_tcp_angular_speed']=None
    candidate['reviewed']['max_tcp_angular_speed']=None
    candidate['max_tcp_linear_speed']=.065
    candidate['reviewed']['max_tcp_linear_speed']=.065
    return candidate


def active_progress():
    candidate=optional_profile();gate=Gate()
    progress=IntroProgress(IntroProgram(candidate,7),gate)
    tick(progress,feedback(candidate,100,0),1.)
    tick(progress,feedback(candidate,100,1),1.1)
    tick(progress,feedback(candidate,101,2),1.2)
    return candidate,gate,progress


def optional_continuous_data(*, historical_optional=True):
    candidate=continuous_data()
    candidate['intro']['max_tcp_angular_speed']=None
    candidate['intro']['reviewed']['max_tcp_angular_speed']=None
    if historical_optional:
        executed=candidate['intro_preview']['execution_profile']['intro']
        executed['max_tcp_angular_speed']=None
        executed['reviewed']['max_tcp_angular_speed']=None
    rebind(candidate)
    return candidate


def test_explicit_reviewed_null_omits_only_robot_angular_monitor():
    program=IntroProgram(optional_profile(),7)
    assert program.max_tcp_angular_speed is None
    script=program.program()
    assert 'fault = 17' not in script
    assert 'fault = 16' in script and '> 0.0650000000' in script
    assert 'fault = 14' in script and '> 0.1060000000' in script
    assert 'packet_age > 0.25' in script and 'fault = 12' in script
    assert 'stopj(0.2)' in script
    assert 'if flag == 0 and not started' in script and 'elif flag == 1 and armed' in script


@pytest.mark.parametrize('where', ['profile', 'reviewed', 'both'])
def test_null_monitor_requires_explicit_key_in_profile_and_review(where):
    candidate=optional_profile()
    if where in ('profile','both'):del candidate['max_tcp_angular_speed']
    if where in ('reviewed','both'):del candidate['reviewed']['max_tcp_angular_speed']
    with pytest.raises(ValueError):IntroProgram(candidate,7)


@pytest.mark.parametrize('value', [0., -1., False, True, '', 'null', math.nan, math.inf, -math.inf])
def test_only_null_or_finite_positive_numeric_monitor_is_accepted(value):
    candidate=optional_profile()
    candidate['max_tcp_angular_speed']=value
    candidate['reviewed']['max_tcp_angular_speed']=value
    with pytest.raises(ValueError):IntroProgram(candidate,7)


def test_null_monitor_still_requires_matching_dynamics_review():
    candidate=optional_profile();candidate['reviewed']['max_tcp_angular_speed']=.15
    with pytest.raises(ValueError,match='review'):IntroProgram(candidate,7)
    candidate=optional_profile();candidate['review']['dynamics']=False
    with pytest.raises(ValueError,match='review'):IntroProgram(candidate,7)


def test_optional_monitor_allows_high_finite_tcp_rotation_after_start_ack():
    candidate,gate,progress=active_progress()
    writer,reader=feedback(candidate,101,3)
    reader['tcp_speed'][3]=.6
    assert tick(progress,(writer,reader),1.3)=='ACTIVE'
    assert gate.reason is None and gate.sent[-1][:3]==(4,1,7)


@pytest.mark.parametrize('field,index,value', [
    ('tcp_speed',0,.066), ('joint_speed',0,.107), ('tcp_speed',3,math.nan),
])
def test_optional_monitor_keeps_linear_joint_and_finite_telemetry_guards(field,index,value):
    candidate,gate,progress=active_progress()
    writer,reader=feedback(candidate,101,3);reader[field][index]=value
    sent=len(gate.sent)
    with pytest.raises(RuntimeError):tick(progress,(writer,reader),1.3)
    assert progress.state=='FAULT' and gate.reason and len(gate.sent)==sent


@pytest.mark.parametrize('state,reason', [(-12,'heartbeat expired'), (-16,'TCP linear speed')])
def test_optional_monitor_keeps_native_heartbeat_and_linear_faults(state,reason):
    candidate,gate,progress=active_progress()
    with pytest.raises(RuntimeError,match=reason):tick(progress,feedback(candidate,state,3),1.3)
    assert progress.state=='FAULT' and reason in gate.reason


def test_optional_monitor_keeps_freshness_stop_and_no_automatic_resume():
    candidate,gate,progress=active_progress();sent=len(gate.sent)
    with pytest.raises(RuntimeError,match='repeated'):
        tick(progress,feedback(candidate,101,3),1.2)
    assert progress.state=='FAULT' and len(gate.sent)==sent
    candidate,gate,progress=active_progress();progress.stop();sent=len(gate.sent)
    with pytest.raises(RuntimeError,match='no longer active'):
        tick(progress,feedback(candidate,101,3),1.3)
    assert progress.state=='STOPPED' and len(gate.sent)==sent


def test_optional_monitor_still_requires_angular_standstill_for_completion():
    candidate,gate,progress=active_progress()
    tick(progress,feedback(candidate,102,3),1.3)
    writer,reader=feedback(candidate,102,3);reader['tcp_speed'][3]=.003
    assert tick(progress,(writer,reader),1.4)=='SETTLING'
    assert progress.steady_since is None and gate.reason is None
    for i in range(9):
        state=tick(progress,feedback(candidate,102,3),1.5+i*.1)
        if state=='COMPLETE':break
    assert progress.state=='COMPLETE'


def test_continuous_preview_accepts_matching_explicit_null_execution_monitor():
    candidate=optional_continuous_data()
    candidate['intro_preview']['execution_record']['trace'][0]['tcp_speed'][3]=.6
    rebind(candidate)
    assert reviewed_profile(candidate).intro['max_tcp_angular_speed'] is None


def test_current_null_monitor_preserves_historical_numeric_execution_guard():
    candidate=optional_continuous_data(historical_optional=False)
    assert reviewed_profile(candidate).intro['max_tcp_angular_speed'] is None
    candidate['intro_preview']['execution_record']['trace'][0]['tcp_speed'][3]=.6
    rebind(candidate)
    with pytest.raises(ValueError,match='telemetry'):reviewed_profile(candidate)


def test_continuous_preview_null_execution_requires_its_own_matching_review():
    candidate=optional_continuous_data()
    executed=candidate['intro_preview']['execution_profile']['intro']
    executed['reviewed']['max_tcp_angular_speed']=.04
    rebind(candidate)
    with pytest.raises(ValueError,match='review'):reviewed_profile(candidate)


@pytest.mark.parametrize('where,index,value', [('trace',0,.011), ('completion',3,.003), ('final',3,.003)])
def test_null_continuous_preview_keeps_linear_and_completion_standstill_checks(where,index,value):
    candidate=optional_continuous_data();record=candidate['intro_preview']['execution_record']
    if where=='trace':record['trace'][0]['tcp_speed'][index]=value
    elif where=='completion':record['trace'][1]['tcp_speed'][index]=value
    else:record['last_independent_feedback']['tcp_speed'][index]=value
    rebind(candidate)
    with pytest.raises(ValueError):reviewed_profile(candidate)


def test_step_program_keeps_its_separate_positive_angular_monitor_requirement():
    _,step=example()
    step['max_tcp_angular_speed']=None;step['reviewed']['max_tcp_angular_speed']=None
    with pytest.raises(ValueError,match='positive max_tcp_angular_speed'):StepProgram(step,7)
