"""可选的引言TCP监视器仍然与停止和返回检查保持独立。"""
import math
import socket

import pytest

from leica_tracker.physical_intro_program import IntroProgram, IntroProgress
from leica_tracker.physical_runtime import reviewed_profile
from test_continuous_preview import continuous_data, rebind
from test_physical_intro_program import Gate, feedback, profile, tick


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Optional-monitor tests must never open a socket')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def optional_linear_profile(*, angular=.15):
    candidate=profile()
    candidate['max_tcp_linear_speed']=None
    candidate['reviewed']['max_tcp_linear_speed']=None
    candidate['max_tcp_angular_speed']=angular
    candidate['reviewed']['max_tcp_angular_speed']=angular
    return candidate


def active_progress(*, angular=.15):
    candidate=optional_linear_profile(angular=angular);gate=Gate()
    progress=IntroProgress(IntroProgram(candidate,7),gate)
    tick(progress,feedback(candidate,100,0),1.)
    tick(progress,feedback(candidate,100,1),1.1)
    tick(progress,feedback(candidate,101,2),1.2)
    return candidate,gate,progress


def optional_continuous_data(*, historical_optional=True):
    candidate=continuous_data()
    candidate['intro']['max_tcp_linear_speed']=None
    candidate['intro']['reviewed']['max_tcp_linear_speed']=None
    if historical_optional:
        executed=candidate['intro_preview']['execution_profile']['intro']
        executed['max_tcp_linear_speed']=None
        executed['reviewed']['max_tcp_linear_speed']=None
    rebind(candidate)
    return candidate


def test_reviewed_null_linear_monitor_omits_only_its_robot_comparison():
    program=IntroProgram(optional_linear_profile(),7)
    assert program.max_tcp_linear_speed is None
    script=program.program()
    assert 'fault = 16' not in script
    assert 'fault = 17' in script and '> 0.1500000000' in script
    assert 'fault = 14' in script and '> 0.1060000000' in script
    assert 'packet_age > 0.25' in script and 'fault = 12' in script
    assert 'is_within_safety_limits' in script and 'fault = 13' in script
    assert 'stopj(0.2)' in script
    assert 'if flag == 0 and not started' in script and 'elif flag == 1 and armed' in script


def test_both_reviewed_null_monitors_keep_robot_joint_and_controller_checks():
    script=IntroProgram(optional_linear_profile(angular=None),7).program()
    assert 'fault = 16' not in script and 'fault = 17' not in script
    assert 'fault = 14' in script and '> 0.1060000000' in script
    assert 'packet_age > 0.25' in script and 'is_within_safety_limits' in script


@pytest.mark.parametrize('where', ['profile', 'reviewed', 'both'])
def test_null_linear_monitor_requires_explicit_key_in_profile_and_review(where):
    candidate=optional_linear_profile()
    if where in ('profile','both'):del candidate['max_tcp_linear_speed']
    if where in ('reviewed','both'):del candidate['reviewed']['max_tcp_linear_speed']
    with pytest.raises(ValueError):IntroProgram(candidate,7)


@pytest.mark.parametrize('value', [0., -1., False, True, '', 'null', math.nan, math.inf, -math.inf])
def test_linear_monitor_rejects_values_other_than_null_or_finite_positive_numbers(value):
    candidate=optional_linear_profile()
    candidate['max_tcp_linear_speed']=value
    candidate['reviewed']['max_tcp_linear_speed']=value
    with pytest.raises(ValueError):IntroProgram(candidate,7)


@pytest.mark.parametrize('reviewed_value', [.05, False])
def test_null_linear_monitor_requires_matching_dynamics_review(reviewed_value):
    candidate=optional_linear_profile()
    candidate['reviewed']['max_tcp_linear_speed']=reviewed_value
    with pytest.raises(ValueError,match='review'):IntroProgram(candidate,7)
    candidate=optional_linear_profile();candidate['review']['dynamics']=False
    with pytest.raises(ValueError,match='review'):IntroProgram(candidate,7)


@pytest.mark.parametrize('angular', [.15, None])
def test_optional_linear_monitor_allows_finite_tcp_translation_after_start_ack(angular):
    candidate,gate,progress=active_progress(angular=angular)
    writer,reader=feedback(candidate,101,3);reader['tcp_speed'][0]=.6
    assert tick(progress,(writer,reader),1.3)=='ACTIVE'
    assert gate.reason is None and gate.sent[-1][:3]==(4,1,7)


@pytest.mark.parametrize('field,index,value', [
    ('tcp_speed',3,.151), ('joint_speed',0,.107), ('tcp_speed',0,math.nan),
])
def test_optional_linear_monitor_keeps_angular_joint_and_finite_telemetry_guards(field,index,value):
    candidate,gate,progress=active_progress()
    writer,reader=feedback(candidate,101,3);reader[field][index]=value
    sent=len(gate.sent)
    with pytest.raises(RuntimeError):tick(progress,(writer,reader),1.3)
    assert progress.state=='FAULT' and gate.reason and len(gate.sent)==sent


@pytest.mark.parametrize('field,index,value', [
    ('joint_speed',0,.107), ('tcp_speed',0,math.nan), ('tcp_speed',3,math.inf),
])
def test_both_optional_monitors_keep_joint_and_finite_telemetry_guards(field,index,value):
    candidate,gate,progress=active_progress(angular=None)
    writer,reader=feedback(candidate,101,3);reader[field][index]=value
    with pytest.raises(RuntimeError):tick(progress,(writer,reader),1.3)
    assert progress.state=='FAULT' and gate.reason


@pytest.mark.parametrize('state', [-12, -13, -14, -15, -16, -17])
def test_optional_monitors_never_ignore_reported_robot_faults(state):
    candidate,gate,progress=active_progress(angular=None)
    with pytest.raises(RuntimeError,match='fault'):
        tick(progress,feedback(candidate,state,3),1.3)
    assert progress.state=='FAULT' and gate.reason


def test_both_optional_monitors_keep_freshness_stop_and_no_automatic_resume():
    candidate,gate,progress=active_progress(angular=None);sent=len(gate.sent)
    with pytest.raises(RuntimeError,match='repeated'):
        tick(progress,feedback(candidate,101,3),1.2)
    assert progress.state=='FAULT' and len(gate.sent)==sent
    candidate,gate,progress=active_progress(angular=None);progress.stop();sent=len(gate.sent)
    with pytest.raises(RuntimeError,match='no longer active'):
        tick(progress,feedback(candidate,101,3),1.3)
    assert progress.state=='STOPPED' and len(gate.sent)==sent


@pytest.mark.parametrize('index,value', [(0,.0006), (3,.003)])
def test_both_optional_monitors_still_require_standstill_to_complete(index,value):
    candidate,gate,progress=active_progress(angular=None)
    tick(progress,feedback(candidate,102,3),1.3)
    writer,reader=feedback(candidate,102,3);reader['tcp_speed'][index]=value
    assert tick(progress,(writer,reader),1.4)=='SETTLING'
    assert progress.steady_since is None and gate.reason is None
    for i in range(9):
        state=tick(progress,feedback(candidate,102,3),1.5+i*.1)
        if state=='COMPLETE':break
    assert progress.state=='COMPLETE'


def test_continuous_preview_accepts_explicit_reviewed_null_executed_linear_monitor():
    candidate=optional_continuous_data()
    candidate['intro_preview']['execution_record']['trace'][0]['tcp_speed'][0]=.6
    rebind(candidate)
    assert reviewed_profile(candidate).intro['max_tcp_linear_speed'] is None


def test_current_null_linear_monitor_preserves_historical_numeric_execution_guard():
    candidate=optional_continuous_data(historical_optional=False)
    assert reviewed_profile(candidate).intro['max_tcp_linear_speed'] is None
    candidate['intro_preview']['execution_record']['trace'][0]['tcp_speed'][0]=.011
    rebind(candidate)
    with pytest.raises(ValueError,match='telemetry'):reviewed_profile(candidate)


def test_continuous_preview_null_linear_execution_requires_matching_review():
    candidate=optional_continuous_data()
    executed=candidate['intro_preview']['execution_profile']['intro']
    executed['reviewed']['max_tcp_linear_speed']=.01
    rebind(candidate)
    with pytest.raises(ValueError,match='review'):reviewed_profile(candidate)


@pytest.mark.parametrize('where,index,value', [
    ('trace',3,.041), ('trace',0,math.nan), ('completion',0,.0006), ('final',0,.0006),
])
def test_null_linear_continuous_preview_keeps_other_monitor_finite_and_standstill_checks(where,index,value):
    candidate=optional_continuous_data();record=candidate['intro_preview']['execution_record']
    if where=='trace':record['trace'][0]['tcp_speed'][index]=value
    elif where=='completion':record['trace'][1]['tcp_speed'][index]=value
    else:record['last_independent_feedback']['tcp_speed'][index]=value
    rebind(candidate)
    with pytest.raises(ValueError):reviewed_profile(candidate)
