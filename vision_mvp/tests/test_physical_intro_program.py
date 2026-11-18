import copy,math
import pytest

from leica_tracker.physical_intro_program import IntroProgram, IntroProgress


def profile():
    start=[0.,-.8,1.,-.7,.5,0.]
    tilt=[0.,-.8,1.,-.7,.5,.2]
    result={'reference_joints':start,'reference_pose':[.1,.2,.3,0.,0.,0.],
            'tcp':[0.,0.,.1,0.,0.,0.],'waypoints':[tilt,start],
            'joint_speed':.08,'joint_acceleration':.1,'max_tcp_angular_speed':.15,
            'max_tcp_linear_speed':.05,'timeout':30.,
            'review':{'robot':True,'mount':True,'tcp':True,'payload':True,
                      'path':True,'dynamics':True,'stop':True,'gesture':True}}
    keys=('reference_joints','reference_pose','tcp','waypoints','joint_speed','joint_acceleration',
          'max_tcp_angular_speed','max_tcp_linear_speed','timeout')
    result['reviewed']={k:copy.deepcopy(result[k]) for k in keys}
    return result


class Gate:
    def __init__(self):self.sent=[];self.reason=None
    def send(self,*args):self.sent.append(args)
    def stop(self,reason):self.reason=reason


def test_fault_is_published_before_blocking_deceleration():
    script=IntroProgram(profile(),7).program()
    fault=script[script.index('    if fault != 0:'):]
    assert fault.index('write_output_integer_register(42, -fault)') < fault.index('stopj(0.2)')


def test_tcp_linear_fault_is_reported_instead_of_ack_timeout():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    tick(progress,feedback(p,100,1),1.1)
    tick(progress,feedback(p,101,2),1.2)
    with pytest.raises(RuntimeError,match='16.*TCP linear speed'):
        tick(progress,feedback(p,-16,2),1.5)
    assert 'TCP linear speed' in gate.reason


def feedback(p, state, sequence, *, joint_speed=None, joints=None):
    writer={'session':7,'state':state,'sequence':sequence}
    reader={'joints':joints or p['reference_joints'],'joint_speed':joint_speed or [0.]*6,
            'tcp_pose':p['reference_pose'],'tcp_speed':[0.]*6}
    return writer,reader


def tick(progress, samples, time):
    writer,reader=samples
    return progress.tick(writer,writer_sampled_at=time,reader=reader,reader_sampled_at=time,now=time)


def test_intro_requires_complete_reviewed_return_path_and_explicit_dynamics():
    p=profile();compiled=IntroProgram(p,7);script=compiled.program()
    assert compiled.robot_model=='ur10e'
    assert 'movej(' in script and 'get_steptime()' in script and 'packet_age > 0.25' in script
    assert 'get_actual_tcp_speed()' in script and '0.1500000000' in script and '0.0500000000' in script
    assert 'stopj(' in script and 'is_within_safety_limits' in script
    assert 'armed = False' in script and 'if flag == 0 and not started' in script
    assert 'elif flag == 1 and armed' in script
    assert 'set_tool' not in script and 'grip' not in script.lower()
    p=profile();p['review']['path']=False
    with pytest.raises(ValueError):IntroProgram(p,7)
    p=profile();p['waypoints']=p['waypoints'][:-1]
    with pytest.raises(ValueError):IntroProgram(p,7)
    p=profile();p['max_tcp_angular_speed']=math.inf
    with pytest.raises(ValueError):IntroProgram(p,7)
    p=profile();p['waypoints'][0][5]+=.01
    with pytest.raises(ValueError):IntroProgram(p,7)
    p=profile();p['joint_speed']*=2
    with pytest.raises(ValueError):IntroProgram(p,7)


def test_numeric_angular_monitor_still_faults_on_excess_tcp_rotation():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    tick(progress,feedback(p,100,1),1.1)
    tick(progress,feedback(p,101,2),1.2)
    writer,reader=feedback(p,101,3)
    reader['tcp_speed'][3]=.151
    with pytest.raises(RuntimeError,match='speed guard'):
        tick(progress,(writer,reader),1.3)
    assert progress.state=='FAULT' and gate.reason=='Intro speed guard'


def test_intro_only_completes_after_ack_and_continuous_fresh_still_return():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    assert gate.sent[-1][:3]==(1,0,7)
    tick(progress,feedback(p,100,1),1.1)
    assert gate.sent[-1][:3]==(2,1,7)
    tick(progress,feedback(p,101,2),1.2)
    assert progress.state=='ACTIVE'
    for n in range(3,13):
        t=1.+n*.1
        tick(progress,feedback(p,102,n),t)
        if progress.state=='COMPLETE':break
    assert progress.state=='COMPLETE'


def test_intro_stop_and_invalid_feedback_latch_without_forcing_return():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    progress.stop()
    assert progress.state=='STOPPED' and gate.reason
    with pytest.raises(RuntimeError):tick(progress,feedback(p,101,1),1.1)
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    tick(progress,feedback(p,100,1),1.1)
    tick(progress,feedback(p,101,2),1.2)
    moving=feedback(p,102,2,joint_speed=[.1,0.,0.,0.,0.,0.])
    assert tick(progress,moving,1.3)=='SETTLING'
    with pytest.raises(RuntimeError):tick(progress,moving,3.4)
    assert progress.state=='FAULT' and gate.reason


def test_intro_requires_independent_advancing_writer_and_reader_samples():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    writer,reader=feedback(p,101,1)
    with pytest.raises(RuntimeError,match='repeated'):
        progress.tick(writer,writer_sampled_at=1.1,reader=reader,reader_sampled_at=1.,now=1.1)
    assert progress.state=='FAULT' and gate.reason


def test_ack_timeout_reports_expected_and_observed_without_sending_again():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    with pytest.raises(RuntimeError,match='expected=1.*observed=0'):
        tick(progress,feedback(p,100,0),1.21)
    assert progress.ack_diagnostic['expected_sequence']==1
    assert progress.ack_diagnostic['observed_sequence']==0
    assert len(gate.sent)==1 and progress.state=='FAULT'


def test_intro_rejects_hold_ack_in_running_state_without_enable():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    with pytest.raises(RuntimeError,match='hold acknowledgement'):
        tick(progress,feedback(p,101,1),1.1)
    assert progress.state=='FAULT' and len(gate.sent)==1 and gate.sent[0][1]==0


def test_intro_completion_dwell_uses_only_samples_after_completion_observed():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    tick(progress,feedback(p,100,1),1.1)
    tick(progress,feedback(p,101,2),1.2)
    writer,reader=feedback(p,102,3)
    progress.tick(writer,writer_sampled_at=1.30,reader=reader,reader_sampled_at=1.29,now=1.31)
    assert progress.steady_since is None
    progress.tick(writer,writer_sampled_at=1.32,reader=reader,reader_sampled_at=1.30,now=1.33)
    assert progress.steady_since is None and progress.state=='SETTLING'
    progress.tick(writer,writer_sampled_at=1.34,reader=reader,reader_sampled_at=1.32,now=1.35)
    assert progress.steady_since==1.32


def test_completion_ack_may_precede_final_position_on_independent_receiver():
    p=profile();gate=Gate();progress=IntroProgress(IntroProgram(p,7),gate)
    tick(progress,feedback(p,100,0),1.)
    tick(progress,feedback(p,100,1),1.1)
    tick(progress,feedback(p,101,2),1.2)
    old_position=list(p['reference_joints']);old_position[5]+=.01
    writer,reader=feedback(p,102,3,joints=old_position,joint_speed=[0,0,0,0,0,.01])
    assert progress.tick(writer,writer_sampled_at=1.30,reader=reader,reader_sampled_at=1.28,now=1.31)=='SETTLING'
    assert progress.steady_since is None
    for n in range(9):
        tick(progress,feedback(p,102,3),1.34+n*.1)
        if progress.state=='COMPLETE':break
    assert progress.state=='COMPLETE' and gate.reason is None
