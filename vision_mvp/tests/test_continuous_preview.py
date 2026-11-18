"""连续运行接受与每点测量不同；仅限离线。"""
import copy

import pytest

from leica_tracker.physical_runtime import reviewed_profile
from test_physical_runtime import data


def continuous_data():
    candidate = data()
    intro = candidate['intro']
    q, pose = intro['reference_joints'], intro['reference_pose']
    trace = []
    for i in range(9):
        trace.append({'sampled_at':100.+i*.1,'writer_age_s':.01,'reader_age_s':.01,
                      'resident_state':101 if i==0 else 102,
                      'ack':20,'reader_ack':20,'writer_timestamp':1000.+i*.1,
                      'reader_timestamp':1000.+i*.1,'pending':[21,99.,1],
                      'joints':list(intro['waypoints'][0] if i==0 else q),
                      'tcp_speed':[0.]*6})
    record = {'outcome':'complete','error':None,'ack_diagnostic':None,'trace':trace,
              'last_independent_feedback':{'sampled_at':100.79,'joints':list(q),'tcp_pose':list(pose),
                                           'joint_speed':[0.]*6,'tcp_speed':[0.]*6}}
    preview = {'status':'OPERATOR_ACCEPTED_CONTINUOUS','robot_ip':candidate['robot_ip'],
               'reference_joints':list(q),'reference_pose':list(pose),'tcp':list(intro['tcp']),
               'waypoints':copy.deepcopy(intro['waypoints']),'operator_decision':'accept',
               'execution_profile':{'mode':'onsite_manual_14_point_sequence',
                                    'robot_ip':candidate['robot_ip'],'intro':copy.deepcopy(intro)},
               'execution_record':record}
    preview['reviewed'] = copy.deepcopy(preview)
    candidate['intro_preview'] = preview
    return candidate


def rebind(candidate):
    preview=candidate['intro_preview']
    preview['reviewed']=copy.deepcopy({k:v for k,v in preview.items() if k!='reviewed'})


def test_accepts_complete_continuous_record_without_inventing_point_observations():
    candidate=continuous_data()
    assert 'observations' not in candidate['intro_preview']
    assert reviewed_profile(candidate).host=='192.0.2.1'


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(operator_decision='reject'),
    lambda p:p['execution_profile'].update(robot_ip='192.0.2.2'),
    lambda p:p['execution_profile']['intro']['waypoints'][1].__setitem__(0,.7),
    lambda p:p['execution_record'].update(outcome='fault'),
    lambda p:p['execution_record'].update(error='timeout'),
    lambda p:p['execution_record'].update(ack_diagnostic={'expected_sequence':7}),
    lambda p:p['execution_record'].update(trace=[]),
    lambda p:p['execution_record']['last_independent_feedback']['joints'].__setitem__(0,.7),
    lambda p:p['execution_record']['last_independent_feedback']['tcp_pose'].__setitem__(0,.7),
    lambda p:p['execution_record']['last_independent_feedback']['tcp_pose'].__setitem__(3,.1),
    lambda p:p['execution_record']['last_independent_feedback']['tcp_speed'].__setitem__(0,.01),
    lambda p:p['execution_record']['last_independent_feedback']['joint_speed'].__setitem__(0,.01),
    lambda p:p['execution_record']['last_independent_feedback'].update(sampled_at=90.),
    lambda p:p['execution_record']['trace'][-1].update(resident_state=101),
    lambda p:p['execution_record']['trace'][-1].update(reader_ack=19),
    lambda p:p['execution_record']['trace'][-1].update(reader_age_s=.2),
    lambda p:p['execution_record']['trace'][-1].update(writer_timestamp=1.),
    lambda p:p['execution_record']['trace'][-1]['tcp_speed'].__setitem__(0,float('nan')),
])
def test_rejects_failed_mismatched_or_invalid_execution_even_if_operator_snapshot_rebound(mutation):
    candidate=continuous_data();mutation(candidate['intro_preview']);rebind(candidate)
    with pytest.raises(ValueError):reviewed_profile(candidate)


def test_rejects_missing_native_completion_or_insufficient_settling():
    candidate=continuous_data()
    for row in candidate['intro_preview']['execution_record']['trace']:row['resident_state']=101
    rebind(candidate)
    with pytest.raises(ValueError):reviewed_profile(candidate)
    candidate=continuous_data()
    candidate['intro_preview']['execution_record']['trace']=candidate['intro_preview']['execution_record']['trace'][:3]
    rebind(candidate)
    with pytest.raises(ValueError):reviewed_profile(candidate)


def test_acceptance_does_not_follow_edited_record_and_keeps_historical_speed_distinct():
    candidate=continuous_data()
    candidate['intro_preview']['execution_record']['trace'][0]['ack']=21
    with pytest.raises(ValueError):reviewed_profile(candidate)
    candidate=continuous_data();intro=candidate['intro']
    intro['joint_speed']=.03;intro['reviewed']['joint_speed']=.03
    assert reviewed_profile(candidate).intro['joint_speed']==.03
    assert candidate['intro_preview']['execution_profile']['intro']['joint_speed']==.02


def test_legacy_step_preview_still_valid():
    assert reviewed_profile(data()).host=='192.0.2.1'
