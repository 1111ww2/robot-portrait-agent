"""草稿配置文件生成绝不能批准或运行安装程序。"""
import pytest
from leica_tracker.physical_draft import draft_from_snapshot
from leica_tracker.physical_runtime import reviewed_profile


def snapshot():
    return {'rtde':{'fresh':True,'tcp_pose_m_rad':[.1,-.5,.3,0.,3.14,0.],
                    'joints_rad':[0.,-1.,-1.,-1.,1.,0.], 'tcp_speed_m_s_rad_s':[0.]*6},
            'primary':{'tcp_offset_m_rad':[0.,0.,.2786,0.,0.,0.]}}


def test_draft_uses_explicit_amplitude_and_simulator_mapping_but_cannot_execute():
    result=draft_from_snapshot(snapshot(),'192.0.2.1',.02,.03,.04)
    assert result['status']=='DRAFT'
    assert result['paths']['MOVE_LEFT'][0]==pytest.approx([.08,-.5,.3])
    assert result['paths']['MOVE_UP'][0]==pytest.approx([.1,-.5,.33])
    assert result['paths']['MOVE_FARTHER'][0]==pytest.approx([.1,-.46,.3])
    assert not any(result['installation']['review'].values())
    assert result['reviewed_paths']=={}
    intro=result['intro'];assert intro['status']=='DRAFT'
    assert len(intro['waypoints'])==14 and intro['waypoints'][-1]==pytest.approx(snapshot()['rtde']['joints_rad'])
    assert len(intro['candidate_average_peak_joint_speed_rad_s'])==14
    assert intro['joint_speed'] is None and intro['joint_acceleration'] is None
    assert all(t%100==0 for t in intro['sample_times_ms'])
    previous=snapshot()['rtde']['joints_rad'];previous_time=0
    for point,stamp,rate in zip(intro['waypoints'],intro['sample_times_ms'],intro['candidate_average_peak_joint_speed_rad_s']):
        delta=max(abs(a-b) for a,b in zip(previous,point))
        assert delta>.02
        assert rate==pytest.approx(delta/((stamp-previous_time)/1000.))
        previous,previous_time=point,stamp
    assert not any(intro['review'].values()) and intro['reviewed']=={}
    with pytest.raises(ValueError,match='review'):reviewed_profile(result)


def test_draft_rejects_missing_current_or_nonfinite_data():
    for invalid in (0.,float('nan')):
        with pytest.raises(ValueError):draft_from_snapshot(snapshot(),'192.0.2.1',invalid,.03,.04)
    value=snapshot();value['rtde']['fresh']=False
    with pytest.raises(ValueError):draft_from_snapshot(value,'192.0.2.1',.02,.03,.04)


def test_simulator_speed_metadata_describes_only_software_limits():
    from leica_tracker.simulator_guidance import joint_velocity
    result=draft_from_snapshot(snapshot(),'192.0.2.1',.02,.03,.04)
    intro=result['intro'];meta=intro['simulator_control_candidate']
    assert meta['source']=='vision_mvp/src/leica_tracker/simulator_guidance.py'
    assert meta['joint_velocity_limit_rad_s']==.6 and meta['joint_acceleration_limit_rad_s2']==1.2
    assert meta['scope']=='SIMULATION_ONLY_NOT_PHYSICAL_LIMIT'
    assert intro['joint_speed'] is None and intro['joint_acceleration'] is None
    assert not any(intro['review'].values()) and result['status']=='DRAFT'
    # 探测实际模拟器算法，而不是物理硬件。
    values=joint_velocity([0.]*6,[.35]*6,[0.]*6,[.6]*6,.1)
    assert max(abs(v) for v in values)==pytest.approx(meta['joint_velocity_limit_rad_s'])
    values=joint_velocity([0.]*6,[.35]*6,[0.]*6,[0.]*6,.1)
    assert max(abs(v) for v in values)==pytest.approx(meta['joint_acceleration_limit_rad_s2']*.1)
