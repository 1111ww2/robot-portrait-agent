import copy,importlib.util,json
from pathlib import Path
import pytest

from leica_tracker.physical_step_preview import SharedPreviewPlan,StepProgram,prepare_shared_step,shared_plan_template,open_step
from test_physical_step_preview import example


def reviewed():
    draft,step=example()
    draft['intro']['reference_pose']=copy.deepcopy(step['reference_pose'])
    dynamic={'joint_tolerance':.001,'joint_speed':.08,'joint_acceleration':.1,
             'max_tcp_angular_speed':.15,'max_tcp_linear_speed':.05,'timeout':30.}
    stop={'resident_timeout':.25,'stopj_acceleration':.2,'watch_interval':.05,'watch_timeout':.3}
    p={'status':'PREVIEW_PLAN_REVIEWED','robot_ip':draft['robot_ip'],
       'reference_joints':copy.deepcopy(draft['intro']['reference_joints']),
       'reference_pose':copy.deepcopy(step['reference_pose']),'tcp':copy.deepcopy(draft['intro']['tcp']),
       'waypoints':copy.deepcopy(draft['intro']['waypoints']),'dynamics':dynamic,'stop':stop,
       'review':{key:True for key in ('robot','mount','tcp','payload','path','dynamics','stop','gesture')},
       'operator_approval':'accept_all_14'}
    p['reviewed']=copy.deepcopy(p)
    return draft,p


def measured(plan, *, index=0,stamp=10.):
    joints=plan['reference_joints'] if index==0 else plan['waypoints'][index-1]
    return {'sampled_at':stamp,'tcp_sampled_at':stamp,'joints':copy.deepcopy(joints),
            'tcp_pose':copy.deepcopy(plan['reference_pose']),'joint_speed':[0.]*6,
            'tcp_speed':[0.]*6,'tcp_offset':copy.deepcopy(plan['tcp'])}


def test_one_review_binds_all_14_points_and_explicit_stop_configuration():
    draft,p=reviewed();plan=SharedPreviewPlan(draft,p)
    profile=plan.step_profile(0,[],measured(p),now=10.)
    compiled=StepProgram(profile,1)
    assert compiled.final_joints==tuple(draft['intro']['waypoints'][0])
    assert compiled.joint_tolerance==.001 and profile['stop']==p['stop']
    assert 'packet_age > 0.2500000000' in compiled.program()
    assert 'stopj(0.2000000000)' in compiled.program()
    p['waypoints'][5][0]+=.1
    with pytest.raises(ValueError):SharedPreviewPlan(draft,p)


def test_missing_or_changed_review_fails_without_fabricating_approval():
    draft,p=reviewed();p['review']['path']=False
    with pytest.raises(ValueError):SharedPreviewPlan(draft,p)
    draft,p=reviewed();p['dynamics']['joint_speed']*=2
    with pytest.raises(ValueError):SharedPreviewPlan(draft,p)
    draft,p=reviewed();p['stop']['resident_timeout']=.4
    with pytest.raises(ValueError):SharedPreviewPlan(draft,p)


def test_next_step_uses_prior_observation_and_checks_fresh_current_state():
    draft,p=reviewed();plan=SharedPreviewPlan(draft,p)
    prior_pose=[.2,.3,.4,0.,0.,0.]
    observations=[{'index':1,'joints':p['waypoints'][0], 'tcp_pose':prior_pose,'sampled_at':9.,'joint_tolerance':.001}]
    current=measured(p,index=1);current['tcp_pose']=prior_pose
    result=plan.step_profile(1,observations,current,now=10.)
    assert result['reference_pose']==prior_pose and result['target_joints']==p['waypoints'][1]
    assert result['reviewed']['reference_pose']==prior_pose
    for key,value in [('sampled_at',9.),('tcp_sampled_at',9.),('tcp_offset',[9.]*6),('joints',[9.]*6)]:
        bad=copy.deepcopy(current);bad[key]=value
        with pytest.raises(ValueError):plan.step_profile(1,observations,bad,now=10.)
    bad=copy.deepcopy(current);bad['tcp_pose'][0]+=.1
    with pytest.raises(ValueError):plan.step_profile(1,observations,bad,now=10.)


def test_shared_step_preparation_uses_only_receive_only_channels():
    from test_physical_intro_bootstrap import setup
    draft,p=reviewed();plan=SharedPreviewPlan(draft,p)
    _,deps,objects=setup()
    result=prepare_shared_step('192.0.2.1',plan,0,[],dependencies=deps)
    assert result['target_joints']==p['waypoints'][0]
    assert 'reader' in objects and 'writer' not in objects and 'command' not in objects


def test_shared_step_preparation_rejects_changed_actual_pose_before_upload():
    from test_physical_intro_bootstrap import setup
    draft,p=reviewed();plan=SharedPreviewPlan(draft,p)
    _,deps,objects=setup(wrong_reference=True)
    with pytest.raises(ValueError):prepare_shared_step('192.0.2.1',plan,0,[],dependencies=deps)
    assert 'command' not in objects and 'writer' not in objects


def test_shared_plan_template_has_no_invented_approval_or_dynamics():
    draft,_=reviewed();template=shared_plan_template(draft)
    assert template['status']=='DRAFT' and template['operator_approval'] is None
    assert not any(template['review'].values()) and template['reviewed']=={}
    assert all(value is None for value in template['dynamics'].values())
    assert all(value is None for value in template['stop'].values())
    with pytest.raises(ValueError):SharedPreviewPlan(draft,template)


def test_shared_stop_settings_are_applied_to_resident_and_watchdog(monkeypatch):
    import leica_tracker.physical_step_preview as module
    draft,p=reviewed();p['stop']={'resident_timeout':.2,'stopj_acceleration':.15,
                                  'watch_interval':.04,'watch_timeout':.2}
    p['reviewed']=copy.deepcopy({k:v for k,v in p.items() if k!='reviewed'})
    plan=SharedPreviewPlan(draft,p);step=plan.step_profile(0,[],measured(p),now=10.)
    script=StepProgram(step,1).program()
    assert 'packet_age > 0.2000000000' in script and 'stopj(0.1500000000)' in script
    calls=[]
    monkeypatch.setattr(module,'open_intro',lambda host,profile,read_guest,**kw:calls.append(kw))
    open_step('192.0.2.1',step,lambda:None)
    assert calls[0]['watch_interval']==.04 and calls[0]['watch_timeout']==.2
    with pytest.raises(ValueError):open_step('192.0.2.1',step,lambda:None,watch_timeout=.3)


def preview_script(monkeypatch,tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts'/'run-physical-preview.py'
    spec=importlib.util.spec_from_file_location('shared_preview_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    return module


def test_unreviewed_shared_plan_does_not_change_running_guest(tmp_path,monkeypatch):
    module=preview_script(monkeypatch,tmp_path)
    draft,p=reviewed();draft_path=tmp_path/'draft.json';plan_path=tmp_path/'plan.json'
    draft_path.write_text(json.dumps(draft));p['review']['path']=False;plan_path.write_text(json.dumps(p))
    calls=[];monkeypatch.setattr(module,'_launcher',lambda:calls.append('launcher'))
    with pytest.raises(ValueError,match='Shared 14-step review'):
        module.main(['--draft',str(draft_path),'--plan',str(plan_path),'--execute'])
    assert calls==[]


def test_single_plan_template_cli_writes_only_unapproved_data(tmp_path,monkeypatch):
    module=preview_script(monkeypatch,tmp_path)
    draft,_=reviewed();draft_path=tmp_path/'draft.json';template_path=tmp_path/'plan.json'
    draft_path.write_text(json.dumps(draft));calls=[]
    monkeypatch.setattr(module,'_launcher',lambda:calls.append('launcher'))
    assert module.main(['--draft',str(draft_path),'--plan-template',str(template_path)])==0
    template=json.loads(template_path.read_text())
    assert template['status']=='DRAFT' and not any(template['review'].values())
    assert template['operator_approval'] is None and template['reviewed']=={} and calls==[]
    with pytest.raises(ValueError):SharedPreviewPlan(draft,template)


def test_next_uses_unchanged_plan_and_fresh_readonly_preparation(tmp_path,monkeypatch):
    from leica_tracker.physical_step_preview import StepSequence
    module=preview_script(monkeypatch,tmp_path)
    draft,p=reviewed();draft_path=tmp_path/'draft.json';plan_path=tmp_path/'plan.json'
    draft_path.write_text(json.dumps(draft));plan_path.write_text(json.dumps(p))
    plan=SharedPreviewPlan(draft,p);calls=[]
    class Session:
        def stop(self):calls.append('stop')
        def close(self):calls.append('close')
    preview=StepSequence(draft,p['robot_ip'],lambda:None,
                         opener=lambda host,step,read_guest:(calls.append('open') or Session()))
    def prepare(host,review,index,observations):
        calls.append('read-only')
        return review.step_profile(index,observations,measured(p),now=10.)
    module.shared_next(preview,plan,draft_path,plan_path,prepare=prepare)
    assert calls==['read-only','open'] and preview.state=='MOVING'


def test_changed_shared_plan_or_draft_latches_without_opening_robot(tmp_path,monkeypatch):
    from leica_tracker.physical_step_preview import StepSequence
    module=preview_script(monkeypatch,tmp_path)
    draft,p=reviewed();draft_path=tmp_path/'draft.json';plan_path=tmp_path/'plan.json'
    draft_path.write_text(json.dumps(draft));plan_path.write_text(json.dumps(p))
    plan=SharedPreviewPlan(draft,p);calls=[]
    preview=StepSequence(draft,p['robot_ip'],lambda:None,opener=lambda *args:calls.append('open'))
    def prepare(host,review,index,observations):
        changed=copy.deepcopy(p);changed['dynamics']['joint_speed']*=2
        plan_path.write_text(json.dumps(changed))
        return review.step_profile(index,observations,measured(p),now=10.)
    with pytest.raises(ValueError,match='changed'):
        module.shared_next(preview,plan,draft_path,plan_path,prepare=prepare)
    assert calls==[] and preview.state=='STOPPED'
