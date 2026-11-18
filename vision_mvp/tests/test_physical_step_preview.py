import copy, importlib.util, json, threading, time
import pytest

from leica_tracker.physical_step_preview import StepProgram,StepSequence,draft_step_template,open_step
from leica_tracker.physical_intro_program import IntroProgress


def example():
    start=[0.,-.8,1.,-.7,.5,0.];tcp=[0.,0.,.1,0.,0.,0.]
    points=[[0.,-.8,1.,-.7,.5,.01*(i+1)] for i in range(14)]
    draft={'status':'DRAFT','robot_ip':'192.0.2.1',
           'intro':{'reference_joints':start,'tcp':tcp,'waypoints':points}}
    step={'status':'STEP_REVIEWED','robot_ip':'192.0.2.1','reference_joints':start,'reference_pose':[.1,.2,.3,0.,0.,0.],
          'tcp':tcp,'target_joints':points[0],'joint_tolerance':.001,'joint_speed':.08,'joint_acceleration':.1,
          'max_tcp_angular_speed':.15,'max_tcp_linear_speed':.05,'timeout':30.,
          'review':{k:True for k in ('robot','mount','tcp','payload','path','dynamics','stop','gesture')}}
    keys=('robot_ip','reference_joints','reference_pose','tcp','target_joints','joint_tolerance','joint_speed',
          'joint_acceleration','max_tcp_angular_speed','max_tcp_linear_speed','timeout')
    step['reviewed']={k:copy.deepcopy(step[k]) for k in keys}
    return draft,step


def test_step_program_has_one_reviewed_move_and_no_implicit_return():
    draft,step=example();script=StepProgram(step,7).program()
    assert StepProgram(step,7).robot_model=='ur10e'
    assert script.count('movej(')==1 and 'armed = False' in script and 'stopj(' in script
    assert 'set_tool' not in script and 'grip' not in script.lower()
    step['target_joints'][5]+=.01
    with pytest.raises(ValueError):StepProgram(step,7)
    draft,step=example();step['review']['path']=False
    with pytest.raises(ValueError):StepProgram(step,7)


def test_next_runs_only_one_point_then_waits_for_explicit_next():
    draft,step=example();events=[]
    class Session:
        def __init__(self,target):
            self.reader=type('Reader',(),{'failure':None,'latest':({'joints':target,'tcp_speed':[0.]*6,
                'joint_speed':[0.]*6,'tcp_pose':[0.]*6},time.monotonic())})()
            self.progress=type('Progress',(),{'program':type('Program',(),{'joint_tolerance':.001})()})()
        def tick(self):events.append('tick');return 'COMPLETE'
        def stop(self):events.append('stop')
        def close(self):events.append('close')
    def opener(host,profile,read_guest):events.append('open');return Session(profile['target_joints'])
    preview=StepSequence(draft,'192.0.2.1',lambda:None,opener=opener)
    assert preview.state=='WAITING' and events==[]
    preview.next(step)
    assert preview.state=='MOVING' and events==['open']
    assert preview.tick()=='WAITING' and preview.index==1
    assert events==['open','tick','close']
    with pytest.raises(ValueError):preview.next(step)
    assert events==['open','tick','close']
    preview.stop()
    with pytest.raises(RuntimeError):preview.next(step)


def test_stop_during_open_latches_and_no_tick_or_following_point_starts():
    draft,step=example();events=[];entered=threading.Event();release=threading.Event()
    class Session:
        def tick(self):events.append('tick');return 'COMPLETE'
        def stop(self):events.append('stop')
        def close(self):events.append('close')
    def opener(*args):entered.set();release.wait(2);return Session()
    preview=StepSequence(draft,'192.0.2.1',lambda:None,opener=opener)
    errors=[]
    def execute():
        try:preview.next(step)
        except RuntimeError as error:errors.append(str(error))
    worker=threading.Thread(target=execute);worker.start()
    assert entered.wait(1)
    preview.stop();release.set();worker.join(2)
    assert errors and events==['stop','close'] and preview.state=='STOPPED'


def test_stop_latches_before_gate_and_keeps_independent_reader_until_explicit_close():
    draft,step=example();events=[]
    class Gate:
        def stop(self,reason):events.append('gate_stop')
    class Session:
        gate=Gate()
        def tick(self):return 'ACTIVE'
        def stop(self):events.append('stop')
        def close(self):events.append('close')
    preview=StepSequence(draft,'192.0.2.1',lambda:{'active':True},opener=lambda *args:Session())
    preview.next(step);preview.stop()
    assert events==['gate_stop','stop'] and preview.state=='STOPPED'
    preview.close()
    assert events==['gate_stop','stop','close']


def test_stop_during_open_invalidates_guest_watcher_immediately():
    draft,step=example();entered=threading.Event();release=threading.Event();observed=[]
    class Session:
        def stop(self):pass
        def close(self):pass
    def opener(host,profile,read_guest):
        entered.set();release.wait(2);observed.append(read_guest());return Session()
    preview=StepSequence(draft,'192.0.2.1',lambda:{'active':True,'camera_connected':True,'camera_fresh':True},opener=opener)
    errors=[]
    def execute():
        try:preview.next(step)
        except RuntimeError as error:errors.append(str(error))
    worker=threading.Thread(target=execute);worker.start();assert entered.wait(1)
    preview.stop();release.set();worker.join(2)
    assert errors and observed[0]['active'] is False and observed[0]['camera_fresh'] is False


def test_stopped_preview_does_not_call_guest_reader():
    draft,_=example();calls=[]
    preview=StepSequence(draft,'192.0.2.1',lambda:calls.append('read'))
    preview.stop()
    assert preview._read_guest()['active'] is False and not calls


def test_close_while_active_stops_gate_before_closing_reader():
    draft,step=example();events=[]
    class Gate:
        def stop(self,reason):events.append('gate_stop')
    class Session:
        gate=Gate()
        def tick(self):return 'ACTIVE'
        def stop(self):events.append('stop')
        def close(self):events.append('close')
    preview=StepSequence(draft,'192.0.2.1',lambda:{'active':True},opener=lambda *args:Session())
    preview.next(step);preview.close()
    assert events==['gate_stop','stop','close'] and preview.state=='STOPPED'


def test_acceptance_requires_all_points_and_explicit_accept():
    draft,step=example()
    class Session:
        def __init__(self,target):
            self.reader=type('Reader',(),{'failure':None,'latest':({'joints':target,'tcp_speed':[0.]*6,
                'joint_speed':[0.]*6,'tcp_pose':[0.]*6},time.monotonic())})()
            self.progress=type('Progress',(),{'program':type('Program',(),{'joint_tolerance':.001})()})()
        def tick(self):return 'COMPLETE'
        def stop(self):pass
        def close(self):pass
    preview=StepSequence(draft,'192.0.2.1',lambda:None,opener=lambda host,profile,read_guest:Session(profile['target_joints']))
    with pytest.raises(ValueError):preview.acceptance('accept')
    for index in range(14):
        current=copy.deepcopy(step)
        current['reference_joints']=draft['intro']['reference_joints'] if index==0 else draft['intro']['waypoints'][index-1]
        if index:current['reference_pose']=preview.observations[-1]['tcp_pose']
        current['target_joints']=draft['intro']['waypoints'][index]
        current['reviewed']={key:copy.deepcopy(current[key]) for key in current['reviewed']}
        preview.next(current);preview.tick()
    with pytest.raises(ValueError):preview.acceptance('reject')
    record=preview.acceptance('accept')
    assert record['observed_indices']==list(range(1,15)) and record['operator_decision']=='accept'
    assert record['reviewed']=={key:value for key,value in record.items() if key!='reviewed'}
    assert len(record['observations'])==14


def test_next_template_uses_previous_observed_pose_but_remains_unapproved():
    draft,_=example()
    observed=[{'index':1,'joints':draft['intro']['waypoints'][0],
               'tcp_pose':[.2,.3,.4,0.,0.,0.],'sampled_at':10.}]
    template=draft_step_template(draft,1,observations=observed)
    assert template['reference_pose']==observed[0]['tcp_pose']
    assert template['status']=='DRAFT' and not any(template['review'].values()) and template['reviewed']=={}
    with pytest.raises(ValueError):StepProgram(template,7)
    observed[0]['joints']=[99.]*6
    with pytest.raises(ValueError):draft_step_template(draft,1,observations=observed)


def test_step_endpoint_tolerance_must_be_reviewed_and_less_than_separation():
    _,step=example();separation=step['target_joints'][5]-step['reference_joints'][5]
    step['joint_tolerance']=separation;step['reviewed']['joint_tolerance']=separation
    with pytest.raises(ValueError,match='tolerance'):StepProgram(step,7)
    step['joint_tolerance']=separation/2
    with pytest.raises(ValueError,match='review'):StepProgram(step,7)
    _,step=example();step['joint_tolerance']=.004;step['reviewed']['joint_tolerance']=.004
    with pytest.raises(ValueError,match='tolerance'):StepProgram(step,7)


def test_step_rejects_previous_joint_position_as_completed_target():
    _,step=example();program=StepProgram(step,7)
    class Gate:
        def send(self,*args):pass
        def stop(self,reason):pass
    progress=IntroProgress(program,Gate())
    reader={'joints':step['reference_joints'],'joint_speed':[0.]*6,
            'tcp_pose':step['reference_pose'],'tcp_speed':[0.]*6}
    def tick(state,ack,t):
        return progress.tick({'session':7,'state':state,'sequence':ack},writer_sampled_at=t,
                             reader=reader,reader_sampled_at=t,now=t)
    tick(100,0,1.);tick(100,1,1.1);tick(101,2,1.2)
    assert tick(102,3,1.3)=='SETTLING'
    with pytest.raises(RuntimeError,match='stationary'):
        tick(102,3,3.4)


def test_step_bootstrap_rejects_current_pose_already_inside_target_tolerance():
    from test_physical_intro_bootstrap import setup
    _,step=example();step['target_joints'][-1]=.002;step['reviewed']['target_joints'][-1]=.002
    _,deps,objects=setup();make_rtde=deps['rtde_factory']
    def at_target(host,**kwargs):
        source=make_rtde(host,**kwargs)
        if kwargs.get('receive_only'):source.latest[0]['joints'][-1]=.002
        return source
    deps['rtde_factory']=at_target
    with pytest.raises(RuntimeError,match='already within'):
        open_step('192.0.2.1',step,lambda:None,dependencies=deps)
    assert 'command' not in objects


def test_later_step_rejects_pose_not_bound_to_previous_observation():
    draft,step=example()
    class Session:
        def __init__(self,target):
            self.reader=type('Reader',(),{'failure':None,'latest':({'joints':target,'tcp_speed':[0.]*6,
                'joint_speed':[0.]*6,'tcp_pose':[.5,.6,.7,0.,0.,0.]},time.monotonic())})()
            self.progress=type('Progress',(),{'program':type('Program',(),{'joint_tolerance':.001})()})()
        def tick(self):return 'COMPLETE'
        def stop(self):pass
        def close(self):pass
    preview=StepSequence(draft,'192.0.2.1',lambda:None,opener=lambda host,p,r:Session(p['target_joints']))
    preview.next(step);preview.tick()
    second=copy.deepcopy(step);second['reference_joints']=draft['intro']['waypoints'][0]
    second['target_joints']=draft['intro']['waypoints'][1]
    second['reviewed']={key:copy.deepcopy(second[key]) for key in second['reviewed']}
    with pytest.raises(ValueError,match='previous observed'):
        preview.validate_next(second)


def test_preview_launcher_preflights_first_review_before_touching_guest(tmp_path,monkeypatch):
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts'/'run-physical-preview.py'
    spec=importlib.util.spec_from_file_location('preview_preflight_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    touched=[]
    monkeypatch.setattr(module,'_launcher',lambda:touched.append('launcher'))
    draft,step=example();draft_path=tmp_path/'draft.json';draft_path.write_text(json.dumps(draft))
    reviews=tmp_path/'reviews';reviews.mkdir()
    with pytest.raises(ValueError,match='First step review'):
        module.main(['--draft',str(draft_path),'--reviews',str(reviews),'--execute'])
    assert not touched
    unreviewed=copy.deepcopy(step);unreviewed['status']='DRAFT'
    (reviews/'01.json').write_text(json.dumps(unreviewed))
    with pytest.raises(ValueError,match='First step review'):
        module.main(['--draft',str(draft_path),'--reviews',str(reviews),'--execute'])
    assert not touched
