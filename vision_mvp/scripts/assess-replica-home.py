"""离线比较标称模型，不建立网络连接，也不验证碰撞或动力学。"""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from urx_browser_simulator.kinematics import solve_ik,forward_kinematics,pose_error
from leica_tracker.physical_spatial import SIM_HOME,HOME_JOINTS,nominal_fk
from leica_tracker.simulator_spatial import input_target_3d,pose_at_xyz

tcp=[0,0,.2786,0,0,0]
current_q=np.radians([-90,-60,-100,-60,90,0])
current_pose,_=forward_kinematics(current_q,tcp)
home_pose,_=forward_kinematics(HOME_JOINTS,tcp)
cases=[('current_home',current_q,current_pose),('simulator_home',np.array(HOME_JOINTS),home_pose)]
raised=home_pose.copy();raised[2]+=.10
rq,re=solve_ik(raised,np.array(HOME_JOINTS),tcp,iterations=200)
cases.append(('simulator_home_raised_10cm',rq,raised))
results=[]
for name,neutral_q,neutral in cases:
    records=[]
    for command in itertools.product([-1.,-.5,0.,.5,1.],[-1.,0.,1.],[-1.,0.,1.]):
        offset=np.array(input_target_3d(*command))-SIM_HOME
        q=neutral_q.copy();worst_base=0.;worst_error=0.
        for f in [.2,.4,.6,.8,1.]:
            target=np.array(neutral);target[:3]+=offset*f
            q,_=solve_ik(target,q,tcp,iterations=120,position_tolerance_m=1e-5,orientation_tolerance_rad=1e-4)
            actual,_=forward_kinematics(q,tcp);error=pose_error(actual,target)
            worst_error=max(worst_error,float(np.linalg.norm(error[:3])))
            worst_base=max(worst_base,abs(math.degrees(q[0]-neutral_q[0])))
        records.append(dict(command=command,position_error_m=worst_error,
            endpoint_orientation_error_rad=float(np.linalg.norm(error[3:])),base_excursion_deg=worst_base,
            reachable=bool(worst_error<.001 and np.linalg.norm(error[3:])<.01),
            inside_base_limit=bool(worst_base<=3.01)))
    summary=dict(name=name,home_joints_deg=np.degrees(neutral_q).tolist(),home_tcp=list(neutral),
        samples=len(records),reachable=sum(s['reachable'] for s in records),
        reachable_with_base_limit=sum(s['reachable'] and s['inside_base_limit'] for s in records),
        max_base_excursion_deg=max(s['base_excursion_deg'] for s in records),
        worst_position_error_m=max(s['position_error_m'] for s in records))
    print(json.dumps(summary),flush=True)
    results.append(dict(**summary,records=records))
report=dict(analysis_only=True,model='Nominal UR5e DH, 278.6 mm TCP',
    limitations=['Sparse IK sampling is not collision validation or proof of all continuous paths.',
                 'No motion, upload, controller safety-limit, payload or dynamic test.',
                 'Simulator home is a candidate, not an instruction to move physical hardware.'],results=results)
Path('docs/lab/2026-09-09-replica-home-assessment.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
