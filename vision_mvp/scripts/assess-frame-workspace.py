"""离线比较候选方案，全程不连接机器人或仿真器。"""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from urx_browser_simulator.kinematics import solve_ik, forward_kinematics, joint_transforms, pose_error

offset = [0, 0, .2786, 0, 0, 0]
def metrics(q):
    frames = joint_transforms(q)
    tcp, _ = forward_kinematics(q, offset)
    jac = np.column_stack([np.r_[np.cross(t[:3, 2], tcp[:3]-t[:3, 3]), t[:3, 2]] for t in frames[:-1]])
    sv = np.linalg.svd(jac, compute_uv=False)
    if sv[-1] < 1e-8:
        return {'translation_joint_gain_rad_per_m': 1e9, 'wrist_margin_deg': 0.}
    # 固定工具中心点（TCP）的角速度为零，比较各单位平移方向上的最大单关节速率；这个量只用于比较。
    gain = np.linalg.norm(np.linalg.solve(jac, np.eye(6)[:, :3]), axis=1).max()
    margin = min(abs(q[4]-k*math.pi) for k in range(-3,4))
    return {'translation_joint_gain_rad_per_m': float(gain), 'wrist_margin_deg': math.degrees(margin)}

candidates = [('current', .5, .2093, math.pi)]
for depth, height in [(.35,.30),(.45,.21),(.45,.30),(.45,.40),(.5,.30),(.55,.30)]:
    candidates.append((f'down_{depth}_{height}', -depth, height, math.pi))
candidates += [('outward_045_030',-.45,.30,math.pi/2), ('tilted_045_030',-.45,.30,3*math.pi/4)]
results=[]
for name,y,z,rx in candidates:
    target=[0,y,z,rx,0,0]
    seed=[-math.pi/2 if y>0 else math.pi/2,-math.pi/2,math.pi/2,-math.pi/2,-math.pi/2,0]
    neutral,e=solve_ik(target,seed,offset,iterations=250,position_tolerance_m=1e-5,orientation_tolerance_rad=1e-4)
    samples=[]
    for dx,dy,dz in itertools.product([-.25,-.125,0,.125,.25],[-.10,0,.10],[-.10,0,.10]):
        p=np.array(target); p[:3]+=[dx,dy,dz]
        # 从中立姿态沿线段逐步求解，保持同一逆运动学分支。
        q=neutral.copy()
        max_jump=0.
        for fraction in [.25,.5,.75,1.]:
            waypoint=np.array(target)+(p-np.array(target))*fraction
            nq,e=solve_ik(waypoint,q,offset,iterations=160,position_tolerance_m=1e-4,orientation_tolerance_rad=1e-3)
            max_jump=max(max_jump,float(np.max(np.abs(nq-q))))
            q=nq
        actual,_=forward_kinematics(q,offset)
        err=pose_error(actual,p)
        ok=bool(np.linalg.norm(err[:3])<.001 and np.linalg.norm(err[3:])<.01)
        samples.append({'offset_m':[dx,dy,dz],'ik_ok':ok,'position_error_m':float(np.linalg.norm(err[:3])),
                        'max_quarter_segment_joint_change_rad':max_jump,**metrics(q)})
    good=[s for s in samples if s['ik_ok']]
    result={'name':name,'tcp_pose':target,'joints_deg':np.degrees(neutral).tolist(),
            'neutral_metrics':metrics(neutral),'reachable_samples':len(good),'total_samples':len(samples),
            'worst_translation_joint_gain_rad_per_m':max(s['translation_joint_gain_rad_per_m'] for s in good) if good else None,
            'min_wrist_margin_deg':min(s['wrist_margin_deg'] for s in good) if good else None,'samples':samples}
    results.append(result)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
Path('docs/lab/2026-09-07-frame-workspace-comparison.json').write_text(json.dumps({
    'analysis_only':True,'model':'teacher UR5e DH; existing RG6 TCP offset 278.6 mm',
    'assumptions':['fixed TCP orientation per candidate','frame geometry and mounting transform not supplied',
                   'no collision, payload, torque, controller safety limit or dynamic validation',
                   'sparse samples and IK seed branch do not prove continuous workspace reachability'],
    'half_range_m':{'horizontal':.25,'depth':.10,'vertical':.10},'results':results},indent=2)+'\n')
