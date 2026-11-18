"""从明确的状态文件中生成一个不可执行的安装草案。这不会连接到机器人或移动机器人。
模拟器说明和明确的振幅只构成未经审查的候选点；它们不是物理相机校准、碰撞检查、安全路径或授权。
实际安装必须重新捕获状态快照；文件本身无法证明捕获时间或未改变的设备。所有审查都从假的开始。"""
import argparse
import ipaddress
import json
import math
from pathlib import Path


_AXES={'MOVE_LEFT':(0,-1),'MOVE_RIGHT':(0,1),'MOVE_UP':(2,1),'MOVE_DOWN':(2,-1),
       'MOVE_CLOSER':(1,-1),'MOVE_FARTHER':(1,1)}


def _six(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=6 or
            any(type(v) not in (float,int) or not math.isfinite(v) for v in value)):
        raise ValueError('A six-value finite status vector is required')
    return list(value)


def draft_from_snapshot(snapshot,host,horizontal,vertical,depth):
    try:host=str(ipaddress.ip_address(host))
    except ValueError:raise ValueError('Explicit robot IP required') from None
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
           for v in (horizontal,vertical,depth)):
        raise ValueError('Explicit finite positive horizontal, vertical and depth amplitudes required')
    amplitudes=(horizontal,depth,vertical)
    if not isinstance(snapshot,dict) or not isinstance(snapshot.get('rtde'),dict) or snapshot['rtde'].get('fresh') is not True:
        raise ValueError('Fresh RTDE status snapshot required')
    try:
        pose=_six(snapshot['rtde']['tcp_pose_m_rad']);joints=_six(snapshot['rtde']['joints_rad'])
        speed=_six(snapshot['rtde']['tcp_speed_m_s_rad_s']);tcp=_six(snapshot['primary']['tcp_offset_m_rad'])
    except (KeyError,TypeError):raise ValueError('Status snapshot incomplete') from None
    if math.hypot(*speed[:3])>=.0005 or math.hypot(*speed[3:])>=.002:
        raise ValueError('Status snapshot was not stationary')
    paths={}
    for cue,(axis,sign) in _AXES.items():
        target=list(pose[:3]);target[axis]+=sign*amplitudes[axis]
        if not math.isfinite(target[axis]):raise ValueError('Candidate target overflow')
        paths[cue]=[target]
    # 采样模拟器的历史*关节偏移*，并将它们添加到新的观测物理关节中。标称模拟IK不是校准的物理IK；
    # 这些是近似仅限审查的点，永远不是路线。
    from .simulator_reference_guidance import ReferenceMapper, curiosity_offset
    fixture_path=Path(__file__).resolve().parents[2]/'config'/'simulator-reference-2026-10-28.json'
    fixture=json.loads(fixture_path.read_text(encoding='utf-8-sig'))
    if fixture.get('scope')!='historical_simulator_reference_only':
        raise ValueError('Unexpected simulator reference fixture')
    tilted=_six(fixture['joints_rad']);tilted[4]-=.28
    retract=ReferenceMapper(tilted,_six(fixture['tcp_offset_m_rad']))._solve([0.,-.10,0.])
    # 在模拟器的100毫秒网格上选择14个代表性时间：倾斜、两个招手波（外/后）和返回。
    # 这些是离散预览候选项，不是模拟器连续运动的复制。
    sample_times=[300,700,1000,1500,2000,2300,2800,3100,3400,3900,4300,4700,5100,5600]
    waypoints=[[base+offset for base,offset in zip(joints,curiosity_offset(ms,retract))]
               for ms in sample_times]
    prior=[joints]+waypoints[:-1]
    intervals=[(t-prev)/1000. for prev,t in zip([0]+sample_times[:-1],sample_times)]
    average_peak=[max(abs(a-b) for a,b in zip(before,after))/dt
                  for before,after,dt in zip(prior,waypoints,intervals)]
    return {'status':'DRAFT','robot_ip':host,'reviewed_robot_ip':None,'reviewed_installation':{},
            'note':'Simulator-axis candidates only; stale snapshot and all physical paths must be reviewed. Never execute this draft.',
            'installation':{'reference':pose,'joints':joints,'tcp':tcp,
                'lower':None,'upper':None,'speed':None,'acceleration':None,
                'base_limit':None,'base_rate':None,
                'review':{k:False for k in ('reference','tcp','payload','path','dynamics','direction','stop')}},
            'paths':paths,'reviewed_paths':{},'hold_timeout':None,'action_timeout':None,
            'gate_timeout':None,'max_joint_speed':None,'reviewed_runtime':{},
            'intro_preview':{'status':'NOT_ACCEPTED','observed_indices':[],
                             'operator_decision':None,'reviewed':{}},
            'intro':{'status':'DRAFT','reference_joints':joints,'reference_pose':pose,'tcp':tcp,
                'waypoints':waypoints,'sample_times_ms':sample_times,
                'note':'Approximate simulator joint-offset samples, not its continuous motion or physical collision-checked path.',
                'candidate_average_peak_joint_speed_rad_s':average_peak,
                'candidate_speed_note':'Finite-difference interval averages only; not peak velocity, acceleration, or a safe robot limit.',
                'simulator_control_candidate':{
                    'scope':'SIMULATION_ONLY_NOT_PHYSICAL_LIMIT',
                    'source':'vision_mvp/src/leica_tracker/simulator_guidance.py',
                    'joint_velocity_limit_rad_s':.6,
                    'joint_acceleration_limit_rad_s2':1.2,
                    'note':'Simulator software clipping and speedj acceleration; not measured peaks, physical settings or safety approval.'},
                'joint_speed':None,'joint_acceleration':None,'max_tcp_angular_speed':None,
                'max_tcp_linear_speed':None,'timeout':None,'reviewed':{},
                'review':{k:False for k in ('robot','mount','tcp','payload','path','dynamics','stop','gesture')}}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True,help='local read-only physical_status JSON')
    parser.add_argument('--robot-ip',required=True)
    parser.add_argument('--horizontal-m',type=float,required=True,help='candidate horizontal displacement')
    parser.add_argument('--vertical-m',type=float,required=True,help='candidate vertical displacement')
    parser.add_argument('--depth-m',type=float,required=True,help='candidate depth displacement')
    args=parser.parse_args(argv)
    data=json.loads(args.snapshot.read_text(encoding='utf-8-sig'))
    print(json.dumps(draft_from_snapshot(data,args.robot_ip,args.horizontal_m,args.vertical_m,args.depth_m),indent=2,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
