"""把本地教学仿真器设置到已保存的历史关节参考位，不连接硬件，也不改变 TCP 或虚拟夹爪。
请使用教学仿真器自己的 Python 环境。"""
import argparse,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'vision_mvp/config/simulator-reference-2026-10-28.json'
SIMULATOR_URL='http://127.0.0.1:8766'

def vector(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=6 or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in value)):
        raise ValueError('Expected six finite numeric values')
    return [float(x) for x in value]

def validate_reference(ref):
    if not isinstance(ref,dict) or ref.get('scope')!='historical_simulator_reference_only':
        raise ValueError('Reference must be explicitly marked simulation-only')
    joints=vector(ref.get('joints_rad'));tcp=vector(ref.get('tcp_offset_m_rad'));pose=vector(ref.get('measured_tcp_m_rad'))
    if any(abs(x)>2*math.pi for x in joints):raise ValueError('Invalid joint reference')
    return joints,tcp,pose

def orientation_error(a,b):
    def quat(r):
        angle=math.hypot(*r)
        if angle<1e-12:return (1.,0.,0.,0.)
        scale=math.sin(angle/2)/angle
        return (math.cos(angle/2),*(scale*x for x in r))
    dot=abs(sum(x*y for x,y in zip(quat(a),quat(b))))
    return 2*math.acos(min(1.,max(0.,dot)))

def set_reference(client,ref):
    joints,tcp,measured=validate_reference(ref)
    health=client.health();state=client.state()
    if (health.get('service')!='urx-browser-simulator' or health.get('backend')!='sim' or
        not health.get('instance_id') or state.get('service')!='urx-browser-simulator' or
        state.get('backend')!='sim' or state.get('instance_id')!=health['instance_id']):
        raise RuntimeError('Verified local simulator identity is required')
    if (state.get('urx',{}).get('clients')!=0 or state.get('urx',{}).get('program_running') is not False
            or state.get('urx',{}).get('control_owner')!='idle'
            or max(abs(v) for v in vector(state.get('joint_velocity')))>.001):
        raise RuntimeError('Simulator is busy; stop its current controller first')
    if math.dist(vector(state.get('active_tcp_offset')),tcp)>1e-6:
        raise RuntimeError('Simulator TCP differs from the recorded reference; no tool change sent')
    client.move_joints(joints,acceleration=.5,velocity=.35,wait=True,timeout=45.)
    state=client.state()
    if state.get('instance_id')!=health['instance_id'] or state.get('backend')!='sim':
        raise RuntimeError('Simulator identity changed while moving')
    if math.dist(vector(state.get('active_tcp_offset')),tcp)>1e-6:
        raise RuntimeError('Simulator TCP changed while setting the reference')
    actual=vector(state.get('joints'));pose=vector(state.get('tcp_pose'))
    error=max(abs(a-b) for a,b in zip(actual,joints))
    if error>.0011:raise RuntimeError('Simulator did not reach the reference joints')
    return {'simulated':True,'source':'historical_read_only_reference','max_joint_error_rad':error,
            'tcp_position_difference_mm':1000*math.dist(pose[:3],measured[:3]),
            'tcp_orientation_difference_rad':orientation_error(pose[3:],measured[3:]),
            'simulated_tcp_pose_m_rad':pose,'tcp_command_sent':False,'gripper_command_sent':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=DEFAULT)
    parser.add_argument('--execute',action='store_true',help='Set pose in the local simulator')
    args=parser.parse_args();path=args.reference.resolve()
    if not path.is_relative_to(ROOT):parser.error('Reference must be inside this workspace')
    ref=json.loads(path.read_text(encoding='utf-8'));joints,tcp,_=validate_reference(ref)
    if not args.execute:
        print(json.dumps({'mode':'offline_preview','simulated':True,'joints_rad':joints,'tcp_offset_m_rad':tcp}));return
    from urx_browser_simulator.student_client import SimulatorClient
    print(json.dumps(set_reference(SimulatorClient(SIMULATOR_URL),ref)),flush=True)

if __name__=='__main__':main()
