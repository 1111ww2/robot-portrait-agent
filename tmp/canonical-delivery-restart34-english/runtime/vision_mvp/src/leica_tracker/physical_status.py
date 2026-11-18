"""用于现场操作员的只读UR10e状态快照。此工具接收主和RTDE遥测数据，并仅发送仪表板查询。
它从不发送机器人程序、目标、重置、工具I/O、夹具或电源命令。必须在现场明确调用；旧样本不是当前状态。"""
import argparse
import ipaddress
import json
import math
import socket
import time

from .robot_state import StateReceiver
from .ur_rtde_transport import RTDEChannel


_QUERIES = ('safetystatus', 'robotmode', 'running', 'is in remote control')


def _vector(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 6 or
            any(type(v) not in (float, int) or not math.isfinite(v) for v in value)):
        return None
    return list(value)


def _fresh(sample, stamp, now):
    return (isinstance(sample, dict) and type(stamp) in (int, float)
            and math.isfinite(stamp) and 0 <= now-stamp <= .5)


def _close(source):
    if source is not None:
        try: source.close()
        except (OSError, RuntimeError, ValueError): pass


def _label(reply, prefix):
    if not reply.startswith(prefix): return None
    value = reply[len(prefix):].strip()
    return value if value and len(value) <= 64 and value.replace('_', '').isalnum() else None


def read_dashboard(host, *, connect=socket.create_connection, clock=time.monotonic):
    """仅记录了仪表板状态查询；未知响应为空。"""
    with connect((host, 29999), timeout=.7) as sock:
        # 一个固定的截止日期限制了即使是发送字节速度非常慢的同行。
        deadline = clock()+3.5
        pending = bytearray()
        def readline():
            while b'\n' not in pending:
                remaining = deadline-clock()
                if remaining <= 0 or len(pending) >= 1024:
                    raise TimeoutError('Dashboard response expired')
                sock.settimeout(min(.7,remaining))
                chunk = sock.recv(1024-len(pending))
                if not chunk: raise OSError('Dashboard disconnected')
                pending.extend(chunk)
            end = pending.index(b'\n')+1
            line = bytes(pending[:end]); del pending[:end]
            return line.decode('ascii', errors='replace').strip()
        if not readline(): raise OSError('Dashboard greeting unavailable')
        replies = []
        for query in _QUERIES:
            sock.sendall((query+'\n').encode('ascii'))
            replies.append(readline())
    safety = _label(replies[0], 'Safetystatus:')
    mode = _label(replies[1], 'Robotmode:')
    running = replies[2].removeprefix('Program running:').strip() if replies[2].startswith('Program running:') else ''
    remote = replies[3].lower()
    boolean = lambda text: {'true': True, 'false': False}.get(text.lower())
    return {'safety_status': safety, 'robot_mode': mode,
            'program_running': boolean(running), 'remote_control': boolean(remote)}


def inspect_robot(ip, *, state_factory=StateReceiver, rtde_factory=RTDEChannel,
                  dashboard_reader=read_dashboard, clock=time.monotonic,
                  sleep=time.sleep, primary_wait=2.):
    """收集独立的最新只读快照；缺失的数据保持为空。"""
    try: ip = str(ipaddress.ip_address(ip))
    except ValueError: raise ValueError('Provide an explicit robot IP address') from None
    started = clock()
    out = {'connected': False,
           'note': 'Channels are sampled independently, not simultaneously. connected only means a channel responded during this check; connected and fresh do not establish motion readiness or safety. A single fresh sample cannot rule out queued data. RTDE unavailable can mean missing dependencies or unavailable registers, not necessarily a disconnected robot. Checks may take more than two seconds. Dashboard age is measured from the start of its query.',
           'elapsed_seconds': None,
           'primary': {'available': False, 'fresh': False, 'age_seconds': None, 'mode': None, 'tcp_pose_m_rad': None,
                       'tcp_offset_m_rad': None},
           'rtde': {'available': False, 'fresh': False, 'age_seconds': None, 'tcp_pose_m_rad': None,
                    'tcp_speed_m_s_rad_s': None, 'joints_rad': None},
           'dashboard': {'available': False, 'fresh': False, 'age_seconds': None, 'safety_status': None, 'robot_mode': None,
                         'program_running': None, 'remote_control': None}}
    sample_times = {}
    source = None
    try:
        source = state_factory(ip)
        deadline = clock()+primary_wait
        while source.latest[0] is None and not source.failure and clock()<deadline:
            sleep(.02)
        sample, stamp = source.latest
        if not source.failure and _fresh(sample, stamp, clock()):
            raw_mode = sample.get('mode')
            if isinstance(raw_mode, dict):
                fields = ('connected','enabled','power','emergency','protective','running','paused','mode','control')
                mode = {k: raw_mode.get(k) if type(raw_mode.get(k)) in (bool,int) else None for k in fields}
            else: mode = None
            pose, tcp = _vector(sample.get('pose')), _vector(sample.get('tcp_offset'))
            if mode is not None or pose is not None or tcp is not None:
                out['primary'] = {'available': True, 'mode': mode, 'tcp_pose_m_rad': pose,
                                  'tcp_offset_m_rad': tcp}
                sample_times['primary'] = stamp
    except (OSError, RuntimeError, ValueError, TimeoutError): pass
    finally:
        _close(source)

    source = None
    try:
        source = rtde_factory(ip, startup_timeout=3., spatial=True, receive_only=True)
        sample, stamp = source.latest
        if not source.failure and _fresh(sample, stamp, clock()):
            pose, speed, joints = (_vector(sample.get(k)) for k in ('tcp_pose','tcp_speed','joints'))
            if any(v is not None for v in (pose,speed,joints)):
                out['rtde'] = {'available': True, 'tcp_pose_m_rad': pose,
                               'tcp_speed_m_s_rad_s': speed, 'joints_rad': joints}
                sample_times['rtde'] = stamp
    except (OSError, RuntimeError, ValueError, TimeoutError, ImportError): pass
    finally:
        _close(source)

    try:
        dashboard_started = clock()
        result = dashboard_reader(ip)
        if isinstance(result,dict):
            for k in ('safety_status','robot_mode'):
                if isinstance(result.get(k),str): out['dashboard'][k] = result[k]
            for k in ('program_running','remote_control'):
                if type(result.get(k)) is bool: out['dashboard'][k] = result[k]
            out['dashboard']['available'] = any(out['dashboard'][k] is not None for k in
                ('safety_status','robot_mode','program_running','remote_control'))
            if out['dashboard']['available']: sample_times['dashboard'] = dashboard_started
    except (OSError, RuntimeError, ValueError, TimeoutError): pass
    now = clock()
    out['elapsed_seconds'] = round(now-started, 3) if now >= started else None
    for name in ('primary','rtde','dashboard'):
        stamp = sample_times.get(name)
        age = now-stamp if stamp is not None else None
        if age is not None and math.isfinite(age) and age >= 0:
            out[name]['age_seconds'] = round(age, 3)
            out[name]['fresh'] = age <= .5
        else:
            out[name]['age_seconds'] = None
            out[name]['fresh'] = False
    out['connected'] = any(out[k]['available'] for k in ('primary','rtde','dashboard'))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', required=True, help='Current robot IP from the pendant')
    args = parser.parse_args(argv)
    try: result = inspect_robot(args.ip)
    except ValueError as e: parser.error(str(e))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['connected'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
