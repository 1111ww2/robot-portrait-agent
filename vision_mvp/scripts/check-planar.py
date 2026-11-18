import json,subprocess,time
from pathlib import Path
from leica_tracker.simulator_follow import synthetic_event
from urx_browser_simulator.student_client import SimulatorClient
root=Path.cwd();runtime=root/'tmp/python-lab-review'
client=SimulatorClient('http://127.0.0.1:8765')
assert client.state()['urx']['clients']==0
client.reset()
with (runtime/'planar-check.jsonl').open('w') as out,(runtime/'planar-check.stderr').open('w') as err:
    p=subprocess.Popen([str(runtime/'.venv/Scripts/python.exe'),'-u','-m','leica_tracker.simulator_fixed_base','--vertical','--duration','17'],stdin=subprocess.PIPE,stdout=out,stderr=err,text=True)
    start=time.monotonic()
    try:
        while p.poll() is None:
            now=time.monotonic();elapsed=now-start
            right,down=(0,-1) if elapsed<3.5 else ((1,-1) if elapsed<7 else ((-1,1) if elapsed<12 else (0,0)))
            event=synthetic_event(now,1.)
            event['command'].update(right=right,down=down)
            if 14<elapsed<15:event['tracking']=False;event['motion_intent_m']['hold']=True
            p.stdin.write(json.dumps(event)+'\n');p.stdin.flush();time.sleep(.05)
    except BrokenPipeError:pass
    finally:p.stdin.close();code=p.wait(timeout=5)
samples=[json.loads(line) for line in (runtime/'planar-check.jsonl').read_text().splitlines() if json.loads(line)['event']=='simulator_sync']
result={'simulated':True,'exit_code':code,'samples':len(samples),'x_range_m':[min(s['tcp_pose'][0] for s in samples),max(s['tcp_pose'][0] for s in samples)],'z_range_m':[min(s['tcp_pose'][2] for s in samples),max(s['tcp_pose'][2] for s in samples)],'max_base_rad':max(abs(s['joints'][0]) for s in samples),'max_depth_error_m':max(abs(s['tcp_pose'][1]+.1333) for s in samples),'errors':(runtime/'planar-check.stderr').read_text()[-1000:]}
print(json.dumps(result))
Path('docs/lab/2026-09-08-planar-simulator-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
assert code==0
assert result['z_range_m'][0]>=.047 and result['z_range_m'][1]<=.46
client.reset()
