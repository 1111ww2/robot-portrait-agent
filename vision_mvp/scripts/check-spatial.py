import json,subprocess,time
from pathlib import Path
from leica_tracker.simulator_follow import synthetic_event
from urx_browser_simulator.student_client import SimulatorClient
runtime=Path.cwd()/'tmp/python-lab-review';client=SimulatorClient('http://127.0.0.1:8765')
assert client.state()['urx']['clients']==0
client.reset()
with (runtime/'spatial-check.jsonl').open('w') as out,(runtime/'spatial-check.stderr').open('w') as err:
    p=subprocess.Popen([str(runtime/'.venv/Scripts/python.exe'),'-u','-m','leica_tracker.simulator_fixed_base','--depth','--duration','20'],stdin=subprocess.PIPE,stdout=out,stderr=err,text=True)
    start=time.monotonic()
    try:
        while p.poll() is None:
            now=time.monotonic();t=now-start
            width=.3 if t<3 else (.4 if t<7 else (.2 if t<12 else (.4 if t<17 else .3)))
            right,down=(0,0) if t<12 else ((1,-1) if t<17 else (0,0))
            event=synthetic_event(now,1.);event['command'].update(right=right,down=down)
            event['image_observation']=dict(shoulder_width=width,confidence=.9)
            if 18<t<19:event['tracking']=False;event['motion_intent_m']['hold']=True
            p.stdin.write(json.dumps(event)+'\n');p.stdin.flush();time.sleep(.05)
    except BrokenPipeError:pass
    finally:p.stdin.close();code=p.wait(timeout=5)
rows=[json.loads(line) for line in (runtime/'spatial-check.jsonl').read_text().splitlines() if json.loads(line)['event']=='simulator_sync']
result=dict(simulated=True,exit_code=code,samples=len(rows),xyz_ranges_m=[[min(r['tcp_pose'][i] for r in rows),max(r['tcp_pose'][i] for r in rows)] for i in range(3)],base_max_deg=max(abs(r['joints'][0])*180/3.141592653589793 for r in rows),calibrated=any(r['depth_calibrated'] for r in rows),errors=(runtime/'spatial-check.stderr').read_text()[-1200:])
print(json.dumps(result));Path('docs/lab/2026-09-08-spatial-simulator-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
assert code==0 and result['calibrated']
assert result['xyz_ranges_m'][1][0]<-.14 and result['xyz_ranges_m'][1][1]>-.125
client.reset()
