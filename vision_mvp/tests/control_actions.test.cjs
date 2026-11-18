const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
function fixture(request){
 const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
 const body=html.match(/async function action\(name,enter=false,stopOrigin=1\)\{[\s\S]*?\n\}/)[0];
 const starts=[],calls=[],configurations=[];
 const context={request:(route,body)=>route==='/state'?Promise.resolve({control:{instance:'session',epoch:1}}):request(route,body),document:{hidden:false},controlHeartbeat:{stop:()=>calls.push('stop'),start:x=>starts.push(x),configure:(value,control)=>configurations.push({value,control})},
  photoFlow:{reset(){}},recorder:{cancel(){}},voice:{cancel(){}},cueLight:{clear(){}},
  copy(){},offline:()=>calls.push('offline'),$:()=>({}),
  lastState:{control:{instance:'session',epoch:1}},stateEpoch:0,controlActionEpoch:0,actionPending:false,
  visitorStopped:false,starting:false};
 vm.createContext(context);vm.runInContext(body,context);
 return {context,starts,calls,configurations,run:context.action};
}
test('Start uses observed epoch and starts independent heartbeat only after acknowledgement',async()=>{
 let payload;const communication={status:'healthy',recoverable:true,remaining_ms:0};
 const f=fixture(async(path,body)=>{payload=body;return {control:{instance:'session',epoch:2},communication}});
 await f.run('start');
 assert.equal(payload.control_protocol,1);assert.equal(payload.control.epoch,1);
 assert.equal(payload.recovery_window_s,3.0);
 assert.equal(f.starts[0].epoch,2);
 assert.equal(f.configurations[0].value,communication);
 assert.equal(f.configurations[0].control.epoch,2);
});
test('Stop cancels heartbeat immediately and late Start response cannot restart it',async()=>{
 let first;const f=fixture((path,body)=>body.action==='start'?new Promise(r=>first=r):Promise.resolve({control:{instance:'session',epoch:3}}));
 const start=f.run('start');for(let i=0;i<10&&!first;i++)await Promise.resolve();assert.equal(typeof first,'function');await f.run('stop');first({control:{instance:'session',epoch:2}});await start;
 assert.equal(f.starts.length,0);assert.equal(f.context.lastState.active,false);
 assert.equal(f.context.lastState.control.epoch,3);assert.equal(f.context.actionPending,false);
});
test('hidden page cannot start movement or heartbeat',async()=>{
 let requests=0;const f=fixture(async()=>{requests++;return {control:{instance:'session',epoch:2}}});
 f.context.document.hidden=true;await f.run('start');
 assert.equal(requests,0);assert.equal(f.starts.length,0);
});

test('Stop while fresh identity is loading prevents sending the old Start',async()=>{
 const f=fixture(()=>{});let resolveState,starts=0;
 f.context.request=(route,body)=>route==='/state'?new Promise(r=>resolveState=r):
  (body.action==='stop'?Promise.resolve({control:{instance:'session',epoch:2}}):(starts++,Promise.resolve({})));
 const pending=f.run('start');await f.run('stop');resolveState({control:{instance:'session',epoch:1}});await pending;
 assert.equal(starts,0);assert.equal(f.starts.length,0);
});

test('Stop preserves its fixed numeric origin and observed epoch without delaying cancellation',async()=>{
 let payload;const f=fixture(async(route,body)=>{payload=body;return {control:{instance:'session',epoch:2}}});
 await f.run('stop',false,5);assert.equal(payload.stop_origin,5);assert.equal(payload.control.epoch,1);assert.equal(f.calls[0],'stop');
 assert.equal(Object.hasOwn(payload,'recovery_window_s'),false);
 await f.run('stop');assert.equal(payload.stop_origin,1);
 await f.run('stop',false,'free text');assert.equal(payload.stop_origin,0);
});

test('a new control acknowledgement cannot retain the previous native motion or photo signal',async()=>{
 const f=fixture(async()=>({control:{instance:'session',epoch:2}}));
 Object.assign(f.context.lastState,{motion:{active:true,fresh:true,action_id:7},feedback:{edge:'right',intensity:1,gesture_id:7},photo_ready:true,photo_latched:true});
 await f.run('restart');
 assert.equal(f.context.lastState.motion.active,false);assert.equal(f.context.lastState.feedback,null);
 assert.equal(f.context.lastState.photo_ready,false);assert.equal(f.context.lastState.photo_latched,false);
});

test('Start while the owned encounter is active preserves heartbeat and photo state',async()=>{
 const f=fixture(async()=>assert.fail('duplicate Start must not be sent'));
 Object.assign(f.context.lastState,{active:true,owner:true,photo_ready:true,photo_latched:true});
 await f.run('start');
 assert.equal(f.calls.length,0);assert.equal(f.starts.length,0);
 assert.equal(f.context.controlActionEpoch,0);
 assert.equal(f.context.lastState.photo_ready,true);assert.equal(f.context.lastState.photo_latched,true);
});

test('an idempotent server Start reply continues above its accepted heartbeat sequence',async()=>{
 const f=fixture(async()=>({control:{instance:'session',epoch:1},already_active:true,control_sequence:42}));
 await f.run('start');
 assert.equal(f.starts[0].sequence,42);
});
