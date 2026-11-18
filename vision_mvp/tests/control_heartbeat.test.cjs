const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

function heartbeatModule(){
  const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
  const block=html.match(/<script id="guest-control-heartbeat">([\s\S]*?)<\/script>/);
  assert.ok(block,'guest control heartbeat module is present');
  const context={module:{exports:{}},globalThis:{}};
  vm.runInNewContext(block[1],context);
  return context.module.exports;
}
const flush=async()=>{for(let i=0;i<5;i++)await Promise.resolve()};
function deferred(){let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no});return {promise,resolve,reject}}
function clock(){
  let now=0,next=1;const timers=new Map(),listeners=new Set();
  const document={hidden:false,addEventListener(name,fn){if(name==='visibilitychange')listeners.add(fn)}};
  const env={document,AbortController,performance:{now:()=>now},
    setTimeout(fn,ms){const id=next++;timers.set(id,{fn,at:now+ms});return id},
    clearTimeout(id){timers.delete(id)},
  };
  async function due(){
    for(;;){const first=[...timers].filter(([,t])=>t.at<=now).sort((a,b)=>a[1].at-b[1].at||a[0]-b[0])[0];
      if(!first)break;timers.delete(first[0]);first[1].fn();await flush()}
  }
  return {env,timers,now:()=>now,
    async advance(ms){const target=now+ms;for(;;){const first=[...timers.values()].filter(t=>t.at<=target).sort((a,b)=>a.at-b.at)[0];if(!first)break;now=first.at;await due()}now=target;await flush()},
    async stall(ms){now+=ms;await due()},
    hidden(value){document.hidden=value;for(const fn of listeners)fn()},
  };
}

test('an immediate heartbeat and fixed cadence continue while an older response is hung',async()=>{
  const c=clock(),first=deferred(),sent=[];
  const heart=heartbeatModule().create(c.env,(body,signal)=>{sent.push({body:{...body},signal,at:c.now()});return sent.length===1?first.promise:Promise.resolve({ok:true,accepted:true})});
  await c.advance(500);
  assert.equal(sent.length,0,'creation alone does not arm control');
  assert.equal(heart.start({instance:'server-a',epoch:4}),true);
  assert.deepEqual(sent[0].body,{instance:'server-a',epoch:4,sequence:1});
  await c.advance(150);
  assert.deepEqual(sent.map(x=>x.at),[500,550,600,650]);
  assert.deepEqual(sent.map(x=>x.body.sequence),[1,2,3,4]);
  assert.equal(sent[0].signal.aborted,true,'a hung request is aborted at 150 ms');
  heart.stop();first.resolve({ok:true,accepted:true});await flush();
});

test('hanging transports use at most three slots and abort each request without queuing more',async()=>{
  const c=clock(),sent=[];
  const heart=heartbeatModule().create(c.env,(body,signal)=>{const reply=deferred();sent.push({body,signal,reply,at:c.now()});return reply.promise});
  heart.start({instance:'server-a',epoch:1});
  await c.advance(500);
  assert.deepEqual(sent.map(x=>x.at),[0,50,100]);
  assert.ok(sent.every(x=>x.signal.aborted));
  assert.equal(c.timers.size,1,'only the independent next cadence timer remains');
  sent[0].reply.resolve({ok:true,accepted:true});await flush();
  await c.advance(50);
  assert.equal(sent.length,4,'a released slot permits the next scheduled pulse');
  assert.equal(sent[3].at,550);
  heart.stop();assert.equal(c.timers.size,0);
});

test('a scheduler stall over the unchanged 300 ms lease fails closed without replay or automatic resume',async()=>{
  const c=clock(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,(body,signal)=>{sent.push({body,signal});return new Promise(()=>{})},reason=>expired.push(reason));
  heart.start({instance:'server-a',epoch:1});
  await c.stall(301);
  assert.equal(sent.length,1);
  assert.equal(sent[0].signal.aborted,true);
  assert.deepEqual(expired,['scheduler_stalled']);
  assert.equal(c.timers.size,0);
  await c.advance(1000);assert.equal(sent.length,1);
  heart.start({instance:'server-a',epoch:2});
  assert.equal(sent.length,2,'only a new explicit start rearms the scheduler');
  assert.equal(sent[1].body.epoch,2);
  heart.stop();
});

test('a delayed but still valid scheduler pulse sends once without catching up missed intervals',async()=>{
  const c=clock(),sent=[];
  const heart=heartbeatModule().create(c.env,body=>{sent.push({...body});return Promise.resolve({ok:true,accepted:true})});
  heart.start({instance:'server-a',epoch:1});await flush();
  await c.stall(300);
  assert.equal(sent.length,2);
  await c.advance(49);assert.equal(sent.length,2);
  await c.advance(1);assert.equal(sent.length,3);
  heart.stop();
});

test('hiding stops and aborts immediately and becoming visible requires another explicit start',async()=>{
  const c=clock(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,(body,signal)=>{sent.push({body,signal});return new Promise(()=>{})},reason=>expired.push(reason));
  heart.start({instance:'server-a',epoch:1});
  c.hidden(true);
  assert.equal(sent[0].signal.aborted,true);
  assert.deepEqual(expired,['page_hidden']);
  assert.equal(heart.start({instance:'server-a',epoch:2}),false);
  c.hidden(false);await c.advance(500);
  assert.equal(sent.length,1);
  assert.equal(heart.start({instance:'server-a',epoch:3}),true);
  assert.equal(sent.length,2);
  heart.stop();
});

test('control sends only with an explicit foreground visibility state',()=>{
  const c=clock(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,body=>{sent.push(body);return new Promise(()=>{})},reason=>expired.push(reason));
  c.hidden(undefined);
  assert.equal(heart.start({instance:'server-a',epoch:1}),false);
  assert.equal(sent.length,0);
  c.hidden(false);heart.start({instance:'server-a',epoch:2});
  c.hidden(undefined);
  assert.deepEqual(expired,['page_hidden']);
  assert.equal(c.timers.size,0);
});

test('ordinary failures and ignored out-of-order replies do not cancel healthy later pulses',async()=>{
  const c=clock(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,body=>{sent.push({...body});return sent.length===1?Promise.reject(Error('temporary transport failure')):Promise.resolve({ok:true,accepted:sent.length!==2})},reason=>expired.push(reason));
  heart.start({instance:'server-a',epoch:1});
  await c.advance(200);
  assert.equal(sent.length,5);
  assert.deepEqual(expired,[]);
  heart.stop();
});

test('a timed-out late refusal cannot cancel a newer healthy heartbeat',async()=>{
  const c=clock(),old=deferred(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,(body,signal)=>{sent.push({body,signal});return sent.length===1?old.promise:Promise.resolve({ok:true,accepted:true})},reason=>expired.push(reason));
  heart.start({instance:'server-a',epoch:1});await c.advance(150);
  assert.equal(sent[0].signal.aborted,true);
  old.resolve({ok:false,fatal:true});await flush();await c.advance(50);
  assert.equal(sent.length,5);
  assert.deepEqual(expired,[]);
  heart.stop();
});

test('fatal server refusal latches stopped while an old generation cannot expire a new one',async()=>{
  const c=clock(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,(body,signal)=>{const reply=deferred();sent.push({body:{...body},signal,reply});return reply.promise},reason=>expired.push(reason));
  heart.start({instance:'server-a',epoch:1});
  heart.stop();assert.equal(sent[0].signal.aborted,true);
  heart.start({instance:'server-b',epoch:2});
  sent[0].reply.resolve({ok:false,fatal:true});await flush();
  assert.deepEqual(expired,[]);
  await c.advance(50);
  assert.deepEqual(sent.map(x=>x.body.epoch),[1,2,2]);
  sent[1].reply.resolve({ok:false,fatal:true});await flush();
  assert.deepEqual(expired,['server_rejected']);
  assert.equal(sent[2].signal.aborted,true,'the other pending request is aborted');
  await c.advance(500);assert.equal(sent.length,3);
});

test('fatal rejected promises stop control but late failures after stop are ignored',async()=>{
  const c=clock(),reply=deferred(),expired=[];
  const heart=heartbeatModule().create(c.env,()=>reply.promise,reason=>expired.push(reason));
  heart.start({instance:'server-a',epoch:1});reply.reject(Object.assign(Error('revoked'),{fatal:true}));await flush();
  assert.deepEqual(expired,['server_rejected']);
  assert.equal(c.timers.size,0);
  const late=deferred();const other=heartbeatModule().create(c.env,()=>late.promise,reason=>expired.push(reason));
  other.start({instance:'server-a',epoch:2});other.stop();late.reject(Object.assign(Error('old'),{fatal:true}));await flush();
  assert.deepEqual(expired,['server_rejected']);
});

test('sequence rollover fails closed and invalid identities cannot start sending',async()=>{
  const c=clock(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,body=>{sent.push({...body});return Promise.resolve({ok:true,accepted:true})},reason=>expired.push(reason));
  for(const identity of [null,{instance:'',epoch:1},{instance:'server-a',epoch:-1},{instance:'server-a',epoch:1.5},{instance:'server-a',epoch:1,sequence:-1},{instance:'server-a',epoch:1,sequence:2147483648}]){
    assert.equal(heart.start(identity),false);
  }
  assert.equal(sent.length,0);
  heart.start({instance:'server-a',epoch:1,sequence:2147483646});
  assert.equal(sent[0].sequence,2147483647);
  await c.advance(50);
  assert.equal(sent.length,1);
  assert.deepEqual(expired,['sequence_exhausted']);
  assert.equal(c.timers.size,0);
});

test('a backwards clock fails closed before sending another packet',async()=>{
  const c=clock(),sent=[],expired=[];
  const heart=heartbeatModule().create(c.env,body=>{sent.push({...body});return Promise.resolve({ok:true,accepted:true})},reason=>expired.push(reason));
  heart.start({instance:'server-a',epoch:1});await flush();
  let now=-1;c.env.performance.now=()=>now;
  await c.advance(50);
  assert.equal(sent.length,1);
  assert.deepEqual(expired,['clock_invalid']);
  assert.equal(c.timers.size,0);
});

test('diagnostics distinguish delayed replies from a stalled foreground scheduler',async()=>{
  const c=clock(),late=deferred();let sent=0;
  const heart=heartbeatModule().create(c.env,()=>++sent===1?late.promise:Promise.resolve({ok:true}));
  heart.start({instance:'private-server-identity',epoch:7});
  await c.advance(200);late.resolve({ok:true});await flush();
  const delayed=heart.diagnostics();
  assert.equal(delayed.control_hb_expiry,0);
  assert.equal(delayed.control_hb_max_gap_ms,50);
  assert.equal(delayed.control_hb_max_rtt_ms,200);
  assert.equal(delayed.control_hb_aborted,1);
  assert.equal(delayed.control_hb_sent,5);
  assert.equal(delayed.control_hb_settled,5);
  heart.stop();
  const stalledClock=clock();
  const stalled=heartbeatModule().create(stalledClock.env,()=>new Promise(()=>{}));
  stalled.start({instance:'private-server-identity',epoch:7});
  await stalledClock.stall(301);
  const diagnosis=stalled.diagnostics();
  assert.equal(diagnosis.control_hb_expiry,4);
  assert.equal(diagnosis.control_hb_max_gap_ms,301);
  assert.equal(diagnosis.control_hb_max_rtt_ms,0);
  assert.equal(diagnosis.control_hb_sent,1);
  assert.equal(diagnosis.control_hb_aborted,1);
});

test('first expiry persists after cleanup and another explicitly started encounter',async()=>{
  const c=clock();let fatal=false;
  const heart=heartbeatModule().create(c.env,()=>Promise.resolve(fatal?{ok:false,fatal:true}:{ok:true}));
  heart.start({instance:'server-a',epoch:1});await flush();
  await c.stall(301);
  assert.equal(heart.diagnostics().control_hb_expiry,4);
  heart.stop();fatal=true;
  heart.start({instance:'server-a',epoch:2});await flush();
  assert.equal(heart.diagnostics().control_hb_expiry,4,'later server refusal must preserve the first recorded cause');
});

test('diagnostic snapshots are immutable bounded numbers with no control identities',async()=>{
  const c=clock(),late=deferred();let sends=0;
  const heart=heartbeatModule().create(c.env,()=>++sends===1?late.promise:Promise.resolve({ok:true}));
  heart.start({instance:'private-server-identity',epoch:898989});
  await c.advance(15000);await c.stall(20000);late.resolve({ok:true});await flush();
  const snapshot=heart.diagnostics();
  assert.equal(snapshot.control_hb_version,1);
  assert.equal(snapshot.control_hb_sent,240);
  assert.equal(snapshot.control_hb_settled,240);
  assert.equal(snapshot.control_hb_max_gap_ms,10000);
  assert.equal(snapshot.control_hb_max_rtt_ms,10000);
  assert.ok(snapshot.control_hb_pending<=3&&snapshot.control_hb_max_pending<=3);
  assert.ok(Object.values(snapshot).every(value=>Number.isFinite(value)&&value>=0&&value<=10000));
  assert.equal(Object.isFrozen(snapshot),true);
  snapshot.control_hb_sent=0;
  assert.equal(heart.diagnostics().control_hb_sent,240);
  assert.notEqual(snapshot,heart.diagnostics());
  assert.ok(!JSON.stringify(snapshot).includes('private-server-identity'));
  assert.ok(!JSON.stringify(snapshot).includes('898989'));
});

test('camera upload diagnostics include the independent heartbeat diagnostics',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
  assert.ok(html.includes('...controlHeartbeat.diagnostics()'));
});
