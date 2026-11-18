const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
function moduleFor(id){const context={module:{exports:{}},globalThis:{}};const block=html.match(new RegExp('<script id="'+id+'">([\\s\\S]*?)<\\/script>'));vm.runInNewContext(block[1],context);return context.module.exports}
const flush=async()=>{for(let i=0;i<5;i++)await Promise.resolve()};
function clock(){let now=0,next=0;const timers=new Map(),listeners=[];
 const env={AbortController,performance:{now:()=>now},document:{hidden:false,addEventListener(name,fn){if(name==='visibilitychange')listeners.push(fn)}},
  setTimeout(fn,ms){const id=++next;timers.set(id,{fn,at:now+ms});return id},clearTimeout(id){timers.delete(id)}};
 async function due(){for(;;){const item=[...timers].filter(([,v])=>v.at<=now).sort((a,b)=>a[1].at-b[1].at||a[0]-b[0])[0];if(!item)break;timers.delete(item[0]);item[1].fn();await flush()}}
 return {env,now:()=>now,async stall(ms){now+=ms;await due()},async advance(ms){const end=now+ms;for(;;){const first=[...timers.values()].filter(v=>v.at<=end).sort((a,b)=>a.at-b.at)[0];if(!first)break;now=first.at;await due()}now=end;await flush()},hidden(){env.document.hidden=true;listeners.forEach(fn=>fn())}};
}
const identity={instance:'same-server',epoch:4};
const healthy={status:'healthy',recoverable:true,remaining_ms:0};

test('negotiated foreground stall sends one fresh pulse in the same encounter without Start',async()=>{
 const c=clock(),sent=[],expired=[];
 const heart=moduleFor('guest-control-heartbeat').create(c.env,body=>{sent.push({...body});return Promise.resolve({ok:true,accepted:true,communication:healthy})},reason=>expired.push(reason));
 heart.start(identity);await flush();heart.configure(healthy,identity);
 await c.stall(700);assert.equal(sent.length,2);assert.equal(sent[1].epoch,4);assert.equal(sent[1].sequence,2);
 assert.deepEqual(expired,[]);await c.advance(50);assert.equal(sent.length,3);heart.stop();
});

test('negotiated lost replies have one fixed bounded deadline and cannot retry forever',async()=>{
 const c=clock(),sent=[],expired=[];
 const heart=moduleFor('guest-control-heartbeat').create(c.env,body=>{sent.push({...body});return Promise.reject(Error('WiFi interrupted'))},reason=>expired.push(reason));
 heart.start(identity);heart.configure(healthy,identity);await flush();await c.advance(3350);
 assert.deepEqual(expired,['recovery_expired']);const count=sent.length;await c.advance(1000);assert.equal(sent.length,count);
});

test('waiting replies do not extend the first local recovery deadline',async()=>{
 const c=clock(),expired=[];
 const heart=moduleFor('guest-control-heartbeat').create(c.env,()=>Promise.resolve({ok:true,accepted:true,communication:{status:'waiting',recoverable:true,remaining_ms:3000}}),reason=>expired.push(reason));
 heart.start(identity);heart.configure(healthy,identity);await c.advance(3050);
 assert.deepEqual(expired,['recovery_expired']);
});

test('real Stop, hidden page and fatal refusal remain terminal during recoverable waiting',async()=>{
 for(const terminal of ['stop','hidden','refusal']){
  const c=clock(),sent=[],expired=[];let refusal=false;
  const heart=moduleFor('guest-control-heartbeat').create(c.env,body=>{sent.push(body);return Promise.resolve(refusal?{ok:false,fatal:true}:{ok:true,accepted:true,communication:healthy})},reason=>expired.push(reason));
  heart.start(identity);await flush();heart.configure(healthy,identity);await c.stall(500);
  if(terminal==='stop')heart.stop();else if(terminal==='hidden')c.hidden();else{refusal=true;await c.advance(50)}
  const count=sent.length;await c.advance(4000);assert.equal(sent.length,count);
  assert.deepEqual(expired,terminal==='hidden'?['page_hidden']:terminal==='refusal'?['server_rejected']:[]);
 }
});

test('held state keeps lights dark and tells the visitor it will reconnect',()=>{
 const api=moduleFor('guest-experience');
 const state={active:true,owner:true,control_fresh:true,tracking:true,stale:false,runtime:{status:'running',reason:'none'},
  communication:{status:'waiting',recoverable:true,remaining_ms:1800},
  motion:{active:true,fresh:true,cue:'MOVE_RIGHT',progress:.8,phase:'moving',action_id:2,sample_age_ms:20},
  feedback:{edge:'right',tone:'red',intensity:1,gesture_id:2}};
 assert.equal(api.signal(state),null);
 const shown=api.present(state);assert.match(shown.title,/reconnect/i);assert.doesNotMatch(shown.caption||'',/start|staff/i);
});

test('held state blocks new countdown even with a working local camera',()=>{
 const photo=moduleFor('guest-photo'),video={videoWidth:640,videoHeight:480,readyState:2,paused:false};
 const state={active:true,owner:true,photo_latched:true,photo_ready:true,control_fresh:true,
  runtime:{status:'running',reason:'none'},motion:{active:false},communication:{status:'waiting',recoverable:true,remaining_ms:2000}};
 assert.equal(photo.canStart(state,video),false);
});

test('communication hold rejects late encoding and upload completions but keeps a finished shared photo',async()=>{
 const photo=moduleFor('guest-photo');
 for(const point of ['encoding','upload','shared']){
  const c=clock(),shown=[],shared=[],hidden=[];let resolveCapture,resolveUpload;
  const capture=new Promise(resolve=>resolveCapture=resolve),upload=new Promise(resolve=>resolveUpload=resolve);
  const flow=photo.createFlow({env:c.env,valid:()=>true,stage:()=>{},capture:()=>capture,
   upload:()=>upload,show:blob=>shown.push(blob),share:info=>shared.push(info),hide:()=>hidden.push(true)});
  flow.update();await c.advance(3800);
  if(point!=='encoding'){resolveCapture('new-photo');await flush()}
  if(point==='shared'){resolveUpload({qr:'finished'});await flush()}
  flow.hold();
  resolveCapture('late-photo');resolveUpload({qr:'late'});await flush();
  if(point==='encoding'){assert.deepEqual(shown,[]);assert.deepEqual(shared,[])}
  if(point==='upload'){assert.deepEqual(shared,[]);assert.equal(hidden.length,1)}
  if(point==='shared'){assert.equal(shared.length,1);assert.equal(hidden.length,0)}
 }
});
