const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
function api(){const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8'),block=html.match(/<script id="guest-photo">([\s\S]*?)<\/script>/);assert.ok(block,'photo module');const context={module:{exports:{}},globalThis:{}};const geometry={module:{exports:{}}};vm.runInNewContext(html.match(/<script id="guest-camera-geometry">([\s\S]*?)<\/script>/)[1],geometry);context.GuestCameraGeometry=geometry.module.exports;vm.runInNewContext(block[1],context);return context.module.exports}
function timers(){let now=0,id=0;const tasks=new Map();return {setTimeout(fn,delay){const n=++id;tasks.set(n,{fn,at:now+delay});return n},clearTimeout(n){tasks.delete(n)},advance(ms){const end=now+ms;for(;;){let first=null;for(const [n,t] of tasks)if(t.at<=end&&(!first||t.at<first.t.at))first={n,t};if(!first)break;tasks.delete(first.n);now=first.t.at;first.t.fn()}now=end}}}
async function flush(){for(let i=0;i<16;i++)await Promise.resolve()}
test('photo flow reports only numeric lifecycle and start-condition diagnostics',()=>{
 const p=api(),t=timers(),video={videoWidth:1280,videoHeight:720,readyState:4,paused:false};
 const flow=p.createFlow({env:t,valid:()=>true,stage(){},capture:async()=>({}),show(){},upload:async()=>({}),share(){},hide(){}});
 const state={active:true,owner:true,photo_latched:true};
 const before=p.diagnostics(state,video,false,flow);
 assert.equal(before.photo_active,true);assert.equal(before.photo_owner,true);
 assert.equal(before.photo_latched,true);assert.equal(before.photo_camera_ready,true);
 assert.equal(before.photo_started,false);assert.equal(before.photo_state,0);
 flow.update();assert.equal(p.diagnostics(state,video,false,flow).photo_state,1);
 assert.equal(p.diagnostics(state,video,false,flow).photo_started,true);
 flow.reset();assert.equal(p.diagnostics(null,null,false,flow).photo_state,0);
});
test('arrival latch starts photography even if the person has since left',()=>{
 const p=api(),video={videoWidth:1280,videoHeight:720,readyState:4,paused:false};
 const state={active:true,owner:true,photo_latched:true,tracking:false,stale:true,phase:'CAMERA_PAUSED'};
 assert.equal(p.canStart(state,video,false),true);
 assert.equal(p.canStart({...state,photo_latched:false},video,false),false);
 assert.equal(p.canStart({...state,active:false},video,false),false);
 assert.equal(p.canStart({...state,owner:false},video,false),false);
 assert.equal(p.canStart(state,video,true),false);
});
test('once latched, tracking changes cannot delay or restart the countdown',async()=>{
 const t=timers(),stages=[],pictures=[];let canStart=true,canCapture=true;
 const flow=api().createFlow({env:t,canStart:()=>canStart,canCapture:()=>canCapture,stage:(...v)=>stages.push(v),capture:async()=>({image:1}),show:b=>pictures.push(b),upload:async()=>({}),share(){},hide(){}});
 flow.update();t.advance(800);canStart=false;flow.update();t.advance(3000);await flush();
 assert.deepEqual(stages.map(x=>x[1]),[null,3,2,1,null]);assert.equal(pictures.length,1);
 flow.update();t.advance(4000);await flush();assert.equal(pictures.length,1);
});
test('a camera failure can pause the latch countdown and recover without losing it',async()=>{
 const t=timers(),stages=[],pictures=[];let canStart=true,canCapture=true;
 const flow=api().createFlow({env:t,canStart:()=>canStart,canCapture:()=>canCapture,stage:(...v)=>stages.push(v),capture:async()=>({image:1}),show:b=>pictures.push(b),upload:async()=>({}),share(){},hide(){}});
 flow.update();t.advance(1800);canStart=false;canCapture=false;flow.update();t.advance(4000);assert.equal(pictures.length,0);
 canCapture=true;flow.update();t.advance(2000);await flush();assert.equal(pictures.length,1);
});
test('waits for photo readiness then counts down once per encounter',async()=>{
 const t=timers(),stages=[],pictures=[],uploads=[];let valid=false;
 const flow=api().createFlow({env:t,valid:()=>valid,stage:(...v)=>stages.push(v),capture:async()=>({image:1}),show:b=>pictures.push(b),upload:async b=>{uploads.push(b);return {qr_data_url:'data:image/png;base64,A'}},share(){},hide(){}});
 flow.update();t.advance(5000);assert.equal(stages.length,0);
 valid=true;flow.update();flow.update();t.advance(800);t.advance(1000);t.advance(1000);t.advance(1000);await flush();
 assert.deepEqual(stages.map(x=>x[1]),[null,3,2,1,null]);assert.equal(pictures.length,1);assert.equal(uploads.length,1);
 flow.update();t.advance(6000);await flush();assert.equal(pictures.length,1);
});
test('pauses for stale input, resumes, and cancels a pending capture on reset',async()=>{
 const t=timers(),pictures=[];let valid=true,resolve;
 const flow=api().createFlow({env:t,valid:()=>valid,stage(){},capture:()=>new Promise(r=>resolve=r),show:x=>pictures.push(x),upload:async()=>({}),share(){},hide(){}});
 flow.update();t.advance(800);valid=false;flow.update();t.advance(4000);assert.equal(pictures.length,0);
 valid=true;flow.update();t.advance(3800);await flush();assert.equal(typeof resolve,'function');flow.reset();resolve({image:1});await flush();assert.equal(pictures.length,0);
});
test('stale input pauses countdown and resumes without silently taking a photo',async()=>{
 const t=timers(),stages=[],pictures=[];let valid=true;
 const flow=api().createFlow({env:t,valid:()=>valid,stage:(...x)=>stages.push(x),capture:async()=>({image:1}),show:x=>pictures.push(x),upload:async()=>({}),share(){},hide(){}});
 flow.update();t.advance(1800);assert.deepEqual(stages.map(x=>x[1]),[null,3,2]);
 valid=false;flow.update();t.advance(6000);await flush();assert.equal(pictures.length,0);assert.equal(stages.at(-1)[0],'paused');
 valid=true;flow.update();t.advance(2000);await flush();assert.equal(pictures.length,1);
});
test('captures a mirrored native-size live video frame, not the screen overlay',async()=>{
 const calls=[],blob={type:'image/jpeg'};const video={videoWidth:1280,videoHeight:720,readyState:4,paused:false};
 const env={document:{createElement:()=>({width:0,height:0,getContext(){return {setTransform(...x){calls.push(['transform',...x])},drawImage(...x){calls.push(['draw',...x])}}},toBlob(cb,type){calls.push(['mime',type]);cb(blob)}})}};
 assert.equal(await api().captureVideo(video,env),blob);assert.deepEqual(calls[0],['transform',-1,0,0,1,1280,0]);assert.equal(calls[1][1],video);assert.deepEqual(calls[1].slice(2),[0,0,1280,720]);
 await assert.rejects(api().captureVideo({videoWidth:0},env));
});
test('portrait photo uses the same rotation and mirror as the uploaded camera frame',async()=>{
 const calls=[],video={videoWidth:720,videoHeight:1280,readyState:4,paused:false};
 const canvas={getContext:()=>({setTransform:(...v)=>calls.push(v),drawImage:(...v)=>calls.push(v)}),toBlob:cb=>cb({size:20})};
 await api().captureVideo(video,{document:{createElement:()=>canvas}},90);
 assert.equal(canvas.width,1280);assert.equal(canvas.height,720);
 assert.deepEqual(calls[0],[0,1,1,0,0,0]);assert.deepEqual(calls[1],[video,0,0,720,1280]);
});
test('late upload after reset cannot show an old QR code',async()=>{
 const t=timers(),shares=[];let resolve,aborted=0;
 const flow=api().createFlow({env:t,valid:()=>true,stage(){},capture:async()=>({image:1}),show(){},upload:()=>new Promise(r=>resolve=r),share:x=>shares.push(x),hide(){},cancelAsync(){aborted++}});
 flow.update();t.advance(3800);await flush();assert.equal(typeof resolve,'function');flow.reset();resolve({qr_data_url:'old'});await flush();assert.equal(shares.length,0);assert.equal(aborted,1);
});
test('upload failure leaves captured preview visible and reports failure',async()=>{
 const t=timers(),shown=[],errors=[];
 const flow=api().createFlow({env:t,valid:()=>true,stage(){},capture:async()=>({image:1}),show:x=>shown.push(x),upload:async()=>{throw Error('down')},share(){},hide(){throw Error('must not hide photo')},error:x=>errors.push(x)});
 flow.update();t.advance(3800);await flush();assert.equal(shown.length,1);assert.deepEqual(errors,['upload']);
});
test('failed upload can be retried only while the original encounter is ready',async()=>{
 const t=timers(),errors=[],shares=[];let valid=true,attempts=0;
 const flow=api().createFlow({env:t,valid:()=>valid,stage(){},capture:async()=>({image:1}),show(){},upload:async()=>{if(++attempts===1)throw Error('retry');return {ok:true}},share:x=>shares.push(x),hide(){},error:x=>errors.push(x)});
 flow.update();t.advance(3800);await flush();assert.deepEqual(errors,['upload']);
 valid=false;assert.equal(flow.retry(),false);assert.equal(attempts,1);assert.equal(errors.at(-1),'notready');
 valid=true;assert.equal(flow.retry(),true);await flush();assert.equal(attempts,2);assert.equal(shares.length,1);
 flow.reset();assert.equal(flow.retry(),false);
});

 test('portrait countdown, sharing and error text is presented in English',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
  assert.match(html,/>This is your best position</);
  assert.match(html,/>Your portrait</);
  assert.match(html,/alt="Scan to download your portrait"/);
  assert.match(html,/Preparing your QR code/);
  assert.match(html,/Scan to download · Available for 5 minutes/);
  assert.match(html,/QR code unavailable/);
  assert.match(html,/Camera unavailable\. Tap New target to try again\./);
 });
