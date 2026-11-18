const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
function api(){const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');const code=html.match(/<script id="ipad-camera">([\s\S]*?)<\/script>/)[1];const context={module:{exports:{}}};const geometry={module:{exports:{}}};vm.runInNewContext(html.match(/<script id="guest-camera-geometry">([\s\S]*?)<\/script>/)[1],geometry);context.GuestCameraGeometry=geometry.module.exports;vm.runInNewContext(code,context);return context.module.exports}
function fixture(){const calls=[],timers=[],transforms=[];let stopped=0,permissions=0,timerId=0;const track={stop(){stopped++},addEventListener(){}};const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};const video={style:{},setAttribute(){},play:async()=>{},remove(){},videoWidth:1280,videoHeight:720,currentTime:1,requestVideoFrameCallback(callback){video.currentTime++;queueMicrotask(callback);return 1},cancelVideoFrameCallback(){}};const canvas={getContext:()=>({setTransform(...x){transforms.push(x)},drawImage(){}}),toBlob:cb=>cb({size:123})};const env={isSecureContext:true,AbortController,navigator:{mediaDevices:{getUserMedia:async()=>{permissions++;return stream}}},document:{createElement:type=>type==='video'?video:canvas,body:{appendChild(){}}},setTimeout:(fn,delay)=>{fn.delay=delay;fn.timerId=++timerId;timers.push(fn);return fn.timerId},clearTimeout:id=>{const i=timers.findIndex(fn=>fn.timerId===id);if(i>=0)timers.splice(i,1)}};const request=async(route,body,extra)=>{calls.push({route,body,extra});if(route==='/camera/start')return {session:'session'};if(route==='/camera/ticket')return {ticket:'ticket'};return {ok:true}};return {env,request,calls,timers,transforms,video,stream,get stopped(){return stopped},get permissions(){return permissions}}}
test('camera opens only explicitly, mirrors one bounded upload and closes tracks',async()=>{const f=fixture(),c=api().createCamera(f.env,f.request,()=>{},()=>{});assert.equal(f.permissions,0);await c.start();assert.equal(f.permissions,1);await f.timers.shift()();assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);assert.equal(f.transforms[0][0],-1);assert.ok(f.transforms[0][4]<=768);await c.stop();assert.equal(f.stopped,1);assert.ok(f.calls.some(x=>x.route==='/camera/stop'))});
test('late permission result after stop cannot reopen capture or upload',async()=>{const f=fixture();let resolve;f.env.navigator.mediaDevices.getUserMedia=()=>new Promise(r=>resolve=r);const c=api().createCamera(f.env,f.request,()=>{},()=>{});const started=c.start();await c.stop();resolve(f.stream);await started;assert.equal(f.stopped,1);assert.equal(f.calls.length,0)});
test('a frozen video frame is not resubmitted under a fresh ticket',async()=>{
 const f=fixture(),c=api().createCamera(f.env,f.request,()=>{},()=>{});
 await c.start();await f.timers.shift()();
 // 初始固定回调会推进视频时间；明确冻结下一个帧，而不是将自动推进回调误认为是冻结。
 f.video.requestVideoFrameCallback=()=>1;
 const pending=f.timers.shift()();for(let i=0;i<5;i++)await Promise.resolve();
 const deadline=f.timers.findIndex(fn=>fn.delay===180);assert.ok(deadline>=0);
 f.timers.splice(deadline,1)[0]();await pending;
 assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);await c.stop();
});
test('insecure page reports unavailable without requesting permission',async()=>{const f=fixture();f.env.isSecureContext=false;const status=[];const c=api().createCamera(f.env,f.request,x=>status.push(x),()=>{});await c.start();assert.equal(f.permissions,0);assert.ok(status.length)});

test('opening state enables cancellation before permission resolves',async()=>{const f=fixture();let resolve;f.env.navigator.mediaDevices.getUserMedia=()=>new Promise(r=>resolve=r);const ui={onDisabled:false,offDisabled:true};const c=api().createCamera(f.env,f.request,()=>{},busy=>{ui.onDisabled=busy;ui.offDisabled=!busy});const pending=c.start();assert.equal(ui.offDisabled,false);await c.stop();resolve(f.stream);await pending;assert.equal(ui.offDisabled,true);assert.equal(f.stopped,1);assert.equal(f.calls.length,0)});
test('a delayed ticket cannot bless an old frame that froze during the request',async()=>{const f=fixture();let issue;const original=f.request;const request=(route,...args)=>route==='/camera/ticket'?new Promise(r=>issue=r):original(route,...args);f.video.requestVideoFrameCallback=()=>1;const c=api().createCamera(f.env,request,()=>{},()=>{});await c.start();const uploading=f.timers.shift()();await Promise.resolve();issue({ticket:'fresh-ticket'});await Promise.resolve();await Promise.resolve();assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,0);if(f.timers.length)f.timers.shift()();await uploading;assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,0);await c.stop()});


test('local preview is available without waiting for a network frame and removed on stop',async()=>{const f=fixture(),previews=[];const c=api().createCamera(f.env,f.request,()=>{},()=>{},video=>previews.push(video));await c.start();assert.equal(previews[0],f.video);assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,0);await c.stop();assert.equal(previews.at(-1),null)});
test('temporary upload failures keep the camera open and retry without another permission',async()=>{const f=fixture();const request=(route,...args)=>route==='/camera/ticket'?Promise.reject(Error('network')):f.request(route,...args);const c=api().createCamera(f.env,request,()=>{},()=>{});await c.start();for(let i=0;i<12;i++){assert.ok(f.timers.length);await f.timers.shift()()}assert.equal(f.stopped,0);assert.equal(f.permissions,1);assert.ok(f.timers.length);await c.stop()});
test('slow uplink with a local preview never asks to open the already open camera',()=>{const view=api().viewState({camera_source:'ipad',stale:true,camera_connected:true},true,false);assert.equal(view.showVeil,false);assert.equal(view.needsCamera,false);assert.equal(view.waiting,true);assert.equal(api().viewState({camera_source:'ipad',stale:true,camera_connected:false},false,true).needsCamera,true)});


test('preview requests 30 fps by default and 60 only when selected',async()=>{const f=fixture();let constraints;f.env.navigator.mediaDevices.getUserMedia=async x=>{constraints=x;return f.stream};const applied=[];f.stream.getVideoTracks()[0].applyConstraints=async x=>applied.push(x);const c=api().createCamera(f.env,f.request,()=>{},()=>{});await c.start();assert.equal(constraints.video.frameRate.ideal,30);await c.setFrameRate(60);assert.equal(applied[0].frameRate.ideal,60);await c.stop()});
test('paused video is resumed before a fresh upload without reopening camera permission',async()=>{const f=fixture();let plays=0;f.video.play=async()=>{plays++;f.video.paused=false};const c=api().createCamera(f.env,f.request,()=>{},()=>{});await c.start();f.video.paused=true;await f.timers.shift()();assert.equal(plays,2);assert.equal(f.permissions,1);await c.stop()});
test('preview monitor counts presented frames independently of uploads and cancels on disposal',()=>{let callback,canceled;const video={requestVideoFrameCallback:f=>{callback=f;return 9},cancelVideoFrameCallback:id=>canceled=id,paused:false};const track={muted:false,readyState:'live',getSettings:()=>({frameRate:30})};const monitor=api().monitorPreview(video,track);callback(0,{presentedFrames:1});callback(1000,{presentedFrames:31});assert.equal(monitor.snapshot().preview_fps,30);track.muted=true;assert.equal(monitor.snapshot().track_muted,true);monitor.stop();assert.equal(canceled,9)});

test('a compositor callback for the current frame waits again instead of treating a healthy camera as lost',async()=>{const f=fixture(),callbacks=[];f.video.requestVideoFrameCallback=cb=>{callbacks.push(cb);return callbacks.length};const c=api().createCamera(f.env,f.request,()=>{},()=>{});await c.start();const pending=f.timers.shift()();for(let i=0;i<5;i++)await Promise.resolve();assert.equal(callbacks.length,1);callbacks[0]();assert.equal(callbacks.length,2);f.video.currentTime++;callbacks[1]();await pending;assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);await c.stop()});

test('upload shares the preview frame observer instead of competing for video callbacks',async()=>{const f=fixture(),callbacks=[];f.env.performance={};f.video.requestVideoFrameCallback=cb=>{callbacks.push(cb);return callbacks.length};const c=api().createCamera(f.env,f.request,()=>{},()=>{});await c.start();const pending=f.timers.shift()();for(let i=0;i<5;i++)await Promise.resolve();assert.equal(callbacks.length,1);callbacks[0](0,{presentedFrames:1});const poll=f.timers.find(fn=>fn.delay===10);assert.ok(poll);await poll();await pending;assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);await c.stop()});

test('freshness can observe decoded frames even when compositor callbacks are throttled',()=>{let decoded=10;const v={requestVideoFrameCallback:()=>1,cancelVideoFrameCallback(){},getVideoPlaybackQuality:()=>({totalVideoFrames:decoded})};const m=api().monitorPreview(v,{});assert.equal(m.sequence(),10);decoded=11;assert.equal(m.sequence(),11);m.stop()});

test('zero playback-quality counts on a live camera do not mask observed presented frames',()=>{let callback;const v={requestVideoFrameCallback:f=>{callback=f;return 1},cancelVideoFrameCallback(){},getVideoPlaybackQuality:()=>({totalVideoFrames:0,droppedVideoFrames:0})};const m=api().monitorPreview(v,{});callback(0,{presentedFrames:20});assert.equal(m.sequence(),20);callback(33,{presentedFrames:21});assert.equal(m.sequence(),21);m.stop()});

test('live ready video can advance after the ticket while compositor callbacks are throttled',async()=>{const f=fixture();f.env.performance={};f.video.readyState=4;f.video.requestVideoFrameCallback=()=>1;const c=api().createCamera(f.env,f.request,()=>{},()=>{});await c.start();const pending=f.timers.shift()();for(let i=0;i<5;i++)await Promise.resolve();f.video.currentTime++;const poll=f.timers.find(fn=>fn.delay===10);await poll();for(let i=0;i<5;i++)await Promise.resolve();assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);await pending;await c.stop()});

test('camera advertises the latched photo flow version as numeric diagnostics',()=>{
 const f=fixture(),c=api().createCamera(f.env,f.request,()=>{},()=>{});
 assert.equal(c.diagnostics().photo_flow_version,4);
});

test('camera reports last upload stage durations without recording frames',async()=>{
  const f=fixture();let now=100;
  const request=async(route,...args)=>{if(route==='/camera/ticket')now+=20;if(route==='/camera/frame')now+=30;return f.request(route,...args)};
  const c=api().createCamera(f.env,request,()=>{},()=>{});await c.start();
  f.env.performance={now:()=>now};
  f.video.requestVideoFrameCallback=cb=>{now+=10;f.video.currentTime++;queueMicrotask(cb);return 1};
  const canvas=f.env.document.createElement('canvas');canvas.toBlob=cb=>{now+=40;cb({size:123})};
  await f.timers.shift()();
  const d=c.diagnostics();
  assert.equal(d.ticket_ms,20);assert.equal(d.fresh_frame_ms,10);assert.equal(d.encode_ms,40);assert.equal(d.upload_ms,30);
  await c.stop();
});

test('camera diagnoses upload failures without closing the preview',async()=>{
  const f=fixture();let now=100;
  const request=async(route,...args)=>{if(route==='/camera/frame'){now+=180;throw Error('frame request failed')}return f.request(route,...args)};
  const c=api().createCamera(f.env,request,()=>{},()=>{});await c.start();f.env.performance={now:()=>now};
  await f.timers.shift()();const d=c.diagnostics();assert.equal(d.error_stage,4);assert.equal(d.error_ms,180);assert.equal(f.stopped,0);await c.stop();
});

test('transient frame-upload failures retry promptly with a fresh ticket and new frame',async()=>{
  const f=fixture();let failed=0;const frames=[];
  const create=f.env.document.createElement;
  f.env.document.createElement=type=>{
    const el=create(type);if(type==='canvas')el.toBlob=cb=>cb({size:123,frameId:f.video.currentTime});return el;
  };
  const request=async(route,...args)=>{
    if(route==='/camera/frame'){
      frames.push(args[0].frameId);
      if(failed++<3)throw Error('transient frame failure');
    }
    return f.request(route,...args);
  };
  const c=api().createCamera(f.env,request,()=>{},()=>{});await c.start();
  for(const delay of [40,100,250]){
    f.video.currentTime++;
    await f.timers.shift()();
    assert.equal(f.timers[0].delay,delay);
  }
  f.video.currentTime++;await f.timers.shift()();
  assert.equal(f.calls.filter(x=>x.route==='/camera/ticket').length,4);
  assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);
  assert.equal(frames.length,4);assert.ok(frames.every((n,i)=>!i||n>frames[i-1]));
  assert.equal(f.stopped,0);await c.stop();
});

test('persistent errors and failures before upload retain the longer retry interval',async()=>{
  const f=fixture();const request=async(route,...args)=>{
    if(route==='/camera/ticket')throw Error('ticket unavailable');
    return f.request(route,...args);
  };
  const c=api().createCamera(f.env,request,()=>{},()=>{});await c.start();
  await f.timers.shift()();assert.equal(f.timers[0].delay,500);await c.stop();
});

test('repeated frame-upload failures settle at a bounded 500ms retry',async()=>{
  const f=fixture();const request=async(route,...args)=>{
    if(route==='/camera/frame')throw Error('frame unavailable');
    return f.request(route,...args);
  };
  const c=api().createCamera(f.env,request,()=>{},()=>{});await c.start();
  for(const delay of [40,100,250,500,500]){
    f.video.currentTime++;await f.timers.shift()();assert.equal(f.timers[0].delay,delay);
  }
  assert.equal(f.permissions,1);assert.equal(f.stopped,0);await c.stop();
});

test('portrait iPad upload uses the selected rotation and landscape output dimensions',async()=>{
 const f=fixture();f.video.videoWidth=720;f.video.videoHeight=1280;
 const canvas=f.env.document.createElement('canvas');
 const c=api().createCamera(f.env,f.request,()=>{},()=>{},()=>{},()=>({}),()=>90);
 await c.start();await f.timers.shift()();
 assert.equal(canvas.width,640);assert.equal(canvas.height,360);
 assert.deepEqual(f.transforms[0],[0,1,1,0,0,0]);
 assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);
 await c.stop();
});

function liveTrial(f){const track=f.stream.getVideoTracks()[0];track.readyState='live';track.muted=false;f.video.readyState=4;f.video.paused=false;f.env.document.hidden=false;return ()=>({photo_active:true})}
test('visible active camera retries transient ticket failures promptly, then backs off without reopening camera',async()=>{
 const f=fixture(),active=liveTrial(f);let failures=0;
 const request=async(route,...args)=>{if(route==='/camera/ticket'&&failures++<4)throw Error('temporary ticket failure');return f.request(route,...args)};
 const c=api().createCamera(f.env,request,()=>{},()=>{},()=>{},active);await c.start();
 for(const delay of [40,100,250,500]){await f.timers.shift()();assert.equal(f.timers[0].delay,delay)}
 await f.timers.shift()();assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);
 assert.equal(f.permissions,1);assert.equal(f.calls.filter(x=>x.route==='/camera/start').length,1);await c.stop();
});
test('active visible capture retries a temporary encode failure with a new ticket and frame',async()=>{
 const f=fixture(),active=liveTrial(f),canvas=f.env.document.createElement('canvas');let encodes=0;
 canvas.toBlob=cb=>cb(encodes++===0?null:{size:123});
 const c=api().createCamera(f.env,f.request,()=>{},()=>{},()=>{},active);await c.start();
 await f.timers.shift()();assert.equal(f.timers[0].delay,40);assert.equal(c.diagnostics().error_stage,3);
 f.video.currentTime++;await f.timers.shift()();
 assert.equal(f.calls.filter(x=>x.route==='/camera/ticket').length,2);assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,1);await c.stop();
});
test('fresh-frame timeout can retry promptly but never uploads the timed-out image',async()=>{
 const f=fixture(),active=liveTrial(f);f.video.requestVideoFrameCallback=()=>1;
 const c=api().createCamera(f.env,f.request,()=>{},()=>{},()=>{},active);await c.start();
 const pending=f.timers.shift()();for(let i=0;i<5;i++)await Promise.resolve();
 const deadline=f.timers.findIndex(x=>x.delay===180);assert.ok(deadline>=0);f.timers.splice(deadline,1)[0]();await pending;
 assert.equal(f.timers[0].delay,40);assert.equal(c.diagnostics().error_stage,2);assert.equal(f.calls.filter(x=>x.route==='/camera/frame').length,0);await c.stop();
});
test('inactive, hidden, muted and ended cameras keep conservative retries for pre-upload faults',async()=>{
 for(const condition of ['inactive','hidden','muted','ended']){
  const f=fixture();liveTrial(f);if(condition==='hidden')f.env.document.hidden=true;
  if(condition==='muted')f.stream.getVideoTracks()[0].muted=true;
  if(condition==='ended')f.stream.getVideoTracks()[0].readyState='ended';
  const request=(route,...args)=>route==='/camera/ticket'?Promise.reject(Error('ticket failure')):f.request(route,...args);
  const c=api().createCamera(f.env,request,()=>{},()=>{},()=>{},()=>({photo_active:condition!=='inactive'}));await c.start();
  await f.timers.shift()();assert.equal(f.timers[0].delay,500,condition);await c.stop();
 }
});

test('a fast ticket waits for the next healthy frame without a redundant 100ms idle gap',async()=>{
 let now=0,next=0,frameNumber=0,ticketCount=0,uploadCount=0;
 const timers=new Map(),frameCallbacks=new Map(),accepted=[],errors=[];
 const setTimeout=(fn,delay)=>{const id=++next;timers.set(id,{fn,at:now+delay});return id};
 const clearTimeout=id=>timers.delete(id);
 const track={readyState:'live',muted:false,getSettings:()=>({frameRate:30}),addEventListener(){},stop(){}};
 const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};
 const video={videoWidth:1280,videoHeight:720,currentTime:0,readyState:4,paused:false,style:{},
  setAttribute(){},play:async()=>{},remove(){},
  requestVideoFrameCallback(fn){const id=++next;frameCallbacks.set(id,fn);return id},
  cancelVideoFrameCallback(id){frameCallbacks.delete(id)}};
 const canvas={getContext:()=>({setTransform(){},drawImage(){}}),toBlob:cb=>cb({size:123})};
 const env={performance:{now:()=>now},isSecureContext:true,AbortController,setTimeout,clearTimeout,
  navigator:{mediaDevices:{getUserMedia:async()=>stream}},
  document:{hidden:false,body:{appendChild(){}},createElement:type=>type==='video'?video:canvas}};
 const delay=(ms,value,onComplete)=>new Promise(resolve=>setTimeout(()=>{if(onComplete)onComplete();resolve(value)},ms));
 const request=route=>{
  if(route==='/camera/start')return Promise.resolve({session:'synthetic'});
  if(route==='/camera/ticket'){ticketCount++;return delay(ticketCount<=2?5:120,{ticket:'ticket'+ticketCount})}
  if(route==='/camera/frame'){uploadCount++;return delay(uploadCount===1?5:120,{ok:true},()=>accepted.push(now))}
  return Promise.resolve({ok:true});
 };
 const camera=api().createCamera(env,request,()=>{},()=>{},()=>{},()=>({photo_active:true}));
 function presentFrame(){
  frameNumber++;video.currentTime=frameNumber/30;
  const callbacks=[...frameCallbacks.values()];frameCallbacks.clear();
  for(const callback of callbacks)callback(now,{presentedFrames:frameNumber});
  setTimeout(presentFrame,1000/30);
 }
 setTimeout(presentFrame,1000/30);
 try{
  await camera.start();
  while(now<400){
   const ready=[...timers].sort((a,b)=>a[1].at-b[1].at)[0];
   if(!ready||ready[1].at>400)break;
   now=ready[1].at;timers.delete(ready[0]);
   const pending=ready[1].fn();if(pending&&pending.catch)pending.catch(error=>errors.push(error.message));
   for(let i=0;i<12;i++)await Promise.resolve();
  }
  // 30fps和成功<=120ms请求：立即等待下一个后票帧，结果是155ms，而不是以前的355ms上传间隙。
  assert.ok(accepted.length>=2,'healthy frames must continue arriving');
  assert.ok(accepted[1]-accepted[0]<300,`successful upload gap must stay below the unchanged freshness deadline: ${JSON.stringify(accepted)}`);
  assert.equal(Object.hasOwn(camera.diagnostics(),'error_stage'),false);
  assert.deepEqual(errors,[]);
 }finally{await camera.stop()}
});

test('camera first-error diagnostics persist across recovery and later errors until camera restart',async()=>{
 const f=fixture();let now=0,tickets=0,uploads=0;
 const request=async(route,...args)=>{
  if(route==='/camera/ticket'){
   now+=++tickets===1?12:8;
   if(tickets===1)throw Object.assign(Error('private response text'),{camera_error_kind:1,camera_http_status:409});
  }
  if(route==='/camera/frame'){
   now+=++uploads===1?17:35;
   if(uploads===2)throw Object.assign(Error('private timeout text'),{camera_error_kind:2});
  }
  return f.request(route,...args);
 };
 const camera=api().createCamera(f.env,request,()=>{},()=>{});await camera.start();
 f.env.performance={now:()=>now};
 await f.timers.shift()();
 f.video.currentTime++;await f.timers.shift()();
 f.video.currentTime++;await f.timers.shift()();
 let d=camera.diagnostics();
 assert.equal(d.camera_first_error_stage,1);assert.equal(d.camera_first_error_kind,1);
 assert.equal(d.camera_first_error_status,409);assert.equal(d.camera_first_error_ms,12);
 assert.equal(d.camera_error_count,2);assert.equal(d.camera_max_ticket_ms,12);assert.equal(d.camera_max_upload_ms,35);
 assert.equal(JSON.stringify(d).includes('private'),false);
 await camera.stop();f.env.performance=undefined;await camera.start();d=camera.diagnostics();
 assert.equal(d.camera_first_error_kind,0);assert.equal(d.camera_error_count,0);await camera.stop();
});

test('camera stage maxima include failed fresh waits and encoding as well as successful uploads',async()=>{
 const f=fixture();let now=0;f.env.performance={};f.video.requestVideoFrameCallback=()=>1;
 const request=async(route,...args)=>{if(route==='/camera/ticket')now+=22;if(route==='/camera/frame')now+=18;return f.request(route,...args)};
 const camera=api().createCamera(f.env,request,()=>{},()=>{});await camera.start();
 f.env.performance.now=()=>now;f.video.requestVideoFrameCallback=()=>1;
 const pending=f.timers.shift()();for(let i=0;i<5;i++)await Promise.resolve();
 now+=180;const deadline=f.timers.findIndex(fn=>fn.delay===180);f.timers.splice(deadline,1)[0]();await pending;
 let d=camera.diagnostics();assert.equal(d.camera_max_ticket_ms,22);assert.equal(d.camera_max_fresh_frame_ms,180);
 assert.equal(d.camera_first_error_stage,2);assert.equal(d.camera_first_error_kind,5);
 const canvas=f.env.document.createElement('canvas');canvas.toBlob=cb=>{now+=31;cb(null)};
 f.video.readyState=4;f.video.paused=false;f.video.currentTime++;
 const again=f.timers.shift()();for(let i=0;i<5;i++)await Promise.resolve();
 f.video.currentTime++;const poll=f.timers.findIndex(fn=>fn.delay===10);f.timers.splice(poll,1)[0]();await again;
 d=camera.diagnostics();assert.equal(d.camera_max_encode_ms,31);assert.equal(d.camera_error_count,2);
 assert.equal(d.camera_first_error_kind,5);await camera.stop();
});

test('camera reports bounded lifetime error count and gaps between successful upload acknowledgements',async()=>{
 const f=fixture();let now=100,fail=false;
 const request=async(route,...args)=>{
  if(route==='/camera/frame'&&fail){now+=20000;throw Error('unavailable')}
  return f.request(route,...args);
 };
 const camera=api().createCamera(f.env,request,()=>{},()=>{});await camera.start();f.env.performance={now:()=>now};
 await f.timers.shift()();now+=170;f.video.currentTime++;await f.timers.shift()();
 assert.equal(camera.diagnostics().camera_max_upload_ack_gap_ms,170);
 fail=true;
 for(let i=0;i<242;i++){f.video.currentTime++;await f.timers.shift()()}
 const d=camera.diagnostics();assert.equal(d.camera_error_count,240);assert.equal(d.camera_max_upload_ms,10000);
 assert.equal(d.camera_first_error_ms,10000);assert.equal(d.camera_first_error_kind,4);
 await camera.stop();
});

function cameraRequestFixture(){
 const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
 const code=html.slice(html.indexOf('async function cameraRequest('),html.indexOf('let photoObjectURL=',html.indexOf('async function cameraRequest(')));
 const timers=new Map();let next=0,seenSignal;
 const context={AbortController,headers:{'X-Guest-Token':'synthetic-private-key'},
  setTimeout(fn,ms){const id=++next;timers.set(id,{fn,ms});return id},clearTimeout:id=>timers.delete(id),
  fetch:(_path,options)=>{seenSignal=options.signal;return new Promise((_resolve,reject)=>{
   const abort=()=>reject(Object.assign(Error('private URL and key'),{name:'AbortError'}));
   if(options.signal.aborted)abort();else options.signal.addEventListener('abort',abort,{once:true});
  })}};
 vm.runInNewContext(code,context);
 return {context,timers,request:(...args)=>context.cameraRequest(...args),signal:()=>seenSignal};
}

test('camera request classifies HTTP refusal without retaining private response text',async()=>{
 const f=cameraRequestFixture();f.context.fetch=async()=>({ok:false,status:413});
 await assert.rejects(f.request('/camera/frame',{},null),error=>error.camera_error_kind===1&&error.camera_http_status===413&&!error.message.includes('synthetic-private-key'));
 assert.equal(f.timers.size,0);
});

test('camera request distinguishes its unchanged local timeout from external cancellation',async()=>{
 for(const external of [false,true]){
  const f=cameraRequestFixture(),controller=new AbortController();
  const pending=f.request('/camera/frame',{},null,controller.signal);
  assert.equal([...f.timers.values()][0].ms,2500);
  if(external)controller.abort();else [...f.timers.values()][0].fn();
  await assert.rejects(pending,error=>error.camera_error_kind===(external?3:2)&&error.camera_http_status===0);
  assert.equal(f.signal().aborted,true);assert.equal(f.timers.size,0);
 }
 const photo=cameraRequestFixture(),controller=new AbortController();const pending=photo.request('/photo',{},null,controller.signal);
 assert.equal([...photo.timers.values()][0].ms,10000);controller.abort();await assert.rejects(pending);
});
