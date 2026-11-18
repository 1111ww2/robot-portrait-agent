const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
test('speed and control interruptions explain a pause without implying immediate retry readiness',()=>{
 const speed=api().present({active:true,runtime:{status:'fault',reason:'speed'}});
 assert.match(speed.caption,/speed/i);assert.doesNotMatch(speed.caption,/tap Start/i);
 const control=api().present({active:true,runtime:{status:'fault',reason:'control'}});
 assert.match(control.caption,/control link/i);assert.doesNotMatch(control.caption,/tap Start/i);
});
test('physical runtime fault is visible instead of offering blind intro retry',()=>{
 const p=api().present({active:true,phase:'INTRO',tracking:true,runtime:{status:'fault',reason:'reference'}});
 assert.equal(p.title,'Robot paused');assert.match(p.caption,/start position/);
 const c=api().present({active:true,phase:'INTRO_UNAVAILABLE',runtime:{status:'fault',reason:'camera'}});
 assert.match(c.caption,/Camera link interrupted/);
});
test('persistent runtime waiting for Start does not falsely demand a position check',()=>{
 const p=api().present({active:false,phase:'PHYSICAL_REFERENCE_HOLD',runtime:{status:'waiting_start',reason:'none'}});
 assert.equal(p.title,'Come as you are');assert.equal(p.caption,'Tap Start to begin.');
});
test('reference hold remains visible even if the visitor encounter is inactive',()=>{const p=api().present({active:false,phase:'PHYSICAL_REFERENCE_HOLD'});assert.equal(p.title,'Reference check needed');assert.ok(p.caption.includes('Staff'))});
test('live controlled reference hold waits without requesting a staff position check',()=>{
 const p=api().present({active:true,phase:'PHYSICAL_REFERENCE_HOLD',control_fresh:true,
  runtime:{status:'running',reason:'none'}});
 assert.equal(p.title,'One moment');assert.doesNotMatch(p.caption,/staff|check/i);
});
test('reference hold does not conceal a stopped runtime, fault or lost control',()=>{
 const v=api(),hold={active:true,phase:'PHYSICAL_REFERENCE_HOLD',control_fresh:true};
 const fault=v.present({...hold,runtime:{status:'fault',reason:'reference'}});
 assert.equal(fault.title,'Robot paused');assert.match(fault.caption,/start position/i);
 for(const state of [
  {...hold,runtime:{status:'stopped',reason:'control'}},
  {...hold,control_fresh:false,runtime:{status:'running',reason:'none'}},
  {...hold,runtime:{status:'running',reason:'control'}},
  {...hold,active:false,runtime:{status:'running',reason:'none'}},
  {...hold,runtime:{status:'starting',reason:'none'}}]){
  const p=v.present(state);assert.equal(p.title,'Reference check needed');assert.match(p.caption,/Staff/);
 }
});
test('failed planning tells visitor to retry instead of waiting forever',()=>{for(const phase of ['AI_NO_INTERACTION_TARGET','AI_INFEASIBLE_TARGET','AI_NETWORK_ERROR','AI_INVALID_RESPONSE']){const p=api().present({active:true,phase});assert.equal(p.title,'Try a new view');assert.ok(p.caption.includes('New target'))}});
function api(){const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8'),block=html.match(/<script id="guest-experience">([\s\S]*?)<\/script>/);assert.ok(block,'guest experience module');const context={module:{exports:{}}};vm.runInNewContext(block[1],context);return context.module.exports}
test('guest captions stay compact and finished differs from pause',()=>{const v=api();assert.equal(v.present({active:true,phase:'AGENT_DONE'}).title,'Your portrait');assert.equal(v.present({stale:true}).title,'One moment');const p=v.present({active:true,phase:'AGENT_WAITING',dialogue:{speech:'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen'}});assert.ok(p.caption.split(/\s+/).length<=12)});
test('arrival describes preparing the photo rather than implying one was taken',()=>{
 const p=api().present({active:true,ready:true,photo_latched:true,phase:'ARRIVED'});
 assert.equal(p.title,'Get ready for your photo');
});
test('right cue labels agree with shared right edge and pose diagnostics never leak',()=>{const p=api().present({active:true,direction:'MOVE_RIGHT',phase:'AGENT_ACTING',dialogue:{speech:'debug head inside green area'}});assert.equal(p.title,'A little right');assert.equal(p.icon,'→');assert.equal(p.caption,'')});
test('image geometry fits landscape and portrait without crop coordinate drift',()=>{const f=api().fit;assert.deepEqual({...f(1280,720,1024,768)},{width:1024,height:576});assert.deepEqual({...f(1280,720,768,1024)},{width:768,height:432});assert.deepEqual({...f(720,1280,1024,768)},{width:432,height:768})});
test('preparing the display never starts an encounter; Start and New target are explicit actions',()=>{
 const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
 assert.match(html,/<button id="visitor-start">Start<\/button>/);
 assert.match(html,/\$\('visitor-start'\)\.onclick=\(\)=>action\('start',true\)/);
 assert.match(html,/\$\('visitor-replan'\)\.onclick=\(\)=>action\('restart',true\)/);
 const prepare=html.match(/\$\('prepare'\)\.onclick=async\(\)=>\{([\s\S]*?)\n\};/);
 assert.ok(prepare);assert.doesNotMatch(prepare[1],/session\.prepare|action\s*\(/);
 assert.match(prepare[1],/\$\('setup'\)\.hidden=true/);
 assert.doesNotMatch(html,/session\.next\(/);
});
test('opening gesture and unavailable gesture communicate the next step',()=>{
 const v=api(),intro=v.present({active:true,phase:'INTRO'}),unavailable=v.present({active:true,phase:'INTRO_UNAVAILABLE'});
 assert.equal(intro.title,'Curious to meet you');
 assert.equal(unavailable.title,'Opening gesture unavailable');
 assert.match(unavailable.caption,/Start/);assert.match(unavailable.caption,/New target/);
});

test('fullscreen toggles the entire interaction, including lights, and can exit',async()=>{
 let requested=0,exited=0;const doc={documentElement:{requestFullscreen:async()=>requested++},exitFullscreen:async()=>exited++};
 assert.equal(await api().fullscreen(doc),'entered');assert.equal(requested,1);
 doc.fullscreenElement=doc.documentElement;assert.equal(await api().fullscreen(doc),'exited');assert.equal(exited,1);
});
test('fullscreen handles prefixed Safari and reports unsupported or denied honestly',async()=>{
 let requested=0;assert.equal(await api().fullscreen({documentElement:{webkitRequestFullscreen:()=>requested++}}),'entered');assert.equal(requested,1);
 assert.equal(await api().fullscreen({documentElement:{}}),'unsupported');
 assert.equal(await api().fullscreen({documentElement:{requestFullscreen:async()=>{throw Error('denied')}}}),'denied');
});
