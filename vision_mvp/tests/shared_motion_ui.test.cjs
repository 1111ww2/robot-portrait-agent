const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const file=path.join(__dirname,'../src/leica_tracker/guest.html');
function source(){return fs.readFileSync(file,'utf8')}
function experience(){const context={module:{exports:{}}};vm.runInNewContext(source().match(/<script id="guest-experience">([\s\S]*?)<\/script>/)[1],context);return context.module.exports}
function moving(changes={}){return {active:true,owner:true,control_fresh:true,stale:true,tracking:false,phase:'PHYSICAL_REFERENCE_HOLD',control:{instance:'run',epoch:2},runtime:{status:'running',reason:'none'},
 motion:{active:true,fresh:true,cue:'MOVE_RIGHT',phase:'moving',action_id:7,progress:.4,sample_age_ms:20},feedback:{edge:'right',tone:'red',intensity:.4,gesture_id:7},...changes}}
function renderFixture({preview=true}={}){
 const lights=[],views=[],actions=[],elements=new Map(),$=id=>{if(!elements.has(id))elements.set(id,{style:{},hidden:false});return elements.get(id)};
 const c={GuestExperience:experience(),IPadCamera:{viewState:()=>({showVeil:true})},cueLight:{set:x=>lights.push(x),clear:()=>lights.push(null)},recorder:{status:()=>({busy:false}),cancel(){}},voice:{skipReply(){},render(){}},photoFlow:{update(){},hold(){}},updateVoiceControls(){},document:{hidden:false,body:{classList:{toggle(){}}}},performance:{now:()=>100},$,
  localPreview:preview?{}:null,lastState:null,lastStateAt:0,VISUAL_ONLY:true,starting:false,actionPending:false,frameReady:true,lastPaintAt:-2000,videoStopped:false,visitorStopped:false,lastHeardText:'',heardUntil:0,heardReplyId:0,copy:(...v)=>views.push(v),action:(...a)=>actions.push(a)};
 vm.createContext(c);vm.runInContext(source().match(/function render\(s\)\{[\s\S]*?\n\}/)[0],c);
 return {c,lights,views,actions,elements,render:c.render};
}
test('actual render keeps accepted native direction and measured light while perception is suspended',()=>{
 const f=renderFixture(),s=moving();f.render(s);
 assert.deepEqual(f.lights.at(-1),s.feedback);
 assert.equal(f.views.at(-1)[0],'→');assert.equal(f.views.at(-1)[1],'A little right');
 assert.equal(f.elements.get('visitor-status').hidden,false);assert.equal(f.elements.get('veil').hidden,true);
});

test('actual recorded recoverable waiting state renders dark without a display-issued Stop',()=>{
 const state=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/native-communication-hold.json'),'utf8'));
 const f=renderFixture();f.render(state);
 assert.equal(f.lights.at(-1),null);assert.equal(f.views.at(-1)[1],'Reconnecting');assert.deepEqual(f.actions,[]);
});
test('native motion feedback cannot fall back to visual state after loss, stale sample or action mismatch',()=>{
 const s=moving({tracking:true,stale:false,phase:'GUIDING'});
 for(const state of [{...s,active:false},{...s,owner:false},{...s,control_fresh:false},{...s,runtime:{status:'fault',reason:'control'}},
  {...s,motion:{...s.motion,fresh:false}},{...s,motion:{...s.motion,sample_age_ms:301}},{...s,feedback:{...s.feedback,gesture_id:6}},
  {...s,motion:{...s.motion,cue:'HOLD'}},{...s,motion:{...s.motion,phase:'unknown'}},{...s,motion:{active:false,fresh:false}}]){
  const f=renderFixture();f.render(state);assert.equal(f.lights.at(-1),null);
 }
});
test('the accepted native cue distinguishes closer from farther although both use all edges',()=>{
 const v=experience();for(const [cue,title] of [['MOVE_CLOSER','Come closer'],['MOVE_FARTHER','A step back']]){
  const s=moving();s.motion.cue=cue;s.feedback.edge='all';assert.equal(v.present(s).title,title);
 }
});
test('JPEG fallback paints fresh native feedback from current state, not an old JPEG header',async()=>{
 const f=renderFixture({preview:false}),s=moving(),old={edge:'left',tone:'red',intensity:1,gesture_id:6};
 Object.assign(f.c,{lastState:s,lastStateAt:100,frameSequence:0,frameBusy:false,shownFrames:0,fps:0,fpsSince:100,headers:{},fitImage(){},setTimeout(){},
  GuestPainter:{present:async()=>{}},fetch:async()=>({status:200,headers:{get:n=>n==='X-Frame-Sequence'?'1':JSON.stringify(old)},blob:async()=>({})})});
 vm.runInContext(source().match(/async function frame\(\)\{[\s\S]*?\nasync function action/)[0].replace(/\nasync function action$/,''),f.c);
 await f.c.frame();assert.deepEqual(f.lights.at(-1),s.feedback);
});
test('preview display staleness does not send a motor Stop in the native workflow',()=>{
 for(const s of [moving(),moving({motion:{active:false,fresh:false}})]){const f=renderFixture({preview:false});f.render(s);assert.deepEqual(f.actions,[])}
});
test('native photo-ready confirmation is green independently of visual tracking',()=>{
 const f=renderFixture(),s=moving({motion:{active:false,fresh:false},photo_ready:true,photo_latched:true,feedback:null,phase:'ARRIVED'});f.render(s);
 assert.equal(f.lights.at(-1).edge,'all');assert.equal(f.lights.at(-1).tone,'green');assert.equal(f.lights.at(-1).intensity,1);
 assert.equal(f.views.at(-1)[1],'Get ready for your photo');
 f.render({...s,active:false});assert.equal(f.lights.at(-1),null);
});
test('actual edge painter uses a green DOM color for authoritative photo confirmation',()=>{
 const c={module:{exports:{}}},properties=new Map(),edges=['left','right','top','bottom'].map(cueEdge=>({dataset:{cueEdge},style:{}}));
 vm.runInNewContext(source().match(/<script id="guest-painter">([\s\S]*?)<\/script>/)[1],c);
 const root={style:{setProperty:(k,v)=>properties.set(k,v)},querySelectorAll:()=>edges};
 c.module.exports.feedback(root,{edge:'all',tone:'green',intensity:1});
 assert.equal(properties.get('--cue-color'),'#39d98a');
 assert.ok(edges.every(edge=>Number(edge.style.opacity)>0));
 c.module.exports.feedback(root,{edge:'right',tone:'red',intensity:.5});
 assert.equal(properties.get('--cue-color'),'#e20612');
});
test('hidden document clears feedback even if a preview callback attempts render',()=>{const f=renderFixture();f.c.document.hidden=true;f.render(moving());assert.equal(f.lights.at(-1),null)});
test('camera orientation changes do not impersonate a user Stop',()=>{
 const actions=[],stage={clientWidth:1000,clientHeight:700},nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,{style:{},value:'',...stage});return nodes.get(id)};
 const c={$,GuestExperience:experience(),GuestCameraGeometry:{plan:()=>({width:1280,height:720})},localPreview:{videoWidth:1280,videoHeight:720,style:{}},cameraRotation:()=>90,
  lastCameraRotation:0,lastState:moving(),visitorStopped:false,orientationChoice:'auto',localStorage:{setItem(){}},action:(...x)=>actions.push(x)};
 vm.createContext(c);vm.runInContext(source().match(/function fitImage\(\)\{[\s\S]*?\n\}/)[0]+'\n'+source().match(/function changeCameraOrientation\(value\)\{[\s\S]*?\n\}/)[0],c);
 c.fitImage();c.changeCameraOrientation('180');assert.deepEqual(actions,[]);
});
