const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=()=>fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
const CUES=['MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER','ARRIVED',
 'INTRO_HELLO','WAIT_MOMENT','WAIT_SECOND','WAIT_FRAME','WAIT_ANGLE','WAIT_LOOK','WAIT_SPOT',
 'COUNTDOWN_THREE','COUNTDOWN_TWO','COUNTDOWN_ONE'];
function moduleFor(id){const block=html().match(new RegExp('<script id="'+id+'">([\\s\\S]*?)<\\/script>'));assert.ok(block,'missing '+id);const context={module:{exports:{}}};vm.runInNewContext(block[1],context);return context.module.exports}
const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve()};
function environment({failResume=false,failPlay=false,fetchClip,decode}={}){
 const env={now:0,document:{hidden:false},contexts:[],plays:[],stops:[],requests:[],timers:new Map(),timerId:0,resumes:0,silentStarts:0,decodes:0};
 env.Math={random:()=>0};env.performance={now:()=>env.now};env.setTimeout=(fn,ms)=>{const id=++env.timerId;env.timers.set(id,{fn,due:env.now+ms});return id};env.clearTimeout=id=>env.timers.delete(id);
 env.advance=ms=>{env.now+=ms;for(const [id,t] of [...env.timers])if(t.due<=env.now){env.timers.delete(id);t.fn()}};
 env.AudioContext=class{
  constructor(){this.state='suspended';this.sampleRate=22050;this.destination={};env.contexts.push(this)}
  resume(){env.resumes++;if(failResume)return Promise.reject(Error('autoplay denied'));this.state='running';return Promise.resolve()}
  createBuffer(){return {silent:true}}
  decodeAudioData(bytes,resolve,reject){env.decodes++;if(decode)return decode(bytes,resolve,reject);const buffer={cue:CUES[new Uint8Array(bytes)[0]-1]};resolve(buffer);return Promise.resolve(buffer)}
  createBufferSource(){const source={buffer:null,connect(){},disconnect(){},start(){if(this.buffer.silent){env.silentStarts++;return}if(failPlay)throw Error('speaker unavailable');env.plays.push(this)},stop(){env.stops.push(this)}};return source}
 };
 env.fetchClip=cue=>{env.requests.push(cue);return fetchClip?fetchClip(cue):Promise.resolve(new Uint8Array([CUES.indexOf(cue)+1]).buffer)};
 return env;
}
function state(cue='MOVE_LEFT',id=1,epoch=1){return {active:true,owner:true,control_fresh:true,control:{instance:'display-a',epoch},runtime:{status:'running',reason:'none'},communication:{status:'healthy',recoverable:true},tracking:false,phase:'PHYSICAL_REFERENCE_HOLD',motion:{active:true,fresh:true,started:true,cue,action_id:id,phase:'moving',progress:.2,sample_age_ms:20},feedback:{edge:{MOVE_LEFT:'left',MOVE_RIGHT:'right',MOVE_UP:'top',MOVE_DOWN:'bottom',MOVE_CLOSER:'all',MOVE_FARTHER:'all'}[cue],tone:'red',intensity:.2,gesture_id:id}}}
function setup(options){const env=environment(options),experience=moduleFor('guest-experience'),statuses=[];const speaker=moduleFor('guest-direction-audio').create(env,{fetchClip:env.fetchClip,motion:experience.motion,photo:experience.photo,onStatus:s=>statuses.push(s)});return {env,speaker,statuses}}
async function ready(options){const result=setup(options);await result.speaker.preload();assert.equal(await result.speaker.unlock(),true);return result}

test('cold audio construction and all decoding finish suspended before camera startup',async()=>{
 const {env,speaker}=setup();await speaker.prepare();assert.equal(env.contexts.length,1);assert.equal(env.contexts[0].state,'suspended');assert.equal(env.resumes,0);assert.equal(env.silentStarts,0);assert.equal(env.plays.length,0);assert.equal(env.decodes,CUES.length);
 const unlocking=speaker.unlock();assert.equal(env.resumes,1,'Start must resume in the same synchronous call');assert.equal(env.contexts.length,1);await unlocking;assert.equal(env.decodes,CUES.length,'Start must reuse decoded buffers');speaker.render(state());assert.equal(env.plays.length,1);
});
test('every production camera entry waits for silent preparation without requiring another visitor action',async()=>{
 const page=html(),source=page.match(/async function startCamera\(\)\{([^}]+)\}/);assert.ok(source,'camera preparation wrapper missing');
 let release;const events=[],context={directionAudio:{prepare(){events.push('prepare');return new Promise(r=>release=r)}},ipadCamera:{start(){events.push('camera');return Promise.resolve()}}};
 vm.runInNewContext(source[0]+';globalThis.open=startCamera;',context);const pending=context.open();assert.deepEqual(events,['prepare']);release();await pending;assert.deepEqual(events,['prepare','camera']);
 assert.equal((page.match(/ipadCamera\.start\(\)/g)||[]).length,1,'camera entry points must use the shared preparation wrapper');
 assert.match(page,/GuestExperience\.directionAudio=directionAudio;directionAudio\.prepare\(\)/);
});

test('all six directions use their matching preloaded clips once per measured action',async()=>{
 const {env,speaker}=await ready();for(let i=0;i<6;i++){const s=state(CUES[i],i+1);speaker.render(s);speaker.render(s);assert.equal(env.plays.at(-1).buffer.cue,CUES[i])}
 assert.equal(env.plays.length,6);assert.equal(new Set(env.requests).size,CUES.length);assert.equal(env.requests.length,CUES.length);
});
test('audio never creates a microphone or a browser speech-recognition session',async()=>{
 const {env,speaker}=await ready();speaker.render(state());assert.equal(env.contexts.length,1);assert.equal(env.resumes,1);assert.equal(env.silentStarts,1);assert.equal(env.plays.length,1);
});
test('zero progress, a first returning sample, and staged direction metadata cannot speak',async()=>{
 const {env,speaker}=await ready();const s=state();s.motion.progress=0;s.feedback.intensity=0;speaker.render(s);s.motion.progress=.5;s.motion.phase='returning';speaker.render(s);assert.equal(env.plays.length,0);
 s.motion.phase='moving';speaker.render(s);assert.equal(env.plays.length,1);
});
test('positive displacement without a literal native movement-start receipt never speaks',async()=>{
 const {env,speaker}=await ready();const s=state();for(const value of [false,undefined,null,'true']){s.motion.started=value;speaker.render(s)}assert.equal(env.plays.length,0);s.motion.started=true;speaker.render(s);assert.equal(env.plays.length,1);
});
test('normal reference-camera HOLD and the same action return/settling do not cut speech',async()=>{
 const {env,speaker}=await ready();const s=state();speaker.render(s);const current=env.plays[0];s.motion.phase='returning';speaker.render(s);s.motion.phase='settling';speaker.render(s);assert.equal(env.stops.includes(current),false);assert.equal(env.plays.length,1);
});
test('a repeated direction in a new action speaks again without overlapping the old clip',async()=>{
 const {env,speaker}=await ready();speaker.render(state('MOVE_RIGHT',1));const old=env.plays[0];speaker.render(state('MOVE_RIGHT',2));assert.equal(env.plays.length,2);assert.ok(env.stops.includes(old));
});
test('an epoch change cancels old playback and permits the same action ID in the new encounter',async()=>{
 const {env,speaker}=await ready();speaker.render(state());const old=env.plays[0];speaker.render(state('MOVE_LEFT',1,2));assert.equal(env.plays.length,2);assert.ok(env.stops.includes(old));
 old.onended();assert.equal(speaker.status().playing,true);
});
test('Stop, ownership loss, communication HOLD, invalid telemetry and backgrounding cancel audio',async()=>{
 for(const mutate of [s=>s.active=false,s=>s.stale=true,s=>s.owner=false,s=>s.control_fresh=false,s=>s.communication.status='waiting',s=>s.motion.fresh=false,s=>s.motion.sample_age_ms=301,s=>s.runtime.status='fault',s=>s.feedback.gesture_id=99]){
  const {env,speaker}=await ready();const s=state();speaker.render(s);mutate(s);speaker.render(s);assert.equal(speaker.status().playing,false);assert.equal(env.stops.length,1);
 }
 const {env,speaker}=await ready();speaker.render(state());env.document.hidden=true;speaker.render(state());assert.equal(speaker.status().playing,false);
});
test('a stale-state watchdog stops speech independently of a blocked polling request',async()=>{
 const {env,speaker}=await ready();speaker.render(state());env.advance(349);assert.equal(speaker.status().playing,true);env.advance(2);assert.equal(speaker.status().playing,false);speaker.render(state());assert.equal(env.plays.length,1);
});
test('mute clears playback, persists across unlock, and never queues missed directions',async()=>{
 const {env,speaker}=await ready();speaker.render(state());speaker.setEnabled(false);speaker.render(state('MOVE_RIGHT',2));await speaker.unlock();assert.equal(speaker.status().playing,false);assert.equal(env.plays.length,1);speaker.setEnabled(true);await speaker.unlock();assert.equal(env.plays.length,1);speaker.render(state('MOVE_UP',3));assert.equal(env.plays.at(-1).buffer.cue,'MOVE_UP');
});
test('late fetch/decode completion cannot play a direction after Stop',async()=>{
 let resolve;const pending=new Promise(r=>resolve=r);const {env,speaker}=setup({fetchClip:cue=>cue==='MOVE_RIGHT'?pending:Promise.resolve(new Uint8Array([CUES.indexOf(cue)+1]).buffer)});
 const unlocked=speaker.unlock();await flush();speaker.render(state('MOVE_RIGHT'));speaker.clear();resolve(new Uint8Array([2]).buffer);await unlocked;await flush();assert.equal(env.plays.length,0);speaker.render({...state('MOVE_RIGHT'),active:false});assert.equal(env.plays.length,0);speaker.render(state('MOVE_RIGHT',2));assert.equal(env.plays.length,1);
});
test('blocked autoplay and unavailable WebAudio show a hint without throwing or sending control',async()=>{
 const {env,speaker,statuses}=setup({failResume:true});assert.equal(await speaker.unlock(),false);speaker.render(state());assert.equal(env.plays.length,0);assert.ok(statuses.some(s=>s.error));
 const other=setup();delete other.env.AudioContext;assert.equal(await other.speaker.unlock(),false);assert.doesNotThrow(()=>other.speaker.render(state()));
});
test('a failed clip start is not retried by every state poll',async()=>{
 const {speaker,statuses}=await ready({failPlay:true});speaker.render(state());speaker.render(state());assert.equal(speaker.status().playing,false);assert.ok(statuses.some(s=>s.error));assert.equal(speaker.diagnostics().direction_audio_errors,1);
});
test('arrival speech requires the current verified photo state and plays once per epoch',async()=>{
 const {env,speaker}=await ready();const s=state();s.motion={active:false,fresh:false};s.ready=true;s.phase='SETTLING';speaker.render(s);assert.equal(env.plays.length,0);
 s.photo_ready=true;s.photo_latched=true;speaker.render(s);speaker.render(s);assert.equal(env.plays.length,1);assert.equal(env.plays[0].buffer.cue,'ARRIVED');
 s.control.epoch=2;speaker.render(s);assert.equal(env.plays.length,2);
});
test('Start unlocks audio before its first awaited request while legacy microphone stays disabled',()=>{
 const page=html(),action=page.slice(page.indexOf('async function action('),page.indexOf('async function sendMessage('));assert.ok(action.indexOf('directionAudio?.unlock()')>=0);assert.ok(action.indexOf('directionAudio?.unlock()')<action.indexOf('await '));
 assert.match(page,/<button id="direction-sound"[^>]*aria-pressed="true"/);assert.match(page,/const VISUAL_ONLY=true/);assert.match(page,/<button hidden disabled id="visitor-mic"/);assert.match(page,/fetch\('\/cue-audio\?cue='/);
});

function quietState(phase){return {...state(),phase,motion:{active:false,fresh:false},feedback:null}}
function observeFor(env,speaker,s,ms){for(let elapsed=0;elapsed<ms;elapsed+=100){env.advance(100);speaker.render(s)}}

test('greeting plays once for an active encounter and is cancelled by Stop',async()=>{
 const {env,speaker}=await ready();const s=quietState('INTRO');speaker.render(s);speaker.render(s);
 assert.equal(env.plays.length,1);assert.equal(env.plays[0].buffer.cue,'INTRO_HELLO');
 speaker.render({...s,active:false});assert.equal(speaker.status().playing,false);
 speaker.render(s);assert.equal(env.plays.length,1);
 speaker.render({...s,control:{...s.control,epoch:2}});assert.equal(env.plays.length,2);
});

test('healthy long framing waits use a spaced nonrepeating library with a three-clip cap',async()=>{
 const {env,speaker}=await ready();const s=quietState('AI_THINKING');speaker.render(s);
 observeFor(env,speaker,s,1400);assert.equal(env.plays.length,0);
 observeFor(env,speaker,s,100);assert.equal(env.plays.length,1);
 assert.match(env.plays[0].buffer.cue,/^WAIT_/);env.plays[0].onended();
 observeFor(env,speaker,s,7900);assert.equal(env.plays.length,1);
 observeFor(env,speaker,s,100);assert.equal(env.plays.length,2);
 assert.notEqual(env.plays[0].buffer.cue,env.plays[1].buffer.cue);env.plays[1].onended();
 observeFor(env,speaker,s,8000);assert.equal(env.plays.length,3);env.plays[2].onended();
 observeFor(env,speaker,s,20000);assert.equal(env.plays.length,3);
});

test('motion preempts waiting speech and recovered waits rebuild their delay',async()=>{
 const {env,speaker}=await ready();const s=quietState('AI_THINKING');speaker.render(s);observeFor(env,speaker,s,1500);
 const old=env.plays[0];speaker.render(state('MOVE_RIGHT'));assert.ok(env.stops.includes(old));
 assert.equal(env.plays.at(-1).buffer.cue,'MOVE_RIGHT');
 speaker.render({...s,communication:{status:'waiting',recoverable:true}});assert.equal(speaker.status().playing,false);
 speaker.render(s);observeFor(env,speaker,s,1400);assert.equal(env.plays.length,2);
 observeFor(env,speaker,s,100);assert.match(env.plays.at(-1).buffer.cue,/^WAIT_/);
});

test('model errors, missing people, ownership loss and communication recovery do not trigger filler speech',async()=>{
 for(const s of [quietState('AI_UNCERTAIN'),quietState('TRACKING_LOST'),{...quietState('AI_THINKING'),owner:false},
  {...quietState('AI_THINKING'),communication:{status:'waiting',recoverable:true}}]){
  const {env,speaker}=await ready();speaker.render(s);observeFor(env,speaker,s,12000);assert.equal(env.plays.length,0);
 }
});

test('spoken countdown follows actual photo stages and cancels on pause/result',async()=>{
 const {env,speaker}=await ready();const s={...quietState('ARRIVED'),photo_ready:true,photo_latched:true};
 speaker.render(s);assert.equal(env.plays.at(-1).buffer.cue,'ARRIVED');
 for(const [number,cue] of [[3,'COUNTDOWN_THREE'],[2,'COUNTDOWN_TWO'],[1,'COUNTDOWN_ONE']]){
  speaker.photoStage('countdown',number);assert.equal(env.plays.at(-1).buffer.cue,cue);
  speaker.render(s);assert.equal(env.plays.at(-1).buffer.cue,cue);
 }
 speaker.photoStage('paused');assert.equal(speaker.status().playing,false);
 speaker.photoStage('countdown',1);assert.equal(speaker.status().playing,true);
 speaker.photoStage('result');assert.equal(speaker.status().playing,false);speaker.render(s);
 assert.equal(speaker.status().playing,false);
});

test('photo intro can allow the arrival sentence to finish before the first spoken number',()=>{
 const {env}=setup(),stages=[];const photo=moduleFor('guest-photo');
 const flow=photo.createFlow({env,valid:()=>true,introDelayMs:1800,stage:(...args)=>stages.push(args),
  capture:async()=>({}),show(){},upload:async()=>({}),share(){},hide(){}});
 flow.update();env.advance(1799);assert.equal(stages.length,1);env.advance(1);
 assert.deepEqual([...stages[1]],['countdown',3]);
});

test('photo callbacks cannot extend the last received state freshness deadline',async()=>{
 const {env,speaker}=await ready();const s={...quietState('ARRIVED'),photo_ready:true,photo_latched:true};
 speaker.render(s);env.advance(300);speaker.photoStage('countdown',3);
 assert.equal(speaker.status().playing,true);env.advance(51);
 assert.equal(speaker.status().playing,false);
});

test('invalid measured motion cannot become a waiting voice during a reference hold',async()=>{
 const {env,speaker}=await ready();const s=state();s.motion.fresh=false;
 speaker.render(s);observeFor(env,speaker,s,5000);assert.equal(env.plays.length,0);
});
