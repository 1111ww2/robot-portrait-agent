const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
function moduleUnderTest(){const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');const code=html.match(/<script id="guest-dialogue">([\s\S]*?)<\/script>/)[1];const context={module:{exports:{}}};vm.runInNewContext(code,context);return context.module.exports}
function environment(){let spoken=[],cancels=0;class Recognition{start(){}abort(){this.aborted=true}};return {isSecureContext:true,SpeechRecognition:Recognition,SpeechSynthesisUtterance:class{constructor(text){this.text=text}},speechSynthesis:{speak:u=>spoken.push(u.text),cancel:()=>cancels++},spoken,get cancels(){return cancels}}}
test('speech is delivered once and canceled on stop',()=>{const api=moduleUnderTest(),env=environment(),v=api.createVoice(env,()=>{},()=>{});v.setEnabled(true);v.render({active:true,dialogue:{id:1,speech:'Hello'}});v.render({active:true,dialogue:{id:1,speech:'Hello'}});assert.deepEqual(env.spoken,['Hello']);v.render({active:false});assert.ok(env.cancels>0)});
test('recognition callback arriving after stop cannot submit text',()=>{const api=moduleUnderTest(),env=environment(),texts=[],v=api.createVoice(env,t=>texts.push(t),()=>{});const rec=v.listen();const callback=rec.onresult;v.cancel();callback({results:[[{transcript:'Continue to the right'}]]});assert.deepEqual(texts,[])});
test('microphone is unavailable on insecure LAN and never starts automatically',()=>{const api=moduleUnderTest(),env=environment();env.isSecureContext=false;const v=api.createVoice(env,()=>{},()=>{});assert.equal(v.available,false);assert.equal(v.listen(),null)});
test('stop phrases bypass the model, but a negated stop is a normal message',()=>{const {isStop}=moduleUnderTest();assert.equal(isStop('\u505c\u6b62\u4e92\u52a8。'),true);assert.equal(isStop('stop'),true);assert.equal(isStop('\u4e0d\u8981\u505c\u6b62'),false)});
test('model speech waits while microphone is listening instead of aborting it',()=>{const api=moduleUnderTest(),env=environment(),texts=[],v=api.createVoice(env,t=>texts.push(t),()=>{});v.setEnabled(true);const rec=v.listen();v.render({active:true,dialogue:{id:2,speech:'A little to your right'}});assert.equal(rec.aborted,undefined);assert.deepEqual(env.spoken,[]);rec.onresult({results:[[{transcript:'I do not want to move'}]]});assert.deepEqual(texts,['I do not want to move']);rec.onend();v.render({active:true,dialogue:{id:2,speech:'A little to your right'}});assert.deepEqual(env.spoken,[])});
function clockEnvironment(){const env=environment();env.now=0;env.performance={now:()=>env.now};env.instances=[];env.utterances=[];env.SpeechRecognition=class{constructor(){env.instances.push(this)}start(){}abort(){this.aborted=true}};env.speechSynthesis.speak=u=>{env.spoken.push(u.text);env.utterances.push(u)};return env}
test('speech completion never opens the microphone without an explicit tap',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});
 v.setHandsFree(true);v.setEnabled(true);const state={active:true,owner:true,dialogue:{id:1,speech:'A little to your right.'}};
 v.render(state);assert.equal(env.spoken.length,1);assert.equal(env.instances.length,0);
 env.utterances[0].onend();env.now=1500;v.render(state);assert.equal(env.instances.length,0);v.arm();assert.equal(env.instances.length,1);
 assert.equal(env.instances[0].lang,'en-AU');
});
test('denied recognition blocks automatic retry and late callbacks after stop do nothing',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),texts=[],v=api.createVoice(env,t=>texts.push(t),()=>{});
 v.arm();v.render({active:true,owner:true});const r=env.instances[0];
 r.onerror({error:'not-allowed'});r.onend();env.now=5000;v.render({active:true,owner:true});assert.equal(env.instances.length,1);
 v.render({active:false});r.onresult({results:[[{transcript:'move right'}]]});assert.deepEqual(texts,[]);
});
test('speech is not replayed on a passive turn with the same utterance',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});v.setEnabled(true);
 v.render({active:true,dialogue:{id:1,speech:'Stay there.'}});env.utterances[0].onend();
 v.render({active:true,dialogue:{id:2,speech:'Stay there.'}});assert.equal(env.spoken.length,1);
});
test('non-owner and stale pages never listen or speak',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});v.setHandsFree(true);v.setEnabled(true);
 v.render({active:true,owner:false,dialogue:{id:1,speech:'Hello'}});v.render({active:true,stale:true});
 assert.equal(env.instances.length,0);assert.equal(env.spoken.length,0);
});
test('explicit stop cancels speech and ignores late microphone callbacks',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),texts=[],v=api.createVoice(env,t=>texts.push(t),()=>{});v.setHandsFree(true);
 v.arm();v.render({active:true,owner:true,dialogue:{present:true}});const rec=env.instances[0];
 v.render({active:false,owner:true,dialogue:{present:false}});rec.onresult({results:[[{transcript:'right'}]]});
 assert.deepEqual(texts,[]);assert.equal(rec.aborted,true);
});
test('a new encounter can speak the same greeting again',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});v.setEnabled(true);
 v.render({active:true,dialogue:{id:1,encounter:1,speech:'Hello there.'}});env.utterances[0].onend();
 v.render({active:true,dialogue:{id:2,encounter:2,speech:'Hello there.'}});assert.equal(env.spoken.length,2);
});
test('background model speech does not interrupt a manual recording',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),texts=[],v=api.createVoice(env,t=>texts.push(t),()=>{});v.setEnabled(true);
 const rec=v.arm();v.render({active:true,owner:true,dialogue:{id:1,speech:'Move right.'}});
 assert.equal(rec.aborted,undefined);assert.deepEqual(env.spoken,[]);
 rec.onresult({results:[[{transcript:'Can you hear me?'}]]});assert.deepEqual(texts,['Can you hear me?']);
 v.render({active:true,owner:true,dialogue:{id:1,speech:'Move right.'}});assert.deepEqual(env.spoken,[]);
});
test('microphone activation does not enable startling automatic speech',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});v.arm();
 v.render({active:true,owner:true,dialogue:{id:1,speech:'Hello there.'}});
 assert.deepEqual(env.spoken.filter(Boolean),[]);
});
test('a visitor microphone toggle cancels capture and late results; can be enabled again',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),texts=[],v=api.createVoice(env,t=>texts.push(t),()=>{});v.arm();const rec=env.instances[0];
 v.setHandsFree(false);rec.onresult({results:[[{transcript:'hello'}]]});env.now=2000;v.render({active:true,owner:true});
 assert.deepEqual(texts,[]);assert.equal(env.instances.length,1);assert.equal(v.status().microphone,false);
 v.arm();assert.equal(env.instances.length,2);assert.equal(v.status().microphone,true);
});
test('voice selection prefers an enhanced English voice and respects an explicit choice',()=>{
 const api=moduleUnderTest(),voices=[{name:'Karen',lang:'en-AU',voiceURI:'k'},{name:'Samantha (Enhanced)',lang:'en-US',voiceURI:'s'},{name:'Chinese',lang:'zh-CN',voiceURI:'c'}];
 assert.equal(api.chooseVoice(voices).voiceURI,'s');assert.equal(api.chooseVoice(voices,'k').voiceURI,'k');
});
test('novelty and robotic system voices are excluded from portrait audition',()=>{
 const api=moduleUnderTest(),voices=[{name:'Bad News',lang:'en-US'},{name:'Zarvox',lang:'en-US'},{name:'Samantha',lang:'en-US'}];
 assert.deepEqual(Array.from(api.portraitVoices(voices),v=>v.name),['Samantha']);
});
test('listening is reported only after the browser actually starts, with error diagnostics',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});const rec=v.arm();
 assert.equal(v.status().listening,false);rec.onstart();assert.equal(v.status().listening,true);
 rec.onerror({error:'not-allowed'});assert.equal(v.diagnostics().voice_error,1);assert.equal(v.diagnostics().voice_listening,false);
});
test('staff can select Chinese recognition without changing English speech output',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});v.setLanguage('zh-CN');const r=v.arm();assert.equal(r.lang,'zh-CN');
 v.setLanguage('unsupported');assert.equal(v.arm().lang,'zh-CN');
});
test('an explicitly opened microphone is not aborted by visual presence flicker',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),texts=[],v=api.createVoice(env,t=>texts.push(t),()=>{});
 const rec=v.arm();rec.onstart();v.render({active:true,owner:true,dialogue:{present:false}});
 rec.onresult({results:[[{transcript:'Can you hear me?'}]]});assert.deepEqual(texts,['Can you hear me?']);
});
test('a browser recognition end cannot reopen the microphone on subsequent state polls',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});const rec=v.arm();rec.onstart();rec.onend();
 for(let i=1;i<=30;i++){env.now=i*1000;v.render({active:true,owner:true,dialogue:{present:true}})}
 assert.equal(env.instances.length,1);assert.equal(v.status().microphone,false);
});
test('a final utterance leaves the microphone off until another explicit click',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});const rec=v.arm();rec.onstart();
 rec.onresult({results:[[{transcript:'Hello'}]]});env.now=5000;v.render({active:true,owner:true});
 assert.equal(v.status().microphone,false);assert.equal(env.instances.length,1);
});
test('speaker diagnostics reflect actual playback and expose errors instead of fake speaking',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});v.setEnabled(true);
 v.render({active:true,owner:true,dialogue:{id:1,speech:'Yes, I can hear you.'}});
 assert.equal(v.status().speaking,false);env.utterances[0].onstart();assert.equal(v.diagnostics().voice_spoken,1);
 env.utterances[0].onerror({error:'not-allowed'});assert.equal(v.diagnostics().voice_speaker_error,1);
});
test('voice timeline distinguishes browser abort from application camera-stale cancellation without transcripts',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});
 const r=v.arm();r.onstart();r.onerror({error:'aborted'});r.onend();
 const events=v.diagnostics().voice_events;
 assert.deepEqual(Array.from(events,e=>e[2]),[1,2,7,8]);assert.equal(events[2][3],6);
 const next=v.arm();next.onstart();env.now=500;v.render({active:true,owner:true,stale:true});
 assert.equal(v.diagnostics().voice_events.at(-1)[2],9);assert.equal(v.diagnostics().voice_events.at(-1)[3],8);
 assert.ok(v.diagnostics().voice_events.every(e=>e.every(Number.isInteger)));
});
test('voice event ring keeps a bounded ordered history across diagnostic polls',()=>{
 const api=moduleUnderTest(),env=clockEnvironment(),v=api.createVoice(env,()=>{},()=>{});
 for(let i=0;i<30;i++){const r=v.arm();r.onstart();r.onend()}
 const events=v.diagnostics().voice_events;assert.equal(events.length,12);
 assert.deepEqual(v.diagnostics().voice_events,events);assert.ok(events[0][0]<events.at(-1)[0]);
});
