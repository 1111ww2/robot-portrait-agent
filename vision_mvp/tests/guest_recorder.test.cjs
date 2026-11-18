const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function setup(transcribe=async()=>({text:'Hello'})){
 const code=fs.readFileSync(require('node:path').join(__dirname,'../src/leica_tracker/guest.html'),'utf8').match(/<script id="guest-recorder">([\s\S]*?)<\/script>/)[1];
 const context={module:{exports:{}}};vm.runInNewContext(code,context);
 const env={isSecureContext:true,Blob,AbortController,now:0,instances:[],stops:0,interval:null};
 env.performance={now:()=>env.now};env.setInterval=f=>(env.interval=f,1);env.clearInterval=()=>env.interval=null;
 env.navigator={mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop:()=>env.stops++}]})}};
 env.MediaRecorder=class{static isTypeSupported(){return true}constructor(){this.state='inactive';this.mimeType='audio/mp4';env.instances.push(this)}start(){this.state='recording'}stop(){this.state='inactive';this.ondataavailable({data:new Blob(['audio'])});return this.onstop()}};
 const texts=[],statuses=[];const recorder=context.module.exports.create(env,transcribe,t=>texts.push(t),s=>statuses.push(s));return {recorder,env,texts,statuses};
}
test('manual recorder stays open without browser speech recognition and submits after finish',async()=>{
 const {recorder,env,texts}=setup();await recorder.start();assert.equal(recorder.status().listening,true);assert.equal(env.instances.length,1);
 env.now=2000;env.interval();assert.equal(recorder.status().listening,true);
 recorder.finish();await new Promise(setImmediate);assert.deepEqual(texts,['Hello']);assert.equal(recorder.status().busy,false);assert.equal(env.stops,1);
});
test('explicit cancellation releases input and rejects delayed transcription',async()=>{
 let resolve;const pending=new Promise(r=>resolve=r);const {recorder,texts}=setup(()=>pending);await recorder.start();recorder.finish();recorder.cancel();resolve({text:'Old request'});await new Promise(setImmediate);assert.deepEqual(texts,[]);assert.equal(recorder.status().busy,false);
});
test('recording has a hard duration ceiling and never restarts itself',async()=>{
 const {recorder,env}=setup();await recorder.start();env.now=15001;env.interval();await new Promise(setImmediate);assert.equal(recorder.status().busy,false);assert.equal(env.instances.length,1);
});
test('permission failure is shown rather than a false listening state',async()=>{
 const {recorder,env,statuses}=setup();env.navigator.mediaDevices.getUserMedia=async()=>{throw Error('denied')};await recorder.start();assert.equal(recorder.status().busy,false);assert.match(statuses.at(-1),/permission/);
});
