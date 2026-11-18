const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const checker=path.join(__dirname,'../scripts/check-motion-light-trace.cjs');
const cues=['MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER'];
function trace(){return cues.map((cue,i)=>({active:true,owner:true,control_fresh:true,phase:'PHYSICAL_REFERENCE_HOLD',tracking:false,stale:true,runtime:{status:'running',reason:'none'},motion:{active:true,fresh:true,cue,phase:'moving',action_id:i+1,progress:.6,sample_age_ms:10},feedback:{edge:['left','right','top','bottom','all','all'][i],tone:'red',intensity:.6,gesture_id:i+1}})).concat({active:true,owner:true,control_fresh:true,phase:'ARRIVED',tracking:false,stale:true,runtime:{status:'running',reason:'none'},motion:{active:false,fresh:false},feedback:null,photo_ready:true,photo_latched:true})}
const recoveryRow=()=>JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/native-communication-hold.json'),'utf8'));

test('actual recorded recovery row is dark in preview and JPEG replay without issuing Stop',async()=>{
 const rows=trace();rows.splice(1,0,recoveryRow());
 const result=await require(checker).validateTrace(rows);
 assert.equal(result.success,true);assert.equal(result.observed_clear_rows,1);
 assert.equal(result.preview_rows,8);assert.equal(result.jpeg_rows,8);assert.equal(result.derived_clear_probes,6);
});

test('browser replay bootstrap supports the same recorded recovery row without browser or network interfaces',async()=>{
 const style=()=>({setProperty(k,v){this[k]=v},getPropertyValue(k){return this[k]||''}});
 const edges=['left','right','top','bottom'].map(cueEdge=>({dataset:{cueEdge},style:style()})),nodes=new Map();
 const get=id=>{if(!nodes.has(id))nodes.set(id,{style:style(),hidden:false,textContent:''});return nodes.get(id)};
 get('cue-frame').querySelectorAll=()=>edges;
 const pageContext=vm.createContext({document:{hidden:false,getElementById:get,body:{classList:{toggle(){}}}},
  performance:{now:()=>100},setTimeout:()=>1,clearTimeout(){},requestAnimationFrame:()=>1,cancelAnimationFrame(){}});
 const shown=[];let closed=false;
 const page={setDefaultTimeout(){},async setViewportSize(){},async route(){},async setContent(){},
  async addScriptTag({content}){vm.runInContext(content,pageContext)},
  async evaluate(fn,arg){pageContext.replayArgument=arg;const result=await vm.runInContext('('+fn.toString()+')(replayArgument)',pageContext);shown.push({state:arg.s,result});return result},
  async screenshot(){}};
 const browser={async newPage(){return page},async close(){closed=true}};
 const fakeRequire=name=>name==='playwright'?{chromium:{async launch(){return browser}}}:
  name==='node:fs'?{...fs,mkdirSync(){}}:require(name);
 const context={module:{exports:{}},require:fakeRequire,__dirname:path.dirname(checker),__filename:checker,
  process:{env:{}},setTimeout,clearTimeout,console};
 vm.runInNewContext(fs.readFileSync(checker,'utf8'),context);
 const result=await context.module.exports.browserReplay([recoveryRow(),...trace()],path.join(__dirname,'../../tmp/offline-recovery-replay-unused'));
 const paused=shown.filter(item=>item.state.communication&&item.state.communication.status==='waiting');
 assert.equal(paused.length,2,'both preview and JPEG paths must inspect an observed recovery row');
 for(const {result:snapshot} of paused){assert.equal(snapshot.title,'Reconnecting');assert.equal(snapshot.stopCalls,0);assert.ok(Object.values(snapshot.edges).every(value=>!(value>0)))}
 assert.equal(result.browser_checks,18);assert.equal(closed,true);
});
test('replay checker uses actual render and JPEG source and confirms measured red cues and green confirmation',async()=>{
 assert.ok(fs.existsSync(checker),'Reusable trace checker must exist');
 const result=await require(checker).validateTrace(trace());
 assert.equal(result.success,true);assert.equal(result.observed_motion_rows,6);assert.equal(result.lit_reference_hold_rows,6);
 assert.equal(result.observed_photo_rows,1);assert.equal(result.directions.length,6);assert.equal(result.preview_rows,7);assert.equal(result.jpeg_rows,7);
 assert.equal(result.derived_clear_probes,6);assert.match(result.source_sha256,/^[a-f0-9]{64}$/);
});
test('replay checker fails an incomplete observed trace instead of synthesizing successful directions',async()=>{
 assert.ok(fs.existsSync(checker),'Reusable trace checker must exist');
 await assert.rejects(require(checker).validateTrace(trace().slice(0,1)),/Incomplete observed trace/);
});
test('replay checker rejects a native cue with mismatched measured light',async()=>{
 assert.ok(fs.existsSync(checker),'Reusable trace checker must exist');
 const input=trace();input[0].feedback.edge='right';
 await assert.rejects(require(checker).validateTrace(input),/Invalid native motion/);
});
