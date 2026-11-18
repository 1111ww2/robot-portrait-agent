#!/usr/bin/env node
/* 仅用软件回放经所有者身份验证的原生 /state 记录。回放时拦截全部浏览器请求，不启动服务器、相机、控制器或模型接口。 */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const ROOT=path.resolve(__dirname,'../..'),GUEST=path.join(__dirname,'../src/leica_tracker/guest.html');
const CUES={MOVE_LEFT:['left','A little left'],MOVE_RIGHT:['right','A little right'],MOVE_UP:['top','A little higher'],MOVE_DOWN:['bottom','A little lower'],MOVE_CLOSER:['all','Come closer'],MOVE_FARTHER:['all','A step back']};
function sources(){
 const html=fs.readFileSync(GUEST,'utf8'),script=id=>{const m=html.match(new RegExp('<script id="'+id+'">([\\s\\S]*?)<\\/script>'));assert.ok(m,'Production module missing: '+id);return m[1]};
 const render=html.match(/function render\(s\)\{[\s\S]*?\n\}/),frame=html.match(/async function frame\(\)\{[\s\S]*?\nasync function action/);
 assert.ok(render&&frame,'Production full render/JPEG functions must be extractable');
 return {html,experience:script('guest-experience'),painter:script('guest-painter'),render:render[0],frame:frame[0].replace(/\nasync function action$/,''),sha:crypto.createHash('sha256').update(html).digest('hex')};
}
function loadModule(source){const c={module:{exports:{}}};vm.runInNewContext(source,c);return c.module.exports}
function expected(s,index){
 const controlled=s.active&&s.owner===true&&s.control_fresh===true&&s.runtime&&s.runtime.status==='running'&&s.runtime.reason==='none',m=s.motion,f=s.feedback;
 if(m&&m.active===true&&m.fresh===true){
  assert.ok(controlled&&CUES[m.cue]&&['moving','returning','settling'].includes(m.phase)&&Number.isSafeInteger(m.action_id)&&m.action_id>0
   &&Number.isFinite(m.progress)&&m.progress>=0&&m.progress<=1&&Number.isFinite(m.sample_age_ms)&&m.sample_age_ms>=0&&m.sample_age_ms<=300
   &&f&&f.gesture_id===m.action_id&&f.edge===CUES[m.cue][0]&&f.tone==='red'&&Number.isFinite(f.intensity)&&f.intensity>=0&&f.intensity<=1,'Invalid native motion at trace row '+index);
  return {edge:f.edge,tone:'red',intensity:f.intensity,title:CUES[m.cue][1],cue:m.cue};
 }
 if(controlled&&m&&m.active===false&&s.photo_ready===true&&s.photo_latched===true)return {edge:'all',tone:'green',intensity:1,title:'Get ready for your photo'};
 return null;
}
function checkDOM(snapshot,wanted,label){
 for(const [edge,opacity] of Object.entries(snapshot.edges)){
  const lit=!!(wanted&&wanted.intensity>0&&(wanted.edge==='all'||wanted.edge===edge));
  assert.equal(opacity>0,lit,label+': '+edge+' visibility');
 }
 if(wanted){assert.equal(snapshot.title,wanted.title,label+': direction/photo text');assert.equal(snapshot.color,wanted.tone==='green'?'#39d98a':'#e20612',label+': actual DOM color')}
}
function fixture(src,preview){
 let now=100,timerId=0;const timers=new Map(),nodes=new Map(),actions=[];
 const style=()=>({setProperty(k,v){this[k]=v}}),edges=['left','right','top','bottom'].map(cueEdge=>({dataset:{cueEdge},style:style()}));
 const $=id=>{if(!nodes.has(id))nodes.set(id,{style:style(),hidden:false,textContent:''});return nodes.get(id)};
 const root=$('cue-frame');root.querySelectorAll=()=>edges;
 const c={GuestExperience:loadModule(src.experience),IPadCamera:{viewState:()=>({showVeil:true})},recorder:{status:()=>({busy:false}),cancel(){}},voice:{skipReply(){},render(){}},photoFlow:{update(){},hold(){}},updateVoiceControls(){},document:{hidden:false,body:{classList:{toggle(){}}}},performance:{now:()=>now},$,
  setTimeout:(fn,ms)=>{const id=++timerId;timers.set(id,{at:now+ms,fn});return id},clearTimeout:id=>timers.delete(id),
  localPreview:preview?{}:null,lastState:null,lastStateAt:0,VISUAL_ONLY:true,starting:false,actionPending:false,frameReady:true,lastPaintAt:now,videoStopped:false,visitorStopped:false,lastHeardText:'',heardUntil:0,heardReplyId:0,
  frameSequence:0,frameBusy:false,shownFrames:0,fps:0,fpsSince:now,headers:{},fitImage(){},action:(...v)=>actions.push(v),
  copy:(icon,title,caption)=>{$('title').textContent=title;$('visitor-status').hidden=false}};
 c.GuestPainter=loadModule(src.painter);c.GuestPainter.present=async()=>{};c.cueLight=c.GuestPainter.createFeedbackAnimator(root,c);
 // 故意使用错误的 JPEG 标头，验证原生状态优先于缓存的图像元数据。
 c.fetch=async()=>({status:200,headers:{get:key=>key==='X-Frame-Sequence'?String(c.frameSequence+1):JSON.stringify({edge:'left',tone:'red',intensity:1,gesture_id:-1})},blob:async()=>({})});
 vm.createContext(c);vm.runInContext(src.render+'\n'+src.frame,c);
 return {c,actions,render:s=>c.render(s),frame:()=>c.frame(),snapshot:()=>({title:$('title').textContent,color:root.style['--cue-color'],edges:Object.fromEntries(edges.map(e=>[e.dataset.cueEdge,Number(e.style.opacity||0)]))}),
  advance(ms){now+=ms;for(const [id,timer] of [...timers])if(timer.at<=now){timers.delete(id);timer.fn()}}};
}
function clearProbes(base){
 return [
  ['Stop',{...base,active:false,motion:{active:false,fresh:false},feedback:null,photo_ready:false,photo_latched:false}],
  ['owner lost',{...base,owner:false}],['control expired',{...base,control_fresh:false}],
  ['sample expired',{...base,motion:{...base.motion,sample_age_ms:301}}],
  ['wrong action',{...base,feedback:{...base.feedback,gesture_id:base.motion.action_id+1}}],
  ['hidden document',base,true]
 ];
}
async function validateTrace(rows){
 assert.ok(Array.isArray(rows)&&rows.length,'Trace must be a nonempty JSON array');const src=sources(),preview=fixture(src,true),jpeg=fixture(src,false),directions=new Set();
 const report={success:false,source_sha256:src.sha,observed_rows:rows.length,preview_rows:0,jpeg_rows:0,observed_motion_rows:0,lit_reference_hold_rows:0,observed_photo_rows:0,observed_clear_rows:0,derived_clear_probes:0,expiry_checks:0};let base;
 for(let i=0;i<rows.length;i++){
  const s=rows[i];assert.ok(s&&typeof s==='object'&&Object.prototype.hasOwnProperty.call(s,'motion'),'Native public state missing at row '+i);
  const want=expected(s,i);preview.render(s);checkDOM(preview.snapshot(),want,'preview row '+i);report.preview_rows++;
  jpeg.render(s);await jpeg.frame();checkDOM(jpeg.snapshot(),want,'JPEG row '+i);report.jpeg_rows++;
  if(want&&want.cue){directions.add(want.cue);report.observed_motion_rows++;base=s;if(!s.tracking&&s.phase==='PHYSICAL_REFERENCE_HOLD'&&want.intensity>0)report.lit_reference_hold_rows++}
  else if(want)report.observed_photo_rows++;else report.observed_clear_rows++;
 }
 assert.ok(directions.size===6&&report.observed_photo_rows>0&&report.lit_reference_hold_rows>0,'Incomplete observed trace: need all six measured directions, reference-hold light and photo readiness');
 for(const [label,s,hidden] of clearProbes(base)){
  for(const f of [preview,jpeg]){f.c.document.hidden=false;f.render(base);f.c.document.hidden=!!hidden;f.render(s);await f.frame();checkDOM(f.snapshot(),null,'derived '+label);f.c.document.hidden=false}
  report.derived_clear_probes++;
 }
 preview.render(base);preview.advance(351);checkDOM(preview.snapshot(),null,'actual animator no-update expiry');report.expiry_checks++;
 assert.equal(preview.actions.length+jpeg.actions.length,0,'Native display must not issue Stop during replay');
 report.directions=[...directions];report.success=true;return report;
}
function insideWorkspace(candidate){const absolute=path.resolve(candidate),relative=path.relative(ROOT,absolute);assert.ok(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative),'Replay output must stay inside the workspace');return absolute}
async function browserReplay(rows,destination){
 const debug=message=>{if(process.env.LEICA_REPLAY_DEBUG==='1')console.error('REPLAY '+message)};debug('launch');
 const bounded=async(promise,label)=>{let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error(label+' timed out after 20 seconds')),20000)})])}finally{clearTimeout(timer)}};
 const {chromium}=require('playwright'),src=sources(),browser=await chromium.launch({channel:'msedge',headless:true,timeout:20000});
 const output=insideWorkspace(destination);fs.mkdirSync(output,{recursive:true});const screenshots=[];
 try{
  debug('page');const page=await bounded(browser.newPage(),'Browser page creation');page.setDefaultTimeout(10000);let blockedRequests=0;
  await bounded(page.setViewportSize({width:1024,height:768}),'Viewport setup');
  await bounded(page.route('**/*',route=>{blockedRequests++;return route.abort()}),'Request blocking');
  // 沿用实际页面结构、样式、模块和完整渲染流程；省略入口脚本以免启动现场服务。
  debug('markup');await page.setContent(src.html.slice(0,src.html.indexOf('<script id="guest-camera-geometry">')),{waitUntil:'domcontentloaded',timeout:10000});
  debug('modules');
  await page.addScriptTag({content:src.experience});await page.addScriptTag({content:src.painter});
  await page.addScriptTag({content:`
var $=id=>document.getElementById(id),IPadCamera={viewState:()=>({showVeil:true})},recorder={status:()=>({busy:false}),cancel(){}},voice={skipReply(){},render(){}},photoFlow={update(){},hold(){}},updateVoiceControls=()=>{};
var localPreview={},lastState=null,lastStateAt=0,VISUAL_ONLY=true,starting=false,actionPending=false,frameReady=true,lastPaintAt=performance.now(),videoStopped=false,visitorStopped=false,lastHeardText='',heardUntil=0,heardReplyId=0;
var frameSequence=0,frameBusy=false,shownFrames=0,fps=0,fpsSince=performance.now(),headers={},fitImage=()=>{},stopCalls=0,action=()=>{stopCalls++};
var cueLight=GuestPainter.createFeedbackAnimator($('cue-frame'),globalThis);
var copy=(icon,title,caption)=>{$('visitor-status').hidden=false;$('arrow').textContent=icon;$('title').textContent=title;$('detail').textContent=caption||''};
${src.render}
var frame=(()=>{const setTimeout=()=>{};${src.frame};return frame})();
$('setup').hidden=true;$('visitor-tools').hidden=true;$('viewport').style.width='100%';$('viewport').style.height='100%';
GuestPainter.present=async()=>{};globalThis.fetch=async()=>({status:200,headers:{get:key=>key==='X-Frame-Sequence'?String(frameSequence+1):JSON.stringify({edge:'left',tone:'red',intensity:1,gesture_id:-1})},blob:async()=>({})});
globalThis.replay=async(s,preview)=>{cueLight.clear();localPreview=preview?{}:null;render(s);if(!preview)await frame();return {title:$('title').textContent,color:$('cue-frame').style.getPropertyValue('--cue-color'),edges:Object.fromEntries([...$('cue-frame').querySelectorAll('[data-cue-edge]')].map(e=>[e.dataset.cueEdge,Number(e.style.opacity)])),stopCalls}};
`});
  debug('cases');let checked=0;const representatives=new Map();let photo,recovery;
  for(let i=0;i<rows.length;i++){const want=expected(rows[i],i);if(want&&want.cue&&!representatives.has(want.cue)&&want.intensity>.1)representatives.set(want.cue,rows[i]);if(want&&!want.cue)photo=rows[i];
   if(!recovery&&rows[i].communication&&rows[i].communication.recoverable===true&&['waiting','ready'].includes(rows[i].communication.status))recovery=rows[i]}
  const cases=[...representatives.entries(),...(recovery?[['COMMUNICATION_HOLD',recovery]]:[]),['PHOTO_READY',photo],['STOP',clearProbes([...representatives.values()][0])[0][1]]];
  for(const [label,state] of cases){debug(label);assert.ok(state,'Missing browser replay state '+label);const want=label==='STOP'?null:expected(state,label);
   for(const preview of [true,false]){const snapshot=await page.evaluate(({s,preview})=>globalThis.replay(s,preview),{s:state,preview});checkDOM(snapshot,want,'headless '+label);assert.equal(snapshot.stopCalls,0);checked++;
    const file=path.join(output,label.toLowerCase()+'-'+(preview?'preview':'jpeg')+'.png');await page.screenshot({path:file,timeout:10000});screenshots.push(path.relative(ROOT,file).replaceAll('\\','/'))}
  }
  return {browser:'headless-msedge',browser_checks:checked,blocked_requests:blockedRequests,screenshots};
 }finally{await bounded(browser.close(),'Browser cleanup')}
}
async function main(argv){
 const input=argv[0];if(!input||input.startsWith('-'))throw Error('Usage: node check-motion-light-trace.cjs <ui-state.json> [--screenshots <workspace-directory>]');
 const rest=argv.slice(1);assert.ok(rest.length===0||(rest.length===2&&rest[0]==='--screenshots'),'Unknown replay arguments');
 const rows=JSON.parse(fs.readFileSync(path.resolve(input),'utf8').replace(/^\uFEFF/,'')),report=await validateTrace(rows);
 if(rest.length)Object.assign(report,await browserReplay(rows,rest[1]));console.log(JSON.stringify(report,null,2));return report;
}
module.exports={validateTrace,browserReplay};
if(require.main===module)main(process.argv.slice(2)).catch(error=>{console.error(JSON.stringify({success:false,error:error.message}));process.exitCode=1});
