#!/usr/bin/env node
/* 按需使用上层进程管理的临时服务器测试实际拍照界面。开始、停止、控制心跳和原生执行均由上层负责；本脚本不打开设备相机、人物照片或外部网址。 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const ROOT=path.resolve(__dirname,'../..'),RESERVED_PORTS=new Set([8767,8768,8777,8779,15900,16080,39999,40001,40002,40004,40011]);
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
async function bounded(promise,milliseconds){
 let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Photo driver operation deadline exceeded')),milliseconds)})])}finally{clearTimeout(timer)}
}
function demand(value){if(!value)throw Error('Invalid private photo-driver configuration')}
function inside(candidate){
 demand(typeof candidate==='string'&&candidate.length>0);const absolute=path.resolve(candidate),relative=path.relative(ROOT,absolute);
 demand(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative));
 let existing=absolute;while(!fs.existsSync(existing)){const parent=path.dirname(existing);demand(parent!==existing);existing=parent}
 const real=fs.realpathSync(existing),realRelative=path.relative(ROOT,real);
 demand(!realRelative.startsWith('..')&&!path.isAbsolute(realRelative));return absolute;
}
function validateConfig(input){
 demand(input&&typeof input==='object'&&!Array.isArray(input));
 const origin=new URL(input.origin);demand(origin.protocol==='http:'&&origin.hostname==='127.0.0.1'&&origin.username===''&&origin.password===''
  &&origin.origin===input.origin&&Number(origin.port)>=1024&&Number(origin.port)<=65535&&!RESERVED_PORTS.has(Number(origin.port)));
 demand(typeof input.token==='string'&&/^[A-Za-z0-9_-]{24,128}$/.test(input.token));
 demand(typeof input.client==='string'&&input.client.length<=128&&/^app-native-validation-owner(?:-[A-Za-z0-9_-]+)?$/.test(input.client));
 const fixture_image_is_mirrored=input.fixture_image_is_mirrored===undefined?false:input.fixture_image_is_mirrored;demand(typeof fixture_image_is_mirrored==='boolean');
 const fixture_image_path=inside(input.fixture_image_path),output=inside(input.output),stat=fs.statSync(fixture_image_path);
 demand(stat.isFile()&&stat.size>0&&stat.size<=8*1024*1024&&/\.(png|jpe?g)$/i.test(fixture_image_path)&&!fs.existsSync(output));
 return {origin:origin.origin,token:input.token,client:input.client,fixture_image_path,fixture_image_is_mirrored,output};
}
function allowedRequest(origin,url,method){
 try{const parsed=new URL(url);if(parsed.origin!==origin)return false;
  if(method==='GET'&&parsed.pathname==='/cue-audio')return [...parsed.searchParams.keys()].length===1
   &&parsed.searchParams.getAll('cue').length===1
   &&['MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER','ARRIVED','UNLOCK',
      'INTRO_HELLO','WAIT_MOMENT','WAIT_SECOND','WAIT_FRAME','WAIT_ANGLE','WAIT_LOOK','WAIT_SPOT',
      'COUNTDOWN_THREE','COUNTDOWN_TWO','COUNTDOWN_ONE'].includes(parsed.searchParams.get('cue'));
  if(method==='GET')return ['/', '/state','/frame.jpg','/favicon.ico','/leica-logo.svg'].includes(parsed.pathname)||parsed.pathname.startsWith('/assets/');
  return method==='POST'&&parsed.pathname==='/photo'&&parsed.search==='';
 }catch{return false}
}
function jpegDimensions(bytes){
 if(!Buffer.isBuffer(bytes)||bytes.length<4||bytes[0]!==255||bytes[1]!==216)throw Error('Invalid captured JPEG');
 let offset=2;
 while(offset+4<=bytes.length){
  if(bytes[offset++]!==255)throw Error('Invalid captured JPEG');while(bytes[offset]===255)offset++;
  const marker=bytes[offset++];if(marker===217||marker===218)break;
  if(marker===1||marker>=208&&marker<=215)continue;
  if(offset+2>bytes.length)break;const length=bytes.readUInt16BE(offset);
  if(length<2||offset+length>bytes.length)break;
  if([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker)){
   if(length<8)break;const height=bytes.readUInt16BE(offset+3),width=bytes.readUInt16BE(offset+5);
   if(!width||!height)break;return {width,height};
  }
  offset+=length;
 }
 throw Error('Captured JPEG has no complete dimensions');
}
async function closeOwnedBrowser({page,context,browser},report,{bound=bounded,clock=()=>performance.now()}={}){
 report.cleanup_stages=[];
 let contextClosed=!context||Boolean(browser&&!browser.contexts().includes(context));
 if(context&&!contextClosed)context.once('close',()=>{contextClosed=true});
 const cleanup=async(stage,action,limit)=>{
  const started=clock();
  try{
   await bound(action(),limit);
   report.cleanup_stages.push({stage,success:true,elapsed_ms:Math.max(0,Math.round(clock()-started))});
  }catch(error){
   const message=String(error?.message||'');
   const kind=/operation deadline exceeded/.test(message)?'outer_deadline':
    /net::ERR_ABORTED/.test(message)?'navigation_aborted':/net::ERR_FAILED/.test(message)?'navigation_failed':
    /Owned (context|browser) remained/.test(message)?'owned_resource_open':
    /Target.*closed|has been closed|TargetClosed/.test(message)?'target_closed':
    /Timeout|timeout|timed out/.test(message)?'timeout':'other';
   const error_type=['Error','TimeoutError','TargetClosedError'].includes(error?.name)?error.name:'UnknownError';
   report.cleanup_stages.push({stage,success:false,elapsed_ms:Math.max(0,Math.round(clock()-started)),error_type,kind});
   report.cleanup_error=true;report.cleanup_error_stage??=stage;report.success=false;
  }
 };
 // 页面卸载期间保留浏览器上下文的 HTTP 拦截规则。
 if(page)await cleanup('page_goto_about_blank',async()=>{
  await page.goto('about:blank',{waitUntil:'domcontentloaded',timeout:5000});
  await new Promise(resolve=>setTimeout(resolve,50));
 },6000);
 // 先关闭显式创建的上下文，再关闭浏览器。两次关闭共用十秒预算；
 // 即使其中一步失败，仍尝试清理其余资源，并将本次验证记为失败。
 const deadline=clock()+10000,remaining=()=>Math.max(1,Math.min(10000,deadline-clock()));
 if(context)await cleanup('context_close',async()=>{
  await context.close({reason:'Photo driver completed'});
  contextClosed=contextClosed||Boolean(browser&&!browser.contexts().includes(context));
  if(!contextClosed)throw Error('Owned context remained open');
 },remaining());
 if(browser)await cleanup('browser_close',async()=>{
  await browser.close({reason:'Photo driver completed'});
  if(browser.isConnected())throw Error('Owned browser remained connected');
 },remaining());
 contextClosed=contextClosed||Boolean(context&&browser&&!browser.contexts().includes(context));
 report.context_closed_after_cleanup=contextClosed;
 report.browser_connected_after_cleanup=browser?browser.isConnected():false;
 report.cleanup_complete=!report.cleanup_error&&contextClosed&&!report.browser_connected_after_cleanup;
 if(!report.cleanup_complete)report.success=false;
}
async function runBrowser(config){
 const report={success:false,input_mode:'public_fixture_video',fixture_image_is_mirrored:config.fixture_image_is_mirrored,browser:'headless-msedge',photo_posts:0,photo_status:null,
  forwarded_control_requests:0,rejected_mutation_requests:0,rejected_other_requests:0,route_errors:0,page_errors:0,read_requests:0};
 let browser,context,page,photoBytes,stage='fixture_read';
 try{
  fs.mkdirSync(config.output,{recursive:false});const fixture=fs.readFileSync(config.fixture_image_path),mime=/\.png$/i.test(config.fixture_image_path)?'image/png':'image/jpeg';
  const expectedHTML=fs.readFileSync(path.join(ROOT,'vision_mvp/src/leica_tracker/guest.html'));
  stage='browser_launch';const {chromium}=require('playwright');browser=await chromium.launch({channel:'msedge',headless:true,timeout:20000});
  context=await bounded(browser.newContext({viewport:{width:1024,height:768},serviceWorkers:'block'}),20000);
  page=await bounded(context.newPage(),20000);page.setDefaultTimeout(10000);
  page.on('pageerror',()=>{report.page_errors++});
  page.on('response',response=>{try{const url=new URL(response.url());if(url.origin===config.origin&&url.pathname==='/photo')report.photo_status=response.status()}catch{}});
  // 在上下文层拦截请求，以覆盖旧页面卸载时的保活请求；仅绑定页面会过早丢失拦截规则。
  await context.route('**/*',async route=>{
   const request=route.request(),method=request.method(),url=request.url();
   if(!allowedRequest(config.origin,url,method)){
    if(method!=='GET')report.rejected_mutation_requests++;else report.rejected_other_requests++;
    return route.abort().catch(()=>{});
   }
   try{
    if(method==='POST'){
     report.photo_posts++;const bytes=request.postDataBuffer();
     if(!bytes||bytes.length>4*1024*1024)throw Error('Invalid bounded photo upload');
     const dimensions=jpegDimensions(bytes);if(photoBytes)throw Error('More than one photo upload');
     photoBytes=Buffer.from(bytes);report.capture_dimensions=dimensions;report.capture_bytes=bytes.length;report.capture_sha256=digest(bytes);
    }else report.read_requests++;
    return await route.continue({headers:{...request.headers(),'x-guest-token':config.token,'x-guest-client':config.client,
     ...(method==='POST'?{'origin':config.origin}:{})}});
   }catch{report.route_errors++;return route.abort().catch(()=>{})}
  });
  stage='production_page';const response=await page.goto(config.origin+'/?key='+encodeURIComponent(config.token),{waitUntil:'domcontentloaded',timeout:20000});
  if(!response||response.status()!==200)throw Error('Production page unavailable');
  report.served_html_sha256=digest(await bounded(response.body(),10000));if(report.served_html_sha256!==digest(expectedHTML))throw Error('Served production source differs');
  stage='preview_setup';report.preview_dimensions=await bounded(page.evaluate(async ({dataURL,mirrored})=>{
   const numbers=[];const observer=new MutationObserver(()=>{
    const value=document.getElementById('photo-number').textContent;
    if(/^[123]$/.test(value)&&numbers.at(-1)?.number!==Number(value))numbers.push({number:Number(value),at:performance.now()});
   });observer.observe(document.getElementById('photo-countdown'),{subtree:true,childList:true,characterData:true,attributes:true});
   globalThis.__photoDriverEvidence={numbers,observer};
   const image=new Image();image.src=dataURL;await image.decode();
   if(!image.naturalWidth||!image.naturalHeight||Math.max(image.naturalWidth,image.naturalHeight)>4096)throw Error('Invalid fixture dimensions');
   const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
   const context=canvas.getContext('2d'),paint=()=>{
    // 上传并归一化的画面已做镜像处理。只对这个原始视频输入还原镜像，再由正式的 captureVideo 镜像一次。
    context.setTransform(mirrored?-1:1,0,0,1,mirrored?canvas.width:0,0);context.drawImage(image,0,0);
   };paint();
   const stream=canvas.captureStream(15),video=document.createElement('video');video.muted=true;video.autoplay=true;video.playsInline=true;video.srcObject=stream;
   globalThis.__photoDriverEvidence.refresh=setInterval(paint,50);
   document.getElementById('viewport').insertBefore(video,document.getElementById('crop'));await video.play();
   if(video.readyState<2)await new Promise((resolve,reject)=>{video.addEventListener('loadeddata',resolve,{once:true});setTimeout(()=>reject(Error('Fixture preview unavailable')),5000)});
   globalThis.__photoDriverEvidence.stream=stream;localPreview=video;video.addEventListener('resize',fitImage);document.getElementById('camera').hidden=true;fitImage();
   // 这里只关闭设置界面。拍照流程与状态计时器照常运行，不触发开始、停止或心跳。
   document.getElementById('enter').click();return {width:video.videoWidth,height:video.videoHeight};
  },{dataURL:'data:'+mime+';base64,'+fixture.toString('base64'),mirrored:config.fixture_image_is_mirrored}),10000);
  stage='photo_lifecycle';await page.waitForFunction(()=>{
   const qr=document.getElementById('photo-qr'),result=document.getElementById('photo-result');
   const photo=document.getElementById('photo-preview');
   return !result.hidden&&!qr.hidden&&qr.complete&&qr.naturalWidth>0&&photo.complete&&photo.naturalWidth>0&&document.getElementById('photo-share-status').textContent==='Scan to download · Available for 5 minutes';
  },null,{timeout:30000});
  const observed=await bounded(page.evaluate(()=>{
   const photo=document.getElementById('photo-preview'),canvas=document.createElement('canvas');canvas.width=photo.naturalWidth;canvas.height=photo.naturalHeight;
   const context=canvas.getContext('2d');context.drawImage(photo,0,0);
   const rgb=x=>[...context.getImageData(Math.floor(x),Math.floor(canvas.height/2),1,1).data].slice(0,3);
   return {numbers:globalThis.__photoDriverEvidence.numbers,qr:document.getElementById('photo-qr').src,qrVisible:!document.getElementById('photo-qr').hidden,
    previewWidth:photo.naturalWidth,previewHeight:photo.naturalHeight,edgeRGB:{left:rgb(canvas.width/4),right:rgb(canvas.width*3/4)},
    state:{active:lastState?.active,owner:lastState?.owner,photo_latched:lastState?.photo_latched,photo_ready:lastState?.photo_ready}};
  }),10000);
  report.countdown=observed.numbers.map(item=>item.number);report.countdown_intervals_ms=observed.numbers.slice(1).map((item,index)=>Math.round(item.at-observed.numbers[index].at));
  report.photo_state=observed.state;report.qr_visible=observed.qrVisible;report.capture_edge_rgb=observed.edgeRGB;
  if(JSON.stringify(report.countdown)!=='[3,2,1]'||report.photo_posts!==1||report.photo_status!==200||!photoBytes
   ||report.page_errors||report.route_errors||Object.values(observed.state).some(value=>value!==true))throw Error('Incomplete production photo lifecycle');
  if(report.capture_dimensions.width!==observed.previewWidth||report.capture_dimensions.height!==observed.previewHeight
   ||report.capture_dimensions.width!==report.preview_dimensions.width||report.capture_dimensions.height!==report.preview_dimensions.height)throw Error('Capture dimensions differ');
  if(!/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(observed.qr))throw Error('Invalid rendered QR image');
  stage='evidence';const qr=Buffer.from(observed.qr.split(',')[1],'base64');
  if(!qr.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))throw Error('QR image is not PNG');
  fs.writeFileSync(path.join(config.output,'capture.jpg'),photoBytes,{flag:'wx'});fs.writeFileSync(path.join(config.output,'qr.png'),qr,{flag:'wx'});
  await page.screenshot({path:path.join(config.output,'photo-ui.png'),timeout:10000});
  report.qr_png_sha256=digest(qr);report.artifacts={capture:'capture.jpg',qr:'qr.png',screenshot:'photo-ui.png'};report.success=true;
 }catch{report.error_stage=stage}
 finally{
  await closeOwnedBrowser({page,context,browser},report);
  if(fs.existsSync(config.output))fs.writeFileSync(path.join(config.output,'result.json'),JSON.stringify(report,null,2));
 }
 return report;
}
function readPrivateInput(){return new Promise((resolve,reject)=>{
 let value='';const timer=setTimeout(()=>{process.stdin.destroy();reject(Error('Private configuration timeout'))},10000);
 process.stdin.setEncoding('utf8');process.stdin.on('data',part=>{value+=part;if(Buffer.byteLength(value)>16384){clearTimeout(timer);process.stdin.destroy();reject(Error('Private configuration too large'))}});
 process.stdin.on('end',()=>{clearTimeout(timer);resolve(value)});process.stdin.on('error',()=>{clearTimeout(timer);reject(Error('Private configuration unavailable'))});
})}
async function main(argv,{print=value=>console.log(JSON.stringify(value,null,2)),readInput=readPrivateInput}={}){
 if(argv.length===0){const report={execute:false,no_runtime_interfaces_opened:true,private_input_required_for_execute:true};print(report);return report}
 if(argv.length!==1||argv[0]!=='--execute')throw Error('Invalid driver arguments');
 const config=validateConfig(JSON.parse(await readInput())),report=await runBrowser(config);print(report);return report;
}
module.exports={validateConfig,allowedRequest,jpegDimensions,closeOwnedBrowser,runBrowser,main};
if(require.main===module)main(process.argv.slice(2)).then(report=>{if(report.success===false)process.exitCode=1}).catch(()=>{
 console.log(JSON.stringify({success:false,error_stage:'configuration',no_private_input_echoed:true}));process.exitCode=1;
});
