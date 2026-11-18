const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const driver=require('../scripts/check-live-photo-flow.cjs');
const ROOT=path.resolve(__dirname,'../..');
function fixture(){
 const dir=fs.mkdtempSync(path.join(ROOT,'tmp/live-photo-driver-test-'));
 const image=path.join(dir,'source.png');
 fs.writeFileSync(image,Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aMZQAAAAASUVORK5CYII=','base64'));
 return {dir,config:{origin:'http://127.0.0.1:59001',token:'t'.repeat(32),client:'app-native-validation-owner',fixture_image_path:image,output:path.join(dir,'result')}};
}
test('private browser driver accepts only an ephemeral loopback test owner and workspace files',()=>{
 const {config}=fixture(),got=driver.validateConfig(config);
 assert.equal(got.origin,config.origin);assert.equal(got.output,config.output);
 for(const origin of ['http://127.0.0.1:8767','http://127.0.0.1:8777','http://127.0.0.1:8768','http://127.0.0.1:8779','http://192.168.1.2:59001','https://127.0.0.1:59001','http://localhost:59001','http://x@127.0.0.1:59001','http://127.0.0.1:59001/state'])assert.throws(()=>driver.validateConfig({...config,origin}));
 for(const client of ['ipad-user','app-native-validation-owner\r\nInjected: yes','app-native-validation-owner/other'])assert.throws(()=>driver.validateConfig({...config,client}));
 assert.throws(()=>driver.validateConfig({...config,output:ROOT}));
 assert.throws(()=>driver.validateConfig({...config,fixture_image_path:path.join(ROOT,'PROJECT_MEMORY.md')}));
 assert.throws(()=>driver.validateConfig({...config,fixture_image_is_mirrored:'true'}));
 assert.equal(driver.validateConfig({...config,fixture_image_is_mirrored:true}).fixture_image_is_mirrored,true);
});
test('browser network boundary never forwards control, camera mutations or credentials to another origin',()=>{
 const origin='http://127.0.0.1:59001';
 for(const url of ['/', '/?key=secret','/state','/frame.jpg?after=2','/assets/a.js','/favicon.ico'])assert.equal(driver.allowedRequest(origin,origin+url,'GET'),true);
 assert.equal(driver.allowedRequest(origin,origin+'/photo','POST'),true);
 for(const route of ['/action','/control-heartbeat','/camera/start','/camera/frame','/camera/stop','/physical-feedback','/reference'])assert.equal(driver.allowedRequest(origin,origin+route,'POST'),false);
 assert.equal(driver.allowedRequest(origin,'http://127.0.0.1:59002/state','GET'),false);
 assert.equal(driver.allowedRequest(origin,'https://example.com/asset.js','GET'),false);
 assert.equal(driver.allowedRequest(origin,origin+'/operator-state','GET'),false);
 assert.equal(driver.allowedRequest(origin,origin+'/photo?again=1','POST'),false);
});
test('photo UI driver allows only the fixed authenticated cue audio GETs',()=>{
 const origin='http://127.0.0.1:59001';
 for(const cue of ['MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER','ARRIVED','UNLOCK'])
  assert.equal(driver.allowedRequest(origin,origin+'/cue-audio?cue='+cue,'GET'),true);
 for(const query of ['', '?cue=UNKNOWN','?cue=MOVE_LEFT&cue=MOVE_RIGHT','?cue=MOVE_LEFT&path=.env'])
  assert.equal(driver.allowedRequest(origin,origin+'/cue-audio'+query,'GET'),false);
 assert.equal(driver.allowedRequest(origin,origin+'/cue-audio?cue=MOVE_LEFT','POST'),false);
});
test('JPEG dimensions come from encoded SOF bytes and invalid data is rejected',()=>{
 const jpeg=Buffer.from([255,216,255,192,0,17,8,2,128,3,192,3,1,17,0,2,17,0,3,17,0,255,217]);
 assert.deepEqual(driver.jpegDimensions(jpeg),{width:960,height:640});
 assert.throws(()=>driver.jpegDimensions(Buffer.from('not an image')));
 assert.throws(()=>driver.jpegDimensions(jpeg.subarray(0,10)));
});
test('default CLI remains offline and never consumes private input',async()=>{
 const result=await driver.main([],{print:()=>{},readInput:()=>{throw Error('default read stdin')}});
 assert.equal(result.execute,false);assert.equal(result.no_runtime_interfaces_opened,true);
});
function ownedResources({gotoError=false,contextStaysOpen=false,browserStaysConnected=false,contextElapsed=0}={}){
 const calls=[],limits=[];let connected=true,closed=false,now=1000;const listeners=new Map();
 const context={once:(name,callback)=>listeners.set(name,callback),close:async()=>{
  calls.push('context.close');now+=contextElapsed;
  if(!contextStaysOpen){closed=true;listeners.get('close')?.()}
 }};
 const browser={contexts:()=>closed?[]:[context],isConnected:()=>connected,close:async()=>{
  calls.push('browser.close');if(!browserStaysConnected){connected=false;closed=true;listeners.get('close')?.()}
 }};
 const page={goto:async(url,options)=>{
  calls.push('page.goto');assert.equal(url,'about:blank');assert.equal(options.timeout,5000);
  if(gotoError)throw Error('net::ERR_ABORTED at http://127.0.0.1:50000/?key=sentinel-private-token');
 }};
 return {resources:{page,context,browser},calls,limits,options:{clock:()=>now,bound:async(promise,limit)=>{
  limits.push(limit);return await promise;
 }}};
}
test('owned cleanup closes the explicit context before browser and proves both closed',async()=>{
 const f=ownedResources(),report={success:true};
 await driver.closeOwnedBrowser(f.resources,report,f.options);
 assert.deepEqual(f.calls,['page.goto','context.close','browser.close']);
 assert.equal(report.cleanup_complete,true);assert.equal(report.context_closed_after_cleanup,true);
 assert.equal(report.browser_connected_after_cleanup,false);assert.equal(report.success,true);
 assert.deepEqual(report.cleanup_stages.map(row=>[row.stage,row.success]),[
  ['page_goto_about_blank',true],['context_close',true],['browser_close',true]]);
 assert.deepEqual(f.limits,[6000,10000,10000]);
});
test('unload failure still closes owned resources and logs only typed safe failure',async()=>{
 const f=ownedResources({gotoError:true}),report={success:true};
 await driver.closeOwnedBrowser(f.resources,report,f.options);
 assert.deepEqual(f.calls,['page.goto','context.close','browser.close']);
 assert.equal(report.success,false);assert.equal(report.cleanup_complete,false);
 assert.equal(report.cleanup_error_stage,'page_goto_about_blank');
 assert.equal(report.cleanup_stages[0].kind,'navigation_aborted');
 assert.equal(report.browser_connected_after_cleanup,false);
 assert.equal(JSON.stringify(report).includes('sentinel-private-token'),false);
 assert.equal(JSON.stringify(report).includes('127.0.0.1'),false);
});
test('cleanup does not accept resolved close calls while owned resources remain open',async()=>{
 for(const changes of [{contextStaysOpen:true},{browserStaysConnected:true}]){
  const f=ownedResources(changes),report={success:true};
  await driver.closeOwnedBrowser(f.resources,report,f.options);
  assert.deepEqual(f.calls,['page.goto','context.close','browser.close']);
  assert.equal(report.success,false);assert.equal(report.cleanup_complete,false);
  assert.equal(report.cleanup_stages.find(row=>!row.success).kind,'owned_resource_open');
 }
});
test('context and browser retirement share the existing ten-second teardown deadline',async()=>{
 const f=ownedResources({contextElapsed:6000}),report={success:true};
 await driver.closeOwnedBrowser(f.resources,report,f.options);
 assert.deepEqual(f.limits,[6000,10000,4000]);
 assert.equal(report.cleanup_complete,true);
});
test('production photo countdown captures one JPEG and displays returned QR on a software-only ephemeral server',
 {skip:process.env.LEICA_LIVE_PHOTO_BROWSER_TEST!=='1',timeout:60000},async()=>{
 const {dir,config}=fixture(),html=fs.readFileSync(path.join(ROOT,'vision_mvp/src/leica_tracker/guest.html'));
 const received=[];let sharePosts=0,reads=0;
 const qr=fs.readFileSync(config.fixture_image_path).toString('base64');
 const server=http.createServer((request,response)=>{
  const url=new URL(request.url,'http://127.0.0.1'),reply=(status,type,body)=>{response.writeHead(status,{'Content-Type':type,'Content-Length':Buffer.byteLength(body),'Cache-Control':'no-store'});response.end(body)};
  const authenticated=request.headers['x-guest-token']===config.token&&request.headers['x-guest-client']===config.client;
  received.push({method:request.method,path:url.pathname,authenticated});
  if(url.pathname==='/')return reply(200,'text/html',html);
  if(!authenticated)return reply(403,'text/plain','fixture request identity rejected');
  if(url.pathname==='/state'){
   reads++;return reply(200,'application/json',JSON.stringify({active:true,owner:true,control_fresh:true,camera_source:'ipad',camera_connected:true,tracking:true,stale:false,phase:'ARRIVED',photo_ready:true,photo_latched:true,motion:{active:false,fresh:false},feedback:{edge:'all',tone:'green',intensity:1,gesture_id:1},runtime:{status:'running',reason:'none'},dialogue:{},composition:{}}));
  }
  if(url.pathname==='/frame.jpg'){response.writeHead(204,{'Content-Length':'0'});return response.end()}
  if(url.pathname==='/photo'&&request.method==='POST'){
   assert.equal(request.headers.origin,config.origin);assert.equal(request.headers['content-type'],'image/jpeg');
   const parts=[];request.on('data',part=>parts.push(part));request.on('end',()=>{sharePosts++;assert.deepEqual(driver.jpegDimensions(Buffer.concat(parts)),{width:96,height:64});reply(200,'application/json',JSON.stringify({qr_data_url:'data:image/png;base64,'+qr,expires_seconds:300}))});return;
  }
  return reply(404,'text/plain','missing');
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));config.origin='http://127.0.0.1:'+server.address().port;
 // 使用真正的图像输入和真正的视频/画布/JPEG编码。端点是一个标记的测试夹具；此测试不接受原生/AI。
 const {chromium}=require('playwright'),browser=await chromium.launch({channel:'msedge',headless:true});
 try{
  const page=await browser.newPage();await page.setContent('<canvas id="c" width="96" height="64"></canvas>');
  const source=await page.evaluate(()=>{const c=document.querySelector('canvas'),x=c.getContext('2d');x.fillStyle='#ff0000';x.fillRect(0,0,48,64);x.fillStyle='#0000ff';x.fillRect(48,0,48,64);return c.toDataURL('image/png').split(',')[1]});fs.writeFileSync(config.fixture_image_path,Buffer.from(source,'base64'));
 }finally{await browser.close()}
 try{
  config.fixture_image_is_mirrored=true;
  const result=await driver.runBrowser(driver.validateConfig(config));
  assert.equal(result.success,true);assert.deepEqual(result.countdown,[3,2,1]);assert.equal(result.photo_posts,1);assert.equal(sharePosts,1);assert.ok(reads>1);
  assert.deepEqual(result.capture_dimensions,{width:96,height:64});assert.equal(result.forwarded_control_requests,0);
  assert.equal(result.fixture_image_is_mirrored,true);
  assert.ok(result.capture_edge_rgb.left[0]>220&&result.capture_edge_rgb.left[2]<30,'Encoded left side must match normalized red-left input');
  assert.ok(result.capture_edge_rgb.right[2]>220&&result.capture_edge_rgb.right[0]<30,'Encoded right side must match normalized blue-right input');
  assert.ok(fs.existsSync(path.join(config.output,'qr.png')));assert.ok(fs.existsSync(path.join(config.output,'photo-ui.png')));
  assert.deepEqual(received.filter(x=>x.method==='POST'),[{method:'POST',path:'/photo',authenticated:true}]);
  assert.deepEqual(received.filter(x=>!x.authenticated),[],'No request may escape the context HTTP boundary during cleanup');
  assert.equal(JSON.stringify(result).includes(config.token),false);assert.equal(JSON.stringify(result).includes(config.client),false);assert.equal(JSON.stringify(result).includes(config.origin),false);
 }finally{await new Promise(resolve=>server.close(resolve))}
});
