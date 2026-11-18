const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
function api(){const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');const block=html.match(/<script id="guest-camera-geometry">([\s\S]*?)<\/script>/);assert.ok(block,'shared camera geometry');const context={module:{exports:{}}};vm.runInNewContext(block[1],context);return context.module.exports}
const landscape={innerWidth:1024,innerHeight:768,navigator:{platform:'MacIntel',maxTouchPoints:5}};
test('auto corrects the portrait iPad source in landscape, but preserves normal landscape and other devices',()=>{
 const g=api();assert.equal(g.rotation({videoWidth:720,videoHeight:1280},landscape,'auto'),90);
 assert.equal(g.rotation({videoWidth:1280,videoHeight:720},landscape,'auto'),0);
 assert.equal(g.rotation({videoWidth:720,videoHeight:1280},{...landscape,innerWidth:768,innerHeight:1024},'auto'),0);
 assert.equal(g.rotation({videoWidth:720,videoHeight:1280},{...landscape,navigator:{}},'auto'),0);
 assert.equal(g.rotation({videoWidth:720,videoHeight:1280},landscape,'0'),0);
 assert.equal(g.rotation({videoWidth:720,videoHeight:1280},landscape,'270'),270);
});
test('raw clockwise rotation then mirror puts a right-facing mirrored head upright, without cropping',()=>{
 const g=api(),p=g.plan(720,1280,90,768);
 assert.equal(p.width,768);assert.equal(p.height,432);
 const [a,b,c,d,e,f]=p.transform;
 const map=(x,y)=>[a*x+c*y+e,b*x+d*y+f];
 assert.deepEqual(map(0,0),[0,0]);assert.deepEqual(map(432,768),[768,432]);
 // 头部在原始肖像框架中位于左侧（镜像后右侧）。
 assert.ok(map(100,300)[1]<map(300,300)[1]);
});
test('all quarter turns cover the full output and mirror once',()=>{
 const g=api();for(const rotation of [0,90,180,270]){
  const p=g.plan(720,1280,rotation,768),[a,b,c,d,e,f]=p.transform;
  const corners=[[0,0],[p.drawWidth,0],[0,p.drawHeight],[p.drawWidth,p.drawHeight]].map(([x,y])=>[a*x+c*y+e,b*x+d*y+f]);
  assert.equal(Math.min(...corners.map(x=>x[0])),0);assert.equal(Math.max(...corners.map(x=>x[0])),p.width);
  assert.equal(Math.min(...corners.map(x=>x[1])),0);assert.equal(Math.max(...corners.map(x=>x[1])),p.height);
  assert.equal(a*d-b*c,-1);
 }
});
