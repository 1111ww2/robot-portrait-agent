const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
const block=html.match(/<script id="guest-painter">([\s\S]*?)<\/script>/);
function player(){
  assert.ok(block,'Guest painter component must exist');
  const context={module:{exports:{}},globalThis:{}};
  vm.runInNewContext(block[1],context);
  return context.module.exports;
}
function canvas(){
  let width=0,height=0;
  const surface={pixels:null,clears:0,getContext(){return {drawImage(image){surface.pixels=image.label}}}};
  Object.defineProperties(surface,{width:{get:()=>width,set:v=>{width=v;surface.pixels=null;surface.clears++}},height:{get:()=>height,set:v=>{height=v;surface.pixels=null;surface.clears++}}});
  return surface;
}
function image(label){return {label,width:768,height:432,close(){this.closed=true}}}
test('successive same-size frames never clear the visible canvas',()=>{
  const p=player(),c=canvas();p.paint(c,image('first'));
  const clears=c.clears;
  p.paint(c,image('second'));
  assert.equal(c.pixels,'second');assert.equal(c.clears,clears);
});
test('pending background decode keeps previous pixels until the new frame is ready',async()=>{
  const p=player(),c=canvas();p.paint(c,image('previous'));
  let resolve;const pending=new Promise(r=>resolve=r),next=image('new');
  const task=p.present(c,{}, {createImageBitmap:()=>pending});
  assert.equal(c.pixels,'previous');
  resolve(next);await task;
  assert.equal(c.pixels,'new');assert.equal(next.closed,true);
});
test('both decode paths failing retain the last good picture and revoke the temporary URL',async()=>{
  const p=player(),c=canvas();p.paint(c,image('last-good'));
  let revoked=false;
  const env={createImageBitmap:async()=>{throw Error('decode')},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL:()=>{revoked=true}},Image:class{decode(){return Promise.reject(Error('bad JPEG'))}}};
  await assert.rejects(p.present(c,{},env));
  assert.equal(c.pixels,'last-good');assert.equal(revoked,true);
});

function border(){
  const nodes=['left','right','top','bottom'].map(side=>({dataset:{cueEdge:side},style:{opacity:'0'}}));
  return {nodes,style:{setProperty(){}},querySelectorAll(){return nodes}};
}
test('right arm signal lights only the right screen edge',()=>{
  const p=player(),b=border();p.feedback(b,{edge:'right',intensity:1,tone:'green'});
  assert.deepEqual(b.nodes.map(n=>n.style.opacity),['0','1','0','0']);
});
test('changing direction clears the previous edge instead of accumulating flashes',()=>{
  const p=player(),b=border();p.feedback(b,{edge:'right',intensity:1,tone:'green'});
  p.feedback(b,{edge:'left',intensity:.5,tone:'green'});
  assert.deepEqual(b.nodes.map(n=>n.style.opacity),['0.85','0','0','0']);
});
test('stop or lost tracking immediately clears all edge lights',()=>{
  const p=player(),b=border();p.feedback(b,{edge:'all',intensity:1,tone:'green'});
  p.feedback(b,null);
  assert.deepEqual(b.nodes.map(n=>n.style.opacity),['0','0','0','0']);
});

test('any real directional movement is highly visible while zero stays dark',()=>{
  const p=player(),b=border();
  p.feedback(b,{edge:'left',intensity:.01,tone:'green'});
  assert.ok(Number(b.nodes[0].style.opacity)>=.7);
  p.feedback(b,{edge:'left',intensity:0,tone:'green'});
  assert.deepEqual(b.nodes.map(n=>n.style.opacity),['0','0','0','0']);
});
test('each directional light fades inward across a quarter of the screen',()=>{
  assert.match(html,/#cue-frame\{position:fixed;inset:0/);
  for (const [edge,dimension,direction] of [
    ['left','width','right'],['right','width','left'],
    ['top','height','bottom'],['bottom','height','top']
  ]) {
    const pattern=new RegExp(`\\[data-cue-edge="${edge}"\\]\\{[^}]*${dimension}:var\\(--cue-spread,25%\\);[^}]*linear-gradient\\(to ${direction},var\\(--cue-color\\)[^}]*transparent`);
    assert.match(html,pattern);
  }
});

function fakeClock(){
  let now=0,next=1;const raf=new Map(),timers=new Map();
  return {performance:{now:()=>now},document:{hidden:false},
    requestAnimationFrame(fn){const id=next++;raf.set(id,fn);return id},
    cancelAnimationFrame(id){raf.delete(id)},
    setTimeout(fn,ms){const id=next++;timers.set(id,{fn,at:now+ms});return id},
    clearTimeout(id){timers.delete(id)},
    step(ms){now+=ms;for(const [id,t] of [...timers])if(t.at<=now){timers.delete(id);t.fn()}
      const current=[...raf];raf.clear();for(const [,fn] of current)fn(now)},
  };
}
test('same-edge live light changes smoothly without dimming first feedback',()=>{
  const p=player(),b=border(),clock=fakeClock(),a=p.createFeedbackAnimator(b,clock);
  a.set({edge:'left',intensity:.1,tone:'green'});const first=Number(b.nodes[0].style.opacity);
  assert.ok(first>=.7,'first movement must be visible immediately');
  a.set({edge:'left',intensity:1,tone:'green'});
  assert.equal(Number(b.nodes[0].style.opacity),first);
  clock.step(40);const halfway=Number(b.nodes[0].style.opacity);
  assert.ok(halfway>first && halfway<1);
  clock.step(50);assert.equal(Number(b.nodes[0].style.opacity),1);
});
test('same-edge pulse finishes smoothly; stop, direction change and stale clear immediately',()=>{
  const p=player(),b=border(),clock=fakeClock(),a=p.createFeedbackAnimator(b,clock);
  a.set({edge:'left',intensity:.4,tone:'green'});a.set({edge:'left',intensity:1,tone:'green'});
  a.set(null);assert.deepEqual(b.nodes.map(n=>n.style.opacity),['0','0','0','0']);
  clock.step(40);assert.equal(b.nodes[0].style.opacity,'0');
  a.set({edge:'left',intensity:1,tone:'green'});a.set({edge:'right',intensity:.1,tone:'green'});
  assert.equal(b.nodes[0].style.opacity,'0');assert.ok(Number(b.nodes[1].style.opacity)>=.7);
  a.set({edge:'right',intensity:0,tone:'green'});clock.step(40);
  assert.ok(Number(b.nodes[1].style.opacity)>0&&Number(b.nodes[1].style.opacity)<.7);
  clock.step(40);assert.equal(b.nodes[1].style.opacity,'0');
  a.set({edge:'right',intensity:1,tone:'green'});a.set({edge:'right',intensity:0,tone:'green'});
  a.set(null);assert.equal(b.nodes[1].style.opacity,'0');
  a.set({edge:'right',intensity:1,tone:'green'});clock.step(350);
  assert.deepEqual(b.nodes.map(n=>n.style.opacity),['0','0','0','0']);
});
test('hidden tab cannot replay old light after visibility resumes',()=>{
  const p=player(),b=border(),clock=fakeClock(),a=p.createFeedbackAnimator(b,clock);
  a.set({edge:'left',intensity:1,tone:'green'});clock.document.hidden=true;a.set({edge:'left',intensity:.4,tone:'green'});
  assert.equal(b.nodes[0].style.opacity,'0');clock.document.hidden=false;clock.step(400);
  assert.equal(b.nodes[0].style.opacity,'0');
});
