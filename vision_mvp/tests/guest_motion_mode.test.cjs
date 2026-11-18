const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
const block=html.match(/<script id="guest-painter">([\s\S]*?)<\/script>/);
function painter(){
  assert.ok(block);
  const context={module:{exports:{}},globalThis:{}};
  vm.runInNewContext(block[1],context);
  return context.module.exports;
}
function display(){
  const values={};
  const nodes=['left','right','top','bottom'].map(side=>({dataset:{cueEdge:side},style:{opacity:'0'}}));
  return {values,nodes,style:{setProperty:(name,value)=>{values[name]=value}},querySelectorAll(){return nodes}};
}
test('measured near and far guidance choose consistent visible screen extent',()=>{
  const p=painter(),d=display();
  p.feedback(d,{edge:'left',intensity:.5,tone:'green',motion_mode:'near'});
  assert.equal(d.values['--cue-spread'],'16%');
  assert.ok(Number(d.nodes[0].style.opacity)>=.7);
  p.feedback(d,{edge:'right',intensity:.5,tone:'green',motion_mode:'far'});
  assert.equal(d.values['--cue-spread'],'25%');
  assert.equal(d.nodes[0].style.opacity,'0');
  assert.ok(Number(d.nodes[1].style.opacity)>=.7);
});
test('unsupported modes and non-direction feedback retain default extent and HOLD clears',()=>{
  const p=painter(),d=display();
  p.feedback(d,{edge:'top',intensity:.5,motion_mode:'near'});
  p.feedback(d,{edge:'all',intensity:1,motion_mode:'other'});
  assert.equal(d.values['--cue-spread'],'25%');
  p.feedback(d,null);
  assert.deepEqual(d.nodes.map(n=>n.style.opacity),['0','0','0','0']);
});
test('new gesture switches extent on the same edge and stale timeout clears it',()=>{
  const p=painter(),d=display();let now=0,next=1;const raf=new Map(),timers=new Map();
  const clock={performance:{now:()=>now},document:{hidden:false},
    requestAnimationFrame(fn){const id=next++;raf.set(id,fn);return id},
    cancelAnimationFrame(id){raf.delete(id)},
    setTimeout(fn,ms){const id=next++;timers.set(id,{fn,at:now+ms});return id},
    clearTimeout(id){timers.delete(id)},
    step(ms){now+=ms;for(const [id,t] of [...timers])if(t.at<=now){timers.delete(id);t.fn()}
      const current=[...raf];raf.clear();for(const [,fn] of current)fn(now)},
  };
  const a=p.createFeedbackAnimator(d,clock);
  a.set({edge:'left',intensity:1,tone:'green',gesture_id:1,motion_mode:'far'});
  assert.equal(d.values['--cue-spread'],'25%');
  a.set({edge:'left',intensity:.5,tone:'green',gesture_id:2,motion_mode:'near'});
  assert.equal(d.values['--cue-spread'],'16%');
  assert.ok(Number(d.nodes[0].style.opacity)>=.7);
  clock.step(350);
  assert.deepEqual(d.nodes.map(n=>n.style.opacity),['0','0','0','0']);
});
