const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

const html=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/guest.html'),'utf8');
const css=html.match(/<style>([\s\S]*?)<\/style>/)[1];

test('visitor presentation uses a restrained black, white and red palette',()=>{
  assert.match(css,/--paper:#111111/i);
  assert.match(css,/--ink:#f5f5f3/i);
  assert.match(css,/--accent:#e20612/i);
  assert.doesNotMatch(css,/--green/);
});

test('directional edge feedback uses Leica red and photo confirmation uses green',()=>{
  const c={module:{exports:{}}},properties=new Map(),root={style:{setProperty:(k,v)=>properties.set(k,v)},querySelectorAll:()=>[]};
  vm.runInNewContext(html.match(/<script id="guest-painter">([\s\S]*?)<\/script>/)[1],c);
  c.module.exports.feedback(root,{edge:'right',tone:'red',intensity:1});
  assert.equal(properties.get('--cue-color'),'#e20612');
  c.module.exports.feedback(root,{edge:'all',tone:'green',intensity:1});
  assert.equal(properties.get('--cue-color'),'#39d98a');
  for(const edge of ['left','right','top','bottom']){
    assert.match(css,new RegExp(`\\[data-cue-edge="${edge}"\\]\\{[^}]*linear-gradient\\(`));
  }
});


test('official Leica mark is self-contained and used in live and portrait views',()=>{
  const match=css.match(/--leica-mark:url\("data:image\/svg\+xml;base64,([A-Za-z0-9+/=]+)"\)/);
  assert.ok(match,'offline Leica mark');
  const official=fs.readFileSync(path.join(__dirname,'../src/leica_tracker/leica-logo.svg'));
  assert.deepEqual(Buffer.from(match[1],'base64'),official);
  assert.match(html,/<header[^>]*>[\s\S]*?class="leica-mark live-mark"[^>]*aria-label="Leica"/);
  assert.match(html,/<section id="photo-result"[\s\S]*?class="leica-mark portrait-mark"[^>]*aria-label="Leica"/);
});

test('the portrait presents the project question beside the photo',()=>{
  assert.match(html,/<section id="photo-result"[\s\S]*?The camera chooses where you stand\. Who is framing whom\?/);
  assert.match(html,/<header[^>]*>[\s\S]*?Who Frames Whom/);
});
