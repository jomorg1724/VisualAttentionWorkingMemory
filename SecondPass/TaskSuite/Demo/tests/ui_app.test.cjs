'use strict';
const {test}=require('node:test'); const assert=require('node:assert/strict'); const fs=require('node:fs'); const path=require('node:path'); const ui=require('../web/atlas.js');
test('Try state cannot be bypassed by global answers and a new example clears the prior reveal',()=>{
 assert.equal(typeof ui.transition,'function');
 let state={answers:true,explanation:true,rate:2};
 state=ui.transition(state,'try'); assert.equal(state.answers,false); assert.equal(state.explanation,false);
 state=ui.transition(state,'answers'); assert.equal(state.answers,false);
 state=ui.transition(state,'judgment',1); assert.equal(state.response,1); assert.equal(state.revealed,true); assert.equal(state.explanation,true);
 state=ui.transition(state,'new-example'); assert.equal(state.revealed,false); assert.equal(state.answers,false); assert.equal(state.response,undefined); assert.equal(state.rate,2);
 const cancelled=ui.transition(state,'cancel-judgment'); assert.equal(cancelled.trying,false); assert.equal(cancelled.answers,false);
});
test('native coverage IDs become readable variant controls without changing tokens',()=>{
 assert.equal(typeof ui.variantLabel,'function');
 assert.equal(ui.variantLabel('contrast/bank/contrast_pair/[0.025,0.08]'),'Increment 0.025 · pedestal 0.08');
 assert.equal(ui.variantLabel('krauzlis_cued_motion/B12/event_type/"foil"'),'Event type · foil');
});
test('offline entry has local synchronous data before player, bundled fonts, accessibility and controls',()=>{
 const file=path.join(__dirname,'../web/index.html'); assert.ok(fs.existsSync(file),'entry point exists'); const html=fs.readFileSync(file,'utf8');
 assert.ok(html.indexOf('src="data.js"')<html.indexOf('src="atlas.js"'));
 assert.match(html,/vendor\/fonts.css/); assert.match(html,/id="app"/); assert.match(html,/class="skip-link"/);
 for(const control of ['play-all','pause-all','answers']) assert.match(html,new RegExp('data-global="'+control+'"'));
 assert.ok(!/https?:\/\//.test(html));
});
test('display is integer nearest-neighbor without stimulus overlays or tween effects',()=>{
 const file=path.join(__dirname,'../web/atlas.css'); assert.ok(fs.existsSync(file),'authored stylesheet exists'); const css=fs.readFileSync(file,'utf8');
 assert.match(css,/image-rendering:\s*pixelated/); assert.match(css,/prefers-reduced-motion/); assert.match(css,/@media print/);
 assert.ok(!/linear-gradient|backdrop-filter/.test(css));
 assert.match(css,/--scene-size:\s*300px/); assert.match(css,/--scene-size:\s*200px/);
});
