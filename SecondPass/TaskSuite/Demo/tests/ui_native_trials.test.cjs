'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const ui=require('../web/atlas.js');
const data={manifest:JSON.parse(fs.readFileSync(require('node:path').join(__dirname,'../artifacts/manifest.json'))),tasks:JSON.parse(fs.readFileSync(require('node:path').join(__dirname,'../content/tasks.json'))),sources:[]};

test('every retained native trial plays all frames once, holds the final observation, and completes once',()=>{
  for(const e of data.manifest.episodes){
    const seen=[];let finished=0;
    const p=new ui.Player(e,{loop:false,onChange:p=>seen.push(p.index),onComplete:()=>finished++});
    p.play();p.tick(0);
    for(let i=1;i<=e.frame_count;i++)p.tick(i*300);
    assert.deepEqual([...new Set(seen)],Array.from({length:e.frame_count},(_,i)=>i),e.id);
    assert.equal(p.index,e.frame_count-1);assert.equal(p.playing,false);assert.equal(p.loops,0);assert.equal(finished,1);
    p.tick(999999);assert.equal(finished,1);
  }
});
test('image memory defaults to four study images; explicit empty controls retain their exact identity',()=>{
  const route=ui.resolveRoute('#task=image_recognition',data);
  assert.equal(route.cell.condition_id,'N4_H4');assert.equal(route.episode.metadata.study_frames.length,4);
  const empty=ui.resolveRoute('#task=image_recognition&condition=N0_H5',data);
  assert.equal(empty.episode.metadata.study_frames.length,0);assert.equal(empty.episode.metadata.probe_frames.length,5);
  const html=ui.renderDetail(data,empty,{});
  assert.match(html,/Empty-list control: no study images/);assert.match(html,/same probe repeated 5/);
  assert.match(html,/data-study-load/);assert.match(html,/data-probe-hold/);
  assert.ok(html.indexOf('data-condition')<html.indexOf('data-player='));
});
test('every primary condition is described with its actual native varying quantity',()=>{
  for(const cell of data.manifest.cells){
    const label=ui.conditionLabel(cell.task_id,cell), k=cell.kwargs;
    if('load' in k)assert.match(label,new RegExp(`${k.load} study images`));
    if('probe_hold' in k)assert.match(label,new RegExp(`probe ${k.probe_hold} frames`));
    if('delay' in k)assert.match(label,new RegExp(`${k.delay} blank delay frames`));
    if('baseline_transitions' in k)assert.match(label,new RegExp(`${k.baseline_transitions} baseline motion transitions`));
  }
});
test('human response buttons wait for completion and answer-dependent variants stay hidden',()=>{
  const route=ui.resolveRoute('#task=image_recognition&condition=N4_H4',data);
  const state=ui.transition({answers:true,explanation:true,trialComplete:true},'try');
  assert.equal(state.trialComplete,false);
  const html=ui.renderDetail(data,route,state);
  assert.match(html,/data-judgment="0" disabled/);assert.match(html,/data-judgment="1" disabled/);
  assert.ok(!html.includes('membership_position'));assert.ok(!html.includes('data-variant'));
  assert.match(ui.renderDetail(data,route,{...state,trialComplete:true}),/data-judgment="1" >/);
});
