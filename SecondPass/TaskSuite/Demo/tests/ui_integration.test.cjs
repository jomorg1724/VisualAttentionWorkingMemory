'use strict';
const assert=require('node:assert/strict'),test=require('node:test');
const fs=require('node:fs'),path=require('node:path');
const UI=require('../web/atlas.js');
const root=path.resolve(__dirname,'..');
const data={manifest:JSON.parse(fs.readFileSync(path.join(root,'artifacts/manifest.json'))),tasks:['sensory','spatial'].flatMap(g=>JSON.parse(fs.readFileSync(path.join(root,`content/${g}_tasks.json`)))),sources:['sensory','spatial'].flatMap(g=>JSON.parse(fs.readFileSync(path.join(root,`content/${g}_sources.json`))))};
test('native code indexing is not a source link and local code records have no fake external link',()=>{
 const task=data.tasks.find(t=>t.id==='orientation_ring'),cell=data.manifest.cells.find(c=>c.task_id===task.id),e=data.manifest.episodes.find(e=>e.id===cell.showcase_id);
 const html=UI.renderDetail(data,{task,cell,episode:e,reader:true},{answers:true,explanation:true});
 assert.ok(!html.includes('data-reference="target_location"'));
 assert.ok(!html.includes('href="#" target="_blank"'));
 assert.ok(html.includes('WorkingMemory/SpatialTaskBattery/stimuli.py'));
});
test('curated sensory worked prose is shown only with answer/explanation reveal',()=>{
 const task=data.tasks.find(t=>t.id==='motion_direction'),cell=data.manifest.cells.find(c=>c.task_id===task.id),e=data.manifest.episodes.find(e=>e.id===cell.showcase_id);
 const route={task,cell,episode:e};
 assert.ok(UI.renderDetail(data,route,{answers:true,explanation:true}).includes('128 surviving square-domain identities'));
 assert.ok(!UI.renderDetail(data,route,{answers:false,trying:true}).includes('128 surviving square-domain identities'));
});
