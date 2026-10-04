'use strict';
const {test}=require('node:test'); const assert=require('node:assert/strict');
const ui=require('../web/atlas.js');
const e={id:'opaque-episode',task_id:'krauzlis_cued_motion',condition_id:'B12',label:1,label_meaning:'SECRET_ANSWER',frames:['a.png','b.png','c.png'],frame_count:3,poster:'a.png',gif:'a.gif',frame_zip:'a.zip',metadata_path:'a.json',phases:[{index:0,phase:'cue',cue_visible:true},{index:1,phase:'baseline',cue_visible:false},{index:2,phase:'postevent',cue_visible:false}],metadata:{event_type:'SECRET_EVENT',first_postchange_frame:2},covered_variants:['SECRET_VARIANT']};
const task={id:e.task_id,title:'Target change',group:'selection',question:'Did motion change at the cued patch?',intro:'A task.',sections:{rules:['Select the target.'],network:['Compare selected evidence.']},properties:[],references:[]};
const data={manifest:{catalog:{tasks:[{id:e.task_id,labels:['no','yes'],conditions:[{id:'B12'}]}]},episodes:[e],cells:[{task_id:e.task_id,condition_id:'B12',episode_ids:[e.id],showcase_id:e.id}]},tasks:[task],sources:[]};

test('deep links preserve task, native condition and exact episode through reader mode',()=>{
 assert.equal(typeof ui.resolveRoute,'function');
 const route=ui.resolveRoute('#task=krauzlis_cued_motion&condition=B12&example=opaque-episode&mode=reader',data);
 assert.equal(route.episode.id,e.id); assert.equal(route.reader,true);
 assert.match(ui.detailLink(e,true),/mode=reader/);
 assert.equal(ui.resolveRoute('#conditions',data).view,'conditions');
 assert.equal(ui.resolveRoute('#task=not-real',data).view,'not-found');
 assert.equal(ui.resolveRoute('#task=krauzlis_cued_motion&condition=not-real',data).view,'not-found');
});

test('Try judgment removes answer selectors, variant clues, metadata, event markers and download leaks from DOM',()=>{
 assert.equal(typeof ui.renderDetail,'function');
 const route={view:'detail',task,episode:e,cell:data.manifest.cells[0],reader:true};
 const html=ui.renderDetail(data,route,{answers:true,explanation:true,trying:true,revealed:false});
 for(const secret of ['SECRET_ANSWER','SECRET_EVENT','SECRET_VARIANT','first_postchange_frame','a.json','postevent']) assert.ok(!html.includes(secret),`leaked ${secret}`);
 assert.match(html,/data-judgment="0"/); assert.match(html,/data-judgment="1"/);
 assert.match(html,/Example 1/); assert.match(html,/Informal exploration/);
 const revealed=ui.renderDetail(data,route,{answers:true,explanation:true,trying:true,revealed:true,response:1});
 assert.match(revealed,/SECRET_ANSWER/); assert.match(revealed,/a.json/);
});

test('observer timeline does not announce Krauzlis event, explanation uses the recorded index',()=>{
 assert.equal(typeof ui.phaseGroups,'function');
 assert.deepEqual(ui.phaseGroups(e,false).map(p=>p.phase),['cue','motion']);
 const phases=ui.phaseGroups(e,true); assert.equal(phases[2].start,2); assert.equal(phases[2].phase,'postevent');
});

test('duration counts come from actual selected-patch schedule up to the displayed transition',()=>{
 assert.equal(typeof ui.durationEvidence,'function');
 const d={metadata:{target_location:1,directions_by_patch:[[3,3,3],[0,1,0,2,0,3,0,1]],moving_frames:[2,3,4,5,6,7,8,9]}};
 assert.deepEqual(ui.durationEvidence(d,1).counts,[0,0,0,0]);
 assert.deepEqual(ui.durationEvidence(d,4).counts,[2,1,0,0]);
 assert.deepEqual(ui.durationEvidence(d,33).counts,[4,2,1,1]);
 assert.equal(ui.durationEvidence(d,33).finalDirection,1);
 assert.equal(ui.durationEvidence(d,33).winner,0);
});

test('sensory worked examples read exact native photometry keys and ring does not invent a sign glyph',()=>{
 const base={...e,frame_count:2,frames:['a.png','b.png'],label:0,label_meaning:'frame 0'};
 const chromatic=ui.renderAnalysis({...base,task_id:'chromatic_increment',metadata:{chromatic_increment:0.045,axis_linear_rgb:[0.95,-0.3,0],frame_colors:[[0.4,0.3,0.5],[0.3,0.4,0.5]],frame_luminances:[0.36,0.36]}});
 assert.match(chromatic,/0.95/); assert.match(chromatic,/0.36/);
 const spectrum=ui.renderAnalysis({...base,task_id:'natural_spectrum',metadata:{betas_by_frame:[-0.175,0.125],actual_common_rms:0.15,beta_delta:0.3}});
 assert.match(spectrum,/-0.175/); assert.match(spectrum,/0.125/);
 const ring=ui.renderAnalysis({...e,task_id:'orientation_ring',frame_count:4,metadata:{target_location:2,cue_sign:1,rotations_degrees:[15,-15,15,-15],cue_frames:[0,1,2]}});
 assert.ok(!ring.includes(' · sign <strong>'),'ring metadata cue_sign is not a visible sign instruction');
});

test('worked sensory reasoning identifies why the alternative is wrong from native values',()=>{
 const base={...e,frame_count:2,frames:['a.png','b.png'],label:1,label_meaning:'frame 1'};
 const contrast=ui.renderAnalysis({...base,task_id:'contrast',metadata:{frame_contrasts:[0.08,0.14],contrast_increment:0.06,pedestal:0.08}});
 assert.match(contrast,/B has amplitude 0.14/);assert.match(contrast,/A has 0.08/);
 const orientation=ui.renderAnalysis({...base,task_id:'orientation',metadata:{signed_orientation_degrees:10,base_orientation_degrees:80}});
 assert.match(orientation,/positive/);assert.match(orientation,/negative/);
});

test('gallery and wall enumerate catalog cells, never a hard-coded duplicate inventory',()=>{
 assert.equal(typeof ui.renderGallery,'function');
 const cells=Array.from({length:35},(_,i)=>({...data.manifest.cells[0],condition_id:'C'+i}));
 const tasks=Array.from({length:13},(_,i)=>({...task,id:'t'+i}));
 const fixture={...data,tasks,manifest:{...data.manifest,cells:[...cells.map((c,i)=>({...c,task_id:'t'+i%13}))]}};
 assert.equal((ui.renderGallery(fixture,{},false).match(/data-player=/g)||[]).length,13);
 assert.equal((ui.renderGallery(fixture,{},false).match(/Static poster/g)||[]).length,13);
 assert.equal((ui.renderGallery(fixture,{},true).match(/data-player=/g)||[]).length,35);
});
