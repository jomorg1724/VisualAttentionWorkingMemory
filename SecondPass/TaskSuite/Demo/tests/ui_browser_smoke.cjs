/* Real-browser smoke using installed Chrome + Node's native CDP/WebSocket.
 * Read-only actual exporter snapshot; no renders, build writes or dependencies.
 * Run: node SecondPass/TaskSuite/Demo/tests/ui_browser_smoke.cjs
 */
'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),http=require('node:http'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process');
const root=path.resolve(__dirname,'..');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const tasks=['sensory_tasks.json','spatial_tasks.json'].flatMap(name=>{const p=path.join(root,'content',name);return fs.existsSync(p)?read(p):[];});
const sources=['sensory_sources.json','spatial_sources.json'].flatMap(name=>{const p=path.join(root,'content',name);return fs.existsSync(p)?read(p):[];});
const metaDir=path.join(root,'dist/assets/metadata');
assert.ok(fs.existsSync(metaDir),'Native assets must exist before this browser test');
const episodes=fs.readdirSync(metaDir).filter(f=>f.endsWith('.json')).map(f=>read(path.join(metaDir,f)));
const cells=[];
for(const e of episodes){let c=cells.find(c=>c.task_id===e.task_id&&c.condition_id===e.condition_id);if(!c){c={task_id:e.task_id,condition_id:e.condition_id,kwargs:e.kwargs,episode_ids:[],showcase_id:e.id};cells.push(c);}c.episode_ids.push(e.id);}
const data={manifest:{catalog:read(path.join(root,'../catalog.json')),episodes,cells},tasks:tasks.filter(t=>cells.some(c=>c.task_id===t.id)),sources,coverage:{}};
assert.ok(data.tasks.length,'Actual authored task content must exist');
const errors=[],checks=[];let chrome,server,ws,tmp;
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
 server=http.createServer((req,res)=>{
  const pathname=new URL(req.url,'http://localhost').pathname;
  if(pathname==='/data.js'){res.setHeader('Content-Type','text/javascript');res.end('window.ATLAS='+JSON.stringify(data)+';');return;}
  const rel=pathname==='/'?'index.html':pathname.slice(1);
  const base=rel.startsWith('assets/')?'dist':'web'; const file=path.resolve(root,base,rel);
  if(!file.startsWith(root+path.sep)||!fs.existsSync(file)){res.statusCode=404;res.end('Not found');return;}
  res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':file.endsWith('.html')?'text/html':file.endsWith('.png')?'image/png':'application/octet-stream');
  res.end(fs.readFileSync(file));
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const url='http://127.0.0.1:'+server.address().port;
 tmp=fs.mkdtempSync(path.join(os.tmpdir(),'atlas-ui-browser-'));
 chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--remote-debugging-port=0','--user-data-dir='+tmp,'about:blank'],{stdio:['ignore','ignore','pipe']});
 let stderr='';chrome.stderr.on('data',chunk=>stderr+=chunk.toString());
 for(let i=0;i<120&&!fs.existsSync(path.join(tmp,'DevToolsActivePort'));i++)await delay(100);
 assert.ok(fs.existsSync(path.join(tmp,'DevToolsActivePort')),stderr);
 const port=fs.readFileSync(path.join(tmp,'DevToolsActivePort'),'utf8').split('\n')[0];
 const target=await(await fetch('http://127.0.0.1:'+port+'/json/new?about:blank',{method:'PUT'})).json();
 ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
 let seq=0;const pending=new Map();
 ws.onmessage=event=>{const msg=JSON.parse(event.data);if(msg.id){const p=pending.get(msg.id);pending.delete(msg.id);msg.error?p.reject(msg.error):p.resolve(msg.result);}else if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails);};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const out=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(out.exceptionDetails)throw new Error(JSON.stringify(out.exceptionDetails));return out.result.value;};
 async function waitFor(expression){for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(50);}throw new Error('Timed out: '+expression);}
 const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
 await call('Runtime.enable');await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url});await waitFor('!!window.atlasApplication');
 assert.equal(await evaluate('document.querySelectorAll(".task-tile").length'),data.tasks.length);
 await evaluate('document.querySelector(".task-tile").scrollIntoView()');
 await waitFor('document.querySelector("[data-scene]").complete');
 await delay(650);
 assert.ok(await evaluate('Array.from(atlasApplication.scheduler.players).some(p=>p.index>0||p.loops>0)'));
 await click('[data-global="pause-all"]');
 const before=await evaluate('Array.from(atlasApplication.scheduler.players).map(p=>p.index)');await delay(500);
 assert.deepEqual(await evaluate('Array.from(atlasApplication.scheduler.players).map(p=>p.index)'),before);
 checks.push('gallery actual snapshot count; visible playback; pause all freezes');
 await evaluate('location.hash="#conditions"');await waitFor('document.querySelectorAll(".task-tile").length==='+cells.filter(c=>data.tasks.some(t=>t.id===c.task_id)).length);
 checks.push('condition wall includes every snapshot cell');
 const selected=episodes.find(e=>e.task_id==='motion_duration_cued')||episodes.find(e=>data.tasks.some(t=>t.id===e.task_id));
 const link='#'+new URLSearchParams({task:selected.task_id,condition:selected.condition_id,example:selected.id}).toString();
 await evaluate('location.hash='+JSON.stringify(link));await waitFor('!!document.querySelector(".detail-player")');
 await click('[data-action="next-frame"]');assert.equal(await evaluate('document.querySelector("[data-frame]").value'),'1');
 await click('[data-action="explanation"]');assert.ok(await evaluate('!!document.querySelector(".analysis-panel")'));
 await click('[data-action="try"]');assert.equal(await evaluate('document.querySelector("[data-frame]").value'),'0');
 assert.equal(await evaluate('document.querySelectorAll(".analysis-panel,[data-variant],[data-section=metadata],.answer-box,.downloads a").length'),0);
 assert.equal(await evaluate('document.querySelector("[data-global=answers]").disabled'),true);
 assert.ok(await evaluate('Array.from(document.querySelector("[data-example]").options).every(o=>/^Example \\d+$/.test(o.textContent))'));
 await click('[data-judgment="0"]');assert.ok(await evaluate('!!document.querySelector(".answer-box")'));
 if(selected.task_id==='motion_duration_cued'){
  await evaluate('const s=document.querySelector("[data-frame]");s.value=4;s.dispatchEvent(new Event("input",{bubbles:true}));');
  const expected=[0,0,0,0];selected.metadata.directions_by_patch[selected.metadata.target_location].slice(0,3).forEach(d=>expected[d]++);
  const actual=await evaluate('({counts:Array.from(document.querySelectorAll("[data-direction-count]")).map(el=>Number(el.textContent)), state:atlasApplication.getState(), players:Array.from(atlasApplication.scheduler.players).map(p=>({index:p.index,task:p.episode.task_id,moving:p.episode.metadata.moving_frames,classes:document.querySelector("[data-player]").className}))})');
  assert.deepEqual(actual.counts,expected,JSON.stringify(actual));
 }
 checks.push('detail stepping, Try UI leak suppression, reveal and actual schedule counters');
 await click('[data-action="observer"]');await click('[data-action="expand"]');
 assert.ok(await evaluate('Array.from(document.querySelectorAll("details")).every(d=>d.open)'));
 const frame=await evaluate('document.querySelector("[data-frame]").value');
 await evaluate('location.hash += "&mode=reader"');await waitFor('!!document.querySelector(".reader-view")');
 assert.equal(await evaluate('document.querySelector("[data-frame]").value'),frame);
 checks.push('expand all and reader mode preserve selected frame');
 for(const width of [1440,900,390]){
  await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:false});
  await delay(100);
  assert.ok(await evaluate('document.documentElement.scrollWidth <= innerWidth'),`detail overflow at ${width}`);
  assert.ok(await evaluate('document.querySelector("[data-global-rate]").getBoundingClientRect().width>0'),`global rate unavailable at ${width}`);
  const size=await evaluate('document.querySelector("[data-scene]").getBoundingClientRect().width');assert.ok([200,400,500].includes(size),`noninteger scene ${size}`);
 }
 checks.push('desktop/tablet/phone detail no horizontal overflow; integer scenes');
 await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
 await call('Page.navigate',{url});await waitFor('!!window.atlasApplication');
 assert.equal(await evaluate('atlasApplication.scheduler.playing'),false);
 await delay(400);assert.ok(await evaluate('Array.from(atlasApplication.scheduler.players).every(p=>!p.playing&&p.index===0)'));
 checks.push('real browser reduced motion starts paused');
 const checked=[];
 for(const cell of cells.filter(c=>data.tasks.some(t=>t.id===c.task_id))){
  const episode=episodes.find(e=>e.id===cell.showcase_id);
  const address='#'+new URLSearchParams({task:episode.task_id,condition:episode.condition_id,example:episode.id}).toString();
  await evaluate('location.hash='+JSON.stringify(address));
  await waitFor('atlasApplication.getRoute().episode?.id==='+JSON.stringify(episode.id));
  await click('[data-action="explanation"]');
  assert.equal(await evaluate('Number(document.querySelector("[data-frame]").max)'),episode.frame_count-1);
  await evaluate('(()=>{const slider=document.querySelector("[data-frame]");slider.value=slider.max;slider.dispatchEvent(new Event("input",{bubbles:true}));})()');
  assert.equal(await evaluate('Number(document.querySelector("[data-frame]").value)'),episode.frame_count-1);
  assert.ok(await evaluate('document.documentElement.scrollWidth<=innerWidth'),`phone overflow ${episode.id}`);
  if(episode.task_id==='image_recognition')assert.equal(await evaluate('document.querySelectorAll(".study-filmstrip figure").length'),episode.metadata.study_frames.length);
  for(const asset of [episode.gif,episode.frame_zip,episode.metadata_path,episode.poster])assert.ok(fs.existsSync(path.join(root,'dist',asset)),asset);
  checked.push(episode.task_id+'/'+episode.condition_id);
 }
 checks.push(`all ${checked.length} actual native cells: explanation, final-frame seek, phone overflow, downloads; complete recognition filmstrips`);
 assert.deepEqual(errors,[],'no uncaught browser exceptions');
 console.log(JSON.stringify({status:'passed',snapshot:{tasks:data.tasks.length,cells:cells.filter(c=>data.tasks.some(t=>t.id===c.task_id)).length,episodes:episodes.length},checks,consoleExceptions:errors.length},null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{ws?.close();chrome?.kill();if(server)await new Promise(r=>server.close(r));if(tmp){await delay(500);fs.rmSync(tmp,{recursive:true,force:true});}});
