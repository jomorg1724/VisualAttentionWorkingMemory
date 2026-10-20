'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>(100*x).toFixed(1)+'%';
const findTask=id=>GUIDE.tasks.find(t=>t.id===id);
const answerLabel=(task,i)=>task.labels[i]==='frame 0'?'First image (native frame 0)':task.labels[i]==='frame 1'?'Second image (native frame 1)':task.labels[i];
function trialHint(id){
 if(id==='orientation_cued')return 'Read the small + or − glyph above the cued patch. It specifies the requested sign of rotation. In the renderer’s image coordinates, positive angles rotate clockwise and negative angles counterclockwise.';
 if(id==='orientation'||id==='orientation_ring')return 'Compare the orientation of the stripes across the two observations. Positive angles rotate clockwise in the image; negative angles rotate counterclockwise. The stripe orientation repeats every 180°.';
 if(id==='chromatic_increment')return 'The positive color direction increases red and decreases green while maintaining the renderer’s luminance constraint. Choose the image farther in that color direction.';
 if(/^vda(4|9|16)$/.test(id))return 'These replay groups are no-change controls used to compare cue validity. The model reports whether any item changes anywhere in the array. Change trials and attention interventions are analyzed in the findings.';
 return '';
}
let cleanup=()=>{},routeGeneration=0;
const loaders={};
function loadTask(id){
 if(window.GUIDE_TASKS?.[id])return Promise.resolve(window.GUIDE_TASKS[id]);
 if(!loaders[id])loaders[id]=new Promise((resolve,reject)=>{
  const s=document.createElement('script');s.src=`assets/guide/${id}/data.js`;
  s.onload=()=>resolve(window.GUIDE_TASKS[id]);s.onerror=()=>{delete loaders[id];s.remove();reject(new Error('Could not load the saved trial. Reload the page or check the local server.'))};document.head.append(s);
 });
 return loaders[id];
}
function subject(){return '<p class="subject-note"><span class="status-dot"></span>Measured results · previous 19-task model · checkpoint 204,289</p>'}
function figure(name,alt,caption){
 return `<figure class="figure-card"><button class="expand-figure" data-figure="${esc(name)}" aria-label="Enlarge figure: ${esc(alt)}"><img src="assets/guide/${esc(name)}.png" alt="${esc(alt)}" loading="lazy"></button><span class="zoom-note">Select figure to enlarge</span><figcaption>${caption}</figcaption></figure>`;
}
function sources(keys){return `<div class="source-links">${keys.map(k=>{const r=GUIDE.references[k];return r?`<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.authors)} (${r.year}) ↗</a>`:''}).join('')}</div>`}
function details(title,body,extra=''){return `<details class="disclosure" ${extra}><summary>${title}</summary><div>${body}</div></details>`}
function sidebar(kind,current){
 let items=kind==='trials'?GUIDE.tasks.map(t=>({id:t.id,title:t.title,group:t.family})):FINDINGS;
 const groups=[...new Set(items.map(x=>x.group))];
 return `<aside class="sidebar"><div class="sidebar-inner"><p class="eyebrow">${kind==='trials'?'19 tasks · one observer':'A finding at a time'}</p>
 <label class="mobile-select"><span class="field-label">${kind==='trials'?'Choose a task':'Choose a finding'}</span><select id="section-select" class="select">${items.map(x=>`<option value="${x.id}" ${x.id===current?'selected':''}>${esc(x.title)}</option>`).join('')}</select></label>
 <nav class="sidebar-links" aria-label="${kind==='trials'?'Tasks':'Findings'}">${groups.map(g=>`<div class="group-label">${g}</div>${items.filter(i=>i.group===g).map(i=>`<a href="#${kind}/${i.id}${kind==='trials'?'/0':''}" ${i.id===current?'aria-current="page"':''} class="${i.id===current?'active':''}">${esc(i.title)}</a>`).join('')}`).join('')}</nav></div></aside>`;
}
function wireSidebar(kind){if($('section-select'))$('section-select').onchange=e=>location.hash=`${kind}/${e.target.value}${kind==='trials'?'/0':''}`}

const heatStops=[[8,12,21],[45,61,114],[80,127,160],[141,191,168],[242,223,140]];
function color(v){let x=Math.min(1,Math.max(0,v))*(heatStops.length-1),a=Math.min(heatStops.length-2,Math.floor(x)),f=x-a;return heatStops[a].map((c,i)=>Math.round(c*(1-f)+heatStops[a+1][i]*f))}
function drawHeat(canvas,map,max,opacity=1){
 const ctx=canvas.getContext('2d');ctx.globalAlpha=opacity;
 for(let y=0;y<10;y++)for(let x=0;x<10;x++){ctx.fillStyle=`rgb(${color(Math.sqrt(map[y][x]/Math.max(max,1e-9)))})`;ctx.fillRect(x*canvas.width/10,y*canvas.height/10,canvas.width/10,canvas.height/10)}
 ctx.globalAlpha=1;
}
function drawInput(canvas,sprite,frame){
 const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;
 ctx.fillStyle='#808080';ctx.fillRect(0,0,canvas.width,canvas.height);
 if(sprite.complete&&sprite.naturalWidth)ctx.drawImage(sprite,frame*100,0,100,100,0,0,canvas.width,canvas.height);
}
const phaseColors={sample:'#8ab6a5',study:'#8ab6a5',reference:'#8ab6a5',search:'#d6a070',test:'#d6a070',probe:'#d6a070',change:'#d6a070',comparison:'#d6a070',delay:'#dfe3dd',blank:'#dfe3dd',cue:'#729baf'};
function phaseRuns(phases){let runs=[];phases.forEach((p,i)=>{if(!i||p!==phases[i-1])runs.push({p,start:i,n:1});else runs.at(-1).n++});return runs}
const traceColors=['#16776f','#c27743','#627aad','#a26194'];
function traceSvg(g){
 const W=800,H=210,L=48,R=18,T=16,B=37,n=g.time.length;
 const x=i=>L+i/Math.max(1,n-1)*(W-L-R), y=v=>T+(1-v)*(H-T-B);
 let out=`<svg class="trace-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Condition-average regional attention over all trial frames"><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="#f7f8f5"/>`;
 for(const r of phaseRuns(g.phases))out+=`<rect x="${x(r.start)}" y="${T}" width="${Math.max(0,x(Math.min(n-1,r.start+r.n))-x(r.start))}" height="${H-T-B}" fill="${phaseColors[r.p]||'#edf0eb'}" opacity=".18"/>`;
 for(const v of [0,.5,1])out+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#d9e1dc"/><text x="${L-10}" y="${y(v)+4}" text-anchor="end" font-size="12" fill="#5d7073">${v.toFixed(1)}</text>`;
 for(let j=0;j<g.region_names.length;j++){
  const lo=g.ci_low.map((r,i)=>`${x(i)},${y(r[j])}`).join(' ');
  const hi=[...g.ci_high].reverse().map((r,i)=>`${x(n-1-i)},${y(r[j])}`).join(' ');
  out+=`<polygon points="${lo} ${hi}" fill="${traceColors[j]}" opacity=".12"/><polyline points="${g.mean_mass.map((r,i)=>`${x(i)},${y(r[j])}`).join(' ')}" fill="none" stroke="${traceColors[j]}" stroke-width="2.5"/>`;
 }
 for(const i of [...new Set([0,Math.floor((n-1)/2),n-1])])out+=`<text x="${x(i)}" y="${H-18}" text-anchor="middle" font-size="12" fill="#5d7073">${i+1}</text>`;
 out+=`<text x="${W/2}" y="${H-2}" text-anchor="middle" font-size="12" fill="#5d7073">Displayed frame (one recurrent update per frame)</text><line id="trace-cursor" x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" stroke="#173038" stroke-dasharray="4 3"/><text x="13" y="94" transform="rotate(-90 13 94)" text-anchor="middle" font-size="11" fill="#5d7073">Attention mass</text></svg>`;
 return out;
}

async function home(generation){
 $('main').innerHTML=`<div class="container"><section class="hero">
 <div><p class="eyebrow">An interactive field guide</p><h1>See. Remember.<br> <em>Select.</em></h1><p class="lead">How one recurrent visual observer learns what to look at—and what to keep in mind.</p>
 <div class="actions"><a class="button primary" href="#trials/motion_memory_search/0">Watch a real trial <span>→</span></a><a class="button link" href="#results/priority">Explore the discoveries ↗</a></div>
 <p class="caption muted">Start with the experience. Open the neuroscience and mathematics when you want to go deeper.</p></div>
 <div class="hero-stage"><div class="hero-stage-top"><span class="eyebrow">A remembered motion guides search</span><span class="badge">Recorded trial</span></div>
 <div class="hero-panes"><div><p>What the model sees</p><canvas id="hero-input" width="300" height="300" aria-label="Actual recorded motion-search input"></canvas></div><div><p>Where it allocates attention</p><canvas id="hero-attention" width="300" height="300" aria-label="The same trial's spatial attention"></canvas></div></div>
 <div class="hero-stage-bottom"><button id="hero-play" class="button" disabled>Loading trial…</button><span id="hero-phase">Loading saved frames</span></div><p class="small" style="color:#a8bebb;margin-top:14px">Right: this trial’s attention, brighter = more weight. Same 10 × 10 sites throughout the movie.</p></div></section>
 <div class="stats-strip"><div><strong>19</strong><span>visual tasks</span></div><div><strong>437</strong><span>tested conditions</span></div><div><strong>55,936</strong><span>native evaluation trials</span></div></div>
 <section><div class="section-intro"><h2>Follow your curiosity.</h2><p>Each path reveals one idea at a time. The detailed evidence is always one step away.</p></div>
 <div class="entry-grid"><a class="entry-card" href="#trials/motion_memory_search/0"><span class="card-no">01 / EXPERIENCE</span><h3>Watch the task unfold.</h3><p>Replay the actual images. Pause at a memory delay. Reveal attention only when you are ready.</p><span class="arrow">Enter the trial explorer →</span></a>
 <a class="entry-card" href="#results/priority"><span class="card-no">02 / DISCOVER</span><h3>Ask what the evidence says.</h3><p>Explore causal attention effects, remembered content, and the connection to human and animal experiments.</p><span class="arrow">Read a finding →</span></a>
 <a class="entry-card" href="#model/0"><span class="card-no">03 / UNDERSTAND</span><h3>Build the mechanism step by step.</h3><p>Follow pixels into representations, queries into weights, and selected information into memory.</p><span class="arrow">Go inside the model →</span></a></div></section>
 <p class="home-note">The findings here describe the frozen 19-task observer at checkpoint <strong>204,289</strong>. <a href="#library">Evidence and experiment history</a> identify the separate spatial-query run and the earlier VDA experiments.</p></div>`;
 const d=await loadTask('motion_memory_search');if(generation!==routeGeneration)return;
 const g=d.groups[0],sprite=new Image();let frame=25,timer=null;
 function draw(){drawInput($('hero-input'),sprite,frame);drawHeat($('hero-attention'),g.example_map[frame],d.ownMax);$('hero-phase').textContent=`${g.phases[frame]} · frame ${frame+1} of ${g.time.length}`}
 function stop(){clearInterval(timer);timer=null;if($('hero-play'))$('hero-play').textContent='Replay sequence ↻'}
 sprite.onload=()=>{if(generation!==routeGeneration)return;draw();$('hero-play').disabled=false;$('hero-play').textContent='Replay sequence ↻'};
 sprite.onerror=()=>{if(generation===routeGeneration)$('hero-phase').textContent='Trial image could not load. Please reload.'};sprite.src=g.sprite;
 $('hero-play').onclick=()=>{if(timer){stop();return}frame=0;draw();$('hero-play').textContent='Pause';timer=setInterval(()=>{frame++;if(frame>=g.time.length){stop();return}draw()},200)};
 const hidden=()=>{if(document.hidden)stop()};document.addEventListener('visibilitychange',hidden);
 cleanup=()=>{stop();sprite.onload=null;document.removeEventListener('visibilitychange',hidden)};
}

async function trials(id,groupIndex,generation){
 const task=findTask(id)||findTask('motion_memory_search');id=task.id;
 $('main').innerHTML=`<div class="workspace">${sidebar('trials',id)}<article class="content"><div class="page-heading">${subject()}<p class="eyebrow">${task.family} / Trial explorer</p><h1>${esc(task.title)}.</h1><p class="deck">${esc(task.instruction)}</p></div><div id="trial-content" class="loading">Loading recorded frames and attention…</div></article></div>`;
 wireSidebar('trials');
 const d=await loadTask(id);if(generation!==routeGeneration)return;
 groupIndex=Math.max(0,Math.min(d.groups.length-1,Number(groupIndex)||0));
 const g=d.groups[groupIndex];let frame=0,timer=null,expanded=false,challenge=false,completed=false;
 const sprite=new Image();
 const result=FINDINGS.find(f=>f.task===id)||FINDINGS.find(f=>f.group===task.family)||FINDINGS[0];
 const variantKeys=[...new Set(task.variants.flatMap(v=>Object.keys(v)))];
 const variants=variantKeys.map(k=>`<li><strong>${esc(k.replaceAll('_',' '))}:</strong> ${[...new Set(task.variants.map(v=>v[k]))].map(esc).join(', ')}</li>`).join('');
 $('trial-content').className='';
 $('trial-content').innerHTML=`<div class="trial-topline"><label><span class="field-label">Recorded condition group</span><select id="condition" class="select">${d.groups.map((r,i)=>`<option value="${i}" ${i===groupIndex?'selected':''}>${esc(r.title)}</option>`).join('')}</select></label><div class="trial-stat"><strong>${pct(task.accuracy)}</strong> across ${task.conditions} native conditions · ${task.trials.toLocaleString()} trials</div></div>
 ${trialHint(id)?`<p class="caption muted" style="margin:0 0 20px">${esc(trialHint(id))}</p>`:''}
 <section class="trial-box" aria-label="Recorded trial player"><p class="trial-instruction">${esc(task.question)}</p><div class="trial-stage"><div class="trial-stage-head"><span class="phase-pill" id="phase-name"></span><span id="frame-label"></span></div>
 <div class="canvas-panes" id="canvas-panes"><div class="canvas-pane"><h3>Actual model input</h3><canvas id="trial-input" width="300" height="300" aria-label="Native trial image at the selected frame"></canvas><p>Original 100 × 100 pixels, enlarged.</p></div>
 <div class="canvas-pane attention-pane" hidden><h3>This trial · attention overlay</h3><canvas id="trial-overlay" width="300" height="300" aria-label="This trial's attention overlaid on its exact image"></canvas><div class="heat-legend"><span>0</span><span></span><span>${d.ownMax.toFixed(3)}</span></div><p>Weight per site · square-root color scale.</p></div>
 <div class="canvas-pane attention-pane" hidden><h3>Condition average · ${g.n} trials</h3><canvas id="trial-mean" width="300" height="300" aria-label="Condition-average attention map"></canvas><div class="heat-legend"><span>0</span><span></span><span>${d.colorMax.toFixed(3)}</span></div><p>Fixed image coordinates · separate scale shown.</p></div></div></div>
 <div class="trial-controls"><div class="playbar"><button id="play" class="button primary" disabled>Play</button><button id="back-frame" class="button icon-button" aria-label="Previous frame">‹</button><input id="frame-range" type="range" min="0" max="${g.time.length-1}" value="0" aria-label="Displayed frame"><button id="next-frame" class="button icon-button" aria-label="Next frame">›</button><select class="select speed" id="speed" aria-label="Playback speed"><option value="500">2 fps</option><option value="200" selected>5 fps</option><option value="100">10 fps</option></select></div>
 <div class="timeline" aria-label="Trial phase sequence">${phaseRuns(g.phases).map(r=>`<span class="${r.p}" style="flex:${r.n}" title="${r.p}: ${r.n} frames"></span>`).join('')}</div>
 <div class="phase-key">${[...new Set(g.phases)].map(p=>`<span style="--key:${phaseColors[p]||'#c8d9d1'}">${esc(p)}</span>`).join('')}</div>
 <div class="viewer-actions"><button class="button" id="toggle-attention" aria-expanded="false">Reveal attention</button><button class="button" id="try-trial">Try the judgment</button><button class="button" id="show-answer" aria-expanded="false">Reveal recorded answer</button>${g.gif?`<a class="button" href="${g.gif}" download>Download GIF ↓</a>`:''}</div>
 <p class="trial-meta">Viewing speed is adjustable; one displayed frame is one model update. Every original frame is retained. This is a selected demonstration, not a human performance test.</p></div>
 <div class="attention-trace" id="attention-trace" hidden><h3>How priority moves through this condition</h3><p class="small muted">Regional mass adds patch weights. Bands: 95% cluster-bootstrap intervals over ${g.clusters} resampling clusters.</p><div class="trace-legend">${g.region_names.map((name,i)=>`<span><b style="background:${traceColors[i]}"></b>${esc(name)}</span>`).join('')}</div>${traceSvg(g)}</div>
 <div class="response-box" id="response" hidden><h3>Your judgment</h3><p id="response-prompt">Watch the complete sequence, then choose an answer.</p><div id="answer-buttons" class="answer-buttons" hidden>${task.labels.map((l,i)=>`<button class="button" data-answer="${i}">${esc(answerLabel(task,i))}</button>`).join('')}</div><p class="answer-result" id="answer-result"></p></div></section>
 ${details('What varies in this task?',`<p>${esc(task.why)}</p><p>The training and native evaluation span <strong>${task.conditions} conditions</strong>. Their parameters are:</p><ul>${variants}</ul><p>Units: S/sample and H/exposure count visible frames; D/delay counts blank frames. Motion spacing and orientation differences are degrees. Each native condition has 128 evaluation trials. The replay menu selects ${d.groups.length} analysis groups for comparing attention; those groups are a subset or pooling of native conditions.</p><button class="button" id="load-conditions">View every native condition and accuracy</button><div id="condition-table"></div>`)}
 ${details('How was this replay and attention average made?',`<p>${esc(d.context)}</p><p>This replay is the first trial meeting the recorded group definition, independently of whether its answer was correct. Source: <strong>${esc(g.example.cell)}</strong>, trial index <strong>${g.example.index}</strong>. The group’s terminal accuracy is ${pct(g.accuracy)} across ${g.n} trials.</p><p>Own attention belongs to the displayed input. The separate mean averages each patch weight over all selected trials. The colors use a square-root transform to reveal small weights; both legends give probabilities. “Other regions (mean)” is the average of separate alternative-region masses.</p><p><a href="assets/guide/provenance.json" target="_blank">Source hashes and build record ↗</a> · <a href="../../TaskNeuroscience/Results/attention/index.html#task=${id}" target="_blank">Full condition-by-time analysis ↗</a></p>`)}
 <div class="next-read"><div><p>Connect the trial to the science</p><h3>${esc(result.title)}</h3></div><a class="button" href="#results/${result.id}">Explore this finding →</a></div>`;
 function draw(){
  drawInput($('trial-input'),sprite,frame);
  if(expanded){drawInput($('trial-overlay'),sprite,frame);drawHeat($('trial-overlay'),g.example_map[frame],d.ownMax,.5);drawHeat($('trial-mean'),g.mean_map[frame],d.colorMax)}
  $('phase-name').textContent=g.phases[frame];$('frame-label').textContent=`Frame ${frame+1} / ${g.time.length}`;$('frame-range').value=frame;
  const cursor=$('trace-cursor');if(cursor){const x=48+frame/Math.max(1,g.time.length-1)*734;cursor.setAttribute('x1',x);cursor.setAttribute('x2',x)}
 }
 function stop(){clearInterval(timer);timer=null;if($('play'))$('play').textContent=frame===g.time.length-1?'Replay':'Play'}
 function finish(){stop();completed=true;if(challenge){$('answer-buttons').hidden=false;$('response-prompt').textContent=task.question;$('announcement').textContent='Sequence complete. Choose your answer.'}}
 function start(reset=false){stop();if(reset||frame===g.time.length-1)frame=0;draw();$('play').textContent='Pause';timer=setInterval(()=>{if(frame>=g.time.length-1){finish();return}frame++;draw()},Number($('speed').value))}
 function toggleAttention(force){expanded=force??!expanded;$('canvas-panes').classList.toggle('expanded',expanded);document.querySelectorAll('.attention-pane').forEach(p=>p.hidden=!expanded);$('attention-trace').hidden=!expanded;$('toggle-attention').textContent=expanded?'Hide attention':'Reveal attention';$('toggle-attention').setAttribute('aria-expanded',String(expanded));draw()}
 function reveal(answer){
  stop();challenge=false;$('try-trial').textContent='Try the judgment';
  $('response').hidden=false;$('answer-buttons').hidden=true;
  $('response-prompt').textContent=`Correct answer: ${answerLabel(task,g.answer)}`;
  $('answer-result').textContent=`${answer===undefined?'':answer===g.answer?'Your judgment is correct. ':'Your judgment differs from the recorded label. '}The model chose “${answerLabel(task,g.choice)}” (${pct(g.probabilities[g.choice])} final response probability). ${g.choice===g.answer?'The model answered correctly.':'The model made an error on this trial.'}`;
  $('show-answer').setAttribute('aria-expanded','true');
  for(const cid of ['condition','toggle-attention','frame-range','back-frame','next-frame','show-answer'])$(cid).disabled=false;
 }
 $('condition').onchange=e=>location.hash=`trials/${id}/${e.target.value}`;
 $('play').onclick=()=>timer?stop():start();
 $('speed').onchange=()=>{if(timer)start()};
 $('frame-range').oninput=e=>{stop();frame=Number(e.target.value);draw()};
 $('back-frame').onclick=()=>{stop();frame=Math.max(0,frame-1);draw()};
 $('next-frame').onclick=()=>{stop();frame=Math.min(g.time.length-1,frame+1);draw()};
 $('toggle-attention').onclick=()=>toggleAttention();
 $('show-answer').onclick=()=>reveal();
 $('try-trial').onclick=()=>{
  if(challenge){stop();challenge=false;$('try-trial').textContent='Try the judgment';$('response').hidden=true;for(const cid of ['condition','toggle-attention','frame-range','back-frame','next-frame','show-answer'])$(cid).disabled=false;return}
  challenge=true;completed=false;toggleAttention(false);$('response').hidden=false;$('answer-buttons').hidden=true;$('answer-result').textContent='';$('response-prompt').textContent='Watch the complete sequence, then choose an answer.';$('try-trial').textContent='Cancel judgment';
  for(const cid of ['condition','toggle-attention','frame-range','back-frame','next-frame','show-answer'])$(cid).disabled=true;
  start(true);
 };
 document.querySelectorAll('[data-answer]').forEach(b=>b.onclick=()=>{if(completed)reveal(Number(b.dataset.answer));$('try-trial').textContent='Try the judgment'});
 $('load-conditions').onclick=()=>{$('condition-table').innerHTML=conditionsTable(d.cells);$('load-conditions').hidden=true};
 const hidden=()=>{if(document.hidden)stop()};document.addEventListener('visibilitychange',hidden);
 sprite.onload=()=>{if(generation!==routeGeneration)return;draw();$('play').disabled=false};
 sprite.onerror=()=>{if(generation===routeGeneration)$('phase-name').textContent='Image unavailable — reload page'};
 sprite.src=g.sprite;draw();
 cleanup=()=>{stop();sprite.onload=null;document.removeEventListener('visibilitychange',hidden)};
}
function conditionsTable(cells){
 return `<div class="table-wrap"><table class="task-table"><caption class="sr-only">Accuracy for every native condition</caption><thead><tr><th>Condition</th><th>Trials</th><th>Correct</th></tr></thead><tbody>${cells.map(c=>`<tr><td>${esc(c.cell)}</td><td>${c.n}</td><td>${pct(c.accuracy)}</td></tr>`).join('')}</tbody></table></div>`;
}
function resultView(id){
 const f=FINDINGS.find(x=>x.id===id)||FINDINGS[0];
 $('main').innerHTML=`<div class="workspace">${sidebar('results',f.id)}<article class="content"><div class="page-heading">${subject()}<p class="eyebrow">${f.group} / A measured finding</p><h1>${f.title}</h1><p class="deck">${f.deck}</p></div>
 <div class="finding-metric"><div><strong>${f.metric}</strong><div class="unit">${f.unit}</div></div><p class="note">${f.metricNote}</p></div>
 ${figure(f.figure,f.alt,f.caption)}
 <div class="result-reading"><h3>What happened?</h3><p>${f.observe}</p></div>
 <section class="science-block"><p class="eyebrow">The neuroscience connection</p><h3>From the experiment to the model.</h3><p>${f.science}</p><p class="connection">${f.connection}</p>${sources(f.refs)}</section>
 ${details('How to interpret this experiment',`<p>${f.detail}</p>`)}
 ${f.extraFigure?details(f.extraTitle,figure(f.extraFigure,f.extraTitle,f.extraCaption)):''}
 ${details('Definitions and evidence',`<p><strong>Percentage points:</strong> absolute difference between percentages; a drop from 80% to 60% is 20 points. <strong>Paired trials:</strong> the same stimulus is replayed across experimental conditions. <strong>Confidence interval:</strong> uncertainty due to the sampled trials, under the stated resampling procedure.</p><p>The observer is fixed at checkpoint 204,289. These results measure this trained model across stimulus conditions. Every figure is rebuilt from saved observations; the schematic attention demonstration in “Inside the model” is labeled separately.</p><p><a href="assets/guide/neuroscience-results.pdf" target="_blank">Read the full neuroscience paper ↗</a> · <a href="assets/guide/provenance.json" target="_blank">Inspect source provenance ↗</a></p>`)}
 <div class="bottom-nav"><a class="button" href="#trials/${f.task}/0">Watch the related task ↗</a><a class="button primary" href="#results/${f.next}">Next finding →</a></div></article></div>`;
 wireSidebar('results');
}

function modelGraphic(step){
 let grid='';for(let y=0;y<10;y++)for(let x=0;x<10;x++){
  let active=(x<3&&y<3),fill=step===0?['#4c8a80','#6b9c8e','#c69661'][(x*3+y*7)%3]:active?'#dec180':'#355f63';
  grid+=`<rect x="${30+x*13}" y="${55+y*13}" width="10" height="10" rx="1" fill="${fill}" opacity="${active||step===0?'.95':'.45'}"/>`;
 }
 const title=['100 visual sites','Previous global memory','Site-specific keys + values','One query · 100 comparisons','Selected evidence updates memory','Task-specific response'][step];
 const right=['64 numbers per site','One 64-number query','Key decides weight','One spatial distribution','Memory changes the next query','A probability for each answer'][step];
 return `<svg viewBox="0 0 420 245" role="img" aria-label="Schematic: ${title}"><text x="210" y="23" text-anchor="middle" fill="#d6e9df" font-size="15">${title}</text>${grid}<path d="M172 115H235" stroke="#91bba9" stroke-width="2" fill="none"/><path d="M229 109l8 6-8 6" stroke="#91bba9" fill="none" stroke-width="2"/><rect x="250" y="76" width="137" height="80" rx="10" fill="#254a4d" stroke="#78a293"/><text x="318" y="111" text-anchor="middle" fill="#e8dfb5" font-size="17">${['Visual vector','Query q','Key k / value v','Weight α','Memory h','Choice'][step]}</text><text x="318" y="137" text-anchor="middle" fill="#afcec0" font-size="12">${['Learned fusion','From remembered state','Learned projections','Softmax across sites','Recurrent update','Task output head'][step]}</text>${step===4?'<path d="M320 168V205H95V191" fill="none" stroke="#d5b778" stroke-width="2" stroke-dasharray="5 4"/>':''}<text x="210" y="237" text-anchor="middle" fill="#afcec0" font-size="13">${right}</text></svg>`;
}
function modelView(raw){
 const step=Math.min(5,Math.max(0,Number(raw)||0)),s=MODEL_STEPS[step];
 $('main').innerHTML=`<div class="container"><div class="page-heading"><p class="eyebrow">Inside the model / One operation at a time</p><h1>Follow the information.</h1><p class="deck">Start with an image. End with a choice. Open the mathematics at the step where it becomes useful.</p></div>
 <nav class="model-stepper" aria-label="Architecture steps">${MODEL_STEPS.map((x,i)=>`<a href="#model/${i}" class="${i===step?'active':''}" ${i===step?'aria-current="step"':''}><strong>0${i+1}</strong>${x.tag.split('/')[1]}</a>`).join('')}</nav>
 <div class="model-layout"><div class="model-visual">${modelGraphic(step)}<p>Explanatory schematic · colors illustrate information flow.</p></div><div class="model-copy"><p class="eyebrow">${s.tag}</p><h2>${s.title}</h2><p>${s.body}</p><div class="intuition">${s.intuition}</div></div></div>
 ${details('Show the math and define every symbol',`<div class="math-box">${esc(s.equation)}</div><p>${s.terms}</p><p>${s.deep}</p>`)}
 ${step===3?`<section class="toy"><p class="eyebrow">Try the calculation / Four-site teaching example</p><h3>How does a score become a weight?</h3><p class="muted">Move a score. Every weight changes because they share the same denominator. These numbers are an illustrative calculation.</p><div class="toy-grid">${['Top left','Top right','Bottom left','Bottom right'].map((n,i)=>`<label class="toy-site">${n}<input type="range" id="score-${i}" min="-4" max="4" step=".25" value="${i===0?2:0}" aria-label="${n} illustrative attention score"><span id="score-label-${i}"></span><output id="weight-${i}"></output><div class="toy-bar"><i id="weight-bar-${i}"></i></div></label>`).join('')}</div><p class="tagline" id="toy-sum"></p></section>`:''}
 ${step===0?flowDetails():''}
 ${step===1?details('What changed in the newer spatial-query experiment?',`<p>The analyzed 19-task checkpoint generates its query from global memory. The new architecture adds a learned projection of the previous spatial-memory grid directly into that query. The aim is to let detailed distributed memory influence selection more directly.</p><p>The new run began from random initialization. Its recovered latest checkpoint is 22,458; its best monitoring checkpoint is 17,366. The monitoring mean AUC at that best checkpoint was 0.8047. Its full final evaluation is unfinished. <a href="#library">See the separate experiment record →</a></p>`):''}
 <div class="bottom-nav">${step?`<a class="button" href="#model/${step-1}">← Previous step</a>`:'<a class="button" href="#trials/motion_memory_search/0">Watch the input ↗</a>'}${step<5?`<a class="button primary" href="#model/${step+1}">Next: ${MODEL_STEPS[step+1].tag.split('/')[1]} →</a>`:'<a class="button primary" href="#results/priority">See what this mechanism does →</a>'}</div></div>`;
 if(step===3){
  function update(){const scores=[0,1,2,3].map(i=>Number($('score-'+i).value)),ex=scores.map(Math.exp),sum=ex.reduce((a,b)=>a+b);scores.forEach((v,i)=>{$('score-label-'+i).textContent='Score '+v.toFixed(2);$('weight-'+i).textContent=pct(ex[i]/sum);$('weight-bar-'+i).style.width=(100*ex[i]/sum)+'%'});$('toy-sum').textContent='All four weights sum to 100%. Weight = exp(score) ÷ sum of exp(all scores).'}
  [0,1,2,3].forEach(i=>$('score-'+i).oninput=update);update();
 }
}
function flowDetails(){return details('Go deeper: pixels, cropping, motion and the Fourier transform',`
 <p><strong>Grayscale intensity</strong> is one number per pixel describing its brightness. Here it is the mean of the normalized red, green and blue values. Image coordinates increase to the right (x) and downward (y).</p>
 <div class="flow-list"><div><strong>1. Extract local crops</strong>A 24 × 24 crop is taken at each of 100 sites. Sites are spaced 10 pixels apart, so neighboring crops share pixels. A border around the whole image supplies positions needed near its edges.</div><div><strong>2. Compare two observations</strong>The crop’s mean is subtracted, and its prepared pattern is compared with an earlier crop at each candidate displacement. The lags are 1, 2 and 4 frames.</div><div><strong>3. Keep the alternatives</strong>Horizontal and vertical shifts each range from −4 to +4: nine possibilities per axis, giving 81 combinations. The resulting 9 × 9 table describes displacement alternatives.</div></div>
 <p><strong>Why subtract the mean?</strong> It removes the crop’s constant brightness component so spatial variation drives the match. Normalization by crop energies reduces sensitivity to positive amplitude scaling. Energy is the sum of squared prepared pixel values; an energy-based factor reduces the influence of weak prepared signals.</p>
 <p><strong>Why a search radius?</strong> The comparison must specify which shifts to evaluate. Four pixels is an implementation parameter setting that range. A true displacement can exceed it; then its exact shift is outside this bank. The 24-pixel crop width and four-pixel shift radius are different quantities.</p>
 <p><strong>Why two padding operations?</strong> First, pad the whole image to extract edge crops. After crop preparation, extend each 24 × 24 crop to a 32 × 32 zero-filled array for the correlation calculation. The latter prevents circular wraparound from contaminating the tested small shifts.</p>
 <p><strong>What does the FFT do?</strong> It expresses each prepared crop as coefficients of spatial waves. Multiplying the conjugated coefficients of the earlier crop by those of the later crop and applying an inverse transform computes circular cross-correlation. The retained small-shift entries equal the intended overlap sums with this padding. The code reads the 81 candidate entries from that larger table, normalizes their scores, and supplies the displacement evidence to the model.</p>
 <p>The fixed correspondence calculation supplies local temporal evidence. Learned appearance, fusion, recurrence and readout determine how it is used. <a href="reference.html#optical-flow">Open the extended optical-flow reference ↗</a> · <a href="reference.html#architecture">Detailed implementation and equations ↗</a></p>`)}

function library(){
 $('main').innerHTML=`<div class="container"><div class="page-heading"><p class="eyebrow">Evidence / Keep the experiments identifiable</p><h1>The measurements behind the story.</h1><p class="library-intro">The main guide follows one frozen observer across 19 tasks. Its checkpoint, trial counts and analysis definitions stay attached to the results.</p></div>
 <div class="library-grid"><section class="library-card"><p class="eyebrow">Neuroscience synthesis</p><h3>The argument in 18 pages.</h3><p>12 figures and 34 primary references connect the measured behavior, retained information and causal interventions to neuroscience experiments.</p><a class="button primary" href="assets/guide/neuroscience-results.pdf" target="_blank">Open the PDF ↗</a></section>
 <section class="library-card"><p class="eyebrow">Inspect the evidence</p><h3>Trials, maps and methods.</h3><p>The guide contains 63 selected condition groups and 888 recorded frames. Every native condition is listed in its task’s detail panel.</p><div class="actions"><a class="button" href="../../TaskNeuroscience/Results/attention/index.html" target="_blank">Complete attention atlas ↗</a><a class="button" href="assets/guide/provenance.json" target="_blank">Source record ↗</a></div></section></div>
 <section style="margin-top:45px"><div class="section-intro"><div><p class="eyebrow">Previous model · checkpoint 204,289</p><h2 style="margin-top:10px">Every task, in view.</h2></div><p>128 trials per native condition. Accuracy averages the conditions within each task equally.</p></div>
 <div class="table-wrap"><table class="task-table"><caption class="sr-only">Native task performance for checkpoint 204289</caption><thead><tr><th>Task</th><th>Conditions</th><th>Trials</th><th>Correct</th></tr></thead><tbody>${GUIDE.tasks.map(t=>`<tr><td><a href="#trials/${t.id}/0">${esc(t.title)}</a></td><td>${t.conditions}</td><td>${t.trials.toLocaleString()}</td><td><span class="accuracy-bar"><i style="width:${100*t.accuracy}%"></i></span>${pct(t.accuracy)}</td></tr>`).join('')}</tbody></table></div></section>
 ${details('Sampling, uncertainty and model identity',`<p>The analyzed weights are checkpoint 204,289, SHA-256 <code style="overflow-wrap:anywhere">${GUIDE.checkpointSha256}</code>. Native evaluation includes 437 conditions and 55,936 trials. Intervention tests replay matched pixels. Population decoders use 192 fitting and 128 held-out trials; selected diagnostic readouts fit separate conditions.</p><p>Wilson intervals summarize event rates. Paired bootstrap intervals quantify matched intervention effects. Clustered analyses keep repeated scenes or photograph identities together. The primary attention comparisons use Holm correction across tasks. Trial intervals describe stimulus sampling for this trained observer.</p><p>The replays retain lossless input sprites and recorded attention. GIF downloads are indexed-color viewing copies at five frames per second; the controlled player uses the original PNG sprites. The first trial satisfying a condition is displayed, independently of its correctness.</p>`)}
 <section class="epoch-block"><p class="eyebrow">Separate experiment / Spatial memory → query</p><h3>A new connection, trained from scratch.</h3><p>This architecture adds a direct projection from distributed spatial memory into the attention query. It tests the hypothesis suggested by the static-search information-use gap.</p>
 <div class="epoch-metrics"><div><strong>${GUIDE.newExperiment.latest.toLocaleString()}</strong><span>latest recovered update / 38,000 target</span></div><div><strong>${GUIDE.newExperiment.best.toLocaleString()}</strong><span>best monitoring checkpoint</span></div><div><strong>${GUIDE.newExperiment.bestMonitoringAuc.toFixed(4)}</strong><span>best monitoring mean AUC</span></div></div>
 <p>Run record: stopped at its time cap; weights recovered; pod deleted. Full final evaluation is unfinished. Monitoring AUC measures score discrimination and is a different measurement from the full native accuracies above. <a href="#model/1">Inspect the added connection →</a></p></section>
 ${details('Earlier results and the long-form reference',`<p>The October 11 manuscript preserves the earlier 13-task, 16-task and 19-task snapshots, including a VDA remote-benefit result. The current neuroscience synthesis analyzes checkpoint 204,289 and its matched follow-ups, which show a different VDA outcome. Read each result with its checkpoint and protocol.</p><div class="actions"><a class="button" href="reference.html">Open the archived reference ↗</a><a class="button" href="reference.html#vda-tasks">Earlier VDA protocols ↗</a><a class="button" href="task-atlas/index.html">Original 13-task atlas ↗</a></div>`)}
 <section style="margin-top:35px"><p class="eyebrow">Primary literature</p><h2 style="margin-top:10px">Follow an idea to its source.</h2><label><span class="sr-only">Filter references by author or topic</span><input class="search-input" id="reference-search" placeholder="Search author, title or year…" type="search"></label><ul class="ref-list" id="references-list"></ul></section></div>`;
 function showRefs(){const q=$('reference-search').value.toLowerCase();$('references-list').innerHTML=Object.values(GUIDE.references).filter(r=>`${r.authors} ${r.title} ${r.year}`.toLowerCase().includes(q)).map(r=>`<li><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a><small>${esc(r.authors)} · ${r.year}</small></li>`).join('')||'<li>No matching references.</li>'}
 $('reference-search').oninput=showRefs;showRefs();
}

async function route(){
 cleanup();cleanup=()=>{};const generation=++routeGeneration;
 let path=location.hash.slice(1)||'home';
 const aliases={top:'home',introduction:'home',tasks:'trials/motion_memory_search/0','vda-tasks':'trials/vda4/0',architecture:'model/0','attention-memory':'model/1','optical-flow':'model/0',neuroscience:'results/priority','neuro-causal':'results/vda',experiments:'results/priority',references:'library','independent-findings':'results/readout'};
 if(aliases[path]){path=aliases[path];history.replaceState(null,'','#'+path)}
 const [kind,id,group]=path.split('/');
 document.querySelectorAll('[data-nav]').forEach(a=>{if(a.dataset.nav===kind)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
 window.scrollTo(0,0);
 try{if(kind==='home')await home(generation);else if(kind==='trials')await trials(id,group,generation);else if(kind==='results')resultView(id);else if(kind==='model')modelView(id);else if(kind==='library')library();else {history.replaceState(null,'','#home');await home(generation)}
 }catch(error){if(generation===routeGeneration){$('main').innerHTML=`<div class="container"><h1>The saved evidence could not load.</h1><p style="margin-top:20px">${esc(error.message)}</p><a class="button" href="assets/guide/neuroscience-results.pdf">Read the PDF</a></div>`}console.error(error)}
 if(generation===routeGeneration){$('announcement').textContent=$('main').querySelector('h1')?.textContent||'Guide updated';document.title=($('main').querySelector('h1')?.textContent||'Selection & Memory')+' · Selection & Memory'}
}
document.addEventListener('click',e=>{
 if(e.target.closest('.skip-link')){e.preventDefault();$('main').focus();$('main').scrollIntoView();return}
 const b=e.target.closest('.expand-figure');if(!b)return;
 const card=b.closest('figure');$('large-figure').src=b.querySelector('img').src;$('large-figure').alt=b.querySelector('img').alt;$('large-caption').textContent=card.querySelector('figcaption').textContent;$('figure-dialog').showModal();
});
$('close-figure').onclick=()=>$('figure-dialog').close();
$('figure-dialog').onclick=e=>{if(e.target===$('figure-dialog'))$('figure-dialog').close()};
window.addEventListener('hashchange',route);
route();
