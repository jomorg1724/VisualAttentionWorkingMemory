/* Teaching figures use synthetic fields, never model results. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const bind = (ids, fn) => { ids.forEach(id => $(id).addEventListener('input', fn)); fn(); };
  const softmax = a => { const m=Math.max(...a), e=a.map(x=>Math.exp(x-m)), s=e.reduce((x,y)=>x+y,0); return e.map(x=>x/s); };
  const ink='#172d36', teal='#087f81', coral='#c66249';
  const text = (c,s,x,y,size=17) => { c.fillStyle=ink;c.font=`${size}px sans-serif`;c.fillText(s,x,y); };
  const axes = (c) => { c.clearRect(0,0,c.canvas.width,c.canvas.height);c.fillStyle='#fcfbf8';c.fillRect(0,0,c.canvas.width,c.canvas.height); };
  window.addEventListener('scroll',()=> { const d=document.documentElement; $('reading-progress').style.width=`${100*d.scrollTop/Math.max(1,d.scrollHeight-d.clientHeight)}%`; },{passive:true});
  const nav=[...document.querySelectorAll('.sidebar nav a')];
  new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting)nav.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));}),{rootMargin:'-5% 0px -70% 0px'}).observe($('introduction'));
  const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting)nav.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));}),{rootMargin:'-5% 0px -70% 0px'});
  document.querySelectorAll('.chapter').forEach(e=>observer.observe(e));
  $('print-paper').addEventListener('click',()=>window.print());
  let opened=[];window.addEventListener('beforeprint',()=>{opened=[...document.querySelectorAll('details:not([open])')];opened.forEach(d=>d.open=true);});window.addEventListener('afterprint',()=>opened.forEach(d=>d.open=false));
  const refs=[...document.querySelectorAll('.bibliography li')];refs.forEach((r,i)=>r.value=i+1);
  $('reference-search').addEventListener('input',()=>{let n=0;refs.forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes($('reference-search').value.toLowerCase());if(!r.hidden)n++;});$('reference-empty').hidden=n>0;});
  document.querySelectorAll('.cite').forEach(a=>a.addEventListener('click',()=>{ $('reference-search').value='';refs.forEach(r=>r.hidden=false);$('reference-empty').hidden=true;}));
  const tasks=window.MANUSCRIPT_TASKS;
  tasks.forEach(t=>{const o=document.createElement('option');o.value=t.id;o.textContent=t.title;$('atlas-task').append(o);});
  $('atlas-task').addEventListener('change',()=>{$('task-atlas').src='task-atlas/index.html#task='+encodeURIComponent($('atlas-task').value);});
  const demands={motion_direction:[0,0,1,0],orientation:[1,0,1,0],contrast:[1,0,1,0],spatial_frequency:[1,0,1,0],chromatic_increment:[1,0,1,0],contour:[1,1,1,0],natural_spectrum:[1,0,1,0],orientation_ring:[1,1,1,0],orientation_cued:[1,1,1,1],motion_duration_cued:[0,1,1,1],krauzlis_cued_motion:[0,1,1,1],spatial_binding:[1,1,1,1],image_recognition:[1,0,1,1]};
  $('task-demand-map').innerHTML='<div class="table-wrap"><table><thead><tr><th>Task</th><th>Appearance</th><th>Spatial selection</th><th>Temporal comparison / integration</th><th>Delay / list memory</th></tr></thead><tbody>'+tasks.map(t=>'<tr><th>'+t.title+'</th>'+demands[t.id].map(v=>`<td style="text-align:center;color:${v?teal:'#aaa'}" aria-label="${v?'Design demand':'Not a central demand'}">${v?'●':'—'}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';
  function flow(){
    const dx=+$('flow-dx').value,dy=+$('flow-dy').value,tau=+$('flow-tau').value,kind=$('flow-texture').value,N=24;
    $('flow-dx-value').textContent=dx+' px';$('flow-dy-value').textContent=dy+' px';$('flow-tau-value').textContent=tau.toFixed(3);
    const dots=[[5,5],[15,7],[9,17],[19,18],[18,3],[3,14]];
    const sample=(x,y)=>kind==='flat'?0.5:kind==='bar'?0.25+0.5/(1+Math.exp(-(x-11))):kind==='periodic'?0.5+0.35*Math.cos(2*Math.PI*x/5)*Math.cos(2*Math.PI*y/5):0.2+0.65*Math.min(1,dots.reduce((s,[u,v])=>s+Math.exp(-((x-u)**2+(y-v)**2)/2),0));
    const A=[],B=[];for(let y=0;y<N;y++)for(let x=0;x<N;x++){A.push(sample(x,y));B.push(sample(x-dx,y-dy));}
    const win=a=>{const mean=a.reduce((s,v)=>s+v,0)/(N*N);return a.map((v,i)=>(v-mean)*(.5-.5*Math.cos(2*Math.PI*(i%N)/(N-1)))*(.5-.5*Math.cos(2*Math.PI*Math.floor(i/N)/(N-1))));};
    const a=win(A),b=win(B),norm=Math.max(1e-8,Math.sqrt(a.reduce((s,v)=>s+v*v,0)*b.reduce((s,v)=>s+v*v,0))),corr=[];
    for(let sy=-4;sy<=4;sy++)for(let sx=-4;sx<=4;sx++){let s=0;for(let y=0;y<N;y++)for(let x=0;x<N;x++){if(x+sx>=0&&x+sx<N&&y+sy>=0&&y+sy<N)s+=a[y*N+x]*b[(y+sy)*N+x+sx];}corr.push(s/norm);}
    const p=softmax(corr.map(v=>v/tau)),ex=p.reduce((s,v,i)=>s+v*(i%9-4),0),ey=p.reduce((s,v,i)=>s+v*(Math.floor(i/9)-4),0),entropy=-p.reduce((s,v)=>s+v*Math.log(Math.max(v,1e-30)),0);
    const c=$('flow-canvas').getContext('2d');axes(c);
    [A,B].forEach((arr,j)=>{text(c,j?'Later patch':'Earlier patch',j*360+30,28);arr.forEach((v,i)=>{c.fillStyle=`rgb(${v*255},${v*255},${v*255})`;c.fillRect(j*360+30+i%N*11,46+Math.floor(i/N)*11,11.1,11.1);});});
    text(c,'Displacement probability',750,28);const max=Math.max(...p);p.forEach((v,i)=>{const r=v/max;c.fillStyle=`rgb(${245-237*r},${247-120*r},${240-111*r})`;c.fillRect(750+i%9*29,46+Math.floor(i/9)*29,28,28);});
    text(c,'x: −4 → +4; y: −4 ↓ +4',750,337,15);text(c,'Applied displacement: '+dx+', '+dy+' pixels',30,337,15);
    $('flow-readout').textContent=`Expected displacement (${ex.toFixed(2)}, ${ey.toFixed(2)}) px · entropy ${entropy.toFixed(2)} / ${Math.log(81).toFixed(2)} nats. ${kind==='flat'?'No texture: every displacement has equal probability.':'Window boundaries and ambiguous textures can bias the expectation. This example uses fixed correlation without learned refinement.'}`;
  }
  bind(['flow-texture','flow-dx','flow-dy','flow-tau'],flow);
  function windows(){const t=+$('window-time').value;$('window-time-value').textContent=t;let s='';for(let i=0;i<=40;i++){const ref=i>=t-15&&i<=t-8,recent=i>=t-3&&i<=t,gap=i>=t-7&&i<=t-4;const fill=ref?teal:recent?coral:gap?'#b7c1c0':'#ecece6';s+=`<rect x="${20+i*26}" y="30" width="23" height="46" rx="3" fill="${fill}" ${i===t?'stroke="#172d36" stroke-width="3"':''}/><text x="${31+i*26}" y="96" text-anchor="middle" font-size="12">${i}</text>`;}const nr=Math.max(0,Math.min(8,t-7));s+=`<text x="20" y="127" font-size="17" fill="${ink}">Reference (teal): ${nr}/8 frames available · Gap (gray): up to 4 · Recent (coral): ${Math.min(4,t+1)}/4</text>`;$('window-svg').innerHTML=s;}
  bind(['window-time'],windows);
  let selected=22;
  function attention(){const bias=+$('attention-bias').value;$('attention-bias-value').textContent=bias.toFixed(1);const uniform=$('attention-base').value==='uniform';const p=softmax(Array.from({length:100},(_,i)=>(uniform?0:3*Math.exp(-((i%10-2)**2+(Math.floor(i/10)-5)**2)/8))+(i===selected?bias:0)));const c=$('attention-canvas').getContext('2d');axes(c);const max=Math.max(...p);
    ['Shared spatial priority','Appearance × same weight','Motion × same weight'].forEach((label,j)=>{text(c,label,30+j*360,28);for(let i=0;i<100;i++){const x=30+j*360+i%10*28,y=45+Math.floor(i/10)*28,r=p[i]/max;c.fillStyle=j===0?`rgba(8,127,129,${.07+.93*r})`:`rgba(198,98,73,${.04+.9*r})`;c.fillRect(x,y,27,27);if(j===1){c.strokeStyle=`rgba(23,45,54,${.1+.9*r})`;c.beginPath();c.moveTo(x+5,y+22);c.lineTo(x+22,y+5);c.stroke();}if(j===2){c.strokeStyle=`rgba(23,45,54,${.1+.9*r})`;c.beginPath();c.moveTo(x+4,y+14);c.lineTo(x+22,y+14);c.lineTo(x+17,y+9);c.moveTo(x+22,y+14);c.lineTo(x+17,y+19);c.stroke();}if(i===selected){c.strokeStyle=ink;c.lineWidth=2;c.strokeRect(x,y,27,27);c.lineWidth=1;}}});text(c,'Click a site in the first grid; arrow keys also move the selected site.',30,347,15);
    const a=p[selected];$('attention-readout').textContent=`Site (${selected%10}, ${Math.floor(selected/10)}) · attention ${(100*a).toFixed(2)}% · spatial write modulation ${(a/(a+.01)).toFixed(3)} · odds multiplier exp(bias) = ${Math.exp(bias).toPrecision(3)}. Colors are scaled to this frame's largest weight; all panels share the same weights.`;
  }
  bind(['attention-bias','attention-base'],attention);$('attention-reset').addEventListener('click',()=>{$('attention-bias').value=0;attention();});
  $('attention-canvas').addEventListener('click',e=>{const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)*1080/r.width,y=(e.clientY-r.top)*360/r.height;if(x>=30&&x<310&&y>=45&&y<325){selected=Math.floor((y-45)/28)*10+Math.floor((x-30)/28);attention();}});
  $('attention-canvas').addEventListener('keydown',e=>{const d={ArrowLeft:-1,ArrowRight:1,ArrowUp:-10,ArrowDown:10}[e.key];if(d){e.preventDefault();selected=Math.max(0,Math.min(99,selected+d));attention();}});
  function memory(){const a=+$('memory-alpha').value,w=+$('memory-write').value,g=a/(a+.01),q=1-g*w;$('memory-alpha-value').textContent=a.toFixed(3);$('memory-write-value').textContent=w.toFixed(2);let s=`<path d="M60 20V220H1010" fill="none" stroke="${ink}"/>`;for(let k=0;k<=30;k+=5)s+=`<text x="${60+k*31}" y="245" text-anchor="middle" font-size="15">${k}</text>`;[0,.5,1].forEach(v=>{s+=`<path d="M60 ${220-v*190}H990" stroke="#ddd"/><text x="45" y="${226-v*190}" text-anchor="end" font-size="15">${v}</text>`;});s+=`<polyline points="${Array.from({length:31},(_,k)=>`${60+k*31},${220-190*q**k}`).join(' ')}" fill="none" stroke="${teal}" stroke-width="4"/><text x="740" y="266" font-size="15">Subsequent recurrent updates</text>`;$('memory-svg').innerHTML=s;const half=q===1?'infinite':q===0?'less than one':(Math.log(.5)/Math.log(q)).toFixed(2);$('memory-readout').textContent=`Attention modulation g = ${g.toFixed(3)} · effective write gw = ${(g*w).toFixed(3)} · direct retention per update = ${q.toFixed(3)} · half-life = ${half} updates. Fixed candidate = 0; initial memory = 1.`;}
  bind(['memory-alpha','memory-write'],memory);
})();
