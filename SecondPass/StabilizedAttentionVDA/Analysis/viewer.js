const $=id=>document.getElementById(id);let group=null,t=0,timer=null,version=0;const cache=new Map();
function armLabel(arm){if(arm==='sham')return 'No intervention';const [site,dose]=arm.split('_');return `${+dose<0?'Inhibit':'Stimulate'} ${site==='cued'?'cued':'opposite'} location (${dose}) during change`}
function load(src){if(!cache.has(src))cache.set(src,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src}));return cache.get(src)}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play'}
function options(){stop();const size=+$('task').value,groups=ATLAS.filter(g=>g.metadata.size===size);$('group').replaceChildren();
for(const g of groups){const o=document.createElement('option');o.value=g.id;o.textContent=`Cue ${g.metadata.cue===0?'top-left':'bottom-right'} · ${g.metadata.event==='uncued'?'opposite change':g.metadata.event==='cued'?'cued change':'no change'} · ${armLabel(g.arm)}`;$('group').append(o)}
const preferred=groups.find(g=>g.metadata.cue===size-1&&g.metadata.event==='uncued'&&g.arm==='cued_-4_change');if(preferred)$('group').value=preferred.id;select()}
function select(){stop();group=ATLAS.find(g=>g.id===$('group').value);t=0;draw()}
async function draw(){const stamp=++version,mode=$('example').value,g=group,frame=t,mean=mode==='mean',paths=mean?g.mean:g.examples[+mode];
const sources=[paths.image,paths.overlay,paths.map,g.mean.map];const images=await Promise.all(sources.map(load));if(stamp!==version)return;
['original','overlay','map','meanmap'].forEach((id,i)=>{const ctx=$(id).getContext('2d');ctx.imageSmoothingEnabled=false;ctx.clearRect(0,0,100,100);ctx.drawImage(images[i],frame*100,0,100,100,0,0,100,100)});
$('title1').textContent=mean?'Condition mean image':'Actual model input';$('title3').textContent=mean?'Condition mean attention':'Its projected attention';
$('description').textContent=`VDA-${g.metadata.size} · 100% cue · ${g.metadata.event==='catch'?'no physical change':'18° change'} · ${armLabel(g.arm)} · ${g.n} scenes in this condition. Opposite-location changes deliberately violate the cue.`;
$('time').value=frame;$('frame').textContent=`t=${frame} · ${g.phases[frame]} · frame ${frame+1}/7`;
$('caution').textContent=mean?'The mean image blends changing Gabor orientations and all three cue colors. It is explicitly paired with the condition mean map.':'The first three panels belong to the same individual movie. The fourth averages 48 separate scenes, pooling colors and signs; it is not the explanation of this particular image.'}
$('task').onchange=options;$('group').onchange=select;$('example').onchange=draw;$('time').oninput=()=>{stop();t=+$('time').value;draw()};
$('prev').onclick=()=>{stop();t=(t+6)%7;draw()};$('next').onclick=()=>{stop();t=(t+1)%7;draw()};$('play').onclick=()=>{if(timer){stop();return}timer=setInterval(()=>{t=(t+1)%7;draw()},600);$('play').textContent='Pause'};options();
