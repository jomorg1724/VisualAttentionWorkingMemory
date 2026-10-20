// Logic and asset-contract tests in a simulated DOM; this is not a browser render.
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('assert');
const htmlPath=path.join(__dirname,'Results','attention','index.html');
const source=fs.readFileSync(htmlPath,'utf8');
const elements=new Map(),assets=new Set();let tick=null;
const context={globalAlpha:1,fillRect(){},clearRect(){},drawImage(){},imageSmoothingEnabled:false};
class Element{
 constructor(tag){this.tag=tag;this.value='';this.children=[];this.width=300;this.height=300;this.textContent='';}
 append(e){this.children.push(e);if(this.tag==='select'&&this.children.length===1)this.value=e.value;}
 replaceChildren(...items){this.children=items;if(this.tag==='select')this.value=items[0]?.value??'';}
 getContext(){return context;}
 set src(p){assets.add(p);this._src=p;}
 get src(){return this._src;}
 set href(p){if(p&&!p.startsWith('http'))assets.add(p);this._href=p;}
}
for(const m of source.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)){
 const el=new Element(m[1]);const value=m[0].match(/\bvalue="([^"]+)"/);if(value)el.value=value[1];elements.set(m[2],el);
}
class ImageStub extends Element{constructor(){super('img');this.complete=false;this.naturalWidth=0;}}
const sandbox={document:{getElementById:id=>{assert(elements.has(id),id);return elements.get(id)},createElement:tag=>new Element(tag)},
 Image:ImageStub,URLSearchParams,location:{hash:'#task=vda4'},history:{replaceState(){}},setInterval:f=>{tick=f;return 1},clearInterval:()=>{tick=null},console};
vm.createContext(sandbox);vm.runInContext(source.match(/<script>([\s\S]*)<\/script>/)[1],sandbox);
let conditionCount=0,frameCount=0;
for(const option of elements.get('task').children){
 elements.get('task').value=option.value;elements.get('task').onchange();
 for(const group of elements.get('condition').children){
  elements.get('condition').value=group.value;elements.get('condition').onchange();
  const max=Number(elements.get('frame').max);
  for(let f=0;f<=max;f++){
   elements.get('frame').value=String(f);elements.get('frame').oninput();
   assert.equal(elements.get('clock').textContent,`Frame ${f} / ${max}`);frameCount++;
  }
  elements.get('next').onclick();assert.equal(elements.get('clock').textContent,`Frame ${max} / ${max}`);
  elements.get('play').onclick();assert.equal(elements.get('play').textContent,'Pause');tick();
  assert.equal(elements.get('clock').textContent,`Frame 0 / ${max}`);
  elements.get('play').onclick();assert.equal(tick,null);conditionCount++;
 }
}
for(const asset of assets)assert(fs.existsSync(path.resolve(path.dirname(htmlPath),asset)),asset);
assert.equal(conditionCount,63);assert.equal(frameCount,888);
console.log(JSON.stringify({simulated_dom:true,conditions:conditionCount,frames:frameCount,assets:assets.size,play_pause_wraparound:true,syntax_and_runtime:'passed',browser_render:false},null,2));
