"""CPU-only task movies and a compact construction check; no model execution."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import sys,json,base64,io,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from WorkingMemory.SpatialTaskBattery.stimuli import *
def png(a):
    out=io.BytesIO();Image.fromarray(np.rint(a.transpose(1,2,0)*255).astype('uint8')).save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
def main():
    start=time.time();torch.set_num_threads(1);examples=[];checks={};out=HERE/'previews';out.mkdir(exist_ok=True)
    settings={'orientation_cued':dict(delay=4),'motion_duration_cued':dict(delay=4),'spatial_binding':dict(delay=4),'krauzlis_cued_motion':dict(baseline_transitions=12),'image_recognition':dict(load=4,probe_hold=4)}
    for task,condition in settings.items():
        stream=SpatialBatteryStream(67973001,'test');needed={'target','foil','catch'} if task=='krauzlis_cued_motion' else {0,1};found={}
        for i in range(100):
            x,y,m=stream.batch(1,task,condition);kind=m[0]['event_type'] if task=='krauzlis_cued_motion' else int(y[0])
            if kind in needed and kind not in found:found[kind]=(x[0].numpy(),m[0])
            if needed.issubset(found):break
        assert needed.issubset(found)
        for kind,(frames,meta) in found.items():
            name=f'{task}_{kind}';pil=[Image.fromarray(np.rint(f.transpose(1,2,0)*255).astype('uint8')) for f in frames];pil[0].save(out/(name+'.gif'),save_all=True,append_images=pil[1:],duration=220,loop=0,optimize=False)
            indices=sorted(set([0,1,2,len(frames)-2,len(frames)-1]+meta.get('cue_frames',[])[:1]+meta.get('probe_frames',[])[:1]+[meta.get('first_postchange_frame',len(frames)//2)]));indices=[i for i in indices if 0<=i<len(frames)]
            strip=Image.new('RGB',(120*len(indices),148),'#f9f7ef');draw=ImageDraw.Draw(strip)
            for j,i in enumerate(indices):strip.paste(pil[i],(j*120+10,25));draw.text((j*120+12,5),f'Frame {i}',fill='#16383f')
            strip.save(out/(name+'_strip.png'));examples.append(dict(id=name,task=task,condition=condition,example_type=str(kind),metadata=meta,frames=[png(f) for f in frames]))
    # One batch per task: valid shapes, pixels and exact checkpointable replay.
    for task in TASK_CLASSES:
        stream=SpatialBatteryStream(68973001,'train');condition=TRAIN_CONDITIONS[task][-1];state=copy.deepcopy(stream.state_dict());x,y,m=stream.batch(8,task,condition);stream.load_state_dict(state);xx,yy,mm=stream.batch(8,task,condition);assert torch.equal(x,xx) and torch.equal(y,yy) and m==mm
        checks[task]=dict(shape=list(x.shape),label_histogram=np.bincount(y.numpy(),minlength=TASK_CLASSES[task]).tolist(),sampler_pixels_labels_metadata_replay_equal=True)
    for task in ('orientation_cued','motion_duration_cued','spatial_binding'):
        a=SpatialBatteryStream(77,'val');b=SpatialBatteryStream(77,'val');x,y,m=a.batch(8,task,dict(delay=0));z,zy,zm=b.batch(8,task,dict(delay=24));n=10 if task=='motion_duration_cued' else 3;tail=2 if task=='spatial_binding' else 1;assert torch.equal(x[:,:n],z[:,:n]) and torch.equal(x[:,-tail:],z[:,-tail:]) and torch.equal(y,zy);checks[task]['paired_delays_preserve_evidence']=True
    s=SpatialBatteryStream(42);events=[];sides=[]
    for i in range(200):label,side=s._case('krauzlis_cued_motion');events.append(s._event_kind);sides.append(side);s.counts['krauzlis_cued_motion']+=1
    assert {k:events[:100].count(k) for k in ('target','foil','catch')}==dict(target=57,foil=29,catch=14)
    checks['krauzlis_cued_motion'].update(first100_event_counts={k:events[:100].count(k) for k in ('target','foil','catch')},first100_target_side_counts=np.bincount(sides[:100]).tolist())
    s=SpatialBatteryStream(31,'test');x,y,m=s.batch(8,'image_recognition',dict(load=0,probe_hold=3));assert int(y.sum())==0;checks['image_recognition']['empty_set_all_negative']=True
    for meta in m:assert meta['probe_raster_sha256'] not in meta['study_raster_sha256']
    checks['status']='passed';checks['elapsed_seconds']=time.time()-start;checks['no_model_or_gpu_execution']=True;(HERE/'construction_checks.json').write_text(json.dumps(checks,indent=2));(out/'movies.json').write_text(json.dumps(examples,separators=(',',':')))
    html='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spatial task battery · stimuli preview</title><style>body{font:16px/1.55 system-ui;margin:0;color:#16383f;background:#faf8f1}header{padding:30px 6vw;background:#16383f;color:#f7f5eb}h1{font:38px Georgia;font-weight:400;margin:0}main{max-width:1050px;margin:auto;padding:25px}.card{background:white;padding:20px;border:1px solid #d4ded5;border-radius:10px;margin:16px 0}select,button{padding:8px;font:inherit;border:1px solid #bacbc3;border-radius:5px;background:white}canvas{display:block;width:min(80vw,400px);height:min(80vw,400px);image-rendering:pixelated;border:1px solid #b9cac1;margin:20px auto}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}input{flex:1}pre{white-space:pre-wrap;font:13px/1.45 monospace}.note{font-size:14px;color:#5c7577}a{color:#257f83}</style><header><h1>Five tasks, actual input movies</h1><p>Stimulus preview only. No trained performance is claimed.</p></header><main><div class="card"><select id="example"></select><div class="controls"><button id="play">▶Play</button><input id="frame" type="range" min="0" value="0"><span id="counter"></span></div><canvas id="scene" width="100" height="100"></canvas><p class="note">These are the exact100×100RGB frames supplied to the model. Playback is slowed to220ms per frame for inspection; it does not define physiological timing. Labels/angles/locations below are analysis metadata and never enter the model.</p><pre id="metadata"></pre></div><p><a href="../PROTOCOL.md">Protocol and timing</a> · <a href="../SOURCES.md">Primary sources</a> · <a href="../construction_checks.json">Construction checks</a> · <a href="../recognition_manifest.json">Disjoint source/raster hashes</a></p></main><script>const DATA=PAYLOAD;let selected=DATA[0],timer=null,version=0;let chooser=document.getElementById('example');DATA.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=e.task+' · '+e.example_type;chooser.append(o)});function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶Play'}async function draw(){let token=++version,t=Number(document.getElementById('frame').value);document.getElementById('counter').textContent='Frame '+t+'/'+(selected.frames.length-1);let im=new Image();im.src=selected.frames[t];await im.decode();if(token===version)document.getElementById('scene').getContext('2d').drawImage(im,0,0)}function select(){stop();selected=DATA[Number(chooser.value)];document.getElementById('frame').max=selected.frames.length-1;document.getElementById('frame').value=0;document.getElementById('metadata').textContent=JSON.stringify(selected.metadata,null,2);draw()}chooser.onchange=select;document.getElementById('frame').oninput=draw;document.getElementById('play').onclick=()=>{if(timer){stop();return}document.getElementById('play').textContent='Pause';timer=setInterval(()=>{let f=document.getElementById('frame');f.value=(Number(f.value)+1)%selected.frames.length;draw()},220)};select();</script></html>'''
    (out/'index.html').write_text(html.replace('PAYLOAD',json.dumps(examples,separators=(',',':'))),encoding='utf-8');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
