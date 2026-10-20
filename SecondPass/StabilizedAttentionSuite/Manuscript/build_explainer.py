"""Build the guided explainer from frozen measurements. No inference or training."""
from pathlib import Path
import json, hashlib, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from SecondPass.MemoryGuidedSearch.dataset import TASKS
from SecondPass.TaskNeuroscience.attention_report import CONTEXT

RESULTS=ROOT/'SecondPass/TaskNeuroscience/Results'
ASSETS=HERE/'assets/guide'
ASSETS.mkdir(parents=True,exist_ok=True)
inputs={}
def read(path):
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf8'))
def copy(source,dest):
    inputs[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(source,dest)

TITLES={
'motion_direction':'Read a motion direction','orientation':'Compare two orientations',
'contrast':'Compare contrast','spatial_frequency':'Compare spatial frequency',
'chromatic_increment':'Compare color','contour':'Find an aligned contour',
'natural_spectrum':'Compare image detail','orientation_ring':'Follow a spatial cue',
'orientation_cued':'Remember a location and a rule','motion_duration_cued':'Accumulate motion evidence',
'krauzlis_cued_motion':'Detect a relevant motion change','spatial_binding':'Remember what was where',
'image_recognition':'Recognize a studied image','vda4':'VDA · 4 locations','vda9':'VDA · 9 locations',
'vda16':'VDA · 16 locations','memory_guided_search':'Search for a remembered object',
'popout_search':'Find the singleton','motion_memory_search':'Search for a remembered motion'}

INTRO={
'motion_direction':('Perception','Two dot images contain a displacement. Judge the direction linking them.','A moving pattern can be identified from correspondence across observations.'),
'orientation':('Perception','Compare the angle of the stripes in the first and second images. Carrier phase can change independently.','The observer must retain orientation while allowing other pixel details to change.'),
'contrast':('Perception','Choose the image whose light and dark stripes differ more strongly from the mean brightness.','Contrast is the size of light–dark variation, rather than the overall brightness of the image.'),
'spatial_frequency':('Perception','Choose the image with more closely spaced stripes.','Spatial frequency counts cycles of a pattern per distance. Finer stripes have higher spatial frequency.'),
'chromatic_increment':('Perception','Compare two colored patterns and judge which has the positive change along the task’s specified RGB color axis.','The renderer keeps its numerical luminance constraint while changing color. This probes information carried by the color pathway.'),
'contour':('Perception','Choose which image contains a contour formed by locally aligned stripe elements.','The spatial arrangement of the elements matters, alongside their individual orientations.'),
'natural_spectrum':('Perception','Compare two versions of one photograph. Choose the version with greater relative fine-scale detail.','The two images differ in their balance of coarse and fine spatial structure.'),
'orientation_ring':('Selection','A ring marks one striped patch. Compare its orientation across observations and report the sign of its rotation.','Other locations contain competing information. The cue identifies the comparison that matters.'),
'orientation_cued':('Selection','Remember the cued location and requested rotation sign. After the blank delay, report whether that patch rotated in the requested direction.','The cue carries both a spatial instruction and a rule; a change at another location is insufficient.'),
'motion_duration_cued':('Memory','Follow the cued dot patch across eight motion transitions. Report the direction that occurred most often, after the retention interval.','The final direction can differ from the correct answer. Success requires information from the earlier stream.'),
'krauzlis_cued_motion':('Selection','Remember the cued dot patch. Report a direction change there; ignore changes confined to the other patch.','The cue disappears before the critical motion. The model must preserve which location is relevant.'),
'spatial_binding':('Memory','Remember the orientations at four locations. After a delay, two exchange orientations. A later cue asks whether its location participated.','Remembering which orientation belonged where allows the later location cue to select an answer.'),
'image_recognition':('Memory','Study a sequence of photographs. After a blank interval, decide whether the probe image appeared in that sequence.','Each study image can differ. More study images increase memory load; the probe can remain visible for several frames.'),
'vda4':('Selection','Use a probabilistic cue, then compare two arrays of four striped patches. Report an orientation change anywhere in the array.','The cue suggests where a change is likely. A change away from the cue still requires a positive response.'),
'vda9':('Selection','Use a probabilistic cue, then compare two arrays of nine striped patches. Report an orientation change anywhere.','Nine candidate locations increase spatial competition within the same image.'),
'vda16':('Selection','Use a probabilistic cue, then compare two arrays of sixteen striped patches. Report an orientation change anywhere.','Sixteen locations test spatial selection at a denser display scale.'),
'memory_guided_search':('Search','Remember the central sample. After a blank interval, choose its matching location among four candidates.','Feature patterns, irregular contour objects and procedural fractals can have distinct or similar alternatives.'),
'popout_search':('Search','Choose the one item whose color or orientation differs from the other three.','Changing the feature difference and viewing duration reveals how quickly a reliable location decision develops.'),
'motion_memory_search':('Search','Remember the direction of the central moving dots. After a delay, choose the candidate moving in that direction.','Sample and search dots have independent layouts. The remembered direction must transfer to a new pixel pattern.')}

GIF_TASKS={'motion_memory_search','memory_guided_search','image_recognition','orientation_cued','krauzlis_cued_motion','vda4'}
font=ImageFont.truetype(str(HERE/'task-atlas/vendor/font-3.ttf'),17)
catalog=[]; counts={'groups':0,'frames':0,'gifs':0}
for task,spec in TASKS.items():
    d=read(RESULTS/task/'attention_dynamics.json')
    behavior=read(RESULTS/task/'behavior.json')
    groups=[]; folder=ASSETS/task;folder.mkdir(exist_ok=True)
    for number,g in enumerate(d['groups']):
        row={k:g[k] for k in ['title','n','clusters','accuracy','time','aligned_time','phases','mean_map','example_map','example','region_names','uniform_mass','mean_mass','ci_low','ci_high']}
        # Retain recorded decisions and true labels for the same displayed trial.
        ref=g['example']; npz=RESULTS/task/'native'/(ref['cell']+'.npz')
        inputs[str(npz.relative_to(ROOT))]=hashlib.sha256(npz.read_bytes()).hexdigest()
        with np.load(npz) as z:
            idx=ref['index']; row['answer']=int(z['y'][idx])
            row['probabilities']=z['p'][idx].tolist()
            row['choice']=int(z['p'][idx].argmax())
            row['temporal']=z['temporal'][idx].tolist()
            np.testing.assert_allclose(g['example_map'],z['attention'][idx],atol=1e-7)
        metadata=read(npz.with_suffix('.json'))[ref['index']]
        row['metadata']=metadata
        image=RESULTS/task/'figures'/g['sprite']
        copy(image,folder/g['sprite'])
        row['sprite']=f"assets/guide/{task}/{g['sprite']}"
        sprite=Image.open(image).convert('RGB')
        assert sprite.size==(100*len(g['time']),100)
        row['gif']=None
        if task in GIF_TASKS and number==0:
            frames=[]
            for i,phase in enumerate(g['phases']):
                canvas=Image.new('RGB',(320,365),'#10262d')
                frame=sprite.crop((100*i,0,100*(i+1),100)).resize((300,300),Image.Resampling.NEAREST)
                canvas.paste(frame,(10,10))
                draw=ImageDraw.Draw(canvas)
                draw.text((12,323),f"{i+1:02d} / {len(g['time']):02d}   {phase.upper()}",font=font,fill='#dfece8')
                frames.append(canvas)
            gif=folder/'trial.gif'
            frames[0].save(gif,save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=False,disposal=2)
            row['gif']=f'assets/guide/{task}/trial.gif';counts['gifs']+=1
        groups.append(row);counts['groups']+=1;counts['frames']+=len(g['time'])
    payload={'task':task,'groups':groups,'context':CONTEXT[task],
             'colorMax':d['figures']['color_max'],
             'ownMax':max(np.asarray(g['example_map']).max() for g in groups),
             'cells':behavior['cells']}
    # Script payloads also support direct file:// reading without fetch/CORS.
    (folder/'data.js').write_text('window.GUIDE_TASKS=window.GUIDE_TASKS||{};\nwindow.GUIDE_TASKS['+json.dumps(task)+']='+json.dumps(payload,separators=(',',':')).replace('</','<\\/')+';',encoding='utf8')
    family,instruction,why=INTRO[task]
    catalog.append(dict(id=task,title=TITLES[task],family=family,instruction=instruction,why=why,
                        labels=spec['labels'],question=spec['question'],conditions=behavior['conditions'],
                        trials=behavior['trials'],accuracy=behavior['equal_cell_accuracy'],
                        ci=behavior['equal_cell_ci95'],groups=len(groups),
                        variants=[c['kwargs'] for c in spec['conditions']]))

pdfroot=ROOT/'output/pdf/neuroscience_results'
for p in (pdfroot/'figures').glob('*.png'):copy(p,ASSETS/p.name)
copy(pdfroot/'Shared_Priority_Neuroscience_Results.pdf',ASSETS/'neuroscience-results.pdf')
refs=read(pdfroot/'references.json')
recovery=read(ROOT/'SecondPass/SpatialMemoryQuerySuite/CloudRuntime/recovery_20261018/full_recovery_verified.json')
meta=dict(checkpoint=204289,checkpointSha256='596bc351783b3ddbfa8d37a9cf8d65146d1333dcc0097d40f0f75665b6451bcf',
          date='20 October 2026',tasks=catalog,references=refs,counts=counts,
          newExperiment=dict(latest=recovery['checkpoints_restored'][0]['step'],best=recovery['checkpoints_restored'][1]['step'],
                             bestMonitoringAuc=recovery['checkpoints_restored'][1]['best'],target=38000))
(ASSETS/'catalog.js').write_text('window.GUIDE='+json.dumps(meta,separators=(',',':')).replace('</','<\\/')+';',encoding='utf8')
if not (HERE/'reference.html').exists():shutil.copy2(HERE/'index.html',HERE/'reference.html')
shutil.copy2(HERE/'explainer.html',HERE/'index.html')
receipt=dict(tasks=len(catalog),native_conditions=sum(t['conditions'] for t in catalog),
             native_trials=sum(t['trials'] for t in catalog),**counts,
             source_hashes=inputs,new_inference=False,new_training=False,
             input_display='Original 100x100 lossless sprite frames, nearest-neighbor display.',
             gif_format='200 ms per frame; indexed-color viewing copies with external frame labels.',
             checkpoint=204289)
(ASSETS/'provenance.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print({k:v for k,v in receipt.items() if k!='source_hashes'})
