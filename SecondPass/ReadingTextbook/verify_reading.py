"""CPU-only document checks. Does not import or execute model/training code."""
import hashlib,json,re
from pathlib import Path
import fitz
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
pdf=HERE/'output/pdf/visual_memory_textbook.pdf'
doc=fitz.open(pdf)
checks=[]
for i,page in enumerate(doc):
 spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
 outside=[s['text'] for s in spans if s['bbox'][0]<32 or s['bbox'][2]>page.rect.width-30 or s['bbox'][1]<20 or s['bbox'][3]>page.rect.height-20]
 text=page.get_text(); checks.append(dict(page=i+1,words=len(text.split()),outside_page_safe_bounds=outside,replacement_character='\ufffd' in text))
 page.get_pixmap(matrix=fitz.Matrix(.8,.8)).save(HERE/f'tmp/pdfs/page_{i+1:02}.png')
for start in range(0,len(doc),12):
 sheet=Image.new('RGB',(1080,1140),'#dce2e5');d=ImageDraw.Draw(sheet)
 for j,i in enumerate(range(start,min(start+12,len(doc)))):
  im=Image.open(HERE/f'tmp/pdfs/page_{i+1:02}.png').convert('RGB');im.thumbnail((250,354))
  x=(j%4)*270+10;y=(j//4)*380+20;sheet.paste(im,(x,y));d.text((x,y-15),str(i+1),fill='black')
 sheet.save(HERE/f'tmp/pdfs/montage_{start//12+1}.png')
md=(HERE/'visual_memory_textbook.md').read_text()
catalog=json.loads((ROOT/'SecondPass/TaskSuite/catalog.json').read_text())
assert len(catalog['tasks'])==13
assert sum(len(t['conditions']) for t in catalog['tasks'])==35
assert sum(c['selection_eligible'] for t in catalog['tasks'] for c in t['conditions'])==32
for task in catalog['tasks']: assert '**Task key:** `'+task['id']+'`' in md
assert (25**2+13**2+7**2)*2*8*16==215808
assert 64*7*7==3136
assert 32*82*9+82+32*32+32==24754
files=['PreAttentiveVision/TemporalIntegration/accumulators.py','WorkingMemory/PlainBaseline/accum.py','SecondPass/JointTraining/worker.py','SecondPass/JointTraining/core.py','SecondPass/JointTraining/continuation_v3.py','SecondPass/JointTraining/AMENDMENT_V3.md','SecondPass/TaskSuite/README.md','SecondPass/TaskSuite/catalog.json','SecondPass/TaskSuite/suite.py','PreAttentiveVision/neuroscience_stimuli.py','PreAttentiveVision/natural_stimuli.py','WorkingMemory/PlainBaseline/variants.py','WorkingMemory/SpatialTaskBattery/stimuli.py','WorkingMemory/stimuli.py','SecondPass/SpatialReadout/BRIEF.md','SecondPass/JointTraining/TechnicalReport/architecture_microstimulation.md','SecondPass/JointTraining/TechnicalReport/architecture_metrics_notes.md']
manifest={'edition':'2026-09-24','status':'New final ConvGRU described as planned, per user clarification and brief. No execution claim.','sources':[{'path':f,'sha256':hashlib.sha256((ROOT/f).read_bytes()).hexdigest()} for f in files]}
original=Path('/Users/jonathanmorgan/Downloads/87252611-f998-4054-8a87-a46d9a26e873.pdf')
manifest['supplied_pdf']={'filename':original.name,'sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result={'pages':len(doc),'source_words':len(md.split()),'all_13_task_sections':True,'primary_cells':35,'selection_eligible_cells':32,'architecture_arithmetic_checks':'passed','pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'page_checks':checks,'visual_review':'See rendered page montages; human/model visual inspection required separately.'}
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='page_checks'},indent=2))
print('Sparse pages',[(p['page'],p['words']) for p in checks if p['words']<100])
assert not any(p['outside_page_safe_bounds'] or p['replacement_character'] for p in checks),checks
