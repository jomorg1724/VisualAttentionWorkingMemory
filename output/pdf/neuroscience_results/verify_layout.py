from pathlib import Path
import fitz, json, hashlib
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
files = sorted((root/'rendered').glob('page-*.png'))
for start in range(0,len(files),6):
    canvas=Image.new('RGB',(1200,1100),'#dddddd')
    for j,f in enumerate(files[start:start+6]):
        im=Image.open(f)
        im.thumbnail((390,510))
        x=(j%3)*400+(400-im.width)//2
        y=(j//3)*550+25
        canvas.paste(im,(x,y))
        ImageDraw.Draw(canvas).text(((j%3)*400+10,(j//3)*550+5),f.stem,fill='black')
    canvas.save(root/'rendered'/f'contact-{start//6+1}.png')
doc=fitz.open(root/'main.pdf')
outside=[]
for i,p in enumerate(doc):
    for b in p.get_text('blocks'):
        if b[0]<0 or b[1]<0 or b[2]>p.rect.width+.1 or b[3]>p.rect.height+.1:
            outside.append([i+1,list(b[:4])])
text='\n'.join(p.get_text() for p in doc)
audit={'page_count':len(doc),'word_count':len(text.split()),'pages_with_no_text':[i+1 for i,p in enumerate(doc) if not p.get_text().strip()], 'text_blocks_outside_page':outside,'unresolved_question_mark_pair':'??' in text,'pdf_sha256':hashlib.sha256((root/'main.pdf').read_bytes()).hexdigest()}
(root/'layout_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
