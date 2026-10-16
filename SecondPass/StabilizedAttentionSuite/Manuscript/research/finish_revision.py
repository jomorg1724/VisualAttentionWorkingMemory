from pathlib import Path
import re
r=Path(__file__).resolve().parents[1]
for name,a,b in [('flow',3,4),('architecture',4,5),('memory',5,6),('neuroscience',6,7),('experiments',7,8),('results',8,9)]:
    p=r/'sections'/f'{name}.html'
    s=p.read_text(encoding='utf-8').replace(f'{a:02} ·',f'{b:02} ·')
    s=re.sub(r'(<h3[^>]*>)'+str(a)+r'\.',r'\g<1>'+str(b)+'.',s)
    p.write_text(s,encoding='utf-8')
p=r/'sections/vda.html';p.write_text(p.read_text(encoding='utf-8').replace('02b ·','03 ·'),encoding='utf-8')
p=r/'sections/experiments.html';s=p.read_text(encoding='utf-8');index=s.index('<section id="reading-guide"')
(r/'sections/reading.html').write_text(s[index:].replace('08 · Reading and provenance','Reading and provenance'),encoding='utf-8')
p.write_text(s[:index],encoding='utf-8')
p=r/'build.py';s=p.read_text(encoding='utf-8').replace("'experiments','results']","'experiments','results','reading']").replace('09 · References','10 · References')
p.write_text(s,encoding='utf-8')
print('Updated chapter numbering and placed reading guide after results.')
